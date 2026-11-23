"""Common functionality for the Approximate Nearest Neighbour recommender models.

The ANN wrappers in this package (annoy, nmslib, faiss) speed up inference on top of an
underlying matrix factorization model by building nearest neighbour indices over the
model's item factors. A problem with keeping a separate index is that the factors and the
index can drift out of sync: for instance :meth:`partial_fit_items` mutates the model's
item factors in place, so between that update and the completion of an index rebuild the
live index still represents the *old* factors, while queries read the *new* factors.

To keep serving consistent, the factors used by the ANN indices are bound together into an
immutable :class:`ANNModelVersion` snapshot. New factors and new indices are built off to
the side and are only published through a single atomic swap, so that an online reader
always observes a version whose indices were built from exactly the factors stored inside
that same version. If the index rebuild fails, the previous (still usable) version remains
published instead of leaving a half built index live.
"""

import logging
import threading
import time

import numpy as np

import implicit.gpu
from implicit.recommender_base import RecommenderBase

log = logging.getLogger("implicit")


class ANNModelVersion:
    """An immutable, self consistent snapshot of factors bound to the ANN indices.

    Every attribute on a version is either frozen at build time (the factor snapshot,
    the ``max_norm`` used by the inner product transformation) or is an index that was
    built from that exact factor snapshot. A version is only ever replaced wholesale by
    another fully built version, never mutated in place.

    Parameters
    ----------
    version : int
        Monotonically increasing identifier for this version.
    item_factors : numpy.ndarray
        Snapshot of the underlying model's item factors at build time. These are the
        factors that every index in this version was built from.
    user_factors : numpy.ndarray
        Snapshot of the underlying model's user factors at build time.
    similar_items_index : object, optional
        ANN index for the :meth:`ANNModelBase.similar_items` call, or None when
        approximate similar item lookups are disabled.
    recommend_index : object, optional
        ANN index for the :meth:`ANNModelBase.recommend` call, or None when
        approximate recommendation is disabled.
    max_norm : float, optional
        Maximum row norm of ``item_factors``, used to convert the angular/cosine ANN
        distance back into an inner product score.
    built_at : float, optional
        Epoch timestamp (as returned by :func:`time.time`) marking when this version's
        indices finished building.
    item_available_at : dict, optional
        Mapping of itemid -> epoch timestamp recording when the factor for that item
        first became recommendable (had a non zero latent vector). Used to answer when a
        newly added item can first be recommended.
    """

    def __init__(
        self,
        version,
        item_factors,
        user_factors,
        similar_items_index=None,
        recommend_index=None,
        max_norm=None,
        built_at=None,
        item_available_at=None,
    ):
        self.version = version
        self.item_factors = item_factors
        self.user_factors = user_factors
        self.similar_items_index = similar_items_index
        self.recommend_index = recommend_index
        self.max_norm = max_norm
        self.built_at = built_at if built_at is not None else time.time()
        # copy on the way in so later mutations of the caller's dict can't leak in
        self.item_available_at = dict(item_available_at) if item_available_at else {}

    def is_item_recommendable(self, itemid):
        """Returns whether itemid has been indexed as recommendable in this version."""
        return itemid in self.item_available_at


class ANNModelBase(RecommenderBase):
    """Base class for ANN recommender wrappers.

    Holds the underlying matrix factorization ``model`` together with the currently
    published :class:`ANNModelVersion`. Subclasses are responsible for building their
    flavour of ANN index by implementing :meth:`_build_similar_items_index` and
    :meth:`_build_recommend_index`; all of the version bookkeeping, atomic swapping and
    failure rollback lives here.

    Parameters
    ----------
    model : MatrixFactorizationBase
        A matrix factorization model to use for the factors.
    approximate_similar_items : bool, optional
        whether or not to build an ANN index for computing similar_items
    approximate_recommend : bool, optional
        whether or not to build an ANN index for the recommend call
    """

    def __init__(self, model, approximate_similar_items=True, approximate_recommend=True):
        self.model = model
        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend

        # the currently published, self consistent (factors + indices) view. None until
        # the first successful fit / index rebuild.
        self._active_version = None
        # monotonically increasing counter used to name versions
        self._next_version = 1
        # itemid -> timestamp of when the item first had a non zero factor. Carried
        # forward across versions so the history survives index rebuilds.
        self._item_available_at = {}
        # guards the swap so concurrent readers never observe a partial publish
        self._version_lock = threading.RLock()

    def __getstate__(self):
        # a threading.RLock isn't picklable; drop it and recreate on unpickle
        state = self.__dict__.copy()
        state["_version_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._version_lock = threading.RLock()

    # ------------------------------------------------------------------
    # version access
    # ------------------------------------------------------------------
    @property
    def active_version(self):
        """The currently published :class:`ANNModelVersion`.

        Raises a :class:`ValueError` if the model hasn't been fit yet, matching the
        behaviour of reading factors off an untrained matrix factorization model.
        """
        if self._active_version is None:
            raise ValueError("The model needs to be fit before calling this method")
        return self._active_version

    @property
    def version(self):
        """Monotonically increasing id of the currently published version."""
        return self.active_version.version

    @property
    def item_factors(self):
        """Item factors of the active version, bound to its indices."""
        return self.active_version.item_factors

    @property
    def user_factors(self):
        """User factors of the active version."""
        return self.active_version.user_factors

    @property
    def similar_items_index(self):
        """ANN index used for similar_items lookups in the active version."""
        return self.active_version.similar_items_index

    @property
    def recommend_index(self):
        """ANN index used for recommend lookups in the active version."""
        return self.active_version.recommend_index

    @property
    def max_norm(self):
        """Inner product rescaling norm of the active version."""
        return self.active_version.max_norm

    # ------------------------------------------------------------------
    # training / version lifecycle
    # ------------------------------------------------------------------
    def fit(self, user_items, show_progress=True, callback=None):
        """Trains the underlying model and atomically publishes a matching index.

        The model is trained first, then a full set of indices is built from the freshly
        trained factors. The new :class:`ANNModelVersion` only replaces the previously
        published one once every index has built successfully, so a failure during the
        index build leaves the previous consistent view in place.
        """
        self.model.fit(user_items, show_progress=show_progress, callback=callback)
        # a full refit is a fresh model of the (possibly changed) catalogue: drop the
        # availability history so timestamps can't leak for items that no longer exist
        self._item_available_at = {}
        self._publish_new_version(show_progress=show_progress)

    def partial_fit_items(self, itemids, item_users, show_progress=False):
        """Incrementally updates item factors then atomically rebuilds the indices.

        The underlying model's :meth:`partial_fit_items` is called first to update (or
        add) the factors for ``itemids`` -- the model array changes immediately, but the
        online indices keep serving the previous version until the rebuild finishes.

        The indices are rebuilt off to the side from the updated factors. On success the
        new version is published in a single atomic swap; if the rebuild raises, the
        previous version stays live and the exception is re-raised to the caller.

        Parameters
        ----------
        itemids : array_like
            An array of itemids to calculate new factors for.
        item_users : csr_matrix
            Sparse matrix containing the liked users for each item in itemids.
        show_progress : bool, optional
            Whether to show a progress bar while rebuilding the indices.
        """
        self.model.partial_fit_items(itemids, item_users)
        try:
            self._publish_new_version(show_progress=show_progress)
        except Exception:
            log.exception(
                "Failed rebuilding ANN indices after partial_fit_items; "
                "keeping version %s live",
                self._active_version.version if self._active_version else None,
            )
            raise

    def rebuild_index(self, show_progress=False):
        """Rebuilds the ANN indices from the current model factors.

        Useful after directly mutating the underlying ``self.model`` outside of
        :meth:`fit` / :meth:`partial_fit_items`. As with the other update paths, a failed
        rebuild leaves the previously published version untouched.

        Returns
        -------
        int
            The version id that was published (unchanged ids are not reused).
        """
        self._publish_new_version(show_progress=show_progress)
        return self.version

    def _publish_new_version(self, show_progress=False):
        item_factors = self._snapshot_item_factors()
        user_factors = self._snapshot_user_factors()

        # record when items first become recommendable: an item is recommendable from
        # the moment its latent factor is non zero and it is included in a built index.
        now = time.time()
        norms = np.linalg.norm(item_factors, axis=1)
        for itemid in np.flatnonzero(norms != 0):
            self._item_available_at.setdefault(int(itemid), now)

        # build everything off to the side against local variables; nothing here touches
        # the published version until the swap at the end.
        similar_items_index = None
        recommend_index = None
        max_norm = None
        if self.approximate_similar_items:
            similar_items_index = self._build_similar_items_index(item_factors, show_progress)
        if self.approximate_recommend:
            recommend_index, max_norm = self._build_recommend_index(item_factors, show_progress)

        version = ANNModelVersion(
            version=self._next_version,
            item_factors=item_factors,
            user_factors=user_factors,
            similar_items_index=similar_items_index,
            recommend_index=recommend_index,
            max_norm=max_norm,
            built_at=now,
            item_available_at=self._item_available_at,
        )
        self._next_version += 1

        # single atomic publish point: readers either see the old version or the complete
        # new one, never a half built index.
        with self._version_lock:
            self._active_version = version

    def _snapshot_item_factors(self):
        factors = self.model.item_factors
        if implicit.gpu.HAS_CUDA and isinstance(factors, implicit.gpu.Matrix):
            factors = factors.to_numpy()
        return np.ascontiguousarray(np.copy(factors), dtype=np.float32)

    def _snapshot_user_factors(self):
        factors = self.model.user_factors
        if implicit.gpu.HAS_CUDA and isinstance(factors, implicit.gpu.Matrix):
            factors = factors.to_numpy()
        return np.ascontiguousarray(np.copy(factors), dtype=np.float32)

    # ------------------------------------------------------------------
    # hooks implemented by the annoy / nmslib / faiss backends
    # ------------------------------------------------------------------
    def _build_similar_items_index(self, item_factors, show_progress):
        raise NotImplementedError()

    def _build_recommend_index(self, item_factors, show_progress):
        """Returns a tuple of (recommend_index, max_norm)."""
        raise NotImplementedError()

    # ------------------------------------------------------------------
    # new item observability
    # ------------------------------------------------------------------
    def item_recommendable_at(self, itemid):
        """Returns when an item first became recommendable.

        Parameters
        ----------
        itemid : int
            The itemid to look up.

        Returns
        -------
        float or None
            The epoch timestamp (as returned by :func:`time.time`) of the index version
            in which the item's factor was first non zero and therefore searchable.
            Returns None if the item has never been part of a successfully built index
            (for instance because it has no interactions yet and a zero factor, or
            because every rebuild since it was added has failed).
        """
        return self._item_available_at.get(int(itemid))

    def recommendable_items(self):
        """Returns the itemids that are currently recommendable, with their timestamps.

        Returns
        -------
        dict
            Mapping of itemid -> epoch timestamp of the first version in which the item
            had a non zero factor.
        """
        return dict(self._item_available_at)

    # ------------------------------------------------------------------
    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError(
            "similar_users isn't implemented with ANN models yet. (note you can call "
            "self.model.similar_users to get the same functionality on the inner model class)"
        )

    def save(self, file):
        raise NotImplementedError(".save isn't implemented for ANN models yet")

    @classmethod
    def load(cls, file):
        raise NotImplementedError(".load isn't implemented for ANN models yet")
