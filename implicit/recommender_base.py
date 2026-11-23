"""Base class for recommendation algorithms in this package"""

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix

from .id_map import (
    IdMap,
    build_interaction_matrix,
    dedupe_keys,
    map_indices_to_keys,
    resolve_indices,
)


class ModelFitError(Exception):
    pass


class RecommenderBase(metaclass=ABCMeta):
    """Defines a common interface for all recommendation models"""

    # Optional stable business-key maps. Models that were trained on plain integer indices
    # leave these as None; models trained from (or attached to) an IdMap expose them so
    # that update/recommend calls can be made by business key and survive a save/load.
    user_id_map = None
    item_id_map = None

    def _ensure_id_map(self, kind, size):
        """Return the ``kind`` IdMap, seeding it with integer keys if the model predates maps.

        Seeding with ``range(size)`` is what lets an older model trained on positional
        indices accept its first business key: existing rows keep their index, and the new
        key is appended *after* every existing row instead of colliding with it.
        """
        id_map = getattr(self, f"{kind}_id_map", None)
        if id_map is None:
            id_map = IdMap(range(int(size)))
            setattr(self, f"{kind}_id_map", id_map)
        return id_map

    def set_id_maps(self, user_id_map=None, item_id_map=None):
        """Attach stable business-key maps to this model.

        Either argument may be an :class:`~implicit.id_map.IdMap`, a sequence of keys
        (indices assigned in order), or ``None`` to leave the existing map untouched.
        """
        if user_id_map is not None:
            self.user_id_map = _to_id_map(user_id_map)
        if item_id_map is not None:
            self.item_id_map = _to_id_map(item_id_map)
        return self

    def _num_users(self):
        factors = getattr(self, "user_factors", None)
        return None if factors is None else factors.shape[0]

    def _num_items(self):
        factors = getattr(self, "item_factors", None)
        return None if factors is None else factors.shape[0]

    def _resolve_user_ids(self, userids, allow_new=False):
        id_map = self.user_id_map
        if allow_new and id_map is None:
            size = self._num_users()
            if size is not None:
                id_map = self._ensure_id_map("user", size)
        return resolve_indices(id_map, userids, num_known=self._num_users(), allow_new=allow_new)

    def _resolve_item_ids(self, itemids, allow_new=False):
        id_map = self.item_id_map
        if allow_new and id_map is None:
            size = self._num_items()
            if size is not None:
                id_map = self._ensure_id_map("item", size)
        return resolve_indices(id_map, itemids, num_known=self._num_items(), allow_new=allow_new)

    def _map_user_ids(self, ids):
        return map_indices_to_keys(self.user_id_map, ids)

    def _map_item_ids(self, ids):
        return map_indices_to_keys(self.item_id_map, ids)

    def _resolve_user_items(self, user_items):
        """Validate that ``user_items`` is expressed in the model's item index space.

        Keyed models identify items by position in ``item_id_map`` (which is also the row
        order of ``item_factors``). A matrix built with a different map - for example one
        that has had new SKUs appended since the model was last updated - cannot be
        aligned column-by-column, so we reject it rather than silently rank the wrong
        items.
        """
        if self.item_id_map is None or len(self.item_id_map) == 0:
            return user_items
        if not isinstance(user_items, csr_matrix):
            return user_items

        self._validate_user_items_columns(user_items, "user_items")
        return user_items

    def _validate_user_items_columns(self, user_items, kind="user_items"):
        """Error early when an update matrix doesn't line up with a keyed model.

        Only keyed models are checked: an unkeyed model keeps its historical behaviour of
        accepting matrices whose columns are a prefix of the model's positional index
        space (existing tests and callers rely on that).
        """
        if not isinstance(user_items, csr_matrix):
            return
        map_kind = "item" if kind == "user_items" else "user"
        id_map = getattr(self, f"{map_kind}_id_map")
        if id_map is None or len(id_map) == 0:
            return
        expected = self._num_items() if kind == "user_items" else self._num_users()
        if expected is not None and user_items.shape[1] != expected:
            raise ValueError(
                f"{kind} has {user_items.shape[1]} columns but the model has {expected}; "
                f"the columns must be positions in the model's {map_kind}_id_map. If you "
                "have interactions with new entities, update those entities first (e.g. "
                "partial_fit_items_by_key before partial_fit_users_by_key)."
            )

    def fit_from_keys(
        self,
        users,
        items,
        values=None,
        on_duplicate="sum",
        show_progress=True,
        callback=None,
    ):
        """Build an interaction matrix from business keys and fit the model on it.

        Keys already present in ``self.user_id_map`` / ``self.item_id_map`` keep their
        index; new keys are appended. This makes it safe to refit against a re-sorted
        product catalog: an existing SKU never changes row just because the feed order
        changed.

        Parameters
        ----------
        users : sequence
            User key per interaction, or a per-row sequence of user keys.
        items : sequence
            Item key per interaction, or a per-row sequence of item keys.
        values : sequence or float, optional
            Interaction weights. Defaults to ``1.0``.
        on_duplicate : {"sum", "max", "error", "last", "first"}, optional
            How to combine repeated (user, item) pairs. See
            :func:`implicit.id_map.build_interaction_matrix`.
        show_progress : bool, optional
            Whether to show a progress bar during fitting.
        callback : Callable, optional
            Callback forwarded to :meth:`fit`.

        Returns
        -------
        self
            The fitted model, with ``user_id_map`` / ``item_id_map`` populated.
        """
        matrix, user_id_map, item_id_map = build_interaction_matrix(
            users,
            items,
            values=values,
            user_id_map=self.user_id_map,
            item_id_map=self.item_id_map,
            on_duplicate=on_duplicate,
        )
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.fit(matrix, show_progress=show_progress, callback=callback)
        return self

    def partial_fit_users_by_key(self, userids, user_items, on_duplicate="error"):
        """Incrementally update user factors, addressing users by business key.

        Existing keys update their existing rows; unseen keys are appended (their factors
        are computed from ``user_items``). Unlike :meth:`partial_fit_users`, integer
        values are *always* treated as business keys here, never as row positions - so an
        integer SKU/user id can't accidentally index a different row.

        Parameters
        ----------
        userids : sequence
            Business keys of the users to update.
        user_items : csr_matrix
            One row per entry in ``userids``. Columns are item positions in the model's
            ``item_id_map``.
        on_duplicate : {"error", "first", "last"}, optional
            What to do when a key is repeated in ``userids``:

            * ``"error"`` (default) raises :class:`ValueError`;
            * ``"first"`` keeps the first matching row;
            * ``"last"`` keeps the last matching row.
        """
        positions = self._dedupe_update_keys(userids, on_duplicate, "user")
        unique_ids, positions = positions
        self._check_update_supported("user")
        indices = self._update_id_map("user", unique_ids)
        rows = user_items[positions]
        return self._partial_fit_positions("user", indices, rows)

    def partial_fit_items_by_key(self, itemids, item_users, on_duplicate="error"):
        """Incrementally update item factors, addressing items by business key.

        Mirror of :meth:`partial_fit_users_by_key` for items. Integer values are always
        treated as business keys.
        """
        unique_ids, positions = self._dedupe_update_keys(itemids, on_duplicate, "item")
        self._check_update_supported("item")
        indices = self._update_id_map("item", unique_ids)
        rows = item_users[positions]
        return self._partial_fit_positions("item", indices, rows)

    def _partial_fit_positions(self, kind, indices, rows):
        """Call the model's positional partial-fit implementation.

        Keyed updates go through here so they bypass the guard in the public
        ``partial_fit_users`` / ``partial_fit_items`` methods (which reject positional
        updates on a keyed model).

        The maps may describe more entities than the factor arrays yet: a key is appended
        to the map as soon as it appears in an interaction matrix, but its factors are
        allocated when it is first updated. When updating users, the matrix columns index
        items, so grow ``item_factors`` to the item map's size (and vice versa). A
        freshly-appended row is all zeros, so it contributes nothing to the solve for the
        rows being updated now - and this removes any ordering requirement between the
        item and user update calls.
        """
        opposite_map = self.item_id_map if kind == "user" else self.user_id_map
        opposite_size = self._num_items() if kind == "user" else self._num_users()
        if opposite_map is not None and opposite_size is not None:
            needed = len(opposite_map)
            if needed > opposite_size:
                if kind == "user":
                    self._grow_factors_for_fit(self._num_users(), needed)
                else:
                    self._grow_factors_for_fit(needed, self._num_items())

        positional = getattr(self, f"_partial_fit_{kind}s", None)
        if positional is not None:
            return positional(indices, rows)
        return getattr(self, f"partial_fit_{kind}s")(indices, rows)

    def _check_update_supported(self, kind):
        """Fail *before* mutating the key map when a model has no incremental update."""
        if not hasattr(self, f"partial_fit_{kind}s"):
            raise NotImplementedError(
                f"{type(self).__name__} doesn't support incremental "
                f"partial_fit_{kind}s updates"
            )

    def _check_positional_update_allowed(self, kind):
        """Reject positional partial-fit calls on a model keyed by business keys.

        This is the guard that makes the mapping worthwhile: once a model identifies rows
        by SKU/user key, ``partial_fit_users([5], ...)`` would silently write into
        whatever entity happens to occupy row 5. Callers must use the ``*_by_key``
        variants so the key decides the row.
        """
        id_map = getattr(self, f"{kind}_id_map")
        if id_map is None or len(id_map) == 0 or id_map.is_positional():
            return
        raise ValueError(
            f"this model is keyed by business {kind} keys, so partial_fit_{kind}s can't take "
            f"positional indices; use partial_fit_{kind}s_by_key with the {kind} keys instead"
        )

    def _update_id_map(self, kind, keys):
        id_map = getattr(self, f"{kind}_id_map")
        if id_map is None:
            size = self._num_users() if kind == "user" else self._num_items()
            if size is None:
                raise ValueError(
                    f"can't add {kind} keys to an unfitted model; call fit/fit_from_keys first"
                )
            id_map = self._ensure_id_map(kind, size)
        return id_map.add_many(keys)

    def _check_positional_growth(self, kind, max_position):
        """Reject positional growth on a model keyed by non-integer business keys."""
        id_map = getattr(self, f"{kind}_id_map")
        if id_map is None or len(id_map) == 0 or max_position < len(id_map):
            return
        if id_map.is_positional():
            return
        raise ValueError(
            f"can't grow {kind} factors by position on a model keyed by business keys; "
            f"use partial_fit_{kind}s_by_key with the new {kind} keys instead"
        )

    def _sync_positional_id_map(self, kind, new_size, max_position):
        """Keep a placeholder integer map in sync when factors are grown positionally.

        An unkeyed model that grows via ``partial_fit_users([100], ...)`` still has an
        index space, so we record positions up to the new size as integer keys.
        """
        id_map = getattr(self, f"{kind}_id_map")
        if id_map is None or max_position < len(id_map):
            return
        if len(id_map) == 0 or id_map.is_positional():
            id_map.add_many(range(len(id_map), new_size))

    @staticmethod
    def _dedupe_update_keys(keys, on_duplicate, name):
        """Validate business keys handed to a partial-fit call.

        Partial updates receive one matrix row per key. If the same key appears twice the
        intent is ambiguous (two different rows would overwrite the same factors), so by
        default we raise rather than silently pick one. Callers can opt in to
        ``on_duplicate='first'`` or ``'last'`` to select a row explicitly.

        Returns ``(unique_keys, positions)`` where ``positions`` selects the rows to keep.
        """
        if on_duplicate not in ("error", "first", "last"):
            raise ValueError(
                "on_duplicate must be one of 'error', 'first' or 'last', "
                f"got {on_duplicate!r}"
            )
        # a bare string is a single business key, not a sequence of characters; likewise a
        # numpy scalar is a single key
        if isinstance(keys, (str, bytes)) or np.isscalar(keys):
            keys = [keys]
        return dedupe_keys(keys, on_duplicate=on_duplicate, name=name)

    @abstractmethod
    def fit(self, user_items, show_progress=True, callback=None):
        """
        Trains the model on a sparse matrix of user/item/weight

        Parameters
        ----------
        user_items : csr_matrix
            A sparse CSR matrix of shape (number_of_users, number_of_items). The nonzero
            entries in this matrix are the items that are liked by each user.
            The values are how confident you are that the item is liked by the user.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        callback: Callable, optional
            Callable function on each epoch with such arguments as epoch, elapsed time and progress
        """

    @abstractmethod
    def recommend(
        self,
        userid,
        user_items,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
        items=None,
    ):
        """
        Recommends items for users.

        This method allows you to calculate the top N recommendations for a user or
        batch of users. Passing an array of userids instead of a single userid will
        tend to be more efficient, and allows multi-thread processing on the CPU.

        This method has options for filtering out items from the results. You can both
        filter out items that have already been liked by the user with the
        filter_already_liked_items parameter, as well as pass in filter_items to filter
        out other items for all users in the batch. By default all items in the training
        dataset are scored, but by setting the 'items' parameter you can restrict down to
        a subset.

        Example usage::

            # calculate the top recommendations for a single user
            ids, scores = model.recommend(0, user_items[0])

            # calculate the top recommendations for a batch of users
            userids = np.arange(10)
            ids, scores = model.recommend(userids, user_items[userids])

        Parameters
        ----------
        userid : Union[int, array_like]
            The userid or array of userids to calculate recommendations for. When the
            model has a ``user_id_map`` (it was trained from business keys) pass the
            business keys here instead; results are returned as business keys too.
        user_items : csr_matrix
            A sparse matrix of shape (users, number_items). This lets us look
            up the liked items and their weights for the user. This is used to filter out
            items that have already been liked from the output, and to also potentially
            recalculate the user representation. Each row in this sparse matrix corresponds
            to a row in the userid parameter: that is the first row in this matrix contains
            the liked items for the first user in the userid array. When the model has key
            maps, the columns are item positions in the model (see
            :func:`implicit.id_map.build_interaction_matrix`).
        N : int, optional
            The number of results to return
        filter_already_liked_items: bool, optional
            When true, don't return items present in the training set that were rated
            by the specified user.
        filter_items : array_like, optional
            List of extra item ids to filter out from the output. Business keys are
            accepted when the model has an ``item_id_map``.
        recalculate_user : bool, optional
            When true, don't rely on stored user embeddings and instead recalculate from the
            passed in user_items. This option isn't supported by all models.
        items: array_like, optional
            Array of extra item ids. When set this will only rank the items in this array instead
            of ranking every item the model was fit for. This parameter cannot be used with
            filter_items. Business keys are accepted when the model has an ``item_id_map``.

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays. When calculating for a single user these array will
            be 1-dimensional with N items. When passed an array of userids, these will be
            2-dimensional arrays with a row for each user. If the model was trained from
            business keys, the item ids are returned as an ``object`` array of those keys.
        """

    @abstractmethod
    def similar_users(self, userid, N=10, filter_users=None, users=None):
        """
        Calculates the most similar users for a userid or array of userids

        Parameters
        ----------
        userid : Union[int, array_like]
            The userid or an array of userids to retrieve similar users for.
        N : int, optional
            The number of similar users to return
        filter_users: array_like, optional
            An array of user ids to filter out from the results being returned
        users: array_like, optional
            An array of user ids to include in the output. If not set all users in the training
            set will be included. Cannot be used with the filter_users options

        Returns
        -------
        tuple
            Tuple of (userids, scores) arrays
        """

    @abstractmethod
    def similar_items(
        self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        """
        Calculates a list of similar items

        Parameters
        ----------
        itemid : Union[int, array_like]
            The item id or an array of item ids to retrieve similar items for
        N : int, optional
            The number of similar items to return
        recalculate_item : bool, optional
            When true, don't rely on stored item state and instead recalculate from the
            passed in item_users
        item_users : csr_matrix, optional
            A sparse matrix of shape (itemid, number_users). This lets us look
            up the users for each item. This is only needs to be set when setting
            recalculate_item=True. This should have the same number of rows as
            the itemid parameter, with the first row of the sparse matrix corresponding
            to the first item in the itemid array.
        filter_items: array_like, optional
            An array of item ids to filter out from the results being returned
        items: array_like, optional
            An array of item ids to include in the output. If not set all items in the training
            set will be included. Cannot be used with the filter_items options

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays
        """

    @abstractmethod
    def save(self, file):
        """Saves the model to a file, using the numpy `.npz` format

        Parameters
        ----------
        file : str or io.IOBase
            Either the filename or an open file-like object to save the model to

        See Also
        --------
        load
        numpy.savez
        """

    @classmethod
    def load(cls, fileobj_or_path) -> "RecommenderBase":
        """Loads the model from a file

        Parameters
        ----------
        fileobj_or_path : str or io.IOBase
            Either the filename or an open file-like object to load the model from

        Returns
        -------
        RecommenderBase
            The model loaded up from disk

        See Also
        --------
        save
        numpy.load
        """
        if isinstance(fileobj_or_path, str) and not fileobj_or_path.endswith(".npz"):
            fileobj_or_path = fileobj_or_path + ".npz"
        with np.load(fileobj_or_path, allow_pickle=False) as data:
            ret = cls()
            for k, v in data.items():
                if v.shape == ():
                    v = v.item()
                if k == "dtype":
                    v = np.dtype(str(v))
                elif k in ("user_id_map", "item_id_map"):
                    v = IdMap.from_json(v)
                setattr(ret, k, v)
            return ret

    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
        warnings.warn(
            "rank_items is deprecated. Use recommend with the 'items' parameter instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.recommend(
            userid,
            user_items,
            recalculate_user=recalculate_user,
            items=selected_items,
            filter_already_liked_items=False,
        )

    def _grow_factors_for_fit(self, num_users, num_items):
        """Pad factor arrays so a refit can cover newly appended keys.

        ``fit`` normally leaves already-initialized factors untouched. For a keyed model
        that has had keys appended since it was last trained, the matrix is now larger than
        the factor arrays; preserve the existing rows and append zero rows, which the
        solver fills in during training.

        Returns
        -------
        tuple
            ``(previous_num_users, previous_num_items)`` so callers can fix up any
            per-model bias columns in the appended rows.
        """
        previous = [None, None]
        for index, (name, num_rows) in enumerate(
            (("user_factors", num_users), ("item_factors", num_items))
        ):
            factors = getattr(self, name, None)
            if factors is None:
                continue
            previous[index] = factors.shape[0]
            if factors.shape[0] >= num_rows:
                continue
            if isinstance(factors, np.ndarray):
                setattr(
                    self,
                    name,
                    np.concatenate(
                        [
                            factors,
                            np.zeros(
                                (num_rows - factors.shape[0], factors.shape[1]),
                                dtype=factors.dtype,
                            ),
                        ]
                    ),
                )
            elif hasattr(factors, "resize"):
                # GPU Matrix: resize zero-fills the appended rows
                factors.resize(num_rows, factors.shape[1])
            else:
                raise TypeError(
                    f"don't know how to grow {name} of type {type(factors).__name__}"
                )
        return previous[0], previous[1]

    def _check_factors(self, user_factors, item_factors):
        is_nan = np.any(np.isnan(user_factors), axis=None)
        is_nan |= np.any(np.isnan(item_factors), axis=None)
        if is_nan:
            raise ModelFitError("NaN encountered in factors")

    def _save_id_maps(self, args):
        """Add ``user_id_map`` / ``item_id_map`` to a ``np.savez`` kwargs dict.

        Maps are JSON strings, which keeps the saved model loadable with
        ``allow_pickle=False`` (integer indices and arbitrary pickled keys would otherwise
        be a code-execution hazard on load).
        """
        for kind in ("user", "item"):
            id_map = getattr(self, f"{kind}_id_map")
            if id_map is not None and len(id_map):
                args[f"{kind}_id_map"] = id_map.to_json()
        return args


def _to_id_map(value):
    if isinstance(value, IdMap):
        return value
    return IdMap(value)


class IdMapDelegatingMixin:
    """Delegate ``user_id_map`` / ``item_id_map`` to an inner ``model``.

    ANN wrappers keep the trained factors on an inner matrix-factorization model; exposing
    the inner model's key maps (rather than keeping a second copy) keeps key resolution and
    ``fit_from_keys`` consistent across the wrapper and the model it wraps.
    """

    @property
    def user_id_map(self):
        return self.model.user_id_map

    @user_id_map.setter
    def user_id_map(self, value):
        self.model.user_id_map = value

    @property
    def item_id_map(self):
        return self.model.item_id_map

    @item_id_map.setter
    def item_id_map(self, value):
        self.model.item_id_map = value

    def _num_users(self):
        return self.model._num_users()  # pylint: disable=protected-access

    def _num_items(self):
        return self.model._num_items()  # pylint: disable=protected-access

    def _resolve_user_items(self, user_items):
        return self.model._resolve_user_items(user_items)  # pylint: disable=protected-access

