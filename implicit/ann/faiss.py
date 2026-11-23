import logging
import warnings

import faiss
import numpy as np
from scipy.sparse import csr_matrix

import implicit.gpu
from implicit.ann.base import ANNModelBase
from implicit.utils import _batch_call, _filter_items_from_results

log = logging.getLogger("implicit")


# pylint:  disable=no-value-for-parameter


class FaissModel(ANNModelBase):
    """
    Speeds up inference calls to MatrixFactorization models by using
    `Faiss <https://github.com/facebookresearch/faiss>`_ to create approximate nearest neighbours
    indices of the latent factors.

    The factors searched by the indices are bound together into an immutable version
    snapshot (see :class:`implicit.ann.base.ANNModelVersion`). Refitting the model or
    calling :meth:`partial_fit_items` builds a fresh set of indices off to the side and
    atomically swaps in a new version, so online queries always use indices built from
    the factors in the same version -- a failed rebuild leaves the previous version live.

    Parameters
    ----------
    model : MatrixFactorizationBase
        A matrix factorization model to use for the factors
    nlist : int, optional
        The number of cells to use when building the Faiss index.
    nprobe : int, optional
        The number of cells to visit to perform a search.
    use_gpu : bool, optional
        Whether or not to enable run Faiss on the GPU. Requires faiss to have been
        built with GPU support.
    approximate_similar_items : bool, optional
        whether or not to build an Faiss index for computing similar_items
    approximate_recommend : bool, optional
        whether or not to build an Faiss index for the recommend call

    Attributes
    ----------
    similar_items_index : faiss.IndexIVFFlat
        Faiss index of the currently published version, looking up similar items in the
        cosine space formed by the latent item_factors

    recommend_index : faiss.IndexIVFFlat
        Faiss index of the currently published version, looking up similar items in the
        inner product space formed by the latent item_factors
    """

    def __init__(
        self,
        model,
        approximate_similar_items=True,
        approximate_recommend=True,
        nlist=400,
        nprobe=20,
        use_gpu=None,
    ):
        if use_gpu is None:
            use_gpu = implicit.gpu.HAS_CUDA

        super().__init__(
            model,
            approximate_similar_items=approximate_similar_items,
            approximate_recommend=approximate_recommend,
        )
        # hyper-parameters for FAISS
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu

    def _create_index(self, factors, gpu_resources):
        if self.use_gpu:
            return faiss.GpuIndexIVFFlat(
                gpu_resources, factors, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
        quantizer = faiss.IndexFlat(factors)
        return faiss.IndexIVFFlat(quantizer, factors, self.nlist, faiss.METRIC_INNER_PRODUCT)

    def _build_recommend_index(self, item_factors, show_progress):
        log.debug("Building faiss recommendation index")
        gpu_resources = faiss.StandardGpuResources() if self.use_gpu else None

        # build up an inner product index here
        index = self._create_index(item_factors.shape[1], gpu_resources)
        index.train(item_factors)
        index.add(item_factors)
        index.nprobe = self.nprobe

        # faiss searches raw inner products, no angular transformation / max_norm needed
        return index, None

    def _build_similar_items_index(self, item_factors, show_progress):
        log.debug("Building faiss similar items index")
        gpu_resources = faiss.StandardGpuResources() if self.use_gpu else None

        # build up cosine index for similar_items, using an inner product
        # index on normalized vectors
        norms = np.linalg.norm(item_factors, axis=1)
        norms[norms == 0] = 1e-10

        normalized = (item_factors.T / norms).T.astype("float32")
        index = self._create_index(item_factors.shape[1], gpu_resources)
        index.train(normalized)
        index.add(normalized)
        index.nprobe = self.nprobe
        return index

    def similar_items(
        self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        if items is not None and self.approximate_similar_items:
            raise NotImplementedError("using an items filter isn't supported with ANN lookup")

        count = N
        if filter_items is not None:
            count += len(filter_items)

        if not self.approximate_similar_items or (self.use_gpu and count >= 1024):
            return self.model.similar_items(
                itemid,
                N,
                recalculate_item=recalculate_item,
                item_users=item_users,
                filter_items=filter_items,
                items=items,
            )

        version = self.active_version

        # support recalculate_item if possible. TODO: refactor this
        if hasattr(self.model, "_item_factor"):
            factors = self.model._item_factor(itemid, item_users, recalculate_item)  # pylint: disable=protected-access
        elif recalculate_item:
            raise NotImplementedError(f"recalculate_item isn't supported with {self.model}")
        else:
            # use the factor snapshot bound to the index being queried
            factors = version.item_factors[itemid]

        if np.isscalar(itemid):
            factors /= np.linalg.norm(factors)
            factors = factors.reshape(1, -1)
        else:
            factors /= np.linalg.norm(factors, axis=1)[:, None]

        scores, ids = version.similar_items_index.search(factors.astype("float32"), count)

        if np.isscalar(itemid):
            ids, scores = ids[0], scores[0]

        if filter_items is not None:
            ids, scores = _filter_items_from_results(itemid, ids, scores, filter_items, N)

        return ids, scores

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
        if (filter_already_liked_items or recalculate_user) and not isinstance(
            user_items, csr_matrix
        ):
            raise ValueError("user_items needs to be a CSR sparse matrix")

        if items is not None and self.approximate_recommend:
            raise NotImplementedError("using a 'items' list with ANN search isn't supported")

        # batch computation is tricky with filter_already_liked_items (requires querying a
        # different number of rows per user). Instead just fallback to a faiss query per user
        if filter_already_liked_items and not np.isscalar(userid):
            return _batch_call(
                self.recommend,
                userid,
                user_items=user_items,
                N=N,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=filter_items,
                recalculate_user=recalculate_user,
                items=items,
            )

        if not self.approximate_recommend:
            warnings.warn(
                "Calling recommend on a FaissModel with approximate_recommend=False",
                stacklevel=2,
            )
            return self.model.recommend(
                userid,
                user_items,
                N=N,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=filter_items,
                recalculate_user=recalculate_user,
                items=items,
            )

        version = self.active_version

        # support recalculate_user if possible (TODO: come back to this since its a bit of a hack)
        if hasattr(self.model, "_user_factor"):
            user = self.model._user_factor(userid, user_items, recalculate_user)  # pylint: disable=protected-access
        elif recalculate_user:
            raise NotImplementedError(f"recalculate_user isn't supported with {self.model}")
        else:
            # use the factor snapshot bound to the index being queried
            user = version.user_factors[userid]

        # calculate the top N items, removing the users own liked items from
        # the results
        count = N
        if filter_items:
            count += len(filter_items)
            filter_items = np.array(filter_items)

        if filter_already_liked_items:
            user_likes = user_items[0].indices
            filter_items = (
                np.append(filter_items, user_likes) if filter_items is not None else user_likes
            )
            count += len(user_likes)

        # the GPU variant of faiss doesn't support returning more than 1024 results.
        # fall back to the exact match when this happens
        if self.use_gpu and count >= 1024:
            return self.model.recommend(
                userid,
                user_items,
                N=N,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=filter_items,
                recalculate_user=recalculate_user,
                items=items,
            )

        if np.isscalar(userid):
            query = user.reshape(1, -1).astype("float32")
        else:
            query = user.astype("float32")

        scores, ids = version.recommend_index.search(query, count)

        if np.isscalar(userid):
            ids, scores = ids[0], scores[0]

        if filter_items is not None:
            ids, scores = _filter_items_from_results(userid, ids, scores, filter_items, N)

        return ids, scores
