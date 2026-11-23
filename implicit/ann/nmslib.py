import logging

import nmslib
import numpy as np
from scipy.sparse import csr_matrix

from implicit.ann.base import ANNModelBase
from implicit.utils import _batch_call, _filter_items_from_results, augment_inner_product_matrix

log = logging.getLogger("implicit")


class NMSLibModel(ANNModelBase):
    """Speeds up inference calls to MatrixFactorization models by using
    `NMSLib <https://github.com/nmslib/nmslib>`_ to create approximate nearest neighbours
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
    method : str, optional
        The NMSLib method to use
    index_params: dict, optional
        Optional params to send to the createIndex call in NMSLib
    query_params: dict, optional
        Optional query time params for the NMSLib 'setQueryTimeParams' call
    approximate_similar_items : bool, optional
        whether or not to build an NMSLIB index for computing similar_items
    approximate_recommend : bool, optional
        whether or not to build an NMSLIB index for the recommend call

    Attributes
    ----------
    similar_items_index : nmslib.FloatIndex
        NMSLib index of the currently published version, looking up similar items in the
        cosine space formed by the latent item_factors

    recommend_index : nmslib.FloatIndex
        NMSLib index of the currently published version, looking up similar items in the
        inner product space formed by the latent item_factors
    """

    def __init__(
        self,
        model,
        approximate_similar_items=True,
        approximate_recommend=True,
        method="hnsw",
        index_params=None,
        query_params=None,
        **kwargs,
    ):
        super().__init__(
            model,
            approximate_similar_items=approximate_similar_items,
            approximate_recommend=approximate_recommend,
        )
        if index_params is None:
            index_params = {"M": 16, "post": 0, "efConstruction": 400}
        if query_params is None:
            query_params = {"ef": 90}

        self.method = method
        self.index_params = index_params
        self.query_params = query_params

    def _build_similar_items_index(self, item_factors, show_progress):
        # nmslib can be a little chatty when first imported, disable some of
        # the logging
        logging.getLogger("nmslib").setLevel(logging.WARNING)

        log.debug("Building nmslib similar items index")
        index = nmslib.init(method=self.method, space="cosinesimil")

        # there are some numerical instability issues here with
        # building a cosine index with vectors with 0 norms, hack around this
        # by just not indexing them
        norms = np.linalg.norm(item_factors, axis=1)
        ids = np.arange(item_factors.shape[0])

        # delete zero valued rows from the matrix
        nonzero_item_factors = np.delete(item_factors, ids[norms == 0], axis=0)
        ids = ids[norms != 0]

        index.addDataPointBatch(nonzero_item_factors, ids=ids)
        index.createIndex(self.index_params, print_progress=show_progress)
        index.setQueryTimeParams(self.query_params)
        return index

    def _build_recommend_index(self, item_factors, show_progress):
        logging.getLogger("nmslib").setLevel(logging.WARNING)

        log.debug("Building nmslib recommendation index")
        max_norm, extra = augment_inner_product_matrix(item_factors)
        index = nmslib.init(method="hnsw", space="cosinesimil")
        index.addDataPointBatch(extra)
        index.createIndex(self.index_params, print_progress=show_progress)
        index.setQueryTimeParams(self.query_params)
        return index, max_norm

    def similar_items(
        self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        if not self.approximate_similar_items:
            return self.model.similar_items(
                itemid,
                N,
                item_users=item_users,
                recalculate_item=recalculate_item,
                filter_items=filter_items,
                items=items,
            )

        if items is not None:
            raise NotImplementedError("using an items filter isn't supported with ANN lookup")

        version = self.active_version

        # support recalculate_item if possible. TODO: refactor this
        if hasattr(self.model, "_item_factor"):
            factors = self.model._item_factor(itemid, item_users, recalculate_item)  # pylint: disable=protected-access
        elif recalculate_item:
            raise NotImplementedError(f"recalculate_item isn't supported with {self.model}")
        else:
            # use the factor snapshot bound to the index being queried
            factors = version.item_factors[itemid]

        count = N
        if filter_items is not None:
            count += len(filter_items)

        if np.isscalar(itemid):
            ids, scores = version.similar_items_index.knnQuery(factors, count)
        else:
            results = version.similar_items_index.knnQueryBatch(factors, count)
            ids = np.stack([result[0] for result in results])
            scores = np.stack([result[1] for result in results])

        scores = 1.0 - scores
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

        if not self.approximate_recommend:
            return self.model.recommend(
                userid,
                user_items,
                N=N,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=filter_items,
                recalculate_user=recalculate_user,
                items=items,
            )

        # batch computation is hard here, fallback to looping over items
        if not np.isscalar(userid):
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

        query = np.append(user, 0)
        ids, scores = version.recommend_index.knnQuery(query, count)
        scaling = version.max_norm * np.linalg.norm(query)
        scores = scaling * (1.0 - (scores))

        if filter_items is not None:
            ids, scores = _filter_items_from_results(userid, ids, scores, filter_items, N)

        return ids, scores
