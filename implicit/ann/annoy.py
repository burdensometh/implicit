import logging

import annoy
import numpy as np
from scipy.sparse import csr_matrix

from implicit.ann.base import ANNModelBase
from implicit.utils import _batch_call, _filter_items_from_results, augment_inner_product_matrix

log = logging.getLogger("implicit")


class AnnoyModel(ANNModelBase):
    """Speeds up inference calls to MatrixFactorization models by using an
    `Annoy <https://github.com/spotify/annoy>`_ index to calculate similar items and
    recommend items.

    The factors searched by the indices are bound together into an immutable version
    snapshot (see :class:`implicit.ann.base.ANNModelVersion`). Refitting the model or
    calling :meth:`partial_fit_items` builds a fresh set of indices off to the side and
    atomically swaps in a new version, so online queries always use indices built from
    the factors in the same version -- a failed rebuild leaves the previous version live.

    Parameters
    ----------
    model : MatrixFactorizationBase
        A matrix factorization model to use for the factors
    n_trees : int, optional
        The number of trees to use when building the Annoy index. More trees gives higher precision
        when querying.
    search_k : int, optional
        Provides a way to search more trees at runtime, giving the ability to have more accurate
        results at the cost of taking more time.
    approximate_similar_items : bool, optional
        whether or not to build an Annoy index for computing similar_items
    approximate_recommend : bool, optional
        whether or not to build an Annoy index for the recommend call

    Attributes
    ----------
    similar_items_index : annoy.AnnoyIndex
        Annoy index of the currently published version, looking up similar items in the
        cosine space formed by the latent item_factors
    recommend_index : annoy.AnnoyIndex
        Annoy index of the currently published version, looking up similar items in the
        inner product space formed by the latent item_factors
    """

    def __init__(
        self,
        model,
        approximate_similar_items=True,
        approximate_recommend=True,
        n_trees=50,
        search_k=-1,
    ):
        super().__init__(
            model,
            approximate_similar_items=approximate_similar_items,
            approximate_recommend=approximate_recommend,
        )
        self.n_trees = n_trees
        self.search_k = search_k

    def _build_similar_items_index(self, item_factors, show_progress):
        log.debug("Building annoy similar items index")
        index = annoy.AnnoyIndex(item_factors.shape[1], "angular")
        for i, row in enumerate(item_factors):
            index.add_item(i, row)
        index.build(self.n_trees)
        return index

    def _build_recommend_index(self, item_factors, show_progress):
        log.debug("Building annoy recommendation index")
        max_norm, extra = augment_inner_product_matrix(item_factors)
        index = annoy.AnnoyIndex(extra.shape[1], "angular")
        for i, row in enumerate(extra):
            index.add_item(i, row)
        index.build(self.n_trees)
        return index, max_norm

    def similar_items(
        self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        if items is not None and self.approximate_similar_items:
            raise NotImplementedError("using an items filter isn't supported with ANN lookup")

        count = N
        if filter_items is not None:
            count += len(filter_items)

        if not self.approximate_similar_items:
            return self.model.similar_items(
                itemid,
                N,
                recalculate_item=recalculate_item,
                item_users=item_users,
                filter_items=filter_items,
                items=items,
            )

        # annoy doesn't have a batch mode we can use
        if not np.isscalar(itemid):
            return _batch_call(
                self.similar_items,
                itemid,
                N=N,
                recalculate_item=recalculate_item,
                item_users=item_users,
                filter_items=filter_items,
            )

        version = self.active_version

        # support recalculate_item if possible. TODO: refactor this
        if hasattr(self.model, "_item_factor"):
            factor = self.model._item_factor(itemid, item_users, recalculate_item)  # pylint: disable=protected-access
        elif recalculate_item:
            raise NotImplementedError(f"recalculate_item isn't supported with {self.model}")
        else:
            # use the factor snapshot bound to the index being queried
            factor = version.item_factors[itemid]

        if len(factor.shape) != 1:
            factor = factor.squeeze()

        ids, scores = version.similar_items_index.get_nns_by_vector(
            factor, N, search_k=self.search_k, include_distances=True
        )
        ids, scores = np.array(ids), np.array(scores)

        if filter_items is not None:
            ids, scores = _filter_items_from_results(itemid, ids, scores, filter_items, N)

        return ids, 1 - (scores**2) / 2

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

        # batch computation isn't supported by annoy, fallback to looping over items
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
        ids, scores = version.recommend_index.get_nns_by_vector(
            query, count, include_distances=True, search_k=self.search_k
        )
        ids, scores = np.array(ids), np.array(scores)

        if filter_items is not None:
            ids, scores = _filter_items_from_results(userid, ids, scores, filter_items, N)

        # convert the distances from euclidean to cosine distance,
        # and then rescale the cosine distance to go back to inner product
        scaling = version.max_norm * np.linalg.norm(query)
        scores = scaling * (1 - (scores**2) / 2)
        return ids, scores
