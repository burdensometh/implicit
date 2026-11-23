"""Base class for recommendation algorithms in this package"""

import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np


class ModelFitError(Exception):
    pass


# Reasons why a piece of feedback or a target item does not participate in an
# explanation. Keeping these as string constants avoids typos and makes it
# easier to filter on the reason in downstream code.
EXCLUDED_NEGATIVE_FEEDBACK = "negative_feedback"
EXCLUDED_ZERO_CONTRIBUTION = "zero_contribution"
EXCLUDED_FILTERED_TARGET = "filtered_target"


# A single feedback entry (or target item) that was kept out of the score
# decomposition, together with the reason why.
Exclusion = namedtuple("Exclusion", ["itemid", "confidence", "reason"])


class Explanation(
    namedtuple(
        "Explanation",
        [
            "total_score",
            "selected_contributions",
            "other_contributions",
            "excluded",
            "user_factors_source",
            "feedback",
            "stored_user_score",
            "user_weights",
        ],
    )
):
    """Result of explaining a user/item score.

    Behaves like the legacy ``(total_score, top_contributions, user_weights)``
    tuple returned by :meth:`RecommenderBase.explain`, while also exposing the
    full decomposition as named attributes.

    Attributes
    ----------
    total_score : float
        The predicted score for the explained user/item pair, decomposed over
        the feedback actually used by the explanation.
    selected_contributions : list
        The top N ``(itemid, score)`` contributions, sorted by descending
        score. These are the contributions the explanation highlights.
    other_contributions : list
        The remaining participating ``(itemid, score)`` contributions that did
        not fit into the top N, also sorted by descending score. Together with
        ``selected_contributions`` these sum to ``total_score``.
    excluded : list
        :class:`Exclusion` entries for feedback and target items that do not
        participate in the decomposition. Each entry records the ``itemid``,
        the ``confidence`` from the feedback matrix and one of the
        ``EXCLUDED_*`` reason constants. Negative feedback, zero-contribution
        feedback and filtered target items are reported separately here.
    user_factors_source : str
        Which user factor representation this explanation was requested for:
        ``"recalculated"`` when ``explain`` was called with
        ``recalculate_user=True`` (the score is built from factors solved over
        :attr:`feedback`, and :attr:`stored_user_score` holds the comparable
        stored-factors score) or ``"stored"`` (the default) otherwise.
    feedback : csr_matrix
        The (single-row) feedback matrix that was actually used to build the
        explanation, after any preprocessing such as the alpha scaling.
    stored_user_score : float or None
        The score computed from the model's stored user factors for comparison
        (``None`` unless ``recalculate_user=True`` was used or the user has no
        stored factors). Compare this against ``total_score`` to see how much
        the score changes when the user factors are recalculated from the
        supplied feedback.
    user_weights : ndarray
        A factorized representation of the user (the Cholesky decomposition of
        the user linear equation). Passing this back in to future ``explain``
        calls for the same user/feedback speeds them up.
    """

    __slots__ = ()

    def __new__(
        cls,
        total_score,
        selected_contributions,
        other_contributions,
        excluded=None,
        user_factors_source="stored",
        feedback=None,
        stored_user_score=None,
        user_weights=None,
    ):
        return super().__new__(
            cls,
            total_score,
            selected_contributions,
            other_contributions,
            list(excluded) if excluded is not None else [],
            user_factors_source,
            feedback,
            stored_user_score,
            user_weights,
        )

    @property
    def top_contributions(self):
        """Alias for :attr:`selected_contributions`."""
        return self.selected_contributions

    @property
    def contributions(self):
        """All participating contributions (selected and other), in order."""
        return self.selected_contributions + self.other_contributions

    def excluded_by_reason(self, reason):
        """Returns the excluded ``Exclusion`` entries matching ``reason``."""
        return [entry for entry in self.excluded if entry.reason == reason]

    def __iter__(self):
        # backwards compatibility: explain() used to return
        # (total_score, top_contributions, user_weights)
        yield self.total_score
        yield self.selected_contributions
        yield self.user_weights

    def __getitem__(self, index):
        # keep index access consistent with the legacy 3-tuple layout above
        return (self.total_score, self.selected_contributions, self.user_weights)[index]


class RecommenderBase(metaclass=ABCMeta):
    """Defines a common interface for all recommendation models"""

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
            The userid or array of userids to calculate recommendations for
        user_items : csr_matrix
            A sparse matrix of shape (users, number_items). This lets us look
            up the liked items and their weights for the user. This is used to filter out
            items that have already been liked from the output, and to also potentially
            recalculate the user representation. Each row in this sparse matrix corresponds
            to a row in the userid parameter: that is the first row in this matrix contains
            the liked items for the first user in the userid array.
        N : int, optional
            The number of results to return
        filter_already_liked_items: bool, optional
            When true, don't return items present in the training set that were rated
            by the specified user.
        filter_items : array_like, optional
            List of extra item ids to filter out from the output
        recalculate_user : bool, optional
            When true, don't rely on stored user embeddings and instead recalculate from the
            passed in user_items. This option isn't supported by all models.
        items: array_like, optional
            Array of extra item ids. When set this will only rank the items in this array instead
            of ranking every item the model was fit for. This parameter cannot be used with
            filter_items

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays. When calculating for a single user these array will
            be 1-dimensional with N items. When passed an array of userids, these will be
            2-dimensional arrays with a row for each user.
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
                if k == "dtype":
                    v = np.dtype(str(v))
                elif v.shape == ():
                    v = v.item()
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

    @staticmethod
    def _check_factors(user_factors, item_factors):
        is_nan = np.any(np.isnan(user_factors), axis=None)
        is_nan |= np.any(np.isnan(item_factors), axis=None)
        if is_nan:
            raise ModelFitError("NaN encountered in factors")
