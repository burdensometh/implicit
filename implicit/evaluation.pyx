# distutils: language = c++

import cython
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from tqdm.auto import tqdm

from libc.math cimport fmin
from libcpp.unordered_set cimport unordered_set

from .utils import check_random_state


def train_test_split(ratings, train_percentage=0.8, random_state=None):
    """ Randomly splits the ratings matrix into two matrices for training/testing.

    Parameters
    ----------
    ratings : coo_matrix
        A sparse matrix to split
    train_percentage : float
        What percentage of ratings should be used for training
    random_state : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.
    Returns
    -------
    (train, test) : csr_matrix, csr_matrix
        A tuple of csr_matrices for training/testing """

    ratings = ratings.tocoo()
    random_state = check_random_state(random_state)
    random_index = random_state.random(len(ratings.data))
    train_index = random_index < train_percentage
    test_index = random_index >= train_percentage

    train = csr_matrix((ratings.data[train_index],
                        (ratings.row[train_index], ratings.col[train_index])),
                       shape=ratings.shape, dtype=ratings.dtype)

    test = csr_matrix((ratings.data[test_index],
                       (ratings.row[test_index], ratings.col[test_index])),
                      shape=ratings.shape, dtype=ratings.dtype)

    test.data[test.data < 0] = 0
    test.eliminate_zeros()

    return train, test


cdef _choose(rng, int n, float frac):
    """Given a range of numbers, select *approximately* 'frac' of them _without_
    replacement.

    Parameters
    ----------
    rng : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.
    n: int
        The upper bound on the range to sample from. Will draw from range(0 -> n).
    frac: float
        The fraction of the total range to be sampled. Must be in interval (0 -> 1).

    Returns
    -------
    ndarray
        An array of randomly sampled integers in the range (0 -> n).

    """

    size = max(1, int(n * frac))
    arr = rng.choice(n, size=size, replace=False)
    return arr


cdef _take_tails(arr, int n, return_complement=False, shuffled=False):
    """
    Given an array of (optionally shuffled) integers in the range 0->n, take the indices
    of the last 'n' occurrences of each integer (tail) -- subject to shuffling.

    Concretely, given an array of 25 integers in the range 0->4:

    arr = [4 0 1 2 0 3 1 3 3 2 0 3 2 3 3 1 1 2 3 2 3 4 4 0 4]

    Return the index of the last 'n' elements for each unique integer. For n=2:

    idx = [ 4 23 16 15  9  3 18 20 21 24]

    Such that:

    arr[idx] = [0 0 1 1 2 2 3 3 4 4]

    Parameters
    ----------
    arr: ndarray
        The input array. This should be an array of integers in the range 0->n, where
        the ordered unique set of integers in said array should produce an array of
        consecutive integers. Concretely, the array [1, 0, 1, 1, 0, 3] would be invalid,
        but the array [1, 0, 1, 1, 0, 2] would not be.
    n: int
        The number of elements in the tail to take. Note that no checks are made to
        ensure that this value is correct (i.e. that a given integer occurs > n times in
        the given array). Invalid values of 'n' will produce IndexErrors.
    return_complement: bool
        If True, returns the complement (i.e. heads) of the tail indices.
        Default is False (only returns tails).
    shuffled: bool
        Optionally indicate whether you wish the dataset to be shuffled. This will act
        to randomise the 'tails' selected.

    Returns
    -------
    output: ndarray or tuple
        If 'return_complement' is False, this will return an array of integers
        corresponding to the tails

    """

    idx = arr.argsort()
    sorted_arr = arr[idx]

    end = np.bincount(sorted_arr).cumsum() - 1
    start = end - n
    ranges = np.linspace(start, end, num=n + 1, dtype=int)[1:]

    if shuffled:
        shuffled_idx = (sorted_arr + np.random.random(arr.shape)).argsort()
        tails = shuffled_idx[np.ravel(ranges, order="f")]
    else:
        tails = np.ravel(ranges, order="f")

    heads = np.setdiff1d(idx, tails)

    if return_complement:
        return idx[tails], idx[heads]
    else:
        return idx[tails]


cpdef leave_k_out_split(
    ratings, int K=1, float train_only_size=0.0, random_state=None
):
    """Implements the 'leave-k-out' split protocol for a ratings matrix. Default
    parameters will produce a 'leave-one-out' split.

    This will create two matrices, one where each eligible user (i.e. user with > K + 1
    ratings) will have a single rating held in the test set, and all other ratings held
    in the train set. Optionally, a percentage of users can be reserved to appear _only_
    in the train set. By default, all eligible users may appear in the test set.

    Parameters
    ----------
    ratings : csr_matrix
        The input ratings CSR matrix to be split.
    K : int
        The total number of samples to be 'left out' in the test set.
    train_only_size : float
        The size (as a fraction) of the users set that should appear *only* in the
        training matrix.
    random_state : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.

    Returns
    -------
    (train, test) : csr_matrix, csr_matrix
        A tuple of CSR matrix corresponding to training/testing matrices.

    """

    if K < 1:
        raise ValueError("The 'K' must be >= 1.")
    if not 0.0 <= train_only_size < 1.0:
        raise ValueError("The 'train_only_size' must be in the range (0.0 <= x < 1.0).")

    ratings = ratings.tocoo()  # this will sort row/cols unless ratings is COO.
    random_state = check_random_state(random_state)

    users = ratings.row
    items = ratings.col
    data = ratings.data

    unique_users, counts = np.unique(users, return_counts=True)

    # get only users with n + 1 interactions
    candidate_mask = counts > K + 1

    # keep a given subset of users _only_ in the training set.
    if train_only_size > 0.0:
        train_only_mask = ~np.isin(
            unique_users, _choose(random_state, len(unique_users), train_only_size)
        )
        candidate_mask = train_only_mask & candidate_mask

    # get unique users who appear in the test set
    unique_candidate_users = unique_users[candidate_mask]
    full_candidate_mask = np.isin(users, unique_candidate_users)

    # get all users, items and ratings that match specified requirements to be
    # included in test set.
    candidate_users = users[full_candidate_mask]
    candidate_items = items[full_candidate_mask]
    candidate_data = data[full_candidate_mask]

    test_idx, train_idx = _take_tails(
        candidate_users, K, shuffled=True, return_complement=True
    )

    # get all remaining remaining candidate user-item pairs, and prepare to append to
    # training set.
    train_idx = np.setdiff1d(np.arange(len(candidate_users), dtype=int), test_idx)

    # build test matrix
    test_users = candidate_users[test_idx]
    test_items = candidate_items[test_idx]
    test_data = candidate_data[test_idx]
    test_mat = csr_matrix(
        (test_data, (test_users, test_items)), shape=ratings.shape, dtype=ratings.dtype
    )

    # build training matrix
    train_users = np.r_[users[~full_candidate_mask], candidate_users[train_idx]]
    train_items = np.r_[items[~full_candidate_mask], candidate_items[train_idx]]
    train_data = np.r_[data[~full_candidate_mask], candidate_data[train_idx]]
    train_mat = csr_matrix(
        (train_data, (train_users, train_items)),
        shape=ratings.shape,
        dtype=ratings.dtype,
    )

    return train_mat, test_mat


@cython.boundscheck(False)
def precision_at_k(model, train_user_items, test_user_items, int K=10,
                   show_progress=True, int num_threads=1):
    """ Calculates P@K for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated p@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads)['precision']


@cython.boundscheck(False)
def mean_average_precision_at_k(model, train_user_items, test_user_items, int K=10,
                                show_progress=True, int num_threads=1):
    """ Calculates MAP@K for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated MAP@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads)['map']


@cython.boundscheck(False)
def ndcg_at_k(model, train_user_items, test_user_items, int K=10,
              show_progress=True, int num_threads=1):
    """ Calculates ndcg@K for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated ndcg@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads)['ndcg']


@cython.boundscheck(False)
def AUC_at_k(model, train_user_items, test_user_items, int K=10,
             show_progress=True, int num_threads=1):
    """ Calculate limited AUC for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated AUC@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads)['auc']


@cython.boundscheck(False)
def ranking_metrics_at_k(model, train_user_items, test_user_items, int K=10,
                         show_progress=True, int num_threads=1):
    """ Calculates ranking metrics for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    dict[str, float]
        the calculated metrics@k
    """
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    cdef int users = test_user_items.shape[0], items = test_user_items.shape[1]
    cdef int u, i, batch_idx
    # precision
    cdef double relevant = 0, pr_div = 0, total = 0
    # map
    cdef double mean_ap = 0, ap = 0
    # ndcg
    cdef double[:] cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cdef double[:] cg_sum = np.cumsum(cg)
    cdef double ndcg = 0, idcg
    # auc
    cdef double mean_auc = 0, auc, hit, miss, num_pos_items, num_neg_items

    cdef int[:] test_indptr = test_user_items.indptr
    cdef int[:] test_indices = test_user_items.indices

    cdef int[:, :] ids
    cdef int[:] batch

    cdef unordered_set[int] likes


    batch_size = 1000
    start_idx = 0

    # get an array of userids that have at least one item in the test set
    to_generate = np.arange(users, dtype="int32")
    to_generate = to_generate[np.ediff1d(test_user_items.indptr) > 0]

    progress = tqdm(total=len(to_generate), disable=not show_progress)

    while start_idx < len(to_generate):
        batch = to_generate[start_idx: start_idx + batch_size]
        ids, _ = model.recommend(batch, train_user_items[np.asarray(batch)], N=K)
        start_idx += batch_size

        with nogil:
            for batch_idx in range(len(batch)):
                u = batch[batch_idx]
                likes.clear()
                for i in range(test_indptr[u], test_indptr[u+1]):
                    likes.insert(test_indices[i])

                pr_div += fmin(K, likes.size())
                ap = 0
                hit = 0
                miss = 0
                auc = 0
                idcg = cg_sum[min(K, likes.size()) - 1]
                num_pos_items = likes.size()
                num_neg_items = items - num_pos_items

                for i in range(K):
                    if likes.find(ids[batch_idx, i]) != likes.end():
                        relevant += 1
                        hit += 1
                        ap += hit / (i + 1)
                        ndcg += cg[i] / idcg
                    else:
                        miss += 1
                        auc += hit
                auc += ((hit + num_pos_items) / 2.0) * (num_neg_items - miss)
                mean_ap += ap / fmin(K, likes.size())
                mean_auc += auc / (num_pos_items * num_neg_items)
                total += 1

        progress.update(len(batch))

    progress.close()
    return {
        "precision": relevant / pr_div,
        "map": mean_ap / total,
        "ndcg": ndcg / total,
        "auc": mean_auc / total
    }


class ImpressionLogError(ValueError):
    """ Raised when an impression log fails validation.

    Carries the indices of the offending records so that they can be
    located in the source logs.

    Attributes
    ----------
    invalid_propensity_indices : ndarray or None
        Indices of impression records whose propensity is not a valid
        probability (not finite, <= 0 or > 1). None if there are no
        such records.
    invalid_reward_indices : ndarray or None
        Indices of impression records whose reward is not finite.
        None if there are no such records.
    duplicate_indices : ndarray or None
        Indices of impression records whose (user_id, item_id) pair
        appears more than once in the log. None if there are no such
        records.
    """

    def __init__(self, message, invalid_propensity_indices=None,
                 invalid_reward_indices=None, duplicate_indices=None):
        super().__init__(message)
        self.invalid_propensity_indices = invalid_propensity_indices
        self.invalid_reward_indices = invalid_reward_indices
        self.duplicate_indices = duplicate_indices


def _format_record_indices(indices, max_shown=10):
    """ Formats up to 'max_shown' record indices for an error message. """
    shown = ", ".join(str(i) for i in indices[:max_shown])
    if len(indices) > max_shown:
        shown += f", ... ({len(indices)} total)"
    return shown


def truncated_ips_at_k(user_ids, item_ids, propensities, rewards,
                       ranking_user_ids, ranking_item_ids,
                       K=None, max_weight=10.0, discount=None):
    """ Estimates the expected per-user reward of offline candidate rankings
    from logged impressions, using a truncated inverse propensity score (IPS)
    estimator.

    Recommendation logs generated by a previous (logging) policy are biased:
    items the policy favoured are shown more often, so a naive hit rate over
    the logs overestimates common items and says little about items that were
    rarely shown. IPS corrects this by weighting each logged impression by
    the inverse of its known impression probability (propensity), truncated
    to 'max_weight' to bound the variance::

        w = min(1 / propensity, max_weight)

    For each user, the candidate ranking is linked back to the impression
    log by (user_id, item_id) pairs, and the score is::

        score = (1 / num_users) * sum_u sum_{j < K} discount(j) * reward * w

    where the inner sum runs over the top-K positions of the candidate
    ranking of user 'u'. Positions whose (user_id, item_id) pair does not
    appear in the impression log contribute 0 to the estimate; they are
    reported through the 'num_unsupported' and 'non_estimable_user_ids'
    keys of the returned dict.

    Parameters
    ----------
    user_ids : array_like
        The user id of each logged impression.
    item_ids : array_like
        The item id of each logged impression.
    propensities : array_like
        The probability that the logging policy showed the item to the user,
        for each impression. Every value must be in the interval (0, 1].
    rewards : array_like
        The observed feedback (e.g. click) for each impression.
    ranking_user_ids : array_like
        The user id of each candidate ranking to evaluate. Each user may
        appear at most once.
    ranking_item_ids : array_like
        A 2-D array with one row per user in 'ranking_user_ids', containing
        the item ids of the candidate ranking in rank order (most
        recommended first). This matches the 'ids' array returned by
        'model.recommend'.
    K : int, optional
        Only evaluate the top-K positions of each candidate ranking.
        Defaults to the full width of 'ranking_item_ids'.
    max_weight : float, optional
        Weights are truncated to this value (equivalently, propensities are
        clipped to '1 / max_weight'). Must be >= 1. Defaults to 10.
    discount : None or "log2", optional
        Optional per-position discount. With "log2", position 'j' (0-based)
        is discounted by '1 / log2(j + 2)' like in DCG. Defaults to no
        discount.

    Returns
    -------
    dict
        A dict with the following keys:

        - 'score': the truncated IPS estimate of the expected per-user
          reward of the candidate rankings over the top-K positions.
        - 'user_scores': the per-user contributions to 'score'.
        - 'num_users': the number of candidate rankings evaluated.
        - 'num_positions': the total number of ranking positions evaluated
          ('num_users * K').
        - 'num_matched': the number of ranking positions that could be
          linked to a logged impression.
        - 'num_unsupported': the number of ranking positions without a
          matching impression. These positions contribute 0 to the score.
        - 'non_estimable_user_ids': ids of users whose candidate ranking
          could not be linked to any impression. The score is not estimable
          for these users.
        - 'weight_sum': the sum of the (truncated) weights over matched
          positions.
        - 'effective_sample_size': the effective sample size of the weighted
          estimate, 'weight_sum ** 2 / sum(weights ** 2)'.
        - 'max_observed_weight': the largest weight over matched positions.
        - 'num_truncated': the number of matched positions whose weight hit
          'max_weight'.

    Raises
    ------
    ImpressionLogError
        If the impression log contains invalid propensities, non-finite
        rewards or duplicate (user_id, item_id) records. The indices of the
        offending records are available on the exception attributes
        'invalid_propensity_indices', 'invalid_reward_indices' and
        'duplicate_indices'.

    References
    ----------
    .. [1] Schnabel et al. "Recommendations as Treatments: Debiasing Learning
       and Evaluation." ICML 2016.
    """
    user_ids = np.asarray(user_ids)
    item_ids = np.asarray(item_ids)
    propensities = np.asarray(propensities, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    ranking_user_ids = np.asarray(ranking_user_ids)
    ranking_item_ids = np.asarray(ranking_item_ids)

    if not all(a.ndim == 1 for a in (user_ids, item_ids, propensities, rewards)):
        raise ValueError("user_ids, item_ids, propensities and rewards must be 1-D arrays")
    num_impressions = len(user_ids)
    if not (len(item_ids) == num_impressions and len(propensities) == num_impressions
            and len(rewards) == num_impressions):
        raise ValueError(
            "user_ids, item_ids, propensities and rewards must have the same"
            f" length, got {len(user_ids)}, {len(item_ids)},"
            f" {len(propensities)} and {len(rewards)}"
        )

    if ranking_user_ids.ndim != 1:
        raise ValueError("ranking_user_ids must be a 1-D array")
    if ranking_item_ids.ndim != 2 or ranking_item_ids.shape[0] != len(ranking_user_ids):
        raise ValueError(
            "ranking_item_ids must be a 2-D array with one row of ranked item"
            " ids per user in ranking_user_ids"
        )

    num_users, ranking_width = ranking_item_ids.shape
    if num_users == 0:
        raise ValueError("ranking_user_ids must contain at least one user")
    if K is None:
        K = ranking_width
    elif not 1 <= K <= ranking_width:
        raise ValueError(f"K must be in the range [1, {ranking_width}], got {K}")
    if max_weight < 1.0:
        raise ValueError(f"max_weight must be >= 1.0, got {max_weight}")
    if discount not in (None, "log2"):
        raise ValueError(f"discount must be None or 'log2', got {discount!r}")

    ranking_users, ranking_user_counts = np.unique(ranking_user_ids, return_counts=True)
    if len(ranking_users) != num_users:
        duplicated = ranking_users[ranking_user_counts > 1]
        raise ValueError(
            "ranking_user_ids contains duplicate users"
            f" [{_format_record_indices(duplicated)}]"
        )

    ranked_items = ranking_item_ids[:, :K]
    sorted_items = np.sort(ranked_items, axis=1)
    repeated = (sorted_items[:, 1:] == sorted_items[:, :-1]).any(axis=1)
    if repeated.any():
        raise ValueError(
            "ranking_item_ids contains repeated items within a single ranking"
            f" for users [{_format_record_indices(ranking_user_ids[repeated])}]"
        )

    invalid_propensities = np.nonzero(
        ~np.isfinite(propensities) | (propensities <= 0.0) | (propensities > 1.0)
    )[0]
    invalid_rewards = np.nonzero(~np.isfinite(rewards))[0]

    # combine (user_id, item_id) pairs into a single void scalar so that pairs
    # can be compared and searched with 1-D numpy operations
    pair_dtype = np.result_type(user_ids, item_ids, ranking_user_ids, ranking_item_ids)
    void_dtype = np.dtype([("user", pair_dtype), ("item", pair_dtype)])

    impression_pairs = np.empty((num_impressions, 2), dtype=pair_dtype)
    impression_pairs[:, 0] = user_ids
    impression_pairs[:, 1] = item_ids
    impression_pairs = impression_pairs.view(void_dtype).ravel()

    unique_pairs, pair_rows, pair_inverse, pair_counts = np.unique(
        impression_pairs, return_index=True, return_inverse=True, return_counts=True
    )
    duplicate_indices = np.nonzero(pair_counts[pair_inverse] > 1)[0]

    if len(invalid_propensities) or len(invalid_rewards) or len(duplicate_indices):
        problems = []
        if len(invalid_propensities):
            problems.append(
                f"{len(invalid_propensities)} records have propensities outside"
                f" (0, 1] at indices [{_format_record_indices(invalid_propensities)}]"
            )
        if len(invalid_rewards):
            problems.append(
                f"{len(invalid_rewards)} records have non-finite rewards at"
                f" indices [{_format_record_indices(invalid_rewards)}]"
            )
        if len(duplicate_indices):
            problems.append(
                f"{len(duplicate_indices)} records share the same (user_id, item_id)"
                f" pair as another record at indices"
                f" [{_format_record_indices(duplicate_indices)}]"
            )
        raise ImpressionLogError(
            "invalid impression log: " + "; ".join(problems),
            invalid_propensity_indices=invalid_propensities if len(invalid_propensities) else None,
            invalid_reward_indices=invalid_rewards if len(invalid_rewards) else None,
            duplicate_indices=duplicate_indices if len(duplicate_indices) else None,
        )

    # link every position of the candidate rankings back to the impression
    # record with the same (user_id, item_id) pair
    position_pairs = np.empty((num_users * K, 2), dtype=pair_dtype)
    position_pairs[:, 0] = np.repeat(ranking_user_ids, K)
    position_pairs[:, 1] = ranked_items.ravel()
    position_pairs = position_pairs.view(void_dtype).ravel()

    matched = np.zeros(num_users * K, dtype=bool)
    impression_rows = np.zeros(num_users * K, dtype=np.intp)
    if num_impressions:
        locations = np.searchsorted(unique_pairs, position_pairs)
        locations = np.minimum(locations, len(unique_pairs) - 1)
        matched = unique_pairs[locations] == position_pairs
        impression_rows[matched] = pair_rows[locations[matched]]

    weights = np.minimum(1.0 / propensities, max_weight)

    if discount == "log2":
        position_discount = np.tile(1.0 / np.log2(np.arange(2, K + 2)), num_users)
    else:
        position_discount = np.ones(num_users * K)

    contributions = np.zeros(num_users * K)
    matched_rows = impression_rows[matched]
    contributions[matched] = (
        rewards[matched_rows] * weights[matched_rows] * position_discount[matched]
    )
    user_scores = contributions.reshape(num_users, K).sum(axis=1)

    matched_weights = weights[matched_rows]
    num_matched = len(matched_rows)
    weight_sum = float(matched_weights.sum())
    if num_matched:
        effective_sample_size = weight_sum ** 2 / float((matched_weights ** 2).sum())
        max_observed_weight = float(matched_weights.max())
    else:
        effective_sample_size = 0.0
        max_observed_weight = 0.0

    matched = matched.reshape(num_users, K)
    return {
        "score": float(user_scores.mean()),
        "user_scores": user_scores,
        "num_users": num_users,
        "num_positions": num_users * K,
        "num_matched": num_matched,
        "num_unsupported": num_users * K - num_matched,
        "non_estimable_user_ids": ranking_user_ids[~matched.any(axis=1)],
        "weight_sum": weight_sum,
        "effective_sample_size": effective_sample_size,
        "max_observed_weight": max_observed_weight,
        "num_truncated": int((matched_weights >= max_weight).sum()),
    }
