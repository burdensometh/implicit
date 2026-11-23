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


def item_groups_by_train_count(train_user_items, num_groups=None, thresholds=None):
    """ Assigns items to popularity groups, using training interactions only.

    Items are bucketed by the number of users that interacted with them in the
    training matrix. Items that never appear in the training data (for instance
    new items introduced by a validation/test split) are not bucketed at all:
    they are assigned the dedicated group id ``-1`` ('unseen') and are excluded
    from the boundary computation, so they can never distort the bucketing of
    the training items.

    Parameters
    ----------
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains the elements that were used
        to train the model. This is the only input that determines the groups.
    num_groups : int, optional
        The number of popularity groups to create. Boundaries are placed on
        quantiles of the positive train interaction counts, so that each group
        holds a similar number of training items. Defaults to 3 (tail/torso/head)
        when neither num_groups nor thresholds is given. Because interaction
        counts are heavily tied on small datasets, the effective number of
        non-empty groups can be smaller than requested.
    thresholds : array_like of int, optional
        Explicit interaction-count cut points. Each threshold ``t`` starts a new
        group for items with count ``>= t``: with thresholds ``[t0, t1]`` the
        groups are counts in ``[1, t0)``, ``[t0, t1)`` and ``[t1, inf)``.
        Mutually exclusive with num_groups.

    Returns
    -------
    (groups, labels) : (ndarray, list of str)
        groups is an int32 array of shape (n_items,) with the group id of each
        item, or -1 for items unseen in the training data. labels[g] is a
        human-readable description of the count interval of group g.
    """
    if num_groups is not None and thresholds is not None:
        raise ValueError("Only one of 'num_groups' and 'thresholds' can be set")
    if num_groups is None and thresholds is None:
        num_groups = 3

    # popularity of each item = number of training interactions (train data only)
    counts = np.asarray(train_user_items.getnnz(axis=0)).astype(np.int64)

    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=np.float64)
        if thresholds.ndim != 1 or not len(thresholds):
            raise ValueError("thresholds must be a non-empty 1-d array of cut points")
        if np.any(thresholds < 1):
            raise ValueError("thresholds must be >= 1")
        # counts are integral: ceiling keeps the partition identical while
        # giving exact integer boundaries
        boundaries = np.unique(np.ceil(thresholds).astype(np.int64))
    else:
        if num_groups < 1:
            raise ValueError("num_groups must be >= 1")
        num_groups = int(num_groups)
        positive = counts[counts > 0]
        if len(positive):
            quantiles = np.linspace(0.0, 1.0, num_groups + 1)[1:-1]
            boundaries = np.unique(np.ceil(np.quantile(positive, quantiles)).astype(np.int64))
        else:
            boundaries = np.empty(0, dtype=np.int64)

    groups = np.searchsorted(boundaries, counts, side="right").astype(np.int32)
    groups[counts == 0] = -1

    labels = []
    lo = 1
    for boundary in boundaries:
        labels.append(f"[{lo}, {boundary})")
        lo = int(boundary)
    labels.append(f"[{lo}, inf)")

    return groups, labels


@cython.boundscheck(False)
def ranking_metrics_at_k_by_item_group(model, train_user_items, test_user_items, int K=10,
                                       num_groups=None, thresholds=None,
                                       show_progress=True, int num_threads=1):
    """ Calculates the hit rate (precision@K) broken down by item popularity group.

    Reporting only the overall hit rate can hide regressions on long-tail items:
    popular items dominate the average, so a drop on rarely interacted items -
    and on the small-shop users who depend on them - goes unnoticed. This
    function buckets items by their number of *training* interactions (see
    item_groups_by_train_count) and reports the hit rate of each group
    separately, next to the overall figure.

    All groups are evaluated on the same candidate users (users with at least
    one item in the test set) and the same ground-truth labels
    (test_user_items) as ranking_metrics_at_k, so the group results are
    directly comparable to the overall numbers: the overall precision reported
    here equals ranking_metrics_at_k(...)['precision'], and the per-group
    'relevant' counts sum up to the overall one.

    The grouping is determined by the training matrix only. Items that never
    appear in the training data (e.g. new items introduced by the
    validation/test split) are placed in a dedicated 'unseen' group and never
    influence the bucket boundaries of the training items.

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model. Also the only input used to bucket items.
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    num_groups : int, optional
        Number of popularity groups to bucket training items into (quantile
        boundaries). Defaults to 3. Mutually exclusive with thresholds.
    thresholds : array_like of int, optional
        Explicit interaction-count cut points for the groups, see
        item_groups_by_train_count. Mutually exclusive with num_groups.
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather
        than help.

    Returns
    -------
    dict
        A dict with keys:

        - 'overall': dict with the metrics over all candidate users:
          'precision' (identical to ranking_metrics_at_k), 'relevant' (total
          hits in the top-K), 'effective_denominator' (sum over users of
          min(K, #test items of the user)), 'num_users' and 'empty'.
        - 'groups': list of per-group dicts, one per popularity bucket ordered
          by increasing train interaction count, plus a final group
          'unseen' holding items without any training interaction. Each entry
          has: 'group', 'label', 'precision' (None when the group is empty),
          'relevant' (hits in the top-K on items of this group),
          'effective_denominator' (sum over candidate users of
          min(K, #test items of the user in this group)), 'test_items' (total
          test labels in the group), 'num_users' (candidate users with at
          least one test label in the group), 'num_items' (training items
          assigned to the group) and 'empty' (True when no test label falls
          in the group, in which case 'precision' is None).
        - 'K': the value of K the metrics were computed at.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    # bucket items by training popularity; this only ever looks at the
    # training matrix, so validation-only items can't distort the bucketing
    item_groups, labels = item_groups_by_train_count(
        train_user_items, num_groups=num_groups, thresholds=thresholds)

    cdef int users = test_user_items.shape[0]
    cdef int items = max(train_user_items.shape[1], test_user_items.shape[1])
    cdef int n_bucketed = len(labels)
    cdef int unseen = n_bucketed  # index of the 'unseen' group
    cdef int G = n_bucketed + 1

    # items that only exist in the test matrix are 'unseen' as well
    if item_groups.shape[0] < items:
        item_groups = np.concatenate(
            [item_groups, np.full(items - item_groups.shape[0], -1, dtype=np.int32)])
    cdef int[:] item_group = item_groups

    cdef int u, i, g, batch_idx
    # per-group accumulators
    cdef double[:] relevant = np.zeros(G)
    cdef double[:] pr_div = np.zeros(G)
    cdef double[:] group_test_items = np.zeros(G)
    cdef double[:] group_users = np.zeros(G)
    # per-user scratch: number of test items of the user in each group
    cdef int[:] likes_in_group = np.zeros(G, dtype=np.int32)
    # overall accumulators
    cdef double total = 0, overall_pr_div = 0

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
                for g in range(G):
                    likes_in_group[g] = 0
                for i in range(test_indptr[u], test_indptr[u+1]):
                    likes.insert(test_indices[i])
                    g = item_group[test_indices[i]]
                    if g < 0:
                        g = unseen
                    likes_in_group[g] += 1

                overall_pr_div += fmin(K, likes.size())
                for g in range(G):
                    if likes_in_group[g] > 0:
                        pr_div[g] += fmin(K, likes_in_group[g])
                        group_users[g] += 1
                        group_test_items[g] += likes_in_group[g]

                for i in range(K):
                    # only count hits: a recommended id found in 'likes' is a
                    # valid test item, so the group lookup is always in bounds
                    if likes.find(ids[batch_idx, i]) != likes.end():
                        g = item_group[ids[batch_idx, i]]
                        if g < 0:
                            g = unseen
                        relevant[g] += 1
                total += 1

        progress.update(len(batch))

    progress.close()

    item_groups = np.asarray(item_group)
    num_items = np.bincount(np.where(item_groups < 0, unseen, item_groups), minlength=G)

    groups = []
    for g in range(G):
        groups.append({
            "group": g if g < n_bucketed else "unseen",
            "label": labels[g] if g < n_bucketed else "unseen (no train interactions)",
            "precision": relevant[g] / pr_div[g] if pr_div[g] > 0 else None,
            "relevant": int(relevant[g]),
            "effective_denominator": int(pr_div[g]),
            "test_items": int(group_test_items[g]),
            "num_users": int(group_users[g]),
            "num_items": int(num_items[g]),
            "empty": pr_div[g] == 0,
        })

    relevant_total = 0.0
    for g in range(G):
        relevant_total += relevant[g]

    return {
        "K": K,
        "overall": {
            "precision": relevant_total / overall_pr_div if overall_pr_div > 0 else None,
            "relevant": int(relevant_total),
            "effective_denominator": int(overall_pr_div),
            "num_users": int(total),
            "empty": total == 0,
        },
        "groups": groups,
    }
