import time
import warnings

import numpy as np
import scipy.sparse
import threadpoolctl


def nonzeros(m, row):
    """returns the non zeroes of a row in csr_matrix"""
    for index in range(m.indptr[row], m.indptr[row + 1]):
        yield m.indices[index], m.data[index]


_checked_blas_config = False


def check_blas_config():
    """checks to see if using OpenBlas/Intel MKL. If so, warn if the number of threads isn't set
    to 1 (causes severe perf issues when training - can be 10x slower)"""
    # don't warn repeatedly
    global _checked_blas_config  # pylint: disable=global-statement
    if _checked_blas_config:
        return

    for api in threadpoolctl.threadpool_info():
        num_threads = api["num_threads"]
        if api["user_api"] != "blas" or num_threads == 1:
            continue

        internal_api = api["internal_api"]
        if internal_api == "openblas":
            warnings.warn(
                f"OpenBLAS is configured to use {num_threads} threads. It is highly recommended"
                " to disable its internal threadpool by setting the environment variable"
                " 'OPENBLAS_NUM_THREADS=1' or by calling 'threadpoolctl.threadpool_limits(1,"
                ' "blas")\'. Having OpenBLAS use a threadpool can lead to severe performance'
                " issues here.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif internal_api == "mkl":
            warnings.warn(
                f"Intel MKL BLAS is configured to use {num_threads} threads. It is highly"
                " recommended to disable its internal threadpool by setting the environment"
                " variable 'MKL_NUM_THREADS=1' or by callng 'threadpoolctl.threadpool_limits(1,"
                ' "blas")\'. Having MKL use a threadpool can lead to severe performance issues',
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            # probably using blis, which is by default single threaded. warn anyways
            # for when threadpoolctl gets support for VecLib/Accelerate etc
            warnings.warn(
                f"BLAS library {internal_api} is configured to use {num_threads} threads."
                " It is highly recommended to disable its internal threadpool by calling"
                " 'threadpoolctl.threadpool_limits(1, \"blas\")'.",
                RuntimeWarning,
                stacklevel=2,
            )

    _checked_blas_config = True


def check_random_state(random_state):
    """Validate the random state.

    Check a random seed or existing numpy rng
    and get back an initialized numpy.randon.Generator

    Parameters
    ----------
    random_state : int, None, np.random.RandomState or np.random.Generator
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.
    """
    # backwards compatibility
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state.rand_int(2**31))

    # otherwise try to initialize a new one, and let it fail through
    # on the numpy side if it doesn't work
    return np.random.default_rng(random_state)


def augment_inner_product_matrix(factors):
    """This function transforms a factor matrix such that an angular nearest neighbours search
    will return top related items of the inner product.

    This involves transforming each row by adding one extra dimension as suggested in the paper:
    "Speeding Up the Xbox Recommender System Using a Euclidean Transformation for Inner-Product
    Spaces" https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf

    Basically this involves transforming each feature vector so that they have the same norm, which
    means the cosine of this transformed vector is proportional to the dot product (if the other
    vector in the cosine has a 0 in the extra dimension)."""
    norms = np.linalg.norm(factors, axis=1)
    max_norm = norms.max()

    # add an extra dimension so that the norm of each row is the same
    # (max_norm)
    extra_dimension = np.sqrt(max_norm**2 - norms**2)
    return max_norm, np.append(factors, extra_dimension.reshape(norms.shape[0], 1), axis=1)


def _batch_call(func, ids, *args, N=10, id_dtype=np.int32, score_dtype=np.float32, **kwargs):
    # we're running in batch mode, just loop over each item and call the scalar version of the
    # function
    output_ids = np.zeros((len(ids), N), dtype=id_dtype)
    output_scores = np.zeros((len(ids), N), dtype=score_dtype)

    user_items = kwargs.pop("user_items") if "user_items" in kwargs else None
    item_users = kwargs.pop("item_users") if "item_users" in kwargs else None

    # pylint: disable=unsubscriptable-object
    for i, idx in enumerate(ids):
        current_kwargs = kwargs
        if user_items is not None:
            current_kwargs = {"user_items": user_items[i], **kwargs}
        elif item_users is not None:
            current_kwargs = {"item_users": item_users[i], **kwargs}

        batch_ids, batch_scores = func(idx, *args, N=N, **current_kwargs)

        # pad out to N items if we're returned fewer
        missing_items = N - len(batch_ids)
        if missing_items > 0:
            batch_ids = np.append(batch_ids, np.full(missing_items, -1))
            batch_scores = np.append(
                batch_scores, np.full(missing_items, -np.finfo(np.float32).max)
            )

        output_ids[i] = batch_ids[:N]
        output_scores[i] = batch_scores[:N]

    return output_ids, output_scores


def _get_exclude_items(user_items, filter_items, filter_already_liked_items):
    """Returns the item ids that should be filtered out of a recommend call.

    This combines the explicit filter_items list with the items the user has already
    liked (when filter_already_liked_items is set) into a single array.
    """
    exclude = None
    if filter_items is not None:
        exclude = np.asarray(filter_items, dtype=np.int64)
    if filter_already_liked_items:
        liked = user_items[0].indices
        exclude = liked if exclude is None else np.concatenate([exclude, liked])
    if exclude is None:
        return np.empty(0, dtype=np.int64)
    return exclude


def _expand_candidate_ids(query_fn, N, exclude, num_items, max_candidates=None):
    """Collects candidate item ids from an approximate index in expanding batches.

    Filtering the results of an ANN lookup can leave too few candidates to fill the
    requested number of recommendations - this happens for instance with users that have
    already interacted with (or had filtered out) all of the top candidates returned by
    the index. Rather than issuing a single oversized query, this queries the index in
    batches, expanding the requested number of candidates each round and skipping both
    the excluded items and any candidate that has already been seen, until N usable
    candidates have been collected or the search upper bound has been reached.

    Parameters
    ----------
    query_fn : Callable[[int], Tuple[array_like, array_like]]
        Function that queries the index, returning up to count (item ids, scores).
    N : int
        The number of candidates to collect.
    exclude : array_like or None
        Item ids to exclude from the results.
    num_items : int
        The total number of items in the index.
    max_candidates : int, optional
        Upper bound on the number of candidates to retrieve from the index. Defaults to
        the number of items in the index.

    Returns
    -------
    numpy.ndarray
        The collected item ids, in the order the index returned them. This may contain
        fewer than N entries if the search upper bound was reached.
    """
    limit = num_items if max_candidates is None else min(num_items, int(max_candidates))
    if limit <= 0 or N <= 0:
        return np.empty(0, dtype=np.int64)

    seen = set(np.asarray(exclude).tolist()) if exclude is not None else set()

    collected = []
    # start off with the same sized request as an unfiltered lookup: enough for the N
    # results plus the items we need to skip over. Only expand from here if it isn't
    # enough.
    count = max(1, min(limit, N + len(seen)))
    while True:
        ids, _ = query_fn(count)
        ids = np.asarray(ids)
        for item in ids.tolist():
            if item < 0 or item >= num_items or item in seen:
                continue
            seen.add(item)
            collected.append(item)
            if len(collected) >= N:
                break

        # some indexes (annoy/nmslib) return fewer than requested when they run out of
        # items - there's no point in expanding the search in that case
        exhausted = len(ids) < count
        if len(collected) >= N or count >= limit or exhausted:
            break

        # expand the search: double the number of requested candidates, at least
        # increasing it by N each round, up to the search upper bound
        count = min(limit, max(count * 2, count + N))

    return np.array(collected, dtype=np.int64)


def _exact_inner_product_scores(model, user, item_ids):
    """Scores the given items exactly against the user factors with a matrix factorization model.

    This is used to fill in any recommendations that an approximate search couldn't find
    enough candidates for, using the same inner product scoring that the exact recommend
    call uses.
    """
    item_factors = model.item_factors
    if _is_gpu_matrix(item_factors):
        # index on the GPU first, so we only copy the selected rows back to the host
        factors = item_factors[np.asarray(item_ids)].to_numpy()
    else:
        factors = item_factors[np.asarray(item_ids)]
    factors = np.asarray(factors, dtype=np.float64)
    user = np.asarray(user, dtype=np.float64).reshape(-1)
    return factors.dot(user)


def _is_gpu_matrix(factors):
    """Returns True if the passed in factors are a GPU matrix rather than a numpy array"""
    cls = type(factors)
    return cls.__name__ == "Matrix" and cls.__module__.startswith("implicit.gpu")


def _sort_by_score(ids, scores, N):
    """Sorts ids by descending score, keeping the index order for tied scores"""
    order = np.argsort(-np.asarray(scores), kind="stable")
    return np.asarray(ids)[order][:N], np.asarray(scores)[order][:N]


def _recommend_with_expanding_search(
    model, user, N, exclude, num_items, query_fn, max_candidates=None
):
    """Recommends items for a single user with an approximate index and expanding search.

    The top candidates returned by an approximate index are frequently all items the user
    has already interacted with (or otherwise asked to filter out) for users with a long
    history. This gathers candidates from the index in expanding batches, skipping seen
    candidates, until enough usable items are collected or the search upper bound is hit.
    If the index still can't supply N results the shortfall is filled in with the exact
    inner product scoring over the remaining items. The final output is sorted by the
    exact factor scores, keeping the index ordering for ties.

    Parameters
    ----------
    model : MatrixFactorizationBase
        The inner model providing the item factors used for exact scoring.
    user : array_like
        The user factor vector to score against.
    N : int
        The number of recommendations to return.
    exclude : array_like or None
        Item ids to exclude from the results.
    num_items : int
        The total number of items in the index.
    query_fn : Callable[[int], Tuple[array_like, array_like]]
        Function returning up to count (item ids, scores) from the approximate index.
    max_candidates : int, optional
        Upper bound on the number of candidates to retrieve from the index.

    Returns
    -------
    tuple
        Tuple of (itemids, scores) arrays.
    """
    if N <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    user = np.asarray(user).reshape(-1)
    exclude_ids = np.asarray(exclude) if exclude is not None else np.empty(0, dtype=np.int64)

    candidate_ids = _expand_candidate_ids(
        query_fn, N, exclude_ids, num_items, max_candidates=max_candidates
    )

    # if we couldn't find enough candidates with the approximate index, fill in the
    # shortfall with the exact factor scores over the items we haven't seen yet
    if len(candidate_ids) < N:
        mask = np.ones(num_items, dtype=bool)
        mask[candidate_ids] = False
        exclude_ids = exclude_ids[(exclude_ids >= 0) & (exclude_ids < num_items)]
        if exclude_ids.size:
            mask[exclude_ids] = False
        remaining = np.nonzero(mask)[0]
        if remaining.size:
            remaining_scores = _exact_inner_product_scores(model, user, remaining)
            order = np.argsort(-remaining_scores, kind="stable")
            shortfall = N - len(candidate_ids)
            candidate_ids = np.concatenate([candidate_ids, remaining[order][:shortfall]])

    if not len(candidate_ids):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    scores = _exact_inner_product_scores(model, user, candidate_ids)
    return _sort_by_score(candidate_ids, scores, N)


def _filter_items_from_results(queryid, ids, scores, filter_items, N):
    if np.isscalar(queryid):
        mask = np.isin(ids, filter_items, invert=True)
        ids, scores = ids[mask][:N], scores[mask][:N]
    else:
        rows = len(queryid)
        filtered_scores = np.zeros((rows, N), dtype=scores.dtype)
        filtered_ids = np.zeros((rows, N), dtype=ids.dtype)
        for row in range(rows):
            mask = np.isin(ids[row], filter_items, invert=True)
            filtered_ids[row] = ids[row][mask][:N]
            filtered_scores[row] = scores[row][mask][:N]
        ids, scores = filtered_ids, filtered_scores
    return ids, scores


class ParameterWarning(Warning):
    pass


def check_csr(user_items):
    if not isinstance(user_items, scipy.sparse.csr_matrix):
        class_name = user_items.__class__.__name__
        start = time.time()
        user_items = user_items.tocsr()
        warnings.warn(
            f"Method expects CSR input, and was passed {class_name} instead. "
            f"Converting to CSR took {time.time() - start} seconds",
            ParameterWarning,
        )
    return user_items
