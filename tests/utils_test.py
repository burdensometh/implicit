import numpy as np

from implicit.utils import (
    _expand_candidate_ids,
    _filter_items_from_results,
    _get_exclude_items,
)


def _make_query_fn(ranking):
    """Returns a query function over a fixed ranking, emulating an ANN index that
    returns at most `count` of the top ranked ids (padding with -1 if needed)."""
    ranking = np.asarray(ranking)

    def query_fn(count):
        ids = ranking[:count]
        if len(ids) < count:
            ids = np.concatenate([ids, np.full(count - len(ids), -1)])
        return ids, np.zeros(len(ids))

    return query_fn


def test_expand_candidate_ids_no_filtering():
    ranking = np.arange(100)[::-1]
    ids = _expand_candidate_ids(_make_query_fn(ranking), 10, None, 100)
    np.testing.assert_array_equal(ids, ranking[:10])


def test_expand_candidate_ids_skips_excluded():
    ranking = np.arange(100)
    exclude = np.arange(20)
    ids = _expand_candidate_ids(_make_query_fn(ranking), 10, exclude, 100)
    np.testing.assert_array_equal(ids, ranking[20:30])


def test_expand_candidate_ids_keeps_expanding_until_enough():
    # exclude the first 60 items; a single lookup sized for N + len(exclude) is enough
    ranking = np.arange(100)
    exclude = np.arange(60)
    ids = _expand_candidate_ids(_make_query_fn(ranking), 10, exclude, 100)
    np.testing.assert_array_equal(ids, ranking[60:70])


def test_expand_candidate_ids_expands_when_index_is_approximate():
    # emulate an index where the first (oversized) lookup only surfaces excluded ids
    # and expanding is required to find usable candidates: the ranking stays fixed.
    ranking = np.concatenate([np.arange(50), np.arange(50, 100)])

    def query_fn(count):
        ids = ranking[:count]
        return ids, np.zeros(len(ids))

    ids = _expand_candidate_ids(query_fn, 10, np.arange(50), 100)
    np.testing.assert_array_equal(ids, ranking[50:60])


def test_expand_candidate_ids_respects_upper_bound():
    ranking = np.arange(100)
    ids = _expand_candidate_ids(_make_query_fn(ranking), 10, None, 100, max_candidates=5)
    np.testing.assert_array_equal(ids, ranking[:5])


def test_expand_candidate_ids_dedupes():
    # an index that repeats ids in its results shouldn't yield duplicates
    ranking = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11])
    ids = _expand_candidate_ids(_make_query_fn(ranking), 4, None, 100, max_candidates=len(ranking))
    np.testing.assert_array_equal(np.unique(ids), ids)
    np.testing.assert_array_equal(ids, np.array([1, 2, 3, 4]))


def test_get_exclude_items():
    from scipy.sparse import csr_matrix

    user_items = csr_matrix(np.array([[0, 1, 0, 1, 0]]))

    np.testing.assert_array_equal(_get_exclude_items(user_items, None, True), [1, 3])
    np.testing.assert_array_equal(_get_exclude_items(user_items, None, False), [])
    np.testing.assert_array_equal(_get_exclude_items(user_items, [4, 2], False), [4, 2])
    np.testing.assert_array_equal(_get_exclude_items(user_items, [4, 2], True), [4, 2, 1, 3])


def test_filter_items_from_results_scalar():
    ids = np.array([0, 1, 2, 3, 4])
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    filtered_ids, filtered_scores = _filter_items_from_results(0, ids, scores, [1, 3], 2)
    np.testing.assert_array_equal(filtered_ids, [0, 2])
    np.testing.assert_array_equal(filtered_scores, [5.0, 3.0])
