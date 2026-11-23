"""Tests for category-aware candidate re-ranking."""

import math

import pytest

from implicit.reranking import (
    UNCATEGORIZED,
    RerankingError,
    apply_category_caps,
    category_aware_rerank,
    filter_purchased,
    order_candidates,
    rerank,
    rerank_candidates,
    rerank_to_json,
    rerank_with_report,
    selection_arrays,
)


# (item_id, score, category)
CANDIDATES = [
    ("a", 0.99, "books"),
    ("b", 0.95, "books"),
    ("c", 0.90, "books"),
    ("d", 0.85, "toys"),
    ("e", 0.80, "toys"),
    ("f", 0.75, "books"),
    ("g", 0.70, "home"),
    ("h", 0.65, None),
    ("i", 0.60, "toys"),
]


def test_basic_category_cap_prefers_high_scores():
    # At most 2 books in the greedy pass, target 5.
    records = rerank(CANDIDATES, {"books": 2}, n=5)
    assert [r["item_id"] for r in records] == ["a", "b", "d", "e", "g"]
    assert [r["rank"] for r in records] == [1, 2, 3, 4, 5]
    # original model scores are untouched
    assert [r["score"] for r in records] == [0.99, 0.95, 0.85, 0.80, 0.70]
    # model ranks reflect the raw model order after filtering
    assert [r["model_rank"] for r in records] == [1, 2, 4, 5, 7]
    assert all(r["category"] == CANDIDATES[ord(r["item_id"]) - 97][2] for r in records)


def test_purchased_filter_then_backfill_shortfall():
    # Remove 3 of the strongest candidates; caps still apply, parked items
    # back-fill the remaining slots.
    records = rerank(CANDIDATES, {"books": 2, "toys": 1}, n=5, already_purchased={"a", "b", "d"})
    # eligible model order: c(books), e(toys), f(books), g(home), h(None), i(toys)
    # greedy: c (books=1), e (toys=1), f blocked? books=2 -> f admitted(books=2),
    # g admitted (home), h admitted -> 5 already.
    assert [r["item_id"] for r in records] == ["c", "e", "f", "g", "h"]

    report = rerank_with_report(
        CANDIDATES, {"books": 2, "toys": 1}, n=5, already_purchased={"a", "b", "d"}
    )
    assert report["num_filtered_purchased"] == 3
    assert report["num_eligible"] == 6
    assert report["shortfall"] == 0
    assert report["category_counts"]["books"] == 2


def test_pool_smaller_than_n_returns_short_list():
    records = rerank(CANDIDATES, {"books": 2}, n=20, already_purchased={"a", "b", "c", "d"})
    assert len(records) == 5
    assert [r["rank"] for r in records] == [1, 2, 3, 4, 5]

    report = rerank_with_report(
        CANDIDATES, {"books": 2}, n=20, already_purchased={"a", "b", "c", "d"}
    )
    assert report["shortfall"] == 15


def test_uncategorized_is_own_group_unlimited_by_default():
    candidates = [
        ("u1", 0.9, None),
        ("u2", 0.8, None),
        ("u3", 0.7, None),
        ("x1", 0.6, "cat"),
    ]
    records = rerank(candidates, {"cat": 1}, n=4)
    assert [r["item_id"] for r in records] == ["u1", "u2", "u3", "x1"]
    assert all(r["category"] is not UNCATEGORIZED or r["item_id"].startswith("u") for r in records)


def test_uncategorized_cap_via_none_key_and_keyword():
    candidates = [("u1", 0.9, None), ("u2", 0.8, None), ("x1", 0.7, "cat")]
    records = rerank(candidates, {None: 1}, n=2)
    assert [r["item_id"] for r in records] == ["u1", "x1"]

    records = rerank(candidates, {"cat": 1}, n=2, uncategorized_cap=1)
    assert [r["item_id"] for r in records] == ["u1", "x1"]


def test_ties_keep_model_input_order():
    candidates = [
        ("p", 0.5, "a"),
        ("q", 0.5, "b"),
        ("r", 0.5, "a"),
        ("s", 0.9, "b"),
    ]
    records = rerank(candidates, {"a": 10, "b": 10}, n=4)
    assert [r["item_id"] for r in records] == ["s", "p", "q", "r"]
    assert [r["model_rank"] for r in records] == [1, 2, 3, 4]


def test_strict_caps_no_backfill_when_n_is_none():
    records = rerank(CANDIDATES, {"books": 1, "toys": 1, "home": 1})
    # uncategorized has no cap, but greedy alone decides membership
    assert [r["item_id"] for r in records] == ["a", "d", "g", "h"]


def test_backfill_disabled_leaves_slots_open():
    records = rerank(CANDIDATES, {"books": 1}, n=5, backfill=False)
    assert [r["item_id"] for r in records] == ["a", "d", "e", "g", "h"]


def test_default_cap_for_unlisted_categories():
    candidates = [
        ("a", 0.9, "x"),
        ("b", 0.8, "x"),
        ("c", 0.7, "y"),
        ("d", 0.6, "y"),
    ]
    # strict: caps limit the result even though more slots were requested
    records = rerank(candidates, {}, n=4, default_cap=1, backfill=False)
    assert [r["item_id"] for r in records] == ["a", "c"]
    # with back-filling, parked items take the leftover slots in model order
    records = rerank(candidates, {}, n=4, default_cap=1)
    assert [r["item_id"] for r in records] == ["a", "c", "b", "d"]


def test_duplicate_item_ids_keep_first_occurrence():
    candidates = [("a", 0.9, "x"), ("a", 0.8, "x"), ("b", 0.7, "y")]
    normalized_input = candidates
    report = rerank_with_report(normalized_input, {"x": 10}, n=3)
    assert [r["item_id"] for r in report["selections"]] == ["a", "b"]
    assert report["num_filtered_duplicate"] == 1


def test_mapping_records_and_recomputation_roundtrip():
    candidates = [
        {"item_id": "a", "score": 0.9, "category": "x"},
        {"id": "b", "score": 0.8, "category": "y"},
        {"item_id": "c", "score": 0.7, "category": "x"},
    ]
    first = rerank(candidates, {"x": 1, "y": 1}, n=3)
    # feeding the output back in must reproduce the identical ranking
    second = rerank(first, {"x": 1, "y": 1}, n=3)
    assert first == second
    ids, scores, ranks = selection_arrays(first)
    assert ids == ("a", "b", "c")
    assert scores == (0.9, 0.8, 0.7)
    assert ranks == (1, 2, 3)
    # json output is consumer-ready
    payload = rerank_to_json(first)
    assert '"item_id"' in payload and '"model_rank"' in payload and '"rank"' in payload


def test_bare_id_score_pair_is_a_single_candidate():
    # a single two-tuple must not be mistaken for parallel arrays
    records = rerank([("only", 0.42, "x"), ("pair", 0.9, "y")], {"x": 1, "y": 1}, n=2)
    assert [r["item_id"] for r in records] == ["pair", "only"]
    records = rerank([("solo", 0.42)], {"x": 10}, n=1)
    assert records[0]["item_id"] == "solo"
    records = rerank([("solo", 0.42, "x")], {"x": 10}, n=1)
    assert records[0]["item_id"] == "solo"


def test_parallel_arrays_with_category_mapping():
    ids = ["a", "b", "c", "d"]
    scores = [0.4, 0.3, 0.2, 0.1]
    categories = {"a": "x", "b": "y", "c": "x", "d": "y"}
    records = rerank((ids, scores), {"x": 1}, n=3, categories=categories)
    assert [r["item_id"] for r in records] == ["a", "b", "d"]
    assert rerank_candidates is rerank and category_aware_rerank is rerank


def test_tuple_inputs_and_numpy_like_scalars():
    class Scalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    candidates = [(Scalar("a"), Scalar(0.9), Scalar("x")), ("b", 0.8, "y")]
    records = rerank(candidates, {"x": 1, "y": 1}, n=2)
    assert [r["item_id"] for r in records] == ["a", "b"]
    assert isinstance(records[0]["score"], float)


def test_pipeline_stage_helpers_are_consistent():
    normalized_input = [
        {"item_id": c[0], "score": c[1], "category": c[2]} for c in CANDIDATES
    ]
    eligible, purchased, _ = filter_purchased(normalized_input, {"a"})
    assert [c["item_id"] for c in purchased] == ["a"]
    ordered = order_candidates(eligible)
    assert ordered[0]["item_id"] == "b"
    selected, parked, counts, backfilled = apply_category_caps(
        ordered, {"books": 1}, n=3
    )
    assert [c["item_id"] for c in selected] == ["b", "d", "e"]
    assert counts == {"books": 1, "toys": 2}
    assert backfilled == 0
    assert any(c["item_id"] == "c" for c in parked)


def test_input_validation():
    with pytest.raises(RerankingError):
        rerank([("a", "not-a-score", "x")], {"x": 1}, n=1)
    with pytest.raises(RerankingError):
        rerank([{"id": "a"}], {"x": 1}, n=1)
    with pytest.raises(RerankingError):
        rerank(CANDIDATES, {"x": -1}, n=1)
    with pytest.raises(RerankingError):
        rerank(CANDIDATES, {"x": 1.5}, n=1)
    with pytest.raises(RerankingError):
        rerank(CANDIDATES, {"x": 1}, n=-2)
    with pytest.raises(RerankingError):
        rerank("not-a-sequence", {"x": 1}, n=1)


def test_nan_scores_sort_last_without_crashing():
    candidates = [("a", math.nan, "x"), ("b", 0.5, "x"), ("c", 0.4, "y")]
    records = rerank(candidates, {"x": 10, "y": 10}, n=3)
    assert [r["item_id"] for r in records] == ["b", "c", "a"]
