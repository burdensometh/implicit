from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

import implicit
from implicit.feedback import (
    FeedbackConfig,
    FeedbackMatrix,
    build_feedback_matrix,
    fit_feedback,
)

CUTOFF = datetime(2024, 6, 1)
BASE_WEIGHTS = {"view": 1.0, "cart": 3.0, "purchase": 5.0, "reject": -6.0}


def event(user, item, type_, days_ago, event_id=None):  # noqa: A002 - short test helper
    return {
        "user": user,
        "item": item,
        "type": type_,
        "event_id": event_id if event_id is not None else f"{user}:{item}:{type_}:{days_ago}",
        "timestamp": CUTOFF - timedelta(days=days_ago),
    }


def config(**kwargs):
    params = {"training_cutoff": CUTOFF, "event_weights": BASE_WEIGHTS}
    params.update(kwargs)
    return FeedbackConfig(**params)


def value(feedback, user, item):
    return feedback.matrix[feedback.user_index[user], feedback.item_index[item]]


# ---------------------------------------------------------------------------------------------
# deduplication by event identity
# ---------------------------------------------------------------------------------------------


def test_replayed_events_are_collapsed():
    events = [event("u1", "i1", "view", 1, event_id="e1")] * 5
    feedback = build_feedback_matrix(events, config())
    assert feedback.shape == (1, 1)
    assert feedback.stats.events_received == 5
    assert feedback.stats.events_after_dedup == 1
    assert feedback.stats.duplicates_removed == 4
    assert feedback.stats.conflicts == 0
    # one un-decayed view at age 1
    assert_allclose(value(feedback, "u1", "i1"), 1.0)


def test_dedup_is_order_independent():
    events = [
        event("u1", "i1", "view", 2, event_id="e1"),
        event("u2", "i2", "purchase", 1, event_id="e2"),
        event("u1", "i1", "view", 2, event_id="e1"),
        event("u3", "i3", "cart", 3, event_id="e3"),
        event("u2", "i2", "purchase", 1, event_id="e2"),
    ]
    forward = build_feedback_matrix(events, config())
    backward = build_feedback_matrix(list(reversed(events)), config())
    assert_array_equal(forward.user_ids, backward.user_ids)
    assert_array_equal(forward.item_ids, backward.item_ids)
    assert (forward.matrix != backward.matrix).nnz == 0
    assert forward.stats == backward.stats


def test_duplicate_identical_payload_is_not_a_conflict():
    events = [event("u1", "i1", "view", 1, event_id="e1")] * 3
    feedback = build_feedback_matrix(events, config())
    assert feedback.stats.conflicts == 0


def test_conflict_latest_policy_keeps_newest_payload():
    events = [
        event("u1", "i1", "view", 5, event_id="dup"),
        event("u1", "i1", "purchase", 1, event_id="dup"),
    ]
    feedback = build_feedback_matrix(events, config(conflict_policy="latest"))
    assert feedback.stats.conflicts == 1
    assert feedback.stats.duplicates_removed == 1
    assert_allclose(value(feedback, "u1", "i1"), 5.0)


def test_conflict_earliest_policy_keeps_oldest_payload():
    events = [
        event("u1", "i1", "view", 5, event_id="dup"),
        event("u1", "i1", "purchase", 1, event_id="dup"),
    ]
    feedback = build_feedback_matrix(events, config(conflict_policy="earliest"))
    assert_allclose(value(feedback, "u1", "i1"), 1.0)


def test_conflict_resolution_is_order_independent():
    events = [
        event("u1", "i1", "view", 5, event_id="dup"),
        event("u1", "i1", "purchase", 1, event_id="dup"),
        event("u2", "i2", "cart", 4, event_id="dup2"),
        event("u2", "i2", "reject", 3, event_id="dup2"),
    ]
    for policy in ("latest", "earliest"):
        forward = build_feedback_matrix(events, config(conflict_policy=policy))
        backward = build_feedback_matrix(list(reversed(events)), config(conflict_policy=policy))
        assert (forward.matrix != backward.matrix).nnz == 0


def test_event_identity_auto_falls_back_to_tuple():
    events = [
        {"user": "u1", "item": "i1", "type": "view", "event_id": None,
         "timestamp": CUTOFF - timedelta(days=1)},
        {"user": "u1", "item": "i1", "type": "view", "event_id": None,
         "timestamp": CUTOFF - timedelta(days=1)},
    ]
    feedback = build_feedback_matrix(events, config(event_identity="auto"))
    assert feedback.stats.duplicates_removed == 1
    assert_allclose(value(feedback, "u1", "i1"), 1.0)


def test_event_identity_requires_id_by_default():
    events = [
        {"user": "u1", "item": "i1", "type": "view", "event_id": None,
         "timestamp": CUTOFF - timedelta(days=1)},
    ]
    with pytest.raises(ValueError, match="event_id"):
        build_feedback_matrix(events, config())


# ---------------------------------------------------------------------------------------------
# behaviour weight and time decay
# ---------------------------------------------------------------------------------------------


def test_behaviour_type_weighting():
    events = [
        event("u1", "i1", "view", 0),
        event("u1", "i2", "cart", 0),
        event("u1", "i3", "purchase", 0),
        event("u1", "i4", "reject", 0),
    ]
    feedback = build_feedback_matrix(events, config())
    assert_allclose(value(feedback, "u1", "i1"), 1.0)
    assert_allclose(value(feedback, "u1", "i2"), 3.0)
    assert_allclose(value(feedback, "u1", "i3"), 5.0)
    assert_allclose(value(feedback, "u1", "i4"), -6.0)
    assert feedback.stats.positive_pairs == 3
    assert feedback.stats.negative_pairs == 1


def test_time_decay_relative_to_cutoff():
    events = [
        event("u1", "i1", "view", 0),
        event("u1", "i2", "view", 2),
        event("u1", "i3", "view", 4),
    ]
    feedback = build_feedback_matrix(events, config(half_life=timedelta(days=2)))
    assert_allclose(value(feedback, "u1", "i1"), 1.0)
    assert_allclose(value(feedback, "u1", "i2"), 0.5)
    assert_allclose(value(feedback, "u1", "i3"), 0.25)


def test_per_type_half_life():
    events = [
        event("u1", "i1", "view", 10),
        event("u1", "i2", "purchase", 10),
    ]
    feedback = build_feedback_matrix(
        events,
        config(half_life={"view": timedelta(days=2), "cart": timedelta(days=2),
                          "purchase": timedelta(days=10), "reject": timedelta(days=10)}),
    )
    # browsing fades much faster than a durable purchase
    assert_allclose(value(feedback, "u1", "i1"), 2.0**-5, rtol=1e-6)
    assert_allclose(value(feedback, "u1", "i2"), 2.5, rtol=1e-6)


def test_numeric_timestamps_and_half_life():
    events = [
        {"user": 0, "item": 0, "type": "view", "event_id": 1, "timestamp": 1000.0},
        {"user": 0, "item": 1, "type": "view", "event_id": 2, "timestamp": 900.0},
    ]
    cfg = FeedbackConfig(training_cutoff=1000.0, event_weights={"view": 1.0}, half_life=100.0)
    feedback = build_feedback_matrix(events, cfg)
    assert_allclose(value(feedback, 0, 0), 1.0)
    assert_allclose(value(feedback, 0, 1), 0.5)


def test_timedelta_half_life_requires_datetime_cutoff():
    with pytest.raises(TypeError, match="timedelta"):
        FeedbackConfig(training_cutoff=1000.0, event_weights={"view": 1.0},
                       half_life=timedelta(days=1))


def test_datetime_and_numeric_mixing_is_rejected():
    events = [{"user": "u", "item": "i", "type": "view", "event_id": "e", "timestamp": 5.0}]
    with pytest.raises(TypeError, match="datetime"):
        build_feedback_matrix(events, config())


def test_aware_and_naive_datetimes_cannot_be_mixed():
    aware_cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    cfg = FeedbackConfig(training_cutoff=aware_cutoff, event_weights={"view": 1.0})
    events = [{"user": "u", "item": "i", "type": "view", "event_id": "e",
               "timestamp": datetime(2024, 5, 31)}]
    with pytest.raises(TypeError, match="timezone"):
        build_feedback_matrix(events, cfg)


def test_aware_datetimes_are_supported():
    cutoff = datetime(2024, 6, 1, tzinfo=timezone.utc)
    events = [
        {"user": "u", "item": "i", "type": "view", "event_id": "e",
         "timestamp": cutoff - timedelta(hours=12)},
    ]
    cfg = FeedbackConfig(
        training_cutoff=cutoff, event_weights={"view": 1.0}, half_life=timedelta(days=1)
    )
    feedback = build_feedback_matrix(events, cfg)
    assert_allclose(value(feedback, "u", "i"), 0.5 ** 0.5, rtol=1e-6)


def test_future_events_dropped_by_default():
    events = [
        event("u1", "i1", "view", 1),
        {"user": "u1", "item": "i2", "type": "purchase", "event_id": "future",
         "timestamp": CUTOFF + timedelta(days=1)},
    ]
    feedback = build_feedback_matrix(events, config())
    assert feedback.stats.future_dropped == 1
    assert "i2" not in feedback.item_index


def test_future_events_clamped_when_configured():
    events = [
        {"user": "u1", "item": "i2", "type": "purchase", "event_id": "future",
         "timestamp": CUTOFF + timedelta(days=1)},
    ]
    feedback = build_feedback_matrix(
        events, config(future_policy="clamp", half_life=timedelta(days=1))
    )
    # clamped to the cutoff -> no decay -> full weight
    assert_allclose(value(feedback, "u1", "i2"), 5.0)
    assert feedback.stats.future_dropped == 0


def test_max_age_drops_stale_events():
    events = [
        event("u1", "i1", "view", 1),
        event("u1", "i2", "view", 100),
    ]
    feedback = build_feedback_matrix(events, config(max_age=timedelta(days=30)))
    assert feedback.stats.too_old_dropped == 1
    assert "i2" not in feedback.item_index


def test_unknown_event_type_errors_by_default():
    events = [event("u1", "i1", "mystery", 0)]
    with pytest.raises(ValueError, match="unknown type"):
        build_feedback_matrix(events, config())


def test_unknown_event_type_can_be_ignored():
    events = [event("u1", "i1", "mystery", 0), event("u1", "i2", "view", 0)]
    feedback = build_feedback_matrix(events, config(unknown_event_policy="ignore"))
    assert feedback.stats.unknown_type_dropped == 1
    assert feedback.shape == (1, 1)


# ---------------------------------------------------------------------------------------------
# purchase / rejection cancellation
# ---------------------------------------------------------------------------------------------


def _sale_and_return():
    return [
        event("u1", "i1", "purchase", 5, event_id="buy"),
        event("u1", "i1", "reject", 1, event_id="return"),
    ]


def test_net_cancellation():
    feedback = build_feedback_matrix(_sale_and_return(), config(cancellation_policy="net"))
    # 5 - 6 = -1 : a net dislike survives
    assert_allclose(value(feedback, "u1", "i1"), -1.0)
    assert feedback.stats.negative_pairs == 1


def test_net_exact_cancellation_drops_pair():
    events = [
        event("u1", "i1", "purchase", 1, event_id="buy"),
        {"user": "u1", "item": "i1", "type": "reject", "event_id": "return",
         "timestamp": CUTOFF - timedelta(days=1)},
    ]
    cfg = FeedbackConfig(
        training_cutoff=CUTOFF,
        event_weights={"purchase": 6.0, "reject": -6.0},
    )
    feedback = build_feedback_matrix(events, cfg)
    assert feedback.matrix.nnz == 0
    assert feedback.stats.weak_dropped == 1


def test_rejection_wins_policy():
    feedback = build_feedback_matrix(
        _sale_and_return(), config(cancellation_policy="rejection_wins")
    )
    # any rejection makes the pair negative, magnitude is the summed rejection
    assert_allclose(value(feedback, "u1", "i1"), -6.0)


def test_latest_wins_policy():
    feedback = build_feedback_matrix(_sale_and_return(), config(cancellation_policy="latest_wins"))
    # latest event is the return -> negative
    assert_allclose(value(feedback, "u1", "i1"), -6.0)

    repurchase = [
        event("u1", "i1", "reject", 5, event_id="return"),
        event("u1", "i1", "purchase", 1, event_id="buy"),
    ]
    feedback = build_feedback_matrix(repurchase, config(cancellation_policy="latest_wins"))
    assert_allclose(value(feedback, "u1", "i1"), 5.0)


def test_min_confidence_threshold():
    events = [event("u1", "i1", "view", 0), event("u1", "i2", "view", 0)]
    feedback = build_feedback_matrix(events, config(min_confidence=0.5))
    assert feedback.matrix.nnz == 2

    decayed = [event("u1", "i1", "view", 4), event("u1", "i2", "view", 0)]
    feedback = build_feedback_matrix(
        decayed, config(min_confidence=0.5, half_life=timedelta(days=2))
    )
    assert feedback.stats.weak_dropped == 1
    assert "i1" not in feedback.item_index


def test_cancellation_is_order_independent():
    events = _sale_and_return() + [
        event("u2", "i2", "purchase", 2, event_id="b2"),
        event("u2", "i2", "reject", 1, event_id="r2"),
    ]
    for policy in ("net", "latest_wins", "rejection_wins"):
        forward = build_feedback_matrix(events, config(cancellation_policy=policy))
        backward = build_feedback_matrix(list(reversed(events)), config(cancellation_policy=policy))
        assert (forward.matrix != backward.matrix).nnz == 0


# ---------------------------------------------------------------------------------------------
# generalized inputs and vocabularies
# ---------------------------------------------------------------------------------------------


def test_columnar_input_matches_row_input():
    events = [
        event("u1", "i1", "view", 1, event_id="e1"),
        event("u2", "i2", "purchase", 2, event_id="e2"),
    ]
    rows = build_feedback_matrix(events, config())
    columns = build_feedback_matrix(
        users=[e["user"] for e in events],
        items=[e["item"] for e in events],
        event_types=[e["type"] for e in events],
        event_ids=[e["event_id"] for e in events],
        timestamps=[e["timestamp"] for e in events],
        config=config(),
    )
    assert (rows.matrix != columns.matrix).nnz == 0
    assert_array_equal(rows.user_ids, columns.user_ids)


def test_mapping_of_columns():
    events = {
        "user": ["u1", "u1"],
        "item": ["i1", "i2"],
        "type": ["view", "view"],
        "event_id": ["e1", "e2"],
        "timestamp": [CUTOFF, CUTOFF],
    }
    feedback = build_feedback_matrix(events, config())
    assert feedback.shape == (1, 2)


def test_explicit_vocabulary_is_stable():
    events = [event("u2", "i2", "view", 0)]
    feedback = build_feedback_matrix(events, config(), user_ids=["u1", "u2"],
                                     item_ids=["i1", "i2"])
    assert feedback.shape == (2, 2)
    assert feedback.user_index == {"u1": 0, "u2": 1}
    assert feedback.item_index == {"i1": 0, "i2": 1}
    assert feedback.matrix[1, 1] == pytest.approx(1.0)


def test_explicit_vocabulary_rejects_unknown_id():
    events = [event("u9", "i1", "view", 0)]
    with pytest.raises(KeyError, match="u9"):
        build_feedback_matrix(events, config(), user_ids=["u1"])


def test_id_vocabulary_is_sorted_not_insertion_ordered():
    events = [
        event("u_b", "i_x", "view", 0),
        event("u_a", "i_y", "view", 0),
    ]
    forward = build_feedback_matrix(events, config())
    backward = build_feedback_matrix(list(reversed(events)), config())
    assert list(forward.user_ids) == ["u_a", "u_b"]
    assert list(backward.user_ids) == ["u_a", "u_b"]


def test_user_row_helper():
    events = [event("u1", "i1", "view", 0), event("u2", "i2", "view", 0)]
    feedback = build_feedback_matrix(events, config())
    row = feedback.user_row("u2")
    assert row.shape == (1, 2)
    assert row[0, feedback.item_index["i2"]] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"event_weights": {}}, ValueError),
        ({"event_weights": {"view": -1.0}}, ValueError),
        ({"event_weights": {"view": float("nan")}}, ValueError),
        ({"half_life": 0.0}, ValueError),
        ({"half_life": -1.0}, ValueError),
        ({"future_policy": "keep"}, ValueError),
        ({"conflict_policy": "newest"}, ValueError),
        ({"cancellation_policy": "sum"}, ValueError),
        ({"unknown_event_policy": "raise"}, ValueError),
        ({"event_identity": "hash"}, ValueError),
        ({"min_confidence": -1.0}, ValueError),
    ],
)
def test_invalid_config(kwargs, error):
    with pytest.raises(error):
        config(**kwargs)


def test_half_life_mapping_must_cover_every_type():
    with pytest.raises(ValueError, match="missing"):
        config(half_life={"view": timedelta(days=1)})


def test_requires_events():
    with pytest.raises(ValueError, match="no events"):
        build_feedback_matrix([], config())


def test_mixed_events_and_columns_rejected():
    events = [event("u1", "i1", "view", 0)]
    with pytest.raises(ValueError, match="not both"):
        build_feedback_matrix(events, config(), users=["u1"])


def test_unequal_columns_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        build_feedback_matrix(
            users=["u1", "u2"], items=["i1"], event_types=["view"],
            event_ids=["e1"], timestamps=[CUTOFF], config=config(),
        )


def test_missing_event_field_rejected():
    with pytest.raises(ValueError, match="missing fields"):
        build_feedback_matrix([{"user": "u1", "item": "i1"}], config())


# ---------------------------------------------------------------------------------------------
# integration with the repository ALS
# ---------------------------------------------------------------------------------------------


def _toy_events():
    events = []
    # users 0..5 like items aligned with their parity (checkerboard-ish, with noise)
    for user in range(6):
        for item in range(6):
            if item % 2 == user % 2:
                events.append(event(user, item, "purchase", 1, event_id=f"{user}:{item}:buy"))
            elif item % 3 == 0:
                events.append(event(user, item, "reject", 1, event_id=f"{user}:{item}:no"))
    return events


def test_als_actually_fits_the_feedback_matrix():
    feedback = build_feedback_matrix(_toy_events(), config(half_life=timedelta(days=30)))
    model = implicit.als.AlternatingLeastSquares(
        factors=8, iterations=5, regularization=0.01, random_state=0, use_gpu=False
    )
    returned = fit_feedback(model, feedback, show_progress=False)

    assert returned is feedback
    assert model.user_factors is not None
    assert model.item_factors is not None
    assert model.user_factors.shape == (feedback.shape[0], 8)
    assert model.item_factors.shape == (feedback.shape[1], 8)
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()

    # the fit used the confidence values, including negative ones - the model should rank the
    # user's positive items above the items they rejected
    user = 0
    scores = model.item_factors.dot(model.user_factors[user])
    positives = [i for i in range(6) if feedback.matrix[user, i] > 0]
    negatives = [i for i in range(6) if feedback.matrix[user, i] < 0]
    assert positives and negatives
    assert min(scores[positives]) > max(scores[negatives])


def test_fit_feedback_builds_and_fits_from_raw_events():
    events = _toy_events()
    model = implicit.als.AlternatingLeastSquares(
        factors=4, iterations=2, random_state=1, use_gpu=False
    )
    feedback = fit_feedback(model, events, config(), show_progress=False)
    assert isinstance(feedback, FeedbackMatrix)
    assert model.user_factors.shape[0] == feedback.shape[0]


def test_matrix_is_float32_csr_by_default():
    feedback = build_feedback_matrix(_toy_events(), config())
    assert isinstance(feedback.matrix, csr_matrix)
    assert feedback.matrix.dtype == np.float32


def test_custom_dtype():
    feedback = build_feedback_matrix(_toy_events(), config(dtype=np.float64))
    assert feedback.matrix.dtype == np.float64


def test_response_is_reproducible_across_runs():
    events = _toy_events()
    first = build_feedback_matrix(events, config(half_life=timedelta(days=7)))
    second = build_feedback_matrix(events, config(half_life=timedelta(days=7)))
    assert_array_equal(first.matrix.toarray(), second.matrix.toarray())
    assert first.stats == second.stats
    assert_array_equal(first.user_ids, second.user_ids)
