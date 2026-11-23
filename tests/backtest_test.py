import datetime

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import implicit
from implicit.backtest import TimeBasedSplitReport, time_based_split


def _events():
    """A small, deterministic event log around ``cutoff=5``.

    Training (t<=5): u1->10,11  u2->10,12  u3->13  (u4 has no history).
    After the cutoff:
      u1->11 (seen), u1->14 (item unseen in training), u2->11 (valid target),
      u2->10 (seen), u3->13 (seen), u4->10 (no user history),
      u3->12 marked non-positive (ignored).
    """
    return {
        "user": [1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 4, 3],
        "item": [10, 11, 10, 12, 13, 11, 14, 11, 10, 13, 10, 12],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "value": [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        "positive": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    }


def _dense(mat):
    return np.asarray(mat.todense())


def test_time_based_split_returns_aligned_csr():
    train, validation = time_based_split(_events(), cutoff=5)

    assert isinstance(train, csr_matrix)
    assert isinstance(validation, csr_matrix)
    # fixed axis covers the 4 users and 5 items seen anywhere in the log
    assert train.shape == validation.shape == (4, 5)


def test_time_based_split_uses_only_pre_cutoff_events_for_training():
    train, _ = time_based_split(_events(), cutoff=5)

    expected = np.zeros(train.shape)
    expected[0, 0] = 1  # u1 -> item 10
    expected[0, 1] = 1  # u1 -> item 11
    expected[1, 0] = 1  # u2 -> item 10
    expected[1, 2] = 2  # u2 -> item 12 (weighted)
    expected[2, 3] = 1  # u3 -> item 13
    np.testing.assert_array_equal(_dense(train), expected)


def test_time_based_split_targets_are_post_cutoff_and_unseen():
    events = _events()
    _, validation = time_based_split(events, cutoff=5)

    # only (u2, item 11) survives: post-cutoff, unseen at the cutoff, the item
    # has training history, and u2 has history.
    users, items = validation.nonzero()
    assert list(zip(users, items)) == [(1, 1)]
    for u, i in zip(users, items):
        times = [
            t
            for uu, ii, t in zip(events["user"], events["item"], events["timestamp"])
            if uu == u + 1 and ii == i + 10
        ]
        assert max(times) > 5

    # (u2, item 10) is seen at the cutoff and must not be a target
    assert validation[1, 0] == 0


def test_time_based_split_ignores_non_positive_feedback():
    train, validation = time_based_split(_events(), cutoff=5)

    # u3 -> item 12 at t=12 is marked non-positive and must be excluded
    assert train[2, 2] == 0
    assert validation[2, 2] == 0


def test_time_based_split_reports_accounts_without_enough_history():
    _, validation, report = time_based_split(
        _events(), cutoff=5, min_history=2, return_report=True
    )

    assert isinstance(report, TimeBasedSplitReport)
    # u3 has 1 training item, u4 has none; both are below min_history=2
    assert set(report.insufficient_history_users.tolist()) == {3, 4}
    assert report.num_insufficient_history_dropped == 1  # u4 -> item 10
    # their post-cutoff targets are dropped from validation
    assert validation[2].nnz == 0
    assert validation[3].nnz == 0
    assert "training items" in report.summary()


def test_time_based_split_reports_with_default_min_history():
    _, _, report = time_based_split(_events(), cutoff=5, return_report=True)
    # only u4 (no history at all) is below the default min_history=1
    assert set(report.insufficient_history_users.tolist()) == {4}


def test_time_based_split_drops_seen_targets():
    _, validation, report = time_based_split(_events(), cutoff=5, return_report=True)
    # u1->11, u2->10 and u3->13 are re-purchases of items seen at the cutoff
    assert report.num_seen_targets_dropped == 3
    assert validation[0, 1] == 0
    assert validation[1, 0] == 0
    assert validation[2, 3] == 0


def test_time_based_split_requires_item_training_history():
    _, validation, report = time_based_split(_events(), cutoff=5, return_report=True)
    # item 14 only ever appears after the cutoff -> not a fair target
    assert report.num_unknown_item_dropped == 1
    assert validation[0, 4] == 0
    # item 11 has training history and the (u2, 11) pair is new -> kept
    assert validation[1, 1] == 1


def test_time_based_split_fixed_axes_are_stable():
    events = _events()
    user_ids = [1, 2, 3, 4]
    item_ids = list(range(10, 15))

    train, validation = time_based_split(
        events, cutoff=5, user_ids=user_ids, item_ids=item_ids
    )
    assert train.shape == validation.shape == (4, 5)

    # reordering the fixed axes reorders the matrices accordingly
    rev_train, rev_validation = time_based_split(
        events, cutoff=5, user_ids=user_ids[::-1], item_ids=item_ids[::-1]
    )
    np.testing.assert_array_equal(_dense(rev_train), _dense(train)[::-1, ::-1])
    np.testing.assert_array_equal(_dense(rev_validation), _dense(validation)[::-1, ::-1])


def test_time_based_split_fixed_axes_include_post_cutoff_only_ids():
    events = _events()
    # user 5 and item 20 only appear after the cutoff
    events["user"].append(5)
    events["item"].append(20)
    events["timestamp"].append(20)
    events["value"].append(1)
    events["positive"].append(1)

    train, validation = time_based_split(
        events, cutoff=5, user_ids=[1, 2, 3, 4, 5], item_ids=list(range(10, 21))
    )
    assert train.shape == validation.shape == (5, 11)
    # user 5 has no training history, so its target is not kept
    assert validation[4].nnz == 0


def test_time_based_split_rejects_unknown_fixed_ids():
    with pytest.raises(ValueError):
        time_based_split(_events(), cutoff=5, user_ids=[1, 2])
    with pytest.raises(ValueError):
        time_based_split(_events(), cutoff=5, item_ids=[10, 11])


def test_time_based_split_rejects_duplicate_fixed_ids():
    with pytest.raises(ValueError):
        time_based_split(_events(), cutoff=5, user_ids=[1, 1, 2, 3, 4])


def test_time_based_split_supports_datetime_cutoff():
    events = _events()
    start = datetime.datetime(2024, 1, 1)
    events["timestamp"] = [start + datetime.timedelta(days=t) for t in events["timestamp"]]

    numeric_train, numeric_validation = time_based_split(_events(), cutoff=5)
    time_train, time_validation = time_based_split(
        events, cutoff=start + datetime.timedelta(days=5)
    )

    np.testing.assert_array_equal(_dense(time_train), _dense(numeric_train))
    np.testing.assert_array_equal(_dense(time_validation), _dense(numeric_validation))


def test_time_based_split_supports_string_timestamps():
    events = _events()
    start = datetime.datetime(2024, 1, 1)
    events["timestamp"] = [
        (start + datetime.timedelta(days=t)).isoformat() for t in events["timestamp"]
    ]

    numeric_train, numeric_validation = time_based_split(_events(), cutoff=5)
    text_train, text_validation = time_based_split(
        events, cutoff=(start + datetime.timedelta(days=5)).isoformat()
    )

    np.testing.assert_array_equal(_dense(text_train), _dense(numeric_train))
    np.testing.assert_array_equal(_dense(text_validation), _dense(numeric_validation))


def test_time_based_split_supports_2d_and_structured_input():
    events = _events()
    columns = [events[k] for k in ("user", "item", "timestamp", "value", "positive")]
    as_2d = np.column_stack(columns)

    dict_train, dict_validation = time_based_split(events, cutoff=5)
    array_train, array_validation = time_based_split(as_2d, cutoff=5)

    np.testing.assert_array_equal(_dense(array_train), _dense(dict_train))
    np.testing.assert_array_equal(_dense(array_validation), _dense(dict_validation))

    structured = np.array(
        list(zip(*columns)),
        dtype=[
            ("user", "i8"),
            ("item", "i8"),
            ("timestamp", "i8"),
            ("value", "f8"),
            ("positive", "bool"),
        ],
    )
    struct_train, struct_validation = time_based_split(structured, cutoff=5)
    np.testing.assert_array_equal(_dense(struct_train), _dense(dict_train))
    np.testing.assert_array_equal(_dense(struct_validation), _dense(dict_validation))


def test_time_based_split_without_value_or_positive_columns():
    events = _events()
    del events["value"]
    del events["positive"]
    train, validation = time_based_split(events, cutoff=5)

    # without a value column every interaction has weight 1
    assert train[1, 2] == 1
    # without a positive column every post-cutoff event is a candidate, so
    # (u3, item 12) is now considered (and dropped as a seen item? no: unseen
    # at the cutoff, but item 12 has training history and u3 has history)
    assert validation[2, 2] == 1


def test_time_based_split_rejects_missing_columns():
    events = _events()
    del events["timestamp"]
    with pytest.raises(ValueError):
        time_based_split(events, cutoff=5)


def test_time_based_split_rejects_negative_min_history():
    with pytest.raises(ValueError):
        time_based_split(_events(), cutoff=5, min_history=-1)


def test_time_based_split_rejects_empty_events():
    with pytest.raises(ValueError):
        time_based_split({"user": [], "item": [], "timestamp": []}, cutoff=5)


def test_report_drop_counts_are_consistent():
    _, validation, report = time_based_split(_events(), cutoff=5, return_report=True)
    accounted = (
        report.num_validation_targets
        + report.num_seen_targets_dropped
        + report.num_unknown_item_dropped
        + report.num_insufficient_history_dropped
    )
    assert accounted == report.num_validation_candidates
    assert report.num_validation_targets == validation.nnz


def test_report_as_dict_is_json_serialisable():
    import json

    _, _, report = time_based_split(_events(), cutoff=5, return_report=True)
    payload = report.as_dict()
    json.dumps(payload)  # must not raise
    assert payload["num_users"] == 4
    assert payload["num_items"] == 5
    assert payload["num_validation_targets"] == 1


def test_time_based_split_supports_string_ids():
    events = _events()
    user_names = {1: "alice", 2: "bob", 3: "carol", 4: "dave"}
    item_names = {10: "a", 11: "b", 12: "c", 13: "d", 14: "e"}
    events["user"] = [user_names[u] for u in events["user"]]
    events["item"] = [item_names[i] for i in events["item"]]

    numeric_train, numeric_validation = time_based_split(_events(), cutoff=5)
    text_train, text_validation, report = time_based_split(
        events, cutoff=5, return_report=True
    )

    # the ids are sorted, so alice/bob/carol/dave and a/b/c/d/e line up with the
    # numeric axes
    assert set(report.insufficient_history_users.tolist()) == {"dave"}
    np.testing.assert_array_equal(_dense(text_train), _dense(numeric_train))
    np.testing.assert_array_equal(_dense(text_validation), _dense(numeric_validation))


def test_time_based_split_supports_mixed_type_ids():
    # ids that can't be ordered together still get a stable first-seen axis
    events = {
        "user": np.array([1, "a", 1], dtype=object),
        "item": [10, 10, 11],
        "timestamp": [1, 2, 3],
    }
    train, validation = time_based_split(events, cutoff=2)
    assert train.shape == validation.shape == (2, 2)
    # training keeps the two pre-cutoff events; the post-cutoff event targets
    # item 11, which has no training history, so it is dropped
    assert train.nnz == 2
    assert validation.nnz == 0

    # give item 11 some training history via another user before the cutoff
    events["user"] = np.append(events["user"], "a")
    events["item"] = np.append(events["item"], 11)
    events["timestamp"] = np.append(events["timestamp"], 1)
    _, validation = time_based_split(events, cutoff=2)
    assert validation.nnz == 1


def test_time_based_split_supports_pandas_dataframe():
    pandas = pytest.importorskip("pandas")
    events = pandas.DataFrame(_events())

    dict_train, dict_validation = time_based_split(_events(), cutoff=5)
    df_train, df_validation = time_based_split(events, cutoff=5)

    np.testing.assert_array_equal(_dense(df_train), _dense(dict_train))
    np.testing.assert_array_equal(_dense(df_validation), _dense(dict_validation))


class TestEvaluateTimeBasedSplit:
    def _build(self):
        # Each user has a dense training history over items 0..7, excluding its
        # own holdout item (also in 0..7), which is revealed after the cutoff.
        # Items 8 and 9 are only ever used as post-cutoff targets for users 8
        # and 9, so they have no training history and are dropped.
        users, items, times = [], [], []
        for user in range(10):
            for item in range(8):
                if item == user:
                    continue
                users.append(user)
                items.append(item)
                times.append(0)
            users.append(user)
            items.append(user)
            times.append(1)

        events = {
            "user": users,
            "item": items,
            "timestamp": times,
            "value": np.ones(len(users)),
            "positive": np.ones(len(users), dtype=bool),
        }
        return time_based_split(events, cutoff=0)

    def test_evaluate_time_based_split_runs_real_evaluation(self):
        train, validation = self._build()
        assert validation.nnz == 8

        model = implicit.als.AlternatingLeastSquares(
            factors=8, regularization=0.1, iterations=3, use_gpu=False, random_state=0
        )
        model.fit(train, show_progress=False)

        metrics = implicit.backtest.evaluate_time_based_split(
            model, train, validation, K=5, show_progress=False
        )
        assert set(metrics) == {"precision", "map", "ndcg", "auc"}
        for value in metrics.values():
            assert np.isfinite(value)
            assert 0.0 <= value <= 1.0

        # the wrapper should match calling the evaluation API directly
        from implicit.evaluation import ranking_metrics_at_k

        direct = ranking_metrics_at_k(model, train, validation, K=5, show_progress=False)
        assert direct == metrics

    def test_evaluate_time_based_split_rejects_empty_validation(self):
        train, _ = self._build()
        model = implicit.als.AlternatingLeastSquares(
            factors=8, iterations=1, use_gpu=False, random_state=0
        )
        model.fit(train, show_progress=False)
        empty = csr_matrix(train.shape)
        with pytest.raises(ValueError):
            implicit.backtest.evaluate_time_based_split(
                model, train, empty, show_progress=False
            )

    def test_evaluate_time_based_split_rejects_mismatched_axes(self):
        train, validation = self._build()
        model = implicit.als.AlternatingLeastSquares(
            factors=8, iterations=1, use_gpu=False, random_state=0
        )
        model.fit(train, show_progress=False)
        smaller = csr_matrix(validation[:3])
        with pytest.raises(ValueError):
            implicit.backtest.evaluate_time_based_split(
                model, train, smaller, show_progress=False
            )
