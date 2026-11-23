"""Tests for the session based co-occurrence helpers and model integration."""

import datetime

import numpy as np
import pytest

import implicit
from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, TFIDFRecommender
from implicit.sessions import (
    DEFAULT_IDLE_TIMEOUT,
    Sessionizer,
    build_session_matrix,
    build_sessions,
    cooccurrence_counts,
    matrix_from_sessions,
    split_sessions,
)


def assert_sessions_equal(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert list(got) == list(want)


class TestSplitSessions:
    def test_single_session_without_idle_gap(self):
        events = [("u1", "a", 0), ("u1", "b", 60), ("u1", "c", 120)]
        sessions = split_sessions(events, idle_timeout=1800)
        assert_sessions_equal(sessions, [["a", "b", "c"]])
        assert all(session.user == "u1" for session in sessions)

    def test_idle_gap_starts_new_session(self):
        events = [("u1", "a", 0), ("u1", "b", 60), ("u1", "c", 10_000)]
        sessions = split_sessions(events, idle_timeout=1800)
        assert_sessions_equal(sessions, [["a", "b"], ["c"]])

    def test_idle_timeout_is_exclusive(self):
        # a gap exactly equal to the timeout stays in the same session
        events = [("u1", "a", 0), ("u1", "b", 1800)]
        assert_sessions_equal(split_sessions(events, idle_timeout=1800), [["a", "b"]])
        # one second more splits the session
        events = [("u1", "a", 0), ("u1", "b", 1801)]
        assert_sessions_equal(split_sessions(events, idle_timeout=1800), [["a"], ["b"]])

    def test_duplicates_merged_within_session(self):
        events = [("u1", "a", 0), ("u1", "b", 60), ("u1", "a", 120)]
        assert_sessions_equal(split_sessions(events, idle_timeout=1800), [["a", "b"]])

    def test_duplicates_preserved_when_requested(self):
        events = [("u1", "a", 0), ("u1", "b", 60), ("u1", "a", 120)]
        sessions = build_sessions(events, idle_timeout=1800, dedupe=False)
        assert_sessions_equal(sessions, [["a", "b", "a"]])

    def test_events_are_sorted_by_time(self):
        events = [("u1", "b", 120), ("u1", "a", 0), ("u1", "c", 60)]
        assert_sessions_equal(split_sessions(events, idle_timeout=1800), [["a", "c", "b"]])

    def test_users_are_sessions_are_independent(self):
        # interleaved users should not affect each other's session boundaries
        events = [
            ("u1", "a", 0),
            ("u2", "x", 60),
            ("u1", "b", 120),
            ("u2", "y", 180),
        ]
        sessions = split_sessions(events, idle_timeout=1800)
        assert_sessions_equal(sessions, [["a", "b"], ["x", "y"]])
        assert [session.user for session in sessions] == ["u1", "u2"]

    def test_sessions_split_per_user(self):
        events = [("u1", "a", 0), ("u2", "b", 1), ("u1", "c", 2)]
        sessions = split_sessions(events, idle_timeout=60)
        assert_sessions_equal(sessions, [["a", "c"], ["b"]])

    def test_session_attributes(self):
        events = [("u1", "a", 0), ("u1", "b", 60)]
        (session,) = split_sessions(events, idle_timeout=1800)
        assert session.user == "u1"
        assert [event.item for event in session.events] == ["a", "b"]
        assert [event.timestamp for event in session.events] == [0, 60]
        assert session.items == ["a", "b"]

    def test_empty_events(self):
        assert split_sessions([], idle_timeout=1800) == []
        assert build_sessions([], idle_timeout=1800) == []

    def test_datetime_timestamps(self):
        events = [
            ("u1", "a", datetime.datetime(2020, 1, 1, 12, 0)),
            ("u1", "b", datetime.datetime(2020, 1, 1, 12, 10)),
            ("u1", "c", datetime.datetime(2020, 1, 2, 12, 0)),
        ]
        assert_sessions_equal(split_sessions(events, idle_timeout=1800), [["a", "b"], ["c"]])

    def test_numpy_timestamps(self):
        events = [
            ("u1", "a", np.datetime64("2020-01-01T12:00")),
            ("u1", "b", np.datetime64("2020-01-01T12:10")),
            ("u1", "c", np.datetime64("2020-01-02T12:00")),
        ]
        assert_sessions_equal(split_sessions(events, idle_timeout=1800), [["a", "b"], ["c"]])

    def test_numpy_array_input(self):
        events = np.array([[0, 10, 0], [0, 11, 60], [0, 12, 100_000]])
        assert_sessions_equal(split_sessions(events, idle_timeout=1800), [[10, 11], [12]])

    def test_mapping_input(self):
        events = [
            {"user": "u", "item": "a", "timestamp": 0},
            {"user": "u", "item": "b", "timestamp": 60},
        ]
        assert_sessions_equal(split_sessions(events, idle_timeout=1800), [["a", "b"]])

    def test_pandas_dataframe(self):
        pandas = pytest.importorskip("pandas")
        frame = pandas.DataFrame(
            {
                "member": ["u1", "u1", "u1"],
                "product": ["a", "b", "c"],
                "ts": [0, 60, 100_000],
            }
        )
        assert_sessions_equal(split_sessions(frame, idle_timeout=1800), [["a", "b"], ["c"]])

        # explicit column selection also works, even for oddly named columns
        frame = pandas.DataFrame(
            {"who": ["u1", "u1"], "what": ["a", "b"], "when": [0, 60]}
        )
        sessions = split_sessions(frame, idle_timeout=1800, columns=("who", "what", "when"))
        assert_sessions_equal(sessions, [["a", "b"]])

    def test_invalid_idle_timeout(self):
        with pytest.raises(ValueError):
            split_sessions([("u", "a", 0)], idle_timeout=-1)

    def test_default_idle_timeout(self):
        assert DEFAULT_IDLE_TIMEOUT == 30 * 60


class TestSessionMatrix:
    def test_shape_and_values(self):
        events = [
            ("u1", "a", 0),
            ("u1", "b", 60),
            ("u1", "z", 100_000),
            ("u2", "a", 10),
        ]
        matrix = build_session_matrix(events, idle_timeout=1800)
        # columns in order of first appearance: a, b, z
        assert matrix.shape == (3, 3)
        dense = matrix.toarray()
        assert list(dense[0]) == [1, 1, 0]  # u1 session one: a, b
        assert list(dense[1]) == [0, 0, 1]  # u1 session two: z
        assert list(dense[2]) == [1, 0, 0]  # u2 session: a

    def test_cross_session_items_do_not_cooccur(self):
        # same member views 'a' in one session and 'b' in another session, months later
        events = [("u1", "a", 0), ("u1", "b", 60 * 60 * 24 * 90)]
        matrix = build_session_matrix(events, idle_timeout=1800, num_items=2)
        assert matrix.shape == (2, 2)
        # no row contains both items, so the columns share no session
        cooccurrence = matrix.T @ matrix
        assert cooccurrence[0, 1] == 0
        assert cooccurrence[1, 0] == 0

    def test_num_items_keeps_empty_columns(self):
        events = [("u1", 0, 0), ("u1", 2, 60)]
        matrix = build_session_matrix(events, idle_timeout=1800, num_items=5)
        assert matrix.shape == (1, 5)
        assert matrix[0, 1] == 0
        assert matrix[0, 2] == 1

    def test_integer_items_used_as_columns(self):
        events = [(0, 3, 0), (0, 1, 10)]
        matrix = build_session_matrix(events, idle_timeout=1800)
        assert matrix.shape == (1, 4)
        assert matrix[0, 3] == 1
        assert matrix[0, 1] == 1

    def test_item_ids_fixes_column_mapping(self):
        events = [("u", "c", 0), ("u", "a", 10)]
        matrix, item_ids = build_session_matrix(
            events, idle_timeout=1800, item_ids=["a", "b", "c"], return_item_ids=True
        )
        assert list(item_ids) == ["a", "b", "c"]
        assert matrix.shape == (1, 3)
        assert matrix[0, 0] == 1  # 'a'
        assert matrix[0, 1] == 0  # 'b'
        assert matrix[0, 2] == 1  # 'c'

    def test_unknown_item_with_item_ids_raises(self):
        with pytest.raises(ValueError):
            build_session_matrix([("u", "z", 0)], item_ids=["a", "b"])

    def test_duplicate_binary_vs_sum(self):
        events = [("u", "a", 0), ("u", "a", 10), ("u", "b", 20)]
        binary = build_session_matrix(events, idle_timeout=1800, duplicate="binary")
        summed = build_session_matrix(events, idle_timeout=1800, duplicate="sum")
        # columns in order of first appearance: a, b
        assert binary[0, 0] == 1
        assert summed[0, 0] == 2
        assert binary[0, 1] == summed[0, 1] == 1

    def test_min_session_items(self):
        events = [("u", "a", 0), ("u", "z", 100_000), ("u", "y", 100_060)]
        matrix = build_session_matrix(events, idle_timeout=1800, min_session_items=2)
        assert matrix.shape[0] == 1
        assert matrix.toarray()[0].sum() == 2

    def test_invalid_duplicate(self):
        with pytest.raises(ValueError):
            build_session_matrix([("u", "a", 0)], duplicate="max")

    def test_matrix_from_sessions(self):
        matrix = matrix_from_sessions([["a", "b"], ["z"], ["a", "b", "c"]])
        # columns in order of first appearance: a, b, z, c
        assert matrix.shape == (3, 4)
        dense = matrix.toarray()
        assert list(dense[0]) == [1, 1, 0, 0]
        assert list(dense[1]) == [0, 0, 1, 0]
        assert list(dense[2]) == [1, 1, 0, 1]

    def test_matrix_from_sessions_dedupes(self):
        matrix = matrix_from_sessions([["a", "a", "b"]])
        assert matrix.toarray().sum() == 2

    def test_matrix_from_sessions_min_items(self):
        matrix = matrix_from_sessions([["a"], ["a", "b"]], min_session_items=2)
        assert matrix.shape[0] == 1

    def test_empty_input(self):
        matrix = build_session_matrix([])
        assert matrix.shape == (0, 0)

    def test_cooccurrence_counts(self):
        events = [
            ("u1", "a", 0),
            ("u1", "b", 10),
            ("u1", "c", 100_000),  # different session: no co-occurrence with a/b
            ("u2", "a", 5),
            ("u2", "b", 15),
        ]
        counts = cooccurrence_counts(events, idle_timeout=1800)
        dense = counts.toarray()
        assert dense[0, 1] == 2  # a-b in two sessions
        assert dense[1, 0] == 2  # symmetric
        assert dense[0, 2] == 0  # a/c never in the same session
        assert dense[1, 2] == 0
        assert np.all(np.diag(dense) == 0)


class TestFitSessions:
    def make_events(self):
        """Builds two clearly separate product clusters.

        Members 0-3 browse items 1 and 2 together. Member 0 also browses items 3 and 4 in a
        session months later; members 5 and 6 reinforce that 3/4 cluster.
        """
        day = 24 * 60 * 60
        events = []
        timestamp = 0
        for user in range(4):
            for item in (1, 2):
                timestamp += 30
                events.append((user, item, timestamp))
        later = timestamp + 100 * day
        for user in range(4):
            for item in (3, 4):
                later += 30
                events.append((user, item, later))
        return events

    @pytest.mark.parametrize("model_class", [TFIDFRecommender, CosineRecommender, BM25Recommender])
    def test_similar_items_stay_within_sessions(self, model_class):
        events = self.make_events()
        model = model_class(K=3)
        model.fit_sessions(events, idle_timeout=1800, num_items=5, show_progress=False)

        ids, _ = model.similar_items(1, N=2)
        assert 2 in ids
        # item 3 never shared a session with item 1 (browsed months later)
        assert 3 not in ids
        assert 4 not in ids

        ids, _ = model.similar_items(3, N=2)
        assert 4 in ids
        assert 1 not in ids
        assert 2 not in ids

    def test_return_self(self):
        model = TFIDFRecommender(K=3)
        result = model.fit_sessions([(0, 0, 0), (0, 1, 1)], show_progress=False)
        assert result is model

    def test_matches_manual_session_matrix(self):
        events = self.make_events()
        model = TFIDFRecommender(K=3)
        model.fit_sessions(events, idle_timeout=1800, num_items=5, show_progress=False)

        matrix = build_session_matrix(events, idle_timeout=1800, num_items=5)
        expected = TFIDFRecommender(K=3)
        expected.fit(matrix, show_progress=False)

        np.testing.assert_allclose(model.similarity.toarray(), expected.similarity.toarray())

    def test_callback_not_supported(self):
        model = TFIDFRecommender(K=3)
        with pytest.raises(NotImplementedError):
            model.fit_sessions([(0, 0, 0)], show_progress=False, callback=lambda *a: None)

    def test_module_exposed(self):
        assert implicit.sessions.build_session_matrix is build_session_matrix

    def test_model_session_helpers(self):
        events = [("u1", "a", 0), ("u1", "b", 60), ("u1", "c", 100_000)]
        model = TFIDFRecommender(K=2)

        assert_sessions_equal(model.split_sessions(events, idle_timeout=1800), [["a", "b"], ["c"]])

        matrix = model.session_matrix(events, idle_timeout=1800)
        assert matrix.shape == (2, 3)

        cooccurrence = model.session_cooccurrence(events, idle_timeout=1800)
        assert cooccurrence[0, 1] == 1
        assert cooccurrence[0, 2] == 0


class TestSessionizer:
    def test_builds_and_fits(self):
        events = [("u1", "a", 0), ("u1", "b", 60), ("u1", "c", 100_000)]
        sessionizer = Sessionizer(idle_timeout=1800)
        assert_sessions_equal(sessionizer.split_sessions(events), [["a", "b"], ["c"]])
        matrix = sessionizer.build_matrix(events)
        assert matrix.shape == (2, 3)

        model = CosineRecommender(K=2)
        result = sessionizer.fit(model, events, show_progress=False)
        assert result is model
        assert model.similarity is not None

    def test_cooccurrence(self):
        events = [("u1", "a", 0), ("u1", "b", 60)]
        counts = Sessionizer(idle_timeout=1800).cooccurrence(events)
        assert counts[0, 1] == 1

    def test_invalid_idle_timeout(self):
        with pytest.raises(ValueError):
            Sessionizer(idle_timeout=-5)
