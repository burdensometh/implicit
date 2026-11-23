import unittest

import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from implicit.session import (
    SessionRecommendation,
    item_popularity,
    recommend_for_session,
)


def _build_model(n_users=30, n_items=12, factors=8):
    rng = np.random.default_rng(42)
    # structured preferences: half the users favor even items, half odd ones
    data = np.zeros((n_users, n_items), dtype=np.float32)
    for u in range(n_users):
        offset = u % 2
        data[u, offset::2] = rng.poisson(2, size=len(range(offset, n_items, 2)))
    user_items = csr_matrix(data)
    model = AlternatingLeastSquares(factors=factors, random_state=23)
    model.fit(user_items, show_progress=False)
    return model, user_items


class SessionRecommendTest(unittest.TestCase):
    def setUp(self):
        self.model, self.user_items = _build_model()
        self.n_items = self.model.item_factors.shape[0]
        self.popularity = item_popularity(self.user_items)

    # regime: inferred --------------------------------------------------------

    def test_inferred_regime_returns_temporary_factor_and_does_not_mutate_model(self):
        user_factors_before = self.model.user_factors.copy()
        item_factors_before = self.model.item_factors.copy()

        result = recommend_for_session(
            self.model, session_items=[0, 2, 4], popularity=self.popularity, N=5
        )

        self.assertEqual(result.regime, SessionRecommendation.REGIME_INFERRED)
        self.assertIsNotNone(result.user_factors)
        # the inferred vector matches a direct read-only recalculate_user call
        row = csr_matrix(
            (np.ones(3, dtype=np.float32), np.array([0, 2, 4]), np.array([0, 3])),
            shape=(1, self.n_items),
        )
        expected_factor = self.model.recalculate_user(0, row)
        np.testing.assert_allclose(result.user_factors, expected_factor)

        # the persisted model is untouched
        np.testing.assert_array_equal(self.model.user_factors, user_factors_before)
        np.testing.assert_array_equal(self.model.item_factors, item_factors_before)

        # scores are the dot products of the temporary factor with item factors
        expected_scores = self.model.item_factors.dot(result.user_factors)[result.ids]
        np.testing.assert_allclose(result.scores, expected_scores)
        self.assertEqual(len(result.ids), 5)
        # results are sorted by descending score
        self.assertTrue(np.all(np.diff(result.scores) <= 0))

    def test_inferred_session_excludes_known_feedback_items(self):
        session = [1, 3, 5]
        result = recommend_for_session(
            self.model, session_items=session, popularity=self.popularity, N=10
        )
        self.assertEqual(result.regime, SessionRecommendation.REGIME_INFERRED)
        self.assertEqual(set(result.ids).intersection(session), set())

    def test_inferred_with_csr_session_row_and_weights(self):
        row = csr_matrix(
            (np.array([2.0, 3.0]), np.array([0, 6]), np.array([0, 2])),
            shape=(1, self.n_items),
        )
        result = recommend_for_session(
            self.model, session_items=row, popularity=self.popularity, N=4
        )
        self.assertEqual(result.regime, SessionRecommendation.REGIME_INFERRED)
        self.assertEqual(set(result.ids).intersection({0, 6}), set())
        np.testing.assert_array_equal(result.unmodeled_items, np.array([], dtype=np.int64))

    # regime: no preference ---------------------------------------------------

    def test_empty_session_is_no_preference_popularity_fallback(self):
        result = recommend_for_session(
            self.model, session_items=[], popularity=self.popularity, N=5
        )
        self.assertEqual(result.regime, SessionRecommendation.REGIME_NO_PREFERENCE)
        self.assertIsNone(result.user_factors)

        expected_all = np.arange(self.n_items)
        expected = expected_all[np.lexsort((expected_all, -self.popularity))][:5]
        assert_array_equal(result.ids, expected)
        np.testing.assert_allclose(result.scores, self.popularity[expected])

    def test_zero_weight_session_is_no_preference(self):
        result = recommend_for_session(
            self.model,
            session_items=[1, 2],
            session_weights=[0.0, 0.0],
            popularity=self.popularity,
            N=3,
        )
        self.assertEqual(result.regime, SessionRecommendation.REGIME_NO_PREFERENCE)

    # regime: unmodeled only --------------------------------------------------

    def test_all_unmodeled_items_uses_popularity_fallback(self):
        beyond = [self.n_items, self.n_items + 5]
        result = recommend_for_session(
            self.model, session_items=beyond, popularity=self.popularity, N=4
        )
        self.assertEqual(result.regime, SessionRecommendation.REGIME_UNMODELED_ONLY)
        self.assertIsNone(result.user_factors)
        assert_array_equal(result.unmodeled_items, np.array(beyond))

        # fallback uses the popularity order over the eligible candidates, and
        # the unmodeled feedback items (outside the modelled catalogue, so not
        # present in candidates) cannot leak into the result. Ties are broken by
        # ascending item id, matching the implementation's deterministic order.
        eligible = np.setdiff1d(np.arange(self.n_items), beyond)
        order = np.lexsort((eligible, -self.popularity[eligible]))
        expected = eligible[order][:4]
        assert_array_equal(result.ids, expected)
        self.assertEqual(set(result.ids).intersection(beyond), set())

    def test_mixed_session_infers_from_modeled_only_and_tracks_unmodeled(self):
        result = recommend_for_session(
            self.model,
            session_items=[2, self.n_items + 1, 4],
            popularity=self.popularity,
            N=6,
        )
        self.assertEqual(result.regime, SessionRecommendation.REGIME_INFERRED)
        assert_array_equal(result.unmodeled_items, np.array([self.n_items + 1]))
        # the modeled feedback items are still excluded
        self.assertEqual(set(result.ids).intersection({2, 4}), set())

    def test_wide_csr_row_with_unmodeled_columns(self):
        row = csr_matrix(
            (np.ones(2, dtype=np.float32), np.array([0, self.n_items + 2]), np.array([0, 2])),
            shape=(1, self.n_items + 5),
        )
        result = recommend_for_session(
            self.model, session_items=row, popularity=self.popularity, N=5
        )
        self.assertEqual(result.regime, SessionRecommendation.REGIME_INFERRED)
        assert_array_equal(result.unmodeled_items, np.array([self.n_items + 2]))
        self.assertNotIn(0, set(result.ids))

    # filtering and candidate restrictions -----------------------------------

    def test_filter_items_excludes_extra_ids_in_fallback(self):
        top = np.argsort(-self.popularity, kind="stable")[:3]
        result = recommend_for_session(
            self.model,
            session_items=[],
            popularity=self.popularity,
            N=3,
            filter_items=[top[0]],
        )
        self.assertNotIn(top[0], set(result.ids))
        self.assertEqual(len(result.ids), 3)

    def test_popularity_tie_at_nth_position_uses_id_order(self):
        # id 4 is the clear winner; ids 0-3 tie below it and straddle the Nth
        # cut, so the global id-ascending tie-break must still apply
        popularity = np.array([3.0, 3.0, 3.0, 3.0, 5.0])
        result = recommend_for_session(
            self.model,
            session_items=[self.n_items],  # unmodeled -> popularity fallback
            popularity=np.pad(popularity, (0, self.n_items - 5)),
            N=3,
        )
        assert_array_equal(result.ids, np.array([4, 0, 1]))

    def test_items_restricts_candidate_set(self):
        candidates = np.array([2, 4, 6])
        result = recommend_for_session(
            self.model,
            session_items=[0],
            popularity=self.popularity,
            N=3,
            items=candidates,
        )
        self.assertTrue(set(result.ids).issubset(set(candidates)))

    def test_items_and_filter_items_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            recommend_for_session(
                self.model,
                session_items=[0],
                popularity=self.popularity,
                items=[1, 2],
                filter_items=[3],
            )

    def test_popularity_length_must_match_catalogue(self):
        with self.assertRaises(ValueError):
            recommend_for_session(
                self.model,
                session_items=[0],
                popularity=np.zeros(self.n_items - 1),
            )

    def test_unknown_item_in_items_param_raises(self):
        with self.assertRaises(IndexError):
            recommend_for_session(
                self.model,
                session_items=[0],
                popularity=self.popularity,
                items=[self.n_items + 1],
            )

    # unsupported model -------------------------------------------------------

    def test_item_item_model_raises_for_modeled_feedback(self):
        knn = CosineRecommender(K=5)
        knn.fit(self.user_items, show_progress=False)
        self.assertEqual(knn.similarity.shape[0], self.n_items)
        with self.assertRaises(NotImplementedError):
            recommend_for_session(
                knn, session_items=[0, 1], popularity=self.popularity
            )

    def test_item_item_model_still_falls_back_without_modeled_feedback(self):
        knn = CosineRecommender(K=5)
        knn.fit(self.user_items, show_progress=False)
        result = recommend_for_session(
            knn, session_items=[], popularity=self.popularity, N=3
        )
        self.assertEqual(result.regime, SessionRecommendation.REGIME_NO_PREFERENCE)

    # helpers -----------------------------------------------------------------

    def test_item_popularity_matches_column_sums(self):
        popularity = item_popularity(self.user_items)
        np.testing.assert_allclose(
            popularity, np.asarray(self.user_items.sum(axis=0)).reshape(-1)
        )


if __name__ == "__main__":
    unittest.main()
