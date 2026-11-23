import unittest

import numpy as np
from recommender_base_test import RecommenderBaseTestMixin, get_checker_board

from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    FaissAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.gpu import HAS_CUDA


class ApproximateRecommendFilterTestMixin:
    """Tests that the approximate recommend calls can still fill N results when the
    user's top ANN candidates are all filtered out (already liked items / filter_items)"""

    def _get_model(self):
        raise NotImplementedError()

    def test_recommend_filters_long_history(self):
        # user 0 has liked every even item except item 0 - so after filtering the first
        # page of ANN results are all items that have to be skipped over
        plays = get_checker_board(200)
        user_items = plays.T.tocsr()

        model = self._get_model()
        model.fit(plays, show_progress=False)

        liked = set(user_items[0].indices.tolist())
        for N in (10, 50, 99):
            ids, scores = model.recommend(0, user_items[0], N=N)

            self.assertEqual(len(ids), N)
            self.assertEqual(len(set(ids.tolist())), N)
            self.assertFalse(liked & set(ids.tolist()))
            self.assertTrue(np.all(np.diff(scores) <= 1e-6))

    def test_recommend_filters_with_filter_items(self):
        plays = get_checker_board(200)
        user_items = plays.T.tocsr()

        model = self._get_model()
        model.fit(plays, show_progress=False)

        filter_items = list(range(100))
        ids, scores = model.recommend(0, user_items[0], N=10, filter_items=filter_items)
        self.assertEqual(len(ids), 10)
        self.assertFalse(set(filter_items) & set(ids.tolist()))

    def test_recommend_batch_matches_single(self):
        plays = get_checker_board(200)
        user_items = plays.T.tocsr()

        model = self._get_model()
        model.fit(plays, show_progress=False)

        userids = np.arange(10)
        batch_ids, batch_scores = model.recommend(userids, user_items[userids], N=5)
        for i, userid in enumerate(userids):
            ids, scores = model.recommend(userid, user_items[userid], N=5)
            np.testing.assert_array_equal(ids, batch_ids[i])
            np.testing.assert_allclose(scores, batch_scores[i])


# don't require annoy/faiss/nmslib to be installed
try:
    import annoy  # noqa pylint: disable=unused-import

    class AnnoyALSTest(
        unittest.TestCase, RecommenderBaseTestMixin, ApproximateRecommendFilterTestMixin
    ):
        def _get_model(self):
            return AnnoyAlternatingLeastSquares(
                factors=32, regularization=0, random_state=23, use_gpu=False
            )

        def test_pickle(self):
            # pickle isn't supported on annoy indices
            pass

        def test_serialization(self):
            pass

        def test_serialization_without_fit(self):
            pass

    if HAS_CUDA:

        class AnnoyALSGPUTest(unittest.TestCase, RecommenderBaseTestMixin):
            def _get_model(self):
                return AnnoyAlternatingLeastSquares(
                    factors=32, regularization=0, random_state=23, use_gpu=True
                )

            def test_pickle(self):
                # pickle isn't supported on annoy indices
                pass

            def test_serialization(self):
                pass

            def test_serialization_without_fit(self):
                pass

except ImportError:
    pass

try:
    import nmslib  # noqa pylint: disable=unused-import

    class NMSLibALSTest(
        unittest.TestCase, RecommenderBaseTestMixin, ApproximateRecommendFilterTestMixin
    ):
        def _get_model(self):
            return NMSLibAlternatingLeastSquares(
                factors=32,
                regularization=0,
                index_params={"post": 2},
                random_state=23,
                use_gpu=False,
            )

        def test_pickle(self):
            # pickle isn't supported on nmslib indices
            pass

        def test_serialization(self):
            pass

        def test_serialization_without_fit(self):
            pass

    if HAS_CUDA:
        # nmslib doesn't support querying on the gpu, but we should be able to still use a GPU als
        # model with the nmslib index
        class NMSLibALSGPUTest(unittest.TestCase, RecommenderBaseTestMixin):
            def _get_model(self):
                return NMSLibAlternatingLeastSquares(
                    factors=32,
                    regularization=0,
                    index_params={"post": 2},
                    random_state=23,
                    use_gpu=True,
                )

            def test_pickle(self):
                # pickle isn't supported on nmslib indices
                pass

            def test_serialization(self):
                pass

            def test_serialization_without_fit(self):
                pass

except ImportError:
    pass

try:
    import faiss  # noqa pylint: disable=unused-import

    class FaissALSTest(
        unittest.TestCase, RecommenderBaseTestMixin, ApproximateRecommendFilterTestMixin
    ):
        def _get_model(self):
            return FaissAlternatingLeastSquares(
                nlist=1, nprobe=1, factors=32, regularization=0, use_gpu=False, random_state=23
            )

        def test_pickle(self):
            # pickle isn't supported on faiss indices
            pass

        def test_serialization(self):
            pass

        def test_serialization_without_fit(self):
            pass

    if HAS_CUDA:

        class FaissALSGPUTest(unittest.TestCase, RecommenderBaseTestMixin):
            __regularization = 0

            def _get_model(self):
                return FaissAlternatingLeastSquares(
                    nlist=1,
                    nprobe=1,
                    factors=32,
                    regularization=self.__regularization,
                    use_gpu=True,
                    random_state=23,
                )

            def test_large_recommend(self):
                # the GPU version of FAISS can't return more than 1K result (and will assert/exit)
                # this tests out that we fall back in this case to the exact version and don't die
                plays = get_checker_board(2048)
                model = self._get_model()
                model.fit(plays, show_progress=False)

                recs = model.similar_items(0, N=1050)
                self.assertEqual(recs[0][0], 0)

                recs = model.recommend(0, plays.T.tocsr()[0], N=1050)
                self.assertEqual(recs[0][0], 0)

            def test_pickle(self):
                # pickle isn't supported on faiss indices
                pass

            def test_serialization(self):
                pass

            def test_serialization_without_fit(self):
                pass

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
