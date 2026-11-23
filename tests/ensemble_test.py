import unittest

import numpy as np
import pytest
from recommender_base_test import get_checker_board

from implicit.als import AlternatingLeastSquares
from implicit.ensemble import CandidateMerger, RankCalibrator


class RankCalibratorTest(unittest.TestCase):
    def test_quantile_maps_to_unit_interval(self):
        scores = np.linspace(-10, 100, 501)
        calibrated = RankCalibrator("quantile").fit_transform(scores)
        self.assertTrue(np.all(calibrated > 0))
        self.assertTrue(np.all(calibrated <= 1))
        # monotone: preserves the ranking of the scores
        self.assertTrue(np.all(np.diff(calibrated) >= 0))

    def test_quantile_makes_scales_comparable(self):
        # two models with wildly different raw score scales
        scores_a = np.linspace(0, 1, 1000)
        scores_b = np.linspace(1000, 5000, 1000)
        cal_a = RankCalibrator().fit(scores_a)
        cal_b = RankCalibrator().fit(scores_b)
        # the median request of each model lands at roughly the same
        # calibrated score, despite the raw scales being 1000x apart
        self.assertAlmostEqual(cal_a.transform([0.5])[0], 0.5, places=2)
        self.assertAlmostEqual(cal_b.transform([3000])[0], 0.5, places=2)

    def test_zscore_normalizes(self):
        rng = np.random.RandomState(42)
        scores = rng.normal(loc=100, scale=7, size=5000)
        calibrator = RankCalibrator("zscore").fit(scores)
        transformed = calibrator.transform(scores)
        self.assertAlmostEqual(transformed.mean(), 0, places=2)
        self.assertAlmostEqual(transformed.std(), 1, places=2)

    def test_zscore_constant_scores(self):
        calibrator = RankCalibrator("zscore").fit([3.0, 3.0, 3.0])
        assert_array_almost_equal(calibrator.transform([3.0, 5.0]), [0.0, 2.0])

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            RankCalibrator("platt")

    def test_fit_empty(self):
        with pytest.raises(ValueError):
            RankCalibrator().fit([])

    def test_transform_before_fit(self):
        with pytest.raises(ValueError):
            RankCalibrator().transform([1.0])


def assert_array_almost_equal(a, b):
    np.testing.assert_array_almost_equal(np.asarray(a), np.asarray(b))


def _make_merger(**kwargs):
    """Two registered models: 'a' scores items 0..3, 'b' scores items 0..3,
    each mapping local ids to business ids with a partial overlap."""
    merger = CandidateMerger(**kwargs)
    merger.add_model("a", np.array(["item-0", "item-1", "item-2", "item-3"]),
                     validation_scores=[0.0, 1.0, 2.0, 3.0])
    merger.add_model("b", np.array(["item-2", "item-3", "item-4", "item-5"]),
                     validation_scores=[100.0, 200.0, 300.0, 400.0])
    return merger


class CandidateMergerTest(unittest.TestCase):
    def test_merge_combines_shared_items(self):
        merger = _make_merger()
        # both models recommend the shared business item 'item-2',
        # via different local ids (2 for 'a', 0 for 'b')
        ids, scores, sources = merger.merge(
            {"a": ([2], [3.0]), "b": ([0], [400.0])}
        )
        self.assertEqual(list(ids), ["item-2"])
        self.assertEqual(sources, [("a", "b")])
        # both raw scores are the max of their validation range -> quantile 1.0
        self.assertAlmostEqual(scores[0], 2.0)

    def test_merge_keeps_single_model_items(self):
        merger = _make_merger()
        ids, scores, sources = merger.merge(
            {"a": ([0, 1], [3.0, 2.0]), "b": ([3], [400.0])}
        )
        self.assertEqual(set(ids), {"item-0", "item-1", "item-5"})
        by_id = dict(zip(ids.tolist(), sources))
        self.assertEqual(by_id["item-0"], ("a",))
        self.assertEqual(by_id["item-1"], ("a",))
        self.assertEqual(by_id["item-5"], ("b",))

    def test_local_id_collision_not_confused(self):
        merger = _make_merger()
        # the same local id 1 means different business items per model
        ids, _, sources = merger.merge({"a": ([1], [3.0]), "b": ([1], [400.0])})
        self.assertEqual(set(ids), {"item-1", "item-3"})
        by_id = dict(zip(ids.tolist(), sources))
        self.assertEqual(by_id["item-1"], ("a",))
        self.assertEqual(by_id["item-3"], ("b",))

    def test_scores_sorted_descending(self):
        merger = _make_merger()
        ids, scores, _ = merger.merge(
            {"a": ([0, 1], [1.0, 2.0]), "b": ([2, 3], [100.0, 400.0])}
        )
        self.assertTrue(np.all(np.diff(scores) <= 0))
        # 'item-5' is top: recommended by 'b' with its max validation score
        self.assertEqual(ids[0], "item-5")

    def test_merge_top_n(self):
        merger = _make_merger()
        ids, scores, sources = merger.merge(
            {"a": ([0, 1, 2], [3.0, 2.0, 1.0]), "b": ([2, 3], [100.0, 400.0])}, N=2
        )
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(sources), 2)

    def test_combine_mean_and_max(self):
        merger = _make_merger(combine="mean")
        _, scores, _ = merger.merge({"a": ([2], [3.0]), "b": ([0], [200.0])})
        self.assertAlmostEqual(scores[0], (1.0 + 0.5) / 2)

        merger = _make_merger(combine="max")
        _, scores, _ = merger.merge({"a": ([2], [3.0]), "b": ([0], [200.0])})
        self.assertAlmostEqual(scores[0], 1.0)

    def test_merge_weights(self):
        merger = _make_merger()
        ids, scores, _ = merger.merge(
            {"a": ([0], [3.0]), "b": ([3], [400.0])}, weights={"b": 3.0}
        )
        by_id = dict(zip(ids.tolist(), scores.tolist()))
        self.assertAlmostEqual(by_id["item-0"], 1.0)
        self.assertAlmostEqual(by_id["item-5"], 3.0)

    def test_unknown_local_id_dropped(self):
        merger = _make_merger()
        ids, _, sources = merger.merge({"a": ([0, 99], [3.0, 1.0]), "b": ([], [])})
        self.assertEqual(list(ids), ["item-0"])
        self.assertEqual(sources, [("a",)])

    def test_unknown_local_id_raises(self):
        merger = _make_merger(on_unknown="raise")
        with pytest.raises(KeyError):
            merger.merge({"a": ([99], [1.0]), "b": ([], [])})

    def test_dict_mapping(self):
        merger = CandidateMerger()
        merger.add_model("a", {0: "item-0", 1: "item-1"}, validation_scores=[0.0, 1.0])
        merger.add_model("b", {1: "item-1"}, validation_scores=[0.0, 1.0])
        ids, _, sources = merger.merge({"a": ([0, 1, 2], [1.0, 1.0, 1.0]), "b": ([1], [1.0])})
        by_id = dict(zip(ids.tolist(), sources))
        # local id 2 of model 'a' is unknown and dropped; 'item-1' is shared
        self.assertEqual(set(ids), {"item-0", "item-1"})
        self.assertEqual(by_id["item-1"], ("a", "b"))

    def test_unregistered_model(self):
        merger = _make_merger()
        with pytest.raises(KeyError):
            merger.merge({"c": ([0], [1.0])})

    def test_merge_batch(self):
        merger = _make_merger()
        results = merger.merge_batch(
            {
                "a": (np.array([[2, 3], [0, 1]]), np.array([[3.0, 2.0], [3.0, 2.0]])),
                "b": (np.array([[0, 1], [2, 3]]), np.array([[400.0, 300.0], [200.0, 100.0]])),
            }
        )
        self.assertEqual(len(results), 2)
        # first request: both models recommend the shared items 'item-2'/'item-3'
        ids0, _, sources0 = results[0]
        self.assertEqual(set(ids0), {"item-2", "item-3"})
        self.assertEqual(sources0, [("a", "b"), ("a", "b")])
        # second request: four distinct business items, each from a single model
        ids1, _, sources1 = results[1]
        self.assertEqual(set(ids1), {"item-0", "item-1", "item-4", "item-5"})
        self.assertTrue(all(len(s) == 1 for s in sources1))

    def test_invalid_options(self):
        with pytest.raises(ValueError):
            CandidateMerger(combine="product")
        with pytest.raises(ValueError):
            CandidateMerger(on_unknown="ignore")


@pytest.mark.parametrize("method", ["quantile", "zscore"])
def test_end_to_end_regional_models(method):
    """Two regional ALS models with partially overlapping inventories: calibrate
    on shared validation requests, then merge candidates by business item id."""
    user_items_a = get_checker_board(8)
    user_items_b = get_checker_board(8)

    model_a = AlternatingLeastSquares(factors=4, regularization=0, use_gpu=False, random_state=23)
    model_b = AlternatingLeastSquares(factors=4, regularization=0, use_gpu=False, random_state=23)
    model_a.fit(user_items_a, show_progress=False)
    model_b.fit(user_items_b, show_progress=False)

    # region b's local catalog is shifted by 4 business items: local id 0 of
    # region b is business 'item-4', so the same local integer ids mean
    # different items in the two regions
    mapping_a = np.array([f"item-{i}" for i in range(8)])
    mapping_b = np.array([f"item-{i + 4}" for i in range(8)])

    merger = CandidateMerger(method=method)
    merger.add_model("region_a", mapping_a)
    merger.add_model("region_b", mapping_b)

    # learn comparable scales from a shared set of historical validation requests
    validation_users = np.arange(8)
    merger.fit_from_requests("region_a", model_a, user_items_a, validation_users, N=8)
    merger.fit_from_requests("region_b", model_b, user_items_b, validation_users, N=8)

    # merge the candidates of a cross-region account
    recs_a = model_a.recommend(0, user_items_a[0], N=4)
    recs_b = model_b.recommend(0, user_items_b[0], N=4)
    ids, scores, sources = merger.merge(
        {"region_a": recs_a, "region_b": recs_b}, N=6
    )

    # output is keyed on business item ids, with provenance per candidate
    assert len(ids) == 6
    assert all(i.startswith("item-") for i in ids)
    assert len(set(ids)) == len(ids)
    assert all(1 <= len(s) <= 2 for s in sources)
    assert np.all(np.diff(scores) <= 0)

    # every recommended business item comes from a region whose catalog
    # actually contains it
    catalog_a = set(mapping_a)
    catalog_b = set(mapping_b)
    for item, source in zip(ids, sources):
        if "region_a" in source:
            assert item in catalog_a
        if "region_b" in source:
            assert item in catalog_b


if __name__ == "__main__":
    unittest.main()
