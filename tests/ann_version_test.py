import threading
import time
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from implicit.approximate_als import (
    AnnoyAlternatingLeastSquares,
    FaissAlternatingLeastSquares,
    NMSLibAlternatingLeastSquares,
)
from implicit.cpu.als import AlternatingLeastSquares as CPUAlternatingLeastSquares

# the version bookkeeping lives in the shared ANNModelBase, so every optional backend
# that is installed gets exercised against the same set of scenarios
try:
    import annoy  # noqa pylint: disable=unused-import

    HAS_ANNOY = True
except ImportError:
    HAS_ANNOY = False

try:
    import nmslib  # noqa pylint: disable=unused-import

    HAS_NMSLIB = True
except ImportError:
    HAS_NMSLIB = False

try:
    import faiss  # noqa pylint: disable=unused-import

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def _get_data(users=30, items=40, seed=42, density=0.2):
    rng = np.random.RandomState(seed)
    data = rng.random((users, items))
    data[data < (1 - density)] = 0
    return csr_matrix(data.astype(np.float32))


class ANNVersionTestMixin:
    """Backend agnostic checks for factor/index version binding."""

    backend_name = None

    def _get_model(self, **kwargs):
        raise NotImplementedError

    def test_fit_publishes_bound_version(self):
        model = self._get_model()
        user_items = _get_data()
        model.fit(user_items, show_progress=False)

        version = model.active_version
        self.assertEqual(version.version, 1)
        self.assertIsNotNone(version.similar_items_index)
        self.assertIsNotNone(version.recommend_index)

        # the snapshot bound to the indices really is the trained model factors
        np.testing.assert_array_equal(version.item_factors, model.model.item_factors)
        np.testing.assert_array_equal(version.user_factors, model.model.user_factors)

        # the convenience attributes resolve to the active version
        self.assertIs(model.similar_items_index, version.similar_items_index)
        self.assertIs(model.recommend_index, version.recommend_index)
        self.assertEqual(model.version, 1)

    def test_snapshot_is_isolated_from_later_model_mutation(self):
        model = self._get_model()
        model.fit(_get_data(), show_progress=False)

        frozen = model.active_version.item_factors.copy()
        # mutate the underlying model array in place, like partial_fit_items would do
        model.model.item_factors[0] += 1.0

        np.testing.assert_array_equal(model.active_version.item_factors, frozen)

    def test_partial_fit_swaps_version_atomically(self):
        model = self._get_model()
        user_items = _get_data()
        model.fit(user_items, show_progress=False)

        old_version = model.active_version

        # change the sales (interactions) of an existing item and retrain locally
        itemid = 5
        item_users = user_items.T.tocsr()[itemid]
        model.partial_fit_items([itemid], item_users, show_progress=False)

        # a brand new, fully built version is published; the old one is untouched
        self.assertIsNot(model.active_version, old_version)
        self.assertEqual(model.active_version.version, 2)
        self.assertEqual(old_version.version, 1)
        self.assertIsNot(model.active_version.similar_items_index, old_version.similar_items_index)

        # new index is bound to the new factors, not the old ones
        np.testing.assert_array_equal(model.active_version.item_factors, model.model.item_factors)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(
                model.active_version.item_factors, old_version.item_factors
            )

        # serving still works against the new version
        ids, scores = model.recommend(0, user_items.T.tocsr()[0], N=5)
        self.assertEqual(len(ids), 5)
        ids, scores = model.similar_items(0, N=5)
        self.assertEqual(len(ids), 5)

    def test_failed_rebuild_keeps_previous_consistent_view(self):
        model = self._get_model()
        user_items = _get_data()
        model.fit(user_items, show_progress=False)
        old_version = model.active_version
        old_factors = old_version.item_factors.copy()

        def boom(item_factors, show_progress):
            raise RuntimeError("index build failed")

        model._build_recommend_index = boom  # pylint: disable=protected-access

        itemid = 7
        with self.assertRaises(RuntimeError):
            model.partial_fit_items([itemid], user_items.T.tocsr()[itemid], show_progress=False)

        # the model array already changed, but the published view is still the old one ...
        self.assertIs(model.active_version, old_version)
        self.assertEqual(model.version, 1)
        np.testing.assert_array_equal(model.active_version.item_factors, old_factors)
        self.assertIs(model.recommend_index, old_version.recommend_index)

        # ... and it stays fully usable: index and factors are still the same old pair
        ids, scores = model.recommend(0, user_items.T.tocsr()[0], N=5)
        self.assertEqual(len(ids), 5)
        ids, scores = model.similar_items(0, N=5)
        self.assertEqual(len(ids), 5)

        # a later successful rebuild publishes a new version
        del model._build_recommend_index
        model.rebuild_index(show_progress=False)
        self.assertEqual(model.version, 2)
        np.testing.assert_array_equal(
            model.active_version.item_factors, model.model.item_factors
        )

    def test_failed_rebuild_does_not_reuse_version_id(self):
        model = self._get_model()
        model.fit(_get_data(), show_progress=False)
        self.assertEqual(model.version, 1)

        def boom(item_factors, show_progress):
            raise RuntimeError("nope")

        model._build_similar_items_index = boom  # pylint: disable=protected-access
        with self.assertRaises(RuntimeError):
            model.rebuild_index(show_progress=False)
        self.assertEqual(model.version, 1)

        del model._build_similar_items_index
        model.rebuild_index(show_progress=False)
        # failed attempt must not have consumed the version id
        self.assertEqual(model.version, 2)

    def test_new_item_first_recommendable_time_is_queryable(self):
        model = self._get_model()
        user_items = _get_data(users=30, items=40)
        model.fit(user_items, show_progress=False)

        new_itemid = user_items.shape[1] + 3  # a brand new item id
        self.assertIsNone(model.item_recommendable_at(new_itemid))
        self.assertFalse(model.active_version.is_item_recommendable(new_itemid))

        # give the new item some interactions and train it locally
        n_users = user_items.shape[0]
        item_users = csr_matrix(
            (np.ones(5, dtype=np.float32), (np.zeros(5, dtype=int), np.arange(5))),
            shape=(1, n_users),
        )

        start = time.time()
        model.partial_fit_items([new_itemid], item_users, show_progress=False)
        end = time.time()

        available_at = model.item_recommendable_at(new_itemid)
        self.assertIsNotNone(available_at)
        self.assertGreaterEqual(available_at, start - 1e-6)
        self.assertLessEqual(available_at, end + 1e-6)
        self.assertTrue(model.active_version.is_item_recommendable(new_itemid))

        # the new item now lives in an index built from its own factors
        total_items = model.item_factors.shape[0]
        ids, _ = model.recommend(
            0,
            user_items.T.tocsr()[0],
            N=total_items,
            filter_already_liked_items=False,
        )
        self.assertIn(new_itemid, set(ids.tolist()))
        if self.backend_name != "nmslib":
            # nmslib skips zero norm rows in its cosine index, only assert the direct
            # lookup for backends that index every row
            sim_ids, _ = model.similar_items(new_itemid, N=5)
            self.assertIn(new_itemid, set(sim_ids.tolist()))

        # history is carried forward into later versions
        model.rebuild_index(show_progress=False)
        self.assertEqual(model.item_recommendable_at(new_itemid), available_at)

    def test_zero_factor_item_is_not_recommendable(self):
        # item 3 has no interactions at all -> zero factor -> must not be marked
        user_items = _get_data()
        user_items = user_items.tolil()
        user_items[:, 3] = 0
        user_items = user_items.tocsr()

        model = self._get_model()
        model.fit(user_items, show_progress=False)

        self.assertIsNone(model.item_recommendable_at(3))
        self.assertFalse(model.active_version.is_item_recommendable(3))

        # an item with interactions is still tracked
        self.assertIsNotNone(model.item_recommendable_at(0))

    def test_rebuild_index_picks_up_manual_model_changes(self):
        model = self._get_model()
        model.fit(_get_data(), show_progress=False)
        self.assertEqual(model.version, 1)

        published = model.rebuild_index(show_progress=False)
        self.assertEqual(published, 2)
        self.assertEqual(model.active_version.version, 2)

    def test_full_fit_resets_availability_history_but_keeps_version_counter(self):
        model = self._get_model()
        user_items = _get_data(users=30, items=40)
        model.fit(user_items, show_progress=False)

        n_users = user_items.shape[0]
        new_itemid = user_items.shape[1] - 1
        # clear this item first so we can observe it becoming recommendable on its own
        user_items = user_items.tolil()
        user_items[:, new_itemid] = 0
        user_items = user_items.tocsr()
        model.fit(user_items, show_progress=False)
        self.assertIsNone(model.item_recommendable_at(new_itemid))

        item_users = csr_matrix(
            (np.ones(5, dtype=np.float32), (np.zeros(5, dtype=int), np.arange(5))),
            shape=(1, n_users),
        )
        model.partial_fit_items([new_itemid], item_users, show_progress=False)
        self.assertIsNotNone(model.item_recommendable_at(new_itemid))
        self.assertEqual(model.version, 3)

        # a full refit repopulates the history from the factors it actually trained, so
        # the timestamp is rebuilt rather than leaked -- while version ids keep increasing
        model.fit(user_items, show_progress=False)
        self.assertEqual(model.version, 4)
        # the item has no interactions in this catalogue, so it stays un-recommendable
        self.assertIsNone(model.item_recommendable_at(new_itemid))
        # while an item with interactions is tracked afresh
        self.assertIsNotNone(model.item_recommendable_at(0))

    def test_queries_never_break_during_repeated_rebuilds(self):
        model = self._get_model()
        user_items = _get_data()
        model.fit(user_items, show_progress=False)

        errors = []

        def reader():
            for _ in range(50):
                try:
                    model.recommend(0, user_items.T.tocsr()[0], N=5)
                    model.similar_items(0, N=5)
                except Exception as err:  # pragma: no cover - failure path
                    errors.append(err)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()

        for itemid in range(0, 20, 4):
            model.partial_fit_items([itemid], user_items.T.tocsr()[itemid], show_progress=False)

        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])

    def test_underlying_model_is_still_plain_als(self):
        # the wrapper must not change the underlying model type / training behaviour
        model = self._get_model()
        self.assertIsInstance(model.model, CPUAlternatingLeastSquares)
        model.fit(_get_data(), show_progress=False)
        self.assertIsNotNone(model.model.item_factors)

    def test_accessing_version_before_fit_raises(self):
        model = self._get_model()
        with self.assertRaises(ValueError):
            _ = model.active_version
        with self.assertRaises(ValueError):
            _ = model.version


@unittest.skipUnless(HAS_ANNOY, "annoy isn't installed")
class AnnoyANNVersionTest(ANNVersionTestMixin, unittest.TestCase):
    backend_name = "annoy"

    def _get_model(self, **kwargs):
        defaults = dict(factors=16, iterations=5, random_state=23, use_gpu=False)
        defaults.update(kwargs)
        return AnnoyAlternatingLeastSquares(**defaults)


@unittest.skipUnless(HAS_NMSLIB, "nmslib isn't installed")
class NMSLibANNVersionTest(ANNVersionTestMixin, unittest.TestCase):
    backend_name = "nmslib"

    def _get_model(self, **kwargs):
        defaults = dict(
            factors=16,
            iterations=5,
            random_state=23,
            use_gpu=False,
            index_params={"M": 16, "post": 0, "efConstruction": 100},
        )
        defaults.update(kwargs)
        return NMSLibAlternatingLeastSquares(**defaults)


@unittest.skipUnless(HAS_FAISS, "faiss isn't installed")
class FaissANNVersionTest(ANNVersionTestMixin, unittest.TestCase):
    backend_name = "faiss"

    def _get_model(self, **kwargs):
        # the test catalogues are small, so a single clustering cell is enough
        defaults = dict(
            factors=16, iterations=5, random_state=23, use_gpu=False, nlist=1, nprobe=1
        )
        defaults.update(kwargs)
        return FaissAlternatingLeastSquares(**defaults)


if __name__ == "__main__":
    unittest.main()
