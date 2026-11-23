"""Common test functions for all recommendation models"""

import os
import pickle
import random
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import coo_matrix, csr_matrix

from implicit.evaluation import precision_at_k
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.utils import ParameterWarning

# pylint: disable=too-many-public-methods


def get_checker_board(X):
    """Returns a 'checkerboard' matrix: where every even userid has liked
    every even itemid and every odd userid has liked every odd itemid.
    The diagonal is withheld for testing recommend methods"""
    ret = np.zeros((X, X))
    for i in range(X):
        for j in range(i % 2, X, 2):
            ret[i, j] = 1.0
    return csr_matrix(ret - np.eye(X))


class RecommenderBaseTestMixin:
    """Mixin to test a bunch of common functionality in models
    deriving from RecommenderBase"""

    def _get_model(self):
        raise NotImplementedError()

    def test_recommend(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        for userid in range(50):
            ids, _ = model.recommend(userid, user_items[userid], N=1)
            self.assertEqual(len(ids), 1)

            # the top item recommended should be the same as the userid:
            # its the one withheld item for the user that is liked by
            # all the other similar users
            self.assertEqual(ids[0], userid)

        # try asking for more items than possible,
        # should return only the available items
        # https://github.com/benfred/implicit/issues/22
        ids, _ = model.recommend(0, user_items[0], N=10000)
        self.assertTrue(len(ids))

        # filter recommended items using an additional filter list
        # https://github.com/benfred/implicit/issues/26
        ids, _ = model.recommend(0, user_items[0], N=1, filter_items=[0])
        self.assertTrue(0 not in set(ids))

    def test_recommend_batch(self):
        user_items = get_checker_board(50)

        model = self._get_model()
        model.fit(user_items, show_progress=False)

        userids = np.arange(50)
        ids, scores = model.recommend(userids, user_items[userids], N=1)
        for userid in userids:
            assert len(ids[userid]) == 1

            # the top item recommended should be the same as the userid:
            # its the one withheld item for the user that is liked by
            # all the other similar users
            assert ids[userid][0] == userid

            # make sure the batch recommend results match those for a single user
            ids_user, scores_user = model.recommend(userid, user_items[userid], N=1)
            assert np.allclose(ids_user, ids[userid])
            assert np.allclose(scores_user, scores[userid])

        userids = np.array([2, 3, 4])
        ids, _ = model.recommend(userids, user_items[userids], N=1)

        for i, userid in enumerate(userids):
            assert ids[i][0] == userid

        # filter recommended items using an additional filter list
        ids, _ = model.recommend(userids, user_items[userids], N=1, filter_items=[0])
        for i, _ in enumerate(userids):
            assert 0 not in ids[i]

        # also make sure the results when filter_already_liked_items=False match in batch vs
        # scalar mode
        userids = np.arange(50)
        ids, scores = model.recommend(
            userids, user_items[userids], N=5, filter_already_liked_items=False
        )

        for userid in range(50):
            ids_user, scores_user = model.recommend(
                userid, user_items[userid], N=5, filter_already_liked_items=False
            )
            assert np.allclose(scores_user, scores[userid])
            assert np.allclose(ids_user, ids[userid])

    def test_recalculate_user(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        try:
            userids = np.arange(50)
            batch_ids, batch_scores = model.recommend(
                userids, user_items[userids], N=1, recalculate_user=True
            )
        except NotImplementedError:
            # some models don't support recalculating user on the fly, and that's ok
            return

        for userid in range(item_users.shape[1]):
            ids, scores = model.recommend(userid, user_items[userid], N=1)
            self.assertEqual(len(ids), 1)
            user_vector = user_items[userid]

            # we should get the same item if we recalculate_user
            ids_from_liked, scores_from_liked = model.recommend(
                userid=0, user_items=user_vector, N=1, recalculate_user=True
            )
            self.assertEqual(ids[0], ids_from_liked[0])

            # TODO: if we set regularization for the model to be sufficiently high, the
            # scores from recalculate_user are slightly different. Investigate
            # (could be difference between CG and cholesky optimizers?)
            self.assertAlmostEqual(scores[0], scores_from_liked[0], places=3)

            # we should also get the same from the batch recommend call already done
            self.assertEqual(batch_ids[userid][0], ids_from_liked[0])
            self.assertAlmostEqual(batch_scores[userid][0], scores_from_liked[0], places=3)

    def test_evaluation(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        # we've withheld the diagonal for testing, and have verified that in test_recommend
        # it is returned for each user. So p@1 should be 1.0
        p = precision_at_k(
            model, user_items.tocsr(), csr_matrix(np.eye(50)), K=1, show_progress=False
        )
        self.assertEqual(p, 1)

    def test_similar_users(self):
        model = self._get_model()
        model.fit(get_checker_board(50), show_progress=False)

        try:
            for userid in range(50):
                ids, _ = model.similar_users(userid, N=10)
                for r in ids:
                    self.assertEqual(r % 2, userid % 2)
        except NotImplementedError:
            pass

    def test_similar_users_batch(self):
        model = self._get_model()
        model.fit(get_checker_board(256), show_progress=False)
        userids = np.arange(50)

        try:
            ids, scores = model.similar_users(userids, N=10)
        except NotImplementedError:
            # similar users isn't implemented for many models (ItemItemRecommeder/ ANN models)
            return

        self.assertEqual(ids.shape, (50, 10))

        for userid in userids:
            # first user returned should be itself, and score should be ~1.0
            self.assertEqual(ids[userid][0], userid)
            self.assertAlmostEqual(scores[userid][0], 1.0, places=4)

            # the rest of the users should be even or odd depending on the userid
            for r in ids[userid]:
                self.assertEqual(r % 2, userid % 2)

    def test_similar_users_filter(self):
        model = self._get_model()
        # calculating similar users in nearest-neighbours is not implemented yet
        if isinstance(model, ItemItemRecommender):
            return

        model.fit(get_checker_board(256), show_progress=False)
        userids = np.arange(50)

        try:
            ids, _ = model.similar_users(userids, N=10, filter_users=np.arange(52) * 5)
        except NotImplementedError:
            return

        for userid in userids:
            for r in ids[userid]:
                self.assertTrue(r % 5 != 0)

        selected = np.arange(10)
        ids, _ = model.similar_users(userids, N=10, users=selected)
        for userid in userids:
            self.assertEqual(set(ids[userid]), set(selected))

    def test_similar_items(self):
        model = self._get_model()
        user_items = get_checker_board(256)
        item_users = user_items.T.tocsr()
        model.fit(user_items, show_progress=False)

        for itemid in range(50):
            ids, scores = model.similar_items(itemid, N=10)
            for r in ids:
                self.assertEqual(r % 2, itemid % 2)

            try:
                recalculated_ids, recalculated_scores = model.similar_items(
                    itemid, N=10, item_users=item_users[itemid]
                )
                assert np.allclose(ids, recalculated_ids)
                assert np.allclose(scores, recalculated_scores)
            except NotImplementedError:
                continue

    def test_similar_items_batch(self):
        model = self._get_model()
        user_items = get_checker_board(256)
        model.fit(user_items, show_progress=False)
        itemids = np.arange(50)

        def check_results(ids):
            self.assertEqual(ids.shape, (50, 10))
            for itemid in itemids:
                # first item returned should be itself
                self.assertEqual(ids[itemid][0], itemid)

                # the rest of the items should be even or odd depending on the itemid
                for r in ids[itemid]:
                    self.assertEqual(r % 2, itemid % 2)

        ids, _ = model.similar_items(itemids, N=10)
        check_results(ids)
        try:
            ids, _ = model.similar_items(
                itemids, N=10, recalculate_item=True, item_users=user_items.T.tocsr()[itemids]
            )
            check_results(ids)
        except NotImplementedError:
            # some models don't support recalculating user on the fly, and that's ok
            pass

    def test_similar_items_filter(self):
        model = self._get_model()

        model.fit(get_checker_board(256), show_progress=False)
        itemids = np.arange(50)

        ids, _ = model.similar_items(itemids, N=10, filter_items=np.arange(52) * 5)
        for itemid in itemids:
            for r in ids[itemid]:
                self.assertTrue(r % 5 != 0)

        try:
            selected = np.arange(10)
            ids, _ = model.similar_items(itemids, N=10, items=selected)
            for itemid in itemids:
                self.assertEqual(set(ids[itemid]), set(selected))
        except NotImplementedError:
            # some models don't support a 'items' filter on the similar_items call
            pass

    def test_zero_length_row(self):
        # get a matrix where a row/column is 0
        item_users = get_checker_board(50).todense()
        item_users[42] = 0
        item_users[:, 42] = 0

        # also set the last row/column to 0 (test out problem reported here
        # https://github.com/benfred/implicit/issues/86#issuecomment-373385686)
        item_users[49] = 0
        item_users[:, 49] = 0

        model = self._get_model()
        model.fit(csr_matrix(item_users), show_progress=False)

        # item 42 has no users, shouldn't be similar to anything
        for itemid in range(40):
            ids, _ = model.similar_items(itemid, 10)
            self.assertTrue(42 not in ids)

    def test_fit_non_csr_matrix(self):
        # models should be able to fit user_items that are passed as a COO or LiL matrix
        # but should warn on input
        user_items = get_checker_board(50)
        model = self._get_model()

        with pytest.warns(ParameterWarning):
            model.fit(user_items.tocoo(), show_progress=False)

        model = self._get_model()
        with pytest.warns(ParameterWarning):
            model.fit(user_items.tolil(), show_progress=False)

    def test_fit_ordering(self):
        # models should return scores that are decreasing in value
        samples = 1000
        user_count = 100
        item_count = 200

        rng = np.random.RandomState(10)
        itemids = rng.randint(0, item_count, size=samples, dtype=np.int32)
        userids = rng.randint(0, user_count, size=samples, dtype=np.int32)

        likes = coo_matrix((np.ones(samples), (userids, itemids))).tocsr()

        model = self._get_model()

        model.fit(likes, show_progress=False)
        for userid in userids:
            ids, scores = model.recommend(userid, likes[userid])
            print(ids, scores)
            assert np.all(np.diff(scores) <= 0)

    def test_dtype(self):
        # models should be able to accept input of either float32 or float64
        item_users = get_checker_board(50)
        model = self._get_model()
        model.fit(item_users.astype(np.float64), show_progress=False)

        model = self._get_model()
        model.fit(item_users.astype(np.float32), show_progress=False)

    def test_rank_items(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        try:
            selected_items = np.array([1, 2, 3, 4, 5, 6])
            ids, _ = model.recommend(0, user_items[0], items=selected_items, N=20)

            self.assertEqual(len(ids), len(selected_items))

            # ranked list should have same items
            self.assertEqual(set(ids), set(selected_items))

            if not isinstance(model, ItemItemRecommender):
                # items 2,4,6 are in the 'filter_alread_liked_items' set
                # and should be in the last 3 positions (except with itemitemrecommenders
                # where its a little more complicated since the sparse results)
                self.assertEqual(set(ids[3:]), set([2, 4, 6]))

        except NotImplementedError:
            return

        for userid in range(50):
            selected_items = random.sample(range(50), 10)

            ids, _ = model.recommend(
                userid, user_items[userid], items=selected_items, filter_already_liked_items=False
            )
            # ranked list should have same items
            self.assertEqual(set(ids), set(selected_items))

            wrong_neg_items = [-1, -3, -5]
            wrong_pos_items = [51, 300, 200]

            # rank_items should raise IndexError if selected items contains wrong itemids
            with self.assertRaises(IndexError):
                wrong_item_list = selected_items + wrong_neg_items
                model.recommend(userid, user_items[userid], items=wrong_item_list)
            with self.assertRaises(IndexError):
                wrong_item_list = selected_items + wrong_pos_items
                model.recommend(userid, user_items[userid], items=wrong_item_list)

    def test_rank_items_batch(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        selected_items = np.arange(10) * 3
        try:
            ids, _ = model.recommend(np.arange(50), user_items, items=selected_items)
        except NotImplementedError:
            return

        for userid in range(50):
            current_ids = ids[userid]
            self.assertEqual(set(current_ids), set(selected_items))

    def test_recommend_per_user_items(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        userids = np.array([0, 1, 2])
        items = [[0, 2, 4, 6], [1, 3, 5], [2, 4, 6, 8, 10]]
        try:
            ids, scores = model.recommend(userids, user_items[userids], N=10, items=items)
        except NotImplementedError:
            # per-user items aren't supported by ANN models
            return

        # already liked items are filtered out of each users candidate set:
        # user 0 has liked 2,4,6 / user 1 has liked 3,5 / user 2 has liked 4,6,8,10
        self.assertEqual(len(ids), len(userids))
        self.assertEqual(len(scores), len(userids))
        for i, expected in enumerate([{0}, {1}, {2}]):
            self.assertEqual(set(ids[i]), expected)
            self.assertEqual(len(ids[i]), len(scores[i]))

        # results come back in the same order as the userids passed in
        shuffled_userids = np.array([2, 0, 1])
        shuffled_items = [[2, 4], [0, 2], [1, 3]]
        ids, _ = model.recommend(
            shuffled_userids, user_items[shuffled_userids], N=10, items=shuffled_items
        )
        for i, expected in enumerate([{2}, {0}, {1}]):
            self.assertEqual(set(ids[i]), expected)

        # a 2D array of per-user candidates should also work
        ids, _ = model.recommend(
            userids, user_items[userids], N=5, items=np.array([[0, 2], [1, 3], [2, 4]])
        )
        for i, expected in enumerate([{0}, {1}, {2}]):
            self.assertEqual(set(ids[i]), expected)

        # filter_items removes items from each users candidate set, and can
        # leave users with no candidates at all
        ids, scores = model.recommend(
            userids, user_items[userids], N=10, items=items, filter_items=[0, 1]
        )
        self.assertEqual(len(ids[0]), 0)
        self.assertEqual(len(ids[1]), 0)
        self.assertEqual(set(ids[2]), {2})

        # empty candidate sets return empty results, rather than falling back
        # to recommending from the entire catalogue
        ids, scores = model.recommend(userids, user_items[userids], N=10, items=[[], [], []])
        for i in range(len(userids)):
            self.assertEqual(len(ids[i]), 0)
            self.assertEqual(len(scores[i]), 0)

        # candidate sets with fewer than N valid items aren't padded out to N
        ids, scores = model.recommend(
            userids,
            user_items[userids],
            N=10,
            items=[[4], [5, 7], []],
            filter_already_liked_items=False,
        )
        self.assertEqual(set(ids[0]), {4})
        self.assertEqual(set(ids[1]), {5, 7})
        self.assertEqual(len(ids[2]), 0)
        for i in range(len(userids)):
            self.assertTrue(np.all(np.asarray(ids[i]) >= 0))
            self.assertTrue(np.all(np.diff(scores[i]) <= 0))

        # a single user can also get recommendations from their own candidate set
        ids, scores = model.recommend(0, user_items[0], N=10, items=[[0, 2, 4, 6]])
        self.assertEqual(set(ids), {0})

        # batch and scalar calls should agree with each other
        batch_ids, batch_scores = model.recommend(
            userids, user_items[userids], N=3, items=items, filter_already_liked_items=False
        )
        for i, userid in enumerate(userids):
            ids, scores = model.recommend(
                userid, user_items[userid], N=3, items=[items[i]], filter_already_liked_items=False
            )
            assert_array_equal(batch_ids[i], ids)
            assert_array_equal(batch_scores[i], scores)

        # invalid itemids in the candidate sets should raise an IndexError
        with self.assertRaises(IndexError):
            model.recommend(userids, user_items[userids], items=[[0], [1], [50]])
        with self.assertRaises(IndexError):
            model.recommend(userids, user_items[userids], items=[[0], [1], [-1]])

        # passing the wrong number of per-user candidate sets should raise a ValueError
        with self.assertRaises(ValueError):
            model.recommend(userids, user_items[userids], items=[[0], [1]])
        with self.assertRaises(ValueError):
            model.recommend(userids, user_items[userids], items=np.array([[0], [1]]))

    def test_recommend_items_with_filter_items(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        try:
            # shared candidate sets can be combined with filter_items
            ids, _ = model.recommend(
                0, user_items[0], N=10, items=[0, 2, 4], filter_items=[0],
                filter_already_liked_items=False,
            )
        except NotImplementedError:
            # items aren't supported by ANN models
            return
        self.assertEqual(set(ids), {2, 4})

        # an empty shared candidate set returns empty results, rather than
        # falling back to the entire catalogue
        ids, scores = model.recommend(0, user_items[0], N=10, items=[])
        self.assertEqual(len(ids), 0)
        self.assertEqual(len(scores), 0)

        ids, scores = model.recommend(np.arange(3), user_items[np.arange(3)], N=10, items=[])
        self.assertEqual(ids.shape, (3, 0))
        self.assertEqual(scores.shape, (3, 0))

    def test_pickle(self):
        user_items = get_checker_board(50)
        model = self._get_model()
        model.fit(user_items, show_progress=False)

        pickled = pickle.dumps(model)
        reloaded = pickle.loads(pickled)

        # make sure we can call methods on the reloaded index, and get the same results back
        # (https://github.com/benfred/implicit/issues/631)
        ids, _ = model.recommend(0, user_items[0])
        reloaded_ids, _ = reloaded.recommend(0, user_items[0])
        assert_array_equal(ids, reloaded_ids)

        ids, _ = model.similar_items(0)
        reloaded_ids, _ = reloaded.similar_items(0)
        assert_array_equal(ids, reloaded_ids)

    def test_pickle_unfitted_model(self):
        model = self._get_model()
        pickle.loads(pickle.dumps(model))

    def test_invalid_user_items(self):
        user_items = get_checker_board(50)
        model = self._get_model()
        model.fit(user_items, show_progress=False)

        with self.assertRaises(ValueError):
            model.recommend(0, user_items=user_items.tocsc())

        with self.assertRaises(ValueError):
            model.recommend(0, user_items=user_items.tocoo())

    def test_serialization(self):
        likes = get_checker_board(50)

        model = self._get_model()
        model.fit(likes, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "model.npz")
            model.save(filename)
            reloaded = model.load(filename)

            assert_array_equal(model.similar_items(1)[0], reloaded.similar_items(1)[0])
            assert_array_equal(model.similar_items(1)[1], reloaded.similar_items(1)[1])

        # also make  sure we can save/load with file objects instead of filenames
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "model.npz"), "wb+") as f:
                model.save(f)
                f.seek(0)
                reloaded = model.load(f)
                assert_array_equal(model.similar_items(1)[0], reloaded.similar_items(1)[0])
                assert_array_equal(model.similar_items(1)[1], reloaded.similar_items(1)[1])

    def test_serialization_without_fit(self):
        model = self._get_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "model.npz")
            model.save(filename)
            reloaded = model.load(filename)
            assert model.__dict__ == reloaded.__dict__

    def test_fit_callback(self):
        model = self._get_model()

        num_called = 0

        def callback(*args, **kwargs):
            nonlocal num_called
            num_called += 1

        try:
            model.fit(get_checker_board(5), show_progress=False, callback=callback)
        except NotImplementedError:
            # callback isn't supported w/ ItemItem KNN models
            return

        assert num_called >= 1
