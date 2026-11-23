import pickle
import unittest

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from recommender_base_test import RecommenderBaseTestMixin, get_checker_board
from scipy.sparse import coo_matrix, csr_matrix, random, vstack

import implicit
from implicit.als import AlternatingLeastSquares
from implicit.gpu import HAS_CUDA

# pylint: disable=consider-using-f-string


class ALSTest(unittest.TestCase, RecommenderBaseTestMixin):
    def _get_model(self):
        return AlternatingLeastSquares(factors=32, regularization=0, use_gpu=False, random_state=23)


if HAS_CUDA:

    class GPUALSTest(unittest.TestCase, RecommenderBaseTestMixin):
        def _get_model(self):
            return AlternatingLeastSquares(
                factors=32, regularization=0, random_state=23, use_gpu=True
            )

    class GPUALSTestFloat16(unittest.TestCase, RecommenderBaseTestMixin):
        def _get_model(self):
            return AlternatingLeastSquares(
                factors=32, regularization=0, random_state=23, use_gpu=True, dtype=np.float16
            )


@pytest.mark.parametrize("use_gpu", [True, False] if HAS_CUDA else [False])
def test_zero_iterations_with_loss(use_gpu):
    model = AlternatingLeastSquares(
        factors=128, use_gpu=use_gpu, iterations=0, calculate_training_loss=True
    )
    model.fit(csr_matrix(np.ones((10, 10))), show_progress=False)


@pytest.mark.skipif(not HAS_CUDA, reason="test requires gpu")
def test_recalculate_after_cpu_conversion():
    # test out issue reported in https://github.com/benfred/implicit/issues/597
    user_items = get_checker_board(50)
    model = AlternatingLeastSquares(factors=2, use_gpu=True)
    model.fit(user_items, show_progress=False)
    original_ids, _ = model.recommend(0, user_items=user_items[0], recalculate_user=True)

    model = model.to_cpu().to_gpu()
    ids, _ = model.recommend(0, user_items=user_items[0], recalculate_user=True)

    assert_array_equal(ids, original_ids)


@pytest.mark.parametrize("use_gpu", [True, False] if HAS_CUDA else [False])
def test_recalculate_after_pickle(use_gpu):
    user_items = get_checker_board(10)
    model = AlternatingLeastSquares(factors=2, use_gpu=use_gpu, regularization=0.1)
    model.fit(user_items, show_progress=False)
    model._XtX = model._YtY = None

    original_ids, _ = model.recommend(0, user_items=user_items[0], recalculate_user=True)

    model = pickle.loads(pickle.dumps(model))
    ids, _ = model.recommend(0, user_items=user_items[0], recalculate_user=True)

    assert_array_equal(ids, original_ids)


@pytest.mark.parametrize("use_native", [True, False])
def test_cg_nan(use_native):
    # test issue with CG code that was causing NaN values in output:
    # https://github.com/benfred/implicit/issues/19#issuecomment-283164905
    raw = [
        [0.0, 2.0, 1.5, 1.33333333, 1.25, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 1.5, 1.33333333, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0, 1.5, 1.33333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 1.33333333, 1.25, 1.2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 1.33333333, 1.25],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 1.33333333],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    counts = csr_matrix(raw, dtype=np.float64)
    model = AlternatingLeastSquares(
        factors=3,
        regularization=0.01,
        dtype=np.float64,
        use_native=use_native,
        use_cg=True,
        use_gpu=False,
        random_state=23,
    )
    model.fit(counts, show_progress=False)
    rows, cols = model.item_factors, model.user_factors

    assert not np.isnan(np.sum(cols))
    assert not np.isnan(np.sum(rows))


@pytest.mark.parametrize("use_native", [True, False])
@pytest.mark.parametrize("use_gpu", [True, False] if HAS_CUDA else [False])
def test_cg_nan2(use_native, use_gpu):
    # test out Nan appearing in CG code (from https://github.com/benfred/implicit/issues/106)
    Ciu = random(
        m=100,
        n=100,
        density=0.0005,
        format="coo",
        dtype=np.float32,
        random_state=42,
        data_rvs=None,
    ).T.tocsr()

    model = AlternatingLeastSquares(
        factors=32,
        regularization=10,
        iterations=10,
        dtype=np.float32,
        random_state=23,
        use_native=use_native,
        use_gpu=use_gpu,
    )
    model.fit(Ciu, show_progress=False)

    item_factors, user_factors = model.item_factors, model.user_factors
    if use_gpu:
        item_factors, user_factors = item_factors.to_numpy(), user_factors.to_numpy()

    assert np.isfinite(item_factors).all()
    assert np.isfinite(user_factors).all()


@pytest.mark.parametrize("use_native", [True, False])
@pytest.mark.parametrize("use_gpu", [True, False] if HAS_CUDA else [False])
@pytest.mark.parametrize("use_cg", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_factorize(use_native, use_gpu, use_cg, dtype):
    if use_gpu and (not use_cg or dtype != np.float32 or not use_native):
        return

    counts = csr_matrix(
        [
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=np.float64,
    )

    model = AlternatingLeastSquares(
        factors=6,
        regularization=0,
        alpha=2.0,
        dtype=dtype,
        use_native=use_native,
        use_cg=use_cg,
        use_gpu=use_gpu,
        random_state=42,
    )
    model.fit(counts, show_progress=False)
    rows, cols = model.user_factors, model.item_factors

    if use_gpu:
        rows, cols = rows.to_numpy(), cols.to_numpy()

    reconstructed = rows.dot(cols.T)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            assert pytest.approx(counts[i, j], abs=1e-3) == reconstructed[i, j], (
                "failed to reconstruct row=%s, col=%s,"
                " value=%.5f, dtype=%s, cg=%s, native=%s gpu=%s"
                % (i, j, reconstructed[i, j], dtype, use_cg, use_native, use_gpu)
            )


def test_explain():
    counts = csr_matrix(
        [
            [1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [1, 4, 1, 0, 7, 0],
            [1, 1, 0, 0, 0, 0],
            [9, 0, 4, 1, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 2, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    user_items = counts.T.tocsr()

    model = AlternatingLeastSquares(
        factors=4,
        regularization=20,
        alpha=2.0,
        use_native=False,
        use_cg=False,
        use_gpu=False,
        iterations=100,
        random_state=23,
    )
    model.fit(user_items, show_progress=False)

    userid = 0

    # Assert recommendation is the the same if we recompute user vectors
    # TODO: this doesn't quite work with N=10 (because we returns items that should have been
    # filtered with large negative score?) also seems like the dtype is different between
    # recalculate and not
    ids, scores = model.recommend(userid, user_items[userid], N=3)
    recalculated_ids, recalculated_scores = model.recommend(
        userid, user_items[userid], N=3, recalculate_user=True
    )
    for item1, score1, item2, score2 in zip(ids, scores, recalculated_ids, recalculated_scores):
        assert item1 == item2
        assert pytest.approx(score1, abs=1e-4) == score2

    # Assert explanation makes sense
    top_rec, score = recalculated_ids[0], recalculated_scores[0]
    score_explained, contributions, W = model.explain(userid, user_items, itemid=top_rec)
    scores = [s for _, s in contributions]
    items = [i for i, _ in contributions]
    assert pytest.approx(score, abs=1e-4) == score_explained
    assert pytest.approx(score, abs=1e-4) == sum(scores)
    assert scores == sorted(scores, reverse=True)

    assert scores == sorted(scores, reverse=True), "Scores not in order"
    assert [0, 2, 3, 4] == sorted(items), "Items not seen by user"

    # Assert explanation with precomputed user weights is correct
    top_score_explained, top_contributions, W = model.explain(
        userid, user_items, itemid=top_rec, user_weights=W, N=2
    )
    top_scores = [s for _, s in top_contributions]
    top_items = [i for i, _ in top_contributions]

    assert len(top_contributions) == 2
    assert pytest.approx(score, abs=1e-4) == top_score_explained
    assert scores[:2] == top_scores
    assert items[:2] == top_items


@pytest.mark.parametrize("use_gpu", [True, False] if HAS_CUDA else [False])
def test_small_nan(use_gpu):
    # test out case where number of factors is larger than number of users:
    # https://github.com/benfred/implicit/issues/377
    user_item_matrix = coo_matrix((np.ones(10), (np.arange(10), np.arange(10)))).tocsr()
    als = AlternatingLeastSquares(factors=15, use_gpu=use_gpu)
    als.fit(user_item_matrix)

    ids, scores = als.recommend(0, user_item_matrix, 10, filter_already_liked_items=False)

    # shouldn't have any NaN values in the resulting dataset
    assert not np.isnan(scores).any()

    # first item recommended should be the liked item in the training set
    assert ids[0] == 0


@pytest.mark.parametrize("use_gpu", [True, False] if HAS_CUDA else [False])
def test_incremental_retrain(use_gpu):
    likes = get_checker_board(50)

    model = AlternatingLeastSquares(factors=2, regularization=0, use_gpu=use_gpu, random_state=23)
    model.fit(likes, show_progress=False)

    ids, _ = model.recommend(0, likes[0])
    assert ids[0] == 0

    # refit the model for user 0, make them like the same thing as user 1
    model.partial_fit_users([0], likes[1])
    ids, _ = model.recommend(0, likes[1])
    assert ids[0] == 1

    # add a new user at position 100, make sure we can also use that for recommendations
    model.partial_fit_users([100], likes[1])
    ids, _ = model.recommend(100, likes[1])
    assert ids[0] == 1

    # add a new item at position 100, make sure it gets recommended for user right away
    model.partial_fit_items([100], likes[1])
    ids, _ = model.recommend(1, likes[1], N=2)
    assert set(ids) == {1, 100}

    # check to make sure we can index only a single extra item/user
    model.partial_fit_users([101], likes[1])
    model.partial_fit_items([101], likes[1])
    ids, _ = model.recommend(101, likes[1], N=3)
    assert set(ids) == {1, 100, 101}


@pytest.mark.parametrize("use_gpu", [True, False] if HAS_CUDA else [False])
def test_calculate_loss_simple(use_gpu):
    if use_gpu:
        calculate_loss = implicit.gpu.als.calculate_loss

    else:
        calculate_loss = implicit.cpu.als.calculate_loss

    # the only user has liked item 0, but not interacted with item 1
    n_users, n_items = 1, 2
    ratings = coo_matrix(([1.0], ([0], [0])), shape=(n_users, n_items)).tocsr()

    # factors are designed to be perfectly wrong, to test loss function
    item_factors = np.array([[0.0], [1.0]], dtype="float32")
    user_factors = np.array([[1.0]], dtype="float32")

    loss = calculate_loss(ratings, user_factors, item_factors, regularization=0)
    assert loss == pytest.approx(1.0)

    loss = calculate_loss(ratings, user_factors, item_factors, regularization=1.0)
    assert loss == pytest.approx(2.0)


@pytest.mark.skipif(not implicit.gpu.HAS_CUDA, reason="needs cuda build")
@pytest.mark.parametrize("n_users", [2**13, 2**19])
@pytest.mark.parametrize("n_items", [2**19])
@pytest.mark.parametrize("n_samples", [2**20])
@pytest.mark.parametrize("regularization", [0.0, 1.0, 500000.0])
def test_gpu_loss(n_users, n_items, n_samples, regularization):
    # we used to have some errors in the gpu loss function
    # if  n_items * n_users >2**31. Test out that the loss on the gpu
    # matches that on the cpu
    # https://github.com/benfred/implicit/issues/441
    # https://github.com/benfred/implicit/issues/367
    liked_items = np.random.randint(0, n_items, n_samples)
    liked_users = np.random.randint(0, n_users, n_samples)
    ratings = coo_matrix(
        (np.ones(n_samples), (liked_users, liked_items)), shape=(n_users, n_items)
    ).tocsr()

    factors = 32
    item_factors = np.random.random((n_items, factors)).astype("float32")
    user_factors = np.random.random((n_users, factors)).astype("float32")

    gpu_loss = implicit.gpu.als.calculate_loss(ratings, user_factors, item_factors, regularization)
    cpu_loss = implicit.cpu.als.calculate_loss(ratings, user_factors, item_factors, regularization)

    assert gpu_loss == pytest.approx(cpu_loss, rel=1e-4)


def test_calculate_loss_segfault():
    # this code used to segfault, because of a bug in calculate_loss
    factors = 1
    regularization = 0
    n_users, n_items = 4, 4

    item_factors = np.random.random((n_items, factors)).astype("float32")
    user_factors = np.random.random((n_users, factors)).astype("float32")
    c_ui = coo_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(n_users, n_items)).tocsr()

    loss = implicit.cpu.als.calculate_loss(c_ui, user_factors, item_factors, regularization)
    assert loss > 0


def _fitted_partial_model(alpha=1.0):
    likes = get_checker_board(50)
    model = AlternatingLeastSquares(
        factors=8, regularization=0.1, alpha=alpha, use_gpu=False, random_state=23
    )
    model.fit(likes, show_progress=False)
    return likes, model


def test_partial_fit_matches_sequential_partial_fit():
    likes, model = _fitted_partial_model(alpha=2.0)

    userids = np.array([0, 1, 2])
    itemids = np.array([3, 4, 50])
    # new item 50 is liked by the new user 100 and existing users 0/1. Both
    # feedback matrices use the post-batch key spaces (101 users and 101 items)
    user_items = likes[userids]
    user_items = csr_matrix(
        (user_items.data, user_items.indices, user_items.indptr),
        shape=(user_items.shape[0], 101),
    )
    # item side: rows for items 3, 4 taken from the transposed feedback, widened
    # to 101 columns, plus a brand new item 50 interacting with users 0, 1, 100
    existing_item_users = likes.T.tocsr()[itemids[:-1]]
    item_users = csr_matrix(
        (
            existing_item_users.data,
            existing_item_users.indices,
            existing_item_users.indptr,
        ),
        shape=(existing_item_users.shape[0], 101),
    )
    item_users = vstack(
        [
            item_users,
            csr_matrix(
                (np.ones(3, dtype=np.float32), np.array([0, 1, 100]), np.array([0, 3])),
                shape=(1, 101),
            ),
        ]
    ).tocsr()
    new_user_items = csr_matrix(likes[1], shape=(1, 101))

    # reference: the existing one-sided partial fit calls, in the same order
    reference = AlternatingLeastSquares(
        factors=8, regularization=0.1, alpha=2.0, use_gpu=False, random_state=23
    )
    reference.fit(likes, show_progress=False)
    reference.partial_fit_users(
        np.append(userids, 100), vstack([user_items, new_user_items])
    )
    reference.partial_fit_items(itemids, item_users)

    model.partial_fit(
        np.append(userids, 100), vstack([user_items, new_user_items]),
        itemids, item_users,
    )

    np.testing.assert_allclose(model.user_factors, reference.user_factors, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.item_factors, reference.item_factors, rtol=1e-5, atol=1e-6)


def test_partial_fit_adds_new_users_and_items_together():
    likes, model = _fitted_partial_model()

    # a brand new user at id 100 and a brand new item at id 100, refreshed in
    # one atomic batch
    new_user_items = csr_matrix(([1.0], [0], [0, 1]), shape=(1, 101))
    new_item_users = csr_matrix(([1.0], [100], [0, 1]), shape=(1, 101))

    model.partial_fit(np.array([100]), new_user_items, np.array([100]), new_item_users)

    assert model.user_factors.shape == (101, 8)
    assert model.item_factors.shape == (101, 8)

    # the new item should show up as similar to item 0 (both liked by user 100)
    ids, _ = model.similar_items(100, N=2)
    assert 0 in ids

    # recommendations work for the new user
    ids, _ = model.recommend(100, new_user_items, N=5)
    assert 100 in ids


def test_partial_fit_validates_keys_and_shapes_before_touching_model():
    likes, model = _fitted_partial_model()
    original_user_factors = model.user_factors.copy()
    original_item_factors = model.item_factors.copy()

    good_user_items = likes[np.array([0])]
    good_item_users = likes.T.tocsr()[np.array([0])]

    # row count mismatch on the user side
    with pytest.raises(ValueError, match="user_items must contain 1 row"):
        model.partial_fit(np.array([0, 1]), likes[np.array([0])],
                          np.array([0]), good_item_users)

    # row count mismatch on the item side
    with pytest.raises(ValueError, match="item_users must contain 1 row"):
        model.partial_fit(np.array([0]), good_user_items,
                          np.array([0, 1]), good_item_users)

    # user_items references an item id that neither exists nor is in itemids
    bad_user_items = csr_matrix(([1.0], [99], [0, 1]), shape=(1, 100))
    with pytest.raises(ValueError, match="user_items references item id"):
        model.partial_fit(np.array([0]), bad_user_items,
                          np.array([0]), good_item_users)

    # item_users references a user id that neither exists nor is in userids
    bad_item_users = csr_matrix(([1.0], [99], [0, 1]), shape=(1, 100))
    with pytest.raises(ValueError, match="item_users references user id"):
        model.partial_fit(np.array([0]), good_user_items,
                          np.array([0]), bad_item_users)

    # negative ids are rejected
    with pytest.raises(ValueError, match="negative"):
        model.partial_fit(np.array([-1]), good_user_items,
                          np.array([0]), good_item_users)
    with pytest.raises(ValueError, match="negative"):
        model.partial_fit(np.array([0]), good_user_items,
                          np.array([-1]), good_item_users)

    # nothing was committed: the previous complete version is still visible
    np.testing.assert_array_equal(model.user_factors, original_user_factors)
    np.testing.assert_array_equal(model.item_factors, original_item_factors)


def test_partial_fit_failure_keeps_previous_version():
    likes, model = _fitted_partial_model()
    original_user_factors = model.user_factors.copy()
    original_item_factors = model.item_factors.copy()

    # force the item side solve to fail after the user side has succeeded:
    # validation passes (ids are in range), but monkeypatch the native solver
    # to blow up on the 2nd (item side) call
    calls = {"n": 0}
    original_solver = implicit.cpu.als._als._least_squares

    def failing_solver(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("simulated item side failure")
        return original_solver(*args, **kwargs)

    implicit.cpu.als._als._least_squares = failing_solver
    try:
        with pytest.raises(ValueError, match="simulated item side failure"):
            model.partial_fit(
                np.array([0]), likes[np.array([0])],
                np.array([0]), likes.T.tocsr()[np.array([0])],
            )
    finally:
        implicit.cpu.als._als._least_squares = original_solver

    # neither side was committed, and the caches were not invalidated
    np.testing.assert_array_equal(model.user_factors, original_user_factors)
    np.testing.assert_array_equal(model.item_factors, original_item_factors)


def test_partial_fit_invalidates_cached_similarities():
    likes, model = _fitted_partial_model()

    # populate the similarity caches from the previous complete version
    old_item_norms = model.item_norms.copy()
    old_user_norms = model.user_norms.copy()
    assert model._item_norms is not None and model._user_norms is not None

    model.partial_fit(
        np.array([0, 1]), likes[np.array([0, 1])],
        np.array([0, 1]), likes.T.tocsr()[np.array([0, 1])],
    )

    # all cached state must be rebuilt from the new version
    assert model._item_norms is None and model._user_norms is None
    assert model._YtY is None and model._XtX is None
    assert not np.allclose(model.item_norms, old_item_norms)
    assert not np.allclose(model.user_norms, old_user_norms)

    # cached similarity results now use the new factors
    ids, _ = model.similar_items(1, N=2)
    assert ids[0] == 1


def test_partial_fit_requires_fitted_model():
    model = AlternatingLeastSquares(factors=8, use_gpu=False)
    empty = csr_matrix((1, 1))
    with pytest.raises(ValueError, match="hasn't been fit"):
        model.partial_fit(np.array([0]), empty, np.array([0]), empty)


def test_partial_fit_supports_empty_side():
    likes, model = _fitted_partial_model()

    # only users refreshed, no items in this batch
    model.partial_fit(
        np.array([0]), likes[np.array([0])],
        np.array([], dtype=np.intp), csr_matrix((0, likes.shape[0])),
    )
    assert model.item_factors.shape == (likes.shape[1], 8)

    # only items refreshed, no users in this batch
    model.partial_fit(
        np.array([], dtype=np.intp), csr_matrix((0, likes.shape[1])),
        np.array([1]), likes.T.tocsr()[np.array([1])],
    )
    assert model.user_factors.shape == (likes.shape[0], 8)

