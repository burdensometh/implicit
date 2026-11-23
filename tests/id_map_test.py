"""Tests for the stable business-key mapping layer and keyed model updates."""

import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import precision_at_k
from implicit.id_map import IdMap, build_interaction_matrix, dedupe_keys

# pylint: disable=redefined-outer-name


def _model(**kwargs):
    params = dict(factors=8, regularization=0.01, iterations=3, use_gpu=False, random_state=23)
    params.update(kwargs)
    return AlternatingLeastSquares(**params)


def _checker_board_keys(size=6):
    """A keyed checkerboard: even users like even items and vice versa, diagonal withheld.

    Returns ``(users, items)`` flat lists of business keys, one entry per interaction.
    """
    users, items = [], []
    for user in range(size):
        for item in range(user % 2, size, 2):
            if item == user:
                continue
            users.append(f"u{user}")
            items.append(f"sku-{item}")
    return users, items


def _empty_row(model, user_key):
    """A single-row zero interaction matrix for ``user_key`` in model item positions."""
    return csr_matrix(np.zeros((1, len(model.item_id_map))))


def _ones_row(model):
    return csr_matrix(np.ones((1, len(model.item_id_map))))


def _positional_checker_board(size):
    from recommender_base_test import get_checker_board

    return get_checker_board(size)


# --------------------------------------------------------------------------------------
# IdMap
# --------------------------------------------------------------------------------------
def test_id_map_append_only():
    id_map = IdMap(["a", "b", "c"])
    assert id_map.add("b") == 1  # existing key keeps its index
    indices = id_map.add_many(["c", "d", "e", "d"])
    assert_array_equal(indices, [2, 3, 4, 3])  # duplicates share the appended index
    assert id_map.keys == ["a", "b", "c", "d", "e"]
    assert len(id_map) == 5


def test_id_map_lookup():
    id_map = IdMap(["a", "b"])
    assert id_map.index("a") == 0
    assert id_map.key(1) == "b"
    assert id_map.indices(["b", "a"]).tolist() == [1, 0]
    assert id_map.keys_of([1, 0]) == ["b", "a"]
    with pytest.raises(KeyError):
        id_map.index("missing")
    with pytest.raises(KeyError):
        id_map.indices(["a", "missing"])
    with pytest.raises(IndexError):
        id_map.key(3)


def test_id_map_integer_keys_and_numpy_scalars():
    id_map = IdMap([np.int64(7), 3])
    assert id_map.index(7) == 0
    assert id_map.index(np.int32(3)) == 1
    assert id_map.keys == [7, 3]


def test_id_map_is_positional():
    assert IdMap(range(3)).is_positional()
    assert not IdMap(["a", "b"]).is_positional()
    assert not IdMap([0, 2]).is_positional()


def test_id_map_json_roundtrip():
    id_map = IdMap(["a", 1, 2.5, True, None])
    restored = IdMap.from_json(id_map.to_json())
    assert restored == id_map
    assert restored.keys == id_map.keys


def test_id_map_unserializable_keys_rejected():
    id_map = IdMap([("tuple", 1)])
    with pytest.raises(TypeError):
        id_map.to_json()


def test_id_map_unhashable_key_rejected():
    id_map = IdMap()
    with pytest.raises(TypeError):
        id_map.add(["list"])


def test_id_map_copy_is_independent():
    id_map = IdMap(["a"])
    copy = id_map.copy()
    copy.add("b")
    assert id_map.keys == ["a"]
    assert copy.keys == ["a", "b"]


def test_dedupe_keys_modes():
    keys = ["a", "b", "a", "c"]
    with pytest.raises(ValueError):
        dedupe_keys(keys, on_duplicate="error")
    unique, positions = dedupe_keys(keys, on_duplicate="first")
    assert unique == ["a", "b", "c"]
    assert positions.tolist() == [0, 1, 3]
    unique, positions = dedupe_keys(keys, on_duplicate="last")
    assert unique == ["a", "b", "c"]
    assert positions.tolist() == [2, 1, 3]


# --------------------------------------------------------------------------------------
# build_interaction_matrix
# --------------------------------------------------------------------------------------
def test_build_interaction_matrix_basic():
    users = ["u0", "u0", "u1"]
    items = ["a", "b", "a"]
    matrix, user_map, item_map = build_interaction_matrix(users, items, values=[1.0, 2.0, 3.0])
    assert matrix.shape == (2, 2)
    assert user_map.keys == ["u0", "u1"]
    assert item_map.keys == ["a", "b"]
    assert_array_equal(matrix.toarray(), [[1.0, 2.0], [3.0, 0.0]])


def test_build_interaction_matrix_reuses_indices_after_reorder():
    _, user_map, item_map = build_interaction_matrix(["u0"], ["a"])
    # a re-sorted feed puts a brand new sku first; existing keys must not shift
    matrix2, user_map2, item_map2 = build_interaction_matrix(
        ["u0", "u1"], ["new", "a"], user_id_map=user_map, item_id_map=item_map
    )
    assert item_map2.keys == ["a", "new"]
    assert user_map2.keys == ["u0", "u1"]
    assert matrix2[0, 1] == 1.0  # u0 -> "new"
    assert matrix2[1, 0] == 1.0  # u1 -> "a"
    assert item_map2.index("a") == item_map.index("a")


def test_build_interaction_matrix_duplicate_pair_policies():
    users = ["u0", "u0", "u0"]
    items = ["a", "a", "a"]
    values = [1.0, 2.0, 4.0]

    matrix, _, _ = build_interaction_matrix(users, items, values=values)
    assert matrix[0, 0] == 7.0  # sum is the default

    matrix, _, _ = build_interaction_matrix(users, items, values=values, on_duplicate="max")
    assert matrix[0, 0] == 4.0

    matrix, _, _ = build_interaction_matrix(users, items, values=values, on_duplicate="first")
    assert matrix[0, 0] == 1.0

    matrix, _, _ = build_interaction_matrix(users, items, values=values, on_duplicate="last")
    assert matrix[0, 0] == 4.0

    with pytest.raises(ValueError):
        build_interaction_matrix(users, items, values=values, on_duplicate="error")


def test_build_interaction_matrix_row_wise():
    users = ["u0", "u1", "u2"]
    items = [["a", "b"], ["b"], []]
    matrix, user_map, _ = build_interaction_matrix(users, items)
    assert matrix.shape == (3, 2)
    assert matrix[0].toarray().tolist() == [[1.0, 1.0]]
    assert matrix[2].toarray().tolist() == [[0.0, 0.0]]
    assert user_map.keys == ["u0", "u1", "u2"]


def test_build_interaction_matrix_row_wise_values():
    matrix, _, _ = build_interaction_matrix(
        ["u0", "u1"], [["a", "b"], ["a"]], values=[[1.0, 0.5], [2.0]]
    )
    assert matrix[0].toarray().tolist() == [[1.0, 0.5]]
    assert matrix[1, 0] == 2.0


def test_build_interaction_matrix_length_mismatch():
    with pytest.raises(ValueError):
        build_interaction_matrix(["u0"], ["a", "b"])


def test_build_interaction_matrix_explicit_shape():
    matrix, _, _ = build_interaction_matrix(["u0"], ["a"], shape=(3, 4))
    assert matrix.shape == (3, 4)


# --------------------------------------------------------------------------------------
# fit_by_key and recommendation in key space
# --------------------------------------------------------------------------------------
def test_fit_from_keys_recommends_in_key_space():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    # keys are allocated in first-appearance order and are all present
    assert set(model.user_id_map.keys) == {f"u{i}" for i in range(6)}
    assert set(model.item_id_map.keys) == {f"sku-{i}" for i in range(6)}

    ids, _ = model.recommend("u1", _empty_row(model, "u1"), N=3)
    assert all(isinstance(key, str) for key in ids)
    # checkerboard: u1 (odd) should be recommended odd items
    assert all(int(key.split("-")[1]) % 2 == 1 for key in ids)


def test_fit_from_keys_preserves_indices_when_catalog_reordered():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)
    original_users = model.user_id_map.keys.copy()
    original_items = model.item_id_map.keys.copy()

    # refit with the same data but every key listed in a different order
    model.fit_from_keys(list(reversed(users)), list(reversed(items)))

    # existing keys keep their indices; the recovered feed does not reshuffle the model
    assert model.user_id_map.keys == original_users
    assert model.item_id_map.keys == original_items
    assert model.user_factors.shape[0] == len(original_users)
    assert model.item_factors.shape[0] == len(original_items)


def test_partial_fit_users_by_key_appends_new_users():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    num_users = len(model.user_id_map)
    model.partial_fit_users_by_key(["u-new"], _ones_row(model))

    assert len(model.user_id_map) == num_users + 1
    assert model.user_id_map.index("u-new") == num_users
    assert model.user_factors.shape[0] == num_users + 1


def test_partial_fit_does_not_touch_other_users():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    before = model.user_factors.copy()
    model.partial_fit_users_by_key(["u2"], _ones_row(model))

    untouched = [index for index, key in enumerate(model.user_id_map) if key != "u2"]
    assert_array_equal(model.user_factors[untouched], before[untouched])


def test_partial_fit_users_by_key_rejects_duplicates():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    rows = csr_matrix(np.ones((2, len(model.item_id_map))))
    with pytest.raises(ValueError, match="duplicate user key"):
        model.partial_fit_users_by_key(["u1", "u1"], rows)

    # explicit collapsing is allowed
    model.partial_fit_users_by_key(["u1", "u1"], rows, on_duplicate="first")
    model.partial_fit_users_by_key(["u1", "u1"], rows, on_duplicate="last")


def test_partial_fit_items_by_key_appends_new_items():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    num_items = len(model.item_id_map)
    model.partial_fit_items_by_key("sku-new", csr_matrix(np.ones((1, len(model.user_id_map)))))

    assert len(model.item_id_map) == num_items + 1
    assert model.item_id_map.index("sku-new") == num_items
    assert model.item_factors.shape[0] == num_items + 1


def test_partial_fit_items_by_key_rejects_duplicates():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    rows = csr_matrix(np.ones((2, len(model.user_id_map))))
    with pytest.raises(ValueError, match="duplicate item key"):
        model.partial_fit_items_by_key(["sku-a", "sku-a"], rows)


def test_new_interactions_cannot_be_written_to_wrong_user():
    """The original bug: a new SKU must not shift existing rows and leak into old users."""
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    before_factors = model.user_factors.copy()
    before_keys = model.user_id_map.keys.copy()
    old_item_count = len(model.item_id_map)

    # a weekly partial update brings a new SKU that wasn't in the training catalog. Doing
    # this by integer position would append a row and then index existing users off-by-one.
    new_sku_users = np.ones((1, len(model.user_id_map)))
    model.partial_fit_items_by_key(["sku-brand-new"], csr_matrix(new_sku_users))

    # the new sku is appended; no existing user factors were touched
    assert model.item_id_map.index("sku-brand-new") == old_item_count
    assert model.item_factors.shape[0] == old_item_count + 1
    assert model.user_id_map.keys == before_keys
    assert_array_equal(model.user_factors, before_factors)

    # and a key built from the new catalog still resolves to the right row
    new_row = csr_matrix(np.zeros((1, old_item_count + 1)))
    ids, _ = model.recommend("u0", new_row, N=1)
    assert len(ids) == 1


def test_keyed_recommend_returns_keys_and_filters():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)
    row = _empty_row(model, "u0")

    ids, _ = model.recommend("u0", row, N=2)
    assert all(isinstance(key, str) for key in ids)

    ids, _ = model.recommend("u0", row, N=2, filter_items=["sku-0"])
    assert "sku-0" not in set(ids)

    ids, _ = model.recommend("u0", row, N=2, items=["sku-0", "sku-1", "sku-2"])
    assert set(ids).issubset({"sku-0", "sku-1", "sku-2"})

    # unknown key raises rather than silently indexing a row
    with pytest.raises(KeyError):
        model.recommend("unknown-user", row)


def test_keyed_similar_items_returns_keys():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    ids, _ = model.similar_items("sku-0", N=2)
    assert all(isinstance(key, str) for key in ids)


def test_keyed_model_rejects_misaligned_user_items():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    wrong_width = csr_matrix(np.zeros((1, len(model.item_id_map) + 1)))
    with pytest.raises(ValueError, match="item columns"):
        model.recommend("u0", wrong_width)


def test_set_id_maps_on_positional_model():
    model = _model()
    model.fit(_positional_checker_board(6), show_progress=False)
    assert model.item_id_map is None

    original = model.item_factors.shape[0]
    model.set_id_maps(
        user_id_map=[f"u{i}" for i in range(original)],
        item_id_map=[f"sku-{i}" for i in range(original)],
    )
    ids, _ = model.recommend("u0", csr_matrix(np.zeros((1, original))), N=1)
    assert len(ids) == 1
    assert ids[0].startswith("sku-")


# --------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------
def test_save_load_roundtrip_keeps_key_maps():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "model.npz")
        model.save(filename)
        reloaded = model.load(filename)

    assert reloaded.item_id_map == model.item_id_map
    assert reloaded.user_id_map == model.user_id_map

    row = _empty_row(model, "u0")
    assert_array_equal(model.recommend("u0", row, N=3)[0], reloaded.recommend("u0", row, N=3)[0])


def test_save_load_then_partial_fit_by_key():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "model.npz")
        model.save(filename)
        reloaded = model.load(filename)

    num_users = len(reloaded.user_id_map)
    num_items = len(reloaded.item_id_map)
    reloaded.partial_fit_items_by_key(["sku-new"], csr_matrix(np.ones((1, num_users))))
    reloaded.partial_fit_users_by_key(["u-new"], csr_matrix(np.ones((1, num_items + 1))))
    assert reloaded.item_id_map.index("sku-new") == num_items
    assert reloaded.user_id_map.index("u-new") == num_users

    ids, _ = reloaded.recommend("u-new", csr_matrix(np.zeros((1, num_items + 1))), N=2)
    assert len(ids) == 2


def test_save_load_roundtrip_keeps_float_and_int_keys():
    users = [0, 1, 2]
    items = [10.5, 20.5, 30.5]
    model = _model()
    model.fit_from_keys(users, items)
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "model.npz")
        model.save(filename)
        reloaded = model.load(filename)
    assert reloaded.user_id_map.keys == [0, 1, 2]
    assert reloaded.item_id_map.keys == [10.5, 20.5, 30.5]


def test_positional_growth_rejected_on_string_keyed_model():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    rows = csr_matrix(np.ones((1, len(model.item_id_map))))
    with pytest.raises(ValueError, match="by_key"):
        model.partial_fit_users([100], rows)


def test_positional_model_still_supports_integer_updates():
    model = _model()
    model.fit(_positional_checker_board(20), show_progress=False)
    rows = csr_matrix(np.ones((1, 20)))
    model.partial_fit_users([100], rows)
    assert model.user_factors.shape[0] == 101
    ids, _ = model.recommend(100, rows, N=2)
    assert len(ids) == 2


# --------------------------------------------------------------------------------------
# evaluation still works on keyed models (positional internals)
# --------------------------------------------------------------------------------------
def test_evaluation_on_keyed_model():
    users, items = _checker_board_keys(20)
    model = _model(iterations=5)
    model.fit_from_keys(users, items)
    # evaluate in positional space, which is what the evaluation module uses
    train = csr_matrix(np.ones((20, 20)))
    test = csr_matrix(np.eye(20))
    precision = precision_at_k(model, train, test, K=1, show_progress=False)
    assert precision >= 0


# --------------------------------------------------------------------------------------
# BPR shares the same base machinery
# --------------------------------------------------------------------------------------
def test_bpr_keyed_fit_and_update():
    users, items = _checker_board_keys(6)
    model = BayesianPersonalizedRanking(factors=4, iterations=2, use_gpu=False, random_state=42)
    model.fit_from_keys(users, items)
    assert set(model.item_id_map.keys) == {f"sku-{i}" for i in range(6)}

    # BPR has no incremental update; the keyed helper should say so and not touch the map
    keys_before = model.item_id_map.keys.copy()
    with pytest.raises(NotImplementedError):
        model.partial_fit_items_by_key("sku-new", csr_matrix(np.ones((1, 6))))
    assert model.item_id_map.keys == keys_before


# --------------------------------------------------------------------------------------
# Item-item nearest neighbour models
# --------------------------------------------------------------------------------------
def test_knn_keyed_fit_and_recommend():
    import implicit

    users, items = _checker_board_keys(6)
    model = implicit.nearest_neighbours.CosineRecommender(K=5)
    model.fit_from_keys(users, items)

    assert set(model.item_id_map.keys) == {f"sku-{i}" for i in range(6)}
    ids, _ = model.recommend("u0", _empty_row(model, "u0"), N=2)
    assert all(key in model.item_id_map.keys for key in ids)

    ids, _ = model.similar_items("sku-0", N=2)
    assert all(isinstance(key, str) for key in ids)


def test_knn_keyed_save_load():
    import implicit

    users, items = _checker_board_keys(6)
    model = implicit.nearest_neighbours.TFIDFRecommender(K=5)
    model.fit_from_keys(users, items)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "model.npz")
        model.save(filename)
        reloaded = model.load(filename)

    assert reloaded.item_id_map == model.item_id_map
    row = _empty_row(model, "u0")
    assert_array_equal(model.recommend("u0", row, N=2)[0], reloaded.recommend("u0", row, N=2)[0])


def test_keyed_model_rejects_positional_partial_fit():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    # the whole point: a positional update on a keyed model could hit the wrong entity
    rows = csr_matrix(np.ones((1, len(model.item_id_map))))
    with pytest.raises(ValueError, match="by_key"):
        model.partial_fit_users([0], rows)
    with pytest.raises(ValueError, match="by_key"):
        model.partial_fit_items([0], csr_matrix(np.ones((1, len(model.user_id_map)))))

    # and nothing was written
    assert model.user_factors.shape[0] == len(model.user_id_map)


def test_refit_trains_appended_key_rows():
    users, items = _checker_board_keys(6)
    model = _model(iterations=5)
    model.fit_from_keys(users, items)
    old_items = len(model.item_id_map)

    # refit with an extra new SKU that has interactions: its appended row should be
    # trained rather than left at zero
    model.fit_from_keys(users + ["u0"], items + ["sku-extra"])
    assert model.item_factors.shape[0] == old_items + 1
    assert np.abs(model.item_factors[-1]).sum() > 0

    ids, _ = model.recommend("u0", _empty_row(model, "u0"), N=len(model.item_id_map))
    assert "sku-extra" in set(ids)


def test_build_interaction_matrix_nested_users():
    matrix, user_map, item_map = build_interaction_matrix(
        [["u0"], ["u1"]], [["a", "b"], ["b"]]
    )
    assert user_map.keys == ["u0", "u1"]
    assert item_map.keys == ["a", "b"]
    assert matrix.toarray().tolist() == [[1.0, 1.0], [0.0, 1.0]]

    # a row belonging to more than one user is ambiguous and rejected
    with pytest.raises(ValueError, match="exactly one user"):
        build_interaction_matrix([["u0", "u1"], ["u2"]], [["a"], ["b"]])


def test_ann_wrapper_delegates_key_maps():
    import sys
    import types

    if "annoy" not in sys.modules:
        annoy = types.ModuleType("annoy")

        class _FakeIndex:
            def __init__(self, *args, **kwargs):
                pass

            def add_item(self, index, vector):
                pass

            def build(self, n_trees):
                pass

            def get_nns_by_vector(self, vector, n, **kwargs):
                return list(range(n)), [0.0] * n

        annoy.AnnoyIndex = _FakeIndex
        sys.modules["annoy"] = annoy

    from implicit.ann.annoy import AnnoyModel

    users, items = _checker_board_keys(6)
    inner = _model()
    wrapper = AnnoyModel(inner, approximate_similar_items=False, approximate_recommend=False)
    wrapper.fit_from_keys(users, items)

    # the wrapper exposes the inner model's maps, and returns business keys
    assert wrapper.item_id_map is inner.item_id_map
    ids, _ = wrapper.recommend("u0", _empty_row(inner, "u0"), N=1)
    assert ids[0] in inner.item_id_map.keys
    ids, _ = wrapper.similar_items("sku-0", N=1)
    assert ids[0] in inner.item_id_map.keys


def test_weekly_refresh_end_to_end():
    """Full scenario: catalog re-sort + new SKU, save/reopen, per-key partial training."""
    users = ["alice", "alice", "bob", "bob", "carol"]
    items = ["sku-1", "sku-2", "sku-1", "sku-3", "sku-2"]
    model = _model(iterations=5)
    model.fit_from_keys(users, items)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "model.npz")
        model.save(filename)
        reopened = model.load(filename)

    # week 2: catalog re-sorted, a brand new SKU inserted at the front, a new user
    matrix, user_map, item_map = build_interaction_matrix(
        ["alice", "bob", "dave"],
        ["sku-brand-new", "sku-1", "sku-2"],
        user_id_map=reopened.user_id_map,
        item_id_map=reopened.item_id_map,
    )
    # existing SKUs keep their index; the new one is appended
    assert reopened.item_id_map.keys[:3] == ["sku-1", "sku-2", "sku-3"]
    assert reopened.item_id_map.index("sku-brand-new") == 3

    # partial training by key works in either order, and the column widths line up
    reopened.partial_fit_items_by_key(["sku-brand-new"], csr_matrix(np.ones((1, len(user_map)))))
    reopened.partial_fit_users_by_key(["dave"], csr_matrix(np.ones((1, len(item_map)))))
    assert reopened.user_factors.shape[0] == len(reopened.user_id_map)
    assert reopened.item_factors.shape[0] == len(reopened.item_id_map)

    ids, scores = reopened.recommend(
        "dave", csr_matrix(np.zeros((1, len(reopened.item_id_map)))), N=3
    )
    assert len(ids) == 3
    assert all(key in reopened.item_id_map.keys for key in ids)


def test_partial_fit_by_key_either_order():
    users, items = _checker_board_keys(6)
    model = _model()
    model.fit_from_keys(users, items)

    # append keys to both maps first (as build_interaction_matrix would), then update in
    # the opposite order to the one the README suggests
    model.item_id_map.add("sku-new")
    model.user_id_map.add("u-new")
    model.partial_fit_users_by_key(["u-new"], csr_matrix(np.ones((1, len(model.item_id_map)))))
    model.partial_fit_items_by_key(["sku-new"], csr_matrix(np.ones((1, len(model.user_id_map)))))
    assert model.user_factors.shape[0] == len(model.user_id_map)
    assert model.item_factors.shape[0] == len(model.item_id_map)
