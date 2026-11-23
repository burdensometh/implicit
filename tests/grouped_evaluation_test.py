"""Tests for the grouped ranking evaluation (implicit.evaluation).

These tests are deliberately kept free of dataset downloads (no h5py /
movielens dependency) so the grouping logic is verified on small,
hand-computable matrices.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, random

import implicit
from implicit.evaluation import (
    item_groups_by_train_count,
    ranking_metrics_at_k,
    ranking_metrics_at_k_by_item_group,
    train_test_split,
)


class _StaticRecommender:
    """Minimal recommender stub returning fixed top-K item ids per user.

    Users without an entry get no recommendations (padded with -1, which
    can never be a hit).
    """

    def __init__(self, recommendations):
        self.recommendations = recommendations

    def recommend(self, userids, user_items, N=10):
        userids = np.atleast_1d(userids)
        ids = np.full((len(userids), N), -1, dtype=np.int32)
        for row, userid in enumerate(userids):
            recs = self.recommendations.get(int(userid), [])[:N]
            ids[row, : len(recs)] = recs
        return ids, None


def _user_items_matrix(user_items, shape):
    """Builds a binary csr_matrix from a {user: [item, ...]} dict."""
    users, items = [], []
    for user, liked in user_items.items():
        for item in liked:
            users.append(user)
            items.append(item)
    return csr_matrix(
        (np.ones(len(users)), (users, items)), shape=shape, dtype=np.float32
    )


def _get_train_matrix():
    """Train matrix with item interaction counts [3, 2, 1, 0]."""
    return _user_items_matrix(
        {0: [0, 1, 2], 1: [0, 1], 2: [0]},
        shape=(3, 4),
    )


def test_item_groups_by_train_count_thresholds():
    train = _get_train_matrix()
    groups, labels = item_groups_by_train_count(train, thresholds=[2])

    # item0 (3 interactions) and item1 (2) are in the upper group,
    # item2 (1) in the lower group, item3 is unseen in training
    assert groups.tolist() == [1, 1, 0, -1]
    assert labels == ["[1, 2)", "[2, inf)"]


def test_item_groups_by_train_count_quantiles():
    # user i interacts with items 0..i, so item j has 10 - j interactions
    train = _user_items_matrix({u: list(range(u + 1)) for u in range(10)}, shape=(10, 10))
    assert train.getnnz(axis=0).tolist() == list(range(10, 0, -1))

    groups, labels = item_groups_by_train_count(train, num_groups=2)

    # median of counts 1..10 is 5.5, so the split is at 6
    assert labels == ["[1, 6)", "[6, inf)"]
    assert groups.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def test_item_groups_default_is_three_groups():
    train = _get_train_matrix()
    _, labels = item_groups_by_train_count(train)
    assert len(labels) == 3


def test_item_groups_unseen_items_get_dedicated_group():
    train = _get_train_matrix()
    groups, _ = item_groups_by_train_count(train, num_groups=2)
    # item3 has no training interaction: it must not be bucketed at all
    assert groups[3] == -1
    assert set(groups[:3]) >= {0, 1}


def test_item_groups_invalid_params():
    train = _get_train_matrix()
    with pytest.raises(ValueError):
        item_groups_by_train_count(train, num_groups=2, thresholds=[2])
    with pytest.raises(ValueError):
        item_groups_by_train_count(train, num_groups=0)
    with pytest.raises(ValueError):
        item_groups_by_train_count(train, thresholds=[0])
    with pytest.raises(ValueError):
        item_groups_by_train_count(train, thresholds=[])


def test_ranking_metrics_by_item_group_hand_computed():
    train = _get_train_matrix()
    test = _user_items_matrix(
        {0: [2, 3], 1: [1], 2: [0, 2]},
        shape=(3, 4),
    )
    model = _StaticRecommender({0: [2, 1], 1: [1, 0], 2: [3, 0]})

    result = ranking_metrics_at_k_by_item_group(
        model, train, test, K=2, thresholds=[2], show_progress=False
    )

    assert result["K"] == 2
    overall = result["overall"]
    assert overall["relevant"] == 3
    assert overall["effective_denominator"] == 5  # min(2,2) + min(2,1) + min(2,2)
    assert overall["precision"] == pytest.approx(3 / 5)
    assert overall["num_users"] == 3
    assert not overall["empty"]

    groups = result["groups"]
    assert [g["group"] for g in groups] == [0, 1, "unseen"]
    assert [g["label"] for g in groups] == ["[1, 2)", "[2, inf)", "unseen (no train interactions)"]

    tail, head, unseen = groups

    # group 0 = item2: hit once (user0), users 0 and 2 each hold one label of it
    assert tail["relevant"] == 1
    assert tail["effective_denominator"] == 2
    assert tail["precision"] == pytest.approx(0.5)
    assert tail["test_items"] == 2
    assert tail["num_users"] == 2
    assert tail["num_items"] == 1
    assert not tail["empty"]

    # group 1 = items 0,1: hit for user1 (item1) and user2 (item0)
    assert head["relevant"] == 2
    assert head["effective_denominator"] == 2
    assert head["precision"] == pytest.approx(1.0)
    assert head["test_items"] == 2
    assert head["num_users"] == 2
    assert head["num_items"] == 2
    assert not head["empty"]

    # unseen = item3: user0 holds it as a label, never hit
    assert unseen["relevant"] == 0
    assert unseen["effective_denominator"] == 1
    assert unseen["precision"] == pytest.approx(0.0)
    assert unseen["test_items"] == 1
    assert unseen["num_users"] == 1
    assert unseen["num_items"] == 1
    assert not unseen["empty"]

    # the group hits decompose the overall hit count exactly
    assert sum(g["relevant"] for g in groups) == overall["relevant"]


def test_ranking_metrics_by_item_group_empty_group():
    train = _get_train_matrix()
    test = _user_items_matrix({0: [2], 1: [1]}, shape=(3, 4))
    model = _StaticRecommender({0: [2], 1: [1]})

    # threshold 1 leaves the first bucket without any training item
    result = ranking_metrics_at_k_by_item_group(
        model, train, test, K=2, thresholds=[1, 2], show_progress=False
    )

    empty_group, _, _, _ = result["groups"]
    assert empty_group["label"] == "[1, 1)"
    assert empty_group["num_items"] == 0
    assert empty_group["test_items"] == 0
    assert empty_group["effective_denominator"] == 0
    assert empty_group["empty"] is True
    assert empty_group["precision"] is None

    # non-empty groups still report plain floats
    assert result["groups"][1]["precision"] == pytest.approx(1.0)


def test_ranking_metrics_unseen_items_dont_distort_train_bucketing():
    train = _get_train_matrix()
    # the test set introduces item3 (no train interactions) and item4
    # (a column that doesn't even exist in the train matrix); item3 is as
    # popular in validation as the most popular training item
    test = _user_items_matrix(
        {0: [3, 4], 1: [3], 2: [3, 4]},
        shape=(3, 5),
    )
    model = _StaticRecommender({0: [3, 4], 1: [3, 4], 2: [3, 4]})

    result = ranking_metrics_at_k_by_item_group(
        model, train, test, K=2, num_groups=2, show_progress=False
    )

    # the bucket boundaries are the ones of the training counts [3, 2, 1]
    # (median 2), unaffected by the validation-only items
    assert [g["label"] for g in result["groups"]] == [
        "[1, 2)",
        "[2, inf)",
        "unseen (no train interactions)",
    ]
    # consistent with asking for the grouping directly
    _, labels = item_groups_by_train_count(train, num_groups=2)
    assert [g["label"] for g in result["groups"][:-1]] == labels

    tail, head, unseen = result["groups"]
    assert tail["empty"] is True and tail["precision"] is None
    assert head["empty"] is True and head["precision"] is None

    # both validation-only items (including the train-out-of-range column)
    # land in the unseen group
    assert unseen["num_items"] == 2
    assert unseen["test_items"] == 5
    assert unseen["relevant"] == 5
    assert unseen["effective_denominator"] == 5
    assert unseen["precision"] == pytest.approx(1.0)

    assert result["overall"]["precision"] == pytest.approx(1.0)
    assert result["overall"]["num_users"] == 3


def test_ranking_metrics_by_item_group_matches_overall():
    """With a real model, the grouped metrics must decompose the overall ones."""
    mat = random(100, 100, density=0.5, format="csr", dtype=np.float64, random_state=42)
    train, test = train_test_split(mat, 0.8, random_state=42)

    model = implicit.nearest_neighbours.ItemItemRecommender(K=10)
    model.fit(train)

    K = 10
    grouped = ranking_metrics_at_k_by_item_group(
        model, train, test, K=K, num_groups=3, show_progress=False
    )
    overall = ranking_metrics_at_k(model, train, test, K=K, show_progress=False)

    # same candidate users and labels as ranking_metrics_at_k => same overall precision
    assert grouped["overall"]["precision"] == pytest.approx(overall["precision"])

    groups = grouped["groups"]
    assert len(groups) == 4  # 3 popularity buckets + unseen
    assert groups[-1]["group"] == "unseen"

    # hits and labels decompose exactly over the groups
    assert sum(g["relevant"] for g in groups) == grouped["overall"]["relevant"]
    candidate_users = np.ediff1d(test.indptr) > 0
    assert sum(g["test_items"] for g in groups) == test[candidate_users].nnz
    assert grouped["overall"]["num_users"] == int(candidate_users.sum())

    # every group carries a reviewable denominator and an explicit empty status
    for group in groups:
        assert set(group) == {
            "group",
            "label",
            "precision",
            "relevant",
            "effective_denominator",
            "test_items",
            "num_users",
            "num_items",
            "empty",
        }
        if group["empty"]:
            assert group["precision"] is None
            assert group["effective_denominator"] == 0
            assert group["test_items"] == 0
        else:
            assert group["precision"] == pytest.approx(
                group["relevant"] / group["effective_denominator"]
            )


def test_ranking_metrics_by_item_group_invalid_k():
    train = _get_train_matrix()
    with pytest.raises(ValueError):
        ranking_metrics_at_k_by_item_group(None, train, train, K=0, show_progress=False)
