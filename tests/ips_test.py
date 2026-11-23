import numpy as np
import pytest
from scipy.sparse import csr_matrix

from implicit.evaluation import ImpressionLogError, truncated_ips_at_k


def _log():
    # user 1 impressions: item 10 (p=0.5, reward=1), item 11 (p=0.25, reward=0)
    # user 2 impressions: item 10 (p=1.0, reward=1)
    return (
        np.array([1, 1, 2]),
        np.array([10, 11, 10]),
        np.array([0.5, 0.25, 1.0]),
        np.array([1.0, 0.0, 1.0]),
    )


def test_truncated_ips_score():
    user_ids, item_ids, propensities, rewards = _log()
    result = truncated_ips_at_k(
        user_ids,
        item_ids,
        propensities,
        rewards,
        ranking_user_ids=np.array([1, 2]),
        ranking_item_ids=np.array([[10, 11], [10, 12]]),
    )
    # user 1: item 10 -> reward 1 * weight 2, item 11 -> reward 0 * weight 4
    # user 2: item 10 -> reward 1 * weight 1, item 12 -> no matching impression
    assert result["score"] == pytest.approx(1.5)
    assert result["user_scores"] == pytest.approx([2.0, 1.0])
    assert result["num_users"] == 2
    assert result["num_positions"] == 4
    assert result["num_matched"] == 3
    assert result["num_unsupported"] == 1
    assert result["non_estimable_user_ids"].tolist() == []
    assert result["weight_sum"] == pytest.approx(7.0)
    assert result["effective_sample_size"] == pytest.approx(49.0 / 21.0)
    assert result["max_observed_weight"] == pytest.approx(4.0)
    assert result["num_truncated"] == 0


def test_truncated_ips_truncates_weights():
    result = truncated_ips_at_k(
        np.array([1]),
        np.array([10]),
        np.array([0.01]),
        np.array([1.0]),
        np.array([1]),
        np.array([[10]]),
        max_weight=10.0,
    )
    # 1 / 0.01 = 100 gets truncated to 10
    assert result["score"] == pytest.approx(10.0)
    assert result["weight_sum"] == pytest.approx(10.0)
    assert result["max_observed_weight"] == pytest.approx(10.0)
    assert result["num_truncated"] == 1
    assert result["effective_sample_size"] == pytest.approx(1.0)


def test_truncated_ips_respects_k():
    user_ids, item_ids, propensities, rewards = _log()
    result = truncated_ips_at_k(
        user_ids, item_ids, propensities, rewards, np.array([1]), np.array([[11, 10]]), K=1
    )
    # only position 0 (item 11, reward 0) is evaluated
    assert result["score"] == pytest.approx(0.0)
    assert result["num_positions"] == 1
    assert result["num_matched"] == 1


def test_truncated_ips_log2_discount():
    user_ids, item_ids, propensities, rewards = _log()
    result = truncated_ips_at_k(
        user_ids,
        item_ids,
        propensities,
        rewards,
        np.array([1]),
        np.array([[11, 10]]),
        discount="log2",
    )
    # item 11 at position 0: reward 0; item 10 at position 1: 1 * 2 / log2(3)
    assert result["score"] == pytest.approx(2.0 / np.log2(3))


def test_invalid_propensities_are_locatable():
    with pytest.raises(ImpressionLogError) as excinfo:
        truncated_ips_at_k(
            np.array([1, 1, 2, 2]),
            np.array([10, 11, 10, 12]),
            np.array([0.5, -0.1, 1.5, np.nan]),
            np.array([1.0, 0.0, 1.0, 1.0]),
            np.array([1]),
            np.array([[10]]),
        )
    err = excinfo.value
    assert err.invalid_propensity_indices.tolist() == [1, 2, 3]
    assert err.invalid_reward_indices is None
    assert err.duplicate_indices is None
    assert "1, 2, 3" in str(err)


def test_duplicate_impressions_are_locatable():
    with pytest.raises(ImpressionLogError) as excinfo:
        truncated_ips_at_k(
            np.array([1, 1, 2, 1]),
            np.array([10, 11, 10, 10]),  # records 0 and 3 are both (user 1, item 10)
            np.array([0.5, 0.5, 1.0, 0.5]),
            np.array([1.0, 0.0, 1.0, 0.0]),
            np.array([1]),
            np.array([[10]]),
        )
    err = excinfo.value
    assert err.duplicate_indices.tolist() == [0, 3]
    assert err.invalid_propensity_indices is None
    assert "0, 3" in str(err)


def test_invalid_log_reports_all_problems_at_once():
    with pytest.raises(ImpressionLogError) as excinfo:
        truncated_ips_at_k(
            np.array([1, 1, 1]),
            np.array([10, 10, 11]),
            np.array([0.0, 0.5, 0.5]),
            np.array([1.0, 1.0, np.inf]),
            np.array([1]),
            np.array([[10]]),
        )
    err = excinfo.value
    assert err.invalid_propensity_indices.tolist() == [0]
    assert err.invalid_reward_indices.tolist() == [2]
    assert err.duplicate_indices.tolist() == [0, 1]


def test_non_estimable_users_are_reported():
    user_ids, item_ids, propensities, rewards = _log()
    result = truncated_ips_at_k(
        user_ids,
        item_ids,
        propensities,
        rewards,
        np.array([1, 2, 3]),
        np.array([[10, 11], [12, 13], [14, 15]]),
    )
    # users 2 and 3 have no impressions matching their candidate rankings
    assert result["non_estimable_user_ids"].tolist() == [2, 3]
    assert result["num_unsupported"] == 4
    assert result["score"] == pytest.approx(2.0 / 3)


def test_empty_impression_log():
    result = truncated_ips_at_k(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([]),
        np.array([]),
        np.array([1]),
        np.array([[10, 11]]),
    )
    assert result["score"] == 0.0
    assert result["num_matched"] == 0
    assert result["effective_sample_size"] == 0.0
    assert result["non_estimable_user_ids"].tolist() == [1]


def test_mixed_integer_dtypes():
    result = truncated_ips_at_k(
        np.array([1], dtype=np.int32),
        np.array([10], dtype=np.int32),
        np.array([0.5]),
        np.array([2.0]),
        np.array([1], dtype=np.int64),
        np.array([[10]], dtype=np.int64),
    )
    assert result["score"] == pytest.approx(4.0)


def test_max_weight_must_be_at_least_one():
    user_ids, item_ids, propensities, rewards = _log()
    with pytest.raises(ValueError):
        truncated_ips_at_k(
            user_ids, item_ids, propensities, rewards, np.array([1]), np.array([[10]]),
            max_weight=0.5,
        )


def test_k_out_of_range():
    user_ids, item_ids, propensities, rewards = _log()
    with pytest.raises(ValueError):
        truncated_ips_at_k(
            user_ids, item_ids, propensities, rewards, np.array([1]), np.array([[10, 11]]), K=3
        )


def test_duplicate_ranking_users_are_rejected():
    user_ids, item_ids, propensities, rewards = _log()
    with pytest.raises(ValueError, match="duplicate users"):
        truncated_ips_at_k(
            user_ids, item_ids, propensities, rewards,
            np.array([1, 1]), np.array([[10], [11]]),
        )


def test_repeated_items_in_ranking_are_rejected():
    user_ids, item_ids, propensities, rewards = _log()
    with pytest.raises(ValueError, match="repeated items"):
        truncated_ips_at_k(
            user_ids, item_ids, propensities, rewards, np.array([1]), np.array([[10, 10]])
        )


def test_mismatched_log_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        truncated_ips_at_k(
            np.array([1, 2]),
            np.array([10]),
            np.array([0.5, 0.5]),
            np.array([1.0, 1.0]),
            np.array([1]),
            np.array([[10]]),
        )


def test_with_model_recommendations():
    import implicit

    rng = np.random.RandomState(7)
    play = csr_matrix((rng.random((20, 40)) > 0.7).astype(np.float32))
    model = implicit.als.AlternatingLeastSquares(factors=8, iterations=2, random_state=7)
    model.fit(play)

    users = np.arange(5)
    ids, _ = model.recommend(users, play[users], N=10)

    # fabricate an impression log from the top-3 of each candidate ranking
    log_user_ids = np.repeat(users, 3)
    log_item_ids = ids[:, :3].ravel()
    result = truncated_ips_at_k(
        log_user_ids,
        log_item_ids,
        np.full(len(log_user_ids), 0.5),
        np.ones(len(log_user_ids)),
        users,
        ids,
        K=10,
    )
    assert result["num_matched"] == len(log_user_ids)
    # 3 matched positions per user, each with reward 1 and weight 1 / 0.5
    assert result["score"] == pytest.approx(6.0)
