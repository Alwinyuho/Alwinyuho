import numpy as np
import pytest

from shelfrank.ope import (
    N_ITEMS,
    LoggedData,
    action_distribution,
    direct_method,
    doubly_robust,
    ips,
    snips,
)


def simulated_logs(n=200_000, seed=0):
    """Uniform-random logging over 80 items, with known click probabilities."""
    rng = np.random.default_rng(seed)
    click_prob = np.linspace(0.001, 0.02, N_ITEMS)
    item = rng.integers(0, N_ITEMS, n)
    position = rng.integers(0, 3, n)
    click = (rng.random(n) < click_prob[item]).astype(float)
    data = LoggedData(
        item=item,
        position=position,
        click=click,
        propensity=np.full(n, 1 / N_ITEMS),
        affinity=np.zeros((n, N_ITEMS), dtype=np.float32),
        users=np.zeros((n, 4), dtype=np.int64),
    )
    return data, click_prob


def target_policy():
    """Shows only the best 10 items, uniformly, in every position."""
    target = np.zeros((3, N_ITEMS))
    target[:, -10:] = 0.1
    return target


def test_ips_and_snips_recover_the_true_value():
    data, click_prob = simulated_logs()
    truth = click_prob[-10:].mean()
    assert ips(data, target_policy()).mean() == pytest.approx(truth, rel=0.1)
    assert snips(data, target_policy()) == pytest.approx(truth, rel=0.1)


def test_dr_is_unbiased_even_with_a_bad_reward_model():
    data, click_prob = simulated_logs()
    truth = click_prob[-10:].mean()
    wrong_model = np.full((len(data.click), N_ITEMS), 0.05, dtype=np.float32)
    assert direct_method(wrong_model, data, target_policy()).mean() == pytest.approx(0.05)
    assert doubly_robust(wrong_model, data, target_policy()).mean() == pytest.approx(truth, rel=0.1)


def test_action_distribution_rows_sum_to_one():
    dist = action_distribution(np.array([0, 1, 1, 79]), np.array([0, 0, 1, 2]))
    assert dist.shape == (3, N_ITEMS)
    assert np.allclose(dist.sum(axis=1), 1.0)
    assert dist[0, 1] == pytest.approx(0.5)
