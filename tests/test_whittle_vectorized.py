"""Vectorized ``_value_iterate`` vs the scalar value-iteration oracle."""

import random

import numpy as np
import pytest

from fuzzer_tool.core.schedulers import op_whittle as W

ATOL = 1e-12
SEEDS = range(6)


def _vi_scalar(m, P_active, P_passive, reward, gamma, n_iters=W._VALUE_ITERS, v0=None):
    K = len(reward)
    V = list(v0) if v0 is not None else [0.0] * K
    for _ in range(n_iters):
        newV = [0.0] * K
        for s in range(K):
            q_active = reward[s] + gamma * sum(P_active[s][sp] * V[sp] for sp in range(K))
            q_passive = m + gamma * sum(P_passive[s][sp] * V[sp] for sp in range(K))
            newV[s] = max(q_active, q_passive)
        V = newV
    return V


def _problem(seed, K=5, decay=0.1):
    rng = random.Random(seed)
    reward = [rng.uniform(-1.0, 1.0) for _ in range(K)]
    return reward, W._active_kernel(K, reward), W._passive_kernel(K, decay)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("n_iters", [8, 40])
def test_matches_scalar(seed, n_iters):
    reward, Pa, Pp = _problem(seed)
    v0 = [0.1 * i for i in range(len(reward))]
    got = W._value_iterate(0.3, Pa, Pp, reward, 0.95, n_iters=n_iters, v0=v0)
    want = _vi_scalar(0.3, Pa, Pp, reward, 0.95, n_iters=n_iters, v0=v0)
    np.testing.assert_allclose(got, want, atol=ATOL)
    assert isinstance(got, list)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("decay", [0.0, 0.3, 1.0])
def test_table_matches_scalar(seed, decay, monkeypatch):
    reward, _, _ = _problem(seed)
    got = W.whittle_index_table(reward, decay)
    monkeypatch.setattr(W, "_value_iterate", _vi_scalar)
    assert got == W.whittle_index_table(reward, decay)


def test_falsification_wrong_oracle_differs():
    reward, Pa, Pp = _problem(1)
    got = W._value_iterate(0.3, Pa, Pp, reward, 0.95, n_iters=40)
    wrong = _vi_scalar(0.3, Pa, Pp, reward, 0.5, n_iters=40)
    assert not np.allclose(got, wrong, atol=1e-6)


def test_cold_start_does_not_alias_v0():
    reward, Pa, Pp = _problem(2)
    v0 = [0.5] * len(reward)
    W._value_iterate(0.3, Pa, Pp, reward, 0.95, n_iters=8, v0=v0)
    assert v0 == [0.5] * len(reward)


@pytest.mark.parametrize(
    ("K", "m", "gamma", "n_iters"),
    [
        (1, 0.0, 0.95, 8),  # single state
        (5, 0.3, 0.0, 8),  # no discounting: V = max(reward, m)
        (5, 0.3, 0.95, 0),  # zero sweeps: returns v0 / zeros
        (64, 0.3, 0.95, 8),  # wide
        (5, 1e6, 0.95, 40),  # subsidy dwarfs reward
        (5, -1e6, 0.95, 40),
    ],
)
def test_adversarial(K, m, gamma, n_iters):
    reward, Pa, Pp = _problem(3, K=K)
    got = W._value_iterate(m, Pa, Pp, reward, gamma, n_iters=n_iters)
    want = _vi_scalar(m, Pa, Pp, reward, gamma, n_iters=n_iters)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=ATOL)
    assert np.isfinite(got).all()


def test_adversarial_tied_rewards_same_indices(monkeypatch):
    reward = [0.5] * 5
    got = W.whittle_index_table(reward, 0.0)
    monkeypatch.setattr(W, "_value_iterate", _vi_scalar)
    assert got == W.whittle_index_table(reward, 0.0)


def test_accepts_ndarray_kernels():
    reward, Pa, Pp = _problem(4)
    got = W._value_iterate(0.3, np.asarray(Pa), np.asarray(Pp), reward, 0.95, n_iters=8)
    want = _vi_scalar(0.3, Pa, Pp, reward, 0.95, n_iters=8)
    np.testing.assert_allclose(got, want, atol=ATOL)
