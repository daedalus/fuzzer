"""AutoDUCBScheduler: D-UCB whose memory is set by the noise/lag tradeoff."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_ducb_auto import (
    PER_ARM_MAX,
    PER_ARM_MIN,
    AutoDUCBScheduler,
    optimal_memory,
)


def _mse(w: float, sigma2: float, delta2: float) -> float:
    """Discounted-mean error at memory w: noise sigma^2/(2w) + lag^2 delta^2 w^2."""
    return sigma2 / (2.0 * w) + delta2 * w * w


@pytest.mark.parametrize(
    ("sigma2", "delta2"), [(0.04, 1e-8), (0.2, 1e-6), (0.01, 1e-7), (0.25, 4e-9)]
)
def test_optimal_memory_minimises_error(sigma2, delta2) -> None:
    """Falsification: W* beats its 10% neighbours on the error it claims to minimise."""
    w = optimal_memory(sigma2, delta2, lo=1.0, hi=1e9)
    best = _mse(w, sigma2, delta2)
    assert best <= _mse(w * 1.1, sigma2, delta2)
    assert best <= _mse(w / 1.1, sigma2, delta2)


def test_optimal_memory_edges() -> None:
    """Adversarial: no drift -> longest memory; no noise -> shortest; no div-by-zero."""
    assert optimal_memory(0.1, 0.0, lo=8.0, hi=1024.0) == 1024.0
    assert optimal_memory(0.0, 1e-4, lo=8.0, hi=1024.0) == 8.0
    assert optimal_memory(0.0, 0.0, lo=8.0, hi=1024.0) == 1024.0


def _feed(sched: AutoDUCBScheduler, arms: list[str], reward_of, pulls: int) -> None:
    """Round-robin pulls so every arm reaches steady state deterministically."""
    sched.select_op(arms)
    for t in range(pulls):
        arm = arms[t % len(arms)]
        sched.record(arm, True, reward_of(arm, t))


def test_drift_estimate_recovers_ramp_slope() -> None:
    """A noiseless ramp of slope d per own-pull yields delta^2 = d^2.

    1% tolerance: the smoother still holds g's from the slow probe's
    start-up transient, before its lag reached d*(M_s - 1).
    """
    d = 1e-4
    s = AutoDUCBScheduler(rng=RandPool(1))
    _feed(s, ["a"], lambda _a, t: 0.1 + d * t, 3000)

    _sigma2, delta2 = s.drift_stats()
    assert delta2 == pytest.approx(d * d, rel=1e-2)


def _noisy(seed: int, slope: float):
    rng = RandPool(seed)

    def reward(arm: str, t: int) -> float:
        drift = slope * t if arm == "a" else 0.0
        return 0.5 + drift + rng.gauss(0.0, 0.2)

    return reward


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stationary_keeps_long_memory(seed) -> None:
    """Control (Hard Rule 46): pure noise on four seeds -> memory stays long."""
    s = AutoDUCBScheduler(rng=RandPool(seed))
    arms = ["a", "b", "c", "d"]
    _feed(s, arms, _noisy(seed, 0.0), 20_000)

    assert s.per_arm >= PER_ARM_MAX / 4


def test_drift_shortens_memory() -> None:
    """Drifting arm shortens memory below the stationary control's floor."""
    s = AutoDUCBScheduler(rng=RandPool(1))
    arms = ["a", "b", "c", "d"]
    # Slope 1.5e-3 per own pull, 2.25 of drift over 1.5k pulls of "a":
    # g = 0.084 against a null sd of 0.043, so above the resolution limit.
    _feed(s, arms, _noisy(1, 1.5e-3), 6000)

    assert PER_ARM_MIN <= s.per_arm < PER_ARM_MAX / 4


def test_gamma_follows_per_arm() -> None:
    """gamma = 1 - 1/(per_arm * K) after a retune, K = candidate count."""
    s = AutoDUCBScheduler(rng=RandPool(1))
    arms = ["a", "b", "c"]
    _feed(s, arms, _noisy(3, 1.5e-3), 6000)

    assert s.gamma == pytest.approx(1.0 - 1.0 / (s.per_arm * len(arms)))


def test_all_zero_rewards_stay_finite() -> None:
    """Adversarial: zero variance and zero drift never divide by zero."""
    s = AutoDUCBScheduler(rng=RandPool(1))
    _feed(s, ["a", "b"], lambda _a, _t: 0.0, 5000)

    assert s.per_arm == PER_ARM_MAX
    assert s.select_op(["a", "b"]) in ("a", "b")


@pytest.mark.parametrize(
    "kwargs",
    [{"retune_every": 0}, {"fast": 0.5}, {"fast": 64.0, "slow": 8.0}, {"per_arm_min": 0.0}],
)
def test_rejects_bad_params(kwargs) -> None:
    """Adversarial: invalid tuning knobs raise at construction."""
    with pytest.raises(ValueError):
        AutoDUCBScheduler(**kwargs)
