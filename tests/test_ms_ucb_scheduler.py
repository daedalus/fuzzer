"""MultiScaleUCBScheduler: per-arm Lepski choice among memories."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_ms_ucb import MultiScaleUCBScheduler


def _pull(s: MultiScaleUCBScheduler, arm: str, rewards: list[float]) -> None:
    for r in rewards:
        s.record(arm, True, r)


def _noise(seed: int, mu: float, n: int) -> list[float]:
    rng = RandPool(seed)
    return [mu + rng.gauss(0.0, 0.1) for _ in range(n)]


def test_constant_reward_reads_slowest_scale() -> None:
    """Falsification: nothing to track -> the most precise scale is read."""
    s = MultiScaleUCBScheduler(rng=RandPool(1))
    _pull(s, "a", [0.4] * 2000)

    assert s.scale_of("a") == len(s.memories) - 1


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stationary_noise_stays_slow(seed) -> None:
    """Control (Hard Rule 46): pure noise rarely leaves the slowest scale."""
    s = MultiScaleUCBScheduler(rng=RandPool(seed))
    slowest = len(s.memories) - 1
    on_slowest = 0
    rewards = _noise(seed, 0.4, 3000)
    for i, r in enumerate(rewards):
        s.record("a", True, r)
        if i >= 1000 and s.scale_of("a") == slowest:
            on_slowest += 1

    assert on_slowest / 2000 >= 0.9


def test_step_switches_to_fast_scale() -> None:
    """A step 0.1 -> 0.9 is read at a fast scale, ahead of the slow mean."""
    s = MultiScaleUCBScheduler(rng=RandPool(1))
    _pull(s, "a", _noise(1, 0.1, 2000))
    _pull(s, "a", _noise(2, 0.9, 40))

    slowest = len(s.memories) - 1
    assert s.scale_of("a") < slowest
    read = s._arm_mean("a", s._arm_count("a"))
    assert abs(read - 0.9) < abs(s.bank_mean("a", slowest) - 0.9)


def test_readout_count_is_selected_scale_neff() -> None:
    """The UCB count is the effective sample size of the scale being read."""
    s = MultiScaleUCBScheduler(rng=RandPool(1))
    _pull(s, "a", _noise(1, 0.1, 500))
    _pull(s, "a", _noise(2, 0.9, 30))

    j = s.scale_of("a")
    assert s._arm_count("a") == pytest.approx(s.bank_neff("a", j))


def test_prefers_higher_mean_arm() -> None:
    """Equal evidence, higher readout mean wins the index."""
    s = MultiScaleUCBScheduler(rng=RandPool(1))
    _pull(s, "lo", [0.1] * 500)
    _pull(s, "hi", [0.6] * 500)

    assert s.select_op(["lo", "hi"]) == "hi"


def test_all_zero_rewards_stay_finite() -> None:
    """Adversarial: zero variance falls back to the floor, never NaN."""
    s = MultiScaleUCBScheduler(rng=RandPool(1))
    _pull(s, "a", [0.0] * 300)
    _pull(s, "a", [1.0] * 5)

    assert s.select_op(["a", "b"]) == "b"  # unpulled first
    assert 0.0 <= s._arm_mean("a", s._arm_count("a")) <= 1.0


def test_unpulled_and_unknown_arms() -> None:
    """Adversarial: unseen arm has zero count and slowest scale."""
    s = MultiScaleUCBScheduler(rng=RandPool(1))
    s.init_arm("x")
    assert s._arm_count("ghost") == 0.0
    assert s.scale_of("ghost") == len(s.memories) - 1


@pytest.mark.parametrize(
    "kwargs", [{"z": 0.0}, {"base": 0.5}, {"ratio": 1.0}, {"n_scales": 0}, {"exploration": 0.0}]
)
def test_rejects_bad_params(kwargs) -> None:
    """Adversarial: invalid knobs raise at construction."""
    with pytest.raises(ValueError):
        MultiScaleUCBScheduler(**kwargs)
