"""ScaleBank: per-arm discounted estimators at several memories."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.schedulers._multiscale import ScaleBank


def _discounted(rewards: list[float], memory: float) -> tuple[float, float, float]:
    """Reference sums (n, n2, x) for one scale, by direct summation."""
    lam = 1.0 - 1.0 / memory
    t = len(rewards)
    w = [lam ** (t - 1 - s) for s in range(t)]
    return sum(w), sum(v * v for v in w), sum(wi * r for wi, r in zip(w, rewards, strict=True))


def test_mean_and_neff_match_direct_sums() -> None:
    """Falsification: recursive sums equal the closed-form weighted sums."""
    rewards = [1.0, 0.0, 0.5, 1.0, 0.0, 0.25]
    bank = ScaleBank((2.0, 5.0))
    for r in rewards:
        bank.update("a", r)

    for j, memory in enumerate((2.0, 5.0)):
        n, n2, x = _discounted(rewards, memory)
        assert bank.mean("a", j) == pytest.approx(x / n)
        assert bank.n_eff("a", j) == pytest.approx(n * n / n2)


def test_neff_saturates_at_steady_state() -> None:
    """Long-run effective sample count is (1+lam)/(1-lam) = 2M-1."""
    memory = 16.0
    bank = ScaleBank((memory,))
    for _ in range(2000):
        bank.update("a", 0.3)

    assert bank.n_eff("a", 0) == pytest.approx(2 * memory - 1, rel=1e-6)


def test_noise_var_ignores_linear_drift() -> None:
    """Von Neumann estimate: a noiseless ramp has noise variance d^2/2, not M^2 d^2."""
    delta = 0.001
    bank = ScaleBank((8.0, 64.0))
    for t in range(1000):
        bank.update("a", 0.1 + delta * t)

    # E[(r_t - r_{t-1})^2] / 2 = delta^2 / 2 for a pure ramp.
    assert bank.noise_var("a") == pytest.approx(delta * delta / 2.0, rel=1e-6)


def test_unknown_arm_reads_empty() -> None:
    """Adversarial: never-updated arm reads zero evidence, no KeyError."""
    bank = ScaleBank((4.0,))
    assert bank.mean("ghost", 0) == 0.0
    assert bank.n_eff("ghost", 0) == 0.0
    assert bank.noise_var("ghost") == 0.0
    assert bank.pulls("ghost") == 0


def test_single_pull_has_no_noise_estimate() -> None:
    """One pull gives no successive difference, so noise variance is 0, not NaN."""
    bank = ScaleBank((4.0,))
    bank.update("a", 1.0)
    assert bank.noise_var("a") == 0.0
    assert math.isfinite(bank.mean("a", 0))


@pytest.mark.parametrize("memories", [(), (0.5,), (4.0, 2.0), (4.0, 4.0)])
def test_rejects_bad_memories(memories) -> None:
    """Adversarial: memories must be non-empty, >= 1 and strictly increasing."""
    with pytest.raises(ValueError):
        ScaleBank(memories)
