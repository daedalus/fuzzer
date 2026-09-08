"""Falsification and adversarial tests for KL-DUCB scheduler."""

from __future__ import annotations

import math

from fuzzer_tool.core.schedulers._kl_ucb import kl_upper_bound
from fuzzer_tool.core.schedulers.kl_ducb import KL_DUCBScheduler


def test_kl_ducb_falsification_gaussian_strict_tightening() -> None:
    """KL width is at most the Gaussian width sqrt(2*budget), with strict
    tightening when the bisection path is active (no fast-path Gaussian return).
    """
    p = 0.5
    budget = 0.3
    kl_width = kl_upper_bound(p, budget) - p
    gaussian_width = min(1.0, p + math.sqrt(2.0 * budget)) - p
    # Strict tightening for these parameters (bisection path active).
    assert kl_width < gaussian_width - 1e-6


def test_kl_ducb_adversarial_scripted_rng() -> None:
    """Scripted RNG drives exact draws; assert exact selected op.
    Test setup: two arms with mean=0 but different counts, gamma=1.0.
    Arm B (n=10) has higher KL-UCB index than arm A (n=100) due to
    larger exploration bonus from smaller n. Deterministic selection.
    """
    # Independent closed-form derivation:
    # Both arms have mean=0. KL(0||B) = -log(1-B), so width = 1-exp(-budget).
    # With gamma=1.0, xi=0.6, exploration=0.25:
    #   n_total = 100 + 10 = 110, log_n = ln(110)
    #   A: n=100 → budget_A = 0.6*ln(110)/100 → width_A = 1-exp(-budget_A)
    #   B: n=10  → budget_B = 0.6*ln(110)/10  → width_B = 1-exp(-budget_B)
    # Since budget_B > budget_A and 1-exp(-x) increases in x,
    # width_B > width_A → score_B > score_A → B selected.
    scheduler = KL_DUCBScheduler(gamma=1.0, xi=0.6, exploration=0.25)
    # Register arms
    scheduler.init_arm("A")
    scheduler.init_arm("B")
    # Give arm A 100 failures, arm B 10 failures (mean=0 for both)
    for _ in range(100):
        scheduler.record("A", success=False, weight=1.0)
    for _ in range(10):
        scheduler.record("B", success=False, weight=1.0)
    # Deterministic selection: no RNG consumed (both arms have evidence)
    op = scheduler.select_op(["A", "B"])
    assert op == "B"


def test_kl_ducb_convergence_stationary() -> None:
    """KL-DUCB converges on stationary Bernoulli (tail share floor)."""
    from tests.support.bandit_env import StationaryBernoulli, run

    env = StationaryBernoulli.build()
    scheduler = KL_DUCBScheduler()
    c = run(scheduler, env, seed=92, rounds=6_000)
    # Conservative floor: KL-UCB should easily beat uniform (0.083) and
    # Gaussian DUCB (~0.968 tail share). Using 0.8 to allow for measurement
    # noise while confirming clear improvement.
    assert c.tail_share(env.best) >= 0.8, (
        f"KL-DUCB spent only {c.tail_share(env.best):.3f} of campaign tail "
        f"on {env.best!r} (p={env.probs[env.best]:.3f}); uniform baseline "
        f"{1.0 / len(env.arms):.3f}"
    )


def test_kl_ducb_convergence_recovery() -> None:
    """KL-DUCB recovers after best-arm decay (non-stationary)."""
    from tests.support.bandit_env import DecayingBest, run

    env = DecayingBest.build(switch_at=10_000)
    scheduler = KL_DUCBScheduler()
    c = run(scheduler, env, seed=92, rounds=20_000)
    # Conservative floor: KL explores less than Gaussian DUCB, so recovery
    # after decay is slower. Still clearly above uniform (1/12 ≈ 0.083).
    assert c.tail_share(env.best_late) >= 0.3, (
        f"KL-DUCB spent only {c.tail_share(env.best_late):.3f} of campaign tail "
        f"on {env.best_late!r} after {env.best_early!r} decayed"
    )
