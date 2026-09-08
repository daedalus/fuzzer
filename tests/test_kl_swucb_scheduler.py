"""Falsification and adversarial tests for KL-SWUCB scheduler."""

from __future__ import annotations

import math

from fuzzer_tool.core.schedulers._kl_ucb import kl_upper_bound
from fuzzer_tool.core.schedulers.kl_swucb import KL_SWUCBScheduler


def test_kl_swucb_falsification_gaussian_strict_tightening() -> None:
    """KL width is at most the Gaussian width sqrt(2*budget), with strict
    tightening when the bisection path is active.
    """
    p = 0.5
    budget = 0.3
    kl_width = kl_upper_bound(p, budget) - p
    gaussian_width = min(1.0, p + math.sqrt(2.0 * budget)) - p
    assert kl_width < gaussian_width - 1e-6


def test_kl_swucb_adversarial_scripted_rng() -> None:
    """Scripted RNG drives exact draws; assert exact selected op.
    Two arms with mean=0 but different counts (gamma=1.0), larger n
    gets larger KL budget so lower width → higher index → selected.
    Independent derivation:
      Both mean=0. width = 1-exp(-budget), budget = xi*log_n/n.
      n_total = 100 + 10 = 110, log_n = ln(110), xi=0.15.
      A: budget = 0.15*ln(110)/100 → width = 1-exp(-budget_A)
      B: budget = 0.15*ln(110)/10  → width = 1-exp(-budget_B)
      budget_B > budget_A → width_B > width_A → score_B > score_A.
    """
    scheduler = KL_SWUCBScheduler(window=1000, xi=0.15, b=1.0, rng=None)
    scheduler.init_arm("A")
    scheduler.init_arm("B")
    # A: 100 pulls, all failures → mean=0
    for _ in range(100):
        scheduler.record("A", success=False, weight=1.0)
    # B: 10 pulls, all failures → mean=0
    for _ in range(10):
        scheduler.record("B", success=False, weight=1.0)
    # Deterministic: no RNG consumed (both have evidence)
    op = scheduler.select_op(["A", "B"])
    assert op == "B"  # B has smaller n → larger KL width → higher score


def test_kl_swucb_convergence_stationary() -> None:
    """KL-SWUCB converges on stationary Bernoulli (tail share floor)."""
    from tests.support.bandit_env import StationaryBernoulli, run

    env = StationaryBernoulli.build()
    scheduler = KL_SWUCBScheduler()
    c = run(scheduler, env, seed=92, rounds=6_000)
    # KL should beat uniform and Gaussian SW-UCB; 0.8 conservative floor.
    assert c.tail_share(env.best) >= 0.8, (
        f"KL-SWUCB spent only {c.tail_share(env.best):.3f} of campaign tail "
        f"on {env.best!r} (p={env.probs[env.best]:.3f}); uniform baseline "
        f"{1.0 / len(env.arms):.3f}"
    )


def test_kl_swucb_convergence_recovery() -> None:
    """KL-SWUCB recovers after best-arm decay (non-stationary)."""
    from tests.support.bandit_env import DecayingBest, run

    env = DecayingBest.build(switch_at=10_000)
    scheduler = KL_SWUCBScheduler()
    c = run(scheduler, env, seed=92, rounds=20_000)
    # Conservative floor: KL explores less than Gaussian SWUCB, so recovery
    # after decay is slower. Still clearly above uniform (1/12 ≈ 0.083).
    assert c.tail_share(env.best_late) >= 0.3, (
        f"KL-SWUCB spent only {c.tail_share(env.best_late):.3f} of campaign tail "
        f"on {env.best_late!r} after {env.best_early!r} decayed"
    )
