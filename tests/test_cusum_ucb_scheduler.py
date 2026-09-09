"""Falsification and adversarial tests for CUSUM-UCB scheduler."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.schedulers.cusum_ucb import CUSUM_UCBScheduler


def test_cusum_ucb_falsification_width_decreases_with_n() -> None:
    """UCB1-style width strictly decreases as n grows (fixed log_n).

    The post-reset index is mean + b*sqrt(xi*log_n/n). If the denominator
    were ever constant or growing, an arm with more evidence since the
    last reset would get a *wider* confidence band than a less-pulled one
    -- the scheduler would over-explore forever instead of settling.
    """
    s = CUSUM_UCBScheduler(xi=0.6, b=1.0)
    log_n = math.log(1000.0)

    def width(n: float) -> float:
        return s._width(mean=0.0, n=n, log_n=log_n)

    for n in (1.0, 10.0, 100.0, 1000.0):
        w = width(n)
        assert math.isfinite(w) and w > 0.0, f"width non-positive at n={n}"
    assert width(1.0) > width(10.0) > width(100.0) > width(1000.0) - 1e-12


def test_cusum_ucb_validates_constructor_params() -> None:
    """Every paper-native parameter must reject a non-positive value.

    A zero or negative m/epsilon/h/xi silently breaks the algorithm
    (division by zero in the width, a CUSUM statistic that never resets
    or resets on every pull) rather than raising -- catch it at
    construction instead.
    """
    with pytest.raises(ValueError):
        CUSUM_UCBScheduler(m=0)
    with pytest.raises(ValueError):
        CUSUM_UCBScheduler(epsilon=0.0)
    with pytest.raises(ValueError):
        CUSUM_UCBScheduler(h=0.0)
    with pytest.raises(ValueError):
        CUSUM_UCBScheduler(xi=0.0)


def test_cusum_ucb_no_cusum_statistic_before_warmup_completes() -> None:
    """The CUSUM test must not run until an arm has > m pulls since reset.

    Feed maximally adversarial alternating 0/1 rewards through the warmup
    window itself: if the CUSUM statistic were live during warmup, this
    pattern would trigger a spurious reset immediately. It must not --
    the baseline mean isn't even frozen yet.
    """
    s = CUSUM_UCBScheduler(m=10, epsilon=0.05, h=0.5, xi=0.6)
    s.init_arm("A")
    for i in range(10):
        s.record("A", success=(i % 2 == 0))
    assert s.bandit_stats()["cusum_resets"] == 0
    assert s._mu0["A"] is not None  # frozen exactly at n == m
    assert s._g_pos["A"] == 0.0 and s._g_neg["A"] == 0.0


def test_cusum_ucb_freezes_baseline_at_m_and_does_not_drift() -> None:
    """mu0 is fixed at n == m and must not keep tracking the running mean.

    If mu0 kept updating past the warmup window, a slow drift would never
    accumulate enough CUSUM slope to be detected -- the whole point of
    freezing it is to give drift something fixed to be measured against.
    """
    s = CUSUM_UCBScheduler(m=5, epsilon=0.1, h=100.0, xi=0.6)
    s.init_arm("A")
    for _ in range(5):
        s.record("A", success=False)  # mean 0.0, frozen here
    mu0_at_freeze = s._mu0["A"]
    assert mu0_at_freeze == pytest.approx(0.0)
    for _ in range(20):
        s.record("A", success=True)  # running mean now climbing toward 1.0
    assert s._mu0["A"] == pytest.approx(mu0_at_freeze), (
        "mu0 drifted after the warmup window instead of staying frozen"
    )


def test_cusum_ucb_adversarial_detects_exact_pull_of_upward_shift() -> None:
    """Deterministic detection timing, derived independently.

    m=3, epsilon=0.1, h=1.0. Three zero-reward pulls freeze mu0=0 at n=3
    (no CUSUM check fires on the freezing pull itself). Every pull after
    that returns reward=1.0, so s_pos = 1 - 0 - 0.1 = 0.9 each time and
    g_pos accumulates with no decay (rewards never dip below mu0+epsilon):
        n=4: g_pos = 0.9         (<=  h, no reset)
        n=5: g_pos = 1.8         (>   h, reset fires on this pull)
    So the reset must land on exactly the 2nd post-warmup pull, and the
    scheduler's own counts must show the reset (n back to 0) immediately
    after.
    """
    s = CUSUM_UCBScheduler(m=3, epsilon=0.1, h=1.0, xi=0.6)
    s.init_arm("A")
    for _ in range(3):
        s.record("A", success=False)
    assert s.bandit_stats()["cusum_resets"] == 0

    s.record("A", success=True)  # n=4, g_pos=0.9, below threshold
    assert s.bandit_stats()["cusum_resets"] == 0
    assert s._counts["A"] == 4

    s.record("A", success=True)  # n=5, g_pos=1.8 > h -> reset
    assert s.bandit_stats()["cusum_resets"] == 1
    assert s._counts["A"] == 0, "arm's own count must be zeroed by its own detection"


def test_cusum_ucb_global_reset_clears_every_arm_not_just_the_detector() -> None:
    """A detection on one arm must reset every registered arm's state.

    This is the module's central, deliberate design choice (see the class
    docstring): the paper's assumption is that a change is environmental,
    not arm-local, so B's still-in-warmup statistics -- collected before A
    ever triggered anything -- must be wiped too, not left stale.
    """
    s = CUSUM_UCBScheduler(m=3, epsilon=0.1, h=1.0, xi=0.6)
    s.init_arm("A")
    s.init_arm("B")
    for _ in range(3):
        s.record("A", success=False)
    for _ in range(2):
        s.record("B", success=True)  # below m; B's baseline never freezes
    assert s._counts["B"] == 2 and s._sums["B"] == pytest.approx(2.0)

    s.record("A", success=True)  # n=4
    s.record("A", success=True)  # n=5 -> triggers global reset

    assert s.bandit_stats()["cusum_resets"] == 1
    assert s._counts["B"] == 0
    assert s._sums["B"] == 0.0
    assert s._mu0["B"] is None


def test_cusum_ucb_falsification_no_reset_when_reward_never_leaves_margin() -> None:
    """Noise strictly inside +/-epsilon of the baseline must never trigger.

    Every s_pos/s_neg term is <= 0 by construction here, so g_pos/g_neg
    can only decay toward zero (the max(0, ...) floor), never climb. A
    scheduler that reset anyway would mean the margin subtraction was
    dropped or inverted somewhere in the update.
    """
    s = CUSUM_UCBScheduler(m=10, epsilon=0.2, h=0.01, xi=0.6)
    s.init_arm("A")
    for _ in range(10):
        s.record("A", success=False)  # mu0 = 0.0
    mu0 = s._mu0["A"]
    assert mu0 == pytest.approx(0.0)

    # Rewards sit exactly at the boundary: reward - mu0 - epsilon == 0,
    # and mu0 - reward - epsilon == -2*epsilon < 0. Both slopes are <= 0.
    for i in range(500):
        s.record("A", success=(i % 2 == 0), weight=0.2)
        assert s.bandit_stats()["cusum_resets"] == 0, f"spurious reset at pull {i}"


def test_cusum_ucb_total_pulls_survive_resets_but_reset_relative_n_does_not() -> None:
    """bandit_stats separates lifetime pulls from pulls-since-last-reset.

    UCBBase's own ``_total_pulls`` counts every ``record()`` call ever,
    across resets -- it must keep climbing. ``cusum_pulls_since_reset``
    (the ``t - tau`` the UCB width is actually computed from) must drop
    back down when a reset fires, or the exploration width would stay
    permanently tight after the first detection instead of re-opening.
    """
    s = CUSUM_UCBScheduler(m=3, epsilon=0.1, h=1.0, xi=0.6)
    s.init_arm("A")
    for _ in range(3):
        s.record("A", success=False)
    for _ in range(2):
        s.record("A", success=True)  # the 2nd of these triggers a reset

    stats = s.bandit_stats()
    assert stats["cusum_resets"] == 1
    assert stats["cusum_pulls"] == 5, "lifetime pull count must survive the reset"
    assert stats["cusum_pulls_since_reset"] == 0, (
        "pulls-since-reset must drop to 0 immediately after a reset fires"
    )


def test_cusum_ucb_bandit_stats_shape() -> None:
    s = CUSUM_UCBScheduler()
    s.init_arm("A")
    s.record("A", success=True)
    stats = s.bandit_stats()
    for key in (
        "cusum_pulls",
        "cusum_resets",
        "cusum_arms",
        "cusum_pulls_since_reset",
        "cusum_log_n",
    ):
        assert key in stats, f"missing diagnostic key {key!r}"


def test_cusum_ucb_convergence_stationary() -> None:
    """Locates the best arm on a fixed-seed stationary campaign.

    Floor of 0.90 sits comfortably below the measured 0.983 mean over 20
    seeds documented in the class docstring, leaving margin for a single
    fixed-seed run without weakening the assertion into meaninglessness.
    """
    from tests.support.bandit_env import StationaryBernoulli, run

    env = StationaryBernoulli.build()
    c = run(CUSUM_UCBScheduler(), env, seed=92, rounds=20_000)
    assert c.tail_share(env.best) >= 0.90, (
        f"CUSUM-UCB spent only {c.tail_share(env.best):.3f} of campaign tail "
        f"on {env.best!r} (p={env.probs[env.best]:.3f}); uniform baseline "
        f"{1.0 / len(env.arms):.3f}"
    )


@pytest.mark.slow
def test_cusum_ucb_convergence_recovery_after_decay() -> None:
    """Detects the best-arm collapse and recovers (non-stationary).

    Floor of 0.5 sits below the measured 0.890 mean over 20 seeds
    documented in the class docstring; every one of those 20 seeds also
    detected the collapse at all (>=1 reset), which this test checks
    directly rather than inferring it from the tail share alone.
    """
    from tests.support.bandit_env import DecayingBest, run

    env = DecayingBest.build(switch_at=10_000)
    scheduler = CUSUM_UCBScheduler()
    c = run(scheduler, env, seed=92, rounds=20_000)
    assert scheduler.bandit_stats()["cusum_resets"] >= 1, (
        "CUSUM-UCB never detected the best-arm collapse"
    )
    assert c.tail_share(env.best_late) >= 0.5, (
        f"CUSUM-UCB spent only {c.tail_share(env.best_late):.3f} of campaign tail "
        f"on {env.best_late!r} after {env.best_early!r} decayed"
    )
