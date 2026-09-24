"""Pull-indexed environments in tests/support/bandit_env.py (non_ucb step 2).

RottingArms decays an arm by its *own* pulls, so it separates rested
forgetters from round-indexed ones, which DecayingBest cannot.
Fatigue150 reconstructs op_consolidated.py's 150-arm environment.
"""

from __future__ import annotations

import pytest

from tests.support.bandit_env import (
    DecayingBest,
    Fatigue150,
    RottingArms,
    StationaryBernoulli,
    build_arms,
    run,
)


class _Fixed:
    """Scheduler stub that always picks one arm."""

    def __init__(self, arm: str):
        self.arm = arm
        self.records: list[tuple[str, bool]] = []

    def select_op(self, _ops):
        return self.arm

    def record(self, name, success, weight=1.0):
        self.records.append((name, success))


class _RoundRobin(_Fixed):
    def __init__(self, arms):
        super().__init__(arms[0])
        self._arms = arms
        self._i = 0

    def select_op(self, _ops):
        arm = self._arms[self._i % len(self._arms)]
        self._i += 1
        return arm


# ---------------------------------------------------------------------------
# RottingArms
# ---------------------------------------------------------------------------


class TestRottingArms:
    def test_decay_is_indexed_by_own_pulls(self):
        env = RottingArms.build(n_arms=6, rho=0.5, p_floor=0.0)
        a = env.best_early
        p0 = env.p(a, 0)
        for _ in range(3):
            env.observe(a, False, 0)
        assert env.p(a, 0) == pytest.approx(p0 * 0.5**3)

    def test_unpulled_arm_does_not_decay_with_time(self):
        """Falsification: a round-indexed forgetter would move this."""
        env = RottingArms.build(n_arms=6)
        a = env.best_early
        assert env.p(a, 10**9) == env.p(a, 0)

    def test_best_switches_after_rotting(self):
        env = RottingArms.build(n_arms=6, p_high=0.30, p_runner_up=0.18, rho=0.5, p_floor=0.0)
        assert env.best_at(0) == env.best_early
        env.observe(env.best_early, True, 0)  # 0.30 -> 0.15 < 0.18
        assert env.best_at(0) == env.best_late
        assert env.p_max(0) == pytest.approx(0.18)

    def test_floor_is_respected(self):
        """Adversarial: thousands of pulls never go below p_floor."""
        env = RottingArms.build(n_arms=6, rho=0.5, p_floor=0.01)
        for _ in range(5000):
            env.observe(env.best_early, False, 0)
        assert env.p(env.best_early, 0) == pytest.approx(0.01)

    def test_reset_restores_initial_state(self):
        env = RottingArms.build(n_arms=6)
        p0 = env.p(env.best_early, 0)
        env.observe(env.best_early, True, 0)
        env.reset()
        assert env.p(env.best_early, 0) == p0


# ---------------------------------------------------------------------------
# Fatigue150
# ---------------------------------------------------------------------------


class TestFatigue150:
    def test_shape(self):
        env = Fatigue150.build()
        assert len(env.arms) == 150
        assert len(set(env.arms)) == 150

    def test_heavy_tailed_yields(self):
        env = Fatigue150.build()
        ps = sorted((env.p(a, 0) for a in env.arms if a not in env.locked), reverse=True)
        median = ps[len(ps) // 2]
        assert ps[0] >= 10 * median  # rare, dominant yields

    def test_fatigue_on_success_only(self):
        env = Fatigue150.build(fatigue=0.5)
        a = env.best_at(0)
        p0 = env.p(a, 0)
        env.observe(a, False, 0)
        assert env.p(a, 0) == p0  # failures do not fatigue
        env.observe(a, True, 0)
        assert env.p(a, 0) == pytest.approx(p0 * 0.5)

    def test_periodic_unlocks(self):
        env = Fatigue150.build(unlock_every=100, p_unlock=0.4)
        first, second = env.locked[0], env.locked[1]
        assert env.p(first, 99) == 0.0
        assert env.p(first, 100) == pytest.approx(0.4)
        assert env.p(second, 199) == 0.0
        assert env.p(second, 200) == pytest.approx(0.4)
        assert env.best_at(100) == first

    def test_unlocked_arm_fatigues_from_unlock_value(self):
        env = Fatigue150.build(unlock_every=100, p_unlock=0.4, fatigue=0.5)
        first = env.locked[0]
        env.observe(first, True, 100)
        assert env.p(first, 100) == pytest.approx(0.2)

    def test_locked_arm_success_cannot_happen(self):
        """Adversarial: a locked arm's p is exactly 0 until its unlock."""
        env = Fatigue150.build(unlock_every=1000)
        assert all(env.p(a, 0) == 0.0 for a in env.locked)


# ---------------------------------------------------------------------------
# run() integration
# ---------------------------------------------------------------------------


def test_run_drives_observe_and_regret():
    env = RottingArms.build(n_arms=6, rho=0.5, p_floor=0.0)
    a = env.best_early
    p0 = env.p(a, 0)
    c = run(_Fixed(a), env, seed=1, rounds=4)
    # Pulls 0..3 see p0 * 0.5**k; env is reset before the run.
    expected = sum(max(env.p_runner_up, p0 * 0.5**k) - p0 * 0.5**k for k in range(4))
    assert c.regret == pytest.approx(expected)


@pytest.mark.parametrize("build", [RottingArms.build, Fatigue150.build])
def test_run_is_repeatable_on_same_env(build):
    """Control: two runs on one stateful env must agree (reset works)."""
    env = build()
    arms = env.arms
    c1 = run(_RoundRobin(arms), env, seed=7, rounds=2000)
    c2 = run(_RoundRobin(arms), env, seed=7, rounds=2000)
    assert c1.regret == c2.regret
    assert c1.successes == c2.successes


@pytest.mark.parametrize("build", [StationaryBernoulli.build, DecayingBest.build])
def test_stateless_envs_unchanged(build):
    """Falsification: envs without observe/reset still run."""
    env = build()
    c = run(_Fixed(env.arms[0]), env, seed=3, rounds=100)
    assert c.rounds == 100


def test_build_arms_small_sets_unchanged():
    """Adding categories must not leak into arm sets drawn from fewer."""
    from fuzzer_tool.core.operator_categories import category_of

    first_six = {"bit", "byte", "block", "dict", "structural", "radamsa"}
    assert {category_of(a) for a in build_arms(97, n_categories=6)} <= first_six
    assert len(set(build_arms(150, n_categories=9))) == 150
