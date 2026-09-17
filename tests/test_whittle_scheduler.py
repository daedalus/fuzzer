"""Falsification tests for the Whittle-index operator scheduler."""

from __future__ import annotations

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_whittle import (
    WhittleIndexScheduler,
    whittle_index_table,
)


def test_index_monotone_in_reward_when_rested() -> None:
    """Rested chain (passive_decay=0): index strictly decreases as state

    (fatigue) increases, when the reward vector itself decreases with
    fatigue. A birth-death chain that rewards freshness should never rank
    a more-fatigued state above a fresher one.
    """
    reward = [0.9, 0.7, 0.5, 0.3, 0.1]
    indices, indexable = whittle_index_table(reward, passive_decay=0.0)
    assert all(indexable), "expected a single crossing at every state for a monotone reward"
    for s in range(len(indices) - 1):
        assert indices[s] >= indices[s + 1] - 1e-9, (
            f"index not monotone: state {s}={indices[s]} < state {s + 1}={indices[s + 1]}"
        )


def test_index_flat_reward_gives_flat_index() -> None:
    """A reward vector with no dependence on state should not manufacture

    a preference between states out of nothing.
    """
    reward = [0.5] * 5
    indices, indexable = whittle_index_table(reward, passive_decay=0.0)
    assert all(indexable)
    for s in range(1, len(indices)):
        assert abs(indices[s] - indices[0]) < 1e-6


def test_passive_decay_zero_is_rested() -> None:
    """With passive_decay=0.0, an idle arm's state never changes."""
    sched = WhittleIndexScheduler(passive_decay=0.0, floor=0.0, rng=RandPool(seed=1))
    ops = ["a", "b"]
    for op in ops:
        sched.init_arm(op)
    sched.record("a", success=True, weight=1.0)
    state_before = sched._state["b"]
    for _ in range(50):
        sched.select_op(ops)
        sched.record("a", success=True, weight=1.0)  # only "a" ever plays
    assert sched._state["b"] == state_before, "idle arm drifted with passive_decay=0.0"


def test_passive_decay_positive_drifts_idle_arms() -> None:
    """With passive_decay=1.0, an idle arm's state advances every round it

    is not selected -- the restless assumption should actually move the
    state, not just exist as an unused parameter.
    """
    sched = WhittleIndexScheduler(
        n_states=5, passive_decay=1.0, floor=0.0, rng=RandPool(seed=2)
    )
    ops = ["a", "b"]
    for op in ops:
        sched.init_arm(op)
    # Force "a" to always be the one played by giving it an overwhelming
    # empirical edge, then drive several rounds.
    for _ in range(20):
        sched.record("a", success=True, weight=1.0)
    for _ in range(6):
        sched.select_op(ops)
        sched.record("a", success=True, weight=1.0)
    assert sched._state["b"] == 4, "idle arm never drifted to max fatigue under passive_decay=1.0"


def test_converges_to_better_arm_stationary() -> None:
    """Over enough rounds, a scheduler with a clear reward gap should

    prefer the better arm on the tail of the run (basic stationary sanity
    check, not a substitute for the real bandit_env.py harness).
    """
    rng = RandPool(seed=7)
    sched = WhittleIndexScheduler(passive_decay=0.0, floor=0.05, rng=rng)
    ops = ["good", "bad"]
    good_p, bad_p = 0.8, 0.2
    picks_tail = []
    n_rounds = 2000
    for i in range(n_rounds):
        op = sched.select_op(ops)
        p = good_p if op == "good" else bad_p
        success = rng.random() < p
        sched.record(op, success=success, weight=1.0)
        if i >= n_rounds - 200:
            picks_tail.append(op)
    good_share = picks_tail.count("good") / len(picks_tail)
    assert good_share > 0.6, f"expected the better arm to dominate the tail, got {good_share:.2f}"


def test_floor_prevents_permanent_starvation() -> None:
    """An arm driven to its most-fatigued state by early bad luck must

    still be selectable again under a nonzero floor.
    """
    sched = WhittleIndexScheduler(n_states=5, passive_decay=0.0, floor=0.2, rng=RandPool(seed=3))
    ops = ["unlucky", "other"]
    for op in ops:
        sched.init_arm(op)
    for _ in range(10):
        sched.record("unlucky", success=False, weight=1.0)
    picks = set()
    for _ in range(200):
        op = sched.select_op(ops)
        picks.add(op)
        sched.record(op, success=(op != "unlucky"), weight=1.0)
    assert "unlucky" in picks, "floor failed to ever revisit the stuck-fatigued arm"


def test_select_op_edge_cases() -> None:
    sched = WhittleIndexScheduler(rng=RandPool(seed=4))
    assert sched.select_op([]) == ""
    assert sched.select_op(["only"]) == "only"


def test_bandit_stats_shape() -> None:
    sched = WhittleIndexScheduler(rng=RandPool(seed=5))
    ops = ["x", "y", "z"]
    for _ in range(10):
        op = sched.select_op(ops)
        sched.record(op, success=True, weight=1.0)
    stats = sched.bandit_stats()
    assert stats["n_arms"] == 3
    assert set(stats["states"]) == set(ops)
    assert set(stats["indices"]) == set(ops)
    assert set(stats["indexable"]) == set(ops)
    assert stats["best_op"] in ops
    assert stats["whittle_pulls"] == 10
