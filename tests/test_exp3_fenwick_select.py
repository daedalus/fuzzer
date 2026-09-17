"""Exp3Scheduler.select_op samples through per-operator-list Fenwick trees.

The tree must be a drop-in for the linear roulette walk it replaced: same
law, same one ``random()`` per draw, same probabilities handed to record(),
and trees that track ``self.weights`` through records, renormalisation, log
overflow and a changing operator list. ``_select_linear`` is still in the
class (the fallback for duplicate names and degenerate totals), so it serves
as the oracle; Hard Rule 46's control runs it against itself.
"""

import math
import random

import pytest

import fuzzer_tool.core.schedulers.op_exp3 as exp3_mod
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import Exp3Scheduler


def _campaign(select_name, K=40, subsets=3, rounds=4000, seed=0):
    """Drive one scheduler through select/record with rotating op lists."""
    R = random.Random(seed)
    universe = [f"op{i}" for i in range(K)]
    lists = [universe] + [[o for o in universe if R.random() > 0.3] for _ in range(subsets - 1)]
    good = set(R.sample(universe, 4))
    sched = Exp3Scheduler(gamma=0.1, rng=RandPool(seed=seed))
    for o in universe:
        sched.init_arm(o)
    select = getattr(sched, select_name)
    rewards = random.Random(seed + 1)
    picks, probs = [], []
    for t in range(rounds):
        ops = lists[t % subsets]
        op = select(ops)
        picks.append(op)
        probs.append(sched._prob_of(op))
        sched.record(op, op in good and rewards.random() < 0.4)
    return picks, probs, sched


def test_control_linear_matches_itself():
    a = _campaign("_select_linear")
    b = _campaign("_select_linear")
    assert a[0] == b[0] and a[1] == b[1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_draws_match_linear_walk(seed):
    tree_picks, tree_probs, _ = _campaign("select_op", seed=seed)
    lin_picks, lin_probs, _ = _campaign("_select_linear", seed=seed)
    assert tree_picks == lin_picks
    for a, b in zip(tree_probs, lin_probs, strict=True):
        assert math.isclose(a, b, rel_tol=1e-9)


def test_snapshot_law_matches_linear_law():
    _, _, sched = _campaign("select_op", rounds=500)
    ops = [f"op{i}" for i in range(40)]
    sched.select_op(ops)
    tree_law = sched.last_selection_probs()
    lin = Exp3Scheduler(gamma=sched.gamma)
    lin.weights = dict(sched.weights)
    lin._rng = RandPool(seed=0)
    lin._select_linear(ops)
    assert tree_law.keys() == lin.last_selection_probs().keys()
    for op, p in tree_law.items():
        assert math.isclose(p, lin._last_probs[op], rel_tol=1e-9)
    assert math.isclose(sum(tree_law.values()), 1.0, rel_tol=1e-9)


def test_trees_track_weights_after_records():
    _, _, sched = _campaign("select_op", rounds=3000)
    for key in list(sched._trees):
        sched.select_op(list(key))  # syncs that tree
        index, leaf, tree, _e, _p, cap = sched._trees[key]
        assert leaf == [sched.weights[op] for op in key]
        assert math.isclose(sum(tree[i] for i in _roots(len(key))), sum(leaf), rel_tol=1e-12)


def _roots(n):
    out, i = [], n
    while i > 0:
        out.append(i)
        i -= i & -i
    return out


def test_renormalisation_rebuilds_trees():
    """Renormalisation rescales every arm, not just the recorded one, so
    the update log alone cannot bring a tree up to date."""
    sched = Exp3Scheduler(gamma=0.1, window_decay=1.0, rng=RandPool(seed=3))
    ops = ["a", "b", "c"]
    for o in ops:
        sched.init_arm(o)
    sched.select_op(ops)
    renormalised = False
    for _ in range(100):
        before = sched._max_relative
        sched._last_probs = {"a": 1.0}
        sched.record("a", success=True, weight=50.0)
        if sched._max_relative < before:
            renormalised = True
            break
    assert renormalised
    assert sched.weights["b"] < 1.0  # scaled down by the renorm
    sched.select_op(ops)
    assert sched._trees[tuple(ops)][1] == [sched.weights[o] for o in ops]


def test_update_log_is_bounded(monkeypatch):
    monkeypatch.setattr(exp3_mod, "_UPDATE_LOG_MAX", 16)
    sched = Exp3Scheduler(gamma=0.1, rng=RandPool(seed=4))
    ops = [f"op{i}" for i in range(8)]
    for o in ops:
        sched.init_arm(o)
    sched.select_op(ops)  # build the full-list tree, then leave it idle
    for _ in range(200):
        op = sched.select_op(ops[:4])
        sched.record(op, success=True)
    assert len(sched._update_log) <= 16
    sched.select_op(ops)
    assert sched._trees[tuple(ops)][1] == [sched.weights[o] for o in ops]


def test_tree_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(exp3_mod, "_TREE_CACHE_MAX", 4)
    sched = Exp3Scheduler(rng=RandPool(seed=5))
    universe = [f"op{i}" for i in range(12)]
    for n in range(2, 12):
        sched.select_op(universe[:n])
    assert len(sched._trees) == 4


def test_record_probability_for_unoffered_arm_is_one_over_k():
    sched = Exp3Scheduler(rng=RandPool(seed=6))
    sched.select_op(["a", "b", "c", "d"])
    assert sched._prob_of("zzz") == 0.25


def test_duplicate_names_fall_back_to_linear_walk():
    sched = Exp3Scheduler(rng=RandPool(seed=7))
    ops = ["a", "b", "a"]
    assert sched.select_op(ops) in ops
    assert not sched._trees
    # A dict law collapses the repeated name, exactly as the walk always did.
    assert set(sched.last_selection_probs()) == {"a", "b"}


def test_zero_total_falls_back_to_uniform_choice():
    sched = Exp3Scheduler(rng=RandPool(seed=8))
    sched.weights = {"a": 0.0, "b": 0.0}
    sched._invalidate_trees()
    assert sched.select_op(["a", "b"]) in ("a", "b")
    assert sched.last_selection_probs() == {}


def test_assigned_law_overrides_snapshot():
    sched = Exp3Scheduler(rng=RandPool(seed=9))
    sched.select_op(["a", "b"])
    sched._last_probs = {"a": 0.8}
    assert sched._prob_of("a") == 0.8
    assert sched._prob_of("b") == 1.0  # 1 / len(assigned law), as before


def test_one_random_draw_per_select():
    class Counting(RandPool):
        calls = 0

        def random(self):
            type(self).calls += 1
            return super().random()

    sched = Exp3Scheduler(rng=Counting(seed=10))
    ops = [f"op{i}" for i in range(33)]
    for _ in range(50):
        sched.select_op(ops)
    assert Counting.calls == 50
