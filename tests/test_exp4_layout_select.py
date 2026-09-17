"""Exp4Scheduler.select_op draws through a cached per-list layout.

The layout path must be a drop-in for the explicit K x E mixture it replaced:
identical picks, identical ``_last_p_arm``/``_last_expert_xi``/``_last_q``
(exact float equality, not isclose — the arithmetic is reproduced term for
term), one ``random()`` per draw, and correct fallbacks. ``_select_linear``
is the reference implementation and serves as the oracle; the control test
runs it against itself.
"""

import random

import pytest

import fuzzer_tool.core.schedulers.op_exp4 as exp4_mod
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_exp4 import Exp4Scheduler

_ALL = [op for cat in sorted(OPERATOR_CATEGORIES) for op in sorted(OPERATOR_CATEGORIES[cat])]


def _lists(seed, subsets):
    R = random.Random(seed)
    lists = [_ALL]
    for _ in range(subsets - 1):
        keep = R.uniform(0.2, 0.9)
        lists.append([o for o in _ALL if R.random() < keep])
    # A registry-unknown name lands in the uncategorized expert.
    lists.append(lists[-1] + ["not_a_registered_op"])
    return lists


def _campaign(select_name, seed=0, subsets=4, rounds=3000, gamma=0.1):
    lists = _lists(seed, subsets)
    R = random.Random(seed + 7)
    good = set(R.sample(_ALL, 12))
    sched = Exp4Scheduler(gamma=gamma, rng=RandPool(seed=seed))
    select = getattr(sched, select_name)
    rewards = random.Random(seed + 1)
    trace = []
    for t in range(rounds):
        ops = lists[(t * 7 + t // 13) % len(lists)]
        op = select(ops)
        trace.append((op, sched._last_p_arm, dict(sched._last_expert_xi), dict(sched._last_q)))
        sched.record(op, op in good and rewards.random() < 0.5, rewards.random())
    return trace, sched


def test_control_linear_matches_itself():
    assert _campaign("_select_linear")[0] == _campaign("_select_linear")[0]


@pytest.mark.parametrize(("seed", "gamma"), [(0, 0.1), (1, 0.0), (2, 0.5), (3, 1.0)])
def test_layout_path_is_bit_identical_to_linear(seed, gamma):
    fast, fs = _campaign("select_op", seed=seed, gamma=gamma)
    slow, ss = _campaign("_select_linear", seed=seed, gamma=gamma)
    assert fast == slow
    assert fs.weights == ss.weights


def test_weights_actually_moved():
    """Guard the equivalence test against a campaign that never learns."""
    _, sched = _campaign("select_op", rounds=1500)
    assert len(set(sched.weights.values())) > 2


def test_renormalisation_path_stays_identical():
    """Large rewards drive max weight past 1e9 and trigger the rescale."""

    def run(name):
        sched = Exp4Scheduler(gamma=0.9, rng=RandPool(seed=5))
        select = getattr(sched, name)
        ops = _lists(5, 2)[1]
        out = []
        for _ in range(400):
            op = select(ops)
            out.append((op, sched._last_p_arm))
            sched.record(op, True, 1.0)
        return out, max(sched.weights.values())

    fast, wmax = run("select_op")
    assert fast == run("_select_linear")[0]
    assert wmax <= 1e9


def test_one_random_per_draw():
    class Counting(RandPool):
        calls = 0

        def random(self):
            Counting.calls += 1
            return super().random()

    sched = Exp4Scheduler(rng=Counting(seed=0))
    for _ in range(50):
        sched.select_op(_ALL)
    assert Counting.calls == 50


def test_repeated_names_use_linear_path(monkeypatch):
    sched = Exp4Scheduler(rng=RandPool(seed=0))
    ops = _ALL[:10] + _ALL[:3]
    used = []
    real = sched._select_linear
    monkeypatch.setattr(sched, "_select_linear", lambda o: used.append(1) or real(o))
    for _ in range(5):
        assert sched.select_op(ops) in ops
    assert len(used) == 5
    assert sched._layouts[tuple(ops)] is None


def test_non_positive_total_uses_linear_path(monkeypatch):
    sched = Exp4Scheduler(rng=RandPool(seed=0))
    ops = _ALL[:20]
    sched.select_op(ops)
    for k in sched.weights:
        sched.weights[k] = 0.0
    sched.gamma = 0.0
    used = []
    real = sched._select_linear
    monkeypatch.setattr(sched, "_select_linear", lambda o: used.append(1) or real(o))
    sched.select_op(ops)
    assert used == [1]


@pytest.mark.parametrize("ops", [[], ["byte_flip"]])
def test_trivial_lists_match_linear(ops):
    a = Exp4Scheduler(rng=RandPool(seed=0))
    b = Exp4Scheduler(rng=RandPool(seed=0))
    assert a.select_op(ops) == b._select_linear(ops)
    assert (a._last_p_arm, a._last_expert_xi, a._last_q) == (
        b._last_p_arm,
        b._last_expert_xi,
        b._last_q,
    )


def test_layout_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(exp4_mod, "_LAYOUT_CACHE_MAX", 3)
    sched = Exp4Scheduler(rng=RandPool(seed=0))
    lists = [_ALL[i : i + 30] for i in range(6)]
    for ops in lists:
        sched.select_op(ops)
    assert len(sched._layouts) == 3
    assert list(sched._layouts) == [tuple(o) for o in lists[-3:]]
    # An evicted list is laid out again and still draws the linear law.
    ref = Exp4Scheduler(rng=RandPool(seed=1))
    sched._rng = RandPool(seed=1)
    ref.weights = dict(sched.weights)
    ref._ops = set(sched._ops)
    assert sched.select_op(lists[0]) == ref._select_linear(lists[0])
    assert sched._last_expert_xi == ref._last_expert_xi
