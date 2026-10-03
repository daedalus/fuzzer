"""Regression: LasVegasScheduler._eliminate() stalled the fuzzer.

record() runs _eliminate() on every reward, and the scalar version called
_radius() twice per eligible arm: ~400 Python calls per record at fuzzgoat's
arm count. Measured on an 8k-iteration fuzzgoat run: 248s with --las-vegas
vs 42s without, 158s of it inside _eliminate().

The oracle below is the pre-fix scalar code, copied verbatim, so the
vectorized version must drop exactly the same arms in the same order.
"""

from __future__ import annotations

import math
import sys

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_las_vegas import LasVegasScheduler


class _ScalarOracle(LasVegasScheduler):
    """Pre-fix elimination: per-arm Python loop over Hoeffding radii."""

    def _radius_ref(self, n: int) -> float:
        k = max(len(self._mean), 1)
        t = max(self._total_pulls, 1)
        arg = max(t * k / self.delta, 1.0)
        return math.sqrt(math.log(arg) / (2.0 * n))

    def _eliminate(self) -> None:
        if len(self._active) <= 1:
            return
        eligible = [a for a in self._active if self._n.get(a, 0) >= self.min_pulls]
        if len(eligible) < 2:
            return

        best_lcb = max(self._mean[a] - self._radius_ref(self._n[a]) for a in eligible)
        for a in [a for a in eligible if self._mean[a] + self._radius_ref(self._n[a]) < best_lcb]:
            self._active.discard(a)
            self._eliminated.add(a)


def _drive(sched: LasVegasScheduler, n_arms: int, steps: int, seed: int) -> list:
    """Skewed per-arm success rates so eliminations actually happen."""
    rng = RandPool(seed)
    ops = [f"op{i}" for i in range(n_arms)]
    rates = [0.9 if i == 0 else 0.05 for i in range(n_arms)]
    trace = []
    for _ in range(steps):
        op = sched.select_op(ops)
        sched.record(op, rng.random() < rates[int(op[2:])], weight=1.0)
        trace.append((op, frozenset(sched._active), frozenset(sched._eliminated)))
    return trace


def test_regression_lv_oracle_control():
    """Control: the oracle agrees with itself, so the comparison can pass."""
    assert _drive(_ScalarOracle(), 12, 600, 7) == _drive(_ScalarOracle(), 12, 600, 7)


def test_regression_lv_matches_scalar():
    """Falsification: vectorized elimination == scalar oracle, step by step."""
    ref = _drive(_ScalarOracle(), 12, 600, 7)
    got = _drive(LasVegasScheduler(), 12, 600, 7)

    assert any(elim for _, _, elim in ref), "oracle never eliminated: test is vacuous"
    assert got == ref


def test_regression_lv_growth_past_capacity():
    """Adversarial: arms first seen via record() past initial array capacity."""
    n_arms = 300
    ref, got = _ScalarOracle(reopen_interval=50), LasVegasScheduler(reopen_interval=50)
    rng = RandPool(3)
    for step in range(4000):
        name = f"late{step % n_arms}"
        ok = rng.random() < (0.9 if name == "late0" else 0.02)
        ref.record(name, ok)
        got.record(name, ok)
        assert got._active == ref._active
        assert got._eliminated == ref._eliminated


def _loaded(cls: type[LasVegasScheduler], n_arms: int) -> LasVegasScheduler:
    """Every arm eligible, equal means: _eliminate() scans all, drops none."""
    sched = cls()
    for i in range(n_arms):
        for _ in range(sched.min_pulls):
            sched.record(f"op{i}", False)
    return sched


def _py_calls(sched: LasVegasScheduler) -> int:
    """Python-level calls made by one _eliminate(); C calls are not counted."""
    calls = 0

    def _count(_frame, event, _arg):
        nonlocal calls
        calls += event == "call"

    sys.setprofile(_count)
    try:
        sched._eliminate()
    finally:
        sys.setprofile(None)
    return calls


def test_regression_lv_eliminate_cost_control():
    """Control: the scalar oracle's Python call count grows with arm count."""
    assert _py_calls(_loaded(_ScalarOracle, 400)) > _py_calls(_loaded(_ScalarOracle, 40))


def test_regression_lv_eliminate_cost():
    """record() must not pay a Python call per arm: cost independent of K.

    Counts calls instead of timing them, so CI load cannot flake it.
    """
    small = _py_calls(_loaded(LasVegasScheduler, 40))
    assert _py_calls(_loaded(LasVegasScheduler, 400)) == small


def test_regression_lv_growth_via_select():
    """Adversarial: ops lists with duplicates and unseen arms past capacity."""
    ref, got = _ScalarOracle(), LasVegasScheduler()
    rng = RandPool(5)
    for step in range(3000):
        width = 2 + step % 7
        ops = [f"s{(step + j * 37) % 150}" for j in range(width)] + [f"s{step % 150}"]
        a, b = ref.select_op(ops), got.select_op(ops)
        assert a == b
        ok = rng.random() < (0.9 if a == "s0" else 0.03)
        ref.record(a, ok)
        got.record(b, ok)
