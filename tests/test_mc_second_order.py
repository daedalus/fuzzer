"""MonteCarloScheduler second-order blend: default-off identity and synthetic A/B."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler

_OPS = ["a", "b", "c", "d"]
_STEPS = 6000


def _want(x: str, y: str, second_order: bool) -> str:
    """Rewarded successor. Never equals y (repeats are not transitions).

    second_order: depends on (x, y), so first order can do at best 1/3.
    else: depends on y alone, first order suffices.
    """
    step = 1 + (_OPS.index(x) % 3 if second_order else 0)
    return _OPS[(_OPS.index(y) + step) % len(_OPS)]


def _sched(seed: int, **kw) -> MonteCarloScheduler:
    s = MonteCarloScheduler(rng=RandPool(seed=seed), arm_decay=1.0, **kw)
    for op in _OPS:
        s.arm_alpha[op] = 1.0
        s.arm_beta[op] = 1.0
    return s


def _run(s: MonteCarloScheduler, second_order: bool = True) -> float:
    """Average success rate over the last half of the run."""
    hist: list[str] = ["a", "b"]
    wins = 0
    for i in range(_STEPS):
        op = s.select_op(_OPS, prev_op=hist[-1])
        ok = op == _want(hist[-2], hist[-1], second_order)
        s.record(op, ok)
        hist.append(op)
        if i >= _STEPS // 2:
            wins += ok
    return wins / (_STEPS - _STEPS // 2)


class TestDefaultOff:
    def test_default_blend_is_zero_and_chain_untouched(self):
        # Off must cost nothing (Hard Rule 41): no chain bookkeeping at all.
        s = _sched(1)
        assert s.second_order_blend == 0.0
        for op in ("a", "b", "c"):
            s.record(op, True)
        assert len(s._chain2) == 0  # noqa: SLF001

    def test_on_records_triples(self):
        s = _sched(1, second_order_blend=0.5)
        for op in ("a", "b", "c"):
            s.record(op, True)
        assert s._chain2.count("a", "b", "c") == 1  # noqa: SLF001

    def test_off_is_identical_to_no_param(self):
        a, b = _sched(3), _sched(3, second_order_blend=0.0)
        seq_a = [(a.select_op(_OPS), a.record("a", True))[0] for _ in range(200)]
        seq_b = [(b.select_op(_OPS), b.record("a", True))[0] for _ in range(200)]
        assert seq_a == seq_b

    def test_blend_is_clamped(self):
        assert _sched(1, second_order_blend=7.0).second_order_blend == 1.0
        assert _sched(1, second_order_blend=-1.0).second_order_blend == 0.0


class TestBackoff:
    def test_unseen_context_falls_back_to_first_order_path(self):
        s = _sched(2, second_order_blend=1.0, pairwise_blend=0.0)
        s.record("a", True)
        s.record("b", True)
        # Context (a, b) has no recorded successor: must still return a valid op.
        assert s.select_op(_OPS) in _OPS

    def test_no_history_returns_valid_op(self):
        assert _sched(2, second_order_blend=1.0).select_op(_OPS) in _OPS


_PAIR = {"pairwise_blend": 0.5}
_BOTH = {"pairwise_blend": 0.5, "second_order_blend": 0.5}
_SEEDS = range(20, 25)


def _mean(kw: dict, second_order: bool) -> float:
    return sum(_run(_sched(s, **kw), second_order) for s in _SEEDS) / len(_SEEDS)


class TestSyntheticAB:
    def test_control_pairwise_vs_itself_agree(self):
        # Hard Rule 46: identical configs on disjoint seeds must agree, else
        # the comparison below measures noise. Memoryless env: both ~1.0.
        r1 = _run(_sched(10, **_PAIR), False)
        r2 = _run(_sched(11, **_PAIR), False)
        assert abs(r1 - r2) < 0.10

    def test_second_order_beats_first_order_when_history_matters(self):
        # Measured: first order 0.45 (0.09-0.67), with second order 0.57 (0.50-0.67).
        assert _mean(_BOTH, True) > _mean(_PAIR, True) + 0.05

    def test_no_meaningful_loss_when_environment_is_memoryless(self):
        assert _mean(_BOTH, False) > _mean(_PAIR, False) - 0.05

    @pytest.mark.parametrize("w", [0.25, 1.0])
    def test_chain_memory_bounded_under_long_run(self, w):
        s = _sched(40, second_order_blend=w)
        _run(s)
        assert len(s._chain2) <= len(_OPS) ** 2  # noqa: SLF001
