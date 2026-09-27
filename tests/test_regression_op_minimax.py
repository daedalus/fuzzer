"""Regression: MonteCarloScheduler.select_op_minimax (minimax Phase 4).

The root searched ``operators[:beam_width]`` in list order, not the
top-scored ops, so it could only ever return one of the first four
candidates. It scored with the posterior mean although its docstring said
Thompson sample, so it never explored. It had no caller; now wired behind
``fuzz --op-minimax`` on the ``bandit`` meta-strategy.
"""

from __future__ import annotations

from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler
from fuzzer_tool.services.operators import OperatorEngine
from tests.test_regression_scheduler_fallback_precedence import _FakeFuzzer, _FakeMC


class _ScriptedBeta:
    """Returns scripted Beta draws in call order (Hard Rule 39)."""

    def __init__(self, draws):
        self._draws = iter(draws)
        self.calls = 0

    def betavariate(self, _a, _b):
        self.calls += 1
        return next(self._draws)


def _sched(ops: list[str], draws: list[float]) -> MonteCarloScheduler:
    mc = MonteCarloScheduler(rng=_ScriptedBeta(draws))
    for op in ops:
        mc.init_arm(op)
    return mc


def test_regression_root_beam_reaches_top_scored_op():
    # Falsification: the best draw is the 6th op; the old root only
    # searched the first 4 and could not return it.
    ops = [f"o{i}" for i in range(6)]
    draws = [0.1, 0.2, 0.15, 0.05, 0.12, 0.9]

    assert _sched(ops, draws).select_op_minimax(ops) == "o5"


def test_regression_uses_thompson_draws_not_means():
    # Identical posteriors: only the draws can separate them. A mean-based
    # evaluation returns the same op for both scripts.
    ops = ["a", "b"]

    assert _sched(ops, [0.2, 0.9]).select_op_minimax(ops) == "b"
    assert _sched(ops, [0.9, 0.2]).select_op_minimax(ops) == "a"


def test_one_draw_per_op():
    ops = [f"o{i}" for i in range(10)]
    mc = _sched(ops, [i / 10 for i in range(10)])

    mc.select_op_minimax(ops, depth=3, beam_width=4)

    assert mc._rng.calls == len(ops)


def test_adversarial_degenerate_inputs():
    mc = _sched(["x"], [])

    assert mc.select_op_minimax([]) == ""
    assert mc.select_op_minimax(["x"]) == "x"


def test_adversarial_unregistered_and_wide():
    # Ops the scheduler never armed still draw from the Beta(1,1) prior;
    # a pool wider than any beam still returns a member.
    ops = [f"u{i}" for i in range(300)]
    draws = [(i * 37 % 300) / 300 for i in range(300)]
    mc = MonteCarloScheduler(rng=_ScriptedBeta(draws))

    op = mc.select_op_minimax(ops, depth=4, beam_width=8)

    assert op == ops[max(range(300), key=draws.__getitem__)]


class _MinimaxMC(_FakeMC):
    def __init__(self):
        super().__init__()
        self.minimax_calls = 0

    def select_op_minimax(self, ops: list[str]) -> str:
        self.minimax_calls += 1
        return "op_minimax"


def _bandit_fuzzer(op_minimax: bool) -> tuple[_FakeFuzzer, _MinimaxMC]:
    f = _FakeFuzzer()
    f.enable("bandit")
    f.mc = _MinimaxMC()
    f._use_op_minimax = op_minimax
    return f, f.mc


def test_bandit_dispatches_to_minimax_when_flagged():
    f, mc = _bandit_fuzzer(op_minimax=True)

    op = OperatorEngine(f).select_op(["bit_flip", "byte_flip"])

    assert op == "op_minimax"
    assert (mc.minimax_calls, mc.calls) == (1, 0)
    assert f._prev_bandit_op == "op_minimax"


def test_bandit_control_without_flag():
    f, mc = _bandit_fuzzer(op_minimax=False)

    assert OperatorEngine(f).select_op(["bit_flip", "byte_flip"]) == "op_bandit"
    assert (mc.minimax_calls, mc.calls) == (0, 1)
