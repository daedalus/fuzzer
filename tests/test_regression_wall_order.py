"""Regression: comparison-wall ordering in condstmt_solve (minimax Phase 3).

``Z3Solver.solve_comparison_wall`` had no caller. ``--wall-order`` now makes
``_op_condstmt_solve`` solve the head of the wall order (alpha-beta within
overlap components, widest taint first) instead of a random unsolved branch.
"""

from __future__ import annotations

from types import SimpleNamespace

from fuzzer_tool.core.cond_stmt import CondState
from fuzzer_tool.core.smt_solver import Z3Solver
from fuzzer_tool.services.operators import _WALL_WINDOW, OperatorEngine
from tests.support.operator_env import make_minimal_fuzzer
from tests.support.scripted_rng import ScriptedRng

NARROW = (b"z", b"q")  # 1 tainted byte
WIDE = (b"ABCD", b"WXYZ")  # 4 tainted bytes
DATA = b"zABCD---"


def _engine(pairs, wall: bool, rng: ScriptedRng) -> OperatorEngine:
    f = make_minimal_fuzzer(pool=rng)
    f._cmplog = SimpleNamespace(pairs=list(pairs), tokens=[])
    f._use_wall_order = wall
    return OperatorEngine(f)


def _solved(engine: OperatorEngine) -> list[bytes]:
    return [c.base.op_a for c in engine._cond_stmts if c.state is CondState.SOLVED]


def test_regression_wall_order_solves_widest_first():
    # Falsification: rng.choice is never scripted, so a random pick raises
    # StopIteration; the wall head (4-byte taint) must be solved.
    engine = _engine([NARROW, WIDE], wall=True, rng=ScriptedRng(randoms=[0.1]))
    buf = bytearray(DATA)

    out = engine._op_condstmt_solve(buf, 0, DATA)

    assert _solved(engine) == [WIDE[0]]
    assert bytes(out) == DATA.replace(WIDE[0], WIDE[1])


def test_wall_order_control_off():
    # Control: without the flag the scripted choice (index 0) wins.
    engine = _engine([NARROW, WIDE], wall=False, rng=ScriptedRng(randoms=[0.1], choice_idxs=[0]))

    engine._op_condstmt_solve(bytearray(DATA), 0, DATA)

    assert _solved(engine) == [NARROW[0]]


def test_adversarial_wide_wall_is_windowed(monkeypatch):
    # 200 conditions all tainting the same byte form one component; the
    # search must stay bounded to the window, not 200! orderings.
    seen: list[int] = []
    real = Z3Solver.solve_comparison_wall

    def spy(self, conditions, *args, **kwargs):
        seen.append(len(conditions))
        return real(self, conditions, *args, **kwargs)

    monkeypatch.setattr(Z3Solver, "solve_comparison_wall", spy)
    pairs = [(b"-", bytes([i])) for i in range(1, 201)]
    engine = _engine(pairs, wall=True, rng=ScriptedRng(randoms=[0.1]))

    engine._op_condstmt_solve(bytearray(DATA), 0, DATA)

    assert seen == [_WALL_WINDOW]
    assert len(_solved(engine)) == 1


def test_adversarial_untainted_wall_keeps_input_order():
    # No operand occurs in the input: no offsets, all singletons, so the
    # order is the cmplog order and the first unsolved branch is the head.
    pairs = [(b"\x01\x02", b"\x03\x04"), (b"\x05\x06", b"\x07\x08")]
    engine = _engine(pairs, wall=True, rng=ScriptedRng(randoms=[0.1], randints=[0]))

    engine._op_condstmt_solve(bytearray(DATA), 0, DATA)

    assert _solved(engine) == [pairs[0][0]]
