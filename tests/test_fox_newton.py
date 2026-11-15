"""FOX secant step (``core/fox_newton.py``): zero-exec fit of operand vs field.

Each test logs the compares of two or three executed inputs, as the cmplog
drain would, then asks ``SecantLog.propose`` for the field value whose
predicted operand lands on a goal next to the constant. Expected outputs are
derived from the target model, never echoed from the solver.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from fuzzer_tool.core.cond_stmt import CondStmt
from fuzzer_tool.core.fox_newton import (
    MAX_INPUTS,
    MAX_SITES,
    SecantLog,
    flip_goals,
)
from fuzzer_tool.core.gradient_descent import Objective, gradient_descent
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

PC = 0x401000
WIDTH = 2
FIELD_AT = 5
SIZE = 16
CONST = 1000


def _input(x: int, order: str = "little", size: int = SIZE) -> bytes:
    """Filler with one *order*-encoded WIDTH-byte field at FIELD_AT."""
    buf = bytearray(size)
    buf[FIELD_AT : FIELD_AT + WIDTH] = x.to_bytes(WIDTH, order)
    return bytes(buf)


def _cond(op: int, const: int = CONST, pc: int | None = PC, width: int = WIDTH):
    """trace-cmp record: operands little-endian, result = sign(a - b)."""
    result = (op > const) - (op < const)
    a, b = op.to_bytes(width, "little"), const.to_bytes(width, "little")
    return CondStmt.from_cmplog_pair(0, a, b, width, result=result, pc=pc)


def _affine(x: int) -> int:
    """Target maps the field before comparing: 3*x + 7 (no wrap below 21843)."""
    return 3 * x + 7


def _log_runs(xs, order="little", f=_affine) -> SecantLog:
    log = SecantLog()
    for x in xs:
        log.observe(_input(x, order), [_cond(f(x))])
    return log


def _field(buf: bytes, order: str = "little") -> int:
    return int.from_bytes(buf[FIELD_AT : FIELD_AT + WIDTH], order)


# ── goal selection ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("signs", "offsets"),
    [
        ({-1}, (0, 1)),
        ({1}, (-1, 0)),
        ({0}, (-1, 1)),
        ({-1, 1}, (0,)),
        ({-1, 0, 1}, (-1, 0, 1)),
    ],
)
def test_flip_goals_target_unseen_sides(signs, offsets):
    assert flip_goals(signs, CONST, WIDTH) == tuple(CONST + o for o in offsets)


def test_flip_goals_wrap_at_width():
    # const 0, only a > b seen: 0 - 1 wraps to the all-ones value.
    assert flip_goals({1}, 0, WIDTH) == ((1 << 8 * WIDTH) - 1, 0)


# ── solve ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("order", ["little", "big"])
@pytest.mark.parametrize("idx", [0, 1])
def test_secant_solves_affine_field(order, idx):
    # Both runs had 3x+7 < 1000: goals are 1000 and 1001; idx picks one.
    log = _log_runs([10, 20], order)
    goal = flip_goals({-1}, CONST, WIDTH)[idx]
    # Draws: site, window (LE start / BE end; a low-byte change fits both), goal.
    window = 0 if order == "little" else 1
    rng = ScriptedRng(choice_idxs=[0, window, idx])

    out = log.propose(_input(10, order), rng)

    x = _field(out, order)
    assert abs(_affine(x) - goal) <= 1  # nearest integer root of 3x + 7 = goal
    assert out[:FIELD_AT] == bytes(FIELD_AT) and out[FIELD_AT + WIDTH :] == bytes(9)


def test_secant_exact_when_root_is_integer():
    # 3x + 7 = 1000 + 1 -> x = 331.33..; 3x + 7 = 1000 - 1 -> x = 330.67..;
    # with const 1003 the goal 1003 has the integer root 332.
    log = SecantLog()
    for x in (10, 20):
        log.observe(_input(x), [_cond(_affine(x), const=1003)])
    out = log.propose(_input(10), ScriptedRng(choice_idxs=[0, 0, 0]))
    assert _affine(_field(out)) == 1003


def test_falsify_numeric_descent_misses_affine_field():
    # Falsification: gradient_descent NUMERIC writes the goal into the field,
    # so the target sees 3*goal + 7, never the goal.
    a, b = _affine(20).to_bytes(WIDTH, "little"), CONST.to_bytes(WIDTH, "little")
    # Any draw: the field keeps 20 (operand 67) or takes ~1000 (operand ~3007).
    rng = RandPool(1)
    out = gradient_descent(_input(20), (a, b), rng=rng, objective=Objective.NUMERIC)
    assert abs(_affine(_field(out)) - CONST) > 1


# ── adversarial ──────────────────────────────────────────────────────


def test_adversarial_nonlinear_third_point_rejects():
    # x*x fits any two points with a line; the third run exposes it.
    log = _log_runs([10, 20, 30], f=lambda x: x * x)
    assert log.propose(_input(10), ScriptedRng(choice_idxs=[0, 0, 0])) is None


def test_adversarial_both_operands_vary_declines():
    log = SecantLog()
    for x in (10, 20):
        log.observe(_input(x), [_cond(_affine(x), const=x)])
    assert not log.ready()


def test_adversarial_span_wider_than_compare_declines():
    # Inputs differ at bytes 0 and 9: no WIDTH-byte window covers both.
    log = SecantLog()
    a, b = bytearray(_input(10)), bytearray(_input(20))
    b[0] = 0xFF
    log.observe(bytes(a), [_cond(_affine(10))])
    log.observe(bytes(b), [_cond(_affine(20))])
    assert log.propose(bytes(a), ScriptedRng(choice_idxs=[0, 0, 0])) is None


def test_adversarial_length_change_declines():
    log = SecantLog()
    log.observe(_input(10), [_cond(_affine(10))])
    log.observe(_input(20, size=SIZE + 1), [_cond(_affine(20))])
    assert not log.ready()


def test_adversarial_root_out_of_range_declines():
    # y = x + 60000 with const 100: the root is negative.
    log = SecantLog()
    for x in (10, 20):
        log.observe(_input(x), [_cond(x + 60000, const=100)])
    assert log.propose(_input(10), ScriptedRng(choice_idxs=[0, 0, 0])) is None


def test_adversarial_short_buffer_declines():
    log = _log_runs([10, 20])
    assert log.propose(bytes(FIELD_AT + 1), ScriptedRng(choice_idxs=[0, 0, 0])) is None


def test_ignores_byte_compares_and_odd_widths():
    log = SecantLog()
    for x in (10, 20):
        log.observe(_input(x), [_cond(_affine(x), pc=None), _cond(x, width=3)])
    assert not log.ready()


def test_repeat_pc_in_one_exec_keeps_first():
    # A loop logs the site twice; the second record belongs to another field.
    log = SecantLog()
    for x in (10, 20):
        log.observe(_input(x), [_cond(_affine(x)), _cond(5)])
    out = log.propose(_input(10), ScriptedRng(choice_idxs=[0, 0, 0]))
    assert abs(_affine(_field(out)) - CONST) <= 1


def test_memory_bounded_inputs_and_sites():
    log = SecantLog()
    for x in range(MAX_INPUTS + 5):
        log.observe(_input(x), [_cond(x, pc=PC + 4 * i) for i in range(MAX_SITES // 4)])
    for i in range(MAX_SITES + 10):
        log.observe(_input(1), [_cond(1, pc=0x900000 + i)])
    assert log.live_inputs() <= MAX_INPUTS
    assert log.site_count() <= MAX_SITES


def test_stale_observations_expire():
    # Newer runs push the two useful ones out of the input ring. Their own
    # site never varies (operand 7 every run), so nothing is fittable.
    log = _log_runs([10, 20])
    for x in range(MAX_INPUTS):
        log.observe(_input(x, size=SIZE + 1), [_cond(7, pc=PC + 4)])
    assert log.live_inputs() == MAX_INPUTS
    assert not log.ready()


def test_regression_far_pair_does_not_hide_near_pair():
    # Real runs: the newest two records at a site often come from unrelated
    # seeds (many bytes differ). An older run one field-edit away from the
    # newest must still fit, and ready() must mean propose can succeed.
    log = SecantLog()
    far = bytearray(_input(20))
    far[0], far[SIZE - 1] = 0xAA, 0xBB  # differs outside any WIDTH window
    log.observe(_input(10), [_cond(_affine(10))])
    log.observe(bytes(far), [_cond(_affine(20))])
    log.observe(_input(20), [_cond(_affine(20) + 3)])  # newest; near run 1
    assert log.ready()
    out = log.propose(_input(10), ScriptedRng(choice_idxs=[0, 0, 0]))
    # Line through (10, 37) and (20, 70): slope 3.3, root of 1000 at 301.8.
    assert abs(37 + Fraction(33, 10) * (_field(out) - 10) - CONST) <= 2


def test_regression_ready_implies_fit():
    # Two records whose inputs differ across 7 bytes: orientable, not fittable.
    log = SecantLog()
    a, b = bytearray(_input(10)), bytearray(_input(20))
    b[0] = 0xFF
    log.observe(bytes(a), [_cond(_affine(10))])
    log.observe(bytes(b), [_cond(_affine(20))])
    assert not log.ready()


def test_input_side_on_right_operand():
    # if (1000 == 3x + 7): the constant is op_a, the field drives op_b.
    log = SecantLog()
    for x in (10, 20):
        a, b = CONST.to_bytes(WIDTH, "little"), _affine(x).to_bytes(WIDTH, "little")
        log.observe(_input(x), [CondStmt.from_cmplog_pair(0, a, b, WIDTH, result=1, pc=PC)])
    out = log.propose(_input(10), ScriptedRng(choice_idxs=[0, 0, 0]))
    assert abs(_affine(_field(out)) - CONST) <= 1
