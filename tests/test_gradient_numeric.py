"""Numeric (branch-distance) objective of ``gradient_descent``.

Integer compares (trace-cmp, PC known) are scored as ``|window - goal|``
over both byte orders and solved at ``goal = target + off``, ``off`` in
{-1, 0, +1}: the boundary values that flip any of the six predicates.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.gradient_descent import Objective, gradient_descent, pick_objective
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.scripted_rng import ScriptedRng

WIDTH = 4
FIELD_AT = 32
FIELD = 4090
LIMIT = 4096
# Index into the solver's boundary table (-1, 0, +1).
OFFSETS = {0: -1, 1: 0, 2: 1}


def _buf(value: int, order: str) -> bytes:
    """Zero filler with one *order*-encoded integer field at FIELD_AT."""
    return bytes(FIELD_AT) + value.to_bytes(WIDTH, order) + bytes(28)


def _pair(field: int = FIELD, limit: int = LIMIT) -> tuple[bytes, bytes]:
    """trace-cmp operands are logged little-endian."""
    return field.to_bytes(WIDTH, "little"), limit.to_bytes(WIDTH, "little")


def _expect(order: str, goal: int) -> bytes:
    buf = bytearray(_buf(FIELD, order))
    buf[FIELD_AT : FIELD_AT + WIDTH] = (goal % (1 << 8 * WIDTH)).to_bytes(WIDTH, order)
    return bytes(buf)


@pytest.mark.parametrize("order", ["big", "little"])
@pytest.mark.parametrize("idx", sorted(OFFSETS))
def test_writes_boundary_in_field_order(order, idx):
    rng = ScriptedRng(choice_idxs=[idx])
    out = gradient_descent(_buf(FIELD, order), _pair(), rng=rng, objective=Objective.NUMERIC)
    assert out == _expect(order, LIMIT + OFFSETS[idx])


def test_falsify_bitwise_misses_big_endian_field():
    # Falsification: the old objective cannot see a BE field holding a LE operand.
    buf = _buf(FIELD, "big")
    out = gradient_descent(buf, _pair(), objective=Objective.BITWISE)
    assert out[FIELD_AT : FIELD_AT + WIDTH] != LIMIT.to_bytes(WIDTH, "big")


def test_adversarial_zero_target_wraps():
    # goal = 0 - 1 wraps to the all-ones value of the width (signed -1).
    rng = ScriptedRng(choice_idxs=[0])
    pair = _pair(FIELD, 0)
    out = gradient_descent(_buf(FIELD, "big"), pair, rng=rng, objective=Objective.NUMERIC)
    assert out[FIELD_AT : FIELD_AT + WIDTH] == b"\xff" * WIDTH


def test_adversarial_odd_width_falls_back_to_bitwise():
    # Dense overlap keeps both paths off the RNG; an empty ScriptedRng
    # proves the numeric path draws no boundary offset for width 3.
    pair = (b"abc", b"xyw")
    buf = b"xyz" * 40
    numeric = gradient_descent(buf, pair, rng=ScriptedRng(), objective=Objective.NUMERIC)
    assert numeric == gradient_descent(buf, pair, objective=Objective.BITWISE)


def test_adversarial_buffer_shorter_than_width():
    rng = ScriptedRng(choice_idxs=[1])
    assert gradient_descent(b"\x01\x02", _pair(), rng=rng, objective=Objective.NUMERIC) == (
        b"\x01\x02"
    )


def test_adversarial_goal_already_present():
    buf = _buf(LIMIT, "big")
    out = gradient_descent(
        buf, _pair(), rng=ScriptedRng(choice_idxs=[1]), objective=Objective.NUMERIC
    )
    assert out == buf


def test_respects_max_len():
    rng = ScriptedRng(choice_idxs=[1])
    out = gradient_descent(
        _buf(FIELD, "big"), _pair(), max_len=FIELD_AT + WIDTH, rng=rng, objective=Objective.NUMERIC
    )
    assert out == _expect("big", LIMIT)[: FIELD_AT + WIDTH]


def test_pick_objective():
    assert pick_objective(0x401000) is Objective.NUMERIC
    assert pick_objective(None) is Objective.BITWISE


def _engine(pc, choice_idxs):
    pair = _pair()
    cmplog = SimpleNamespace(pair_pc=lambda a, b: pc if (a, b) == pair else None)
    ctx = SimpleNamespace(
        cmplog_pairs=[pair], cmplog=cmplog, max_len=4096, _rng=ScriptedRng(choice_idxs=choice_idxs)
    )
    return SimpleNamespace(ctx=ctx)


def test_operator_uses_numeric_when_pc_known():
    eng = _engine(0x401000, [0, 2])
    out = OperatorEngine._op_gradient_descent(eng, bytearray(_buf(FIELD, "big")), 0, None)
    assert bytes(out) == _expect("big", LIMIT + 1)


def test_operator_uses_bitwise_without_pc():
    # One scripted draw (the pair): the bitwise path must not draw an offset.
    eng = _engine(None, [0])
    buf = _buf(FIELD, "big")
    out = OperatorEngine._op_gradient_descent(eng, bytearray(buf), 0, None)
    want = gradient_descent(buf, _pair(), max_len=4096, objective=Objective.BITWISE)
    assert (bytes(out) if out else buf) == want


def test_fuzzy_field_when_transformed():
    # Field holds source+3 (e.g. a header offset): no exact hit, so the
    # numerically closest window wins. Non-zero filler keeps it unique.
    shift = 3
    buf = b"A" * FIELD_AT + (FIELD + shift).to_bytes(WIDTH, "big") + b"A" * 28
    out = gradient_descent(buf, _pair(), rng=RandPool(seed=1), objective=Objective.NUMERIC)
    goals = {(LIMIT + off).to_bytes(WIDTH, "big") for off in OFFSETS.values()}
    assert out[FIELD_AT : FIELD_AT + WIDTH] in goals
    assert out[:FIELD_AT] == buf[:FIELD_AT]


def test_multiple_hits_pick_by_rng():
    # Two exact BE hits: the second choice() draw selects which one is rewritten.
    second = FIELD_AT + 16
    buf = bytearray(_buf(FIELD, "big"))
    buf[second : second + WIDTH] = FIELD.to_bytes(WIDTH, "big")
    out = gradient_descent(
        bytes(buf), _pair(), rng=ScriptedRng(choice_idxs=[1, 1]), objective=Objective.NUMERIC
    )
    assert out[second : second + WIDTH] == LIMIT.to_bytes(WIDTH, "big")
    assert out[FIELD_AT : FIELD_AT + WIDTH] == FIELD.to_bytes(WIDTH, "big")


def test_adversarial_mismatched_operand_widths_use_bitwise():
    pair = (FIELD.to_bytes(8, "little"), LIMIT.to_bytes(WIDTH, "little"))
    buf = _buf(FIELD, "big")
    numeric = gradient_descent(buf, pair, rng=ScriptedRng(), objective=Objective.NUMERIC)
    assert numeric == gradient_descent(buf, pair, objective=Objective.BITWISE)
