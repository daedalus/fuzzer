"""cmp_solve: overwrite a partially matching window with a cmp operand.

honggfuzz mangle_CmpSolve. Draws, in order: pair index, operand index,
variant (VARIANT_PLUS -> +1, VARIANT_MINUS -> -1 as a little-endian int,
anything else exact). Expected outputs derive from int arithmetic, not
from a run.
"""

from types import SimpleNamespace

from fuzzer_tool.core.cmp_solve import (
    MAX_INT_LEN,
    MAX_OPERAND_LEN,
    VARIANT_EXACT,
    VARIANT_MINUS,
    VARIANT_PLUS,
    cmp_solve,
    partial_window,
)
from fuzzer_tool.core.operator_registry import REGISTRY
from tests.support.scripted_rng import ScriptedRng


def _le_add(val: bytes, delta: int) -> bytes:
    n = len(val)
    return ((int.from_bytes(val, "little") + delta) % (1 << (8 * n))).to_bytes(n, "little")


def _rng(pair=0, operand=0, variant=VARIANT_EXACT):
    return ScriptedRng(randints=[pair, operand, variant])


# ── partial_window ───────────────────────────────────────────────────


def test_window_is_lowest_partial_offset():
    data = b"....MAGX....MAXX"
    assert partial_window(data, b"MAGI") == data.index(b"MAGX")


def test_full_match_is_not_partial():
    """Falsification: the operand already in place is skipped."""
    data = b"MAGI" + b"." * 8 + b"MXXX"
    assert partial_window(data, b"MAGI") == data.index(b"MXXX")


def test_no_shared_byte_no_window():
    assert partial_window(b"\x00" * 64, b"MAGI") is None


def test_only_full_matches_no_window():
    assert partial_window(b"MAGIMAGI", b"MAGI") is None


def test_window_must_fit():
    """Adversarial: a byte match whose window runs off either end."""
    assert partial_window(b"Ixxx", b"MAGI") is None
    assert partial_window(b"xxM", b"MAGI") is None
    assert partial_window(b"MA", b"MAGI") is None


def test_window_matches_brute_force():
    """Lazy merge agrees with an O(n*m) scan on mixed inputs."""
    import itertools

    def brute(data, val):
        n = len(val)
        for off in range(len(data) - n + 1):
            m = sum(a == b for a, b in zip(data[off : off + n], val, strict=True))
            if 0 < m < n:
                return off
        return None

    alphabet = b"ab\x00"
    for k in range(0, 7):
        for data in itertools.product(alphabet, repeat=k):
            data = bytes(data)
            for val in (b"ab", b"ba", b"a\x00b", b"aaa"):
                assert partial_window(data, val) == brute(data, val), (data, val)


# ── cmp_solve ────────────────────────────────────────────────────────


def test_exact_overwrite():
    data = b"..MAGX.."
    out = cmp_solve(data, [(b"MAGI", b"\x00\x01")], _rng())
    assert out == b"..MAGI.."


def test_second_operand_drawn():
    data = b"..\x00\x02.."
    out = cmp_solve(data, [(b"MAGI", b"\x00\x01")], _rng(operand=1))
    assert out == b"..\x00\x01.."


def test_plus_one_little_endian():
    val = b"\xff\x01"
    data = b"\x00\x01zz"
    out = cmp_solve(data, [(val, b"zz")], _rng(variant=VARIANT_PLUS))
    assert out == _le_add(val, 1) + b"zz"


def test_minus_one_little_endian():
    val = b"\x00\x02"
    data = b"zz\x00\x03"
    out = cmp_solve(data, [(val, b"q")], _rng(variant=VARIANT_MINUS))
    assert out == b"zz" + _le_add(val, -1)


def test_plus_one_wraps():
    val = b"\xff\xff"
    out = cmp_solve(b"\xff\x00", [(val, b"q")], _rng(variant=VARIANT_PLUS))
    assert out == _le_add(val, 1) == b"\x00\x00"


def test_wide_operand_stays_exact():
    """Past MAX_INT_LEN the operand is a string: no +/-1."""
    val = b"A" * (MAX_INT_LEN + 1)
    data = b"A" * MAX_INT_LEN + b"B"
    out = cmp_solve(data, [(val, b"q")], _rng(variant=VARIANT_PLUS))
    assert out == val


def test_length_preserved():
    data = b"xxMAGXyy"
    out = cmp_solve(data, [(b"MAGI", b"q")], _rng())
    assert len(out) == len(data)


def test_declines_without_window():
    """Adversarial: unusable operands or inputs return None (a no-op)."""
    long_val = b"A" * (MAX_OPERAND_LEN + 1)
    cases = [
        (b"", [(b"MAGI", b"q")]),
        (b"\x00" * 16, [(b"MAGI", b"q")]),  # no shared byte
        (b"MAGI", [(b"MAGI", b"q")]),  # only a full match
        (b"M", [(b"M", b"q")]),  # 1-byte operand cannot partially match
        (b"A" * 64, [(long_val, b"q")]),  # over the operand cap
        (b"MA", [(b"MAGI", b"q")]),  # operand longer than input
    ]
    for data, pairs in cases:
        assert cmp_solve(data, pairs, _rng()) is None, (data, pairs)


def test_no_pairs_declines():
    assert cmp_solve(b"data", [], ScriptedRng()) is None


# ── Registry and handler ─────────────────────────────────────────────


def test_registered_in_adaptive_band():
    assert "cmp_solve" in REGISTRY.names()
    assert REGISTRY.category_of("cmp_solve") == "adaptive"


def test_handler_applies_and_truncates():
    from fuzzer_tool.services.operators import OperatorEngine

    ctx = SimpleNamespace(cmplog_pairs=[(b"MAGI", b"q")], _rng=_rng(), max_len=64)
    out = OperatorEngine._op_cmp_solve(SimpleNamespace(ctx=ctx), bytearray(b"MAGX"), 0, b"")
    assert out == bytearray(b"MAGI")


def test_handler_noop_without_pairs():
    from fuzzer_tool.services.operators import OperatorEngine

    ctx = SimpleNamespace(cmplog_pairs=[], _rng=ScriptedRng(), max_len=64)
    assert OperatorEngine._op_cmp_solve(SimpleNamespace(ctx=ctx), bytearray(b"x"), 0, b"") is None
