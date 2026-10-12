"""Tests for line_blend, field_minmax, nm_reflect and opposition.

Randomness is scripted (Hard Rule 39): shape, parameter (when the operator has
one), then phase (width > 1 only), so every expected child is exact.  Expected
values come from an independent per-field oracle written with Python ints.
"""

import random

import pytest

from fuzzer_tool.core.mutations.generic import (
    _LINE_BLEND_NUMS,
    field_minmax,
    line_blend,
    nm_reflect,
    opposition,
)
from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

B1, B2, L2, B4, L4 = range(5)
_SHAPES = {B1: (1, "big"), B2: (2, "big"), L2: (2, "little"), B4: (4, "big"), L4: (4, "little")}


class Scripted:
    def __init__(self, shape, param=None, phase=0):
        self._draws = [shape] + ([] if param is None else [param]) + [phase]

    def randrange(self, n):
        return self._draws.pop(0)


class NoDraw:
    """Any draw raises: a declined call must not consume randomness."""

    def randrange(self, n):
        raise AssertionError("declined call drew randomness")


def be(*v, w=4):
    return b"".join(x.to_bytes(w, "big") for x in v)


def fields(x, shape, count=None):
    w, order = _SHAPES[shape]
    return [int.from_bytes(x[i : i + w], order) for i in range(0, len(x) - w + 1, w)][:count]


def clip(v, w):
    return max(0, min(v, (1 << (8 * w)) - 1))


# --- line_blend -------------------------------------------------------------
def test_line_blend_numerators_exclude_the_parents_and_cover_both_extrapolations():
    assert 0 not in _LINE_BLEND_NUMS and 8 not in _LINE_BLEND_NUMS  # t=0 is a, t=1 is b
    assert min(_LINE_BLEND_NUMS) < 0 and max(_LINE_BLEND_NUMS) > 8  # t<0 and t>1 both reachable
    assert len(set(_LINE_BLEND_NUMS)) == len(_LINE_BLEND_NUMS)


@pytest.mark.parametrize(
    ("num", "expect"), [(-4, 0x00), (-1, 0x0C), (1, 0x14), (4, 0x20), (9, 0x34), (12, 0x40)]
)
def test_line_blend_is_a_plus_t_times_the_gap(num, expect):
    # a=0x10, b=0x30: gap 32, child = a + 32 * num / 8
    got = line_blend(b"\x10", b"\x30", Scripted(B1, _LINE_BLEND_NUMS.index(num)))
    assert got == bytes([expect])


def test_line_blend_reaches_beyond_the_donor():
    # t = 3/2 on 0 -> 1000 lands at 1500: past b, which arith_mid can never do
    out = line_blend(be(0), be(1000), Scripted(B4, _LINE_BLEND_NUMS.index(12), phase=2))
    assert out == be(1500)
    # t = -1/2 goes the other way, past a
    out = line_blend(be(1000), be(2000), Scripted(B4, _LINE_BLEND_NUMS.index(-4), phase=2))
    assert out == be(500)


def test_line_blend_floors_negative_steps():
    # gap -3, t = 1/8: floor(-3/8) = -1, so 0x10 -> 0x0f (truncation would leave 0x10)
    assert line_blend(b"\x10", b"\x0d", Scripted(B1, _LINE_BLEND_NUMS.index(1))) == b"\x0f"


def test_line_blend_saturates_instead_of_wrapping():
    assert line_blend(b"\xf0", b"\xff", Scripted(B1, _LINE_BLEND_NUMS.index(12))) == b"\xff"
    assert line_blend(b"\x10", b"\x00", Scripted(B1, _LINE_BLEND_NUMS.index(12))) == b"\x00"


@pytest.mark.parametrize("shape", [B1, B2, L2, B4, L4])
@pytest.mark.parametrize("num", _LINE_BLEND_NUMS)
def test_line_blend_matches_the_per_field_oracle(shape, num):
    w, _ = _SHAPES[shape]
    # every byte differs, so the grid (phase 0) covers the whole 16-byte buffer
    a = bytes((i * 37 + 11) % 256 for i in range(16))
    b = bytes((x + 101) % 256 for x in a)
    out = line_blend(a, b, Scripted(shape, _LINE_BLEND_NUMS.index(num)))
    want = [
        clip(x + (((y - x) * num) >> 3), w)
        for x, y in zip(fields(a, shape), fields(b, shape), strict=True)
    ]
    assert fields(out, shape) == want


def test_line_blend_keeps_shared_bytes_and_length():
    a, b = b"MAGIC\x10tail", b"MAGIC\x30tail"
    out = line_blend(a, b, Scripted(B1, _LINE_BLEND_NUMS.index(4)))
    assert out == b"MAGIC\x20tail"
    assert len(line_blend(a, b + b"extra", Scripted(B1, 0))) == len(a)


def test_line_blend_declines_without_drawing():
    assert line_blend(b"abc", b"abc", NoDraw()) == b"abc"
    assert line_blend(b"", b"x", NoDraw()) == b""
    assert line_blend(b"abc", b"", NoDraw()) == b"abc"


# --- field_minmax -----------------------------------------------------------
def test_field_minmax_min_and_max_per_byte():
    a, b = b"\x10\x90", b"\x20\x40"
    assert field_minmax(a, b, Scripted(B1, 0)) == b"\x10\x40"
    assert field_minmax(a, b, Scripted(B1, 1)) == b"\x20\x90"


def test_field_minmax_wide_fields_compare_numerically_not_bytewise():
    a, b = be(0x01FF, w=2), be(0x0200, w=2)
    assert field_minmax(a, b, Scripted(B2, 0)) == a  # 0x01ff < 0x0200
    assert field_minmax(a, b, Scripted(B2, 1)) == b
    # a bytewise min would give 0x0100: falsification of that reading
    assert field_minmax(a, b, Scripted(B2, 0)) != b"\x01\x00"


def test_field_minmax_little_endian_orders_by_value():
    a, b = (0x0100).to_bytes(2, "little"), (0x00FF).to_bytes(2, "little")
    assert field_minmax(a, b, Scripted(L2, 0)) == b
    assert field_minmax(a, b, Scripted(L2, 1)) == a


def test_field_minmax_mixes_parents_per_field():
    a, b = be(5, 90, 7), be(9, 40, 7)
    assert field_minmax(a, b, Scripted(B4, 0)) == be(5, 40, 7)
    assert field_minmax(a, b, Scripted(B4, 1)) == be(9, 90, 7)


def test_field_minmax_min_and_max_are_ordered_and_idempotent():
    rnd = random.Random(1)
    for _ in range(40):
        a = bytes(rnd.randrange(256) for _ in range(12))
        b = bytes(rnd.randrange(256) for _ in range(12))
        lo = field_minmax(a, b, Scripted(B4, 0))
        hi = field_minmax(a, b, Scripted(B4, 1))
        assert all(x <= y for x, y in zip(fields(lo, B4), fields(hi, B4), strict=True))
        assert field_minmax(lo, b, Scripted(B4, 0)) == lo  # min(min(a,b), b) == min(a,b)
        assert len(lo) == len(hi) == len(a)


def test_field_minmax_declines_without_drawing():
    assert field_minmax(b"abc", b"abc", NoDraw()) == b"abc"
    assert field_minmax(b"", b"x", NoDraw()) == b""
    assert field_minmax(b"abc", b"", NoDraw()) == b"abc"


def test_field_minmax_short_donor_is_padded_with_the_base_tail():
    assert field_minmax(b"\x20\x77", b"\x10", Scripted(B1, 0)) == b"\x10\x77"


# --- nm_reflect -------------------------------------------------------------
@pytest.mark.parametrize(("param", "expect"), [(0, 0x34), (1, 0x40), (2, 0x58)])
def test_nm_reflect_contracts_reflects_and_expands(param, expect):
    # a=0x10 through the centroid 0x28 of b=0x30, c=0x20: alpha = 1/2, 1, 2
    assert nm_reflect(b"\x10", b"\x30", b"\x20", Scripted(B1, param)) == bytes([expect])


def test_nm_reflect_alpha_one_is_the_parallelogram_rule():
    rnd = random.Random(2)
    for _ in range(60):
        x, y, z = (rnd.randrange(0x40, 0xC0) for _ in range(3))
        want = clip(y + z - x, 1)
        assert nm_reflect(bytes([x]), bytes([y]), bytes([z]), Scripted(B1, 1)) == bytes([want])


def test_nm_reflect_reflects_through_a_single_point_when_donors_agree():
    assert nm_reflect(b"\x10", b"\x30", b"\x30", Scripted(B1, 1)) == b"\x50"


def test_nm_reflect_saturates_both_ways():
    assert nm_reflect(b"\xf0", b"\x10", b"\x10", Scripted(B1, 1)) == b"\x00"
    assert nm_reflect(b"\x00", b"\xf0", b"\xf0", Scripted(B1, 2)) == b"\xff"


def test_nm_reflect_wide_fields_and_endianness():
    # the differing bytes are the low two, so phase 2 aligns the 4-byte grid
    assert nm_reflect(be(100), be(900), be(500), Scripted(B4, 1, phase=2)) == be(1300)
    a, b, c = ((v).to_bytes(2, "little") for v in (0x0100, 0x0900, 0x0500))
    assert nm_reflect(a, b, c, Scripted(L2, 1)) == (0x0D00).to_bytes(2, "little")


def test_nm_reflect_keeps_bytes_where_all_three_agree():
    assert nm_reflect(b"AA\x10ZZ", b"AA\x30ZZ", b"AA\x20ZZ", Scripted(B1, 1)) == b"AA\x40ZZ"


def test_nm_reflect_declines_when_all_agree_without_drawing():
    assert nm_reflect(b"abc", b"abc", b"abc", NoDraw()) == b"abc"
    assert nm_reflect(b"", b"x", b"y", NoDraw()) == b""


# --- opposition -------------------------------------------------------------
def test_opposition_mirrors_within_the_range_of_three():
    # range [0x10, 0x30]: lo + hi - x
    assert opposition(b"\x10", b"\x30", b"\x20", Scripted(B1)) == b"\x30"
    assert opposition(b"\x30", b"\x10", b"\x20", Scripted(B1)) == b"\x10"
    assert opposition(b"\x25", b"\x10", b"\x30", Scripted(B1)) == b"\x1b"


def test_opposition_midpoint_is_a_fixed_point():
    assert opposition(b"\x20", b"\x10", b"\x30", Scripted(B1)) == b"\x20"


def test_opposition_never_leaves_the_range_of_the_parents():
    rnd = random.Random(3)
    for _ in range(80):
        a, b, c = (bytes(rnd.randrange(256) for _ in range(12)) for _ in range(3))
        out = opposition(a, b, c, Scripted(B4))
        for fo, fa, fb, fc in zip(*(fields(x, B4) for x in (out, a, b, c)), strict=True):
            assert min(fa, fb, fc) <= fo <= max(fa, fb, fc)


def test_opposition_wide_fields_are_numeric():
    a, b, c = be(0x01FF, w=2), be(0x0200, w=2), be(0x01FF, w=2)
    assert opposition(a, b, c, Scripted(B2)) == be(0x0200, w=2)  # a is the low end
    assert opposition(a, b, c, Scripted(B2)) != b"\x01\x00"


def test_opposition_declines_when_all_agree_without_drawing():
    assert opposition(b"abc", b"abc", b"abc", NoDraw()) == b"abc"
    assert opposition(b"", b"x", b"y", NoDraw()) == b""


def test_opposition_short_donors_padded_with_the_base_tail():
    assert opposition(b"\x10\x77", b"\x30", b"\x20", Scripted(B1)) == b"\x30\x77"


# --- all four on a real RandPool ---------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda a, b, c, r: line_blend(a, b, r),
        lambda a, b, c, r: field_minmax(a, b, r),
        lambda a, b, c, r: nm_reflect(a, b, c, r),
        lambda a, b, c, r: opposition(a, b, c, r),
    ],
)
def test_ops_run_on_randpool_and_keep_the_base_length(call):
    rnd = random.Random(4)
    rng = RandPool(seed=5)
    for _ in range(200):
        n = rnd.randrange(1, 40)
        a, b, c = (bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 50))) for _ in range(3))
        a = a[:n]
        out = call(a, b, c, rng)
        assert isinstance(out, bytes)
        assert len(out) == len(a)


# --- engine wiring ----------------------------------------------------------
class _Fuzzer:
    def __init__(self, corpus, max_len=64):
        self.corpus = corpus
        self.seed_meta = {}
        self.max_len = max_len
        self._rng = RandPool(seed=11)


def engine(corpus, max_len=64):
    return OperatorEngine(_Fuzzer(corpus, max_len))


_NAMES = ("line_blend", "field_minmax", "nm_reflect", "opposition")


def test_registered_structural_and_needs_no_lineage():
    for name in _NAMES:
        assert name in REGISTRY.names()
        assert REGISTRY.category_of(name) == "structural"
        assert name not in _AVAILABLE  # donors come from _distinct_donors, not parent_key


@pytest.mark.parametrize("op", ["_op_line_blend", "_op_field_minmax"])
def test_engine_two_seed_ops_need_one_distinct_donor(op):
    # crossing fields (a < b in the first, a > b in the second): min/max mix the parents
    a, b = bytearray(be(1, 70)), bytearray(be(9, 7))
    assert getattr(engine([a]), op)(a[:], 0, a) is None
    assert getattr(engine([a, bytearray(a)]), op)(a[:], 0, a) is None  # duplicate of the base
    seeds = [a, b]
    got = {bytes(r) for _ in range(60) if (r := getattr(engine(seeds), op)(a[:], 0, a)) is not None}
    assert got
    assert all(len(x) == 8 and x not in (bytes(a), bytes(b)) for x in got)


def test_field_minmax_aligned_grid_over_ordered_parents_returns_a_parent():
    a, b = be(1, 2), be(9, 7)  # a < b in every field
    # phase 3 aligns the 4-byte grid with the integers: min is a, max is b
    assert field_minmax(a, b, Scripted(B4, 0, phase=3)) == a
    assert field_minmax(a, b, Scripted(B4, 1, phase=3)) == b


def test_engine_field_minmax_never_returns_a_parent():
    # a misaligned grid mixes bytes of both parents; an aligned one repeats a parent -> declined
    a, b = bytearray(be(1, 2)), bytearray(be(9, 7))
    eng = engine([a, b])
    outs = [eng._op_field_minmax(a[:], 0, a) for _ in range(200)]
    assert all(o is None or bytes(o) not in (bytes(a), bytes(b)) for o in outs)


@pytest.mark.parametrize("op", ["_op_nm_reflect", "_op_opposition"])
def test_engine_three_seed_ops_need_two_distinct_donors(op):
    a, b = bytearray(be(1, 2)), bytearray(be(9, 7))
    assert getattr(engine([a]), op)(a[:], 0, a) is None
    assert getattr(engine([a, b]), op)(a[:], 0, a) is None
    assert getattr(engine([a, bytearray(a), b]), op)(a[:], 0, a) is None
    seeds = [bytearray(be(i * 1000, i * 77, i * 5)) for i in (1, 4, 9)]
    got = {
        bytes(r)
        for _ in range(60)
        if (r := getattr(engine(seeds), op)(seeds[0][:], 0, seeds[0])) is not None
    }
    assert got
    assert all(len(x) == 12 and x != bytes(seeds[0]) for x in got)


@pytest.mark.parametrize("op", [f"_op_{n}" for n in _NAMES])
def test_engine_truncates_to_max_len_and_declines_on_empty(op):
    seeds = [bytearray(be(i * 1000, i * 77, i * 5)) for i in (1, 4, 9)]
    eng = engine(seeds, max_len=5)
    outs = [getattr(eng, op)(seeds[0][:], 0, seeds[0]) for _ in range(40)]
    assert all(o is None or len(o) <= 5 for o in outs)
    assert getattr(engine(seeds), op)(bytearray(), 0, seeds[0]) is None
