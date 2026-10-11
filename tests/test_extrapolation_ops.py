"""Tests for lineage_step, diff_evo, bit_majority and field_median.

Randomness is scripted (Hard Rule 39): shape, parameter (when the operator has
one), then phase (width > 1 only), so every expected child is exact.
"""

import pytest

from fuzzer_tool.core.mutations.generic import (
    bit_majority,
    diff_evo,
    field_median,
    lineage_step,
)
from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.corpus_manager import seed_key
from fuzzer_tool.services.operators import OperatorEngine

B1, B2, L2, B4, L4 = range(5)


class Scripted:
    def __init__(self, shape, param=None, phase=0):
        self._draws = [shape] + ([] if param is None else [param]) + [phase]

    def randrange(self, n):
        return self._draws.pop(0)


def be(*v, w=4):
    return b"".join(x.to_bytes(w, "big") for x in v)


# --- lineage_step -----------------------------------------------------------
def test_lineage_step_repeats_the_step():
    assert lineage_step(b"\x20", b"\x10", Scripted(B1, 0)) == b"\x30"  # t=1
    assert lineage_step(b"\x20", b"\x10", Scripted(B1, 1)) == b"\x40"  # t=2


def test_lineage_step_saturates_up_and_down():
    assert lineage_step(b"\xa0", b"\x10", Scripted(B1, 1)) == b"\xff"
    assert lineage_step(b"\x10", b"\xa0", Scripted(B1, 1)) == b"\x00"
    assert lineage_step(be(0xFFFF0000), be(0x00000000), Scripted(B4, 0)) == be(0xFFFFFFFF)


def test_lineage_step_wide_fields_little_endian():
    c = (0x0200).to_bytes(2, "little")
    p = (0x0100).to_bytes(2, "little")
    assert lineage_step(c, p, Scripted(L2, 1)) == (0x0400).to_bytes(2, "little")


def test_lineage_step_keeps_shared_bytes_and_length():
    p = b"MAGIC\x10tail"
    c = b"MAGIC\x20tail"
    assert lineage_step(c, p, Scripted(B1, 0)) == b"MAGIC\x30tail"


def test_lineage_step_equal_or_empty_is_declined():
    assert lineage_step(b"abc", b"abc", Scripted(B1, 0)) == b"abc"
    assert lineage_step(b"", b"x", Scripted(B1, 0)) == b""
    assert lineage_step(b"abc", b"", Scripted(B1, 0)) == b"abc"


def test_lineage_step_short_parent_padded_with_child_tail():
    assert lineage_step(b"\x20\x77", b"\x10", Scripted(B1, 0)) == b"\x30\x77"


# --- diff_evo ---------------------------------------------------------------
@pytest.mark.parametrize(("param", "expect"), [(0, 0x18), (1, 0x20), (2, 0x30)])
def test_diff_evo_scales_f_half_one_two(param, expect):
    # a + F * (0x30 - 0x20) with F = 1/2, 1, 2
    assert diff_evo(b"\x10", b"\x30", b"\x20", Scripted(B1, param)) == bytes([expect])


def test_diff_evo_half_step_floors_on_odd_difference():
    assert diff_evo(b"\x10", b"\x25", b"\x20", Scripted(B1, 0)) == b"\x12"  # 5>>1 = 2
    assert diff_evo(b"\x10", b"\x20", b"\x25", Scripted(B1, 0)) == b"\x0d"  # -5>>1 = -3


def test_diff_evo_saturates():
    assert diff_evo(b"\xf0", b"\xff", b"\x00", Scripted(B1, 2)) == b"\xff"
    assert diff_evo(b"\x10", b"\x00", b"\xff", Scripted(B1, 2)) == b"\x00"


def test_diff_evo_untouched_where_donors_agree():
    a = b"AA\x10ZZ"
    b = b"BB\x30ZZ"
    c = b"BB\x20ZZ"
    assert diff_evo(a, b, c, Scripted(B1, 1)) == b"AA\x20ZZ"


def test_diff_evo_equal_donors_declined():
    assert diff_evo(b"\x10", b"\x20", b"\x20", Scripted(B1, 1)) == b"\x10"


def test_diff_evo_four_byte_big_endian():
    out = diff_evo(be(1000), be(5000), be(3000), Scripted(B4, 1))
    assert out == be(3000)


# --- bit_majority -----------------------------------------------------------
def test_bit_majority_truth_table():
    assert bit_majority(bytes([0b1100]), bytes([0b1010]), bytes([0b0110])) == bytes([0b1110])
    assert bit_majority(b"\xf0", b"\xf0", b"\x0f") == b"\xf0"  # two agree -> kept


def test_bit_majority_keeps_shared_header_drops_unique_bits():
    a, b, c = b"HDR\x01", b"HDR\x02", b"HDR\x04"
    assert bit_majority(a, b, c) == b"HDR\x00"


def test_bit_majority_length_and_padding():
    # short donors padded with a's tail, so the tail is a's
    assert bit_majority(b"\xff\xaa\xbb", b"\x00", b"\x00") == b"\x00\xaa\xbb"
    assert bit_majority(b"", b"x", b"y") == b""
    assert bit_majority(b"\xff", b"\x00\x11\x22", b"\x00\x33") == b"\x00"


def test_bit_majority_idempotent_on_agreement():
    assert bit_majority(b"abc", b"abc", b"xyz") == b"abc"


# --- field_median -----------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b", "c", "expect"),
    [(0x10, 0x90, 0x30, 0x30), (0x90, 0x10, 0x30, 0x30), (0x30, 0x10, 0x90, 0x30), (5, 5, 9, 5)],
)
def test_field_median_one_byte(a, b, c, expect):
    assert field_median(bytes([a]), bytes([b]), bytes([c]), Scripted(B1)) == bytes([expect])


def test_field_median_never_invents_values():
    for w, shape in ((2, B2), (4, B4)):
        a, b, c = be(0x00FF, 0x1234, w=w), be(0x0100, 0x1200, w=w), be(0x0001, 0x1300, w=w)
        out = field_median(a, b, c, Scripted(shape))
        got = [int.from_bytes(out[i : i + w], "big") for i in (0, w)]
        assert got == [0x00FF, 0x1234]


def test_field_median_all_equal_declined_and_endian():
    assert field_median(b"abc", b"abc", b"abc", Scripted(B1)) == b"abc"
    a, b, c = (x.to_bytes(2, "little") for x in (0x0100, 0x0005, 0x0300))
    assert field_median(a, b, c, Scripted(L2)) == a


# --- engine wiring ----------------------------------------------------------
class _Fuzzer:
    def __init__(self, corpus, meta=None, max_len=64):
        self.corpus = corpus
        self.seed_meta = meta or {}
        self._use_lineage = True
        self.max_len = max_len
        self._rng = RandPool(seed=11)


def engine(corpus, meta=None):
    return OperatorEngine(_Fuzzer(corpus, meta))


def test_registered_structural_and_lineage_gate():
    for name in ("lineage_step", "diff_evo", "bit_majority", "field_median"):
        assert name in REGISTRY.names()
        assert REGISTRY.category_of(name) == "structural"
    assert _AVAILABLE["lineage_step"](_Fuzzer([]), None) is True
    assert _AVAILABLE["lineage_step"](type("N", (), {})(), None) is False
    for name in ("diff_evo", "bit_majority", "field_median"):
        assert name not in _AVAILABLE  # no lineage needed


def test_engine_lineage_step_resolves_parent_in_corpus():
    p, c = bytearray(be(0x100)), bytearray(be(0x180))
    meta = {bytes(c): {"parent_key": seed_key(bytes(p))}, bytes(p): {}}
    eng = engine([p, c], meta)
    out = eng._op_lineage_step(bytearray(c), 0, c)
    assert out is not None
    assert bytes(out) != bytes(c)


def test_engine_lineage_step_declines_without_resolvable_parent():
    c = bytearray(be(0x180))
    assert engine([c], {bytes(c): {"parent_key": "gone"}})._op_lineage_step(c[:], 0, c) is None
    assert engine([c], {bytes(c): {"parent_key": None}})._op_lineage_step(c[:], 0, c) is None
    assert engine([c], {})._op_lineage_step(c[:], 0, c) is None


@pytest.mark.parametrize("op", ["_op_diff_evo", "_op_bit_majority", "_op_field_median"])
def test_engine_three_parent_ops_need_two_distinct_donors(op):
    a, b, c = bytearray(be(1, 2)), bytearray(be(9, 7)), bytearray(be(5, 3))
    assert getattr(engine([a], {}), op)(a[:], 0, a) is None
    assert getattr(engine([a, b], {}), op)(a[:], 0, a) is None
    # duplicates of the base or of each other do not count as donors
    assert getattr(engine([a, bytearray(a), b], {}), op)(a[:], 0, a) is None
    out = getattr(engine([a, b, c], {}), op)(a[:], 0, a)
    assert out is None or len(out) <= 64


@pytest.mark.parametrize("op", ["_op_diff_evo", "_op_field_median"])
def test_engine_three_parent_ops_produce_children(op):
    seeds = [bytearray(be(i * 1000, i * 77, i * 5)) for i in (1, 4, 9)]
    got = {
        bytes(r)
        for _ in range(60)
        if (r := getattr(engine(seeds, {}), op)(seeds[0][:], 0, seeds[0])) is not None
    }
    assert got  # reachable
    assert all(len(x) == 12 for x in got)
