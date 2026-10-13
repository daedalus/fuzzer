"""Tests for the pattern-table gaps found against AFL++/libFuzzer/honggfuzz/Radamsa.

Float bit patterns (``structured._FLOAT_PATTERNS``), Unicode sequences
(``_FUNNY_UNICODE``) and decimal width boundaries (``generic._DECIMAL_EDGES``,
drawn by ``ascii_num_replace``).  Oracles are Python's own float and Unicode
semantics, not the tables under test.
"""

import math
import struct
import sys
import unicodedata

import numpy as np
import pytest

from fuzzer_tool.core.mutations import _FUNNY_UNICODE, generic
from fuzzer_tool.core.mutations import structured as S
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine


def _d(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _f(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# --- float patterns -------------------------------------------------------------
_F64_MANT, _F32_MANT = (1 << 52) - 1, (1 << 23) - 1


def _pair_with(pred):
    return [(w, n) for w, n in S._FLOAT_PATTERNS if pred(_d(w), _f(n), w, n)]


def test_float_table_has_the_smallest_normal():
    hits = _pair_with(lambda d, f, w, n: d == sys.float_info.min and f == math.ldexp(1.0, -126))
    assert len(hits) == 1


def test_float_table_has_the_negative_extremes():
    assert _pair_with(
        lambda d, f, w, n: d == -sys.float_info.max and f == -math.ldexp(2 - 2**-23, 127)
    )
    assert _pair_with(
        lambda d, f, w, n: d == -math.ldexp(1.0, -1074) and f == -math.ldexp(1.0, -149)
    )


def test_float_table_has_a_nan_with_every_payload_bit_set():
    hits = _pair_with(
        lambda d, f, w, n: (
            math.isnan(d) and w & _F64_MANT == _F64_MANT and n & _F32_MANT == _F32_MANT
        )
    )
    assert hits


@pytest.mark.parametrize("delta", [-1, 0, 2])
def test_float_table_brackets_the_integer_exactness_boundary(delta):
    # 2**53 (double) and 2**24 (float) are the first values where x + 1 == x.
    hits = _pair_with(lambda d, f, w, n: d == 2.0**53 + delta and f == 2.0**24 + delta)
    assert len(hits) == 1


def test_exactness_boundary_entries_behave_as_labelled():
    wide, narrow = next((w, n) for w, n in S._FLOAT_PATTERNS if _d(w) == 2.0**53)
    assert _d(wide) + 1.0 == _d(wide)
    assert np.float32(_f(narrow)) + np.float32(1.0) == np.float32(_f(narrow))
    below = next(_d(w) for w, n in S._FLOAT_PATTERNS if _d(w) == 2.0**53 - 1)
    assert below + 1.0 != below


def test_float_squeeze_with_the_extended_table_keeps_length_and_always_writes():
    zeros = bytes(64)
    for seed in range(200):
        out = S.float_squeeze(zeros, RandPool(seed=seed))
        assert len(out) == 64
        assert out != zeros  # no table entry is all zero bits


def test_float_table_entries_are_unique():
    assert len(set(S._FLOAT_PATTERNS)) == len(S._FLOAT_PATTERNS)


# --- unicode ----------------------------------------------------------------------
CASE_GROW = (0x130, 0x23A, 0x23E, 0x149, 0x1F0, 0x3B0, 0x587, 0xFB01, 0xFB03, 0x1F80, 0x1FB3)
LINE_AND_BIDI = (0x85, 0x2028, 0x2029, 0x200E, 0x200F, 0x2066, 0x2067, 0x2068, 0x2069, 0x61C)


@pytest.mark.parametrize("cp", CASE_GROW + LINE_AND_BIDI)
def test_unicode_table_holds_the_code_point_once(cp):
    assert _FUNNY_UNICODE.count(chr(cp).encode("utf-8")) == 1


@pytest.mark.parametrize("cp", CASE_GROW)
def test_case_mapping_changes_the_length(cp):
    s = chr(cp)
    mapped = (s.upper(), s.lower())
    grows_bytes = any(len(m.encode()) > len(s.encode()) for m in mapped)
    grows_points = any(len(m) > 1 for m in mapped)
    assert grows_bytes or grows_points


@pytest.mark.parametrize("cp", LINE_AND_BIDI)
def test_line_and_bidi_entries_are_format_or_separator_characters(cp):
    assert unicodedata.category(chr(cp)) in {"Cc", "Cf", "Zl", "Zp"}


def test_bidi_isolates_have_the_isolate_classes():
    classes = [unicodedata.bidirectional(chr(cp)) for cp in (0x2066, 0x2067, 0x2068, 0x2069)]
    assert classes == ["LRI", "RLI", "FSI", "PDI"]


class _Fuzzer:
    def __init__(self, rng, max_len=64):
        self.corpus = [bytearray(b"abc")]
        self.seed_meta = {}
        self.max_len = max_len
        self._rng = rng


class _Pick:
    """choice() returns a fixed entry; randint() returns its upper bound (append)."""

    def __init__(self, entry):
        self.entry = entry

    def choice(self, seq):
        assert self.entry in seq
        return self.entry

    def randint(self, lo, hi):
        return hi


@pytest.mark.parametrize("cp", CASE_GROW + LINE_AND_BIDI)
def test_utf8_insert_appends_the_sequence_as_valid_utf8(cp):
    entry = chr(cp).encode("utf-8")
    eng = OperatorEngine(_Fuzzer(_Pick(entry)))
    buf = bytearray(b"abc")
    eng._op_utf8_insert(buf, 0, buf)
    assert bytes(buf) == b"abc" + entry
    assert bytes(buf).decode("utf-8") == "abc" + chr(cp)


def test_utf8_insert_declines_at_max_len():
    eng = OperatorEngine(_Fuzzer(_Pick(_FUNNY_UNICODE[0]), max_len=3))
    buf = bytearray(b"abc")
    eng._op_utf8_insert(buf, 0, buf)
    assert bytes(buf) == b"abc"


# --- decimal width boundaries -------------------------------------------------------
_EDGES = (
    127, 128, 255, 256, 32767, 32768, 65535, 65536,
    2147483647, 2147483648, 4294967295, 4294967296,
    9007199254740992,
    9223372036854775807, 9223372036854775808,
    18446744073709551615, 18446744073709551616,
)  # fmt: skip


@pytest.mark.parametrize("value", _EDGES)
def test_decimal_edges_hold_each_width_boundary(value):
    assert value in generic._DECIMAL_EDGES


class _Num:
    """randint() picks the strategy; choice() pops scripted indices."""

    def __init__(self, strategy, picks):
        self.strategy, self.picks = strategy, list(picks)

    def randint(self, lo, hi):
        assert (lo, hi) == (0, 4)
        return self.strategy

    def choice(self, seq):
        return seq[self.picks.pop(0)]


def test_random_num_boundary_strategy_returns_a_table_value():
    for i, edge in enumerate(generic._DECIMAL_EDGES):
        assert generic._random_num(_Num(4, [i])) == edge


def test_ascii_num_replace_writes_a_boundary_number():
    i = generic._DECIMAL_EDGES.index(2147483647)
    assert generic.ascii_num_replace(b"len=1234", _Num(4, [0, i])) == b"len=2147483647"


def test_ascii_num_replace_negates_a_boundary_for_a_negative_token():
    i = generic._DECIMAL_EDGES.index(2147483648)
    assert generic.ascii_num_replace(b"x=-12;", _Num(4, [0, i])) == b"x=-2147483648;"


def test_decimal_edges_are_ascending_and_unique():
    edges = generic._DECIMAL_EDGES
    assert list(edges) == sorted(set(edges))
