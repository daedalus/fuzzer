"""Tests for line_ending_swap, nul_inject (core/mutations/text_edges.py) and
masked_merge, position_marginal (core/mutations/generic.py), plus engine wiring.

Randomness is scripted: every function draws only through ``rng.randrange(n)``.
"""

import random

import pytest

from fuzzer_tool.core.mutations.generic import _MERGE_MASKS, masked_merge, position_marginal
from fuzzer_tool.core.mutations.text_edges import line_ending_swap, nul_inject
from fuzzer_tool.core.operator_registry import _AVAILABLE, REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine


class S:
    def __init__(self, *values):
        self._v = list(values)

    def randrange(self, n):
        v = self._v.pop(0)
        assert 0 <= v < n, f"scripted draw {v} outside [0, {n})"
        return v


class NoDraw:
    def randrange(self, n):
        raise AssertionError("declined call drew randomness")


# --- line_ending_swap -------------------------------------------------------
# draws: source ending among those present (order CRLF, LF, CR), destination among
# the other three of (CRLF, LF, CR, LFCR), scope (0 = all, 1 = one occurrence),
# and, for scope 1, which occurrence.
def test_line_ending_swap_lf_to_crlf_everywhere():
    # present = [LF]; dst options for LF = [CRLF, CR, LFCR]; idx 0 -> CRLF
    assert line_ending_swap(b"a\nb\nc", S(0, 0, 0)) == b"a\r\nb\r\nc"


def test_line_ending_swap_crlf_to_lf_and_to_cr():
    # present = [CRLF]; dst options = [LF, CR, LFCR]
    assert line_ending_swap(b"a\r\nb\r\n", S(0, 0, 0)) == b"a\nb\n"
    assert line_ending_swap(b"a\r\nb\r\n", S(0, 1, 0)) == b"a\rb\r"
    assert line_ending_swap(b"a\r\nb", S(0, 2, 0)) == b"a\n\rb"


def test_line_ending_swap_single_occurrence_makes_mixed_endings():
    out = line_ending_swap(b"a\nb\nc\nd", S(0, 0, 1, 1))  # LF -> CRLF, scope one, the 2nd LF
    assert out == b"a\nb\r\nc\nd"


def test_line_ending_swap_cr_lone_is_distinguished_from_crlf():
    # a lone CR and a CRLF: present = [CRLF, CR]; pick CR (idx 1); its options are
    # [CRLF, LF, LFCR], so idx 1 is LF
    assert line_ending_swap(b"a\rb\r\nc", S(1, 1, 0)) == b"a\nb\r\nc"
    # pick CRLF (idx 0) -> LF (idx 0), the lone CR is untouched
    assert line_ending_swap(b"a\rb\r\nc", S(0, 0, 0)) == b"a\rb\nc"


def test_line_ending_swap_without_endings_declines_without_drawing():
    assert line_ending_swap(b"no newline here", NoDraw()) == b"no newline here"
    assert line_ending_swap(b"", NoDraw()) == b""


def test_line_ending_swap_always_changes_the_input_when_it_acts():
    rnd = random.Random(1)
    rng = RandPool(seed=2)
    alphabet = b"ab\r\n "
    for _ in range(300):
        data = bytes(alphabet[rnd.randrange(len(alphabet))] for _ in range(rnd.randrange(1, 30)))
        out = line_ending_swap(data, rng)
        has = b"\n" in data or b"\r" in data
        assert (out != data) == has


# --- nul_inject ---------------------------------------------------------------
# draws: which printable run (>= 4 bytes), interior offset in the run, mode
# (0 = overwrite a byte with NUL, 1 = insert a NUL).
def test_nul_inject_overwrites_an_interior_byte():
    assert nul_inject(b"\x01hello\x02", S(0, 1, 0)) == b"\x01he\x00lo\x02"


def test_nul_inject_inserts_a_nul():
    assert nul_inject(b"\x01hello\x02", S(0, 1, 1)) == b"\x01he\x00llo\x02"


def test_nul_inject_picks_among_runs():
    data = b"abcd\xffwxyz"
    assert nul_inject(data, S(1, 0, 0)) == b"abcd\xffw\x00yz"


def test_nul_inject_never_touches_the_first_byte_of_a_run():
    for _ in range(100):
        out = nul_inject(b"\x01hello\x02", RandPool(seed=_))
        assert out[1:2] == b"h"


def test_nul_inject_needs_a_printable_run_of_four():
    assert nul_inject(b"abc\xffdef", NoDraw()) == b"abc\xffdef"
    assert nul_inject(b"", NoDraw()) == b""


# --- masked_merge ---------------------------------------------------------------
# draws: pattern index, phase within the pattern period.
def test_masked_merge_nibble_stripes():
    # mask 0xF0 everywhere: high nibbles from a, low nibbles from b
    assert masked_merge(b"\xab\xcd", b"\x12\x34", S(0, 0)) == b"\xa2\xc4"
    # mask 0x0F: the other way round
    assert masked_merge(b"\xab\xcd", b"\x12\x34", S(1, 0)) == b"\x1b\x3d"


def test_masked_merge_alternating_bytes_and_phase():
    idx = _MERGE_MASKS.index((0xFF, 0x00))
    assert masked_merge(b"\xaa\xbb\xcc\xdd", b"\x11\x22\x33\x44", S(idx, 0)) == b"\xaa\x22\xcc\x44"
    assert masked_merge(b"\xaa\xbb\xcc\xdd", b"\x11\x22\x33\x44", S(idx, 1)) == b"\x11\xbb\x33\xdd"


def test_masked_merge_is_a_bitwise_select():
    rnd = random.Random(3)
    masks = _MERGE_MASKS
    for _ in range(200):
        a = bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 20)))
        b = bytes(rnd.randrange(256) for _ in range(len(a)))
        pi, ph = rnd.randrange(len(masks)), None
        ph = rnd.randrange(len(masks[pi]))
        out = masked_merge(a, b, S(pi, ph))
        period = masks[pi][ph:] + masks[pi][:ph]
        for i, (x, y, o) in enumerate(zip(a, b, out, strict=True)):
            m = period[i % len(period)]
            assert o == (x & m) | (y & ~m & 0xFF)


def test_masked_merge_pads_a_short_donor_with_the_base_tail_and_keeps_length():
    out = masked_merge(b"\xaa\xbb\xcc", b"\x11", S(0, 0))
    # the donor's missing tail is the base's own, so only byte 0 can differ
    assert out == b"\xa1\xbb\xcc"


def test_masked_merge_empty_declines_without_drawing():
    assert masked_merge(b"", b"x", NoDraw()) == b""
    assert masked_merge(b"x", b"", NoDraw()) == b"x"


# --- position_marginal -----------------------------------------------------------
# draws: window length - 1, window start, then one parent index per position
# (0 = the base, i = the i-th donor).
def test_position_marginal_takes_each_byte_from_the_chosen_parent():
    base, d1, d2 = b"AAAA", b"BBBB", b"CCCC"
    out = position_marginal(base, [d1, d2], S(3, 0, 1, 2, 0, 1))  # len 4, start 0
    assert out == b"BCAB"


def test_position_marginal_leaves_bytes_outside_the_window():
    base, d1 = b"AAAAAA", b"BBBBBB"
    out = position_marginal(base, [d1, d1], S(1, 2, 1, 2))  # window len 2 at 2
    assert out == b"AABBAA"


def test_position_marginal_values_come_from_the_parents_at_that_position():
    rnd = random.Random(4)
    rng = RandPool(seed=6)
    for _ in range(200):
        n = rnd.randrange(1, 40)
        base = bytes(rnd.randrange(256) for _ in range(n))
        donors = [bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 50))) for _ in range(3)]
        out = position_marginal(base, donors, rng)
        padded = [d[:n] if len(d) >= n else d + base[len(d) :] for d in donors]
        assert len(out) == n
        for i, o in enumerate(out):
            assert o in {base[i], *(d[i] for d in padded)}


def test_position_marginal_declines_without_drawing_on_empty_or_no_donors():
    assert position_marginal(b"", [b"x"], NoDraw()) == b""
    assert position_marginal(b"abc", [], NoDraw()) == b"abc"


# --- engine wiring ----------------------------------------------------------------
class _Fuzzer:
    def __init__(self, corpus, max_len=64):
        self.corpus = corpus
        self.seed_meta = {}
        self.max_len = max_len
        self._rng = RandPool(seed=11)


def engine(corpus, max_len=64):
    return OperatorEngine(_Fuzzer(corpus, max_len))


_NAMES = (
    "varint_reencode",
    "sign_extend",
    "bcd_pack",
    "negabinary",
    "bit_compress",
    "bit_matrix_transpose",
    "timestamp_specials",
    "line_ending_swap",
    "nul_inject",
    "masked_merge",
    "position_marginal",
)


def test_all_eleven_registered_structural_and_need_no_lineage():
    for name in _NAMES:
        assert name in REGISTRY.names(), name
        assert REGISTRY.category_of(name) == "structural"
        assert name not in _AVAILABLE


_SOLO = (
    "varint_reencode",
    "sign_extend",
    "bcd_pack",
    "negabinary",
    "bit_compress",
    "bit_matrix_transpose",
    "timestamp_specials",
    "line_ending_swap",
    "nul_inject",
)


@pytest.mark.parametrize("name", _SOLO)
def test_engine_solo_ops_return_bytearrays_within_max_len_and_decline_on_empty(name):
    data = bytearray(b"id=1234\r\nname=hello\r\n" + bytes(range(40)))
    eng = engine([data], max_len=48)
    op = getattr(eng, f"_op_{name}")
    outs = [op(data[:], 0, data) for _ in range(80)]
    assert any(o is not None for o in outs)
    assert all(o is None or (isinstance(o, bytearray) and len(o) <= 48) for o in outs)
    assert op(bytearray(), 0, data) is None


@pytest.mark.parametrize("name", ["masked_merge"])
def test_engine_merge_needs_one_distinct_donor_and_never_returns_a_parent(name):
    a, b = bytearray(b"\x12\x34\x56\x78"), bytearray(b"\x9a\xbc\xde\xf0")
    op = getattr(engine([a]), f"_op_{name}")
    assert op(a[:], 0, a) is None
    eng = engine([a, b])
    outs = [getattr(eng, f"_op_{name}")(a[:], 0, a) for _ in range(120)]
    got = {bytes(o) for o in outs if o is not None}
    assert got and bytes(a) not in got and bytes(b) not in got


def test_engine_position_marginal_needs_two_donors_and_never_returns_the_base():
    a, b, c = bytearray(b"AAAAAAAA"), bytearray(b"BBBBBBBB"), bytearray(b"CCCCCCCC")
    assert engine([a])._op_position_marginal(a[:], 0, a) is None
    two = engine([a, b])
    assert two._op_position_marginal(a[:], 0, a) is None  # needs at least two donors
    eng = engine([a, b, c])
    outs = [eng._op_position_marginal(a[:], 0, a) for _ in range(120)]
    got = {bytes(o) for o in outs if o is not None}
    assert got and bytes(a) not in got
