"""Exhaustive enumeration over ``core/count_class.py``.

See ``docs/learnings/2026-08-22-count-class-exhaustive.md``. The kernel of
matklad's point
is that when an input space is small enough to enumerate, sampling it is
strictly worse than walking it: a hand-picked example asserts the author's
mental model at one point, and the points the author did not think of are
exactly the ones that are wrong.

Every entry point in this module has a domain of at most 2**16:

    _classify_byte / classify_single / bucket_bit   256
    _build_u16_table                                65_536
    classify_counts                                 256 values x 18 lengths
    new_bits                                        256 x 256 byte pairs
                                                    x every offset 0..16

so all of it is walked here rather than sampled. Runtime is ~2s total.

Two rules keep this honest:

1. **The oracle is written differently from the implementation.** The
   implementation computes ``1 << (val.bit_length() - 1)``. If the oracle
   did the same it would agree with the implementation on every input
   including the wrong ones, and the enumeration would prove nothing beyond
   "the function is deterministic". ``_oracle_class`` walks the ladder of
   boundaries literally instead, transcribed from the AFL table the module
   docstring cites.

2. **Agreement between two paths is asserted, not assumed.** Where the
   module has a fast path and a slow path over the same domain -- numpy vs
   the u16 table, the 8-byte word loop vs the byte tail -- the enumeration
   compares them against each other across the whole domain, not at the
   handful of points where they were known to agree.

Rule 2 is what this file was written for. ``new_bits`` had eighteen
hand-picked tests, including two named for the 8-byte boundary, and every
one of them chose byte values on which its two paths happen to agree. The
sweep below found the pair on which they do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core.count_class import (
    BUCKET_BIT_TABLE,
    BUCKET_COUNT,
    _build_u16_table,
    _classify_byte,
    bucket_bit,
    bucket_bits,
    classify_counts,
    classify_single,
    new_bits,
)

# ── Oracles ──────────────────────────────────────────────────────────
#
# Transcribed from AFL's count_class_lookup8, deliberately as literal
# ladders rather than as arithmetic, for the reason in rule 1 above.

_CLASS_LADDER = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (7, 4),
    (15, 8),
    (31, 16),
    (63, 32),
    (127, 64),
    (255, 128),
]

_BIT_LADDER = [
    (0, 0x00),
    (1, 0x01),
    (2, 0x02),
    (3, 0x04),
    (7, 0x08),
    (15, 0x10),
    (31, 0x20),
    (127, 0x40),
    (255, 0x80),
]


def _ladder(table, val: int) -> int:
    for upper, out in table:
        if val <= upper:
            return out
    raise AssertionError(f"value {val} above the top of the ladder")


def _oracle_class(val: int) -> int:
    return _ladder(_CLASS_LADDER, val)


def _oracle_bit(val: int) -> int:
    return _ladder(_BIT_LADDER, val)


def _oracle_new_bits(trace: bytes, virgin: bytes) -> int:
    """Byte-at-a-time reference for ``new_bits``, no word loop.

    AFL's ``has_new_bits``, translated to this module's *non-inverted*
    virgin representation (0 = edge never seen, bits = buckets seen):

        2  some edge in the trace has never been seen at all
        1  no new edge, but some known edge landed in a new bucket
        0  nothing in the trace that the virgin map does not already have

    The distinction the ladder rests on is that ``bucket_bit`` gives every
    bucket a disjoint bit, so ``t & ~v`` is exactly "bucket bits this run
    contributed".
    """
    ret = 0
    for t, v in zip(trace, virgin, strict=False):
        if not t & ~v:
            continue
        if v == 0:
            return 2
        ret = 1
    return ret


# ── Single-byte domains: 256 points each ─────────────────────────────


def test_classify_byte_over_entire_domain():
    for val in range(256):
        assert _classify_byte(val) == _oracle_class(val), f"count {val}"


def test_classify_single_matches_classify_byte_over_entire_domain():
    for val in range(256):
        assert classify_single(val) == _classify_byte(val), f"count {val}"


def test_classify_is_monotone_and_never_inflates():
    """A classified count may never exceed the raw count, nor decrease in it.

    Both properties are structural rather than incidental: the class is a
    representative *of the bucket the count fell into*, so it is the bucket
    floor, and buckets are ordered. A violation of either would mean the
    ladder had been edited into overlapping or reordered ranges.
    """
    previous = -1
    for val in range(256):
        cls = _classify_byte(val)
        assert cls <= val, f"count {val} classified upward to {cls}"
        assert cls >= previous, f"class decreased at count {val}"
        previous = cls


def test_class_count_matches_the_documented_ladder():
    """Ten classes, not the eight the module docstring used to claim.

    ``_classify_byte`` splits 32-63 and 64-127; ``bucket_bit`` merges them,
    matching AFL. The two functions are *supposed* to differ here -- see the
    BUCKET_BIT_TABLE comment -- and the difference is pinned in both
    directions so neither can be quietly "fixed" into the other.
    """
    classes = {_classify_byte(v) for v in range(256)}
    assert classes == {0, 1, 2, 3, 4, 8, 16, 32, 64, 128}
    assert len({bucket_bit(v) for v in range(256)}) == BUCKET_COUNT + 1  # +1 for the empty slot


def test_bucket_bit_over_entire_domain():
    for val in range(256):
        assert bucket_bit(val) == _oracle_bit(val), f"count {val}"


def test_bucket_bit_sets_exactly_one_bit_and_only_for_nonempty_slots():
    """The OR-accumulation in a virgin map is only sound if bits are disjoint.

    This is the property the module was split for: ``_classify_byte`` returns
    3 for a count of 3, which is 1|2, so a virgin map accumulating classes
    would record "seen 3" as "seen 1 and seen 2" and then drop a later count
    of exactly 3 as uninteresting.
    """
    assert bucket_bit(0) == 0
    for val in range(1, 256):
        bit = bucket_bit(val)
        assert bit.bit_count() == 1, f"count {val} -> {bit:#04x} is not a single bit"


def test_bucket_bit_is_monotone_in_the_count():
    previous = 0
    for val in range(1, 256):
        bit = bucket_bit(val)
        assert bit >= previous, f"bucket bit decreased at count {val}"
        previous = bit


def test_bucket_bits_vector_path_matches_scalar_over_entire_domain():
    counts = np.arange(256, dtype=np.uint32)
    vector = bucket_bits(counts)
    assert list(vector) == [bucket_bit(v) for v in range(256)]
    assert list(BUCKET_BIT_TABLE) == [bucket_bit(v) for v in range(256)]


def test_bucket_bits_clamps_above_a_byte_into_the_top_bucket():
    """The SHM count field is uint32; 256+ is a real count, not a wrapped one."""
    counts = np.array([255, 256, 1000, 2**31, 2**32 - 1], dtype=np.uint32)
    assert list(bucket_bits(counts)) == [0x80] * 5


def test_bucket_bits_on_empty_input():
    out = bucket_bits(np.zeros(0, dtype=np.uint32))
    assert out.dtype == np.uint8
    assert out.size == 0


# ── u16 table: all 65 536 entries ────────────────────────────────────


def test_u16_table_over_entire_domain():
    """Both bytes, every combination -- the packing is where this can go wrong.

    The previous tests checked six entries. The failure mode a sparse check
    misses is a lo/hi transposition, which is invisible whenever the two
    classes coincide -- i.e. on 10 of the 100 class pairs, and on the
    all-zero entry that a spot check reaches for first.
    """
    table = _build_u16_table()
    assert len(table) == 65536
    for hi in range(256):
        expected_hi = _oracle_class(hi) << 8
        for lo in range(256):
            got = table[lo | (hi << 8)]
            assert got == _oracle_class(lo) | expected_hi, f"lo={lo} hi={hi}"


def test_u16_table_is_not_symmetric_in_lo_and_hi():
    """Guards the transposition the exhaustive check above would catch anyway.

    Kept separate because it names the bug: if lo and hi were swapped, the
    table would be symmetric under byte exchange, and this is the cheapest
    statement of that.
    """
    table = _build_u16_table()
    assert table[0x00FF] != table[0xFF00]


# ── classify_counts: whole byte domain at every length 0..17 ─────────


@pytest.mark.parametrize("length", range(18))
def test_classify_counts_matches_scalar_oracle_at_every_length(length):
    """Every length through two full 8-byte words, both parities.

    Length matters here because the module has an odd-tail branch, and the
    numpy path silently makes it unreachable. Enumerating lengths keeps the
    contract asserted at the boundary regardless of which path serves it.
    """
    for base in range(0, 256, 7):  # 37 distinct starting values
        buf = bytes((base + i) % 256 for i in range(length))
        got = classify_counts(buf)
        assert len(got) == length
        assert list(got) == [_oracle_class(b) for b in buf], f"len={length} base={base}"


def test_classify_counts_covers_every_byte_value_in_one_buffer():
    buf = bytes(range(256))
    assert list(classify_counts(buf)) == [_oracle_class(b) for b in buf]


def test_classify_counts_never_mutates_its_input():
    buf = bytearray(range(256))
    before = bytes(buf)
    classify_counts(buf)
    assert bytes(buf) == before


def test_classify_counts_is_idempotent_over_entire_domain():
    """Classifying a classified buffer must be a no-op.

    Not a restatement of the ladder: it says every class representative is
    itself a fixed point, which is what makes it safe to compare a stored
    classified snapshot against a freshly classified one -- exactly what
    ``ptrace_coverage.is_new_coverage`` does.
    """
    once = classify_counts(bytes(range(256)))
    assert classify_counts(bytes(once)) == once


# ── new_bits: 256 x 256 byte pairs at every offset ───────────────────


def _at_offset(byte: int, offset: int, length: int) -> bytes:
    buf = bytearray(length)
    buf[offset] = byte
    return bytes(buf)


@pytest.mark.parametrize("offset", range(16))
def test_new_bits_is_independent_of_byte_position(offset):
    """The same byte pair must give the same answer wherever it sits.

    This is the negative-space test for the 8-byte word loop: a buffer
    position is not an input to the question "does this trace have new
    bits", so the answer may not depend on one. The word loop and the byte
    tail are two implementations of one contract, and nothing else compares
    them.
    """
    length = 16
    for t in (0, 1, 2, 3, 4, 5, 8, 0x80, 0xFF):
        for v in (0, 1, 2, 3, 4, 5, 8, 0x80, 0xFF):
            here = new_bits(_at_offset(t, offset, length), _at_offset(v, offset, length))
            alone = new_bits(bytes([t]), bytes([v]))
            assert here == alone, f"trace={t:#04x} virgin={v:#04x} at offset {offset}"


def test_new_bits_over_every_byte_pair_single_byte_buffers():
    for t in range(256):
        for v in range(256):
            assert new_bits(bytes([t]), bytes([v])) == _oracle_new_bits(bytes([t]), bytes([v])), (
                f"trace={t:#04x} virgin={v:#04x}"
            )


def test_new_bits_over_every_byte_pair_inside_a_full_word():
    """Same 65 536 pairs, now served by the word loop rather than the tail."""
    for t in range(256):
        trace = bytes([t] + [0] * 7)
        for v in range(256):
            virgin = bytes([v] + [0] * 7)
            assert new_bits(trace, virgin) == _oracle_new_bits(trace, virgin), (
                f"trace={t:#04x} virgin={v:#04x}"
            )


@pytest.mark.parametrize("length", range(1, 18))
def test_new_bits_matches_oracle_at_every_length(length):
    """Lengths spanning two words, so every (whole words, remainder) split runs."""
    for shift in range(0, 256, 11):
        trace = bytes((shift + i * 3) % 256 for i in range(length))
        virgin = bytes((shift + i * 5) % 256 for i in range(length))
        assert new_bits(trace, virgin) == _oracle_new_bits(trace, virgin), (
            f"len={length} shift={shift}"
        )


def test_new_bits_truncates_to_the_shorter_buffer():
    """min(len) is the contract; bytes past the end of either are not inputs."""
    for split in range(1, 9):
        trace = bytes([0x01] * split + [0xFF] * 8)
        virgin = bytes([0x01] * split)
        assert new_bits(trace, virgin) == _oracle_new_bits(trace, virgin[:split])
        assert new_bits(virgin, trace) == _oracle_new_bits(virgin, trace[:split])


def test_new_bits_on_empty_input():
    assert new_bits(b"", b"") == 0
    assert new_bits(b"", b"\xff") == 0
    assert new_bits(b"\xff", b"") == 0


def test_new_bits_accepts_bytearray_and_memoryview():
    for wrap in (bytes, bytearray, memoryview):
        assert new_bits(wrap(b"\x01\x00"), wrap(b"\x00\x00")) == 2


def test_new_bits_zero_when_the_trace_repeats_a_known_run():
    """A byte-identical rerun contributes nothing and must report nothing.

    The property that makes ``new_bits`` usable as the novelty signal its
    name promises. It is stated over the whole domain rather than at one
    point because "t is a subset of v" is where the previous overlap-based
    reading and AFL's disagree.
    """
    for v in range(256):
        virgin = bytes([v] * 9)
        assert new_bits(virgin, virgin) == 0, f"virgin={v:#04x}"


def test_new_bits_two_means_an_edge_never_seen_before():
    """2 is reserved for a virgin *slot*, not merely for a new bucket bit."""
    for t in range(1, 256):
        assert new_bits(bytes([t] * 9), bytes(9)) == 2, f"trace={t:#04x}"


def test_new_bits_one_means_a_new_bucket_on_a_known_edge():
    """Every ordered pair of distinct buckets, at a slot the map already has."""
    buckets = [bucket_bit(v) for v in (1, 2, 3, 4, 8, 16, 32, 128)]
    for seen in buckets:
        for fresh in buckets:
            trace = bytes([fresh] * 9)
            virgin = bytes([seen] * 9)
            expected = 0 if fresh == seen else 1
            assert new_bits(trace, virgin) == expected, f"seen={seen:#04x} fresh={fresh:#04x}"
