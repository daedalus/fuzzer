"""Logarithmic count classification for edge hit counts.

Ports AFL's count_class_lookup16 table that bucketizes raw 0-255 hit counts
into 10 logarithmic classes: 0, 1, 2, 3, 4-7, 8-15, 16-31, 32-63, 64-127,
128+.

Note that this is one class finer than the *bucket bit* ladder further
down, which merges 32-63 with 64-127 to stay bit-identical to AFL's
count_class_lookup8. The two are supposed to differ; the docstring here
previously claimed the coarser ladder for both, which is why the
enumeration in tests/test_count_class_exhaustive.py pins each separately.

This normalizes edge frequencies before comparison, preventing the fuzzer
from distinguishing between "hit 50 times" and "hit 100 times" when both
are in the same bucket. It reduces noise and improves deduplication.

The u16 lookup table classifies TWO bytes at once: it maps
(count_lo | count_hi << 8) to (class_lo | class_hi << 8), giving
O(1) classification for both bytes per table lookup.

NumPy path: classify_counts and new_bits are vectorized. numpy is an
unconditional import here, so the u16-table loop in classify_counts is
reachable only for an empty buffer and LOOKUP_U16 is never built in
normal operation -- it is kept, and tested, as the reference the table
packing is checked against.

Separately, ``bucket_bit`` / ``bucket_bits`` map a raw count to AFL's
count_class_lookup8 *bit*, which is what a virgin map needs. See the
comment above them for why the classify_* representative values cannot
be reused for that.
"""

from array import array

import numpy as np

_NP_CLASSIFY_TABLE = np.array(
    [
        _classify_byte(i)
        if (
            _classify_byte := lambda val: val if val <= 3 else min(1 << (val.bit_length() - 1), 128)
        )
        else 0
        for i in range(256)
    ],
    dtype=np.uint8,
)


def _classify_byte(val: int) -> int:
    """Classify a single hit count value.

    Maps 0-3 to identity, then largest power-of-2 <= val, capped at 128.
    """
    if val <= 3:
        return val
    b = 1 << (val.bit_length() - 1)
    return min(b, 128)


def _build_u16_table() -> array:
    """Build a 65536-entry lookup table that classifies 2 bytes at once.

    For a u16 value v = lo | (hi << 8), the entry is:
        classify(lo) | (classify(hi) << 8)

    This lets us classify an entire trace buffer in half the iterations.
    Uses array('H') (2 bytes per entry) instead of list[int] (~28 bytes
    per entry) to reduce retained memory from ~2.6 MB to ~128 KB.
    """
    table = array("H", [0]) * 65536
    for lo in range(256):
        cl = _classify_byte(lo)
        for hi in range(256):
            ch = _classify_byte(hi)
            table[lo | (hi << 8)] = cl | (ch << 8)
    return table


# Lazily built — only constructed on first access via __getattr__.
# In normal operation numpy is always available (imported above), so
# classify_counts() uses the vectorized numpy path and LOOKUP_U16 is
# never needed.  Building it at import time wasted ~2.6 MB of retained
# memory for a code path that is effectively dead.
def __getattr__(name: str):
    """PEP 562 module-level lazy attribute resolution."""
    if name == "LOOKUP_U16":
        table = _build_u16_table()
        globals()["LOOKUP_U16"] = table
        return table
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def classify_counts(trace_bits):
    """Classify edge hit counts using the logarithmic lookup table.

    Each byte's count is independently bucketized into one of 9 classes.
    Uses numpy when available (~120x faster on 131K buffers).

    Args:
        trace_bits: Raw edge bitmap where each byte is a hit count (0-255).

    Returns:
        Classified trace bitmap.
    """
    if len(trace_bits) > 0:
        arr = (
            np.frombuffer(trace_bits, dtype=np.uint8)
            if isinstance(trace_bits, bytes | bytearray)
            else np.asarray(trace_bits, dtype=np.uint8)
        )
        return bytearray(_NP_CLASSIFY_TABLE[arr])

    result = bytearray(trace_bits)
    length = len(result)

    i = 0
    end = length - 1
    while i < end:
        raw = result[i] | (result[i + 1] << 8)
        classified = LOOKUP_U16[raw]  # noqa: F821 — resolved via module __getattr__
        result[i] = classified & 0xFF
        result[i + 1] = (classified >> 8) & 0xFF
        i += 2

    if length & 1:
        result[length - 1] = LOOKUP_U16[result[length - 1]]  # noqa: F821 — resolved via module __getattr__

    return result


def classify_single(val: int) -> int:
    """Classify a single hit count value.

    Returns one of: 0, 1, 2, 3, 4, 8, 16, 32, 64, 128 -- ten values, not
    the nine this docstring used to list. 64 is reachable (any count in
    64..127 classifies to it); omitting it made the class ladder read as
    if it matched the eight-bucket ``bucket_bit`` ladder, which it does
    not.
    """
    return _classify_byte(val)


# ── Bucket bits (for virgin maps) ────────────────────────────────────
#
# _classify_byte() returns a *representative value* per bucket
# (0, 1, 2, 3, 4, 8, 16, 32, 64, 128).  Those are not disjoint bits:
# class 3 is 0b11, which is class 1 OR'd with class 2.  That is harmless
# when comparing two classified traces byte-for-byte, but it is wrong for
# a virgin map, which accumulates by OR — an edge seen once and then
# twice leaves virgin == 0b11, and a later hit count of exactly 3 then
# reports no new bucket and is silently dropped.
#
# AFL's count_class_lookup8 avoids this by giving every bucket its own
# bit.  BUCKET_BIT_TABLE is that mapping, kept separate so the existing
# classify_* semantics stay exactly as they are:
#
#     count      bit
#     0          0x00   (empty slot — never sets a bit)
#     1          0x01
#     2          0x02
#     3          0x04
#     4-7        0x08
#     8-15       0x10
#     16-31      0x20
#     32-127     0x40
#     128+       0x80
#
# The SHM `count` field is uint32, not AFL's uint8, so counts above 255
# are real values rather than a wrapped counter.  They are clamped into
# the 128+ bucket to keep this ladder bit-identical to AFL's.  Extending
# it upward (256+, 1024+, …) would add buckets AFL does not have and
# would change which inputs are judged interesting, so it belongs in its
# own patch with its own A/B rather than riding along here.

BUCKET_COUNT = 8  # number of non-empty buckets; bits fit in a uint8


def bucket_bit(count: int) -> int:
    """Map a raw hit count to its bucket bit.

    Returns 0 for count <= 0 (an empty slot occupies no bucket), otherwise
    exactly one of the eight bits above.
    """
    if count <= 0:
        return 0
    if count <= 3:
        return 1 << (count - 1)  # 1, 2, 3 -> 0x01, 0x02, 0x04
    if count <= 7:
        return 0x08
    if count <= 15:
        return 0x10
    if count <= 31:
        return 0x20
    if count <= 127:
        return 0x40  # AFL merges 32-63 and 64-127 into one bucket
    return 0x80


BUCKET_BIT_TABLE = np.array([bucket_bit(i) for i in range(256)], dtype=np.uint8)


def bucket_bits(counts) -> np.ndarray:
    """Vectorized ``bucket_bit`` over an array of raw hit counts.

    Args:
        counts: Array-like of raw counts (uint32 in the SHM edge table).

    Returns:
        uint8 array of bucket bits, one per input count.  Counts above
        255 clamp into the 128+ bucket.
    """
    arr = np.asarray(counts)
    if arr.size == 0:
        return np.zeros(0, dtype=np.uint8)
    clamped = np.minimum(arr, 255).astype(np.uint8)
    return BUCKET_BIT_TABLE[clamped]


def _as_u8(buf, length: int) -> np.ndarray:
    """View the first ``length`` bytes of a buffer as uint8 without copying."""
    if isinstance(buf, bytes | bytearray | memoryview):
        return np.frombuffer(memoryview(buf)[:length], dtype=np.uint8)
    return np.asarray(buf, dtype=np.uint8)[:length]


def new_bits(
    trace: bytes | bytearray,
    virgin: bytes | bytearray,
) -> int:
    """Check whether a bucketed trace contributes coverage a virgin map lacks.

    Both arguments hold *bucket bits* as produced by ``bucket_bit`` /
    ``bucket_bits``: one disjoint bit per bucket, 0 for an untouched slot.
    The virgin map accumulates by OR, so ``trace & ~virgin`` is exactly the
    set of bucket bits this run contributed.

    AFL's ``has_new_bits``, in this module's non-inverted representation:

    Returns:
        0 = nothing new; every bit in the trace is already in the map
        1 = a known edge landed in a bucket the map had not recorded
        2 = an edge the map had never seen at all (its slot was 0)

    Note that 1 is *not* "the two maps overlap". An input replayed against a
    map that already contains it returns 0. Prior to the P2-6 exhaustive
    sweep this returned 1 for any overlap, so a byte-identical rerun
    reported new coverage; see tests/test_count_class_exhaustive.py.
    """
    length = min(len(trace), len(virgin))
    if length == 0:
        return 0

    t = _as_u8(trace, length)
    v = _as_u8(virgin, length)

    # Bits present in the trace and absent from the map. Vectorised over the
    # whole buffer rather than word-at-a-time: the previous hand-rolled
    # 8-byte loop and its byte-wise tail implemented two *different*
    # contracts, so the answer depended on where a byte sat relative to an
    # 8-byte boundary.
    contributed = t & ~v
    if not contributed.any():
        return 0

    # A slot the map has never touched. Where v == 0, contributed == t, so
    # this is the subset of the above with an empty virgin slot.
    if np.any(v == 0):
        return 2 if np.any((v == 0) & (t != 0)) else 1
    return 1
