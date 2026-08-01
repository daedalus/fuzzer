"""Logarithmic count classification for edge hit counts.

Ports AFL's count_class_lookup16 table that bucketizes raw 0-255 hit counts
into 8 logarithmic classes: 0, 1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+.

This normalizes edge frequencies before comparison, preventing the fuzzer
from distinguishing between "hit 50 times" and "hit 100 times" when both
are in the same bucket. It reduces noise and improves deduplication.

The u16 lookup table classifies TWO bytes at once: it maps
(count_lo | count_hi << 8) to (class_lo | class_hi << 8), giving
O(1) classification for both bytes per table lookup.

NumPy path: when numpy is available, classify_counts and
classify_and_new_bits use vectorized operations for 100-400x speedup
over the pure-Python loop on 131K buffers.
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

    Returns one of: 0, 1, 2, 3, 4, 8, 16, 32, 128.
    """
    return _classify_byte(val)


def new_bits(
    trace: bytes | bytearray,
    virgin: bytes | bytearray,
) -> int:
    """Check if a classified trace has new coverage vs a virgin map.

    Implements AFL's has_new_bits semantics:
    - For each byte, if trace[i] & virgin[i] is nonzero: overlap (potential new info)
    - If trace[i] & ~virgin[i] is nonzero: trace has bits virgin doesn't (new edge)
    - If trace[i] is nonzero and virgin[i] is 0: entirely new edge

    Returns:
        0 = no new bits
        1 = overlap — trace has bits where virgin also has bits (count changed)
        2 = new edge — trace has bits where virgin is 0
    """
    length = min(len(trace), len(virgin))
    if length == 0:
        return 0

    has_overlap = False

    # Process 8 bytes at a time for efficiency
    i = 0
    while i + 7 < length:
        t = int.from_bytes(trace[i : i + 8], "little")
        v = int.from_bytes(virgin[i : i + 8], "little")

        # New edge: trace has bits where virgin is 0
        if t & ~v:
            return 2

        # Overlap: trace has bits where virgin also has bits
        if t & v:
            has_overlap = True

        i += 8

    # Handle remaining bytes
    while i < length:
        t = trace[i]
        v = virgin[i]

        if t and not v:
            return 2
        if t and v:
            has_overlap = True

        i += 1

    return 1 if has_overlap else 0
