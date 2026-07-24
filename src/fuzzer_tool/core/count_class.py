"""Logarithmic count classification for edge hit counts.

Ports AFL's count_class_lookup16 table that bucketizes raw 0-255 hit counts
into 8 logarithmic classes: 0, 1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+.

This normalizes edge frequencies before comparison, preventing the fuzzer
from distinguishing between "hit 50 times" and "hit 100 times" when both
are in the same bucket. It reduces noise and improves deduplication.

The u16 lookup table classifies TWO bytes at once: it maps
(count_lo | count_hi << 8) to (class_lo | class_hi << 8), giving
O(1) classification for both bytes per table lookup.
"""


def _classify_byte(val: int) -> int:
    """Classify a single hit count value."""
    if val == 0:
        return 0
    if val == 1:
        return 1
    if val == 2:
        return 2
    if val == 3:
        return 3
    if val <= 7:
        return 4
    if val <= 15:
        return 8
    if val <= 31:
        return 16
    if val <= 127:
        return 32
    return 128


def _build_u16_table() -> list[int]:
    """Build a 65536-entry lookup table that classifies 2 bytes at once.

    For a u16 value v = lo | (hi << 8), the entry is:
        classify(lo) | (classify(hi) << 8)

    This lets us classify an entire trace buffer in half the iterations.
    """
    table = [0] * 65536
    for lo in range(256):
        cl = _classify_byte(lo)
        for hi in range(256):
            ch = _classify_byte(hi)
            table[lo | (hi << 8)] = cl | (ch << 8)
    return table


# Precomputed lookup table
LOOKUP_U16: list[int] = _build_u16_table()


def classify_counts(trace_bits: bytearray | bytes) -> bytearray:
    """Classify edge hit counts using the logarithmic lookup table.

    Processes the trace buffer 2 bytes at a time using the u16 table.
    Each byte's count is independently bucketized into one of 9 classes.

    Args:
        trace_bits: Raw edge bitmap where each byte is a hit count (0-255).

    Returns:
        Classified trace bitmap (new bytearray with bucketized values).
    """
    result = bytearray(trace_bits)
    length = len(result)
    _lookup = LOOKUP_U16  # local var avoids global lookup overhead

    # Process 2 bytes at a time — unrolled for the common 65K-iteration case
    i = 0
    end = length - 1
    while i < end:
        raw = result[i] | (result[i + 1] << 8)
        classified = _lookup[raw]
        result[i] = classified & 0xFF
        result[i + 1] = (classified >> 8) & 0xFF
        i += 2

    # Handle odd trailing byte
    if length & 1:
        result[length - 1] = _lookup[result[length - 1]]

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
