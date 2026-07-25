"""TLV-aware mutation operator.

Scans input for potential length fields (bytes whose value matches the
distance to some boundary within the input) and mutates them with
boundary values: 0, 1, 0x7f, 0x80, 0xff, exact-remaining, off-by-one,
double. Falls back to inserting a random TLV-like 4-byte structure.

Ported from honggfuzz mangle.c:mangle_TlvMutate (line 1173).
"""

import random as _random

# Boundary values for length field mutation
_TLV_BOUNDARIES = [0x00, 0x01, 0x7F, 0x80, 0xFF]


def tlv_mutate(data: bytes, rng=None) -> bytes:
    """Mutate potential TLV length fields in the input.

    Scans the first 4KB (or 10% for large inputs) for bytes whose value
    looks like a length field pointing within the remaining data. When a
    candidate is found (1/8 probability), replaces it with a boundary value.

    Falls back to inserting a random 4-byte TLV structure if no candidate found.

    Args:
        data: Input bytes to mutate.
        rng: Random instance (default: module-level random).

    Returns:
        Mutated bytes (same length unless fallback inserts TLV).
    """
    r = rng or _random
    if len(data) < 4:
        return _insert_tlv_fallback(data, r)

    buf = bytearray(data)

    # Scan limit: first 4KB or 10% for large inputs
    scan_limit = min(len(buf) - 2, 4096)
    if len(buf) > 40960:
        scan_limit = max(scan_limit, len(buf) // 10)

    for off in range(scan_limit):
        b1 = buf[off]
        b2 = 0
        if off + 1 < len(buf):
            b2 = (buf[off] << 8) | buf[off + 1]

        remaining = len(buf) - off - 1
        found = (0 < b1 <= remaining) or (b2 > 0 and b2 <= remaining and b2 < len(buf))

        if found and r.randint(0, 7) == 0:
            # Mutate the length field with a boundary value
            mutations = [
                0x00,  # Zero length
                0x01,  # Minimal
                0x7F,  # Max signed byte
                0x80,  # Min negative as signed
                0xFF,  # Max byte
                remaining & 0xFF,  # Exact remaining
                (remaining + 1) & 0xFF,  # Off by one
                (remaining * 2) & 0xFF,  # Double
            ]
            buf[off] = r.choice(mutations)
            return bytes(buf)

    return _insert_tlv_fallback(data, r)


def _insert_tlv_fallback(data: bytes, rng) -> bytes:
    """Insert a random 4-byte TLV-like structure at a random position."""
    tlv = bytes(
        [
            rng.randint(0, 255),  # Tag
            rng.randint(1, 16),  # Length
            rng.randint(0, 255),  # Value byte 1
            rng.randint(0, 255),  # Value byte 2
        ]
    )
    if not data:
        return tlv
    pos = rng.randint(0, len(data))
    return data[:pos] + tlv + data[pos:]
