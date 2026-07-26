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

    Supports both 1-byte and 2-byte (big-endian) length field detection.
    For 2-byte matches, both bytes are mutated to the new length value.

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

        # 1-byte length: value fits in remaining bytes after this byte
        remaining1 = len(buf) - off - 1
        found1 = 0 < b1 <= remaining1

        # 2-byte length: value fits in remaining bytes after both bytes
        remaining2 = len(buf) - off - 2
        found2 = b2 > 0 and b2 <= remaining2 and b2 < len(buf)

        if (found1 or found2) and r.randint(0, 7) == 0:
            if found2 and not found1:
                # 2-byte length field — mutate both bytes as big-endian
                new_len = r.choice([
                    0x0000, 0x0001, 0x007F, 0x0080, 0x00FF,
                    remaining2 & 0xFFFF,
                    (remaining2 + 1) & 0xFFFF,
                    (remaining2 * 2) & 0xFFFF,
                ])
                buf[off] = (new_len >> 8) & 0xFF
                buf[off + 1] = new_len & 0xFF
            else:
                # 1-byte length field — mutate single byte
                mutations = [
                    0x00,  # Zero length
                    0x01,  # Minimal
                    0x7F,  # Max signed byte
                    0x80,  # Min negative as signed
                    0xFF,  # Max byte
                    remaining1 & 0xFF,  # Exact remaining
                    (remaining1 + 1) & 0xFF,  # Off by one
                    (remaining1 * 2) & 0xFF,  # Double
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
