"""ZigZag varints: protobuf sint32/sint64, Avro, Thrift compact.

ZigZag folds signed integers onto unsigned ones so small magnitudes stay
short, then LEB128 encodes the result:

    n       0   -1   1   -2   ...   INT64_MIN
    zigzag  0    1   2    3   ...   2**64 - 1   (10-byte varint)

``leb128_encode`` / ``sleb128_encode`` never produce this mapping, so a
decoder's sign-unfolding path is only reached by luck without it.

Functions take ``(data, byte_idx, rng, max_len)`` and return the new bytes,
or None to decline (same contract as ``text_codec``).
"""

from fuzzer_tool.core.mutations.generic import encode_uleb128

_BITS = 64
_MASK = (1 << _BITS) - 1
_I32_MIN, _I32_MAX = -(1 << 31), (1 << 31) - 1
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1

# Field widths a signed integer is read from, little-endian.
WIDTHS = (1, 2, 4, 8)

# Values sitting on sign, 7-bit-group and int32/int64 boundaries.
ZIGZAG_EDGES = (0, -1, 1, -64, 64, _I32_MIN, _I32_MAX, _I64_MIN, _I64_MAX)


def zigzag(n: int) -> int:
    """Map signed *n* to its unsigned ZigZag form (64-bit)."""
    return ((n << 1) ^ (n >> (_BITS - 1))) & _MASK


def _splice(data: bytes, start: int, end: int, repl: bytes, max_len: int) -> bytes | None:
    """Replace data[start:end] with *repl*; None if unchanged or too long."""
    if repl == data[start:end]:
        return None
    if len(data) - (end - start) + len(repl) > max_len:
        return None
    return data[:start] + repl + data[end:]


def rewrite(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    """Re-encode the signed LE field at *pos* as a ZigZag varint."""
    width = rng.choice(WIDTHS)
    if pos + width > len(data):
        return None

    value = int.from_bytes(data[pos : pos + width], "little", signed=True)
    return _splice(data, pos, pos + width, encode_uleb128(zigzag(value)), max_len)


def insert_edge(data: bytes, pos: int, rng, max_len: int) -> bytes | None:
    """Insert a boundary value as a ZigZag varint at *pos*."""
    return _splice(data, pos, pos, encode_uleb128(zigzag(rng.choice(ZIGZAG_EDGES))), max_len)


# Append only: tests index this tuple by position.
ZIGZAG_MODES = (rewrite, insert_edge)


def zigzag_encode(data: bytes, byte_idx: int, rng, max_len: int) -> bytes | None:
    """Rewrite or insert a ZigZag varint at *byte_idx*."""
    if not data:
        return None

    mode = rng.choice(ZIGZAG_MODES)
    return mode(data, byte_idx % len(data), rng, max_len)
