"""Variable-length u64 integer encoding/decoding and mutation.

Ported from https://github.com/aki-akaguma/vu64 — a format where the first byte's
leading-ones count signals the length (1-9 bytes), similar to LEB128 but with
little-endian data layout.

Format pattern:
| Prefix     | Precision | Total Bytes |
|------------|-----------|-------------|
| 0xxxxxxx   | 7 bits    | 1 byte      |
| 10xxxxxx   | 14 bits   | 2 bytes     |
| 110xxxxx   | 21 bits   | 3 bytes     |
| 1110xxxx   | 28 bits   | 4 bytes     |
| 11110xxx   | 35 bits   | 5 bytes     |
| 111110xx   | 42 bits   | 6 bytes     |
| 1111110x   | 49 bits   | 7 bytes     |
| 11111110   | 56 bits   | 8 bytes     |
| 11111111   | 64 bits   | 9 bytes     |

Unlike LEB128: zero encodes as single byte 0x00, and data is little-endian.
"""


def vu64_encoded_len(value: int) -> int:
    """Return the byte length of vu64 encoding for a u64 value."""
    if value == 0:
        return 1
    # Count leading zeros - smaller values need fewer bytes
    ldz = value.bit_length()
    # Map bit length to encoded length (1-9 bytes)
    # 1-7 bits -> 1 byte, 8-14 bits -> 2, etc.
    if ldz <= 7:
        return 1
    elif ldz <= 14:
        return 2
    elif ldz <= 21:
        return 3
    elif ldz <= 28:
        return 4
    elif ldz <= 35:
        return 5
    elif ldz <= 42:
        return 6
    elif ldz <= 49:
        return 7
    elif ldz <= 56:
        return 8
    else:
        return 9


def vu64_encode_value(value: int) -> bytes:
    """Encode a u64 integer value as vu64 bytes."""
    if value < 0:
        raise ValueError("vu64 encodes unsigned values only")
    if value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("value exceeds u64 range")

    length = vu64_encoded_len(value)

    if length == 1:
        return bytes([value & 0x7F])

    # Port the exact algorithm from the Rust source: shift the value left by
    # 'length' bits, write little-endian, then construct the length prefix in
    # the most significant bits of the first byte.
    buf = bytearray(9)

    if length <= 8:
        shifted = value << length
        buf[:8] = shifted.to_bytes(8, "little")
        first_byte = buf[0]
        follow_len = length - 1
        # Construct prefix: `!( (!(b1st >> 1)) >> follow_len )`
        # In Python: ~ is bitwise NOT, but we need to mask to 8 bits
        inner = first_byte >> 1  # Clear top bit
        inner = ~inner & 0xFF  # Bitwise NOT, mask to 8 bits
        inner = inner >> follow_len
        buf[0] = (~inner) & 0xFF  # Final NOT, mask to 8 bits
    else:
        # 9-byte case: value fits in last 8 bytes, first byte is 0xFF
        buf[1:] = value.to_bytes(8, "little")
        buf[0] = 0xFF

    return bytes(buf[:length])


def vu64_decode_bytes(data: bytes) -> tuple[int, int]:
    """Decode vu64 bytes, returning (value, bytes_consumed).

    Raises ValueError if data is truncated or invalid.
    """
    if not data:
        raise ValueError("truncated vu64 value")

    first_byte = data[0]
    length = (first_byte.bit_length() % 8) + 1 if first_byte != 0x00 else 1

    # Calculate length from leading ones
    if first_byte == 0x00:
        length = 1
    else:
        # Count leading ones
        leading_ones = 0
        for i in range(7, -1, -1):
            if first_byte & (1 << i):
                leading_ones += 1
            else:
                break
        length = leading_ones + 1

    if len(data) < length:
        raise ValueError("truncated vu64 value")

    if length == 1:
        return data[0], 1

    follow_len = length - 1

    if follow_len < 7:
        # Reconstruct value from shifted portion and first byte data.
        # Match Rust u8 arithmetic: the lsb shift wraps at 8 bits.
        follow_bytes = data[1:length]
        padded = follow_bytes + b"\x00" * (8 - len(follow_bytes))
        val_le = int.from_bytes(padded, "little")

        lsb = (first_byte << length) & 0xFF
        combined = (val_le << 8) | lsb
        result = combined >> length
        if result < (1 << (7 * follow_len)):
            raise ValueError("redundant encoded vu64 value")
        return result, length
    elif follow_len == 7:
        # 8-byte encoding: bytes[1:8] contain value in low 8 bytes
        val_bytes = data[1:9]
        # Pad if needed
        padded = val_bytes + b"\x00" * (9 - len(val_bytes))
        result = int.from_bytes(padded[:8], "little")
        return result, length
    else:  # follow_len == 8, 9-byte encoding
        # 9-byte encoding: bytes[1:9] contain value
        val_bytes = data[1:9]
        result = int.from_bytes(val_bytes, "little")
        return result, length


def vu64_encode(data: bytes, rng, max_len: int = 65536) -> bytes:
    """Mutate input by rewriting or inserting a vu64-encoded integer.

    Scans for a 1-8 byte little-endian unsigned integer and rewrites it as vu64.
    Falls back to inserting a random vu64 value when no candidate is found.
    Returns input unchanged on empty data or when result would exceed max_len.

    Args:
        data: Input bytes.
        rng: Random source (from RandPool).
        max_len: Maximum output length.

    Returns:
        Mutated bytes, or original bytes if no mutation applied.
    """
    if not data:
        return data

    result = bytearray(data)

    # Candidate widths in bytes, smallest first (like leb128_encode)
    widths = [1, 2, 3, 4, 5, 6, 7, 8]

    for width in widths:
        if len(result) < width:
            continue

        idx = rng.randint(0, len(result) - width)
        value = int.from_bytes(result[idx : idx + width], "little")

        try:
            encoded = vu64_encode_value(value)
        except ValueError:
            continue

        if not encoded:
            continue

        # Replacement: old width bytes become encoded vu64 bytes
        new_len = len(result) - width + len(encoded)
        if new_len > max_len:
            continue

        result[idx : idx + width] = encoded
        return bytes(result[:max_len])

    # Fallback: insert a random small vu64 value at a random position
    value = rng.randint(0, 255)
    try:
        encoded = vu64_encode_value(value)
    except ValueError:
        return data

    if len(result) + len(encoded) > max_len:
        return data

    pos = rng.randint(0, len(result))
    return result[:pos] + encoded + result[pos:max_len]


# Test vectors ported from vu64 Rust tests
_TEST_VECTORS = [
    (0x0F0F, b"\x8f\x3c"),
    (0x0F0F_F0F0, b"\xe0\x0f\xff\xf0"),
    (0x0F0F_F0F0_0F0F, b"\xfd\x87\x07\x78\xf8\x87\x07"),
    (0x0F0F_F0F0_0F0F_F0F0, b"\xff\xf0\xf0\x0f\x0f\xf0\xf0\x0f\x0f"),
]


def _test_roundtrip():
    """Verify encoding roundtrips correctly."""
    for value, expected in _TEST_VECTORS:
        encoded = vu64_encode_value(value)
        assert encoded == expected, f"encode({value:#x}) = {encoded!r}, expected {expected!r}"
        decoded, consumed = vu64_decode_bytes(encoded)
        assert decoded == value, f"decode({encoded!r}) = {decoded}, expected {value}"
        assert consumed == len(encoded)
    print("vu64 roundtrip tests passed")
