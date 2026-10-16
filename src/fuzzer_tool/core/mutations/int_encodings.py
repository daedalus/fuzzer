"""Numeric and bit-level field re-encodings.

Parsers that read an integer one way and a writer that wrote it another way
disagree in a handful of well-known places: non-canonical (overlong) varints,
sign versus zero extension of a narrow field, packed BCD, base -2, bit gather
and scatter, a bit matrix read by rows instead of columns, and timestamps at
epoch edges.  Each function here builds exactly one of those from the input.

Every function draws only through ``rng.randrange(n)`` so a scripted source
fixes the result.  All return ``data`` itself when nothing changed.
"""

from __future__ import annotations

import re

import numpy as np

_ORDERS = ("big", "little")
_WIDTHS = (1, 2, 4, 8)


def _pick(seq, rng):
    return seq[rng.randrange(len(seq))]


def _field(data: bytes, widths, rng):
    """``(width, offset, byte order)`` of a field that fits in *data*, else None."""
    fits = [w for w in widths if w <= len(data)]
    if not fits:
        return None
    width = _pick(fits, rng)
    offset = rng.randrange(len(data) - width + 1)
    return width, offset, _pick(_ORDERS, rng)


def _read(data: bytes, offset: int, width: int, order: str) -> int:
    return int.from_bytes(data[offset : offset + width], order)


def _write(data: bytes, offset: int, width: int, order: str, value: int) -> bytes:
    raw = (value & ((1 << (8 * width)) - 1)).to_bytes(width, order)
    return data[:offset] + raw + data[offset + width :]


# --- varints -----------------------------------------------------------------
LEB128, VLQ, QUIC, EBML = range(4)
_VARINT_SCAN = 16  # longest run a base-128 decoder follows


def _groups(value: int) -> int:
    return max(1, (value.bit_length() + 6) // 7)


def _dec_leb(buf: bytes, i: int):
    """Little-endian base-128 (protobuf, DWARF, wasm): high bit continues."""
    value = 0
    for n in range(min(_VARINT_SCAN, len(buf) - i)):
        byte = buf[i + n]
        value |= (byte & 0x7F) << (7 * n)
        if not byte & 0x80:
            return value, n + 1
    return None


def _dec_vlq(buf: bytes, i: int):
    """Big-endian base-128 (MIDI VLQ, ASN.1 OID arcs, SQLite's first 8 bytes)."""
    value = 0
    for n in range(min(_VARINT_SCAN, len(buf) - i)):
        byte = buf[i + n]
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, n + 1
    return None


def _dec_quic(buf: bytes, i: int):
    """QUIC (RFC 9000 section 16): the top two bits give a 1/2/4/8-byte length."""
    if i >= len(buf):
        return None
    n = 1 << (buf[i] >> 6)
    if i + n > len(buf):
        return None
    return int.from_bytes(buf[i : i + n], "big") & ((1 << (8 * n - 2)) - 1), n


def _dec_ebml(buf: bytes, i: int):
    """EBML/Matroska vint: the count of leading zero bits gives the length."""
    if i >= len(buf) or buf[i] == 0:
        return None
    n = 9 - buf[i].bit_length()
    if i + n > len(buf):
        return None
    return int.from_bytes(buf[i : i + n], "big") & ((1 << (7 * n)) - 1), n


def _enc_leb(value: int, extra: int) -> bytes:
    n = _groups(value) + extra
    return bytes(((value >> (7 * k)) & 0x7F) | (0x80 if k < n - 1 else 0) for k in range(n))


def _enc_vlq(value: int, extra: int) -> bytes:
    n = _groups(value) + extra
    return bytes(
        ((value >> (7 * (n - 1 - k))) & 0x7F) | (0x80 if k < n - 1 else 0) for k in range(n)
    )


_QUIC_LADDER = (1, 2, 4, 8)


def _enc_quic(value: int, extra: int) -> bytes:
    minimal = next(i for i, n in enumerate(_QUIC_LADDER) if value < 1 << (8 * n - 2))
    step = min(minimal + extra, len(_QUIC_LADDER) - 1)
    n = _QUIC_LADDER[step]
    return (value | (step << (8 * n - 2))).to_bytes(n, "big")


def _enc_ebml(value: int, extra: int) -> bytes:
    n = min(_groups(value) + extra, 8)
    # An all-ones payload is the reserved "unknown size": widen instead of emitting it.
    while value == (1 << (7 * n)) - 1 and n < 8:
        n += 1
    return ((1 << (7 * n)) | value).to_bytes(n, "big")


VARINT_DECODERS = (_dec_leb, _dec_vlq, _dec_quic, _dec_ebml)
VARINT_ENCODERS = (_enc_leb, _enc_vlq, _enc_quic, _enc_ebml)
# Bits each encoder keeps, the largest value it takes (EBML's all-ones payload is the
# reserved "unknown size", so its top is one below the mask), and how many padding
# steps it can add.  The mask truncates; the maximum clamps.  Masking with the
# maximum clears bit 0 of every EBML value.
VARINT_MASK = ((1 << 63) - 1, (1 << 63) - 1, (1 << 62) - 1, (1 << 56) - 1)
VARINT_MAX = ((1 << 63) - 1, (1 << 63) - 1, (1 << 62) - 1, (1 << 56) - 2)
VARINT_MAX_EXTRA = (6, 6, 3, 4)
_VARINT_VALUES = (0, 1, 0x7F, 0x80, 0xFF, 0x3FFF, 0x4000, 0xFFFF, 0x1FFFFF, 0x7FFFFFFF, 0xFFFFFFFF)


def varint_reencode(data: bytes, rng) -> bytes:
    """Rewrite a varint at a random offset in a non-canonical (overlong) form.

    Draws a format (LEB128, big-endian VLQ, QUIC, EBML), an offset and a padding
    amount.  If a varint decodes at that offset its value is re-encoded with
    padding (``85 80 00`` for 5 in LEB128, ``40 05`` in QUIC): the same number,
    a different length, which is what a parser that trusts the canonical form
    or sizes a buffer from the first byte gets wrong.  If nothing decodes there
    an interesting value is inserted in overlong form instead.

    Length changes.  Returns *data* when it is empty or the rewrite is a no-op.

    Example (LEB128, one padding byte)::

        aa 05 bb  ->  aa 85 00 bb

    Args:
        data: Input bytes.
        rng: Draw source with ``randrange``.

    Returns:
        Mutated bytes.
    """
    if not data:
        return data
    fmt = rng.randrange(len(VARINT_DECODERS))
    offset = rng.randrange(len(data))
    parsed = VARINT_DECODERS[fmt](data, offset)
    if parsed is not None:
        value, span = parsed
        extra = 1 + rng.randrange(VARINT_MAX_EXTRA[fmt])
        raw = VARINT_ENCODERS[fmt](min(value & VARINT_MASK[fmt], VARINT_MAX[fmt]), extra)
        child = data[:offset] + raw + data[offset + span :]
        return data if child == data else child
    value = _pick(_VARINT_VALUES, rng)
    extra = 1 + rng.randrange(VARINT_MAX_EXTRA[fmt])
    return data[:offset] + VARINT_ENCODERS[fmt](value, extra) + data[offset:]


# --- sign extension ------------------------------------------------------------
def sign_extend(data: bytes, rng) -> bytes:
    """Sign-extend a narrow value inside a 2/4/8-byte field.

    Takes the low ``k`` bits (k drawn in 1..8w-1) of a field and widens them as
    a signed k-bit integer: the upper bits become copies of bit k-1.  Half the
    calls force bit k-1 set first, so the extension is not a no-op.  Hits the
    truncate/extend confusions (a 12-bit offset read as unsigned, a 24-bit
    length sign-extended into a 32-bit register) that no whole-field operator
    produces.  Length-preserving.

    Example (16-bit field, k=8, forced)::

        00 80  ->  ff 80

    Args:
        data: Input bytes.
        rng: Draw source with ``randrange``.

    Returns:
        Mutated bytes, same length.
    """
    field = _field(data, _WIDTHS[1:], rng)
    if field is None:
        return data
    width, offset, order = field
    bits = 1 + rng.randrange(8 * width - 1)
    force = rng.randrange(2)
    low = _read(data, offset, width, order) & ((1 << bits) - 1) | (force << (bits - 1))
    if low >> (bits - 1) & 1:
        low |= ((1 << (8 * width)) - 1) ^ ((1 << bits) - 1)
    child = _write(data, offset, width, order, low)
    return data if child == data else child


# --- BCD -------------------------------------------------------------------------
_DIGIT_RUN = re.compile(rb"[0-9]{2,16}")


def _bcd_from_digits(digits: bytes) -> bytes:
    if len(digits) % 2:
        digits = b"0" + digits
    return bytes(((digits[i] - 48) << 4) | (digits[i + 1] - 48) for i in range(0, len(digits), 2))


def _bcd_invalid_nibble(data: bytes, rng) -> bytes:
    i = rng.randrange(len(data))
    high = rng.randrange(2) == 0
    bad = 0xA + rng.randrange(6)
    byte = (bad << 4) | (data[i] & 0x0F) if high else (data[i] & 0xF0) | bad
    return data[:i] + bytes([byte]) + data[i + 1 :]


def _bcd_field(data: bytes, rng) -> bytes:
    field = _field(data, _WIDTHS, rng)
    if field is None:
        return data
    width, offset, order = field
    digits = str(_read(data, offset, width, order) % 10 ** (2 * width))
    return _write(data, offset, width, order, int(digits, 16))


def bcd_pack(data: bytes, rng) -> bytes:
    """Re-encode a number as packed BCD, or plant an invalid BCD nibble.

    Three modes: an ASCII digit run becomes packed BCD (two digits per byte,
    shorter), a 1/2/4/8-byte field has its decimal digits reread as hex (the
    packed-BCD form of the same number), or one nibble of a byte is set to
    0xA-0xF, which no decimal digit can be (the BCD-to-binary conversions that
    skip validation).  Mode 0 falls back to mode 1 when there is no digit run.

    Example (mode 0)::

        id=1234;  ->  id= 12 34 ;

    Args:
        data: Input bytes.
        rng: Draw source with ``randrange``.

    Returns:
        Mutated bytes; shorter in mode 0.
    """
    if not data:
        return data
    mode = rng.randrange(3)
    runs = list(_DIGIT_RUN.finditer(data)) if mode == 0 else []
    if mode == 0 and runs:
        run = runs[rng.randrange(len(runs))]
        return data[: run.start()] + _bcd_from_digits(run.group()) + data[run.end() :]
    child = _bcd_invalid_nibble(data, rng) if mode == 2 else _bcd_field(data, rng)
    return data if child == data else child


# --- negabinary --------------------------------------------------------------------
def negabinary(data: bytes, rng) -> bytes:
    """Convert a field to or from base -2 (negabinary), per Hacker's Delight ch. 12.

    ``(x + M) ^ M`` turns a two's-complement value into its base -2 digits and
    ``(x ^ M) - M`` turns them back, with ``M`` = ``0xAAAA...`` for the field
    width.  The result is a value that is small in base -2 but a wild
    two's-complement integer, and the other way round: -1 becomes 3 and 3 reads
    back as -1.  Width 1/2/4/8, either byte order, length-preserving.

    Example (one byte, to base -2)::

        ff  ->  03

    Args:
        data: Input bytes.
        rng: Draw source with ``randrange``.

    Returns:
        Mutated bytes, same length.
    """
    field = _field(data, _WIDTHS, rng)
    if field is None:
        return data
    width, offset, order = field
    mask = int.from_bytes(b"\xaa" * width, "big")
    x = _read(data, offset, width, order)
    y = ((x + mask) ^ mask) if rng.randrange(2) else ((x ^ mask) - mask)
    child = _write(data, offset, width, order, y)
    return data if child == data else child


# --- pext / pdep ----------------------------------------------------------------------
BIT_MASKS = (0x55, 0xAA, 0x0F, 0xF0, 0x33, 0xCC, 0x81, 0xFE)


def _pext(x: int, mask: int) -> int:
    out = k = 0
    while mask:
        low = mask & -mask
        if x & low:
            out |= 1 << k
        k += 1
        mask ^= low
    return out


def _pdep(x: int, mask: int) -> int:
    out = k = 0
    while mask:
        low = mask & -mask
        if x >> k & 1:
            out |= low
        k += 1
        mask ^= low
    return out


# Per mask byte: (pext table, pdep table, popcount), built on first use from
# the bit loops above. At most 256 entries. A field mask is one byte
# repeated, so pext/pdep of the field is per-byte lookups, concatenated
# popcount bits at a time: 3-17x faster than the bit loop at width 2-8.
# bytes.translate / int.from_bytes variants were measured: faster only at
# width 8, slower at 1-4.
_BIT_TABLES: dict[int, tuple[list[int], list[int], int]] = {}


def _bit_tables(byte: int) -> tuple[list[int], list[int], int]:
    tabs = _BIT_TABLES.get(byte)
    if tabs is None:
        ext = [_pext(v, byte) for v in range(256)]
        dep = [_pdep(v, byte) for v in range(256)]
        tabs = _BIT_TABLES[byte] = (ext, dep, byte.bit_count())
    return tabs


def _pext_bytes(x: int, byte: int, width: int) -> int:
    """pext of a ``width``-byte field through ``byte`` repeated per byte."""
    ext, _dep, c = _bit_tables(byte)
    out = 0
    for b in x.to_bytes(width, "big"):
        out = out << c | ext[b]
    return out


def _pdep_bytes(x: int, byte: int, width: int) -> int:
    """pdep of x into a ``width``-byte field through ``byte`` repeated per byte."""
    _ext, dep, c = _bit_tables(byte)
    lo = (1 << c) - 1
    out = 0
    for i in range(width):
        out |= dep[x >> (i * c) & lo] << (8 * i)
    return out


def bit_compress(data: bytes, rng) -> bytes:
    """Gather (pext) or scatter (pdep) the bits of a field through a mask.

    ``pext`` packs the bits of ``x`` selected by the mask into the low bits;
    ``pdep`` spreads the low bits of ``x`` out to the mask's positions.  The
    mask is a byte repeated across the field (``0x55``, ``0xF0``, ``0x81``...)
    or a random non-zero byte.  The structured moves behind bit-interleaved
    layouts (Morton order, 5-6-5 pixels, packed flag words) that a flip or an
    add cannot make in one step.  Width 1/2/4/8, either order, length-preserving.

    Example (one byte, mask 0x0f, pext)::

        b7  ->  07

    Args:
        data: Input bytes.
        rng: Draw source with ``randrange``.

    Returns:
        Mutated bytes, same length.
    """
    field = _field(data, _WIDTHS, rng)
    if field is None:
        return data
    width, offset, order = field
    pick = rng.randrange(len(BIT_MASKS) + 1)
    byte = BIT_MASKS[pick] if pick < len(BIT_MASKS) else 1 + rng.randrange(255)
    x = _read(data, offset, width, order)
    y = _pext_bytes(x, byte, width) if rng.randrange(2) == 0 else _pdep_bytes(x, byte, width)
    child = _write(data, offset, width, order, y)
    return data if child == data else child


# --- bit-matrix transpose ----------------------------------------------------------------
_TRANSPOSE_DIMS = (8, 16, 32)  # n x n bits: 8, 32 and 128 bytes


def bit_matrix_transpose(data: bytes, rng) -> bytes:
    """Transpose an n x n bit matrix (n = 8, 16, 32) at a random byte offset.

    Row-major bits become column-major: bit (r, c) goes to (c, r).  This is the
    real transpose of Hacker's Delight section 7-3, not a swap of random bit
    pairs, so it is an involution and the way a bit-sliced or column-oriented
    reader sees data written row by row.  Skipped (returned unchanged) when the
    block is symmetric.  Length-preserving.

    Example (8x8, first row all ones)::

        ff 00 00 00 00 00 00 00  ->  80 80 80 80 80 80 80 80

    Args:
        data: Input bytes.
        rng: Draw source with ``randrange``.

    Returns:
        Mutated bytes, same length.
    """
    dims = [n for n in _TRANSPOSE_DIMS if n * n // 8 <= len(data)]
    if not dims:
        return data
    n = _pick(dims, rng)
    size = n * n // 8
    offset = rng.randrange(len(data) - size + 1)
    bits = np.unpackbits(np.frombuffer(data, np.uint8, size, offset)).reshape(n, n)
    block = np.packbits(np.ascontiguousarray(bits.T)).tobytes()
    if block == data[offset : offset + size]:
        return data
    return data[:offset] + block + data[offset + size :]


# --- timestamps ------------------------------------------------------------------------------
# (value, byte width).  Narrowest first so the candidates for a short input are a prefix.
TIMESTAMPS = (
    (0x0000, 2),  # DOS date/time zero
    (0x0021, 2),  # DOS date 1980-01-01
    (0xFFFF, 2),
    (0, 4),
    (1, 4),
    (0x7FFFFFFF, 4),  # 2038-01-19, last signed 32-bit second
    (0x80000000, 4),  # first second that wraps negative
    (0xFFFFFFFF, 4),  # -1, and 2106 unsigned
    (946684800, 4),  # 2000-01-01
    (1000000000, 4),
    (2082844800, 4),  # 1904 (classic Mac) to 1970 offset
    (0, 8),
    (0x7FFFFFFFFFFFFFFF, 8),
    (0x8000000000000000, 8),
    (0xFFFFFFFFFFFFFFFF, 8),
    (253402300799, 8),  # 9999-12-31T23:59:59
    (62135596800, 8),  # year 1 to 1970 offset (.NET)
    (11644473600, 8),  # 1601 to 1970 in seconds (FILETIME)
    (116444736000000000, 8),  # 1970 in FILETIME 100 ns ticks
    (2147483647000, 8),  # 2038 in milliseconds
)


def timestamp_specials(data: bytes, rng) -> bytes:
    """Overwrite a 2/4/8-byte field with a timestamp at an epoch edge.

    The table holds the values date code trips on: 2038 and its wrap, -1, the
    year 2000, 9999-12-31, the 1601/1904/year-1 epoch offsets, DOS dates, and
    the same instants in 100 ns ticks and milliseconds.  Only values that fit
    the input are candidates; byte order is drawn.  Length-preserving.

    Example (4-byte, big endian)::

        00 00 00 00  ->  7f ff ff ff

    Args:
        data: Input bytes.
        rng: Draw source with ``randrange``.

    Returns:
        Mutated bytes, same length.
    """
    choices = [entry for entry in TIMESTAMPS if entry[1] <= len(data)]
    if not choices:
        return data
    value, width = _pick(choices, rng)
    offset = rng.randrange(len(data) - width + 1)
    child = _write(data, offset, width, _pick(_ORDERS, rng), value)
    return data if child == data else child
