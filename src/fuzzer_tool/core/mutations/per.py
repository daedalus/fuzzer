"""PER (Packed Encoding Rules, X.691) length-determinant mutations.

PER has no tags: fields are back-to-back, sized by length determinants
(§11.9, octet-aligned, unconstrained):

    0nnnnnnn             n < 128
    10nnnnnn nnnnnnnn    n < 16384
    11mmmmmm             fragment: m * 16K items follow, m in 1..4

The parser reads the input as a chain of length-prefixed open-type fields
(``02 ab 03 xyz``); that chain is the sniffer and the edit-site list.

    mode         edit                                   decoder branch
    -----------  -------------------------------------  ------------------
    form_flip    02 := 80 02 (and back)                 non-minimal length
    len_skew     n := n-1 / n+1 / 0 / 127 / 128 / 16383  field desync
    frag_header  n := c1..c4 / reserved c5, ff          fragment reassembly
    bit_insert   one bit before a field, tail shifted   unaligned preamble

Returns None to decline: not a PER chain, no site, no-op, or over budget.
"""

from __future__ import annotations

SHORT_MAX = 0x7F
LONG_MAX = 0x3FFF
FRAG_UNIT = 16384
MAX_FRAG = 4
_LONG = 0x80
_FRAG = 0xC0
_LOW6 = 0x3F

# Smallest input the chain parser accepts as PER.
PER_MIN = 4

# Longest chain accepted. Bounds the sniffer, which runs every exec until
# PER goes live: uncapped, a 4 KiB zero seed walked 4096 fields (690 us).
MAX_FIELDS = 16

# Absolute length edges for len_skew (all encodable without fragmenting).
LEN_EDGES = (0, 1, SHORT_MAX, SHORT_MAX + 1, LONG_MAX)

# Fragment counts: legal 1..4, reserved 5 and 63.
FRAG_COUNTS = (1, 2, 3, 4, 5, _LOW6)

BITS = (0, 1)


def parse_per(data: bytes) -> list[tuple[int, int, int]] | None:
    """(start, header octets, length) per field when *data* is exactly a
    chain of 2..MAX_FIELDS length-prefixed fields, at least one non-empty."""
    end = len(data)
    if end < PER_MIN:
        return None

    # Determinant decode inlined: ~20% faster than a helper call per field.
    fields = []
    pos = 0
    while pos < end:
        b = data[pos]
        if b < _LONG:
            n, hdr = b, 1
        elif b < _FRAG:
            if pos + 1 >= end:
                return None
            n, hdr = ((b & _LOW6) << 8) | data[pos + 1], 2
        else:
            m = b & _LOW6
            if not 1 <= m <= MAX_FRAG:
                return None
            n, hdr = m * FRAG_UNIT, 1

        if pos + hdr + n > end:
            return None
        fields.append((pos, hdr, n))
        if len(fields) > MAX_FIELDS:
            return None
        pos += hdr + n

    if len(fields) < 2 or not any(f[2] for f in fields):
        return None
    return fields


def _encode(n: int) -> bytes:
    """Minimal determinant for n <= LONG_MAX."""
    if n <= SHORT_MAX:
        return bytes((n,))
    return bytes((_LONG | (n >> 8), n & 0xFF))


def _splice(data: bytes, field, header: bytes) -> bytes:
    start, hdr, _n = field
    return data[:start] + header + data[start + hdr :]


def form_flip(data, fields, rng) -> bytes | None:
    sites = [f for f in fields if f[2] <= SHORT_MAX]
    if not sites:
        return None

    field = rng.choice(sites)
    n = field[2]
    header = bytes((_LONG, n)) if field[1] == 1 else bytes((n,))
    return _splice(data, field, header)


def len_skew(data, fields, rng) -> bytes | None:
    field = rng.choice(fields)
    n = field[2]
    new = rng.choice((n - 1, n + 1, *LEN_EDGES))
    if not 0 <= new <= LONG_MAX:
        return None
    return _splice(data, field, _encode(new))


def frag_header(data, fields, rng) -> bytes | None:
    field = rng.choice(fields)
    return _splice(data, field, bytes((_FRAG | rng.choice(FRAG_COUNTS),)))


def bit_insert(data, fields, rng) -> bytes | None:
    start = rng.choice(fields)[0]
    tail = data[start:]
    nbits = len(tail) * 8

    # bit | tail, then left-align into len(tail) + 1 octets (7 zero pad bits).
    v = ((rng.choice(BITS) << nbits) | int.from_bytes(tail, "big")) << 7
    return data[:start] + v.to_bytes(len(tail) + 1, "big")


MODES = (form_flip, len_skew, frag_header, bit_insert)


def per_len_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one PER length-determinant edit; None when it cannot apply."""
    fields = parse_per(data)
    if fields is None:
        return None

    out = rng.choice(MODES)(data, fields, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
