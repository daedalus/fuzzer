"""ASN.1 universal-type value edges, schema-free (X.680 / X.690 §8).

``x509_field_mutate`` swaps whole known values; this edits the encoded
contents of any universal primitive the way a type decoder reads it.

    mode         edit                                     decoder branch
    -----------  ---------------------------------------  -----------------
    int_edge     INTEGER := 0/-1/2^31/2^63/2^64/129 octs  bignum / int64 fit
    oid_arc      one OID subidentifier := 0..2^64         arc decode overflow
    real_swap    REAL/INTEGER := inf/NaN/-0/odd forms     §8.5 REAL decode
    str_charset  string := octets outside its alphabet    charset validation
    fixed_len    NULL/BOOLEAN := wrong content length     fixed-size checks

Returns None to decline: unparseable input, no site, no-op, or over budget.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.der import parse_der, serialize_der, tag_sites

TAG_BOOLEAN = 0x01
TAG_INTEGER = 0x02
TAG_NULL = 0x05
TAG_OID = 0x06
TAG_REAL = 0x09

# Two's-complement contents: empty, 0, -1, int8/int32/int64 edges, 2^64, 129-octet.
INTEGERS = (
    b"",
    b"\x00",
    b"\xff",
    b"\x7f",
    b"\x00\x80",
    b"\x7f\xff\xff\xff",
    b"\x00\x80\x00\x00\x00",
    b"\x80\x00\x00\x00",
    b"\x7f" + b"\xff" * 7,
    b"\x80" + b"\x00" * 7,
    b"\x01" + b"\x00" * 8,
    b"\x7f" + b"\xff" * 128,
)

# Subidentifier values: 7-bit, 14-bit, 32-bit and 64-bit carries.
OID_ARCS = (0, 39, 40, 127, 128, 16383, 16384, 2**31 - 1, 2**32, 2**64)

# REAL contents (X.690 §8.5): zero, +inf, -inf, NaN, -0, reserved special,
# binary 1*2^0, 1-octet exponent via length octet, truncated exponent,
# reserved base, huge exponent, decimal NR3/NR2/NR1 edges.
REALS = (
    b"",
    b"\x40",
    b"\x41",
    b"\x42",
    b"\x43",
    b"\x44",
    b"\x80\x00\x01",
    b"\x83\x01\x00\x01",
    b"\x83\x00",
    b"\xb0\x00\x01",
    b"\x81\x7f\xff\x01",
    b"\x03" + b"1.E+999",
    b"\x02" + b"-0.",
    b"\x01" + b"1e",
)

# Per string type: contents outside its alphabet or malformed for its width.
CHARSET_BREAKS = {
    0x0C: (b"\xc0\xaf", b"\xed\xa0\x80", b"\xe2\x82", b"\xf4\x90\x80\x80", b"a\x00b"),
    0x12: (b"12a", b"-1", b"1.5"),
    0x13: (b"a@b", b"a*b", b"a&b", b"a\x00b"),
    0x16: (b"a\x80b", b"\xff"),
    0x1A: (b"a\x7fb", b"\x1b"),
    0x1C: (b"\x00\x00\x00", b"\x00\x11\x00\x00", b"\x00\x00\xd8\x00"),
    0x1E: (b"\x00", b"\xd8\x00", b"\x00a\x00"),
}

# NULL must be empty; BOOLEAN must be one octet.
FIXED_LEN_BREAKS = {
    TAG_BOOLEAN: (b"", b"\xff\xff"),
    TAG_NULL: (b"\x00", b"\x00\x00"),
}

_SEVEN = 0x7F
_MORE = 0x80


def _subids(body: bytes) -> list[int]:
    """Base-128 subidentifiers; body must end on a final (high-bit-clear) octet."""
    out, acc = [], 0
    for b in body:
        acc = (acc << 7) | (b & _SEVEN)
        if not b & _MORE:
            out.append(acc)
            acc = 0
    return out


def _subid(v: int) -> bytes:
    """Minimal base-128 encoding: 128 -> 81 00."""
    out = [v & _SEVEN]
    v >>= 7
    while v:
        out.append(_MORE | (v & _SEVEN))
        v >>= 7
    return bytes(reversed(out))


def int_edge(nodes, rng) -> bytes | None:
    sites = tag_sites(nodes, (TAG_INTEGER,))
    if not sites:
        return None
    return serialize_der(nodes, target=rng.choice(sites), tvalue=rng.choice(INTEGERS))


def oid_arc(nodes, rng) -> bytes | None:
    sites = [n for n in tag_sites(nodes, (TAG_OID,)) if n.value and not n.value[-1] & _MORE]
    if not sites:
        return None

    target = rng.choice(sites)
    arcs = _subids(target.value)
    i = rng.choice(range(len(arcs)))
    arcs[i] = rng.choice(OID_ARCS)
    return serialize_der(nodes, target=target, tvalue=b"".join(map(_subid, arcs)))


def real_swap(nodes, rng) -> bytes | None:
    sites = tag_sites(nodes, (TAG_REAL, TAG_INTEGER))
    if not sites:
        return None
    target = rng.choice(sites)
    return serialize_der(nodes, target=target, tvalue=rng.choice(REALS), ttag=bytes((TAG_REAL,)))


def str_charset(nodes, rng) -> bytes | None:
    sites = tag_sites(nodes, CHARSET_BREAKS)
    if not sites:
        return None
    target = rng.choice(sites)
    return serialize_der(nodes, target=target, tvalue=rng.choice(CHARSET_BREAKS[target.tag[0]]))


def fixed_len(nodes, rng) -> bytes | None:
    sites = tag_sites(nodes, FIXED_LEN_BREAKS)
    if not sites:
        return None
    target = rng.choice(sites)
    return serialize_der(nodes, target=target, tvalue=rng.choice(FIXED_LEN_BREAKS[target.tag[0]]))


MODES = (int_edge, oid_arc, real_swap, str_charset, fixed_len)


def asn1_value_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one ASN.1 value edge; None when it cannot apply."""
    nodes = parse_der(data)
    if not nodes:
        return None

    out = rng.choice(MODES)(nodes, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
