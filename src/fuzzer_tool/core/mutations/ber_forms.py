"""BER-only encodings DER never emits (X.690 §8).

    mode          edit                                    BER rule
    ------------  --------------------------------------  -------------------
    indef_eoc     30 n .. := 30 80 .. 00 00               §8.1.3.6 indefinite
    string_chunk  04 04 abcd := 24 08 04 02 ab 04 02 cd   §8.7.3 constructed
    tag_highform  02 := 1f 02 / 1f 80 02                  §8.1.2.4 high tag
    len_overlong  01 := 83 00 00 01                       §8.1.3.5 long form
    eoc_inject    00 00 between definite-length siblings  §8.1.5 EOC misuse

``der_len_mutate`` already flips 0x80 in without a terminator; indef_eoc
emits the well-formed variant so decoders reach their EOC-matching code.
Returns None to decline: unparseable input, no site, no-op, or over budget.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.der import (
    MAX_LENGTH_BYTES,
    encode_tlv,
    parse_der,
    serialize_der,
    tag_sites,
    walk_der,
)

EOC = b"\x00\x00"
INDEFINITE = b"\x80"
CONSTRUCTED = 0x20
HIGH_TAG = 0x1F
TAG_BIT_STRING = 0x03
_LONG_FORM = 0x80
_CONTINUATION = 0x80

# Primitive string types BER may encode constructed: BIT, OCTET, UTF8,
# Numeric, Printable, T61, IA5, Visible, Universal, BMP.
STRING_TAGS = (0x03, 0x04, 0x0C, 0x12, 0x13, 0x14, 0x16, 0x1A, 0x1C, 0x1E)

# Segment counts for string_chunk.
CHUNKS = (2, 3, 4)

# Redundant leading zero length octets (total stays <= MAX_LENGTH_BYTES).
PAD_BYTES = (1, 2, 3)

# High-tag forms: 0 = minimal (1f nn), 1 = leading 0x80 continuation (1f 80 nn).
HIGH_FORMS = (0, 1)


def segment_string(tag: int, value: bytes, size: int) -> bytes:
    """Primitive *tag* segments of at most *size* payload octets each.

    BIT STRING segments carry their own unused-bits octet: 0 on all but the
    last, which keeps the original (X.690 §8.6.4).
    """
    bits = tag == TAG_BIT_STRING
    unused, payload = (value[:1], value[1:]) if bits else (b"", value)
    t = bytes((tag,))
    starts = range(0, len(payload), size)
    last = len(starts) - 1

    out = []
    for i, s in enumerate(starts):
        head = (unused if i == last else b"\x00") if bits else b""
        out.append(encode_tlv(t, head + payload[s : s + size]))
    return b"".join(out)


def indef_eoc(nodes, rng) -> bytes | None:
    sites = [n for n in walk_der(nodes) if n.constructed and not n.indefinite]
    if not sites:
        return None
    target = rng.choice(sites)
    return serialize_der(nodes, target=target, tvalue=target.value + EOC, tlength=INDEFINITE)


def string_chunk(nodes, rng) -> bytes | None:
    sites = [n for n in tag_sites(nodes, STRING_TAGS) if len(n.value) >= 2]
    if not sites:
        return None

    target = rng.choice(sites)
    tag = target.tag[0]
    n = len(target.value) - (tag == TAG_BIT_STRING)
    if n < 2:
        return None

    size = -(-n // rng.choice(CHUNKS))
    value = segment_string(tag, target.value, size)
    return serialize_der(nodes, target=target, tvalue=value, ttag=bytes((tag | CONSTRUCTED,)))


def tag_highform(nodes, rng) -> bytes | None:
    sites = [n for n in walk_der(nodes) if len(n.tag) == 1]
    if not sites:
        return None

    target = rng.choice(sites)
    first = target.tag[0]
    pad = b"\x80" if rng.choice(HIGH_FORMS) else b""
    ttag = bytes(((first & 0xE0) | HIGH_TAG,)) + pad + bytes((first & HIGH_TAG,))
    return serialize_der(nodes, target=target, ttag=ttag)


def len_overlong(nodes, rng) -> bytes | None:
    sites = [n for n in walk_der(nodes) if not n.indefinite]
    if not sites:
        return None

    target = rng.choice(sites)
    n = len(target.value)
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    pad = rng.choice(PAD_BYTES)
    width = len(raw) + pad
    if width > MAX_LENGTH_BYTES:
        return None

    tlength = bytes((_LONG_FORM | width,)) + bytes(pad) + raw
    return serialize_der(nodes, target=target, tlength=tlength)


def eoc_inject(nodes, rng) -> bytes | None:
    sites = [n for n in walk_der(nodes) if n.parsed_children and not n.indefinite]
    if not sites:
        return None

    target = rng.choice(sites)
    idx = rng.choice(range(len(target.children) + 1))
    kids = target.children
    value = serialize_der(kids[:idx]) + EOC + serialize_der(kids[idx:])
    return serialize_der(nodes, target=target, tvalue=value)


MODES = (indef_eoc, string_chunk, tag_highform, len_overlong, eoc_inject)


def ber_form_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one BER-only encoding; None when it cannot apply."""
    nodes = parse_der(data)
    if not nodes:
        return None

    out = rng.choice(MODES)(nodes, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
