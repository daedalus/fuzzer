"""DER canonicity violations: values BER accepts and DER forbids (X.690 §10-11).

    mode           edit                                   DER rule broken
    -------------  -------------------------------------  ------------------
    bool_noncanon  BOOLEAN ff := 01/7f/80/fe              §11.1 TRUE is ff
    int_pad        INTEGER 05 := 00 05, 80 := ff 80       §8.3.2 minimal
    bits_pad       BIT STRING unused bits := 1s           §11.2.1 zero pad
    set_unsort     SET children := reverse DER order      §11.6 sorted SET OF

Strict decoders (OpenSSL, BoringSSL, Go) reject each; lax ones accept and
can disagree on the value -- the signature-malleability bug class.
Returns None to decline: unparseable input, no site, no-op, or over budget.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.der import parse_der, serialize_der, tag_sites

TAG_BOOLEAN = 0x01
TAG_INTEGER = 0x02
TAG_BIT_STRING = 0x03
TAG_SET = 0x31

# BER TRUE values DER rejects.
BER_TRUES = (0x01, 0x7F, 0x80, 0xFE)

# Legal unused-bit counts; the padding bits are set to 1.
UNUSED_BITS = (1, 2, 3, 4, 5, 6, 7)

_SIGN = 0x80


def bool_noncanon(nodes, rng) -> bytes | None:
    sites = [n for n in tag_sites(nodes, (TAG_BOOLEAN,)) if len(n.value) == 1]
    if not sites:
        return None
    target = rng.choice(sites)
    return serialize_der(nodes, target=target, tvalue=bytes((rng.choice(BER_TRUES),)))


def int_pad(nodes, rng) -> bytes | None:
    sites = [n for n in tag_sites(nodes, (TAG_INTEGER,)) if n.value]
    if not sites:
        return None

    # Sign-extend: same value, one redundant leading octet.
    target = rng.choice(sites)
    pad = b"\xff" if target.value[0] & _SIGN else b"\x00"
    return serialize_der(nodes, target=target, tvalue=pad + target.value)


def bits_pad(nodes, rng) -> bytes | None:
    sites = [n for n in tag_sites(nodes, (TAG_BIT_STRING,)) if len(n.value) >= 2]
    if not sites:
        return None

    # 00 aa a0 -> 03 aa a7: declare k unused bits and set them.
    target = rng.choice(sites)
    k = rng.choice(UNUSED_BITS)
    last = target.value[-1] | ((1 << k) - 1)
    value = bytes((k,)) + target.value[1:-1] + bytes((last,))
    return serialize_der(nodes, target=target, tvalue=value)


def set_unsort(nodes, rng) -> bytes | None:
    sites = [n for n in tag_sites(nodes, (TAG_SET,)) if n.parsed_children and len(n.children) >= 2]
    if not sites:
        return None

    # DER sorts SET OF by encoding; descending order is never canonical
    # unless every child is identical (then the caller declines a no-op).
    target = rng.choice(sites)
    encs = sorted((serialize_der([c]) for c in target.children), reverse=True)
    return serialize_der(nodes, target=target, tvalue=b"".join(encs))


MODES = (bool_noncanon, int_pad, bits_pad, set_unsort)


def der_canon_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one DER canonicity violation; None when it cannot apply."""
    nodes = parse_der(data)
    if not nodes:
        return None

    out = rng.choice(MODES)(nodes, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
