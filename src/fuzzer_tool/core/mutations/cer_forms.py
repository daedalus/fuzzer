"""CER (Canonical Encoding Rules, X.690 §9) boundary encodings.

CER is BER restricted to: indefinite length on every constructed value,
and strings over 1000 octets split into 1000-octet primitive segments.

    mode         edit                                         CER rule
    -----------  -------------------------------------------  -------------
    frag_edge    string := 24 80 {04 999|1000|1001 ..} 00 00  §9.2 segments
    indef_all    every definite constructed := 80 .. 00 00    §9.1 form
    frag_nested  string := 24 80 {24 80 {04 ..} 00 00} 00 00  §9.2 primitive

frag_edge grows a short string past 1000 octets with zero fill so the
segmenting path runs at all. Segment sizes count payload octets (a BIT
STRING's unused-bits octet excluded).
Returns None to decline: unparseable input, no site, no-op, or over budget.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.ber_forms import (
    CONSTRUCTED,
    EOC,
    INDEFINITE,
    STRING_TAGS,
    TAG_BIT_STRING,
    segment_string,
)
from fuzzer_tool.core.mutations.der import DerNode, parse_der, serialize_der, tag_sites

# X.690 §9.2: fragment size for long strings.
CER_SEGMENT = 1000

# Off-by-one around the segment size: short, exact, over.
SEGMENT_SIZES = (CER_SEGMENT - 1, CER_SEGMENT, CER_SEGMENT + 1)


def _constructed(tag: int, inner: bytes) -> bytes:
    """Indefinite-length constructed wrapper: tag|20 80 inner 00 00."""
    return bytes((tag | CONSTRUCTED,)) + INDEFINITE + inner + EOC


def frag_edge(nodes, rng) -> bytes | None:
    sites = tag_sites(nodes, STRING_TAGS)
    if not sites:
        return None

    target = rng.choice(sites)
    tag = target.tag[0]
    size = rng.choice(SEGMENT_SIZES)

    # Grow to CER_SEGMENT + 1 payload octets so at least two segments exist.
    value = target.value
    short = CER_SEGMENT + 1 + (tag == TAG_BIT_STRING) - len(value)
    if short > 0:
        value += bytes(short)

    inner = segment_string(tag, value, size)
    return serialize_der(
        nodes,
        target=target,
        tvalue=inner + EOC,
        tlength=INDEFINITE,
        ttag=bytes((tag | CONSTRUCTED,)),
    )


def _indef(node: DerNode) -> bytes:
    """CER form of *node*: definite constructed values go indefinite."""
    if not node.parsed_children or node.indefinite:
        return node.tag + node.length_bytes + node.value
    return node.tag + INDEFINITE + b"".join(_indef(c) for c in node.children) + EOC


def indef_all(nodes, _rng) -> bytes | None:
    return b"".join(_indef(n) for n in nodes)


def frag_nested(nodes, rng) -> bytes | None:
    sites = tag_sites(nodes, STRING_TAGS)
    if not sites:
        return None

    target = rng.choice(sites)
    tag = target.tag[0]
    nested = _constructed(tag, segment_string(tag, target.value, CER_SEGMENT))
    return serialize_der(
        nodes,
        target=target,
        tvalue=nested + EOC,
        tlength=INDEFINITE,
        ttag=bytes((tag | CONSTRUCTED,)),
    )


MODES = (frag_edge, indef_all, frag_nested)


def cer_form_mutate(data: bytes, rng, max_len: int) -> bytes | None:
    """Apply one CER boundary encoding; None when it cannot apply."""
    nodes = parse_der(data)
    if not nodes:
        return None

    out = rng.choice(MODES)(nodes, rng)
    if out is None or out == data or len(out) > max_len:
        return None
    return out
