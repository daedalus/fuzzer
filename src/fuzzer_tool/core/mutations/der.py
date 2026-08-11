"""BER/DER (ASN.1 TLV) aware mutations for structure-aware fuzzing.

Provides a BER/DER TLV parser and mutation methods targeting ASN.1's
tag-length-value structure:

  - length-field mutations: short<->long form flips, shrink/grow, BER
    indefinite lengths
  - tag mutations: class bits, constructed bit, tag number, 2-byte tags
  - sibling reordering inside constructed values (SEQUENCE/SET children)
  - TLV insertion (fresh or truncated headers)

DER is a subset of BER (definite, minimal lengths); the parser accepts
both and the mutators deliberately emit BER-valid but DER-invalid forms
(non-minimal lengths, indefinite) to exercise parsers' strictness checks.

Usage:
    from fuzzer_tool.core.mutations.der import DerMutator, parse_der

    mutator = DerMutator()
    mutated = mutator.mutate_length(original_der, max_len=4096)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Maximum nesting depth the parser recurses into; deeper nodes are kept as
# opaque leaves so adversarial nesting cannot blow up the recursion.
MAX_DER_DEPTH = 8

# Long-form lengths with more than this many length bytes are rejected.
MAX_LENGTH_BYTES = 4

# Universal-class tags used when swapping tag numbers.
_UNIVERSAL_TAGS = (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x0C, 0x0D, 0x13, 0x16, 0x17)


def _encode_length(n: int, indefinite: bool = False) -> bytes:
    """Encode a DER-minimal definite length, or the BER indefinite marker."""
    if indefinite:
        return b"\x80"
    if n < 0x80:
        return bytes((n,))
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes((0x80 | len(raw),)) + raw


def _long_length(n: int) -> bytes:
    """Encode n in BER long form even when it fits a short-form byte."""
    raw = n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")
    return bytes((0x80 | len(raw),)) + raw


def _parse_length(data: bytes, i: int, end: int) -> tuple[int, int, bool] | None:
    """Decode a BER length at data[i]; (value_len, header_len, indefinite)."""
    b = data[i]
    if b < 0x80:
        return b, 1, False
    if b == 0x80:
        return 0, 1, True
    nbytes = b & 0x7F
    if nbytes == 0 or nbytes > MAX_LENGTH_BYTES or i + 1 + nbytes > end:
        return None
    n = int.from_bytes(data[i + 1 : i + 1 + nbytes], "big")
    return n, 1 + nbytes, False


@dataclass
class DerNode:
    """One ASN.1 TLV: raw tag/length header bytes plus value (children).

    ``parsed_children`` is False when a constructed value could not be
    decoded (unparseable interior) or depth capped the recursion — the
    value is then kept as an opaque byte slice and re-serialized verbatim.
    """

    tag: bytes
    length_bytes: bytes
    constructed: bool
    header_start: int
    value: bytes
    children: list[DerNode] = field(default_factory=list)
    parsed_children: bool = False
    indefinite: bool = False


def _parse_children(data: bytes, start: int, end: int, depth: int) -> list[DerNode] | None:
    """Parse sibling TLVs in [start, end). Returns None when the interior is
    unparseable or the depth cap is hit — the caller keeps the value opaque
    (distinct from an empty value, which yields [])."""
    if depth > MAX_DER_DEPTH:
        return None
    children: list[DerNode] = []
    pos = start
    while pos < end:
        parsed = _parse_node(data, pos, end, depth)
        if parsed is None:
            return None
        node, pos = parsed
        children.append(node)
    return children


def _parse_node(data: bytes, pos: int, end: int, depth: int) -> tuple[DerNode, int] | None:
    """Parse one TLV at data[pos]; returns (node, next_pos) or None."""
    if pos >= end:
        return None
    t0 = data[pos]
    if t0 & 0x1F == 0x1F:  # tag number needs continuation bytes
        i = pos + 1
        while i < end and data[i] & 0x80:
            i += 1
        if i >= end or i - pos + 1 > 6:
            return None
        tag = data[pos : i + 1]
    else:
        tag = data[pos : pos + 1]
    if pos + len(tag) >= end:
        return None
    parsed_len = _parse_length(data, pos + len(tag), end)
    if parsed_len is None:
        return None
    value_len, len_header, indefinite = parsed_len
    header_end = pos + len(tag) + len_header
    if indefinite:
        value_end = end
    else:
        value_end = header_end + value_len
        if value_end > end:
            return None
    value = data[header_end:value_end]
    constructed = bool(t0 & 0x20)
    children: list[DerNode] = []
    parsed_children = False
    if constructed:
        inner = _parse_children(data, header_end, value_end, depth + 1)
        if inner is not None:
            children = inner
            parsed_children = True
    return (
        DerNode(
            tag=tag,
            length_bytes=data[pos + len(tag) : header_end],
            constructed=constructed,
            header_start=pos,
            value=value,
            children=children,
            parsed_children=parsed_children,
            indefinite=indefinite,
        ),
        value_end,
    )


def parse_der(data: bytes) -> list[DerNode] | None:
    """Parse top-level BER/DER TLVs.

    Returns None when the buffer is not structurally valid ASN.1 (empty,
    truncated header, length beyond the buffer, trailing junk) — never
    raises. Nested constructed values that fail to decode are kept as
    opaque leaves rather than failing the whole parse.
    """
    if not data:
        return None
    nodes: list[DerNode] = []
    pos = 0
    while pos < len(data):
        parsed = _parse_node(data, pos, len(data), 0)
        if parsed is None:
            return None
        node, pos = parsed
        nodes.append(node)
    return nodes


def _serialize_node(node, target, tvalue, tlength, ttag) -> bytes:
    """Serialize one node; byte-minimal for untouched subtrees."""
    if node is target:
        if tvalue is not None:
            tag = ttag if ttag is not None else node.tag
            length = tlength if tlength is not None else _encode_length(len(tvalue))
            return tag + length + tvalue
        v = node.value
        if node.parsed_children:
            v = _serialize_nodes(node.children, target, tvalue, tlength, ttag)
        if tlength is None and ttag is None and v == node.value:
            return node.tag + node.length_bytes + node.value
        tag = ttag if ttag is not None else node.tag
        length = tlength if tlength is not None else _encode_length(len(v))
        return tag + length + v
    if node.parsed_children:
        inner = _serialize_nodes(node.children, target, tvalue, tlength, ttag)
        if inner == node.value:
            return node.tag + node.length_bytes + node.value
        return node.tag + _encode_length(len(inner)) + inner
    return node.tag + node.length_bytes + node.value


def _serialize_nodes(nodes, target=None, tvalue=None, tlength=None, ttag=None) -> bytes:
    return b"".join(_serialize_node(n, target, tvalue, tlength, ttag) for n in nodes)


def serialize_der(nodes, target=None, tvalue=None, tlength=None, ttag=None) -> bytes:
    """Re-serialize parsed nodes; byte-minimal for untouched subtrees.

    ``target`` names a node whose tag/length/value is overridden: ``ttag``
    replaces its tag bytes, ``tlength`` its length encoding (e.g. long-form
    or indefinite flips), ``tvalue`` its value bytes. Ancestor lengths
    re-derive from the changed content.
    """
    return _serialize_nodes(nodes, target, tvalue, tlength, ttag)


class DerMutator:
    """BER/DER-aware mutators, one method per operator.

    Each method returns the mutated bytes, or None when the input does not
    parse or no valid mutation site exists (callers fall back).
    """

    _rng = random

    @staticmethod
    def _all_nodes(nodes: list[DerNode]) -> list[DerNode]:
        out: list[DerNode] = []
        stack = list(nodes)
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out

    def mutate_length(self, data: bytes, max_len: int = 4096, rng=None) -> bytes | None:
        """Mutate a TLV length field: form flips, shrink/grow, indefinite."""
        rng = rng or self._rng
        nodes = parse_der(data)
        if not nodes:
            return None
        nodes_all = self._all_nodes(nodes)
        target = nodes_all[rng.randint(0, len(nodes_all) - 1)]
        actions = ["flip_form", "grow", "shrink"]
        if target.constructed and not target.indefinite:
            actions.append("indefinite")
        action = actions[rng.randint(0, len(actions) - 1)]
        value = target.value
        if action == "flip_form":
            short = len(target.length_bytes) == 1 and target.length_bytes[0] < 0x80
            tlength = _long_length(len(value)) if short else _encode_length(len(value))
            if tlength == target.length_bytes:
                return None  # already minimal: nothing to flip
            return serialize_der(nodes, target=target, tlength=tlength)[:max_len]
        if action == "indefinite":
            return serialize_der(nodes, target=target, tlength=b"\x80")[:max_len]
        if action == "grow":
            room = max_len - len(data)
            if room <= 0:
                return None
            delta = rng.randint(1, min(64, room))
            return serialize_der(nodes, target=target, tvalue=value + rng.randbytes(delta))[
                :max_len
            ]
        if not value:
            return None
        k = rng.randint(0, len(value) - 1)
        return serialize_der(nodes, target=target, tvalue=value[:k])[:max_len]

    def mutate_tag(self, data: bytes, max_len: int = 4096, rng=None) -> bytes | None:
        """Mutate a tag byte: class bits, constructed bit, number, 2-byte."""
        rng = rng or self._rng
        nodes = parse_der(data)
        if not nodes:
            return None
        nodes_all = self._all_nodes(nodes)
        target = nodes_all[rng.randint(0, len(nodes_all) - 1)]
        first = target.tag[0]
        action = rng.randint(0, 3)
        if action == 0:  # flip class bits
            ttag = bytes((first ^ 0xC0,)) + target.tag[1:]
        elif action == 1:  # toggle constructed bit
            ttag = bytes((first ^ 0x20,)) + target.tag[1:]
        elif action == 2:  # swap in another universal tag number
            num = _UNIVERSAL_TAGS[rng.randint(0, len(_UNIVERSAL_TAGS) - 1)]
            ttag = bytes(((first & 0xE0) | (num & 0x1F),))
        else:  # extend to a 2-byte tag
            ttag = bytes(((first & 0xE0) | 0x1F, rng.randint(1, 127)))
        if ttag == target.tag:
            return None
        return serialize_der(nodes, target=target, ttag=ttag)[:max_len]

    def reorder_children(self, data: bytes, max_len: int = 4096, rng=None) -> bytes | None:
        """Shuffle, duplicate, or remove a sibling inside a constructed value."""
        rng = rng or self._rng
        nodes = parse_der(data)
        if not nodes:
            return None
        containers = [
            n for n in self._all_nodes(nodes) if n.parsed_children and len(n.children) >= 2
        ]
        if not containers:
            return None
        target = containers[rng.randint(0, len(containers) - 1)]
        children = list(target.children)
        action = rng.randint(0, 2)
        if action == 0:
            rng.shuffle(children)
        elif action == 1:
            idx = rng.randint(0, len(children) - 1)
            children.insert(idx, children[idx])
        else:
            del children[rng.randint(0, len(children) - 1)]
        target.children = children
        return serialize_der(nodes)[:max_len]

    def insert_tlv(self, data: bytes, max_len: int = 4096, rng=None) -> bytes | None:
        """Insert a fresh or truncated TLV at a random sibling position."""
        rng = rng or self._rng
        nodes = parse_der(data)
        if not nodes:
            return None
        containers = [n for n in self._all_nodes(nodes) if n.parsed_children]
        if not containers:
            return None
        target = containers[rng.randint(0, len(containers) - 1)]
        fresh = (
            b"\x05\x00",  # NULL
            bytes((0x02, 0x01, rng.randint(0, 255))),  # INTEGER
            bytes((0x04, 0x02, rng.randint(0, 255), rng.randint(0, 255))),  # OCTET STRING
            b"\x04\x05",  # truncated OCTET STRING: declared 5, no value bytes
            b"\x30\x03\x02\x01\x00",  # nested SEQUENCE
        )
        payload = fresh[rng.randint(0, len(fresh) - 1)]
        if len(data) + len(payload) > max_len:
            return None
        idx = rng.randint(0, len(target.children))
        new_value = (
            serialize_der(target.children[:idx]) + payload + serialize_der(target.children[idx:])
        )
        return serialize_der(nodes, target=target, tvalue=new_value)[:max_len]

    def _generate_random_der(self, max_len: int = 4096, rng=None) -> bytes:
        """Generate a random nested SEQUENCE tree with consistent lengths."""
        rng = rng or self._rng
        primitives = (
            b"\x02\x01\x01",
            b"\x02\x02\x00\xff",
            b"\x06\x03\x2a\x03\x04",
            b"\x05\x00",
            b"\x01\x01\xff",
            b"\x03\x02\x00\xff",
            b"\x04\x02\xab\xcd",
            b"\x0c\x03\x61\x62\x63",
        )

        def gen(depth: int) -> bytes:
            if depth <= 0 or rng.random() < 0.5:
                return primitives[rng.randint(0, len(primitives) - 1)]
            inner = b"".join(gen(depth - 1) for _ in range(rng.randint(1, 3)))
            return b"\x30" + _encode_length(len(inner)) + inner

        inner = gen(3)
        out = b"\x30" + _encode_length(len(inner)) + inner
        return out[:max_len]
