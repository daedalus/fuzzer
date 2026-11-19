"""TLV reshape: regroup sibling frames into a uniform random nesting tree.

``tlv_mutate`` and ``tlv_nest_mutate`` edit bytes inside a fixed nesting.
This changes the nesting itself: a window of k sibling frames becomes one
container tree whose shape is a uniform random labeled tree (Prüfer code,
``core/prufer.py``), so every one of the k^(k-2) shapes is equally likely —
not just the shallow ones a recursive generator favours.

    flat   [A][B][C][D]        code -> parent [3, 0, 3, ROOT]
    nested [D [A [B]] [C]]     D(A(B), C)

An internal node keeps its tag and its value becomes its children's frames
(a constructed frame, as in ASN.1); a leaf keeps its value bytes. Lengths
are recomputed bottom-up, and enclosing frames are repaired via
``resize_tlv_value``. The output is never longer than the input.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from fuzzer_tool.core.prufer import ROOT, children, random_tree
from fuzzer_tool.core.rand_pool import get_default_rand_pool
from fuzzer_tool.core.structural_constraints import TlvNode, parse_tlv, resize_tlv_value

MIN_SIBLINGS = 2  # a 1-node tree is the input itself
MAX_TREE_NODES = 32  # frames per reshaped window: bounds work and depth


class Endian(Enum):
    BIG = "big"
    LITTLE = "little"


class TlvLayout(NamedTuple):
    tag_width: int
    length_width: int
    endian: Endian


# Same probe order as ``_op_tlv_nest_mutate``.
LAYOUTS = (
    TlvLayout(1, 2, Endian.BIG),
    TlvLayout(2, 2, Endian.BIG),
    TlvLayout(1, 4, Endian.BIG),
    TlvLayout(1, 2, Endian.LITTLE),
)


def _sibling_lists(roots: list[TlvNode]) -> list[tuple[TlvNode | None, list[TlvNode]]]:
    """``(parent, siblings)`` for every run of >= MIN_SIBLINGS frames.

    Roots first (parent None), then containers in walk order.
    """
    out: list[tuple[TlvNode | None, list[TlvNode]]] = []
    if len(roots) >= MIN_SIBLINGS:
        out.append((None, roots))
    for root in roots:
        for node in root.walk():
            if len(node.children) >= MIN_SIBLINGS:
                out.append((node, node.children))
    return out


def _build(data: bytes, window: list[TlvNode], parent: list[int], lay: TlvLayout) -> bytes | None:
    """Serialize *window* under the *parent* shape; None if a length overflows."""
    kids = children(parent)
    limit = 1 << (8 * lay.length_width)
    order = lay.endian.value

    # BFS from the root, then emit in reverse: children before parents (no recursion).
    top = parent.index(ROOT)
    bfs = [top]
    for v in bfs:
        bfs.extend(kids[v])

    enc: list[bytes] = [b""] * len(window)
    for v in reversed(bfs):
        node = window[v]
        if kids[v]:
            value = b"".join(enc[c] for c in kids[v])
        else:
            value = data[node.value_start : node.value_end]
        if len(value) >= limit:
            return None
        tag = data[node.tag_offset : node.tag_offset + lay.tag_width]
        enc[v] = tag + len(value).to_bytes(lay.length_width, order) + value
    return enc[top]


def _reshape(data: bytes, lists: list, lay: TlvLayout, rng) -> bytes | None:
    holder, sibs = lists[rng.randint(0, len(lists) - 1)]

    # Window of k consecutive siblings.
    k = min(len(sibs), MAX_TREE_NODES)
    start = rng.randint(0, len(sibs) - k)
    window = sibs[start : start + k]

    tree = _build(data, window, random_tree(k, rng), lay)
    if tree is None:
        return None

    lo, hi = window[0].tag_offset, window[-1].value_end
    if holder is None:
        return data[:lo] + tree + data[hi:]

    # Never longer than before, so no enclosing length can overflow.
    value = data[holder.value_start : lo] + tree + data[hi : holder.value_end]
    return resize_tlv_value(
        data,
        holder,
        value,
        tag_width=lay.tag_width,
        length_width=lay.length_width,
        big_endian=lay.endian is Endian.BIG,
    )


def tlv_reshape(data: bytes, rng=None) -> bytes | None:
    """Rebuild a window of sibling TLV frames as a uniform random tree.

    Uses the first layout in :data:`LAYOUTS` that yields a sibling run.

    Args:
        data: Input bytes.
        rng: ``RandPool``-API source (default: shared pool).

    Returns:
        Reshaped bytes, or None when no layout parses a sibling run or a
        container length would overflow its field.
    """
    r = rng or get_default_rand_pool()
    for lay in LAYOUTS:
        roots = parse_tlv(
            data,
            tag_width=lay.tag_width,
            length_width=lay.length_width,
            big_endian=lay.endian is Endian.BIG,
        )
        lists = _sibling_lists(roots)
        if lists:
            return _reshape(data, lists, lay, r)
    return None
