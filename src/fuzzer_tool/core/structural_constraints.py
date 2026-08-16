"""Length/offset arithmetic goals, and TLV nesting invariants.

Two related gaps in the existing mutators, both a consequence of writing
values blindly rather than deriving them.

**Length/offset pairs.** ``tlv_mutate`` writes boundary constants (0, 1,
0x7f, 0x80, 0xff) into candidate length fields, and the ELF mutator's
``_overlap_section`` writes an all-ones size or a past-EOF offset. Those
reach the interesting branch only by luck: the bug class worth targeting is
a parser that computes ``offset + size`` and compares it against the file
length *without* checking for wraparound, and hitting that needs a pair
where the sum wraps while the offset alone still looks valid. That is an
arithmetic condition, so it can be solved for instead of guessed.

**TLV nesting.** Mutating a leaf value invalidates the length prefix of
every enclosing container. Repairing one level leaves the outer ones wrong,
so the parser rejects the input at the outermost frame and never reaches the
mutated leaf. The invariant is exactly the dependency relation
``field_constraints`` already models — a parent's span covers its children's
length fields — so parsing TLV into ``Field`` objects and running the
existing topological repair fixes every level at once, innermost first.
"""

from __future__ import annotations

import logging

from fuzzer_tool.core.field_constraints import LENGTH, Field, repair

log = logging.getLogger(__name__)

# ── Length/offset arithmetic goals ─────────────────────────────────────

WRAP = "wrap"
"""offset + size wraps past the width, while offset alone stays in range.

The classic OOB read: a bounds check written as ``if (off + size <= len)``
passes because the sum wrapped to something small, then the read uses the
unwrapped size."""

EXACT_FIT = "exact_fit"
"""offset + size == filesize. The boundary a correct parser must accept."""

OFF_BY_ONE = "off_by_one"
"""offset + size == filesize + 1. The boundary it must reject."""

ZERO_SIZE = "zero_size"
"""Valid offset, zero size — empty-region handling."""

SIGNED_NEGATIVE = "signed_negative"
"""Size with the high bit set: non-negative unsigned, negative signed.

Parsers that store a length in a signed int and check ``size <= remaining``
pass this trivially, then use it as a count."""

HUGE_SIZE = "huge_size"
"""Small offset, maximal size — unwrapped overflow of the same check."""

GOALS = (WRAP, EXACT_FIT, OFF_BY_ONE, ZERO_SIZE, SIGNED_NEGATIVE, HUGE_SIZE)


def solve_length_offset(goal: str, width: int, filesize: int, rng=None) -> tuple[int, int] | None:
    """Return ``(offset, size)`` satisfying *goal* for a *width*-byte field.

    Closed form throughout — every goal here is a statement about modular
    arithmetic with an obvious witness, and calling a solver for it would
    cost milliseconds to rediscover what one line states. ``solve_coupled_
    sections`` below handles the case that genuinely needs search.

    Returns None when the goal is unreachable at this width (a wrap needs a
    modulus larger than the file, for instance).
    """
    if width <= 0 or filesize < 0:
        return None
    modulus = 1 << (width * 8)
    limit = modulus - 1

    if goal == WRAP:
        # Need (off + size) mod 2^w < off <= filesize, with size > 0.
        # Taking off in range and size = 2^w - off + k gives a sum of k.
        if filesize == 0 or modulus <= filesize:
            return None
        offset = max(1, filesize // 2)
        overshoot = 1 if rng is None else rng.randint(1, max(1, min(16, offset)))
        size = modulus - offset + overshoot
        if size > limit:
            return None
        return offset, size

    if goal == EXACT_FIT:
        if filesize > limit:
            return None
        offset = filesize // 2
        return offset, filesize - offset

    if goal == OFF_BY_ONE:
        if filesize + 1 > limit:
            return None
        offset = filesize // 2
        return offset, filesize - offset + 1

    if goal == ZERO_SIZE:
        return min(filesize, limit), 0

    if goal == SIGNED_NEGATIVE:
        high_bit = 1 << (width * 8 - 1)
        return min(filesize // 2, limit), high_bit | (filesize & (high_bit - 1))

    if goal == HUGE_SIZE:
        return 0, limit

    return None


def verify_goal(goal: str, offset: int, size: int, width: int, filesize: int) -> bool:
    """True if ``(offset, size)`` actually satisfies *goal*.

    Kept separate from the solver so tests check the property rather than
    re-running the construction that produced it.
    """
    modulus = 1 << (width * 8)
    if not (0 <= offset < modulus and 0 <= size < modulus):
        return False
    wrapped = (offset + size) % modulus

    if goal == WRAP:
        return size > 0 and offset <= filesize and wrapped < offset
    if goal == EXACT_FIT:
        return offset + size == filesize
    if goal == OFF_BY_ONE:
        return offset + size == filesize + 1
    if goal == ZERO_SIZE:
        return size == 0 and offset <= filesize
    if goal == SIGNED_NEGATIVE:
        return bool(size & (1 << (width * 8 - 1)))
    if goal == HUGE_SIZE:
        return size == modulus - 1
    return False


def solve_coupled_sections(
    count: int,
    width: int,
    filesize: int,
    *,
    ordered: bool = True,
    non_overlapping: bool = True,
    wrap_index: int | None = None,
    timeout_ms: int = 200,
) -> list[tuple[int, int]] | None:
    """Solve offsets and sizes for *count* sections under joint constraints.

    Unlike the single-pair goals above, this has no closed form: requiring
    the sections to be ordered *and* non-overlapping *and* one of them to
    wrap couples every field to every other, so each one's admissible range
    depends on all the others. This is the case a solver is for.

    Returns a list of ``(offset, size)``, or None if unsatisfiable.

    Cost: measured at ~10.7ms per call for three 4-byte sections — roughly
    ten times the whole per-execution budget at 800 eps. Deliberately not
    wired to a mutation operator; it is for building a section table once,
    offline or per-seed, not per iteration. The single-pair goals above are
    the hot-path path at ~0.9us.
    """
    if count <= 0:
        return None
    try:
        import z3

        from fuzzer_tool.core.z3_lifecycle import guard_z3_shutdown

        guard_z3_shutdown()
    except ImportError:
        return None

    bits = width * 8
    modulus = 1 << bits
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    offsets = [z3.BitVec(f"off{i}", bits) for i in range(count)]
    sizes = [z3.BitVec(f"size{i}", bits) for i in range(count)]

    for i in range(count):
        solver.add(z3.ULE(offsets[i], filesize))
        solver.add(z3.UGT(sizes[i], 0))

    if ordered:
        for i in range(count - 1):
            solver.add(z3.ULE(offsets[i], offsets[i + 1]))

    if non_overlapping:
        for i in range(count - 1):
            # Zero-extend before adding: in modular arithmetic a wrapping sum
            # satisfies ULE(off + size, next_off) vacuously (0x1000 +
            # 0xFFFFF000 wraps to 0, which is <= anything), so the constraint
            # would permit exactly the overlap it exists to forbid. Widening
            # makes the sum a true integer.
            wide_a = z3.ZeroExt(bits, offsets[i]) + z3.ZeroExt(bits, sizes[i])
            solver.add(z3.ULE(wide_a, z3.ZeroExt(bits, offsets[i + 1])))

    if wrap_index is not None and 0 <= wrap_index < count:
        # The sum must wrap: off + size < off in modular arithmetic.
        solver.add(z3.ULT(offsets[wrap_index] + sizes[wrap_index], offsets[wrap_index]))
        if non_overlapping and wrap_index < count - 1:
            # A wrapping section cannot also end before the next one starts.
            return None

    if solver.check() != z3.sat:
        return None

    model = solver.model()
    out = []
    for i in range(count):
        off = model.eval(offsets[i], model_completion=True).as_long() % modulus
        size = model.eval(sizes[i], model_completion=True).as_long() % modulus
        out.append((off, size))
    return out


# ── TLV nesting ────────────────────────────────────────────────────────

MAX_TLV_DEPTH = 8
"""Nesting depth explored. Deep enough for real container formats, bounded
so a crafted input cannot drive unbounded recursion."""

MAX_TLV_NODES = 4096
"""Total nodes parsed, bounding work on adversarial input."""


class TlvNode:
    """One tag-length-value frame.

    Args:
        tag_offset: Where the tag starts.
        tag_width: Tag width in bytes.
        length_offset: Where the length field starts.
        length_width: Length field width.
        value_start: First byte of the value.
        value_end: One past the last byte of the value.
        big_endian: Byte order of the length field.
        children: Nested frames parsed from within the value.
    """

    __slots__ = (
        "tag_offset",
        "tag_width",
        "length_offset",
        "length_width",
        "value_start",
        "value_end",
        "big_endian",
        "children",
    )

    def __init__(
        self,
        tag_offset: int,
        tag_width: int,
        length_offset: int,
        length_width: int,
        value_start: int,
        value_end: int,
        big_endian: bool = True,
        children: list[TlvNode] | None = None,
    ):
        self.tag_offset = tag_offset
        self.tag_width = tag_width
        self.length_offset = length_offset
        self.length_width = length_width
        self.value_start = value_start
        self.value_end = value_end
        self.big_endian = big_endian
        self.children = children if children is not None else []

    @property
    def size(self) -> int:
        return self.value_end - self.value_start

    def walk(self):
        """Yield this node and every descendant, parents before children."""
        yield self
        for child in self.children:
            yield from child.walk()

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"TlvNode(tag@{self.tag_offset}, len@{self.length_offset}, "
            f"value={self.value_start}:{self.value_end}, kids={len(self.children)})"
        )


def parse_tlv(
    data: bytes,
    *,
    tag_width: int = 1,
    length_width: int = 2,
    big_endian: bool = True,
    start: int = 0,
    end: int | None = None,
    depth: int = 0,
    budget: list[int] | None = None,
) -> list[TlvNode]:
    """Parse a TLV frame sequence, recursing into values that parse cleanly.

    A value is treated as a container only when its bytes parse as a
    complete TLV sequence with nothing left over. Anything else is a leaf,
    which keeps opaque payloads from being misread as structure.
    """
    if end is None:
        end = len(data)
    if budget is None:
        budget = [MAX_TLV_NODES]
    nodes: list[TlvNode] = []
    header = tag_width + length_width
    pos = start

    while pos + header <= end and budget[0] > 0:
        length = int.from_bytes(
            data[pos + tag_width : pos + header], "big" if big_endian else "little"
        )
        value_start = pos + header
        value_end = value_start + length
        if length < 0 or value_end > end:
            break
        budget[0] -= 1
        children: list[TlvNode] = []
        if depth + 1 < MAX_TLV_DEPTH and length >= header:
            candidate = parse_tlv(
                data,
                tag_width=tag_width,
                length_width=length_width,
                big_endian=big_endian,
                start=value_start,
                end=value_end,
                depth=depth + 1,
                budget=budget,
            )
            # Only a *complete* cover counts as nesting.
            if candidate and candidate[-1].value_end == value_end:
                children = candidate
        nodes.append(
            TlvNode(
                tag_offset=pos,
                tag_width=tag_width,
                length_offset=pos + tag_width,
                length_width=length_width,
                value_start=value_start,
                value_end=value_end,
                big_endian=big_endian,
                children=children,
            )
        )
        pos = value_end

    return nodes


def tlv_fields(nodes: list[TlvNode]) -> list[Field]:
    """Length fields for every frame, as ``field_constraints.Field`` objects.

    A parent's span covers its children's length fields, so the existing
    dependency ordering emits children first and ``repair`` fixes the whole
    nest bottom-up in one pass — which is the invariant that was missing.
    """
    fields: list[Field] = []
    for root in nodes:
        for node in root.walk():
            fields.append(
                Field(
                    LENGTH,
                    offset=node.length_offset,
                    width=node.length_width,
                    span=(node.value_start, node.value_end),
                    big_endian=node.big_endian,
                    name=f"tlv_len@{node.length_offset}",
                )
            )
    return fields


def repair_tlv(
    data: bytes, *, tag_width: int = 1, length_width: int = 2, big_endian: bool = True
) -> bytes | None:
    """Restore every length prefix in a TLV nest simultaneously."""
    nodes = parse_tlv(data, tag_width=tag_width, length_width=length_width, big_endian=big_endian)
    if not nodes:
        return None
    fields = tlv_fields(nodes)
    if not fields:
        return None
    return repair(fields, data)


def build_tlv(
    tag: int, value: bytes, *, tag_width: int = 1, length_width: int = 2, big_endian: bool = True
) -> bytes:
    """Serialize one TLV frame. Used by tests and by container synthesis."""
    order = "big" if big_endian else "little"
    return tag.to_bytes(tag_width, order) + len(value).to_bytes(length_width, order) + value


def serialize_tlv(
    node: TlvNode,
    data: bytes,
    *,
    replace: TlvNode | None = None,
    new_value: bytes | None = None,
) -> bytes:
    """Re-emit *node* from *data*, recomputing every length bottom-up.

    Optionally substitutes ``new_value`` for the value of ``replace``.
    Lengths are derived from the serialized children rather than read from
    the buffer, so an edit at any depth propagates outward correctly.

    ``replace`` is matched by ``tag_offset`` — its position in *data* — not
    by object identity, since the caller's node may come from a different
    parse of the same buffer.
    """
    order = "big" if node.big_endian else "little"
    tag = data[node.tag_offset : node.tag_offset + node.tag_width]

    if replace is not None and node.tag_offset == replace.tag_offset:
        value = new_value if new_value is not None else b""
    elif node.children:
        value = b"".join(
            serialize_tlv(child, data, replace=replace, new_value=new_value)
            for child in node.children
        )
    else:
        value = data[node.value_start : node.value_end]

    return tag + len(value).to_bytes(node.length_width, order) + value


def resize_tlv_value(
    data: bytes,
    node: TlvNode,
    new_value: bytes,
    *,
    tag_width: int = 1,
    length_width: int = 2,
    big_endian: bool = True,
) -> bytes | None:
    """Replace one frame's value and repair every enclosing length.

    Rebuilt from the parsed tree rather than spliced-and-reparsed. Splicing
    first would leave stale ancestor lengths in the buffer, and reparsing
    *that* yields a different tree — the surplus bytes get read as sibling
    frames, so the lengths that then get repaired belong to a structure
    nobody asked for and the edit silently disappears.
    """
    if node.value_start > len(data) or node.value_end > len(data):
        return None
    roots = parse_tlv(data, tag_width=tag_width, length_width=length_width, big_endian=big_endian)
    if not roots:
        return None
    out = b"".join(serialize_tlv(root, data, replace=node, new_value=new_value) for root in roots)
    # Anything after the last parsed frame was never part of the structure.
    tail_start = roots[-1].value_end
    return out + data[tail_start:] if tail_start < len(data) else out
