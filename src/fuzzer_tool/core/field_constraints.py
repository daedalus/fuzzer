"""Simultaneous satisfaction of coupled derived fields.

Real formats carry fields whose values are *functions of other bytes*: a
header length that must equal a payload span, a checksum over a region, an
offset that must point at a real structure. Mutating the payload invalidates
all of them at once, and repairing them one at a time does not converge —
fixing a length changes bytes that a checksum covers, so the checksum has to
be recomputed, and if that checksum's own field lies inside another
checksummed span the correction cascades further.

The existing code repairs single fields in isolation
(``serialize_png_chunks`` recomputes one CRC, ``Z3Solver.solve_png_length``
computes one length). Neither knows about the others, so a mutation that
touches two coupled fields leaves at least one wrong and the target rejects
the input before reaching the parser logic being fuzzed.

This module models the fields as a dependency graph and repairs them in
topological order, so every constraint holds *simultaneously* when it
finishes. Two cases need more than ordering:

- **Cycles.** A checksum whose own field lies inside its span depends on
  itself. Recomputation cannot converge. For CRC-32 this is still solvable
  in closed form, because CRC is affine over GF(2): the effect of each
  field bit on the result is independent, so the required field value can be
  derived directly. ``_solve_self_referential_crc`` does that.
- **Coupled algebraic fields.** Two fields constrained against each other
  (a length and an offset that must sum to the file size, say) have no
  ordering that satisfies both by recomputation. These go to z3.

Ordinary acyclic cases never touch z3 — recomputation is exact and orders of
magnitude cheaper.
"""

from __future__ import annotations

import binascii
import logging
import struct

log = logging.getLogger(__name__)

# Field kinds
LENGTH = "length"
CHECKSUM_CRC32 = "crc32"
CHECKSUM_SUM = "sum"
CONSTANT = "constant"
OFFSET = "offset"

MAX_REPAIR_PASSES = 8
"""Ordering makes one pass sufficient for acyclic graphs; extra passes only
matter when a span was resized mid-repair. Bounded so a pathological
self-feeding layout cannot spin."""


class Field:
    """One derived field: a byte range whose value is a function of a span.

    Args:
        kind: One of LENGTH, CHECKSUM_CRC32, CHECKSUM_SUM, CONSTANT, OFFSET.
        offset: Where the field's own bytes live.
        width: Field width in bytes.
        span: ``(start, end)`` the field is computed over. Ignored for
            CONSTANT.
        big_endian: Field byte order.
        value: Required value, for CONSTANT only.
        adjust: Added to the computed value before storing — covers formats
            whose length counts a header the span excludes.
    """

    __slots__ = ("kind", "offset", "width", "span", "big_endian", "value", "adjust", "name")

    def __init__(
        self,
        kind: str,
        offset: int,
        width: int,
        span: tuple[int, int] | None = None,
        big_endian: bool = True,
        value: int | None = None,
        adjust: int = 0,
        name: str = "",
    ):
        self.kind = kind
        self.offset = offset
        self.width = width
        self.span = span
        self.big_endian = big_endian
        self.value = value
        self.adjust = adjust
        self.name = name or f"{kind}@{offset}"

    @property
    def end(self) -> int:
        return self.offset + self.width

    def covers(self, position: int) -> bool:
        """True if *position* lies inside this field's own bytes."""
        return self.offset <= position < self.end

    def span_contains_field(self, other: Field) -> bool:
        """True if this field's span covers any byte of *other*'s field."""
        if self.span is None:
            return False
        start, end = self.span
        return start < other.end and other.offset < end

    def is_self_referential(self) -> bool:
        """True if this field's span covers its own bytes."""
        return self.span_contains_field(self)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Field({self.name}, kind={self.kind}, span={self.span})"


def _pack(value: int, width: int, big_endian: bool) -> bytes:
    return (value & ((1 << (width * 8)) - 1)).to_bytes(width, "big" if big_endian else "little")


def _unpack(data: bytes, offset: int, width: int, big_endian: bool) -> int:
    return int.from_bytes(data[offset : offset + width], "big" if big_endian else "little")


def compute_field(field: Field, data: bytes) -> int | None:
    """Value *field* should hold given the current *data*."""
    if field.kind == CONSTANT:
        return field.value
    if field.span is None:
        return None
    start, end = field.span
    start = max(0, start)
    end = min(len(data), end)
    if start > end:
        return None
    region = data[start:end]

    if field.kind == LENGTH:
        return len(region) + field.adjust
    if field.kind == CHECKSUM_CRC32:
        return (binascii.crc32(region) & 0xFFFFFFFF) + field.adjust
    if field.kind == CHECKSUM_SUM:
        return (sum(region) + field.adjust) & ((1 << (field.width * 8)) - 1)
    if field.kind == OFFSET:
        return (field.value if field.value is not None else start) + field.adjust
    return None


def check(fields: list[Field], data: bytes) -> list[Field]:
    """Fields whose stored value disagrees with their computed value."""
    broken = []
    for field in fields:
        if field.end > len(data):
            broken.append(field)
            continue
        expected = compute_field(field, data)
        if expected is None:
            continue
        actual = _unpack(data, field.offset, field.width, field.big_endian)
        mask = (1 << (field.width * 8)) - 1
        if (expected & mask) != actual:
            broken.append(field)
    return broken


def satisfied(fields: list[Field], data: bytes) -> bool:
    """True when every field holds simultaneously."""
    return not check(fields, data)


# ── Dependency ordering ────────────────────────────────────────────────


def dependency_order(fields: list[Field]) -> tuple[list[Field], list[Field]]:
    """Topologically sort fields; return ``(ordered, cyclic)``.

    Field A depends on field B when A's span covers B's bytes: B must be
    written before A is computed, or A is computed over stale input. Fields
    in a dependency cycle (including self-referential ones) cannot be fixed
    by ordering and are returned separately.
    """
    deps: dict[int, set[int]] = {i: set() for i in range(len(fields))}
    for i, a in enumerate(fields):
        for j, b in enumerate(fields):
            if i != j and a.span_contains_field(b):
                deps[i].add(j)

    self_ref = {i for i, f in enumerate(fields) if f.is_self_referential()}

    ordered: list[Field] = []
    done: set[int] = set()
    remaining = set(range(len(fields))) - self_ref

    # Kahn's algorithm. Whatever cannot be emitted is inside a cycle.
    while remaining:
        ready = [i for i in remaining if deps[i] <= done]
        if not ready:
            break
        for i in sorted(ready):
            ordered.append(fields[i])
            done.add(i)
            remaining.discard(i)

    cyclic = [fields[i] for i in sorted(set(remaining) | self_ref)]
    return ordered, cyclic


# ── Repair ─────────────────────────────────────────────────────────────


def _write(data: bytearray, field: Field, value: int) -> None:
    packed = _pack(value, field.width, field.big_endian)
    data[field.offset : field.end] = packed


def repair(fields: list[Field], data: bytes) -> bytes | None:
    """Make every field hold simultaneously.

    Acyclic fields are recomputed in dependency order; cyclic ones are
    solved. Returns the repaired bytes, or None if some constraint cannot be
    satisfied.
    """
    if not fields:
        return data
    out = bytearray(data)
    ordered, cyclic = dependency_order(fields)

    for _ in range(MAX_REPAIR_PASSES):
        for field in ordered:
            if field.end > len(out):
                return None
            expected = compute_field(field, bytes(out))
            if expected is not None:
                _write(out, field, expected)

        for field in cyclic:
            if not _repair_cyclic(out, field):
                return None

        if satisfied(fields, bytes(out)):
            return bytes(out)

    return bytes(out) if satisfied(fields, bytes(out)) else None


def _repair_cyclic(out: bytearray, field: Field) -> bool:
    """Repair a field whose span covers its own bytes."""
    if field.end > len(out):
        return False
    if field.kind == CHECKSUM_CRC32:
        value = _solve_self_referential_crc(bytes(out), field)
        if value is None:
            return False
        _write(out, field, value)
        return True
    if field.kind == CHECKSUM_SUM:
        value = _solve_self_referential_sum(bytes(out), field)
        if value is None:
            return False
        _write(out, field, value)
        return True
    if field.kind in (LENGTH, CONSTANT, OFFSET):
        expected = compute_field(field, bytes(out))
        if expected is None:
            return False
        _write(out, field, expected)
        return True
    return False


def _solve_self_referential_crc(data: bytes, field: Field) -> int | None:
    """Field value making CRC-32 over a span that includes the field itself.

    CRC-32 is affine over GF(2): ``crc(x ^ d) == crc(x) ^ (crc(d) ^ crc(0))``
    for equal-length inputs. So each field bit flips a fixed, independent
    pattern in the result, and the required value can be built bit by bit
    instead of searched. 32 probes, no solver.
    """
    if field.width != 4:
        return None
    start, end = field.span
    start, end = max(0, start), min(len(data), end)
    if not (start <= field.offset and field.end <= end):
        return None

    def crc_with(value: int) -> int:
        buf = bytearray(data)
        buf[field.offset : field.end] = _pack(value, 4, field.big_endian)
        return (binascii.crc32(bytes(buf[start:end])) & 0xFFFFFFFF) + field.adjust

    base = crc_with(0) & 0xFFFFFFFF
    # Column i is the result delta caused by setting field bit i alone.
    columns = [(crc_with(1 << i) & 0xFFFFFFFF) ^ base for i in range(32)]

    # crc_with(x) = base ^ (M . x), and we need crc_with(x) == x, so
    # (M ^ I) . x = base. Row b holds the coefficients of output bit b.
    rows: list[list[int]] = []
    for bit in range(32):
        mask = 0
        for i in range(32):
            if (columns[i] >> bit) & 1:
                mask |= 1 << i
        mask ^= 1 << bit  # the identity term, moved to the left-hand side
        rows.append([mask, (base >> bit) & 1])

    # Gauss-Jordan over GF(2).
    pivot_row = 0
    pivot_of_col: dict[int, int] = {}
    for col in range(32):
        sel = next(
            (r for r in range(pivot_row, len(rows)) if (rows[r][0] >> col) & 1),
            None,
        )
        if sel is None:
            continue
        rows[pivot_row], rows[sel] = rows[sel], rows[pivot_row]
        for r in range(len(rows)):
            if r != pivot_row and (rows[r][0] >> col) & 1:
                rows[r][0] ^= rows[pivot_row][0]
                rows[r][1] ^= rows[pivot_row][1]
        pivot_of_col[col] = pivot_row
        pivot_row += 1

    # 0 == 1 anywhere means no field value can close the loop.
    for r in range(pivot_row, len(rows)):
        if rows[r][0] == 0 and rows[r][1] == 1:
            return None

    solution = 0
    for col, r in pivot_of_col.items():
        if rows[r][1]:
            solution |= 1 << col

    return solution if crc_with(solution) & 0xFFFFFFFF == solution else None


def _solve_self_referential_sum(data: bytes, field: Field) -> int | None:
    """Field value making a byte-sum over a span that includes the field.

    ``total = rest + sum(field_bytes)``, so the field must satisfy
    ``v == rest + sum(bytes(v))``. Solved by scanning the byte-sum of the
    field (0..255*width), which is a tiny space, rather than the value space.
    """
    start, end = field.span
    start, end = max(0, start), min(len(data), end)
    if not (start <= field.offset and field.end <= end):
        return None
    mask = (1 << (field.width * 8)) - 1
    rest = sum(data[start : field.offset]) + sum(data[field.end : end])

    for digit_sum in range(255 * field.width + 1):
        candidate = (rest + digit_sum + field.adjust) & mask
        if sum(_pack(candidate, field.width, field.big_endian)) == digit_sum:
            return candidate
    return None


# ── Coupled algebraic fields (the z3 case) ─────────────────────────────


def solve_coupled(
    fields: list[Field], data: bytes, relations: list[tuple[str, int, int, int]]
) -> bytes | None:
    """Satisfy fields that are constrained against *each other*.

    *relations* are ``(op, field_index_a, field_index_b, constant)`` with op
    in ``{"sum_eq", "lt", "le", "eq"}`` — e.g. ``("sum_eq", 0, 1, 512)``
    requires ``field0 + field1 == 512``. No recomputation order satisfies
    these, because each field's correct value depends on the other's; this is
    where a solver is actually required rather than convenient.
    """
    try:
        import z3
    except ImportError:
        return None

    solver = z3.Solver()
    solver.set("timeout", 200)
    vars_ = [z3.BitVec(f"f{i}", f.width * 8) for i, f in enumerate(fields)]

    for i, field in enumerate(fields):
        if field.kind == CONSTANT and field.value is not None:
            solver.add(vars_[i] == field.value)
        elif field.kind == LENGTH and field.span is not None:
            expected = compute_field(field, data)
            if expected is not None:
                solver.add(vars_[i] == (expected & ((1 << (field.width * 8)) - 1)))

    for op, a, b, const in relations:
        if not (0 <= a < len(vars_) and 0 <= b < len(vars_)):
            continue
        width = max(fields[a].width, fields[b].width) * 8
        va = z3.ZeroExt(width - fields[a].width * 8, vars_[a])
        vb = z3.ZeroExt(width - fields[b].width * 8, vars_[b])
        if op == "sum_eq":
            solver.add(va + vb == const)
        elif op == "lt":
            solver.add(z3.ULT(va, vb))
        elif op == "le":
            solver.add(z3.ULE(va, vb))
        elif op == "eq":
            solver.add(va == vb)

    if solver.check() != z3.sat:
        return None

    model = solver.model()
    out = bytearray(data)
    for i, field in enumerate(fields):
        if field.end > len(out):
            return None
        value = model.eval(vars_[i], model_completion=True).as_long()
        _write(out, field, value)
    return bytes(out)


# ── Format extractors ──────────────────────────────────────────────────


def png_fields(data: bytes) -> list[Field]:
    """Length and CRC fields for every PNG chunk.

    Each chunk is ``length(4) type(4) data(length) crc(4)``, where the CRC
    covers type+data but *not* the length — so length and CRC are coupled
    through the data span and must be repaired in order.
    """
    fields: list[Field] = []
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return fields
    pos = 8
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        if length > len(data) or pos + 12 + length > len(data):
            break
        data_start = pos + 8
        data_end = data_start + length
        fields.append(
            Field(LENGTH, offset=pos, width=4, span=(data_start, data_end), name=f"len@{pos}")
        )
        fields.append(
            Field(
                CHECKSUM_CRC32,
                offset=data_end,
                width=4,
                span=(pos + 4, data_end),
                name=f"crc@{data_end}",
            )
        )
        pos = data_end + 4
    return fields


def gzip_fields(data: bytes) -> list[Field]:
    """Trailing CRC-32 and ISIZE of a gzip member.

    Both are computed over the *decompressed* payload, which this module
    cannot see, so only their positions are modelled — callers that hold the
    plaintext supply the values. Included so gzip inputs participate in the
    same simultaneity check rather than being silently skipped.
    """
    if len(data) < 18 or data[:3] != b"\x1f\x8b\x08":
        return []
    return [
        Field(CHECKSUM_CRC32, offset=len(data) - 8, width=4, big_endian=False, name="gzip_crc"),
        Field(LENGTH, offset=len(data) - 4, width=4, big_endian=False, name="gzip_isize"),
    ]
