"""Weizz-style structure tags (ISSTA 2020) — consolidated P1 port.

Two independent ways to populate the same tag model, both landing here:

1. **Active / differential** (``get_deps`` + ``place_tags``): flips each byte
   (or bit) of an input, re-executes the target, and observes which cmplog
   operand values changed. Establishes real causality but costs O(len) or
   O(8*len) extra executions — gate with ``max_len`` / ``max_execs`` and
   prefer once-per-lineage.
2. **Passive / cmplog-consuming** (``build_tag_map_from_cmplog`` +
   ``collect_structure_map``): reuses operand pairs already captured by
   ``CmplogCollector`` (+ optional colorization taint regions) and locates
   them in the input via substring matching. No extra executions, no second
   comparison tracer, but weaker than (1): presence, not verified causality.

Both paths write into the same ``ByteTag`` / ``StructureMap`` model, so
downstream consumers (operators, corpus metadata) don't care which one
produced a given map. Prefer (2) as the cheap default; fall back to (1) when
an operator needs confirmed dependencies (e.g. before trusting an ``IS_LEN``
flag enough to auto-repair a checksum).

Tag layout (Python-friendly mirror of Weizz ``struct tag``):

    cmp_id      — stable id of the comparison site / operand group that
                  dominated this byte (0 = untagged)
    parent      — cmp_id of the enclosing chunk (0 = top-level)
    counter     — temporal / nesting height order observed at assignment
    depends_on  — cmp_id this field is derived from (length/CRC-like)
    flags       — TagFlags.IS_LEN | IS_IMPL | IS_CHECKSUM | IS_MAGIC | ...

Contiguous runs of the same ``cmp_id`` become fields; parent/counter nesting
approximates chunks. Tags are approximate — operators must tolerate partial
or wrong tags (same contract as AFLSmart under a bad model).

Gate behind ``--weizz-tags`` / ``--structure-tags`` and a size limit analogous
to Weizz ``-L``. See ``docs/handover/handover_done_2026-09-06.md`` §4, which
absorbed both the Weizz port plan and the P1 tag-map writeup.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntFlag

import xxhash

from fuzzer_tool.core.aho_corasick import scanner_for_pairs

log = logging.getLogger(__name__)

# ── Flags (mirror Weizz TAG_* bits, extended for this tree) ─────────────


class TagFlags(IntFlag):
    NONE = 0
    IS_LEN = 1 << 0  # looks like a length / size field
    IS_IMPL = 1 << 1  # implicit / derived (not directly compared)
    IS_CHECKSUM = 1 << 2  # checksum / CRC-like operand
    IS_MAGIC = 1 << 3  # constant / magic-byte comparison
    IS_INPUT_TO_STATE = 1 << 4  # operand also observed feeding program state
    DIRTY = 1 << 5  # invalidated by a length-changing mutation


# Default size gate (Weizz -L 8k class) shared by both collection paths.
DEFAULT_MAX_LEN = 8192


# ── Per-byte tag ──────────────────────────────────────────────────────


@dataclass(slots=True)
class ByteTag:
    """One tag for one input byte."""

    cmp_id: int = 0
    parent: int = 0
    counter: int = 0
    depends_on: int = 0
    flags: TagFlags = TagFlags.NONE

    def is_tagged(self) -> bool:
        return self.cmp_id != 0

    def is_empty(self) -> bool:
        return self.cmp_id == 0 and self.flags == TagFlags.NONE


# ── Dense / RLE views ────────────────────────────────────────────────


@dataclass
class StructureMap:
    """Structure map for one input seed, however it was produced.

    Prefer the dense ``tags`` list for small seeds; use ``to_rle()`` when
    attaching to corpus metadata so resume stays cheap.
    """

    tags: list[ByteTag] = field(default_factory=list)
    ntypes: int = 0  # distinct cmp_id values (excluding 0)
    max_counter: int = 0
    input_len: int = 0
    # Provenance
    from_cmplog: bool = False
    from_differential: bool = False
    # Optional: byte offsets that participated in differential dep recovery
    dep_bytes: set[int] = field(default_factory=set)
    # Optional dense dep bitvectors kept for debugging / repair glue
    # (populated by get_deps). Keyed by (cmp_id, hit_index) → (v0, v1) byte sets.
    deps: dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]] = field(default_factory=dict)
    exec_count: int = 0
    fully_colorized: bool = False

    def __post_init__(self) -> None:
        if not self.input_len:
            self.input_len = len(self.tags)

    @property
    def length(self) -> int:
        return len(self.tags)

    def field_spans(self) -> list[tuple[int, int, int]]:
        """Return contiguous same-cmp_id runs as (start, end_exclusive, cmp_id).

        Untagged (cmp_id == 0) runs are omitted.
        """
        spans: list[tuple[int, int, int]] = []
        if not self.tags:
            return spans
        i = 0
        n = len(self.tags)
        while i < n:
            cid = self.tags[i].cmp_id
            if cid == 0:
                i += 1
                continue
            j = i + 1
            while j < n and self.tags[j].cmp_id == cid:
                j += 1
            spans.append((i, j, cid))
            i = j
        return spans

    def flagged_spans(self, flag: TagFlags) -> list[tuple[int, int, int]]:
        """Field spans whose first byte carries *flag* (e.g. IS_LEN, IS_CHECKSUM)."""
        out: list[tuple[int, int, int]] = []
        for start, end, cid in self.field_spans():
            if start < len(self.tags) and (self.tags[start].flags & flag):
                out.append((start, end, cid))
        return out

    def chunk_spans(self, parent: int | None = None) -> list[tuple[int, int, int]]:
        """Spans that share the same parent cmp_id (chunk approximation).

        If *parent* is None, return top-level chunks (parent == 0) grouped by
        the dominant child cmp_id sequence. Otherwise return children of that
        parent.
        """
        spans: list[tuple[int, int, int]] = []
        if not self.tags:
            return spans
        i = 0
        n = len(self.tags)
        while i < n:
            t = self.tags[i]
            if t.cmp_id == 0:
                i += 1
                continue
            p = t.parent
            if parent is not None and p != parent:
                i += 1
                continue
            if parent is None and p != 0:
                # skip nested; only top-level when parent is None
                i += 1
                continue
            # extend while same parent and continuous tagged region
            j = i + 1
            while j < n and self.tags[j].cmp_id != 0 and self.tags[j].parent == p:
                j += 1
            spans.append((i, j, p if parent is not None else t.cmp_id))
            i = j
        return spans

    def mark_dirty(self, start: int, end: int) -> None:
        """Flag a range as dirty after a length-changing mutation."""
        end = min(end, len(self.tags))
        for i in range(max(0, start), end):
            self.tags[i].flags |= TagFlags.DIRTY

    def to_rle(self) -> list[tuple[int, int, int, int, int, int]]:
        """Run-length encode: (start, length, cmp_id, parent, counter, flags)."""
        out: list[tuple[int, int, int, int, int, int]] = []
        if not self.tags:
            return out
        i = 0
        n = len(self.tags)
        while i < n:
            t = self.tags[i]
            j = i + 1
            while (
                j < n
                and self.tags[j].cmp_id == t.cmp_id
                and self.tags[j].parent == t.parent
                and self.tags[j].counter == t.counter
                and self.tags[j].depends_on == t.depends_on
                and self.tags[j].flags == t.flags
            ):
                j += 1
            out.append((i, j - i, t.cmp_id, t.parent, t.counter, int(t.flags)))
            i = j
        return out

    @classmethod
    def from_rle(
        cls,
        rle: Sequence[tuple[int, int, int, int, int, int]],
        input_len: int,
        **kwargs,
    ) -> StructureMap:
        tags = [ByteTag() for _ in range(input_len)]
        ntypes = 0
        max_counter = 0
        seen: set[int] = set()
        for start, length, cmp_id, parent, counter, flags in rle:
            for k in range(start, min(start + length, input_len)):
                tags[k] = ByteTag(
                    cmp_id=cmp_id,
                    parent=parent,
                    counter=counter,
                    flags=TagFlags(flags),
                )
            if cmp_id and cmp_id not in seen:
                seen.add(cmp_id)
                ntypes += 1
            max_counter = max(max_counter, counter)
        return cls(
            tags=tags,
            ntypes=ntypes,
            max_counter=max_counter,
            input_len=input_len,
            **kwargs,
        )


# ── Active path: differential dependency recovery ───────────────────────

# A comparison site snapshot: cmp_id → ordered list of (v0, v1) operand pairs
# observed during one execution. Values are bytes (raw operand material).
CmpSnapshot = dict[int, list[tuple[bytes, bytes]]]

# Callback: mutate input → (path_checksum, cmp_snapshot). path_checksum is
# whatever the executor uses to decide "same path" (edge hash, etc.).
ExecFn = Callable[[bytes], tuple[int, CmpSnapshot]]


def _flip(buf: bytearray, idx: int, byte_level: bool) -> int:
    """Flip byte *idx* (all 8 bits) or bit *idx* in *buf*; returns the byte index."""
    if byte_level:
        # Flip all 8 bits of the byte (cheap approximation of Weizz
        # byte-level path that only samples first/last bit of each byte).
        buf[idx] ^= 0xFF
        return idx
    byte_index = idx >> 3
    bit = idx & 7
    buf[byte_index] ^= 1 << (7 - bit)
    return byte_index


def _dep_slot(
    deps: dict[tuple[int, int], tuple[set[int], set[int]]], key: tuple[int, int]
) -> tuple[set[int], set[int]]:
    """(v0 deps, v1 deps) for *key*, created empty on first use."""
    if key not in deps:
        deps[key] = (set(), set())
    return deps[key]


def _diff_site(
    deps: dict[tuple[int, int], tuple[set[int], set[int]]],
    cmp_id: int,
    pairs: list[tuple[bytes, bytes]],
    orig_pairs: list[tuple[bytes, bytes]],
    byte_index: int,
) -> None:
    """Mark *byte_index* as a dep of every operand of *cmp_id* that changed."""
    # Align by hit index; Weizz also handles wrap, we take min length.
    limit = min(len(pairs), len(orig_pairs)) if orig_pairs else len(pairs)
    for j in range(limit):
        v0, v1 = pairs[j]
        if j < len(orig_pairs):
            o_v0, o_v1 = orig_pairs[j]
        else:
            o_v0, o_v1 = b"", b""
        key = (cmp_id, j)
        v0_set, v1_set = _dep_slot(deps, key)
        if v0 != o_v0:
            v0_set.add(byte_index)
        if v1 != o_v1:
            v1_set.add(byte_index)
    # New hits that did not exist in the original snapshot.
    if len(pairs) > len(orig_pairs):
        for j in range(len(orig_pairs), len(pairs)):
            v0, v1 = pairs[j]
            key = (cmp_id, j)
            v0_set, v1_set = _dep_slot(deps, key)
            # Both sides "changed" relative to absence.
            v0_set.add(byte_index)
            v1_set.add(byte_index)


def get_deps(
    data: bytes,
    exec_fn: ExecFn,
    *,
    byte_level: bool | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    max_execs: int = 0,
) -> StructureMap | None:
    """Recover per-byte dependency bitvectors via differential flips.

    Algorithm (Weizz ``get_deps`` simplified):

    1. Run the original input; capture comparison operand map + path checksum.
    2. For each byte (or bit), flip it, re-execute, and record which comparison
       operands changed relative to the original snapshot.
    3. Those changed sites mark the flipped byte as a dependency of that
       comparison operand (v0 / v1).

    Returns ``None`` if the input is empty, over ``max_len``, or the first
    execution fails (empty snapshot and zero checksum treated as failure only
    when the callback raises — otherwise we still return an empty StructureMap).

    Cost: ~``len`` (byte-level) or ``8*len`` (bit-level) extra executions.
    Callers must size-gate and prefer once-per-lineage.
    """
    if not data:
        return StructureMap()
    if len(data) > max_len:
        log.debug("get_deps: skip len=%d > max_len=%d", len(data), max_len)
        return None

    # Heuristic: bit-level is expensive; default to bytes when large.
    if byte_level is None:
        byte_level = len(data) > 512

    try:
        orig_cksum, orig_snap = exec_fn(data)
    except Exception as exc:  # noqa: BLE001 — executor failures are expected
        log.debug("get_deps: original exec failed: %s", exc)
        return None

    exec_count = 1
    n = len(data)
    # deps[(cmp_id, hit_idx)] → (set of bytes affecting v0, set affecting v1)
    deps: dict[tuple[int, int], tuple[set[int], set[int]]] = {}

    # Indices to flip: byte starts, or every bit.
    flip_indices = list(range(n)) if byte_level else list(range(n * 8))

    if max_execs <= 0:
        max_execs = len(flip_indices) + 1

    buf = bytearray(data)

    for idx in flip_indices:
        if exec_count >= max_execs:
            break

        byte_index = _flip(buf, idx, byte_level)

        try:
            _cksum, snap = exec_fn(bytes(buf))
        except Exception:  # noqa: BLE001
            # Restore and continue; a single bad flip must not abort recovery.
            buf[byte_index] = data[byte_index]
            exec_count += 1
            continue
        exec_count += 1

        # Restore
        buf[byte_index] = data[byte_index]

        # Diff operand values per cmp_id / hit slot.
        for cmp_id, pairs in snap.items():
            _diff_site(deps, cmp_id, pairs, orig_snap.get(cmp_id, []), byte_index)

        # Also note path change? Not required for dep recovery; path is for
        # colorization. We keep the flip even if path changes — Weizz does too
        # for operand diffs.

    frozen = {k: (frozenset(v0), frozenset(v1)) for k, (v0, v1) in deps.items()}
    smap = place_tags(n, frozen)
    smap.deps = frozen
    smap.exec_count = exec_count
    smap.from_differential = True
    return smap


def _byte_cmps(
    length: int, deps: dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]]
) -> tuple[list[set[int]], dict[int, int]]:
    """Per-byte set of dependent cmp_ids, and per-cmp_id hit counter."""
    byte_to_cmps: list[set[int]] = [set() for _ in range(length)]
    cmp_counters: dict[int, int] = {}

    for (cmp_id, hit_idx), (v0, v1) in deps.items():
        cmp_counters[cmp_id] = max(cmp_counters.get(cmp_id, 0), hit_idx + 1)
        for b in v0 | v1:
            if 0 <= b < length:
                byte_to_cmps[b].add(cmp_id)
    return byte_to_cmps, cmp_counters


def _min_spans(
    deps: dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]],
) -> dict[int, int]:
    """Smallest dependency span (max - min + 1) seen per cmp_id."""
    span_of: dict[int, int] = {}
    for (cmp_id, _), (v0, v1) in deps.items():
        members = v0 | v1
        if not members:
            continue
        span = max(members) - min(members) + 1
        if cmp_id not in span_of or span < span_of[cmp_id]:
            span_of[cmp_id] = span
    return span_of


def _is_len_dep(
    deps: dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]],
    chosen: int,
    byte_to_cmps: list[set[int]],
    length: int,
) -> bool:
    """Length-field heuristic for *chosen*.

    Its deps span >= 8 bytes while <= 4 bytes are tagged with it, e.g. a
    2/4-byte run whose dependents cover a much wider window.
    """
    members: set[int] = set()
    for (cid, _), (v0, v1) in deps.items():
        if cid == chosen:
            members |= v0 | v1
    return bool(
        members
        and (max(members) - min(members) + 1) >= 8
        and len([b for b in range(length) if chosen in byte_to_cmps[b]]) <= 4
    )


def place_tags(
    length: int,
    deps: dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]],
) -> StructureMap:
    """Assign per-byte tags from dependency bitvectors.

    Contiguous runs of bytes that depend on the same ``cmp_id`` become a
    field. Parent is the most recent non-zero cmp_id that already placed a
    tag (simple nesting heuristic, not a full parse).
    """
    tags = [ByteTag() for _ in range(length)]
    if length == 0 or not deps:
        return StructureMap(tags=tags, input_len=length, from_differential=bool(deps))

    # Aggregate: for each byte, the set of cmp_ids that list it as a dep.
    byte_to_cmps, cmp_counters = _byte_cmps(length, deps)

    # Prefer the cmp_id with the smallest dependency span for a byte
    # (tighter fields). Fall back to min id for stability.
    span_of = _min_spans(deps)

    last_parent = 0
    ntypes = 0
    max_counter = 0
    placed_ids: set[int] = set()

    for i in range(length):
        cmps = byte_to_cmps[i]
        if not cmps:
            continue
        # Choose tightest cmp_id.
        chosen = min(cmps, key=lambda c: (span_of.get(c, length), c))
        tags[i].cmp_id = chosen
        tags[i].parent = last_parent if last_parent != chosen else 0
        tags[i].counter = cmp_counters.get(chosen, 1)
        max_counter = max(max_counter, tags[i].counter)

        if chosen not in placed_ids:
            placed_ids.add(chosen)
            ntypes += 1
            last_parent = chosen

        if _is_len_dep(deps, chosen, byte_to_cmps, length):
            tags[i].flags |= TagFlags.IS_LEN

    return StructureMap(
        tags=tags,
        ntypes=ntypes,
        max_counter=max_counter,
        input_len=length,
        deps=deps,
        from_differential=True,
    )


def synthetic_exec_fn(
    comparisons: Sequence[tuple[int, int, int, Callable[[bytes], tuple[bytes, bytes]]]],
) -> ExecFn:
    """Build an ``exec_fn`` from a list of synthetic comparison sites (test helper).

    Each entry is ``(cmp_id, hit_slot, path_bit_index, operand_fn)`` where
    ``operand_fn(data) -> (v0, v1)`` and ``path_bit_index`` is a byte whose
    value contributes to the path checksum (optional; use -1 for none).
    """

    def _exec(data: bytes) -> tuple[int, CmpSnapshot]:
        snap: CmpSnapshot = {}
        cksum = 0
        for cmp_id, _hit, path_bit, op_fn in comparisons:
            v0, v1 = op_fn(data)
            snap.setdefault(cmp_id, []).append((v0, v1))
            if 0 <= path_bit < len(data):
                cksum = (cksum * 131 + data[path_bit]) & 0xFFFFFFFF
        return cksum, snap

    return _exec


# ── Passive path: consume existing cmplog pairs ──────────────────────────


def _looks_like_magic(op_a: bytes, op_b: bytes) -> bool:
    """Constant / magic comparison: one side very short or all-same."""
    for op in (op_a, op_b):
        if 1 <= len(op) <= 8 and (
            len(set(op)) == 1 or op in (b"\x00" * len(op), b"\xff" * len(op))
        ):
            return True
    # typical fourcc / magic length
    return len(op_a) in (2, 4, 8) and len(op_b) in (2, 4, 8)


def _looks_like_checksum(op: bytes) -> bool:
    """Heuristic: 2/4/8-byte operands that look like CRC/checksum digests.

    Purely shape-based — Weizz never claims a sound parse. Prefer 4-byte
    (CRC-32 / Adler) and avoid tagging short lengths as checksums when they
    already look like lengths (caller should check IS_LEN first).
    """
    if len(op) not in (2, 4, 8):
        return False
    # High entropy-ish: not all zeros/ones and not a tiny integer in either endian.
    if op in (b"\x00" * len(op), b"\xff" * len(op)):
        return False
    for endian in ("little", "big"):
        try:
            v = int.from_bytes(op, endian)
        except ValueError:
            continue
        if 0 < v < 0x10000 and len(op) >= 4:
            # Looks more like a length/count than a CRC.
            return False
    return True


_CMP_ID_MASK = 0xFFFF  # ids are non-zero 16-bit


def _stable_cmp_id(op_a: bytes, op_b: bytes, pc: int | None = None) -> int:
    """Stable non-zero id for a comparison site / operand group.

    Prefer PC when available; otherwise hash the operand pair. The hash is
    seed-free xxh3 (builtin ``hash()`` is salted per process), and ``op_a``
    is length-framed so ``(b"ab", b"c")`` and ``(b"a", b"bc")`` differ.
    """
    if pc is not None and pc != 0:
        # keep low 16 bits non-zero; fold high bits
        cid = (pc ^ (pc >> 16)) & _CMP_ID_MASK
        return cid if cid else 1
    h = xxhash.xxh3_64_intdigest(op_a + op_b, seed=len(op_a)) & _CMP_ID_MASK
    return h if h else 1


@dataclass
class TagCollectorConfig:
    """Knobs for the passive P1 collector (map to CLI later)."""

    max_input_len: int = DEFAULT_MAX_LEN  # Weizz -L class gate
    min_operand_len: int = 1
    max_operand_len: int = 64  # ignore huge memcmp blobs for field tags
    prefer_input_operand: bool = True  # when both sides match, prefer the one in input
    enable_differential: bool = True  # use colorization taints when provided
    once_per_lineage: bool = True  # caller should honour this


def _operand_flags(op_a: bytes, op_b: bytes, n: int) -> TagFlags:
    """Shape flags for a pair: length or checksum (length wins), plus magic."""
    return _pick_flags(_pair_shape(op_a, op_b), n)


# (length floor, flags if a length, flags if not): the input length only
# enters through the length test ``0 < v <= 2n``, so a pair's shape is
# classified once and picked per input.
_Shape = tuple[float, TagFlags, TagFlags]


def _pair_shape(op_a: bytes, op_b: bytes) -> _Shape:
    """Input-length-free half of _operand_flags."""
    floor = min(
        (
            v
            for op in (op_a, op_b)
            if op and len(op) <= 8
            for v in (int.from_bytes(op, "little"), int.from_bytes(op, "big"))
            if v > 0
        ),
        default=math.inf,
    )
    magic = TagFlags.IS_MAGIC if _looks_like_magic(op_a, op_b) else TagFlags.NONE
    # IS_LEN takes priority over checksum on the same operand.
    checksum = _looks_like_checksum(op_a) or _looks_like_checksum(op_b)
    not_len = (TagFlags.IS_CHECKSUM if checksum else TagFlags.NONE) | magic
    return floor, TagFlags.IS_LEN | magic, not_len


def _pick_flags(shape: _Shape, n: int) -> TagFlags:
    floor, if_len, if_not = shape
    return if_len if floor <= n * 2 else if_not


def _claim_span(
    tags: list[ByteTag],
    off: int,
    size: int,
    cid: int,
    counter: int,
    flags: TagFlags,
    dep_bytes: set[int],
) -> bool:
    """Tag still-untagged bytes of [off, off+size); False when none were free."""
    end = off + size
    # only claim still-untagged bytes (shorter operands win)
    any_free = any(tags[i].cmp_id == 0 for i in range(off, end))
    if not any_free:
        return False
    for i in range(off, end):
        if tags[i].cmp_id == 0:
            tags[i] = ByteTag(
                cmp_id=cid,
                parent=0,
                counter=counter,
                flags=flags,
            )
            dep_bytes.add(i)
    return True


def _pair_key(p: tuple[bytes, bytes]) -> tuple[int, int]:
    """Sort key: shorter / more specific operand pairs claim bytes first."""
    a, b = p
    return (min(len(a), len(b)) if a and b else max(len(a), len(b)), -max(len(a), len(b)))


def _sized_operands(op_a: bytes, op_b: bytes, cfg: TagCollectorConfig) -> list[tuple[bytes, str]]:
    """Operands within the configured length bounds, tagged with their side."""
    return [
        (op, side)
        for op, side in ((op_a, "a"), (op_b, "b"))
        if cfg.min_operand_len <= len(op) <= cfg.max_operand_len
    ]


def _claim_pair(
    tags: list[ByteTag],
    candidates: list[tuple[bytes, str]],
    offsets: dict,
    tag: tuple[int, int, TagFlags],
    dep_bytes: set[int],
    first_offset: dict[int, int],
    prefer_input: bool,
) -> None:
    """Claim every in-input occurrence of the pair's operands for tag (cid, counter, flags).

    With *prefer_input*, stop after the first operand that claimed bytes.
    """
    cid, counter, flags = tag
    claimed = False
    for op, _side in candidates:
        for off in offsets.get(op, ()):
            if not _claim_span(tags, off, len(op), cid, counter, flags, dep_bytes):
                continue
            if cid not in first_offset:
                first_offset[cid] = off
            claimed = True
        if claimed and prefer_input:
            break


def _tag_stats(tags: list[ByteTag]) -> tuple[int, int]:
    """(distinct non-zero cmp_ids, max counter) over *tags*."""
    ntypes = len({t.cmp_id for t in tags if t.cmp_id})
    max_counter = max((t.counter for t in tags), default=0)
    return ntypes, max_counter


# One plan per pool: (candidates, cmp_id, counter, shape) in claim order,
# cmp_id -> counter, and operand -> plan indices. Keyed like scanner_for_pairs
# (owner identity + length). The index lets a call visit only the entries
# whose operands occur in the input: scanning every entry cost 23 s per 2.5k
# --hail-mary execs at a ~4k-pair pool.
_PlanEntry = tuple[list[tuple[bytes, str]], int, int, _Shape]
_Plan = tuple[list[_PlanEntry], dict[int, int], dict[bytes, list[int]]]
# One slot per caller kind (with / without a pc map): the mutators build
# without pcs and the seed path with the collector's, on the same pool, so a
# single slot would evict on every alternation.
# Slot value: (pairs owner, pcs owner, key, plan).
_plans: dict[bool, tuple[object, object, tuple, _Plan]] = {}

# Pair -> shape, kept across plan rebuilds: the pool mostly persists when it
# grows or evicts. Cleared, not LRU-trimmed, past the cap (bounded memory).
_SHAPE_CACHE_MAX = 1 << 16
_shape_cache: dict[tuple[bytes, bytes], _Shape] = {}


def _cached_shape(op_a: bytes, op_b: bytes) -> _Shape:
    shape = _shape_cache.get((op_a, op_b))
    if shape is None:
        if len(_shape_cache) >= _SHAPE_CACHE_MAX:
            _shape_cache.clear()
        shape = _shape_cache[(op_a, op_b)] = _pair_shape(op_a, op_b)
    return shape


def _tag_plan(
    pairs: Sequence[tuple[bytes, bytes]],
    pair_pcs: dict[tuple[bytes, bytes], int | None] | None,
    cfg: TagCollectorConfig,
) -> _Plan:
    """Pool-only half of the tag map: sort order, cmp ids, counters, shapes.

    Keyed on the identity and length of *pairs* and *pair_pcs* (both objects
    retained, like ``scanner_for_pairs``). Exact for the collector's own maps:
    it records a pc only when it appends that pair and rebinds ``pairs`` to
    evict. A caller passing a fresh pc dict per call misses every time.
    """
    slot = bool(pair_pcs)
    key = (len(pairs), len(pair_pcs or ()), cfg.min_operand_len, cfg.max_operand_len)
    hit = _plans.get(slot)
    if hit is not None and hit[0] is pairs and hit[1] is (pair_pcs or None) and hit[2] == key:
        return hit[3]

    pcs = pair_pcs or {}
    counter_by_id: dict[int, int] = {}
    plan: list[_PlanEntry] = []
    by_op: dict[bytes, list[int]] = {}
    # Shorter / more specific operands claim bytes first.
    for op_a, op_b in sorted(pairs, key=_pair_key):
        if not op_a and not op_b:
            continue
        # skip pathological sizes
        candidates = _sized_operands(op_a, op_b, cfg)
        if not candidates:
            continue

        cid = _stable_cmp_id(op_a, op_b, pcs.get((op_a, op_b)))
        counter = counter_by_id.setdefault(cid, len(counter_by_id) + 1)
        for op, _side in candidates:
            by_op.setdefault(op, []).append(len(plan))
        plan.append((candidates, cid, counter, _cached_shape(op_a, op_b)))

    built = (plan, counter_by_id, by_op)
    _plans[slot] = (pairs, pair_pcs or None, key, built)
    return built


def build_tag_map_from_cmplog(
    data: bytes,
    pairs: Sequence[tuple[bytes, bytes]],
    *,
    pair_pcs: dict[tuple[bytes, bytes], int | None] | None = None,
    taint_regions: Sequence[tuple[int, int]] | None = None,
    config: TagCollectorConfig | None = None,
) -> StructureMap:
    """Build a StructureMap from cmplog operand pairs (+ optional taints).

    Algorithm (approximate Weizz surgical + tag assignment):

    1. For each (op_a, op_b) locate occurrences of either operand inside
       *data*. Prefer the operand that actually appears in the input.
    2. Assign a stable ``cmp_id`` (from PC when known) to those byte spans.
    3. Optional differential: bytes inside colorization taint regions that
       remain untagged get a soft ``IS_IMPL`` tag keyed by the nearest
       comparison.
    4. Infer length-like / magic / checksum flags from operand shape.
    5. Parent/counter: order comparison sites by first occurrence offset;
       nested spans inherit the enclosing cmp_id as parent and bump counter.

    Args:
        data: Original (or colorized) seed bytes.
        pairs: cmplog operand pairs ``(op_a, op_b)``.
        pair_pcs: optional map pair → program counter.
        taint_regions: optional list of (start, end_inclusive) from colorization.
        config: collector knobs.

    Returns:
        StructureMap (may be empty if nothing matched).
    """
    cfg = config or TagCollectorConfig()
    n = len(data)
    if n == 0 or n > cfg.max_input_len:
        return StructureMap(tags=[ByteTag() for _ in range(n)], input_len=n)

    tags = [ByteTag() for _ in range(n)]
    first_offset: dict[int, int] = {}
    dep_bytes: set[int] = set()

    # Locate every operand in one multi-pattern pass, then consume the results
    # in plan order.  The claim loop below is order-dependent -- shorter, more
    # specific operands take bytes first and later ones only fill what is still
    # untagged -- so the automaton is used to *collect* offsets, never to apply
    # them.  The application order is unchanged, which is what makes the tag
    # map identical byte for byte.
    offsets = scanner_for_pairs(pairs).scan(data, min_len=cfg.min_operand_len)

    plan, counter_by_id, by_op = _tag_plan(pairs, pair_pcs, cfg)
    # Only entries with an operand in the input can claim; sorted indices
    # keep plan (claim) order.
    hits = sorted({i for op in offsets for i in by_op.get(op, ())})
    for i in hits:
        candidates, cid, counter, shape = plan[i]
        _claim_pair(
            tags,
            candidates,
            offsets,
            (cid, counter, _pick_flags(shape, n)),
            dep_bytes,
            first_offset,
            cfg.prefer_input_operand,
        )

    # Parent / nesting: a span whose bytes sit strictly inside another
    # already-tagged span inherits that span's cmp_id as parent.
    # Simple left-to-right pass using field spans after first assignment.
    _assign_parents(tags)

    # Differential soft-tags from colorization taints
    from_diff = False
    if cfg.enable_differential and taint_regions:
        from_diff = _apply_taint_soft_tags(tags, taint_regions, first_offset, counter_by_id)
        if from_diff:
            _assign_parents(tags)

    ntypes, max_counter = _tag_stats(tags)

    return StructureMap(
        tags=tags,
        ntypes=ntypes,
        max_counter=max_counter,
        input_len=n,
        from_cmplog=bool(pairs),
        from_differential=from_diff,
        dep_bytes=dep_bytes,
    )


def _assign_parents(tags: list[ByteTag]) -> None:
    """Set parent cmp_id for nested runs (approximate chunk hierarchy)."""
    n = len(tags)
    # stack of (end_exclusive, cmp_id)
    stack: list[tuple[int, int]] = []
    i = 0
    while i < n:
        cid = tags[i].cmp_id
        if cid == 0:
            stack.clear()
            i += 1
            continue
        # find run end
        j = i + 1
        while j < n and tags[j].cmp_id == cid:
            j += 1
        # pop finished parents
        while stack and stack[-1][0] <= i:
            stack.pop()
        parent = stack[-1][1] if stack else 0
        if parent and parent != cid:
            for k in range(i, j):
                if tags[k].parent == 0:
                    tags[k].parent = parent
        stack.append((j, cid))
        i = j


def _apply_taint_soft_tags(
    tags: list[ByteTag],
    taint_regions: Sequence[tuple[int, int]],
    first_offset: dict[int, int],
    counter_by_id: dict[int, int],
) -> bool:
    """Mark untagged bytes inside taint regions as IS_IMPL near a neighbour cmp."""
    n = len(tags)
    applied = False
    # nearest cmp_id to the left of each position
    nearest: list[int] = [0] * n
    last = 0
    for i in range(n):
        if tags[i].cmp_id:
            last = tags[i].cmp_id
        nearest[i] = last

    for start, end in taint_regions:
        end = min(end + 1, n)  # inclusive → exclusive
        start = max(0, start)
        for i in range(start, end):
            if tags[i].cmp_id != 0:
                continue
            cid = nearest[i]
            if not cid:
                continue
            tags[i] = ByteTag(
                cmp_id=cid,
                parent=0,
                counter=counter_by_id.get(cid, 0),
                flags=TagFlags.IS_IMPL,
            )
            applied = True
    return applied


def collect_structure_map(
    data: bytes,
    cmplog,
    *,
    colorization_result=None,
    config: TagCollectorConfig | None = None,
) -> StructureMap:
    """Build StructureMap from a CmplogCollector instance (+ optional colorize result).

    ``cmplog`` is expected to expose ``.pairs`` and optionally ``._pair_pc``.
    ``colorization_result`` may be a ``ColorizationResult`` (``.taints``) or a
    list of ``(start, end)`` tuples.
    """
    # The collector's own objects, not copies: the scanner and plan caches
    # key on their identity (a copy rebuilt both, 54 ms per seed).
    pairs = getattr(cmplog, "pairs", None) or []
    pair_pcs = getattr(cmplog, "_pair_pc", None) or {}

    taints: list[tuple[int, int]] | None = None
    if colorization_result is not None:
        raw = getattr(colorization_result, "taints", colorization_result)
        taints = []
        for t in raw:
            if hasattr(t, "start"):
                taints.append((t.start, t.end))
            else:
                taints.append((int(t[0]), int(t[1])))

    return build_tag_map_from_cmplog(
        data,
        pairs,
        pair_pcs=pair_pcs,
        taint_regions=taints,
        config=config,
    )


# ── Seed-metadata helpers ────────────────────────────────────────────────


def attach_tags_to_meta(meta: dict, smap: StructureMap) -> dict:
    """Store a compact RLE tag map on seed metadata dict (mutates and returns)."""
    meta = dict(meta)
    meta["weizz_tags_rle"] = smap.to_rle()
    meta["weizz_tags_len"] = smap.input_len
    meta["weizz_tags_ntypes"] = smap.ntypes
    meta["weizz_tags_dirty"] = any(t.flags & TagFlags.DIRTY for t in smap.tags)
    return meta


def load_tags_from_meta(meta: dict) -> StructureMap | None:
    """Restore StructureMap from seed metadata, or None if absent."""
    rle = meta.get("weizz_tags_rle")
    length = meta.get("weizz_tags_len")
    if not rle or not length:
        return None
    return StructureMap.from_rle(rle, int(length), from_cmplog=True)


def inherit_tags_from_parent(
    parent_meta: dict | None,
    parent_data: bytes | None,
    child_data: bytes,
) -> dict:
    """P5 — derived-tag inheritance for a child of a tagged parent.

    Weizz ``use_derived_tags``: children reuse the parent map when the
    mutation was length-preserving; otherwise the map is marked dirty so
    operators skip it until the next collector pass.

    Returns a dict of ``weizz_tags_*`` keys to merge into the child's meta
    (empty when the parent had nothing useful).
    """
    if not parent_meta or parent_data is None:
        return {}
    smap = load_tags_from_meta(parent_meta)
    if smap is None:
        return {}
    out: dict = {}
    if len(child_data) == len(parent_data) and not parent_meta.get("weizz_tags_dirty"):
        # Length-preserving: reuse RLE as-is.
        out["weizz_tags_rle"] = parent_meta.get("weizz_tags_rle")
        out["weizz_tags_len"] = parent_meta.get("weizz_tags_len")
        out["weizz_tags_ntypes"] = parent_meta.get("weizz_tags_ntypes", smap.ntypes)
        out["weizz_tags_dirty"] = False
        out["weizz_tags_inherited"] = True
    else:
        # Length changed or parent already dirty: mark dirty on child so
        # P2/P3 do not consume a misaligned map.
        out["weizz_tags_rle"] = parent_meta.get("weizz_tags_rle")
        out["weizz_tags_len"] = len(child_data)
        out["weizz_tags_ntypes"] = parent_meta.get("weizz_tags_ntypes", smap.ntypes)
        out["weizz_tags_dirty"] = True
        out["weizz_tags_inherited"] = True
    return out


__all__ = [
    "TagFlags",
    "ByteTag",
    "StructureMap",
    "DEFAULT_MAX_LEN",
    "CmpSnapshot",
    "ExecFn",
    "get_deps",
    "place_tags",
    "synthetic_exec_fn",
    "TagCollectorConfig",
    "build_tag_map_from_cmplog",
    "collect_structure_map",
    "attach_tags_to_meta",
    "load_tags_from_meta",
    "inherit_tags_from_parent",
]
