"""Weizz-style structure tags from comparison dependency recovery.

Ports the *technique* from Weizz (ISSTA 2020) — not the AFL/QEMU shell:

1. ``get_deps`` — differential execution that builds per-byte dependency
   bitvectors against comparison operands (which input bytes, when flipped,
   change which comparison operands).
2. ``place_tags`` — assign a packed tag per input byte from those deps
   (same-cmp_id runs → fields; parent/counter nesting → approximate chunks).

Existing cmplog + colorization remain the sole comparison tracer. This module
only consumes operand snapshots; it does not intercept comparisons itself.

See ``docs/handover/handover_weizz_structure_aware_port_2026-08-31.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

log = logging.getLogger(__name__)

# ── Tag flags (mirror Weizz TAG_* bits where useful) ─────────────────

TAG_IS_LEN = 0x01
TAG_IS_INPUT_TO_STATE = 0x02
TAG_CMP_IS_CHECKSUM = 0x04
TAG_IS_IMPL = 0x08  # implicit / derived dependency

# Default size gate (Weizz -L 8k class). Differential work is O(len) execs.
DEFAULT_MAX_LEN = 8192


# ── Data model ───────────────────────────────────────────────────────


@dataclass(slots=True)
class Tag:
    """Per-byte structure tag (Weizz-inspired)."""

    cmp_id: int = 0
    parent: int = 0
    counter: int = 0
    depends_on: int = 0
    flags: int = 0

    def is_empty(self) -> bool:
        return self.cmp_id == 0 and self.flags == 0


@dataclass
class TagsInfo:
    """Tag map for one input seed."""

    tags: list[Tag] = field(default_factory=list)
    ntypes: int = 0
    max_counter: int = 0
    # Optional dense dep bitvectors kept for debugging / repair glue.
    # Keyed by (cmp_id, hit_index) → (v0_bytes, v1_bytes) as frozensets.
    deps: dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]] = field(
        default_factory=dict
    )
    exec_count: int = 0
    fully_colorized: bool = False

    @property
    def length(self) -> int:
        return len(self.tags)

    def field_spans(self) -> list[tuple[int, int, int]]:
        """Return contiguous same-cmp_id runs as (start, end_exclusive, cmp_id)."""
        spans: list[tuple[int, int, int]] = []
        if not self.tags:
            return spans
        start = 0
        cur = self.tags[0].cmp_id
        for i in range(1, len(self.tags)):
            if self.tags[i].cmp_id != cur:
                if cur != 0:
                    spans.append((start, i, cur))
                start = i
                cur = self.tags[i].cmp_id
        if cur != 0:
            spans.append((start, len(self.tags), cur))
        return spans

    def chunk_spans(self) -> list[tuple[int, int, int]]:
        """Contiguous same-parent runs as (start, end_exclusive, parent_id)."""
        spans: list[tuple[int, int, int]] = []
        if not self.tags:
            return spans
        start = 0
        cur = self.tags[0].parent
        for i in range(1, len(self.tags)):
            p = self.tags[i].parent
            if p != cur:
                if cur != 0:
                    spans.append((start, i, cur))
                start = i
                cur = p
        if cur != 0:
            spans.append((start, len(self.tags), cur))
        return spans


# ── Comparison snapshot interface ────────────────────────────────────

# A comparison site snapshot: cmp_id → ordered list of (v0, v1) operand pairs
# observed during one execution. Values are bytes (raw operand material).
CmpSnapshot = dict[int, list[tuple[bytes, bytes]]]

# Callback: mutate input → (path_checksum, cmp_snapshot). path_checksum is
# whatever the executor uses to decide "same path" (edge hash, etc.).
ExecFn = Callable[[bytes], tuple[int, CmpSnapshot]]


# ── get_deps ─────────────────────────────────────────────────────────


def get_deps(
    data: bytes,
    exec_fn: ExecFn,
    *,
    byte_level: bool | None = None,
    max_len: int = DEFAULT_MAX_LEN,
    max_execs: int = 0,
) -> TagsInfo | None:
    """Recover per-byte dependency bitvectors via differential flips.

    Algorithm (Weizz ``get_deps`` simplified):

    1. Run the original input; capture comparison operand map + path checksum.
    2. For each byte (or bit), flip it, re-execute, and record which comparison
       operands changed relative to the original snapshot.
    3. Those changed sites mark the flipped byte as a dependency of that
       comparison operand (v0 / v1).

    Returns ``None`` if the input is empty, over ``max_len``, or the first
    execution fails (empty snapshot and zero checksum treated as failure only
    when the callback raises — otherwise we still return an empty TagsInfo).

    Cost: ~``len`` (byte-level) or ``8*len`` (bit-level) extra executions.
    Callers must size-gate and prefer once-per-lineage.
    """
    if not data:
        return TagsInfo()
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

    def _ensure(key: tuple[int, int]) -> tuple[set[int], set[int]]:
        if key not in deps:
            deps[key] = (set(), set())
        return deps[key]

    # Indices to flip: byte starts, or every bit.
    if byte_level:
        flip_indices = list(range(n))  # treat as "byte i"
    else:
        flip_indices = list(range(n * 8))

    if max_execs <= 0:
        max_execs = len(flip_indices) + 1

    buf = bytearray(data)

    for idx in flip_indices:
        if exec_count >= max_execs:
            break

        if byte_level:
            byte_index = idx
            # Flip all 8 bits of the byte (cheap approximation of Weizz
            # byte-level path that only samples first/last bit of each byte).
            old = buf[byte_index]
            buf[byte_index] = old ^ 0xFF
        else:
            byte_index = idx >> 3
            bit = idx & 7
            buf[byte_index] ^= 1 << (7 - bit)

        try:
            _cksum, snap = exec_fn(bytes(buf))
        except Exception:  # noqa: BLE001
            # Restore and continue; a single bad flip must not abort recovery.
            if byte_level:
                buf[byte_index] = data[byte_index]
            else:
                buf[byte_index] = data[byte_index]
            exec_count += 1
            continue
        exec_count += 1

        # Restore
        if byte_level:
            buf[byte_index] = data[byte_index]
        else:
            buf[byte_index] = data[byte_index]

        # Diff operand values per cmp_id / hit slot.
        for cmp_id, pairs in snap.items():
            orig_pairs = orig_snap.get(cmp_id, [])
            # Align by hit index; Weizz also handles wrap, we take min length.
            limit = min(len(pairs), len(orig_pairs)) if orig_pairs else len(pairs)
            for j in range(limit):
                v0, v1 = pairs[j]
                if j < len(orig_pairs):
                    o_v0, o_v1 = orig_pairs[j]
                else:
                    o_v0, o_v1 = b"", b""
                key = (cmp_id, j)
                v0_set, v1_set = _ensure(key)
                if v0 != o_v0:
                    v0_set.add(byte_index)
                if v1 != o_v1:
                    v1_set.add(byte_index)
            # New hits that did not exist in the original snapshot.
            if len(pairs) > len(orig_pairs):
                for j in range(len(orig_pairs), len(pairs)):
                    v0, v1 = pairs[j]
                    key = (cmp_id, j)
                    v0_set, v1_set = _ensure(key)
                    # Both sides "changed" relative to absence.
                    v0_set.add(byte_index)
                    v1_set.add(byte_index)

        # Also note path change? Not required for dep recovery; path is for
        # colorization. We keep the flip even if path changes — Weizz does too
        # for operand diffs.

    frozen = {
        k: (frozenset(v0), frozenset(v1)) for k, (v0, v1) in deps.items()
    }
    tags = place_tags(n, frozen)

    return TagsInfo(
        tags=tags.tags,
        ntypes=tags.ntypes,
        max_counter=tags.max_counter,
        deps=frozen,
        exec_count=exec_count,
        fully_colorized=False,
    )


def place_tags(
    length: int,
    deps: dict[tuple[int, int], tuple[frozenset[int], frozenset[int]]],
) -> TagsInfo:
    """Assign per-byte tags from dependency bitvectors.

    Contiguous runs of bytes that depend on the same ``cmp_id`` become a
    field. Parent is the most recent non-zero cmp_id that already placed a
    tag (simple nesting heuristic, not a full parse).
    """
    tags = [Tag() for _ in range(length)]
    if length == 0 or not deps:
        return TagsInfo(tags=tags)

    # Aggregate: for each byte, the set of cmp_ids that list it as a dep.
    byte_to_cmps: list[set[int]] = [set() for _ in range(length)]
    cmp_counters: dict[int, int] = {}

    for (cmp_id, hit_idx), (v0, v1) in deps.items():
        cmp_counters[cmp_id] = max(cmp_counters.get(cmp_id, 0), hit_idx + 1)
        for b in v0 | v1:
            if 0 <= b < length:
                byte_to_cmps[b].add(cmp_id)

    # Prefer the cmp_id with the smallest dependency span for a byte
    # (tighter fields). Fall back to min id for stability.
    span_of: dict[int, int] = {}
    for (cmp_id, _), (v0, v1) in deps.items():
        members = v0 | v1
        if not members:
            continue
        span = max(members) - min(members) + 1
        if cmp_id not in span_of or span < span_of[cmp_id]:
            span_of[cmp_id] = span

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

        # Length heuristic: dependency span looks like a length field if the
        # dependent region is large relative to the tag itself (single-byte
        # or 2/4-byte run whose deps cover a much wider window).
        members = set()
        for (cid, _), (v0, v1) in deps.items():
            if cid == chosen:
                members |= v0 | v1
        if members and (max(members) - min(members) + 1) >= 8 and len(
            [b for b in range(length) if chosen in byte_to_cmps[b]]
        ) <= 4:
            tags[i].flags |= TAG_IS_LEN

    return TagsInfo(
        tags=tags,
        ntypes=ntypes,
        max_counter=max_counter,
        deps=deps,
    )


# ── Synthetic helper for unit tests ──────────────────────────────────


def synthetic_exec_fn(
    comparisons: Sequence[tuple[int, int, int, Callable[[bytes], tuple[bytes, bytes]]]],
) -> ExecFn:
    """Build an ``exec_fn`` from a list of synthetic comparison sites.

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
