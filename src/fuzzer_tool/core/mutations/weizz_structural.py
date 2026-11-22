"""P2 — field- and chunk-scoped operators driven by Weizz structure tags.

Port of the second item in
``docs/handover/handover_done_2026-09-06.md``: mutation
operators that treat a per-seed :class:`~fuzzer_tool.core.weizz_tags.StructureMap`
as a set of fields (same-``cmp_id`` runs) and chunks (top-level same-parent
spans), rather than mutating at arbitrary byte offsets.

Three operators, all class-based (:mod:`fuzzer_tool.core.mutator_interface`)
rather than ``_op_*`` methods on ``OperatorEngine`` — this is the first
implementor of that interface, chosen because these operators need no
``Fuzzer`` state beyond what ``MutationContext`` already carries
(``cmplog_pairs``, ``max_len``) plus the new ``weizz_tags_enabled`` field:

- ``weizz_field_mutate`` — pick a tagged field span and mutate only inside
  it (random overwrite, arithmetic bump, or single-bit flip), preferring
  non-``IS_MAGIC`` fields. Length-preserving.
- ``weizz_chunk_mutate`` — AFLSmart-style whole-chunk edit over top-level
  chunk spans: duplicate, delete, or swap two chunks. Length-changing
  (except swap of equal-size chunks).
- ``weizz_chunk_splice`` — AFLSmart's cross-seed chunk edit: replace a
  top-level chunk of the input with, or insert after it, a chunk of the
  *same type* (same ``cmp_id``) taken from another corpus entry. Both seeds'
  maps are built from the same ``cmplog_pairs``, so one ``cmp_id`` names one
  comparison site in both. Declines when no donor shares a chunk type.

**Tag cost model:** the operators build the :class:`StructureMap` from
``data`` and ``context.cmplog_pairs`` via the passive collector
(:func:`build_tag_map_from_cmplog`), memoised in a small in-module LRU
(``_smap_for``) keyed on the input bytes and on the identity and length of
the pairs list, so repeated picks of one seed (and of one donor) skip the
rebuild. There is no persisted once-per-lineage cache yet (§4 step 3 / §6 "Differential pass cost" in the
handover doc is about the *active* ``get_deps`` path, which these operators
do not use). The passive collector is a single substring-matching pass
gated by ``TagCollectorConfig.max_input_len`` (default
``weizz_tags.DEFAULT_MAX_LEN``, mirrored below), so a miss is bounded;
persisting on seed metadata
(``attach_tags_to_meta`` / ``load_tags_from_meta`` already exist for this)
is the natural follow-up once these operators show signal under
``--elo``/paired bench.

Gate: all three operators decline (``is_available`` returns ``False``) unless
``--weizz-tags`` is set (``context.weizz_tags_enabled``), matching the
handover doc's acceptance checklist ("Flag default off").
"""

from __future__ import annotations

import struct
from collections import OrderedDict

from fuzzer_tool.core.mutations.generic import ARITHMETIC_DELTAS
from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.weizz_tags import (
    DEFAULT_MAX_LEN,
    StructureMap,
    TagFlags,
    build_tag_map_from_cmplog,
)

# Widths eligible for the arithmetic sub-mutation (struct-packable).
_ARITH_WIDTHS = (1, 2, 4, 8)
_ARITH_FMT = {1: None, 2: "H", 4: "I", 8: "Q"}

# Cap on how large a duplicated/swapped chunk block we accept per call --
# a single pathological chunk (most of the input tagged as one span) should
# not let one operator call blow past max_len in one step.
_MAX_CHUNK_BLOCK = 4096


# Memo of built maps. The pairs list is rebound (not mutated in place) when
# cmplog evicts, so identity + length of ``pairs`` is a sound staleness key;
# the entry keeps a reference to the list so ``is`` cannot hit a recycled id.
_SMAP_CACHE: OrderedDict[tuple, tuple[object, StructureMap]] = OrderedDict()
_SMAP_CACHE_MAX = 128


def _smap_for(data: bytes, pairs) -> StructureMap:
    """``build_tag_map_from_cmplog`` behind a bounded LRU (see module docstring)."""
    key = (data, id(pairs), len(pairs))
    hit = _SMAP_CACHE.get(key)
    if hit is not None and hit[0] is pairs:
        _SMAP_CACHE.move_to_end(key)
        return hit[1]
    smap = build_tag_map_from_cmplog(bytes(data), pairs)
    _SMAP_CACHE[key] = (pairs, smap)
    if len(_SMAP_CACHE) > _SMAP_CACHE_MAX:
        _SMAP_CACHE.popitem(last=False)
    return smap


def _preconditions_ok(data: bytes, context: MutationContext | None) -> bool:
    return bool(
        context is not None
        and context.weizz_tags_enabled
        and context.cmplog_pairs
        and data
        and len(data) <= DEFAULT_MAX_LEN
    )


class WeizzFieldMutator(MutatorBase):
    """Mutate within a single tagged field, length-preserving."""

    name = "weizz_field_mutate"
    category = "structural"

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        return _preconditions_ok(data, context)

    def mutate(self, data, rng, max_len=0, *, context=None, **ctx):
        if not _preconditions_ok(data, context):
            return None

        smap = _smap_for(data, context.cmplog_pairs)
        spans = smap.field_spans()
        if not spans:
            return None

        # Prefer non-magic fields most of the time; magic bytes are already
        # a well-covered target of dict/interesting-value operators, and
        # constantly clobbering them wastes the tag signal this operator
        # exists to exploit.
        non_magic = [s for s in spans if not (smap.tags[s[0]].flags & TagFlags.IS_MAGIC)]
        pool = non_magic if non_magic and rng.randint(0, 4) < 4 else spans
        start, end, _cmp_id = pool[rng.randint(0, len(pool) - 1)]
        width = end - start

        buf = bytearray(data)
        choice = rng.randint(0, 2)

        if choice == 1 and width in _ARITH_WIDTHS:
            self._arith_bump(buf, start, width, rng)
        elif choice == 2:
            i = rng.randint(start, end - 1)
            buf[i] ^= 1 << rng.randint(0, 7)
        else:
            for i in range(start, end):
                buf[i] = rng.randint(0, 255)

        result = bytes(buf)
        if result == data:
            return None
        return result

    @staticmethod
    def _arith_bump(buf: bytearray, start: int, width: int, rng) -> None:
        delta = rng.choice(ARITHMETIC_DELTAS)
        if rng.randint(0, 1) == 0:
            delta = -delta
        if width == 1:
            buf[start] = (buf[start] + delta) & 0xFF
            return
        fmt_char = _ARITH_FMT[width]
        endian = "<" if rng.randint(0, 1) == 0 else ">"
        fmt = endian + fmt_char
        mask = (1 << (width * 8)) - 1
        val = (struct.unpack_from(fmt, buf, start)[0] + delta) & mask
        struct.pack_into(fmt, buf, start, val)


class WeizzChunkMutator(MutatorBase):
    """Whole-chunk duplicate / delete / swap over top-level chunk spans."""

    name = "weizz_chunk_mutate"
    category = "structural"

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        return _preconditions_ok(data, context)

    def mutate(self, data, rng, max_len=0, *, context=None, **ctx):
        if not _preconditions_ok(data, context):
            return None

        smap = _smap_for(data, context.cmplog_pairs)
        chunks = smap.chunk_spans(parent=None)
        if not chunks:
            return None

        buf = bytearray(data)

        if len(chunks) >= 2 and rng.randint(0, 49) < 17:
            result = self._swap(buf, chunks, rng)
        else:
            result = self._duplicate_or_delete(buf, chunks, rng, max_len)

        if result is None or bytes(result) == data:
            return None
        return bytes(result)

    @staticmethod
    def _pick_two_distinct(n: int, rng) -> tuple[int, int]:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        # n >= 2 is guaranteed by the caller; a handful of retries is enough
        # in practice and a hard cap keeps this from looping forever on a
        # pathological rng.
        for _ in range(8):
            if j != i:
                break
            j = rng.randint(0, n - 1)
        return i, j

    @classmethod
    def _swap(cls, buf: bytearray, chunks: list[tuple[int, int, int]], rng) -> bytearray | None:
        i, j = cls._pick_two_distinct(len(chunks), rng)
        if i == j:
            return None
        (s1, e1, _), (s2, e2, _) = sorted((chunks[i], chunks[j]), key=lambda t: t[0])
        if e1 > s2:  # overlap guard; chunk_spans() shouldn't overlap, but be defensive
            return None
        block1 = bytes(buf[s1:e1])
        block2 = bytes(buf[s2:e2])
        if max(len(block1), len(block2)) > _MAX_CHUNK_BLOCK:
            return None
        return bytearray(bytes(buf[:s1]) + block2 + bytes(buf[e1:s2]) + block1 + bytes(buf[e2:]))

    @staticmethod
    def _duplicate_or_delete(
        buf: bytearray, chunks: list[tuple[int, int, int]], rng, max_len: int
    ) -> bytearray | None:
        idx = rng.randint(0, len(chunks) - 1)
        start, end, _cmp_id = chunks[idx]
        size = end - start
        if size > _MAX_CHUNK_BLOCK:
            return None

        want_duplicate = rng.randint(0, 1) == 0
        if want_duplicate and max_len and len(buf) + size > max_len:
            want_duplicate = False  # fall back to delete rather than decline outright

        if want_duplicate:
            block = bytes(buf[start:end])
            ins = rng.randint(0, len(buf))
            buf[ins:ins] = block
            return buf

        if len(buf) - size < 1:
            return None  # never delete the entire input
        del buf[start:end]
        return buf


class WeizzChunkSpliceMutator(MutatorBase):
    """Replace / insert a same-type top-level chunk taken from another seed."""

    name = "weizz_chunk_splice"
    category = "structural"

    # Donors sampled per call before declining; a miss costs two cached map
    # lookups, so a few tries find a type match without a corpus scan.
    _DONOR_TRIES = 4

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        return _preconditions_ok(data, context) and len(context.corpus) >= 2

    def mutate(self, data, rng, max_len=0, *, context=None, **ctx):
        if not _preconditions_ok(data, context) or len(context.corpus) < 2:
            return None
        data = bytes(data)
        pairs = context.cmplog_pairs
        chunks = _smap_for(data, pairs).chunk_spans(parent=None)
        if not chunks:
            return None
        by_type: dict[int, list[tuple[int, int]]] = {}
        for start, end, cmp_id in chunks:
            if end - start <= _MAX_CHUNK_BLOCK:
                by_type.setdefault(cmp_id, []).append((start, end))
        if not by_type:
            return None

        corpus = context.corpus
        for _ in range(self._DONOR_TRIES):
            donor = bytes(corpus[rng.randint(0, len(corpus) - 1)])
            if donor == data or not donor or len(donor) > DEFAULT_MAX_LEN:
                continue
            donor_chunks = [
                c
                for c in _smap_for(donor, pairs).chunk_spans(parent=None)
                if c[2] in by_type and c[1] - c[0] <= _MAX_CHUNK_BLOCK
            ]
            if not donor_chunks:
                continue
            d_start, d_end, cmp_id = donor_chunks[rng.randint(0, len(donor_chunks) - 1)]
            block = donor[d_start:d_end]
            targets = by_type[cmp_id]
            r_start, r_end = targets[rng.randint(0, len(targets) - 1)]
            replaced = data[:r_start] + block + data[r_end:]
            inserted = data[:r_end] + block + data[r_end:]
            out = inserted if rng.randint(0, 2) == 0 else replaced
            if max_len and len(out) > max_len:
                out = replaced  # insert would overrun; replace never grows by more
            if (max_len and len(out) > max_len) or out == data:
                continue
            return out
        return None


def _register() -> None:
    for mutator in (WeizzFieldMutator(), WeizzChunkMutator(), WeizzChunkSpliceMutator()):
        if mutator.name not in REGISTRY.names():
            REGISTRY.register_mutator(mutator)


_register()

__all__ = ["WeizzFieldMutator", "WeizzChunkMutator", "WeizzChunkSpliceMutator"]
