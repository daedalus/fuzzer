"""P2 — field- and chunk-scoped operators driven by Weizz structure tags.

Port of the second item in
``docs/handover/handover_weizz_structure_aware_port_2026-08-31.md``: mutation
operators that treat a per-seed :class:`~fuzzer_tool.core.weizz_tags.StructureMap`
as a set of fields (same-``cmp_id`` runs) and chunks (top-level same-parent
spans), rather than mutating at arbitrary byte offsets.

Two operators, both class-based (:mod:`fuzzer_tool.core.mutator_interface`)
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

**Tag cost model:** both operators build the :class:`StructureMap` fresh
from ``data`` and ``context.cmplog_pairs`` on every ``mutate()`` call via
the passive collector (:func:`build_tag_map_from_cmplog`) — there is no
once-per-lineage cache yet (§4 step 3 / §6 "Differential pass cost" in the
handover doc is about the *active* ``get_deps`` path, which these operators
do not use). The passive collector is a single substring-matching pass
gated by ``TagCollectorConfig.max_input_len`` (default
``weizz_tags.DEFAULT_MAX_LEN``, mirrored below), so the per-call cost is
bounded but still paid repeatedly; caching on seed metadata
(``attach_tags_to_meta`` / ``load_tags_from_meta`` already exist for this)
is the natural follow-up once these operators show signal under
``--elo``/paired bench.

Gate: both operators decline (``is_available`` returns ``False``) unless
``--weizz-tags`` is set (``context.weizz_tags_enabled``), matching the
handover doc's acceptance checklist ("Flag default off").
"""

from __future__ import annotations

import struct

from fuzzer_tool.core.mutations.generic import ARITHMETIC_DELTAS
from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.weizz_tags import (
    DEFAULT_MAX_LEN,
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

        smap = build_tag_map_from_cmplog(bytes(data), list(context.cmplog_pairs))
        spans = smap.field_spans()
        if not spans:
            return None

        # Prefer non-magic fields most of the time; magic bytes are already
        # a well-covered target of dict/interesting-value operators, and
        # constantly clobbering them wastes the tag signal this operator
        # exists to exploit.
        non_magic = [s for s in spans if not (smap.tags[s[0]].flags & TagFlags.IS_MAGIC)]
        pool = non_magic if non_magic and rng.random() < 0.8 else spans
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
        if rng.random() < 0.5:
            delta = -delta
        if width == 1:
            buf[start] = (buf[start] + delta) & 0xFF
            return
        fmt_char = _ARITH_FMT[width]
        endian = "<" if rng.random() < 0.5 else ">"
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

        smap = build_tag_map_from_cmplog(bytes(data), list(context.cmplog_pairs))
        chunks = smap.chunk_spans(parent=None)
        if not chunks:
            return None

        buf = bytearray(data)

        if len(chunks) >= 2 and rng.random() < 0.34:
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


def _register() -> None:
    for mutator in (WeizzFieldMutator(), WeizzChunkMutator()):
        if mutator.name not in REGISTRY.names():
            REGISTRY.register_mutator(mutator)


_register()

__all__ = ["WeizzFieldMutator", "WeizzChunkMutator"]
