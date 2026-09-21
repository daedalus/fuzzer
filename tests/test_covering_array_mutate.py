"""Tests for core/mutations/covering_array_mutate.py -- covering_array_ihdr.

Covers the P3 "covering-array header-field generator" candidate from
``docs/handover/handover_generators_2026-09-20.md``: guarantees every
pairwise combination of PNG IHDR field boundary values is exercised
across repeated selections, rather than left to chance the way
``png.py``'s own per-field ``_mutate_ihdr`` draws are.
"""

from __future__ import annotations

import random
import struct

from fuzzer_tool.core import covering_array as ca
from fuzzer_tool.core.crc32 import crc32
from fuzzer_tool.core.mutations.covering_array_mutate import (
    _FIELDS,
    _VALUE_SETS,
    IHDR_LEN,
    PngCoveringArrayMutator,
    _apply_row,
)
from fuzzer_tool.core.mutations.png import PngChunk, parse_png_chunks
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.operator_registry import REGISTRY

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _Rng:
    """Minimal RandPool-shaped rng (randint/choice) over a stdlib Random."""

    def __init__(self, seed: int = 7):
        self._r = random.Random(seed)

    def randint(self, a, b):
        return self._r.randint(a, b)

    def choice(self, seq):
        return self._r.choice(seq)


def _make_png(ihdr_data: bytes | None = None, extra: list[PngChunk] | None = None) -> bytes:
    if ihdr_data is None:
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    chunks = [PngChunk(b"IHDR", ihdr_data)]
    if extra:
        chunks.extend(extra)
    chunks.append(PngChunk(b"IEND", b""))
    return _PNG_MAGIC + b"".join(c.serialize() for c in chunks)


# ═══════════════════════════════════════════════════════════════════
# _apply_row
# ═══════════════════════════════════════════════════════════════════


class TestApplyRow:
    def test_writes_every_field_at_its_offset(self):
        base = bytes(IHDR_LEN)
        row = (100, 200, 8, 3, 0, 0, 1)
        out = _apply_row(base, row)
        assert struct.unpack_from(">I", out, 0)[0] == 100
        assert struct.unpack_from(">I", out, 4)[0] == 200
        assert out[8] == 8
        assert out[9] == 3
        assert out[10] == 0
        assert out[11] == 0
        assert out[12] == 1

    def test_preserves_trailing_bytes_past_spec_length(self):
        base = bytes(IHDR_LEN) + b"\xaa\xbb"
        row = (1, 1, 8, 2, 0, 0, 0)
        out = _apply_row(base, row)
        assert len(out) == IHDR_LEN + 2
        assert out[IHDR_LEN:] == b"\xaa\xbb"

    def test_row_order_matches_fields_order(self):
        assert [name for name, _o, _v in _FIELDS] == [
            "width",
            "height",
            "bit_depth",
            "color_type",
            "compression",
            "filter",
            "interlace",
        ]


# ═══════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════


class TestRegistration:
    def test_registered_under_expected_name_and_category(self):
        assert "covering_array_ihdr" in REGISTRY.names()
        assert REGISTRY.category_of("covering_array_ihdr") == "format"


# ═══════════════════════════════════════════════════════════════════
# is_available
# ═══════════════════════════════════════════════════════════════════


class TestIsAvailable:
    def test_available_on_png_magic(self):
        m = PngCoveringArrayMutator()
        ctx = MutationContext()
        assert m.is_available(ctx, _make_png())

    def test_unavailable_on_non_png(self):
        m = PngCoveringArrayMutator()
        ctx = MutationContext()
        assert not m.is_available(ctx, b"not a png at all, just text")

    def test_unavailable_on_empty(self):
        m = PngCoveringArrayMutator()
        ctx = MutationContext()
        assert not m.is_available(ctx, b"")


# ═══════════════════════════════════════════════════════════════════
# mutate: decline cases
# ═══════════════════════════════════════════════════════════════════


class TestMutateDeclines:
    def test_declines_empty_data(self):
        m = PngCoveringArrayMutator()
        assert m.mutate(b"", _Rng()) is None

    def test_declines_non_png(self):
        m = PngCoveringArrayMutator()
        assert m.mutate(b"garbage bytes here", _Rng()) is None

    def test_declines_unparseable_png(self):
        m = PngCoveringArrayMutator()
        # Magic present, nothing valid after it.
        assert m.mutate(_PNG_MAGIC, _Rng()) is None

    def test_declines_png_without_ihdr(self):
        m = PngCoveringArrayMutator()
        data = _PNG_MAGIC + PngChunk(b"IEND", b"").serialize()
        assert m.mutate(data, _Rng()) is None

    def test_declines_truncated_ihdr(self):
        m = PngCoveringArrayMutator()
        data = _make_png(ihdr_data=b"\x00" * 5)
        assert m.mutate(data, _Rng()) is None


# ═══════════════════════════════════════════════════════════════════
# mutate: output validity
# ═══════════════════════════════════════════════════════════════════


class TestMutateOutput:
    def test_output_is_valid_png(self):
        m = PngCoveringArrayMutator()
        data = _make_png()
        out = m.mutate(data, _Rng())
        assert out is not None
        chunks = parse_png_chunks(out)
        assert chunks is not None
        assert chunks[0].chunk_type == b"IHDR"
        assert len(chunks[0].data) >= IHDR_LEN

    def test_crc_is_valid_after_mutation(self):
        m = PngCoveringArrayMutator()
        data = _make_png()
        out = m.mutate(data, _Rng())
        chunks = parse_png_chunks(out)
        ihdr = chunks[0]
        raw = ihdr.serialize()
        stored_crc = struct.unpack_from(">I", raw, len(raw) - 4)[0]
        assert stored_crc == crc32(b"IHDR" + ihdr.data) & 0xFFFFFFFF

    def test_respects_max_len(self):
        m = PngCoveringArrayMutator()
        data = _make_png()
        out = m.mutate(data, _Rng(), max_len=20)
        assert out is None or len(out) <= 20

    def test_preserves_non_ihdr_chunks(self):
        m = PngCoveringArrayMutator()
        extra = PngChunk(b"tEXt", b"hello world")
        data = _make_png(extra=[extra])
        out = m.mutate(data, _Rng())
        chunks = parse_png_chunks(out)
        types = [c.chunk_type for c in chunks]
        assert b"tEXt" in types
        assert chunks[types.index(b"tEXt")].data == b"hello world"

    def test_only_touches_ihdr_bytes_field_values(self):
        m = PngCoveringArrayMutator()
        data = _make_png()
        out = m.mutate(data, _Rng())
        ihdr = parse_png_chunks(out)[0]
        for _name, offset, values in _FIELDS:
            width = 4 if offset in (0, 4) else 1
            if width == 4:
                val = struct.unpack_from(">I", ihdr.data, offset)[0]
            else:
                val = ihdr.data[offset]
            assert val in values


# ═══════════════════════════════════════════════════════════════════
# Round-robin coverage guarantee -- the operator's whole reason to exist
# ═══════════════════════════════════════════════════════════════════


class TestRoundRobinCoverage:
    def test_repeated_calls_achieve_full_pairwise_coverage(self):
        m = PngCoveringArrayMutator()
        rng = _Rng(seed=42)
        data = _make_png()

        observed_rows = set()
        # Enough calls to exhaust the array at least once; the array's
        # own size is an implementation detail, so call generously more
        # times than core/covering_array.py's own test showed it needs
        # (< 500 rows for these exact domains).
        for _ in range(700):
            out = m.mutate(data, rng)
            assert out is not None
            ihdr = parse_png_chunks(out)[0]
            row = []
            for name, offset, _values in _FIELDS:
                if name in ("width", "height"):
                    row.append(struct.unpack_from(">I", ihdr.data, offset)[0])
                else:
                    row.append(ihdr.data[offset])
            observed_rows.add(tuple(row))

        assert ca.verify_coverage(list(observed_rows), _VALUE_SETS, t=2)

    def test_rows_built_lazily_and_cached_on_first_call(self):
        m = PngCoveringArrayMutator()
        assert m._rows is None
        m.mutate(_make_png(), _Rng())
        assert m._rows is not None
        rows_after_first = m._rows
        m.mutate(_make_png(), _Rng())
        # Same list object -- not rebuilt on every call.
        assert m._rows is rows_after_first

    def test_index_advances_round_robin_not_random(self):
        m = PngCoveringArrayMutator()
        rng = _Rng(seed=3)
        data = _make_png()
        m.mutate(data, rng)
        assert m._idx == 1
        m.mutate(data, rng)
        assert m._idx == 2

    def test_declines_when_row_equals_current_ihdr_exactly(self):
        # If the row the round-robin lands on happens to match the
        # input's current field values exactly, mutate() must decline
        # (no-op) rather than return an unchanged buffer.
        m = PngCoveringArrayMutator()
        rng = _Rng(seed=11)
        m._rows = [(1, 1, 8, 2, 0, 0, 0)]
        m._idx = 0
        data = _make_png(ihdr_data=struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        assert m.mutate(data, rng) is None


# ═══════════════════════════════════════════════════════════════════
# Field domains
# ═══════════════════════════════════════════════════════════════════


class TestFieldDomains:
    def test_all_domains_non_empty(self):
        for _name, _offset, values in _FIELDS:
            assert len(values) > 0

    def test_value_sets_matches_fields_order(self):
        assert tuple(v for _n, _o, v in _FIELDS) == _VALUE_SETS

    def test_domains_cover_all_png_spec_valid_color_types(self):
        color_type_values = next(v for n, _o, v in _FIELDS if n == "color_type")
        assert {0, 2, 3, 4, 6} <= set(color_type_values)

    def test_pairwise_coverage_reachable_with_bounded_rows(self):
        # Sanity bound on the domains actually shipped, independent of
        # core/covering_array.py's own test -- catches a domain edit
        # that blows up the row count without anyone noticing.
        rows = ca.generate(_VALUE_SETS, t=2, rng=random.Random(0))
        assert ca.verify_coverage(rows, _VALUE_SETS, t=2)
        assert len(rows) < 500
