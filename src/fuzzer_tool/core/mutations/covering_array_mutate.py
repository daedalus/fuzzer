"""``covering_array_ihdr``: pairwise-covering PNG IHDR field mutation.

The P3 candidate from ``docs/handover/handover_generators_2026-09-20.md``:
PNG's IHDR chunk is 7 small-domain fields (width, height, bit depth, color
type, compression method, filter method, interlace method) packed into 13
bytes. ``png.py``'s existing ``_mutate_ihdr`` picks one field per call and
draws it independently -- across many calls every *individual* boundary
value gets tried, but two fields are never deliberately set together. A
decoder bug gated on a specific *pair* -- e.g. ``color_type=3`` (indexed,
so ``bit_depth`` must be 1/2/4/8 per the PNG spec) paired with
``bit_depth=16`` (valid alone, invalid for indexed) -- is reached only by
the luck of two independent draws landing right in the same call.

This operator instead sweeps a precomputed pairwise covering array
(``core/covering_array.py``): every one of the 487 required (field-pair,
value-pair) combinations across the 7 fields' domains is guaranteed to
appear in some row, in 57 rows here, instead of never or only by chance.
Rows are applied round-robin (not redrawn per call) specifically so that
guarantee holds across repeated selections of this operator, the same way
a test suite exhausts a table instead of sampling it.

Field domains were chosen to mirror boundary values ``png.py``'s own
mutator already uses (``bit_depth``'s ``[0, 1, 2, 4, 8, 16, 255]``,
``interlace``'s ``[0, 1, 42, 255]``) plus the PNG-spec-valid values for
fields it does not enumerate (``color_type``, ``compression``, ``filter``),
rather than inventing an unrelated set -- see ``_FIELDS`` below for the
per-field rationale.

Does not attempt other formats' header fields; PNG was the doc's own
worked example and the only rollout target here. A second format follows
the same pattern (a ``_FIELDS``-shaped table plus the same lazy
build/round-robin operator body) if this earns its keep.
"""

from __future__ import annotations

import struct
from typing import Any

from fuzzer_tool.core import covering_array
from fuzzer_tool.core.mutations.png import parse_png_chunks, serialize_png_chunks
from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# (name, byte offset within IHDR data, value set). Order matches IHDR's own
# on-disk layout; width/height are 4-byte big-endian, the rest are 1 byte.
#
#   width/height: PNG allows any 1..2^31-1; 0 is explicitly invalid (spec
#     4.1.1 forbids width=0). Boundary set: 0 (invalid), 1/2 (smallest
#     valid), 2^31-1 (largest signed-positive), 2^32-1 (unsigned-max,
#     negative if a decoder reads it as int32).
#   bit_depth: same set png.py's own _mutate_ihdr already draws from --
#     valid values are 1/2/4/8/16 depending on color_type, 0 and 255 are
#     always invalid.
#   color_type: PNG spec's five defined values (0 grayscale, 2 truecolor,
#     3 indexed, 4 grayscale+alpha, 6 truecolor+alpha); 1 and 255 are
#     always-undefined values a decoder must reject.
#   compression / filter: the spec defines exactly one method (0) for
#     each; 1 and 255 exercise the "unsupported method" rejection path.
#   interlace: same set png.py's own _mutate_interlace already draws from
#     -- 0/1 are the two defined methods, 42/255 are undefined.
_FIELDS: tuple[tuple[str, int, tuple[int, ...]], ...] = (
    ("width", 0, (0, 1, 2, 0x7FFFFFFF, 0xFFFFFFFF)),
    ("height", 4, (0, 1, 2, 0x7FFFFFFF, 0xFFFFFFFF)),
    ("bit_depth", 8, (0, 1, 2, 4, 8, 16, 255)),
    ("color_type", 9, (0, 1, 2, 3, 4, 6, 255)),
    ("compression", 10, (0, 1, 255)),
    ("filter", 11, (0, 1, 255)),
    ("interlace", 12, (0, 1, 42, 255)),
)

# core/covering_array.generate()'s value_sets parameter, in _FIELDS order.
_VALUE_SETS: tuple[tuple[int, ...], ...] = tuple(vs for _n, _o, vs in _FIELDS)

IHDR_LEN = 13


def _apply_row(ihdr_data: bytes, row: tuple[int, ...]) -> bytes:
    """Write *row*'s values into a copy of *ihdr_data* at each field's offset.

    Only the first 13 bytes are touched; any trailing bytes an already-
    malformed IHDR carries past the spec length are passed through
    unchanged, matching png.py's own ``_mutate_ihdr``.
    """
    data = bytearray(ihdr_data)
    for (name, offset, _values), value in zip(_FIELDS, row, strict=True):
        if name in ("width", "height"):
            struct.pack_into(">I", data, offset, value)
        else:
            data[offset] = value
    return bytes(data)


class PngCoveringArrayMutator(MutatorBase):
    """``covering_array_ihdr``: sweeps a pairwise covering array over IHDR.

    The array is built once, lazily, on first ``mutate()`` call (using the
    fuzzer's own rng, so it is reproducible under ``--seed`` like every
    other draw), then applied round-robin -- see the module docstring for
    why redrawing per call instead would silently drop the coverage
    guarantee this operator exists for.
    """

    name = "covering_array_ihdr"
    category = "format"

    def __init__(self) -> None:
        self._rows: list[tuple[int, ...]] | None = None
        self._idx = 0

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        return bool(data) and data[:8] == _PNG_MAGIC

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        if not data or data[:8] != _PNG_MAGIC:
            return None

        chunks = parse_png_chunks(data)
        if not chunks:
            return None
        ihdr = next((c for c in chunks if c.chunk_type == b"IHDR"), None)
        if ihdr is None or len(ihdr.data) < IHDR_LEN:
            return None

        if self._rows is None:
            self._rows = covering_array.generate(_VALUE_SETS, t=2, rng=rng)
            if not self._rows:  # pragma: no cover - unreachable, domains non-empty
                return None

        row = self._rows[self._idx % len(self._rows)]
        self._idx += 1

        ihdr.data = _apply_row(ihdr.data, row)
        out = serialize_png_chunks(chunks)
        if max_len:
            out = out[:max_len]
        return out if out != data else None


def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY

    m = PngCoveringArrayMutator()
    if m.name not in REGISTRY.names():
        REGISTRY.register_mutator(m)


_register()
