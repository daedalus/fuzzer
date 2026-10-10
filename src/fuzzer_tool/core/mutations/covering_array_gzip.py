"""``covering_array_gzip``: pairwise-covering gzip header field mutation.

Second format for ``core/covering_array.py`` after PNG IHDR (see
``covering_array_mutate.py``; handover ``handover_combinatorics_permutations``
§6 item 2a). The RFC 1952 member header is 10 fixed-offset bytes:

    0   2   1f 8b  magic (kept)
    2   1   CM     compression method (8 = deflate)
    3   1   FLG    FTEXT/FHCRC/FEXTRA/FNAME/FCOMMENT + reserved bits 5-7
    4   4   MTIME  little-endian
    8   1   XFL
    9   1   OS

Decoder bugs live in *combinations*: ``FLG=FEXTRA|FNAME`` makes the parser read
optional fields the body never provides, and only matters for some ``CM``.
Rows are applied round-robin so every field-value pair is hit across repeated
selections. The body is untouched.

Availability is gated on the gzip magic, so the arm costs nothing on other
targets. Whether it earns selection share on a real gzip target is unmeasured,
the same open question as the IHDR arm.
"""

from __future__ import annotations

from typing import Any

from fuzzer_tool.core import covering_array, field_spec
from fuzzer_tool.core.mutator_interface import Availability, MutationContext, MutatorBase

_GZIP_MAGIC = b"\x1f\x8b"
HEADER_LEN = 10

# Values per field: spec-defined ones plus always-invalid edges.
#   CM: 8 deflate; 0-7 reserved; 255 max.
#   FLG: each defined bit alone, all defined (0x1f), reserved bits (0xe0), all.
#   MTIME: zero, one, max (unsigned; negative if read as int32).
#   XFL: 0 none, 2 best, 4 fastest, 255 undefined.
#   OS: 0 FAT, 3 Unix, 11 NTFS, 255 unknown.
_FIELDS: tuple[field_spec.FieldDef, ...] = (
    field_spec.FieldDef("cm", 2, 1, values=(0, 1, 8, 255)),
    field_spec.FieldDef("flg", 3, 1, values=(0, 1, 2, 4, 8, 16, 0x1F, 0xE0, 0xFF)),
    field_spec.FieldDef("mtime", 4, 4, little=True, values=(0, 1, 0xFFFFFFFF)),
    field_spec.FieldDef("xfl", 8, 1, values=(0, 2, 4, 255)),
    field_spec.FieldDef("os", 9, 1, values=(0, 3, 11, 255)),
)
_VALUE_SETS = tuple(f.values for f in _FIELDS)


class GzipCoveringArrayMutator(MutatorBase):
    """``covering_array_gzip``: sweeps a pairwise covering array over the header."""

    availability = Availability.INPUT  # is_available reads only the input

    name = "covering_array_gzip"
    category = "format"

    def __init__(self) -> None:
        self._rows: list[tuple[int, ...]] | None = None
        self._idx = 0

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        return len(data) >= HEADER_LEN and data[:2] == _GZIP_MAGIC

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        if len(data) < HEADER_LEN or data[:2] != _GZIP_MAGIC:
            return None

        # Built once from the fuzzer's rng (reproducible under --seed), then
        # round-robin: redrawing per call would drop the coverage guarantee.
        if self._rows is None:
            self._rows = covering_array.generate(
                _VALUE_SETS, t=2, rng=rng, strategy=covering_array.OPERATOR_STRATEGY
            )

        row = self._rows[self._idx % len(self._rows)]
        self._idx += 1

        out = field_spec.apply_row(data, _FIELDS, row)
        if max_len:
            out = out[:max_len]
        return out if out != data else None


def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY

    m = GzipCoveringArrayMutator()
    if m.name not in REGISTRY.names():
        REGISTRY.register_mutator(m)


_register()
