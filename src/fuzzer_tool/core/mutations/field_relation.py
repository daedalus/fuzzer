"""Relation-polarity header mutation for ZIP, WebP ``VP8X`` and ISO-BMFF ``ftyp``.

``covering_array_*`` hits every pair of field values once, but a pair row that
breaks one header relation tends to break several, and a validator stops at its
first failed check. These operators sweep *relation polarity* instead (see
``core/field_constraint``): one row where every relation holds, then one row
per relation (and per ``_VIOLATE`` -subset) where exactly that relation fails
and all the others hold, so each check is exercised in isolation with the rest
of the header valid.

Same fields, gates and round-robin application as ``covering_array_container``;
only the row source differs. Rows are built once from the fuzzer's rng.

Relations are pairwise (``holds(a, b)`` takes the two named fields' values); a
rule over three fields, such as "stored => comp_size == file_size", has no
pairwise form and is not modelled:

  ZIP     descriptor_sizes     flags bit 3 (data descriptor)  =>  comp_size == 0
          deflate64_version    method == 9           =>  version >= 21
          aes_flag             method == 99          =>  flags bit 0 (encrypted)
          empty_comp_size      file_size == 0        =>  comp_size <= 2
          lengths_sum_fits     name_len + extra_len <= 0xFFFF
  WebP    riff_covers_chunk    riff_size >= chunk_size + 12
          canvas_fits_u32      (width + 1) * (height + 1) < 2**32
          reserved_flag_bits   flags & 0xC1 == 0     =>  reserved == 0
  BMFF    brand_fits_box       major != 0            =>  size >= 16 (or 0 / 1)
          minor_fits_box       minor != 0            =>  size >= 16 (or 0 / 1)

Selection share on a real target is unmeasured, like the covering-array arms.
"""

from __future__ import annotations

from typing import Any, ClassVar

from fuzzer_tool.core import field_constraint
from fuzzer_tool.core.field_constraint import Relation
from fuzzer_tool.core.mutations.covering_array_container import (
    ISOBMFF_FIELDS,
    WEBP_FIELDS,
    ZIP_FIELDS,
    _HeaderCoveringMutator,
)

# Relations violated together per pattern. 1 = each check alone; 2 adds every pair.
_VIOLATE = 2
# Rows drawn per polarity pattern.
_PER_PATTERN = 2

_ZIP_IDX = {f.name: i for i, f in enumerate(ZIP_FIELDS)}
_WEBP_IDX = {f.name: i for i, f in enumerate(WEBP_FIELDS)}
_BMFF_IDX = {f.name: i for i, f in enumerate(ISOBMFF_FIELDS)}


def _zip_relations() -> tuple[Relation, ...]:
    m, c, f, v = _ZIP_IDX["method"], _ZIP_IDX["comp_size"], _ZIP_IDX["flags"], _ZIP_IDX["version"]
    fs, nl, el = _ZIP_IDX["file_size"], _ZIP_IDX["name_len"], _ZIP_IDX["extra_len"]
    return (
        Relation("descriptor_sizes", f, c, lambda flags, comp: not flags & 8 or comp == 0),
        Relation("deflate64_version", m, v, lambda method, ver: method != 9 or ver >= 21),
        Relation("aes_flag", m, f, lambda method, flags: method != 99 or bool(flags & 1)),
        Relation("empty_comp_size", fs, c, lambda size, comp: size != 0 or comp <= 2),
        Relation("lengths_sum_fits", nl, el, lambda name, extra: name + extra <= 0xFFFF),
    )


def _webp_relations() -> tuple[Relation, ...]:
    r, ch, fl = _WEBP_IDX["riff_size"], _WEBP_IDX["chunk_size"], _WEBP_IDX["flags"]
    rs, w, h = _WEBP_IDX["reserved"], _WEBP_IDX["width"], _WEBP_IDX["height"]
    return (
        Relation("riff_covers_chunk", r, ch, lambda riff, chunk: riff >= chunk + 12),
        Relation("canvas_fits_u32", w, h, lambda wd, ht: (wd + 1) * (ht + 1) < 1 << 32),
        Relation("reserved_flag_bits", fl, rs, lambda flags, res: not flags & 0xC1 or res == 0),
    )


def _bmff_relations() -> tuple[Relation, ...]:
    sz, ma, mi = _BMFF_IDX["size"], _BMFF_IDX["major"], _BMFF_IDX["minor"]

    def fits(size: int, other: int) -> bool:
        return other == 0 or size in (0, 1) or size >= 16

    return (
        Relation("brand_fits_box", sz, ma, fits),
        Relation("minor_fits_box", sz, mi, fits),
    )


ZIP_RELATIONS = _zip_relations()
WEBP_RELATIONS = _webp_relations()
ISOBMFF_RELATIONS = _bmff_relations()


class _RelationMutator(_HeaderCoveringMutator):
    """Header mutator whose rows come from ``field_constraint.relation_rows``."""

    relations: ClassVar[tuple[Relation, ...]] = ()

    def _build_rows(self, rng: Any) -> list[tuple[int, ...]]:
        vs = [tuple(f.values) for f in self.fields]
        rows = field_constraint.relation_rows(
            vs, self.relations, violate=_VIOLATE, per_pattern=_PER_PATTERN, rng=rng
        )
        return rows or super()._build_rows(rng)


class ZipRelationMutator(_RelationMutator):
    """``field_relation_zip``: first local file header, one broken check at a time."""

    name = "field_relation_zip"
    fields = ZIP_FIELDS
    relations = ZIP_RELATIONS

    def _gate(self, data: bytes) -> bool:
        return data[:4] == b"PK\x03\x04"


class WebpRelationMutator(_RelationMutator):
    """``field_relation_webp``: RIFF/VP8X sizes, canvas, reserved bits."""

    name = "field_relation_webp"
    fields = WEBP_FIELDS
    relations = WEBP_RELATIONS

    def _gate(self, data: bytes) -> bool:
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP" and data[12:16] == b"VP8X"


class IsobmffRelationMutator(_RelationMutator):
    """``field_relation_isobmff``: ``ftyp`` box size against brand and version."""

    name = "field_relation_isobmff"
    fields = ISOBMFF_FIELDS
    relations = ISOBMFF_RELATIONS

    def _gate(self, data: bytes) -> bool:
        return data[4:8] == b"ftyp"


def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY

    for cls in (ZipRelationMutator, WebpRelationMutator, IsobmffRelationMutator):
        m = cls()
        if m.name not in REGISTRY.names():
            REGISTRY.register_mutator(m)


_register()
