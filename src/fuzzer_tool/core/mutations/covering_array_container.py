"""Pairwise-covering header mutation for WebP ``VP8X``, ISO-BMFF ``ftyp`` and ZIP.

Remaining formats of handover ``handover_combinatorics_permutations`` §6 item 2a,
built like ``covering_array_gzip``: fixed-offset fields, a t=2 covering array
over their values, rows applied round-robin so every field-value pair is hit
across repeated selections. Bodies are untouched. Each arm is gated on its own
magic and costs nothing on other targets.

    WebP VP8X (RIFF, little-endian)        ISO-BMFF ftyp (big-endian)
      4  riff_size                           0  size      (0 = to EOF, 1 = 64-bit)
     16  chunk_size (spec: 10)               8  major     brand
     20  flags (ICC/alpha/EXIF/XMP/anim)    12  minor     version
     21  reserved (3)
     24  width-1 (3)                       ZIP local file header (little-endian)
     27  height-1 (3)                        4 version  6 flags  8 method
                                            18 comp_size  22 file_size
                                            26 name_len  28 extra_len

ISO-BMFF ``tkhd`` is not covered: it sits at a variable depth (moov/trak/tkhd)
and needs a box walker, not fixed offsets. Selection share on a real target is
unmeasured, the same open question as the IHDR and gzip arms.
"""

from __future__ import annotations

from typing import Any, ClassVar

from fuzzer_tool.core import covering_array, field_spec
from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase

_U32 = 0xFFFFFFFF
_U24 = 0xFFFFFF
_U16 = 0xFFFF

_RIFF = b"RIFF"
_WEBP = b"WEBP"
_VP8X = b"VP8X"
_FTYP = b"ftyp"
_ZIP_LFH = b"PK\x03\x04"


def _brand(tag: bytes) -> int:
    return int.from_bytes(tag, "big")


# VP8X flags: 0x02 animation, 0x04 XMP, 0x08 EXIF, 0x10 alpha, 0x20 ICC;
# 0xC1 = reserved bits. Width/height are stored minus one, so 0xFFFFFF
# makes (w * h) overflow 32 bits in a decoder that multiplies them.
WEBP_FIELDS: tuple[field_spec.FieldDef, ...] = (
    field_spec.FieldDef("riff_size", 4, 4, little=True, values=(0, 4, _U32)),
    field_spec.FieldDef("chunk_size", 16, 4, little=True, values=(0, 9, 10, 11, _U32)),
    field_spec.FieldDef("flags", 20, 1, values=(0, 0x02, 0x04, 0x08, 0x10, 0x20, 0x3E, 0xC1, 0xFF)),
    field_spec.FieldDef("reserved", 21, 3, values=(0, 1, _U24)),
    field_spec.FieldDef("width", 24, 3, little=True, values=(0, 1, 0x7FFF, _U24)),
    field_spec.FieldDef("height", 27, 3, little=True, values=(0, 1, 0x7FFF, _U24)),
)

# size 0 = box runs to EOF, 1 = 64-bit largesize follows, 8 = header only.
ISOBMFF_FIELDS: tuple[field_spec.FieldDef, ...] = (
    field_spec.FieldDef("size", 0, 4, values=(0, 1, 8, 16, _U32)),
    field_spec.FieldDef(
        "major",
        8,
        4,
        values=tuple(_brand(b) for b in (b"isom", b"mp42", b"qt  ", b"heic", b"avif")) + (0,),
    ),
    field_spec.FieldDef("minor", 12, 4, values=(0, 1, _U32)),
)

# flags: bit0 encrypted, bit3 data descriptor (sizes/CRC follow the data),
# bit11 UTF-8 names. method: 0 stored, 8 deflate, 9 deflate64, 12 bzip2,
# 14 lzma, 93 zstd, 99 AES.
ZIP_FIELDS: tuple[field_spec.FieldDef, ...] = (
    field_spec.FieldDef("version", 4, 2, little=True, values=(0, 10, 20, 45, _U16)),
    field_spec.FieldDef("flags", 6, 2, little=True, values=(0, 1, 8, 0x800, _U16)),
    field_spec.FieldDef("method", 8, 2, little=True, values=(0, 8, 9, 12, 14, 93, 99, _U16)),
    field_spec.FieldDef("comp_size", 18, 4, little=True, values=(0, 1, _U32)),
    field_spec.FieldDef("file_size", 22, 4, little=True, values=(0, 1, _U32)),
    field_spec.FieldDef("name_len", 26, 2, little=True, values=(0, 1, _U16)),
    field_spec.FieldDef("extra_len", 28, 2, little=True, values=(0, 1, _U16)),
)


class _HeaderCoveringMutator(MutatorBase):
    """Sweeps a pairwise covering array over a fixed-offset header."""

    category = "format"
    fields: ClassVar[tuple[field_spec.FieldDef, ...]] = ()

    def __init__(self) -> None:
        self._rows: list[tuple[int, ...]] | None = None
        self._idx = 0
        self._header_len = max(f.end for f in self.fields)

    def _gate(self, data: bytes) -> bool:
        raise NotImplementedError

    def _ok(self, data: bytes) -> bool:
        return len(data) >= self._header_len and self._gate(data)

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        return self._ok(data)

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        if not self._ok(data):
            return None

        # Built once from the fuzzer's rng (reproducible under --seed), then
        # round-robin: redrawing per call would drop the coverage guarantee.
        if self._rows is None:
            vs = tuple(f.values for f in self.fields)
            self._rows = covering_array.generate(vs, t=2, rng=rng)

        row = self._rows[self._idx % len(self._rows)]
        self._idx += 1

        out = field_spec.apply_row(data, self.fields, row)
        if max_len:
            out = out[:max_len]
        return out if out != data else None


class WebpCoveringArrayMutator(_HeaderCoveringMutator):
    """``covering_array_webp``: RIFF size and the VP8X chunk (flags, canvas)."""

    name = "covering_array_webp"
    fields = WEBP_FIELDS

    def _gate(self, data: bytes) -> bool:
        return data[:4] == _RIFF and data[8:12] == _WEBP and data[12:16] == _VP8X


class IsobmffCoveringArrayMutator(_HeaderCoveringMutator):
    """``covering_array_isobmff``: ``ftyp`` box size, major brand, minor version."""

    name = "covering_array_isobmff"
    fields = ISOBMFF_FIELDS

    def _gate(self, data: bytes) -> bool:
        return data[4:8] == _FTYP


class ZipCoveringArrayMutator(_HeaderCoveringMutator):
    """``covering_array_zip``: first local file header (method, flags, sizes, lengths)."""

    name = "covering_array_zip"
    fields = ZIP_FIELDS

    def _gate(self, data: bytes) -> bool:
        return data[:4] == _ZIP_LFH


def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY

    for cls in (WebpCoveringArrayMutator, IsobmffCoveringArrayMutator, ZipCoveringArrayMutator):
        m = cls()
        if m.name not in REGISTRY.names():
            REGISTRY.register_mutator(m)


_register()
