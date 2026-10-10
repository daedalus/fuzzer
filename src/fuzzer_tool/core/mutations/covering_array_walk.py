"""Covering-array operators whose fields sit at data-dependent offsets.

``covering_array_container.py`` covers fields at *fixed* offsets (WebP VP8X,
ISO-BMFF ftyp, the first ZIP local header). This module covers the rest of the
combinatorics handover section 6 item 2a, where a field is found by walking the
container structure first:

    covering_array_isobmff_tkhd   moov/trak/tkhd (v0 and v1 layouts)
    covering_array_wav            RIFF/WAVE fmt + data sizes
    covering_array_avi            RIFF/AVI  LIST hdrl / avih
    covering_array_webp_vp8       VP8 lossy frame header (plain or in VP8X)
    covering_array_webp_vp8l      VP8L lossless signature + packed size word
    covering_array_zip_eocd       end-of-central-directory record
    covering_array_zip_cd         one central-directory header (random entry)
    covering_array_zip_entry      local header of a later entry (index >= 1)

Each operator owns a fixed list of ``(field, candidate values)``; the pairwise
rows are generated once from those value sets and applied round-robin, so the
coverage guarantee is per field set, not per located layout. Where an input has
several candidates (ZIP entries) one is drawn per call, so pair coverage holds
per row index but not per entry.

A walk that cannot find every field declines (``is_available`` False,
``mutate`` None), so an operator costs nothing on other formats. Walks are
bounded (box/chunk/entry caps) and never raise on garbage input.

Selection share on a real target is unmeasured, like the other format arms.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

from fuzzer_tool.core import covering_array, field_spec
from fuzzer_tool.core.mutations.covering_array_container import ZIP_FIELDS
from fuzzer_tool.core.mutator_interface import Availability, MutationContext, MutatorBase

_U32 = 0xFFFFFFFF
_U24 = 0xFFFFFF
_U16 = 0xFFFF

# Walk caps: a fuzzed size field must never turn the gate into a long loop.
_MAX_NODES = 4096
_MAX_ZIP_ENTRIES = 64
_EOCD_LEN = 22
_EOCD_SEARCH = _EOCD_LEN + 0xFFFF
_CDFH_LEN = 46
_LFH_LEN = 30

# name -> (absolute offset, size, little-endian)
Layout = dict[str, tuple[int, int, bool]]
Values = tuple[tuple[str, tuple[int, ...]], ...]


# --------------------------------------------------------------------------
# Base


class _WalkedCoveringMutator(MutatorBase):
    """Fields come from ``layout(data, rng)``; values are fixed per operator."""

    availability = Availability.INPUT  # is_available reads only the input

    category = "format"
    values: ClassVar[Values] = ()

    def __init__(self) -> None:
        self._rows: list[tuple[int, ...]] | None = None
        self._idx = 0

    def layout(self, data: bytes, rng: Any) -> Layout | None:
        raise NotImplementedError

    def value_sets(self) -> list[tuple[int, ...]]:
        return [v for _, v in self.values]

    def fields_for(self, data: bytes, rng: Any = None) -> tuple[field_spec.FieldDef, ...] | None:
        """FieldDefs located in *data* (declared order), or None if any is missing."""
        found = self.layout(data, rng)
        if found is None:
            return None

        out = []
        for name, vals in self.values:
            if name not in found:
                return None
            off, size, little = found[name]
            if off < 0 or off + size > len(data):
                return None
            out.append(field_spec.FieldDef(name, off, size, little=little, values=vals))
        return tuple(out)

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        return self.fields_for(data) is not None

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        fields = self.fields_for(data, rng)
        if fields is None:
            return None

        # Built once from the fuzzer's rng, then round-robin: redrawing per
        # call would drop the pairwise guarantee.
        if self._rows is None:
            self._rows = covering_array.generate(
                self.value_sets(), t=2, rng=rng, strategy=covering_array.OPERATOR_STRATEGY
            )

        row = self._rows[self._idx % len(self._rows)]
        self._idx += 1
        out = field_spec.apply_row(data, fields, row)
        if max_len:
            out = out[:max_len]
        return out if out != data else None


def _pick(rng: Any, lo: int, hi: int) -> int:
    return rng.randint(lo, hi) if rng is not None and hi > lo else lo


# --------------------------------------------------------------------------
# ISO-BMFF box walker


def _boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """``(type, payload_start, box_end)`` for each box in ``[start, end)``.

    A size running past *end* is clamped (fuzzed files still walk); a size
    smaller than its own header stops the walk.
    """
    pos = start
    for _ in range(_MAX_NODES):
        if pos + 8 > end:
            return
        size = int.from_bytes(data[pos : pos + 4], "big")
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = int.from_bytes(data[pos + 8 : pos + 16], "big")
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            return
        stop = min(pos + size, end)
        yield data[pos + 4 : pos + 8], pos + header, stop
        pos += size


def find_box(
    data: bytes, path: Sequence[bytes], start: int = 0, end: int | None = None
) -> tuple[int, int] | None:
    """``(payload_start, box_end)`` of the first box at *path*, backtracking over siblings."""
    end = len(data) if end is None else end
    for kind, payload, stop in _boxes(data, start, end):
        if kind != path[0]:
            continue
        if len(path) == 1:
            return payload, stop
        found = find_box(data, path[1:], payload, stop)
        if found is not None:
            return found
    return None


# tkhd payload: version/flags, then times, track_id, duration, 8 reserved,
# layer, alt_group, volume, 2 reserved, 36-byte matrix, width, height (16.16).
_TKHD_V0 = {
    "track_id": 12,
    "duration": (20, 4),
    "layer": 32,
    "volume": 36,
    "width": 76,
    "height": 80,
    "end": 84,
}
_TKHD_V1 = {
    "track_id": 20,
    "duration": (28, 8),
    "layer": 44,
    "volume": 48,
    "width": 88,
    "height": 92,
    "end": 96,
}


class IsobmffTkhdCoveringArrayMutator(_WalkedCoveringMutator):
    """``covering_array_isobmff_tkhd``: pairwise sweep of the track header box."""

    name = "covering_array_isobmff_tkhd"
    # Version 1 switches times and duration to 64 bits: a decoder that reads the
    # layout from the wrong version mis-reads every later field.
    values: ClassVar[Values] = (
        ("version", (0, 1, 2, 255)),
        ("flags", (0, 1, 3, 7, _U24)),
        ("track_id", (0, 1, _U32)),
        ("duration", (0, 1, _U32)),
        ("layer", (0, 1, _U16)),
        ("volume", (0, 0x0100, _U16)),
        ("width", (0, 1 << 16, _U32)),
        ("height", (0, 1 << 16, _U32)),
    )

    def layout(self, data: bytes, rng: Any) -> Layout | None:
        found = find_box(data, (b"moov", b"trak", b"tkhd"))
        if found is None:
            return None

        base, stop = found
        if base >= len(data):
            return None
        table = _TKHD_V1 if data[base] == 1 else _TKHD_V0
        if stop - base < table["end"]:
            return None

        dur_off, dur_size = table["duration"]
        return {
            "version": (base, 1, False),
            "flags": (base + 1, 3, False),
            "track_id": (base + table["track_id"], 4, False),
            "duration": (base + dur_off, dur_size, False),
            "layer": (base + table["layer"], 2, False),
            "volume": (base + table["volume"], 2, False),
            "width": (base + table["width"], 4, False),
            "height": (base + table["height"], 4, False),
        }


# --------------------------------------------------------------------------
# RIFF chunk walker


def _riff_chunks(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int, int]]:
    """``(id, header_offset, payload_start, payload_end)`` for chunks in ``[start, end)``."""
    pos = start
    for _ in range(_MAX_NODES):
        if pos + 8 > end:
            return
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        payload = pos + 8
        yield data[pos : pos + 4], pos, payload, min(payload + size, end)
        pos = payload + size + (size & 1)


def riff_find(
    data: bytes, path: Sequence[bytes], start: int = 12, end: int | None = None
) -> tuple[int, int, int] | None:
    """``(header_offset, payload_start, payload_end)`` of the chunk at *path*.

    A path element ``b"LIST:xxxx"`` descends into a ``LIST`` chunk of list type
    ``xxxx``.
    """
    end = len(data) if end is None else end
    head = path[0]
    want, _, list_type = head.partition(b":")
    for cid, header, payload, stop in _riff_chunks(data, start, end):
        if cid != want:
            continue
        if list_type and data[payload : payload + 4] != list_type:
            continue
        if len(path) == 1:
            return header, payload, stop
        found = riff_find(data, path[1:], payload + (4 if list_type else 0), stop)
        if found is not None:
            return found
    return None


def _riff_gate(data: bytes, form: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == form


def _riff_layout(
    data: bytes, form: bytes, chunk_path: Sequence[bytes], need: int
) -> tuple[int, int] | None:
    """``(header_offset, payload_start)`` for a chunk with >= *need* payload bytes."""
    if not _riff_gate(data, form):
        return None
    found = riff_find(data, chunk_path)
    if found is None or found[2] - found[1] < need:
        return None
    return found[0], found[1]


class WavCoveringArrayMutator(_WalkedCoveringMutator):
    """``covering_array_wav``: pairwise sweep of the fmt chunk and the data size."""

    name = "covering_array_wav"
    values: ClassVar[Values] = (
        ("riff_size", (0, 4, _U32)),
        ("fmt_size", (0, 15, 16, 18, _U32)),
        ("audio_format", (0, 1, 3, 0xFFFE, _U16)),
        ("channels", (0, 1, 2, _U16)),
        ("sample_rate", (0, 8000, 44100, _U32)),
        ("byte_rate", (0, 1, _U32)),
        ("block_align", (0, 1, _U16)),
        ("bits", (0, 8, 16, 24, 32, _U16)),
        ("data_size", (0, 1, 0x7FFFFFFF, _U32)),
    )

    def layout(self, data: bytes, rng: Any) -> Layout | None:
        fmt = _riff_layout(data, b"WAVE", (b"fmt ",), 16)
        body = riff_find(data, (b"data",)) if fmt is not None else None
        if fmt is None or body is None:
            return None

        hdr, pay = fmt
        return {
            "riff_size": (4, 4, True),
            "fmt_size": (hdr + 4, 4, True),
            "audio_format": (pay, 2, True),
            "channels": (pay + 2, 2, True),
            "sample_rate": (pay + 4, 4, True),
            "byte_rate": (pay + 8, 4, True),
            "block_align": (pay + 12, 2, True),
            "bits": (pay + 14, 2, True),
            "data_size": (body[0] + 4, 4, True),
        }


# avih: microsec/frame, max bytes/sec, padding, flags, total frames, initial
# frames, streams, suggested buffer, width, height, 4 reserved (56 bytes).
class AviCoveringArrayMutator(_WalkedCoveringMutator):
    """``covering_array_avi``: pairwise sweep of the AVI main header."""

    name = "covering_array_avi"
    values: ClassVar[Values] = (
        ("riff_size", (0, 4, _U32)),
        ("avih_size", (0, 55, 56, _U32)),
        ("microsec", (0, 1, _U32)),
        ("flags", (0, 0x10, 0x100, _U32)),
        ("total_frames", (0, 1, _U32)),
        ("streams", (0, 1, 2, _U32)),
        ("buffer", (0, 1, _U32)),
        ("width", (0, 1, _U32)),
        ("height", (0, 1, _U32)),
    )

    def layout(self, data: bytes, rng: Any) -> Layout | None:
        avih = _riff_layout(data, b"AVI ", (b"LIST:hdrl", b"avih"), 40)
        if avih is None:
            return None

        hdr, pay = avih
        return {
            "riff_size": (4, 4, True),
            "avih_size": (hdr + 4, 4, True),
            "microsec": (pay, 4, True),
            "flags": (pay + 12, 4, True),
            "total_frames": (pay + 16, 4, True),
            "streams": (pay + 24, 4, True),
            "buffer": (pay + 28, 4, True),
            "width": (pay + 32, 4, True),
            "height": (pay + 36, 4, True),
        }


# VP8 frame tag (3 bytes LE): bit0 = not-keyframe, bits1-3 version, bit4 show,
# bits5-23 first-partition size. Width/height are 14 bits plus a 2-bit scale.
class WebpVp8CoveringArrayMutator(_WalkedCoveringMutator):
    """``covering_array_webp_vp8``: pairwise sweep of the lossy frame header."""

    name = "covering_array_webp_vp8"
    values: ClassVar[Values] = (
        ("riff_size", (0, 4, _U32)),
        ("chunk_size", (0, 9, 10, _U32)),
        ("frame_tag", (0x000010, 0x000011, 0x00001E, 0xFFFFF0, _U24)),
        ("start_code", (0x9D012A, 0, 0x9D012B, _U24)),
        ("width", (0, 1, 0x3FFF, 0x4001, _U16)),
        ("height", (0, 1, 0x3FFF, 0x4001, _U16)),
    )

    def layout(self, data: bytes, rng: Any) -> Layout | None:
        vp8 = _riff_layout(data, b"WEBP", (b"VP8 ",), 10)
        if vp8 is None:
            return None

        hdr, pay = vp8
        return {
            "riff_size": (4, 4, True),
            "chunk_size": (hdr + 4, 4, True),
            "frame_tag": (pay, 3, True),
            "start_code": (pay + 3, 3, False),
            "width": (pay + 6, 2, True),
            "height": (pay + 8, 2, True),
        }


# VP8L: 0x2f signature, then a 32-bit LE word: width-1 (14), height-1 (14),
# alpha (1), version (3). The packed word is swept as one field.
class WebpVp8lCoveringArrayMutator(_WalkedCoveringMutator):
    """``covering_array_webp_vp8l``: pairwise sweep of the lossless header."""

    name = "covering_array_webp_vp8l"
    values: ClassVar[Values] = (
        ("riff_size", (0, 4, _U32)),
        ("chunk_size", (0, 4, 5, _U32)),
        ("signature", (0x2F, 0, 0xFF)),
        ("packed", (0, 0x00003FFF, 0x0FFFC000, 0x0FFFFFFF, 0x10000000, 0xE0000000, _U32)),
    )

    def layout(self, data: bytes, rng: Any) -> Layout | None:
        vp8l = _riff_layout(data, b"WEBP", (b"VP8L",), 5)
        if vp8l is None:
            return None

        hdr, pay = vp8l
        return {
            "riff_size": (4, 4, True),
            "chunk_size": (hdr + 4, 4, True),
            "signature": (pay, 1, False),
            "packed": (pay + 1, 4, True),
        }


# --------------------------------------------------------------------------
# ZIP record walkers

_EOCD = b"PK\x05\x06"
_CDFH = b"PK\x01\x02"
_LFH = b"PK\x03\x04"


def _le(data: bytes, off: int, size: int) -> int:
    return int.from_bytes(data[off : off + size], "little")


def zip_eocd_offset(data: bytes) -> int | None:
    """Offset of the last end-of-central-directory record, or None."""
    if data[:2] != b"PK":
        return None
    at = data.rfind(_EOCD, max(0, len(data) - _EOCD_SEARCH))
    return at if at >= 0 and at + _EOCD_LEN <= len(data) else None


def zip_cd_entries(data: bytes) -> list[int]:
    """Offsets of central-directory headers reached from the EOCD."""
    eocd = zip_eocd_offset(data)
    if eocd is None:
        return []

    pos = _le(data, eocd + 16, 4)
    out: list[int] = []
    while (
        len(out) < _MAX_ZIP_ENTRIES
        and data[pos : pos + 4] == _CDFH
        and pos + _CDFH_LEN <= len(data)
    ):
        out.append(pos)
        pos += _CDFH_LEN + _le(data, pos + 28, 2) + _le(data, pos + 30, 2) + _le(data, pos + 32, 2)
    return out


def zip_local_offsets(data: bytes) -> list[int]:
    """Local-header offsets named by the central directory (valid ones only)."""
    offs = (_le(data, cd + 42, 4) for cd in zip_cd_entries(data))
    return [o for o in offs if data[o : o + 4] == _LFH and o + _LFH_LEN <= len(data)]


class ZipEocdCoveringArrayMutator(_WalkedCoveringMutator):
    """``covering_array_zip_eocd``: pairwise sweep of the end-of-central-directory record."""

    name = "covering_array_zip_eocd"
    values: ClassVar[Values] = (
        ("disk", (0, 1, _U16)),
        ("cd_disk", (0, 1, _U16)),
        ("entries_disk", (0, 1, _U16)),
        ("entries_total", (0, 1, 2, _U16)),
        ("cd_size", (0, 1, _U32)),
        ("cd_offset", (0, 1, _U32)),
        ("comment_len", (0, 1, _U16)),
    )

    def layout(self, data: bytes, rng: Any) -> Layout | None:
        at = zip_eocd_offset(data)
        if at is None:
            return None

        return {
            "disk": (at + 4, 2, True),
            "cd_disk": (at + 6, 2, True),
            "entries_disk": (at + 8, 2, True),
            "entries_total": (at + 10, 2, True),
            "cd_size": (at + 12, 4, True),
            "cd_offset": (at + 16, 4, True),
            "comment_len": (at + 20, 2, True),
        }


class ZipCdCoveringArrayMutator(_WalkedCoveringMutator):
    """``covering_array_zip_cd``: pairwise sweep of one central-directory header."""

    name = "covering_array_zip_cd"
    values: ClassVar[Values] = (
        ("version_made", (0, 20, 63, _U16)),
        ("version_needed", (0, 20, 45, _U16)),
        ("flags", (0, 1, 8, 0x0800, _U16)),
        ("method", (0, 8, 99, _U16)),
        ("comp_size", (0, 1, _U32)),
        ("file_size", (0, 1, _U32)),
        ("name_len", (0, 1, _U16)),
        ("extra_len", (0, 1, _U16)),
        ("comment_len", (0, 1, _U16)),
        ("disk_start", (0, 1, _U16)),
        ("int_attr", (0, 1, _U16)),
        ("ext_attr", (0, _U32)),
        ("local_offset", (0, 1, _U32)),
    )

    def layout(self, data: bytes, rng: Any) -> Layout | None:
        entries = zip_cd_entries(data)
        if not entries:
            return None

        at = entries[_pick(rng, 0, len(entries) - 1)]
        return {
            "version_made": (at + 4, 2, True),
            "version_needed": (at + 6, 2, True),
            "flags": (at + 8, 2, True),
            "method": (at + 10, 2, True),
            "comp_size": (at + 20, 4, True),
            "file_size": (at + 24, 4, True),
            "name_len": (at + 28, 2, True),
            "extra_len": (at + 30, 2, True),
            "comment_len": (at + 32, 2, True),
            "disk_start": (at + 34, 2, True),
            "int_attr": (at + 36, 2, True),
            "ext_attr": (at + 38, 4, True),
            "local_offset": (at + 42, 4, True),
        }


class ZipEntryCoveringArrayMutator(_WalkedCoveringMutator):
    """``covering_array_zip_entry``: local header of entry 1.. (entry 0 is ``covering_array_zip``)."""

    name = "covering_array_zip_entry"
    values: ClassVar[Values] = tuple((f.name, f.values) for f in ZIP_FIELDS)

    def layout(self, data: bytes, rng: Any) -> Layout | None:
        locals_ = zip_local_offsets(data)
        if len(set(locals_)) < 2:
            return None

        at = sorted(set(locals_))[_pick(rng, 1, len(set(locals_)) - 1)]
        return {f.name: (at + f.offset, f.size, f.little) for f in ZIP_FIELDS}


_MUTATORS = (
    IsobmffTkhdCoveringArrayMutator,
    WavCoveringArrayMutator,
    AviCoveringArrayMutator,
    WebpVp8CoveringArrayMutator,
    WebpVp8lCoveringArrayMutator,
    ZipEocdCoveringArrayMutator,
    ZipCdCoveringArrayMutator,
    ZipEntryCoveringArrayMutator,
)


def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY

    for cls in _MUTATORS:
        m = cls()
        if m.name not in REGISTRY.names():
            REGISTRY.register_mutator(m)


_register()
