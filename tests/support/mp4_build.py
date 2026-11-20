"""Minimal MP4 builders for config-box mutator tests (avcC / hvcC / esds)."""

from __future__ import annotations

import struct
from enum import Enum

VISUAL_ENTRY_LEN = 78  # VisualSampleEntry fields after the box header
AUDIO_ENTRY_LEN = 28  # AudioSampleEntry v0


def box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + typ + payload


def descr(tag: int, payload: bytes, width: int = 4) -> bytes:
    """MPEG-4 descriptor with an expandable length padded to *width* bytes."""
    n = len(payload)
    groups = [(n >> (7 * k)) & 0x7F for k in range(width - 1, -1, -1)]
    length = bytes([g | 0x80 for g in groups[:-1]] + [groups[-1]])
    return bytes([tag]) + length + payload


def avcc(sps: bytes, pps: bytes) -> bytes:
    """avcC record, 4-byte NAL lengths, one SPS and one PPS (with NAL headers)."""
    rec = bytes([1, sps[1], sps[2], sps[3], 0xFF, 0xE1])
    rec += struct.pack(">H", len(sps)) + sps + b"\x01" + struct.pack(">H", len(pps)) + pps
    return box(b"avcC", rec)


def esds(asc: bytes) -> bytes:
    dsi = descr(5, asc)
    dc = descr(4, bytes([0x40, 0x15]) + bytes(3) + struct.pack(">II", 128000, 96000) + dsi)
    es = descr(3, b"\x00\x01\x00" + dc + descr(6, b"\x02"))
    return box(b"esds", bytes(4) + es)


def stsd(entry_type: bytes, child: bytes) -> bytes:
    fixed = VISUAL_ENTRY_LEN if entry_type == b"avc1" else AUDIO_ENTRY_LEN
    entry = box(entry_type, bytes(fixed) + child)
    return box(b"stsd", bytes(4) + struct.pack(">I", 1) + entry)


def stco(offsets: list[int]) -> bytes:
    return box(
        b"stco",
        bytes(4)
        + struct.pack(">I", len(offsets))
        + b"".join(struct.pack(">I", o) for o in offsets),
    )


class Layout(Enum):
    """Where moov sits relative to mdat."""

    FASTSTART = 0  # moov first: chunk offsets point past it
    MDAT_FIRST = 1


def mp4(cfg_entry: bytes, mdat: bytes = b"\xaa" * 16, layout: Layout = Layout.FASTSTART) -> bytes:
    """ftyp + moov(trak..stbl(stsd, stco)) + mdat; stco points at mdat's payload."""
    ftyp = box(b"ftyp", b"isom" + bytes(4) + b"isom")

    def build(offset: int) -> bytes:
        stbl = box(b"stbl", cfg_entry + stco([offset]))
        return box(b"moov", box(b"trak", box(b"mdia", box(b"minf", stbl))))

    if layout is Layout.MDAT_FIRST:
        mdat_box = box(b"mdat", mdat)
        return ftyp + mdat_box + build(len(ftyp) + 8)
    moov_len = len(build(0))
    return ftyp + build(len(ftyp) + moov_len + 8) + box(b"mdat", mdat)
