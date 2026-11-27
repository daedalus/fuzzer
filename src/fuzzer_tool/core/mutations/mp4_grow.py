"""Grow a missing required MP4 box, with its own required children.

The other MP4 ops edit or transplant boxes that already exist. From an empty
corpus nothing below ``mdia`` ever appears, so qtdemux's sample-table code
stays dark (measured: no ``stbl`` in 2,634 fuzzed GStreamer inputs). One draw
here fills the chain:

    ftyp                 ftyp moov
                                mvhd
              ->                trak  tkhd
                                      mdia  mdhd hdlr
                                            minf  vmhd dinf(dref)
                                                  stbl stsd stts stsc stsz stco

Children go first in their container, so a lying later sibling cannot hide
them; a missing ``moov`` goes after the first top-level box. Ancestor sizes
and chunk offsets are fixed by ``mp4_path.splice``.
"""

from __future__ import annotations

import struct

from fuzzer_tool.core.mutations.mp4_path import BOX_HDR, BoxHit, find_boxes, splice

ROOT = b""  # pseudo-container: the file's top level
U16_ONE = 0x0001
FIXED_1_0 = 0x00010000  # 16.16 fixed point 1.0 (rate, matrix)
FIXED_8_8_ONE = 0x0100  # 8.8 fixed point 1.0 (volume)
MATRIX_W = 0x40000000  # 2.30 fixed point 1.0, matrix[8]
DPI_72 = 0x00480000  # 72 dpi, 16.16
DEPTH_24 = 0x0018
LANG_UND = 0x55C4  # packed ISO-639 "und"
TRACK_ENABLED = 0x000003  # tkhd flags: enabled | in movie
URL_SELF = 0x000001  # 'url ' flags: media in this file
TIMESCALE = 1000
DIM = 16  # template frame size, pixels

# Required children per container, in spec order (ISO/IEC 14496-12).
REQUIRED: dict[bytes, tuple[bytes, ...]] = {
    ROOT: (b"moov",),
    b"moov": (b"mvhd", b"trak"),
    b"trak": (b"tkhd", b"mdia"),
    b"mdia": (b"mdhd", b"hdlr", b"minf"),
    b"minf": (b"vmhd", b"dinf", b"stbl"),
    b"dinf": (b"dref",),
    b"stbl": (b"stsd", b"stts", b"stsc", b"stsz", b"stco"),
}
_GROWABLE = frozenset(REQUIRED) - {ROOT}


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", BOX_HDR + len(payload)) + typ + payload


def _full(version_flags: int, body: bytes) -> bytes:
    """FullBox payload: version/flags word, then fields."""
    return struct.pack(">I", version_flags) + body


_MATRIX = struct.pack(">9I", FIXED_1_0, 0, 0, 0, FIXED_1_0, 0, 0, 0, MATRIX_W)

# Minimal valid leaf payloads: one video track, one 1-byte sample at offset 0.
_LEAVES: dict[bytes, bytes] = {
    b"mvhd": _full(
        0,
        struct.pack(">IIIIIH10x", 0, 0, TIMESCALE, TIMESCALE, FIXED_1_0, FIXED_8_8_ONE)
        + _MATRIX
        + bytes(24)
        + struct.pack(">I", 2),
    ),
    b"tkhd": _full(
        TRACK_ENABLED,
        struct.pack(">IIIII8xHHH2x", 0, 0, 1, 0, TIMESCALE, 0, 0, 0)
        + _MATRIX
        + struct.pack(">II", DIM << 16, DIM << 16),
    ),
    b"mdhd": _full(0, struct.pack(">IIIIHH", 0, 0, TIMESCALE, TIMESCALE, LANG_UND, 0)),
    b"hdlr": _full(0, bytes(4) + b"vide" + bytes(12) + b"\x00"),
    b"vmhd": _full(1, bytes(8)),
    b"dref": _full(0, struct.pack(">I", 1) + _box(b"url ", _full(URL_SELF, b""))),
    b"stsd": _full(
        0,
        struct.pack(">I", 1)
        + _box(
            b"mp4v",
            bytes(6)
            + struct.pack(">H", U16_ONE)
            + bytes(16)
            + struct.pack(">HHIIIH", DIM, DIM, DPI_72, DPI_72, 0, U16_ONE)
            + bytes(32)
            + struct.pack(">Hh", DEPTH_24, -1),
        ),
    ),
    b"stts": _full(0, struct.pack(">III", 1, 1, 1)),
    b"stsc": _full(0, struct.pack(">IIII", 1, 1, 1, 1)),
    b"stsz": _full(0, struct.pack(">III", 0, 1, 1)),
    b"stco": _full(0, struct.pack(">II", 1, 0)),
}


def _template(typ: bytes) -> bytes:
    """*typ* as a full box: a leaf, or a container holding its required children."""
    if typ in _LEAVES:
        return _box(typ, _LEAVES[typ])

    return _box(typ, b"".join(_template(c) for c in REQUIRED[typ]))


_TEMPLATES = {t: _template(t) for c in REQUIRED.values() for t in c}


def _child_types(data: bytes, pos: int, end: int) -> tuple[set[bytes], int]:
    """Types of the boxes tiling [pos, end) up to the first bad size; end of the first."""
    types: set[bytes] = set()
    first_end = -1
    while pos + BOX_HDR <= end:
        size = struct.unpack_from(">I", data, pos)[0]
        if size < BOX_HDR or pos + size > end:
            break

        types.add(data[pos + 4 : pos + 8])
        if first_end < 0:
            first_end = pos + size
        pos += size
    return types, first_end


def _candidates(data: bytes) -> list[tuple[BoxHit | None, bytes]]:
    """(container, missing child) pairs in file order; None is the top level."""
    top, _ = _child_types(data, 0, len(data))
    out: list[tuple[BoxHit | None, bytes]] = [(None, t) for t in REQUIRED[ROOT] if t not in top]

    for hit in find_boxes(data, _GROWABLE):
        typ = data[hit.start + 4 : hit.start + 8]
        have, _ = _child_types(data, hit.start + BOX_HDR, hit.start + hit.size)
        out += [(hit, t) for t in REQUIRED[typ] if t not in have]
    return out


def _grow_root(data: bytes, child: bytes) -> bytes | None:
    """Insert *child* after the first top-level box; None if there is none."""
    _, first_end = _child_types(data, 0, len(data))
    if first_end < 0:
        return None

    return data[:first_end] + child + data[first_end:]


def grow_mp4(data: bytes, max_len: int, rng) -> bytes | None:
    """Add one missing required box (with its subtree); None if nothing is missing.

    One ``rng.choice`` picks the (container, child) pair.
    """
    cands = _candidates(data)
    if not cands:
        return None

    hit, typ = rng.choice(cands)
    child = _TEMPLATES[typ]
    if hit is None:
        out = _grow_root(data, child)
    else:
        payload = data[hit.start + BOX_HDR : hit.start + hit.size]
        out = splice(data, hit, child + payload)

    if out is None or len(out) > max_len:
        return None
    return out
