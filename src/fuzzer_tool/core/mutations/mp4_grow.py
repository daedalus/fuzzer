"""Grow a missing MP4 box, with its own required children.

The other MP4 ops edit or transplant boxes that already exist. From an empty
corpus nothing below ``mdia`` ever appears, so qtdemux's sample-table code
stays dark (measured: no ``stbl`` in 2,634 fuzzed GStreamer inputs). One draw
here fills the chain:

    ftyp                 ftyp moov
                                mvhd
              ->                trak  tkhd
                                      mdia  mdhd hdlr
                                            minf  vmhd|smhd dinf(dref)
                                                  stbl stsd stts stsc stsz stco
                                mvex  trex                    (optional)
                         moof  mfhd traf(tfhd tfdt trun)      (optional, at end)

Track kind (mp4v, avc1+avcC, mp4a+esds) follows the enclosing trak's hdlr when
it has one, so grown parts agree with the track; otherwise a second draw picks
it. Children go first in their container, so a lying later sibling cannot hide
them; ``moov`` goes after the first top-level box. Ancestor sizes and chunk
offsets are fixed by ``mp4_path.splice``.
"""

from __future__ import annotations

import struct
from enum import Enum, auto
from typing import cast

from fuzzer_tool.core.mutations.mp4_path import BOX_HDR, BoxHit, find_boxes, splice
from fuzzer_tool.core.rand_pool import RandPool

ROOT = b""  # pseudo-container: the file's top level
MEDIA_HDR = b"*mhd"  # pseudo-type: vmhd or smhd, by track kind
U16_ONE = 0x0001
FIXED_1_0 = 0x00010000  # 16.16 fixed point 1.0 (rate, matrix)
FIXED_8_8_ONE = 0x0100  # 8.8 fixed point 1.0 (volume)
MATRIX_W = 0x40000000  # 2.30 fixed point 1.0, matrix[8]
DPI_72 = 0x00480000  # 72 dpi, 16.16
DEPTH_24 = 0x0018
LANG_UND = 0x55C4  # packed ISO-639 "und"
TRACK_ENABLED = 0x000003  # tkhd flags: enabled | in movie
URL_SELF = 0x000001  # 'url ' flags: media in this file
TFHD_BASE_IS_MOOF = 0x020000
TRUN_SAMPLE_SIZE = 0x000200
TIMESCALE = 1000
DIM = 16  # template frame size, pixels
SAMPLE_RATE = 48000
HDLR_TYPE_OFF = BOX_HDR + 8  # version/flags + pre_defined


class TrackKind(Enum):
    """Sample entry family grown into stsd."""

    MP4V = auto()
    AVC1 = auto()
    MP4A = auto()


_HANDLER = {TrackKind.MP4V: b"vide", TrackKind.AVC1: b"vide", TrackKind.MP4A: b"soun"}


# Kinds a handler_type allows; an unknown or absent handler allows all.
KINDS: dict[bytes, tuple[TrackKind, ...]] = {
    b"vide": (TrackKind.MP4V, TrackKind.AVC1),
    b"soun": (TrackKind.MP4A,),
}

# Required children per container, in spec order (ISO/IEC 14496-12).
REQUIRED: dict[bytes, tuple[bytes, ...]] = {
    ROOT: (b"moov",),
    b"moov": (b"mvhd", b"trak"),
    b"trak": (b"tkhd", b"mdia"),
    b"mdia": (b"mdhd", b"hdlr", b"minf"),
    b"minf": (MEDIA_HDR, b"dinf", b"stbl"),
    b"dinf": (b"dref",),
    b"stbl": (b"stsd", b"stts", b"stsc", b"stsz", b"stco"),
    b"mvex": (b"trex",),
    b"moof": (b"mfhd", b"traf"),
    b"traf": (b"tfhd", b"tfdt", b"trun"),
}
# Offered too, though a valid file may lack them: fragmented-MP4 structure.
OPTIONAL: dict[bytes, tuple[bytes, ...]] = {ROOT: (b"moof",), b"moov": (b"mvex",)}
_AT_END = frozenset({b"moof"})  # top-level children appended, not after ftyp
# Top-level boxes holding offsets an insert before them would invalidate
# (tfhd base_data_offset, sidx first_offset): no moov is inserted ahead of them.
_OFFSET_ANCHORS = frozenset({b"moof", b"sidx"})
_ALIASES = {MEDIA_HDR: frozenset({b"vmhd", b"smhd", b"hmhd", b"nmhd", b"sthd"})}
_GROWABLE = frozenset(REQUIRED) - {ROOT}


def _box(typ: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", BOX_HDR + len(payload)) + typ + payload


def _full(version_flags: int, body: bytes) -> bytes:
    """FullBox payload: version/flags word, then fields."""
    return struct.pack(">I", version_flags) + body


_MATRIX = struct.pack(">9I", FIXED_1_0, 0, 0, 0, FIXED_1_0, 0, 0, 0, MATRIX_W)
_VISUAL_ENTRY = (
    bytes(6)
    + struct.pack(">H", U16_ONE)
    + bytes(16)
    + struct.pack(">HHIIIH", DIM, DIM, DPI_72, DPI_72, 0, U16_ONE)
    + bytes(32)
    + struct.pack(">Hh", DEPTH_24, -1)
)
_AUDIO_ENTRY = bytes(6) + struct.pack(">H8xHHHHI", U16_ONE, 1, 16, 0, 0, SAMPLE_RATE << 16)
# Decoder configs lifted from ffmpeg output (libx264 16x16, AAC 48 kHz mono).
_AVCC = bytes.fromhex(
    "01f4000affe1001767f4000a919b2bd80880000003008000000302078912cb01000668ebe3c44844fff8f800"
)
_ESDS = bytes.fromhex(
    "0000000003808080250001000480808017401500000000013594000135940580808005118856e500068080800102"
)

_STSD_ENTRY = {
    TrackKind.MP4V: _box(b"mp4v", _VISUAL_ENTRY),
    TrackKind.AVC1: _box(b"avc1", _VISUAL_ENTRY + _box(b"avcC", _AVCC)),
    TrackKind.MP4A: _box(b"mp4a", _AUDIO_ENTRY + _box(b"esds", _ESDS)),
}

# Minimal valid leaves, kind-independent: one 1-byte sample at offset 0.
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
    b"dref": _full(0, struct.pack(">I", 1) + _box(b"url ", _full(URL_SELF, b""))),
    b"stts": _full(0, struct.pack(">III", 1, 1, 1)),
    b"stsc": _full(0, struct.pack(">IIII", 1, 1, 1, 1)),
    b"stsz": _full(0, struct.pack(">III", 0, 1, 1)),
    b"stco": _full(0, struct.pack(">II", 1, 0)),
    b"trex": _full(0, struct.pack(">IIIII", 1, 1, 0, 0, 0)),
    b"mfhd": _full(0, struct.pack(">I", 1)),
    b"tfhd": _full(TFHD_BASE_IS_MOOF, struct.pack(">I", 1)),
    b"tfdt": _full(0, struct.pack(">I", 0)),
    b"trun": _full(TRUN_SAMPLE_SIZE, struct.pack(">II", 1, 1)),
}


def _kind_leaf(typ: bytes, kind: TrackKind) -> bytes | None:
    """Full box for a kind-dependent leaf (hdlr, media header, stsd); None otherwise."""
    if typ == b"hdlr":
        return _box(b"hdlr", _full(0, bytes(4) + _HANDLER[kind] + bytes(12) + b"\x00"))
    if typ == MEDIA_HDR:
        if kind is TrackKind.MP4A:
            return _box(b"smhd", _full(0, bytes(4)))
        return _box(b"vmhd", _full(1, bytes(8)))
    if typ == b"stsd":
        return _box(b"stsd", _full(0, struct.pack(">I", 1) + _STSD_ENTRY[kind]))
    return None


def _template(typ: bytes, kind: TrackKind) -> bytes:
    """*typ* as a full box: a leaf, or a container holding its required children."""
    leaf = _kind_leaf(typ, kind)
    if leaf is not None:
        return leaf
    if typ in _LEAVES:
        return _box(typ, _LEAVES[typ])

    return _box(typ, b"".join(_template(c, kind) for c in REQUIRED[typ]))


_ALL_CHILDREN = {t for d in (REQUIRED, OPTIONAL) for c in d.values() for t in c}
_TEMPLATES = {(t, k): _template(t, k) for t in _ALL_CHILDREN for k in TrackKind}
# Children whose template differs by kind need a kind; the rest never draw one.
_KIND_DEP = frozenset(t for t in _ALL_CHILDREN if len({_TEMPLATES[(t, k)] for k in TrackKind}) > 1)


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


def _missing(container: bytes, have: set[bytes]) -> list[bytes]:
    """Required then optional children of *container* not in *have* (aliases count)."""
    wanted = REQUIRED.get(container, ()) + OPTIONAL.get(container, ())
    return [t for t in wanted if not (_ALIASES.get(t, frozenset({t})) & have)]


def _candidates(data: bytes) -> list[tuple[BoxHit | None, bytes]]:
    """(container, missing child) pairs in file order; None is the top level."""
    top, _ = _child_types(data, 0, len(data))
    out: list[tuple[BoxHit | None, bytes]] = [(None, t) for t in _missing(ROOT, top)]

    for hit in find_boxes(data, _GROWABLE):
        typ = data[hit.start + 4 : hit.start + 8]
        have, _ = _child_types(data, hit.start + BOX_HDR, hit.start + hit.size)
        out += [(hit, t) for t in _missing(typ, have)]
    return out


def _allowed_kinds(data: bytes, hit: BoxHit | None) -> tuple[TrackKind, ...]:
    """Kinds the enclosing trak's hdlr allows; every kind outside a trak or without one."""
    if hit is None:
        return tuple(TrackKind)

    chain = (*hit.ancestors, hit.start)
    traks = [p for p in chain if data[p + 4 : p + 8] == b"trak"]
    if not traks:
        return tuple(TrackKind)

    trak = traks[-1]
    for h in find_boxes(data, frozenset({b"hdlr"})):
        if trak in h.ancestors and h.start + HDLR_TYPE_OFF + 4 <= len(data):
            handler = data[h.start + HDLR_TYPE_OFF : h.start + HDLR_TYPE_OFF + 4]
            return KINDS.get(handler, tuple(TrackKind))
    return tuple(TrackKind)


def _grow_root(data: bytes, typ: bytes, child: bytes) -> bytes | None:
    """Append fragments at the end; insert others after the first top-level box."""
    if typ in _AT_END:
        return data + child

    top, first_end = _child_types(data, 0, len(data))
    if first_end < 0 or top & _OFFSET_ANCHORS:
        return None
    return data[:first_end] + child + data[first_end:]


def grow_mp4(data: bytes, max_len: int, rng: RandPool) -> bytes | None:
    """Add one missing box (with its subtree); None if nothing is missing.

    ``rng.choice`` picks the (container, child) pair, then the track kind
    when the child depends on it and the trak's hdlr leaves more than one.
    """
    cands = _candidates(data)
    if not cands:
        return None

    hit, typ = cast("tuple[BoxHit | None, bytes]", rng.choice(cands))
    kind = TrackKind.MP4V
    if typ in _KIND_DEP:
        kinds = _allowed_kinds(data, hit)
        kind = kinds[0] if len(kinds) == 1 else cast("TrackKind", rng.choice(kinds))

    child = _TEMPLATES[(typ, kind)]
    if hit is None:
        out = _grow_root(data, typ, child)
    else:
        payload = data[hit.start + BOX_HDR : hit.start + hit.size]
        out = splice(data, hit, child + payload)

    if out is None or len(out) > max_len:
        return None
    return out
