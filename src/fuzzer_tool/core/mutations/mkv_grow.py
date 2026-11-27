"""Grow a missing Matroska element, with its own required children.

``webm_chunk_mutate`` edits elements that exist. An empty-corpus GStreamer run
built Segment/Tracks/Cluster yet never CodecPrivate, Audio, BlockGroup, Cues or
SeekHead (0 of 327 MKV inputs). One draw here adds a whole element:

    EBML                 EBML
                         Segment  SeekHead(Seek)  Info  Tracks(TrackEntry ...)
              ->                  Cluster(Timecode SimpleBlock BlockGroup)  Cues

Track kind (V_MPEG4/ISO/AVC or A_AAC, CodecPrivate and Video/Audio to match)
follows the TrackEntry's TrackType when present; otherwise a second draw.
Cluster and Cues go last in the Segment (after Tracks, where demuxers expect
them); other children go first. Sizes are re-encoded by ``serialize_webm``;
bytes after the parsed tree are kept.
"""

from __future__ import annotations

import struct
from enum import Enum, auto
from typing import cast

from fuzzer_tool.core.mutations.webm import (
    Element,
    _encode_size_vint,
    _parse_element,
    parse_webm,
    serialize_webm,
)
from fuzzer_tool.core.rand_pool import RandPool

EBML = 0x1A45DFA3
SEGMENT = 0x18538067
SEEK_HEAD = 0x114D9B74
SEEK = 0x4DBB
INFO = 0x1549A966
TRACKS = 0x1654AE6B
TRACK_ENTRY = 0xAE
TRACK_TYPE = 0x83
CODEC_ID = 0x86
CODEC_PRIVATE = 0x63A2
VIDEO = 0xE0
AUDIO = 0xE1
MEDIA = -1  # pseudo-id: Video or Audio, by kind
CLUSTER = 0x1F43B675
BLOCK_GROUP = 0xA0
CUES = 0x1C53BB6B
CUE_POINT = 0xBB
CUE_POSITIONS = 0xB7
ROOT = 0  # pseudo-container: the file's top level
TIMECODE_SCALE_NS = 1_000_000
DIM = 16
SAMPLE_RATE = 48000.0
TRACK_VIDEO, TRACK_AUDIO = 1, 2
KEYFRAME = 0x80
# Block body: track 1 (vint 0x81), timecode 0 (i16), flags, one payload byte.
_BLOCK = bytes([0x81, 0x00, 0x00, KEYFRAME, 0x00])
# Decoder configs: libx264 16x16 avcC and AAC-LC 48 kHz mono ASC (ffmpeg output).
_AVCC = bytes.fromhex(
    "01f4000affe1001767f4000a919b2bd80880000003008000000302078912cb01000668ebe3c44844fff8f800"
)
_ASC = b"\x11\x88"


class MkvKind(Enum):
    """Track family grown into a TrackEntry."""

    VIDEO = auto()
    AUDIO = auto()


_KIND_OF_TYPE = {TRACK_VIDEO: MkvKind.VIDEO, TRACK_AUDIO: MkvKind.AUDIO}

# Required children per container (Matroska spec order); optional ones offered too.
REQUIRED: dict[int, tuple[int, ...]] = {
    ROOT: (SEGMENT,),
    EBML: (0x4286, 0x42F7, 0x42F2, 0x42F3, 0x4282, 0x4287, 0x4285),
    SEGMENT: (INFO, TRACKS, CLUSTER),
    INFO: (0x2AD7B1, 0x4D80, 0x5741),
    TRACKS: (TRACK_ENTRY,),
    TRACK_ENTRY: (0xD7, 0x73C5, TRACK_TYPE, CODEC_ID, MEDIA),
    CLUSTER: (0xE7, 0xA3),
    BLOCK_GROUP: (0xA1,),
    SEEK_HEAD: (SEEK,),
    SEEK: (0x53AB, 0x53AC),
    CUES: (CUE_POINT,),
    CUE_POINT: (0xB3, CUE_POSITIONS),
    CUE_POSITIONS: (0xF7, 0xF1),
}
OPTIONAL: dict[int, tuple[int, ...]] = {
    SEGMENT: (SEEK_HEAD, CUES),
    INFO: (0x4489,),
    TRACK_ENTRY: (CODEC_PRIVATE,),
    CLUSTER: (BLOCK_GROUP,),
}
_APPEND = frozenset({CLUSTER, CUES})
_ALIASES = {MEDIA: frozenset({VIDEO, AUDIO})}


def _uint(v: int) -> bytes:
    return v.to_bytes(max(1, (v.bit_length() + 7) // 8), "big")


_COMMON_LEAVES: dict[int, bytes] = {
    0x4286: _uint(1),  # EBMLVersion
    0x42F7: _uint(1),  # EBMLReadVersion
    0x42F2: _uint(4),  # EBMLMaxIDLength
    0x42F3: _uint(8),  # EBMLMaxSizeLength
    0x4282: b"matroska",  # DocType
    0x4287: _uint(4),  # DocTypeVersion
    0x4285: _uint(2),  # DocTypeReadVersion
    0x2AD7B1: _uint(TIMECODE_SCALE_NS),
    0x4D80: b"fuzz",  # MuxingApp
    0x5741: b"fuzz",  # WritingApp
    0x4489: struct.pack(">f", 1000.0),  # Duration
    0xD7: _uint(1),  # TrackNumber
    0x73C5: _uint(1),  # TrackUID
    0xE7: _uint(0),  # Cluster Timecode
    0xA3: _BLOCK,  # SimpleBlock
    0xA1: _BLOCK,  # Block
    0x53AB: INFO.to_bytes(4, "big"),  # SeekID
    0x53AC: _uint(0),  # SeekPosition
    0xB3: _uint(0),  # CueTime
    0xF7: _uint(1),  # CueTrack
    0xF1: _uint(0),  # CueClusterPosition
}


def _id_bytes(elem_id: int) -> bytes:
    return elem_id.to_bytes((elem_id.bit_length() + 7) // 8, "big")


def _el(elem_id: int, payload: bytes) -> bytes:
    return _id_bytes(elem_id) + _encode_size_vint(len(payload)) + payload


def _kind_leaf(elem_id: int, kind: MkvKind) -> bytes | None:
    """Full element for a kind-dependent leaf; None otherwise."""
    video = kind is MkvKind.VIDEO
    if elem_id == TRACK_TYPE:
        return _el(TRACK_TYPE, _uint(TRACK_VIDEO if video else TRACK_AUDIO))
    if elem_id == CODEC_ID:
        return _el(CODEC_ID, b"V_MPEG4/ISO/AVC" if video else b"A_AAC")
    if elem_id == CODEC_PRIVATE:
        return _el(CODEC_PRIVATE, _AVCC if video else _ASC)
    if elem_id == MEDIA and video:
        return _el(VIDEO, _el(0xB0, _uint(DIM)) + _el(0xBA, _uint(DIM)))
    if elem_id == MEDIA:
        return _el(AUDIO, _el(0xB5, struct.pack(">f", SAMPLE_RATE)) + _el(0x9F, _uint(1)))
    return None


def _template(elem_id: int, kind: MkvKind) -> bytes:
    """*elem_id* as a full element: a leaf, or a master holding its required children."""
    leaf = _kind_leaf(elem_id, kind)
    if leaf is not None:
        return leaf
    if elem_id in _COMMON_LEAVES:
        return _el(elem_id, _COMMON_LEAVES[elem_id])

    return _el(elem_id, b"".join(_template(c, kind) for c in REQUIRED[elem_id]))


_ALL = {t for d in (REQUIRED, OPTIONAL) for c in d.values() for t in c}
_TEMPLATES = {(t, k): _template(t, k) for t in _ALL for k in MkvKind}
_KIND_DEP = frozenset(t for t in _ALL if len({_TEMPLATES[(t, k)] for k in MkvKind}) > 1)


def _missing(container: int, have: set[int]) -> list[int]:
    wanted = REQUIRED.get(container, ()) + OPTIONAL.get(container, ())
    return [t for t in wanted if not (_ALIASES.get(t, frozenset({t})) & have)]


def _uint_of(el: Element) -> int:
    return int.from_bytes(el.data, "big") if el.data else 0


def _entry_kind(entry: Element | None) -> MkvKind | None:
    """Kind from a TrackEntry's TrackType child; None when absent or unknown."""
    if entry is None:
        return None
    for c in entry.children:
        if c.elem_id == TRACK_TYPE:
            return _KIND_OF_TYPE.get(_uint_of(c))
    return None


Cand = tuple[Element | None, int, MkvKind | None]


def _walk(el: Element, entry: Element | None, out: list[Cand]) -> None:
    """Collect missing children under *el*; unparsed masters (raw data) are skipped."""
    if el.elem_id not in REQUIRED or (el.data and not el.children):
        return

    entry = el if el.elem_id == TRACK_ENTRY else entry
    have = {c.elem_id for c in el.children}
    out += [(el, t, _entry_kind(entry)) for t in _missing(el.elem_id, have)]
    for c in el.children:
        _walk(c, entry, out)


def _candidates(data: bytes) -> list[tuple[Element | None, int]]:
    """(container, missing child) pairs in tree order; None is the top level."""
    return [(el, t) for el, t, _ in _cands(data)[0]]


def _cands(data: bytes) -> tuple[list[Cand], list[Element] | None]:
    els = parse_webm(data)
    if els is None:
        head, _ = _parse_element(data, 0)
        ok = head is not None and head.elem_id == EBML
        return ([(None, SEGMENT, None)] if ok else []), None

    out: list[Cand] = []
    for el in els:
        _walk(el, None, out)
    return out, els


def _grow_root(data: bytes, child: bytes) -> bytes | None:
    head, end = _parse_element(data, 0)
    if head is None:
        return None
    return data[:end] + child + data[end:]


def grow_mkv(data: bytes, max_len: int, rng: RandPool) -> bytes | None:
    """Add one missing element (with its subtree); None if nothing is missing.

    ``rng.choice`` picks the (container, child) pair, then the track kind
    when the child depends on it and no TrackType decides it.
    """
    cands, els = _cands(data)
    if not cands:
        return None

    parent, elem_id, kind = cast("Cand", rng.choice(cands))
    if kind is None and elem_id in _KIND_DEP:
        kind = cast("MkvKind", rng.choice(tuple(MkvKind)))
    child = _TEMPLATES[(elem_id, kind or MkvKind.VIDEO)]

    if parent is None or els is None:
        out = _grow_root(data, child)
    else:
        consumed = len(serialize_webm(els))
        node, _ = _parse_element(child, 0)
        if node is None:
            return None
        if elem_id in _APPEND:
            parent.children.append(node)
        else:
            parent.children.insert(0, node)
        out = serialize_webm(els) + data[consumed:]

    if out is None or len(out) > max_len:
        return None
    return out
