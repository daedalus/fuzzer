"""Grow a missing AVI chunk (or LIST), with its own required children.

``avi_field_mutate`` edits chunks that exist. An empty-corpus GStreamer run
produced one RIFF/AVI header in 2,634 inputs and no parseable chunk, so
avidemux stopped at the header. One draw here adds a whole LIST:

    RIFF AVI             RIFF AVI  LIST hdrl  avih  LIST strl  strh strf
              ->                   LIST movi  00dc
                                   idx1                         (optional)

Stream kind (vids + BITMAPINFOHEADER, or auds + WAVEFORMATEX) follows the
strl's strh fccType when present; otherwise a second draw. movi and idx1 go
last in the RIFF (after hdrl); other children go first. Sizes are
little-endian u32 with even padding; every enclosing size grows by the
inserted length.
"""

from __future__ import annotations

import struct
from enum import Enum, auto
from typing import NamedTuple, cast

from fuzzer_tool.core.mutations.riff_fields import CHUNK_HDR, LIST_TYPE_LEN, RIFF_HDR, sniff_avi
from fuzzer_tool.core.rand_pool import RandPool

ROOT = b"AVI "
U32_MAX = 0xFFFFFFFF
DEPTH_MAX = 8
DIM = 16
FRAME_US = 40000  # 25 fps
SAMPLE_RATE = 48000
AVIF_HASINDEX = 0x10
AVIIF_KEYFRAME = 0x10
BI_SIZE = 40
FCC_MPEG4 = b"FMP4"
WAVE_FORMAT_PCM = 1


class AviKind(Enum):
    """Stream family grown into a strl."""

    VIDEO = auto()
    AUDIO = auto()


_KIND_OF_TYPE = {b"vids": AviKind.VIDEO, b"auds": AviKind.AUDIO}

# Required children per container; LISTs are named by their list type.
REQUIRED: dict[bytes, tuple[bytes, ...]] = {
    ROOT: (b"hdrl", b"movi"),
    b"hdrl": (b"avih", b"strl"),
    b"strl": (b"strh", b"strf"),
    b"movi": (b"00dc",),
}
OPTIONAL: dict[bytes, tuple[bytes, ...]] = {ROOT: (b"idx1",)}
_APPEND = frozenset({b"movi", b"idx1"})
_LISTS = frozenset({b"hdrl", b"strl", b"movi"})


def _chunk(fcc: bytes, body: bytes) -> bytes:
    """Chunk with its pad byte when the body is odd."""
    return fcc + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) & 1 else b"")


_AVIH = struct.pack("<14I", FRAME_US, 0, 0, AVIF_HASINDEX, 1, 0, 1, 0, DIM, DIM, 0, 0, 0, 0)
_COMMON: dict[bytes, bytes] = {
    b"avih": _chunk(b"avih", _AVIH),
    b"00dc": _chunk(b"00dc", b"\x00"),
    b"idx1": _chunk(b"idx1", struct.pack("<4sIII", b"00dc", AVIIF_KEYFRAME, 4, 1)),
}


def _strh(kind: AviKind) -> bytes:
    video = kind is AviKind.VIDEO
    fcc, handler = (b"vids", FCC_MPEG4) if video else (b"auds", bytes(4))
    rate, sample = (25, 0) if video else (SAMPLE_RATE, 2)
    body = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        fcc, handler, 0, 0, 0, 0, 1, rate, 0, 1, 0, U32_MAX, sample, 0, 0, DIM, DIM,
    )  # fmt: skip
    return _chunk(b"strh", body)


def _strf(kind: AviKind) -> bytes:
    if kind is AviKind.VIDEO:
        bih = struct.pack("<IiiHH4sIiiII", BI_SIZE, DIM, DIM, 1, 24, FCC_MPEG4, 0, 0, 0, 0, 0)
        return _chunk(b"strf", bih)
    wfx = struct.pack("<HHIIHHH", WAVE_FORMAT_PCM, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, 0)
    return _chunk(b"strf", wfx)


def _template(name: bytes, kind: AviKind) -> bytes:
    """*name* as full chunk bytes: a leaf, or a LIST holding its required children."""
    if name == b"strh":
        return _strh(kind)
    if name == b"strf":
        return _strf(kind)
    if name in _COMMON:
        return _COMMON[name]

    return _chunk(b"LIST", name + b"".join(_template(c, kind) for c in REQUIRED[name]))


_ALL = {t for d in (REQUIRED, OPTIONAL) for c in d.values() for t in c}
_TEMPLATES = {(t, k): _template(t, k) for t in _ALL for k in AviKind}
_KIND_DEP = frozenset(t for t in _ALL if len({_TEMPLATES[(t, k)] for k in AviKind}) > 1)


class Node(NamedTuple):
    """A RIFF/LIST container: header start, declared size, enclosing starts."""

    name: bytes
    start: int
    size: int
    ancestors: tuple[int, ...]


def _children(data: bytes, node: Node) -> tuple[list[tuple[bytes, int, int]], bool]:
    """(name, start, size) of each child that fits; False if a child lies."""
    pos = node.start + RIFF_HDR
    end = node.start + CHUNK_HDR + node.size
    out: list[tuple[bytes, int, int]] = []
    while pos + CHUNK_HDR <= end:
        fcc = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if pos + CHUNK_HDR + size > end:
            return out, False

        is_list = fcc == b"LIST" and size >= LIST_TYPE_LEN
        name = data[pos + CHUNK_HDR : pos + RIFF_HDR] if is_list else fcc
        out.append((name, pos, size if is_list else -1))
        pos += CHUNK_HDR + size + (size & 1)
    return out, True


def _strl_kind(data: bytes, kids: list[tuple[bytes, int, int]]) -> AviKind | None:
    for name, pos, _ in kids:
        if name == b"strh":
            return _KIND_OF_TYPE.get(data[pos + CHUNK_HDR : pos + RIFF_HDR])
    return None


Cand = tuple[Node, bytes, AviKind | None]


def _walk(data: bytes, node: Node, out: list[Cand]) -> None:
    kids, ok = _children(data, node)
    have = {name for name, _, _ in kids}
    kind = _strl_kind(data, kids) if node.name == b"strl" else None
    wanted = REQUIRED.get(node.name, ()) + OPTIONAL.get(node.name, ())
    out += [(node, t, kind) for t in wanted if t not in have]

    if not ok or len(node.ancestors) >= DEPTH_MAX:
        return
    for name, pos, size in kids:
        if size >= 0 and name in REQUIRED:
            _walk(data, Node(name, pos, size, (*node.ancestors, node.start)), out)


def _cands(data: bytes) -> list[Cand]:
    if not sniff_avi(data):
        return []
    riff_size = min(struct.unpack_from("<I", data, 4)[0], len(data) - CHUNK_HDR)
    out: list[Cand] = []
    _walk(data, Node(ROOT, 0, riff_size, ()), out)
    return out


def _candidates(data: bytes) -> list[tuple[Node, bytes]]:
    """(container, missing child) pairs in tree order."""
    return [(n, t) for n, t, _ in _cands(data)]


def _insert(data: bytes, node: Node, child: bytes, at_end: bool) -> bytes | None:
    """Insert *child* first (after the list type) or last in *node*; grow every size."""
    pos = node.start + (CHUNK_HDR + node.size if at_end else RIFF_HDR)
    out = bytearray(data[:pos] + child + data[pos:])
    for start in (*node.ancestors, node.start):
        size = struct.unpack_from("<I", out, start + 4)[0] + len(child)
        if size > U32_MAX:
            return None
        struct.pack_into("<I", out, start + 4, size)
    return bytes(out)


def grow_avi(data: bytes, max_len: int, rng: RandPool) -> bytes | None:
    """Add one missing chunk (with its subtree); None if nothing is missing.

    ``rng.choice`` picks the (container, child) pair, then the stream kind
    when the child depends on it and no strh decides it.
    """
    cands = _cands(data)
    if not cands:
        return None

    node, name, kind = cast("Cand", rng.choice(cands))
    if kind is None and name in _KIND_DEP:
        kind = cast("AviKind", rng.choice(tuple(AviKind)))
    child = _TEMPLATES[(name, kind or AviKind.VIDEO)]

    out = _insert(data, node, child, at_end=node.name == ROOT and name in _APPEND)
    if out is None or len(out) > max_len:
        return None
    return out
