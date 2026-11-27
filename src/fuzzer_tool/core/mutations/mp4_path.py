"""MP4 config-box plumbing: find boxes under sample entries, splice resized
boxes back, parse ``esds`` descriptors.

``isobmff.parse_boxes`` stops at ``stsd``; decoder configs sit below it:

  moov > trak > mdia > minf > stbl > stsd > avc1|hvc1|mp4a > avcC|hvcC|esds
                                     8 B    78 B (visual) / 28 B (+16/+36 QT v1/v2)
                                            fixed fields before child boxes

A resized box changes every ancestor's size and, when ``moov`` precedes
``mdat``, every chunk offset past the edit; ``splice`` fixes both.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

BOX_HDR = 8
MAX_DEPTH = 16
U32_MAX = 0xFFFFFFFF

CONTAINERS = frozenset(
    {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta", b"mvex"}
    | {b"moof", b"traf", b"sinf", b"schi", b"wave"}
)
VISUAL_ENTRY_LEN = 78
AUDIO_ENTRY_LEN = 28
QT_SOUND_EXTRA = {1: 16, 2: 36}  # QuickTime sound description v1/v2
QT_VERSION_OFF = 8  # u16 inside the audio sample entry body
_FIXED_BEFORE_CHILDREN = {b"stsd": 8} | dict.fromkeys(
    (b"avc1", b"avc3", b"hvc1", b"hev1", b"encv", b"mp4v"), VISUAL_ENTRY_LEN
)
_AUDIO_ENTRIES = frozenset({b"mp4a", b"enca"})
_CHUNK_OFFSETS = {b"stco": 4, b"co64": 8}  # entry width

ES_TAG, DC_TAG, DSI_TAG = 3, 4, 5
FULLBOX_HDR = 4
DC_FIXED_LEN = 13  # objectTypeIndication .. avgBitrate
ES_DEPENDS, ES_URL, ES_OCR = 0x80, 0x40, 0x20
MAX_LEN_BYTES = 4


@dataclass(frozen=True)
class BoxHit:
    """A box at *start* of *size* bytes; ancestors are enclosing box starts."""

    start: int
    size: int
    ancestors: tuple[int, ...]


def _children_at(data: bytes, pos: int, typ: bytes) -> int | None:
    """Offset of a box's first child, or None for a leaf."""
    body = pos + BOX_HDR
    if typ in CONTAINERS:
        return body
    if typ in _FIXED_BEFORE_CHILDREN:
        return body + _FIXED_BEFORE_CHILDREN[typ]
    if typ not in _AUDIO_ENTRIES or body + QT_VERSION_OFF + 2 > len(data):
        return None
    version = struct.unpack_from(">H", data, body + QT_VERSION_OFF)[0]
    return body + AUDIO_ENTRY_LEN + QT_SOUND_EXTRA.get(version, 0)


def _walk(data, pos, end, ancestors, types, hits) -> None:
    if len(ancestors) > MAX_DEPTH:
        return
    while pos + BOX_HDR <= end:
        size = struct.unpack_from(">I", data, pos)[0]
        if size < BOX_HDR or pos + size > end:
            return  # 0 (to EOF), 1 (64-bit) and lying sizes end the level
        typ = data[pos + 4 : pos + 8]
        if typ in types:
            hits.append(BoxHit(pos, size, ancestors))

        child = _children_at(data, pos, typ)
        if child is not None:
            _walk(data, child, pos + size, (*ancestors, pos), types, hits)
        pos += size


def find_boxes(data: bytes, types: frozenset[bytes]) -> list[BoxHit]:
    """Every box of *types*, in file order, with its ancestor chain."""
    hits: list[BoxHit] = []
    _walk(data, 0, len(data), (), types, hits)
    return hits


def _shift_chunk_offsets(out: bytearray, at: int, delta: int) -> None:
    """Add *delta* to stco/co64 entries pointing at or past *at*."""
    for hit in find_boxes(bytes(out), frozenset(_CHUNK_OFFSETS)):
        width = _CHUNK_OFFSETS[bytes(out[hit.start + 4 : hit.start + 8])]
        fmt = ">I" if width == 4 else ">Q"
        first = hit.start + BOX_HDR + FULLBOX_HDR + 4
        if first > hit.start + hit.size:
            continue  # truncated table: no entry count to read
        count = struct.unpack_from(">I", out, first - 4)[0]
        last = min(first + count * width, hit.start + hit.size)

        for off in range(first, last - width + 1, width):
            value = struct.unpack_from(fmt, out, off)[0]
            if value >= at:
                struct.pack_into(fmt, out, off, (value + delta) % (1 << (8 * width)))


def splice(data: bytes, hit: BoxHit, payload: bytes) -> bytes | None:
    """Replace *hit*'s payload; fix ancestor sizes and chunk offsets.

    None when an ancestor size would leave the 32-bit range.
    """
    delta = BOX_HDR + len(payload) - hit.size
    out = bytearray(data[: hit.start])
    out += struct.pack(">I", BOX_HDR + len(payload)) + data[hit.start + 4 : hit.start + 8]
    out += payload + data[hit.start + hit.size :]
    if not delta:
        return bytes(out)

    for anc in hit.ancestors:
        size = struct.unpack_from(">I", out, anc)[0] + delta
        if not BOX_HDR <= size <= U32_MAX:
            return None
        struct.pack_into(">I", out, anc, size)
    _shift_chunk_offsets(out, hit.start + hit.size, delta)
    return bytes(out)


@dataclass(frozen=True)
class Descr:
    """MPEG-4 descriptor: tag at *start*, expandable length at *len_off*."""

    tag: int
    start: int
    len_off: int
    len_width: int
    length: int
    payload_off: int

    @property
    def end(self) -> int:
        return self.payload_off + self.length


@dataclass(frozen=True)
class EsdsLayout:
    """ES_Descriptor > DecoderConfigDescriptor > DecoderSpecificInfo."""

    es: Descr
    dc: Descr
    dsi: Descr | None


def read_descr(buf: bytes, pos: int, limit: int) -> Descr | None:
    """Descriptor at *pos* whose payload ends by *limit*, or None."""
    length, p = 0, pos + 1
    for _ in range(MAX_LEN_BYTES):
        if p >= limit:
            return None
        b = buf[p]
        length = (length << 7) | (b & 0x7F)
        p += 1
        if not b & 0x80:
            break
    if p + length > limit:
        return None
    return Descr(buf[pos], pos, pos + 1, p - pos - 1, length, p)


def write_descr_len(buf: bytearray, len_off: int, width: int, value: int) -> bool:
    """Rewrite an expandable length in place, keeping its width."""
    if value >> (7 * width):
        return False
    groups = [(value >> (7 * k)) & 0x7F for k in range(width - 1, -1, -1)]
    buf[len_off : len_off + width] = bytes([g | 0x80 for g in groups[:-1]] + [groups[-1]])
    return True


def _es_children(payload: bytes, es: Descr) -> int | None:
    """Offset after the ES_Descriptor's optional fields."""
    p = es.payload_off + 2  # ES_ID
    if p >= es.end:
        return None
    flags = payload[p]
    p += 1
    p += 2 if flags & ES_DEPENDS else 0
    if flags & ES_URL:
        if p >= es.end:
            return None
        p += 1 + payload[p]
    p += 2 if flags & ES_OCR else 0
    return p if p < es.end else None


def parse_esds(payload: bytes) -> EsdsLayout | None:
    """Descriptor layout of an ``esds`` payload (after version/flags), or None."""
    es = read_descr(payload, FULLBOX_HDR, len(payload))
    if es is None or es.tag != ES_TAG:
        return None
    p = _es_children(payload, es)
    dc = None if p is None else read_descr(payload, p, es.end)
    if dc is None or dc.tag != DC_TAG or dc.length < DC_FIXED_LEN:
        return None

    q = dc.payload_off + DC_FIXED_LEN
    dsi = read_descr(payload, q, dc.end) if q < dc.end else None
    return EsdsLayout(es, dc, dsi if dsi is not None and dsi.tag == DSI_TAG else None)
