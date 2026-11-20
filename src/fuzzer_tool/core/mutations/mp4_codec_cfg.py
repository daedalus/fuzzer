"""avcC / hvcC / esds field mutator (MP4/MOV decoder configuration).

``isobmff_chunk_mutate`` never parses below ``stsd``, so the decoder
config records get only byte noise. This op edits their fields:

  avcC  version profile compat level | 111111 lengthSizeMinusOne | 111 numSPS
        { u16 len, SPS } numPPS { u16 len, PPS }
  hvcC  version profile .. level@12 .. lengthSizeMinusOne@21 numArrays@22
        { type, u16 count, { u16 len, NALU } }
  esds  descriptor lengths, objectTypeIndication, streamType, buffer, bitrates

FIELD writes an edge value in place (size unchanged). PS_EDIT rewrites one
SPS/PPS field inside avcC through ``h264_ps.edit_ps``, fixes its u16
length, and splices the resized box (ancestor sizes, chunk offsets).
The AudioSpecificConfig inside esds is ``aac_asc_mutate``'s job.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

from fuzzer_tool.core.mutations.h264_ps import edit_ps, escape, unescape
from fuzzer_tool.core.mutations.mp4_path import (
    BOX_HDR,
    find_boxes,
    parse_esds,
    splice,
    write_descr_len,
)
from fuzzer_tool.core.rand_pool import RandPool

CONFIG_TYPES = frozenset({b"avcC", b"hvcC", b"esds"})
AVCC_FIXED = 6
HVCC_FIXED = 23
HVCC_LEVEL_OFF, HVCC_LEN_SIZE_OFF = 12, 21
NAL_TYPE_MASK = 0x1F
PS_TYPES = frozenset({7, 8})
U16_MAX = 0xFFFF

VERSION_EDGES = (0, 1, 2, 0xFF)
W1_EDGES = (0, 1, 0x7F, 0x80, 0xFF)
W2_EDGES = (0, 1, 0x7FFF, 0x8000, 0xFFFF)
W3_EDGES = (0, 1, 0xFFFFFF)
W4_EDGES = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
LEN_SIZE_EDGES = (0xFC, 0xFD, 0xFE, 0xFF, 0x00, 0x03)  # lengthSizeMinusOne 0..3
NUM_SPS_EDGES = (0xE0, 0xE1, 0xE2, 0xFF, 0x00, 0x1F)
HVCC_ARRAY_EDGES = (0x20, 0x21, 0x22, 0x27, 0x28, 0x3F, 0xA1)  # VPS/SPS/PPS/SEI
OTI_EDGES = (0x00, 0x20, 0x40, 0x66, 0x67, 0x69, 0x6B, 0xA5, 0xDD, 0xFF)
STREAM_EDGES = (0x00, 0x05, 0x11, 0x15, 0xFF)
DESCR_LEN_EDGES = (0, 1, 0x7F, 0x0FFFFFFF)  # clamped to the field's width


class Op(IntEnum):
    """Config edit; values are the RNG draw."""

    FIELD = 0
    PS_EDIT = 1


class SiteKind(IntEnum):
    """How a site's value is encoded."""

    INT = 0  # big-endian unsigned
    DESCR_LEN = 1  # MPEG-4 expandable length, width kept


@dataclass(frozen=True)
class Site:
    """An editable field at *off* in a config box payload."""

    off: int
    width: int
    edges: tuple[int, ...]
    kind: SiteKind = SiteKind.INT


def _len_prefixed(p: bytes, pos: int, count: int, sites: list[Site], spans: list) -> int:
    """Walk *count* ``u16 len, NALU`` entries; returns the end offset."""
    for _ in range(count):
        if pos + 2 > len(p):
            return len(p)
        n = struct.unpack_from(">H", p, pos)[0]
        sites.append(Site(pos, 2, W2_EDGES))
        if pos + 2 + n > len(p):
            return len(p)
        spans.append((pos, n))
        pos += 2 + n
    return pos


def _avcc(p: bytes) -> tuple[list[Site], list[tuple[int, int]]]:
    if len(p) < AVCC_FIXED:
        return [], []
    sites = [Site(0, 1, VERSION_EDGES), Site(1, 1, W1_EDGES), Site(2, 1, W1_EDGES)]
    sites += [Site(3, 1, W1_EDGES), Site(4, 1, LEN_SIZE_EDGES), Site(5, 1, NUM_SPS_EDGES)]
    spans: list[tuple[int, int]] = []

    pos = _len_prefixed(p, AVCC_FIXED, p[5] & 0x1F, sites, spans)
    if pos < len(p):
        sites.append(Site(pos, 1, W1_EDGES))
        _len_prefixed(p, pos + 1, p[pos], sites, spans)
    return sites, spans


def _hvcc(p: bytes) -> list[Site]:
    if len(p) < HVCC_FIXED:
        return []
    sites = [Site(0, 1, VERSION_EDGES), Site(1, 1, W1_EDGES), Site(HVCC_LEVEL_OFF, 1, W1_EDGES)]
    sites += [Site(HVCC_LEN_SIZE_OFF, 1, LEN_SIZE_EDGES), Site(HVCC_FIXED - 1, 1, W1_EDGES)]

    pos = HVCC_FIXED
    for _ in range(p[HVCC_FIXED - 1]):
        if pos + 3 > len(p):
            break
        sites += [Site(pos, 1, HVCC_ARRAY_EDGES), Site(pos + 1, 2, W2_EDGES)]
        count = struct.unpack_from(">H", p, pos + 1)[0]
        pos = _len_prefixed(p, pos + 3, count, sites, [])
    return sites


def _esds(p: bytes) -> list[Site]:
    layout = parse_esds(p)
    if layout is None:
        return []
    descrs = [d for d in (layout.es, layout.dc, layout.dsi) if d is not None]
    sites = [Site(d.len_off, d.len_width, DESCR_LEN_EDGES, SiteKind.DESCR_LEN) for d in descrs]

    dc = layout.dc.payload_off
    sites += [Site(dc, 1, OTI_EDGES), Site(dc + 1, 1, STREAM_EDGES), Site(dc + 2, 3, W3_EDGES)]
    sites += [Site(dc + 5, 4, W4_EDGES), Site(dc + 9, 4, W4_EDGES)]
    return sites


def config_sites(typ: bytes, payload: bytes) -> tuple[list[Site], list[tuple[int, int]]]:
    """(editable sites, avcC SPS/PPS spans as (len offset, NAL length))."""
    if typ == b"avcC":
        return _avcc(payload)
    return (_hvcc(payload) if typ == b"hvcC" else _esds(payload)), []


def _write(buf: bytearray, site: Site, value: int) -> None:
    if site.kind is SiteKind.DESCR_LEN:
        write_descr_len(buf, site.off, site.width, min(value, (1 << (7 * site.width)) - 1))
        return
    mask = (1 << (8 * site.width)) - 1
    buf[site.off : site.off + site.width] = (value & mask).to_bytes(site.width, "big")


def _field(data: bytes, hits: list, rng: RandPool) -> bytes | None:
    """FIELD: one site of one config box to an edge value."""
    hit = hits[rng.randint(0, len(hits) - 1)]
    payload = bytearray(data[hit.start + BOX_HDR : hit.start + hit.size])
    sites, _ = config_sites(data[hit.start + 4 : hit.start + 8], bytes(payload))
    if not sites:
        return None

    site = sites[rng.randint(0, len(sites) - 1)]
    _write(payload, site, rng.choice(site.edges))
    return splice(data, hit, bytes(payload))


def _ps_edit(data: bytes, hits: list, rng: RandPool) -> bytes | None:
    """PS_EDIT: one SPS/PPS field inside an avcC, length and boxes fixed."""
    avc = [h for h in hits if data[h.start + 4 : h.start + 8] == b"avcC"]
    if not avc:
        return None
    hit = avc[rng.randint(0, len(avc) - 1)]
    payload = data[hit.start + BOX_HDR : hit.start + hit.size]
    _, spans = _avcc(payload)
    if not spans:
        return None

    len_off, n = spans[rng.randint(0, len(spans) - 1)]
    nal = payload[len_off + 2 : len_off + 2 + n]
    if len(nal) < 2 or nal[0] & NAL_TYPE_MASK not in PS_TYPES:
        return None
    rbsp = edit_ps(nal[0] & NAL_TYPE_MASK, unescape(nal[1:]), rng)
    if rbsp is None:
        return None

    new_nal = nal[:1] + escape(rbsp)
    if len(new_nal) > U16_MAX:
        return None
    new = payload[:len_off] + struct.pack(">H", len(new_nal)) + new_nal + payload[len_off + 2 + n :]
    return splice(data, hit, new)


def mutate_mp4_cfg(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Edit one avcC/hvcC/esds field or one avcC SPS/PPS field; None if declined.

    Draw order: op, box index, site/span index, then the value draw(s).
    """
    hits = find_boxes(data, CONFIG_TYPES)
    if not hits:
        return None

    op = Op(rng.randint(0, len(Op) - 1))
    out = _field(data, hits, rng) if op is Op.FIELD else _ps_edit(data, hits, rng)
    return None if out is None else out[:max_len]
