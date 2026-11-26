"""FLV payload-field mutator (GStreamer ``flvdemux``).

``flv_chunk_mutate`` edits tag headers only; this overwrites the fields
inside tag payloads that flvdemux dispatches on:

  header   flags (bit0 video, bit2 audio)
  audio    flags (codec nibble 10 AAC / 9 enhanced) | AAC packet type | fourcc
  video    flags (bit7 ExHeader, frame type, codec) | AVC packet type | CTS (s24)
           enhanced: flags | fourcc (avc1/hvc1/...) | CTS after CodedFrames
  script   "onMetaData" ECMA count; per known key: AMF type byte, f64 value
           (NaN/inf reach the double->int casts in parse_metadata_item)

Edits are in place; every size field holds.
"""

from __future__ import annotations

import struct

from fuzzer_tool.core.mutations.field_poke import Endian, Field, Site, Table, poke, site
from fuzzer_tool.core.mutations.flv import TAG_AUDIO, TAG_SCRIPT, TAG_VIDEO, parse_flv
from fuzzer_tool.core.rand_pool import RandPool

FLV_HDR = 9
PREV_LEN = 4
TAG_HDR = 11
TAG_TYPE_MASK = 0x1F  # bit 5 is the filter flag
CODEC_SHIFT = 4
LOW_NIBBLE = 0x0F
EX_HEADER = 0x80
AUDIO_AAC = 10
AUDIO_ENHANCED = 9
VIDEO_CTS_CODECS = (7, 12)  # AVC, legacy HEVC
CODED_FRAMES = 1
CTS_FOURCCS = (b"avc1", b"hvc1")
FOURCC_LEN = 4

HEAD_FLAGS = (0, 1, 4, 5, 0xFF)
AUDIO_FLAGS = (0x0F, 0x2F, 0xAF, 0xBF, 0xEF, 0x90, 0x91, 0x94, 0x95, 0x96, 0xFF)
AAC_TYPES = (0, 1, 2, 0xFF)
AUDIO_FOURCCS = (b".mp3", b"mp4a", b"Opus", b"fLaC", b"ac-3", b"\0\0\0\0")
VIDEO_FLAGS = (0x17, 0x27, 0x57, 0x14, 0x1C, 0x90, 0x91, 0x93, 0x96, 0x97, 0xFF)
PKT_TYPES = (0, 1, 2, 0xFF)
CTS_EDGES = (0, 1, 0x7FFFFF, 0x800000, 0xFFFFFF)
VIDEO_FOURCCS = (b"avc1", b"hvc1", b"av01", b"vp09", b"\0\0\0\0")
AMF_TYPES = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)
AMF_NUMBER = 0
F64_LEN = 8
DOUBLE_EDGES = tuple(
    struct.pack(">d", v)
    for v in (0.0, -0.0, -1.0, float("nan"), float("inf"), float("-inf"), 1e300, 2.0**63, 5e-324)
)
ON_META = b"onMetaData"
AMF_ECMA = 8
META_KEYS = (
    b"duration",
    b"width",
    b"height",
    b"framerate",
    b"audiosamplerate",
    b"audiosamplesize",
    b"audiocodecid",
    b"videocodecid",
    b"AspectRatioX",
    b"AspectRatioY",
    b"channels",
    b"samplerate",
    b"times",
    b"filepositions",
)

HEAD = Table((Field("flags", 4, 1, HEAD_FLAGS),))


def sniff_flv_fields(data: bytes) -> bool:
    """FLV magic plus a header, PreviousTagSize and one tag header."""
    return len(data) >= FLV_HDR + PREV_LEN + TAG_HDR and data[:3] == b"FLV"


def _audio(body: bytes) -> tuple[Field, ...]:
    codec = body[0] >> CODEC_SHIFT
    flags = Field("flags", 0, 1, AUDIO_FLAGS)
    if codec == AUDIO_AAC:
        return flags, Field("aac_type", 1, 1, AAC_TYPES)
    if codec == AUDIO_ENHANCED:
        return flags, Field("fourcc", 1, FOURCC_LEN, AUDIO_FOURCCS)
    return (flags,)


def _video(body: bytes) -> tuple[Field, ...]:
    b0 = body[0]
    flags = Field("flags", 0, 1, VIDEO_FLAGS)
    if b0 & EX_HEADER:
        fcc = body[1 : 1 + FOURCC_LEN]
        fourcc = Field("fourcc", 1, FOURCC_LEN, VIDEO_FOURCCS)
        if (b0 & LOW_NIBBLE) == CODED_FRAMES and fcc in CTS_FOURCCS:
            return flags, fourcc, Field("cts", 1 + FOURCC_LEN, 3, CTS_EDGES)
        return flags, fourcc
    if (b0 & LOW_NIBBLE) in VIDEO_CTS_CODECS:
        return flags, Field("pkt_type", 1, 1, PKT_TYPES), Field("cts", 2, 3, CTS_EDGES)
    return (flags,)


def _script(body: bytes) -> tuple[Field, ...]:
    """ECMA count after onMetaData, then type/value of each known key."""
    fields: list[Field] = []
    meta = body.find(len(ON_META).to_bytes(2, "big") + ON_META + bytes([AMF_ECMA]))
    if meta >= 0:
        fields.append(Field("ecma_count", meta + 2 + len(ON_META) + 1, 4))

    for key in META_KEYS:
        p = body.find(len(key).to_bytes(2, "big") + key)
        t = p + 2 + len(key)
        if p < 0 or t >= len(body):
            continue
        fields.append(Field("type", t, 1, AMF_TYPES))
        if body[t] == AMF_NUMBER:
            fields.append(Field("value", t + 1, F64_LEN, DOUBLE_EDGES))
    return tuple(fields)


_TAG_FIELDS = {TAG_AUDIO: _audio, TAG_VIDEO: _video, TAG_SCRIPT: _script}


def _sites(data: bytes) -> list[Site]:
    """Header flags, then one site per tag with fitting fields."""
    sites = [Site(0, FLV_HDR, HEAD)]
    parsed = parse_flv(data)
    if parsed is None:
        return sites

    header, tags, _ = parsed
    pos = len(header)
    for tag in tags:
        pos += PREV_LEN + TAG_HDR
        body, base = tag.data, pos
        pos += len(body)
        fn = _TAG_FIELDS.get(tag.tag_type & TAG_TYPE_MASK)
        if fn is None or not body:
            continue
        s = site(base, len(body), Table(fn(body)))
        if s:
            sites.append(s)
    return sites


def mutate_flv_fields(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """One FLV header/tag-payload field set to an edge; None if declined."""
    if len(data) < FLV_HDR or data[:3] != b"FLV":
        return None
    out = poke(data, _sites(data), Endian.BIG, rng)
    return None if out is None else out[:max_len]
