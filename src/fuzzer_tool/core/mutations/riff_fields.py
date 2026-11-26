"""AVI / WAV payload-field mutators (GStreamer ``avidemux`` / ``wavparse``).

``riff_chunk_mutate`` edits top-level framing only; these walk the nested
LIST tree and overwrite the fields the demuxers branch and divide on:

  RIFF AVI  LIST hdrl  avih            us_frame, tot_frames
                       LIST strl  strh  type, scale, rate, length, ...
                                  strf  BITMAPINFOHEADER | WAVEFORMATEX (by strh type)
                                  indx  wLongsPerEntry (bpe = w*4 in a guint16), entries
            idx1                        16-byte entries: id, flags, offset, size
  RIFF WAVE fmt (EXTENSIBLE subformat), fact, ds64, data size, cue, smpl, acid

Edits are in place; lengths and every size field outside the edit hold.
"""

from __future__ import annotations

from typing import NamedTuple

from fuzzer_tool.core.mutations.field_poke import Endian, Field, Site, Table, poke, site
from fuzzer_tool.core.rand_pool import RandPool

RIFF_HDR = 12
CHUNK_HDR = 8
DEPTH_MAX = 16
LIST_IDS = frozenset({b"LIST", b"RIFF"})  # RIFF: OpenDML AVIX extension
LIST_TYPE_LEN = 4
TYPE_LEN = 4
INDEX_PREFIX = b"ix"  # ix00, ix01, ... standard index

MAGICS = (b"RIFF", b"RF64", b"AVF0", b"BW64")
FCC_TYPES = (b"vids", b"auds", b"txts", b"iavs", b"mids")
HANDLERS = (b"\0\0\0\0", b"DIB ", b"H264", b"XVID", b"MJPG", b"DXSB", b"DXSA")
BITCOUNT_EDGES = (0, 1, 4, 8, 16, 24, 32, 0xFFFF)
FORMAT_EDGES = (0, 1, 2, 3, 6, 7, 0x11, 0x55, 0xFF, 0x2000, 0xFFFE, 0xFFFF)
SUBFORMAT_EDGES = (0, 1, 3, 6, 7, 0x92, 0xFFFE)  # EXTENSIBLE GUID Data1
LONGS_EDGES = (0, 1, 2, 4, 0x4000, 0x4001, 0xFFFF)  # 0x4001*4 wraps to 4
INDEX_TYPE_EDGES = (0, 1, 0x80, 0xFF)  # 0x80: AVI_INDEX_IS_DATA
IDX1_IDS = (b"00dc", b"01wb", b"rec ", b"\0\0\0\0", b"99dc")
IDX1_FLAG_EDGES = (0, 0x10, 0x100, 0xFFFFFFFF)  # 0x10: keyframe
DATA_SIZE_EDGES = (0, 1, 0x7FFFFFFF, 0xFFFFFFFF)  # 0: to EOF; ~0: ds64
F32_EDGES = (0, 0x7FC00000, 0x7F800000, 0xFF800000, 1, 0x7F7FFFFF)  # NaN, +-inf, denormal


class Chunk(NamedTuple):
    """Chunk id, absolute payload offset, payload bytes present."""

    fourcc: bytes
    off: int
    size: int


def walk_riff(data: bytes) -> list[Chunk]:
    """Every chunk under the RIFF header, depth first, LISTs descended.

    Sizes past their parent are clipped; nesting stops at DEPTH_MAX.
    """
    out: list[Chunk] = []
    stack = [(RIFF_HDR, len(data), 0)]
    while stack:
        pos, end, depth = stack.pop()
        if pos + CHUNK_HDR > end:
            continue

        fcc = data[pos : pos + 4]
        body = pos + CHUNK_HDR
        size = min(int.from_bytes(data[pos + 4 : body], "little"), end - body)
        out.append(Chunk(fcc, body, size))

        # Sibling first, child last: the child pops next (document order).
        stack.append((body + size + (size & 1), end, depth))
        if fcc in LIST_IDS and size >= LIST_TYPE_LEN and depth < DEPTH_MAX:
            stack.append((body + LIST_TYPE_LEN, body + size, depth + 1))
    return out


AVIH = Table((Field("us_frame", 0, 4), Field("tot_frames", 16, 4)))
STRH = Table(
    (
        Field("type", 0, 4, FCC_TYPES),
        Field("handler", 4, 4, HANDLERS),
        Field("scale", 20, 4),
        Field("rate", 24, 4),
        Field("start", 28, 4),
        Field("length", 32, 4),
        Field("bufsize", 36, 4),
        Field("samplesize", 44, 4),
    )
)
BIH = Table(
    (
        Field("width", 4, 4),
        Field("height", 8, 4),
        Field("bitcount", 14, 2, BITCOUNT_EDGES),
        Field("compression", 16, 4, HANDLERS),
        Field("num_colors", 32, 4),
    )
)
WFX = Table(
    (
        Field("format", 0, 2, FORMAT_EDGES),
        Field("channels", 2, 2),
        Field("rate", 4, 4),
        Field("av_bps", 8, 4),
        Field("blockalign", 12, 2),
        Field("bits", 14, 2),
        Field("cb_size", 16, 2),
        Field("valid_bits", 18, 2),
        Field("channel_mask", 20, 4),
        Field("subformat", 24, 4, SUBFORMAT_EDGES),
    )
)
INDX = Table(
    (
        Field("longs_per_entry", 0, 2, LONGS_EDGES),
        Field("sub_type", 2, 1),
        Field("index_type", 3, 1, INDEX_TYPE_EDGES),
        Field("entries", 4, 4),
        Field("base_offset", 12, 4),
    )
)
IDX1 = Table(
    (
        Field("id", 0, 4, IDX1_IDS),
        Field("flags", 4, 4, IDX1_FLAG_EDGES),
        Field("offset", 8, 4),
        Field("size", 12, 4),
    ),
    stride=16,
)
VPRP = Table((Field("aspect", 20, 4), Field("fields", 32, 4)))
ONE_U32 = Table((Field("value", 0, 4),))  # dmlh total frames, fact samples
DS64 = Table(tuple(Field(f"word{i}", 4 * i, 4) for i in range(6)))  # riff/data/samples lo,hi
DATA = Table((Field("size", -4, 4, DATA_SIZE_EDGES),))
CUE = Table((Field("count", 0, 4), Field("id", 4, 4), Field("position", 8, 4)))
SMPL = Table((Field("midi_note", 12, 4),))
ACID = Table((Field("tempo", 20, 4, F32_EDGES),))
HEAD = Table((Field("magic", 0, 4, MAGICS),))

_AVI_TABLES = {
    b"avih": AVIH,
    b"strh": STRH,
    b"vprp": VPRP,
    b"dmlh": ONE_U32,
    b"indx": INDX,
    b"idx1": IDX1,
}
_STRF_TABLES = {b"vids": BIH, b"auds": WFX}
_WAV_TABLES = {
    b"fmt ": WFX,
    b"fact": ONE_U32,
    b"ds64": DS64,
    b"data": DATA,
    b"cue ": CUE,
    b"labl": ONE_U32,
    b"note": ONE_U32,
    b"smpl": SMPL,
    b"acid": ACID,
}


def sniff_avi(data: bytes) -> bool:
    return len(data) >= RIFF_HDR and data[:4] == b"RIFF" and data[8:12] == b"AVI "


def sniff_wav(data: bytes) -> bool:
    return len(data) >= RIFF_HDR and data[:4] in (b"RIFF", b"RF64") and data[8:12] == b"WAVE"


def _avi_table(c: Chunk, stream: bytes) -> Table | None:
    if c.fourcc == b"strf":
        return _STRF_TABLES.get(stream)
    if c.fourcc[:2] == INDEX_PREFIX:
        return INDX
    return _AVI_TABLES.get(c.fourcc)


def _avi_sites(data: bytes) -> list[Site]:
    """Sites in document order; strf is typed by the strh before it."""
    sites: list[Site] = []
    stream = b""
    for c in walk_riff(data):
        if c.fourcc == b"strh" and c.size >= TYPE_LEN:
            stream = data[c.off : c.off + TYPE_LEN]
        table = _avi_table(c, stream)
        s = site(c.off, c.size, table) if table else None
        if s:
            sites.append(s)
    return sites


def _wav_sites(data: bytes) -> list[Site]:
    """Header magic first, then chunk sites in document order."""
    sites = [Site(0, RIFF_HDR, HEAD)]
    for c in walk_riff(data):
        table = _WAV_TABLES.get(c.fourcc)
        s = site(c.off, c.size, table) if table else None
        if s:
            sites.append(s)
    return sites


def mutate_avi(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """One AVI header/index field set to an edge; None if declined."""
    if not sniff_avi(data):
        return None
    out = poke(data, _avi_sites(data), Endian.LITTLE, rng)
    return None if out is None else out[:max_len]


def mutate_wav(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """One WAV magic/fmt/fact/ds64/data/cue field set to an edge; None if declined."""
    if not sniff_wav(data):
        return None
    out = poke(data, _wav_sites(data), Endian.LITTLE, rng)
    return None if out is None else out[:max_len]
