"""Matroska CodecPrivate / ContentEncoding mutator (GStreamer matroskademux).

GStreamer's Matroska demuxer has been bitten three times by the track-level
codec configuration, and the container-level ``webm_chunk_mutate`` ops never
reach inside these leaves (CodecID swap is the deepest they go):

- CVE-2026-39043 (SA-2026-0022, fixed 1.28.2): heap overflow in the
  decompressed-buffer size calculation for **bz2-compressed tracks** —
  ``ContentCompAlgo`` 1 with compressed blocks.
- CVE-2026-17072 (SA-2026-0073, fixed 1.28.6): 4-byte OOB read in the FLAC
  stream header parser — a metadata block whose **length field** ends the
  block exactly at the CodecPrivate boundary (the next block header reads
  past it), or a loop that enters with <4 bytes left.
- GStreamer-SA-2026-0089: integer overflow and OOB read in the **Xiph codec
  data parser** — the A_VORBIS CodecPrivate header-count/lacing-length
  bytes that overshoot the buffer.

FFmpeg's Matroska demuxer parses the same leaves, so every edit reaches
both targets. Edits are in place inside one leaf (FLAC block lengths, Xiph
count/lacing) or a rebuilt ContentEncodings subtree (CompAlgo → bz2 1,
2, 3, 0) spliced into the TrackEntry; parent EBML size vints are
recomputed bottom-up by ``serialize_webm``.
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.webm import (
    Element,
    _encode_size_vint,
    _find_all,
    parse_webm,
    serialize_webm,
)
from fuzzer_tool.core.rand_pool import RandPool

TRACK_ENTRY = 0xAE
CODEC_ID = 0x86
CODEC_PRIVATE = 0x63A2
CONTENT_ENCODINGS = 0x6240

FLAC_CODECS = (b"A_FLAC", b"A/FLAC")
VORBIS_CODECS = (b"A_VORBIS", b"A/VORBIS")
FLAC_MAGIC = b"fLaC"

# ContentCompAlgo values: 0 zlib, 1 bz2, 2 lzo1x, 3 header stripping.
ALGO_EDGES = (1, 2, 3, 0)
XIPH_COUNT_EDGES = (0, 1, 3, 255)
XIPH_LENGTH_EDGES = (0, 1, 0x7F, 255)


def _leaf(track: Element, elem_id: int) -> Element | None:
    """First child leaf of *track* with *elem_id* (raw data, no children)."""
    for child in track.children:
        if child.elem_id == elem_id and not child.children:
            return child
    return None


def _comp_algo_positions(encodings: Element) -> list[int]:
    """Byte offsets of ContentCompAlgo id vints inside a ContentEncodings leaf."""
    return [
        pos
        for pos in range(len(encodings.data) - 1)
        if encodings.data[pos : pos + 2] == b"\x42\x54"
    ]


def _flac_headers(priv: bytes) -> list[int]:
    """Offsets of metadata block headers in a FLAC CodecPrivate past ``fLaC``.

    A declared length that overshoots the buffer still yields its header
    offset: the point is to rewrite it, and every header position up to the
    first bad one is a real site.
    """
    heads: list[int] = []
    pos = len(FLAC_MAGIC)
    while pos + 4 <= len(priv):
        heads.append(pos)
        blen = int.from_bytes(priv[pos + 1 : pos + 4], "big")
        if blen >= len(priv):  # already overshooting: later bytes are not headers
            break
        pos += 4 + blen
    return heads


def _xiph_lengths(priv: bytes) -> list[int]:
    """Positions of the lacing length bytes after the header-count byte."""
    lengths: list[int] = []
    pos = 1
    remain = priv[0] if priv else 0  # headers minus one lengths to read
    while pos < len(priv) and remain > 0:
        lengths.append(pos)
        if priv[pos] != 0xFF:  # a terminator ends one length
            remain -= 1
        pos += 1
    return lengths


def _build_comp_algo(algo: int) -> bytes:
    """A minimal ContentEncodings subtree with ContentCompAlgo=algo (bz2=1)."""
    comp_algo = b"\x42\x54\x81" + bytes([algo])
    compression = b"\x50\x34" + _encode_size_vint(len(comp_algo)) + comp_algo
    encoding_type = b"\x50\x33\x81\x00"  # ContentEncodingType: compression
    encoding = b"\x22\x40" + _encode_size_vint(len(compression) + len(encoding_type))
    encoding += compression + encoding_type
    return b"\x62\x40" + _encode_size_vint(len(encoding)) + encoding


def sniff_mkv_codec_cfg(data: bytes) -> bool:
    """Matroska/WebM with a track holding FLAC/Vorbis CodecPrivate or a
    ContentEncodings subtree (CompAlgo), or any track (injection)."""
    elements = parse_webm(data)
    if elements is None:
        return False
    return any(_modes(track) for track in _find_all(elements, TRACK_ENTRY))


def _modes(track: Element) -> list[str]:
    """Modes available for one TrackEntry: flac, xiph, comp."""
    out: list[str] = []
    codec = _leaf(track, CODEC_ID)
    priv = _leaf(track, CODEC_PRIVATE)
    if codec is not None and priv is not None:
        if codec.data in FLAC_CODECS and priv.data[:4] == FLAC_MAGIC and len(priv.data) > 4:
            out.append("flac")
        elif codec.data in VORBIS_CODECS and len(priv.data) > 2:
            out.append("xiph")
    encodings = [
        child
        for child in track.children
        if child.elem_id == CONTENT_ENCODINGS and not child.children
    ]
    if encodings:
        if any(encodings[0].data[pos + 2] == 0x81 for pos in _comp_algo_positions(encodings[0])):
            out.append("comp")  # rewrite the ContentCompAlgo value in place
    elif codec is not None:
        out.append("comp")  # no subtree yet: inject one
    return out


def _edit_flac(priv: Element, rng: RandPool) -> None:
    """Rewrite one FLAC metadata block length to a boundary-landing edge."""
    heads = _flac_headers(priv.data)
    pos = heads[rng.randint(0, len(heads) - 1)]
    remaining = len(priv.data) - (pos + 4)
    edges = [remaining, remaining - 1, remaining - 2, remaining - 3, remaining + 16]
    new_len = min(max(edges[rng.randint(0, len(edges) - 1)], 0), 0xFFFFFF)
    priv.data = priv.data[: pos + 1] + new_len.to_bytes(3, "big") + priv.data[pos + 4 :]


def _edit_xiph(priv: Element, rng: RandPool) -> None:
    """Rewrite the header-count or a lacing-length byte to an edge."""
    if rng.randint(0, 1):
        value = XIPH_COUNT_EDGES[rng.randint(0, len(XIPH_COUNT_EDGES) - 1)]
        priv.data = bytes([value]) + priv.data[1:]
        return
    lengths = _xiph_lengths(priv.data)
    pos = lengths[rng.randint(0, len(lengths) - 1)]
    value = XIPH_LENGTH_EDGES[rng.randint(0, len(XIPH_LENGTH_EDGES) - 1)]
    priv.data = priv.data[:pos] + bytes([value]) + priv.data[pos + 1 :]


def _edit_comp(track: Element, rng: RandPool) -> None:
    """Rewrite the ContentCompAlgo value in place, or inject a subtree."""
    for child in track.children:
        if child.elem_id != CONTENT_ENCODINGS or child.children:
            continue
        for pos in _comp_algo_positions(child):
            if child.data[pos + 2] == 0x81:  # size vint: one payload byte
                algo = ALGO_EDGES[rng.randint(0, len(ALGO_EDGES) - 1)]
                child.data = child.data[: pos + 3] + bytes([algo]) + child.data[pos + 4 :]
                return
    payload = _build_comp_algo(ALGO_EDGES[rng.randint(0, len(ALGO_EDGES) - 1)])
    track.children.append(
        Element(
            elem_id=CONTENT_ENCODINGS,
            id_raw=b"\x62\x40",
            size_raw=_encode_size_vint(len(payload)),
            size_val=len(payload),
            data=payload,
        )
    )


def mutate_mkv_codec_cfg(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Edit one Matroska track's codec config; None if declined.

    Draw order: TrackEntry, mode, field, value.
    """
    elements = parse_webm(data)
    if elements is None:
        return None
    tracks = [(track, mode) for track in _find_all(elements, TRACK_ENTRY) for mode in _modes(track)]
    if not tracks:
        return None
    track, mode = tracks[rng.randint(0, len(tracks) - 1)]

    if mode == "flac":
        priv = _leaf(track, CODEC_PRIVATE)
        assert priv is not None  # _modes() only offers flac with a private leaf
        _edit_flac(priv, rng)
    elif mode == "xiph":
        priv = _leaf(track, CODEC_PRIVATE)
        assert priv is not None  # _modes() only offers xiph with a private leaf
        _edit_xiph(priv, rng)
    else:
        _edit_comp(track, rng)
    out = serialize_webm(elements)
    return None if out == data else out[:max_len]
