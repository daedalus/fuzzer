"""Tests for the Matroska codec-config mutator (`mkv_codec_cfg_mutate`)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.core.mutations.mkv_codec_cfg import (
    ALGO_EDGES,
    XIPH_COUNT_EDGES,
    _build_comp_algo,
    _flac_headers,
    _modes,
    _xiph_lengths,
    mutate_mkv_codec_cfg,
    sniff_mkv_codec_cfg,
)
from fuzzer_tool.core.mutations.webm import (
    Element,
    _encode_size_vint,
    _find_all,
    parse_webm,
    serialize_webm,
)
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

FFMPEG = shutil.which("ffmpeg")

TRACK_ENTRY = 0xAE
CODEC_ID = 0x86
CODEC_PRIVATE = 0x63A2


def _leaf(elem_id: int, id_raw: bytes, data: bytes) -> Element:
    return Element(elem_id, id_raw, _encode_size_vint(len(data)), len(data), data=data)


def _container(elem_id: int, id_raw: bytes, children: list[Element]) -> Element:
    payload = serialize_webm(children)
    return Element(
        elem_id, id_raw, _encode_size_vint(len(payload)), len(payload), children=children
    )


def _mkv(track_children: list[Element]) -> bytes:
    """A minimal matroska file with one TrackEntry built from *children*."""
    header = _container(
        0x1A45DFA3,
        b"\x1a\x45\xdf\xa3",
        [
            _leaf(0x4286, b"\x42\x86", b"\x01"),  # EBMLVersion
            _leaf(0x42F7, b"\x42\xf7", b"\x01"),  # EBMLReadVersion
            _leaf(0x42F2, b"\x42\xf2", b"\x04"),  # EBMLMaxIDLength
            _leaf(0x42F3, b"\x42\xf3", b"\x08"),  # EBMLMaxSizeLength
            _leaf(0x4282, b"\x42\x82", b"matroska"),  # DocType
            _leaf(0x4287, b"\x42\x87", b"\x04"),  # DocTypeVersion
            _leaf(0x4285, b"\x42\x85", b"\x02"),  # DocTypeReadVersion
        ],
    )
    track = _container(TRACK_ENTRY, b"\xae", track_children)
    tracks = _container(0x1654AE6B, b"\x16\x54\xae\x6b", [track])
    segment = _container(0x18538067, b"\x18\x53\x80\x67", [tracks])
    return serialize_webm([header, segment])


# FLAC CodecPrivate: "fLaC" + STREAMINFO (type 0, 34 B) + a 10 B second block.
FLAC_PRIV = b"fLaC" + b"\x00\x00\x00\x22" + bytes(34) + b"\x81\x00\x00\x0a" + bytes(10)
FLAC_MKV = _mkv(
    [
        _leaf(0xD7, b"\xd7", b"\x01"),  # TrackNumber
        _leaf(0x83, b"\x83", b"\x02"),  # TrackType: audio
        _leaf(CODEC_ID, b"\x86", b"A_FLAC"),
        _leaf(CODEC_PRIVATE, b"\x63\xa2", FLAC_PRIV),
    ]
)

# Vorbis CodecPrivate: header count 3, Xiph lacing 30+30, three headers.
VORBIS_PRIV = b"\x02\x1e\x1e" + b"A" * 30 + b"B" * 30 + b"C" * 20
VORBIS_MKV = _mkv(
    [
        _leaf(0xD7, b"\xd7", b"\x01"),
        _leaf(0x83, b"\x83", b"\x02"),
        _leaf(CODEC_ID, b"\x86", b"A_VORBIS"),
        _leaf(CODEC_PRIVATE, b"\x63\xa2", VORBIS_PRIV),
    ]
)

# A track with a zlib ContentEncodings subtree (ContentCompAlgo 0).
COMP_MKV = _mkv(
    [
        _leaf(0xD7, b"\xd7", b"\x01"),
        _leaf(0x83, b"\x83", b"\x02"),
        _leaf(CODEC_ID, b"\x86", b"A_FLAC"),
        _leaf(CODEC_PRIVATE, b"\x63\xa2", FLAC_PRIV),
        _leaf(0x6240, b"\x62\x40", _build_comp_algo(0)),
    ]
)


def _codec_private(data: bytes) -> bytes:
    elements = parse_webm(data)
    assert elements is not None
    track = _find_all(elements, TRACK_ENTRY)[0]
    priv = next(c for c in track.children if c.elem_id == CODEC_PRIVATE)
    return priv.data


class TestParse:
    def test_flac_headers(self):
        assert _flac_headers(FLAC_PRIV) == [4, 42]  # both block headers

    def test_flac_overshoot_stops_scan(self):
        priv = b"fLaC" + b"\x00\x00\x00\x22" + bytes(34)
        assert _flac_headers(priv) == [4]  # second header is inside the block

    def test_xiph_lengths(self):
        assert _xiph_lengths(VORBIS_PRIV) == [1, 2]

    def test_modes_flac(self):
        elements = parse_webm(FLAC_MKV)
        assert _modes(_find_all(elements, TRACK_ENTRY)[0]) == ["flac", "comp"]

    def test_modes_vorbis(self):
        elements = parse_webm(VORBIS_MKV)
        assert _modes(_find_all(elements, TRACK_ENTRY)[0]) == ["xiph", "comp"]

    def test_modes_comp_in_place(self):
        elements = parse_webm(COMP_MKV)
        # ContentEncodings present with a 0x81-sized CompAlgo: no injection.
        assert _modes(_find_all(elements, TRACK_ENTRY)[0]) == ["flac", "comp"]

    def test_comp_algo_bytes(self):
        # id vints: 42 54 (algo), size 81, value; wrapped in 50 34 / 22 40 / 62 40.
        assert _build_comp_algo(1).startswith(b"\x62\x40")
        assert b"\x50\x34\x84\x42\x54\x81\x01" in _build_comp_algo(1)


class TestMutate:
    def test_flac_block_length_end_exact(self):
        # track/mode 0 (flac), header 0 (STREAMINFO), edge 0: its length is
        # rewritten so the block ends exactly at the CodecPrivate boundary
        # (the CVE-2026-17072 4-byte over-read on the next header).
        out = mutate_mkv_codec_cfg(FLAC_MKV, rng=ScriptedRng(randints=[0, 0, 0]))
        priv = _codec_private(out)
        assert int.from_bytes(priv[5:8], "big") == len(priv) - 8
        assert len(out) == len(FLAC_MKV)  # in place, no resize

    def test_flac_block_length_short_tail(self):
        # track/mode 0, header 1, edge 2: leaves 2 trailing bytes, the loop
        # enters with fewer than 4 bytes left and reads past the buffer.
        out = mutate_mkv_codec_cfg(FLAC_MKV, rng=ScriptedRng(randints=[0, 1, 2]))
        priv = _codec_private(out)
        assert int.from_bytes(priv[43:46], "big") == 8

    def test_xiph_count_edge(self):
        # track/mode 0 (xiph), count path, edge 3 (255 headers).
        out = mutate_mkv_codec_cfg(VORBIS_MKV, rng=ScriptedRng(randints=[0, 1, 3]))
        assert _codec_private(out)[0] == XIPH_COUNT_EDGES[3]

    def test_xiph_length_edge(self):
        # track/mode 0 (xiph), length path, position 1, edge 3 (255).
        out = mutate_mkv_codec_cfg(VORBIS_MKV, rng=ScriptedRng(randints=[0, 0, 1, 3]))
        assert _codec_private(out)[2] == 0xFF

    def test_comp_rewrite_in_place(self):
        # track/mode 1 (comp on the encodings track), algo edge 0: bz2.
        out = mutate_mkv_codec_cfg(COMP_MKV, rng=ScriptedRng(randints=[1, 0]))
        assert b"\x42\x54\x81" + bytes([ALGO_EDGES[0]]) in out
        assert len(out) == len(COMP_MKV)  # value rewrite, no resize

    def test_comp_injection(self):
        # track/mode 1 (comp on the plain FLAC track), algo edge 1 (bz2).
        out = mutate_mkv_codec_cfg(FLAC_MKV, rng=ScriptedRng(randints=[1, 1]))
        assert b"\x62\x40" in out and b"\x42\x54\x81" + bytes([ALGO_EDGES[1]]) in out
        assert len(out) > len(FLAC_MKV)  # the subtree was spliced in
        # parent size vints recomputed: the file still round-trips
        elements = parse_webm(out)
        assert elements is not None
        assert serialize_webm(elements) == out

    # Falsification: no EBML structure, or no track to edit.
    def test_not_matroska_declines(self):
        assert mutate_mkv_codec_cfg(b"junk" + bytes(64), rng=ScriptedRng()) is None

    def test_trackless_segment_declines(self):
        header = _mkv([])  # a Segment with an empty Tracks box
        assert mutate_mkv_codec_cfg(header, rng=ScriptedRng()) is None

    # Adversarial: every truncation of a valid file never raises.
    def test_truncations_never_raise(self):
        rng = ScriptedRng(randints=[0] * 800)
        for cut in range(len(FLAC_MKV) + 1):
            out = mutate_mkv_codec_cfg(FLAC_MKV[:cut], rng=rng)
            assert out is None or isinstance(out, bytes)


class TestRegistry:
    def test_sniffer(self):
        assert sniff_mkv_codec_cfg(FLAC_MKV)
        assert sniff_mkv_codec_cfg(VORBIS_MKV)
        assert sniff_mkv_codec_cfg(COMP_MKV)
        assert not sniff_mkv_codec_cfg(b"\x1a\x45\xdf\xa3" + bytes(64) + b"junk")

    def test_gate(self):
        assert format_gate_matches("mkv_codec_cfg_mutate", FLAC_MKV)
        assert not format_gate_matches("mkv_codec_cfg_mutate", bytes(64))

    def test_category(self):
        assert "mkv_codec_cfg_mutate" in OPERATOR_CATEGORIES["format"]


@pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")
class TestFfmpeg:
    def _encode(self, tmp_path) -> bytes:
        path = tmp_path / "flac.mkv"
        cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3"]
        cmd += ["-c:a", "flac", "-f", "matroska", str(path)]
        subprocess.run(cmd, check=True)
        return path.read_bytes()

    def test_mutated_flac_mkv_opens(self, tmp_path):
        # The demuxer must still walk the EBML tree after the edit (FFmpeg's
        # own Matroska demuxer consumes the same leaves); a minimal hand-built
        # Tracks-only fixture cannot even be opened, so a real one is used.
        # The real file holds one STREAMINFO block whose end-exact edge is its
        # own value, so the short-tail edge (leave 1 byte: the CVE-2026-17072
        # <4-bytes-left loop entry) is scripted.
        data = self._encode(tmp_path)
        out = mutate_mkv_codec_cfg(data, rng=ScriptedRng(randints=[0, 0, 1]))
        assert out is not None
        path = tmp_path / "mutated.mkv"
        path.write_bytes(out)
        cmd = [FFMPEG, "-v", "error", "-i", str(path), "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0
