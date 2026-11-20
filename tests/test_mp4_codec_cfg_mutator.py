"""Tests for the avcC / hvcC / esds field mutator (`mp4_codec_cfg_mutate`)."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess

import pytest

from fuzzer_tool.core.mutations.h264_ps import UE_EDGES, parse_ps, unescape
from fuzzer_tool.core.mutations.mp4_codec_cfg import (
    LEN_SIZE_EDGES,
    OTI_EDGES,
    Op,
    config_sites,
    mutate_mp4_cfg,
)
from fuzzer_tool.core.mutations.mp4_path import find_boxes, parse_esds
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.mp4_build import avcc, box, esds, mp4, stsd
from .support.scripted_rng import ScriptedRng

FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
CFG = frozenset({b"avcC", b"hvcC", b"esds"})
SPS = b"\x67\x42\xc0\x1e\xed\x08\xf2"  # baseline 64x48 (see test_h264_ps_mutator)
PPS = b"\x68\xce\x0f\xc8"
AVC = mp4(stsd(b"avc1", avcc(SPS, PPS)))
AAC = mp4(stsd(b"mp4a", esds(b"\x12\x10")))


def _payload(data: bytes, typ: bytes) -> bytes:
    (hit,) = find_boxes(data, frozenset({typ}))
    return data[hit.start + 8 : hit.start + hit.size]


def _site_index(data: bytes, typ: bytes, off: int) -> int:
    sites, _ = config_sites(typ, _payload(data, typ))
    return [s.off for s in sites].index(off)


def _hvcc() -> bytes:
    rec = bytearray(23)
    rec[0], rec[21], rec[22] = 1, 0xFF, 1
    rec += bytes([0x21]) + struct.pack(">H", 1) + struct.pack(">H", 3) + b"\x42\x01\x01"
    return box(b"hvcC", bytes(rec))


class TestSites:
    def test_avcc(self):
        sites, spans = config_sites(b"avcC", _payload(AVC, b"avcC"))
        assert [s.off for s in sites] == [0, 1, 2, 3, 4, 5, 6, 6 + 2 + len(SPS), 9 + len(SPS)]
        assert spans == [(6, len(SPS)), (9 + len(SPS), len(PPS))]

    def test_hvcc(self):
        sites, spans = config_sites(b"hvcC", _payload(mp4(stsd(b"avc1", _hvcc())), b"hvcC"))
        assert [s.off for s in sites] == [0, 1, 12, 21, 22, 23, 24, 26]
        assert spans == []

    def test_esds(self):
        payload = _payload(AAC, b"esds")
        layout = parse_esds(payload)
        offs = [s.off for s in config_sites(b"esds", payload)[0]]
        assert layout.dc.payload_off in offs
        assert layout.dsi.len_off in offs

    # Adversarial: a lying SPS count/length stops the walk, never raises.
    def test_truncated_avcc(self):
        sites, spans = config_sites(b"avcC", b"\x01\x42\xc0\x1e\xff\xff\x00\x40")
        assert spans == []
        assert len(sites) >= 6


class TestMutate:
    def test_avcc_length_size(self):
        idx = _site_index(AVC, b"avcC", 4)
        rng = ScriptedRng(randints=[Op.FIELD, 0, idx], choice_idxs=[2])
        out = mutate_mp4_cfg(AVC, rng=rng)
        assert _payload(out, b"avcC")[4] == LEN_SIZE_EDGES[2]
        assert len(out) == len(AVC)

    def test_esds_object_type(self):
        payload = _payload(AAC, b"esds")
        idx = _site_index(AAC, b"esds", parse_esds(payload).dc.payload_off)
        out = mutate_mp4_cfg(AAC, rng=ScriptedRng(randints=[Op.FIELD, 0, idx], choice_idxs=[3]))
        new = _payload(out, b"esds")
        assert new[parse_esds(new).dc.payload_off] == OTI_EDGES[3]

    def test_ps_edit_rewrites_length_and_boxes(self):
        fields, _ = parse_ps(7, unescape(SPS[1:]))
        idx = [f.name for f in fields].index("pic_width_in_mbs_minus1")
        big = UE_EDGES.index(65535)  # 31-bit codeword: the SPS grows
        rng = ScriptedRng(randints=[Op.PS_EDIT, 0, 0, idx], choice_idxs=[big])
        out = mutate_mp4_cfg(AVC, rng=rng)

        rec = _payload(out, b"avcC")
        sps_len = struct.unpack_from(">H", rec, 6)[0]
        assert sps_len > len(SPS)
        new_fields, _ = parse_ps(7, unescape(rec[9 : 8 + sps_len]))
        assert {f.name: f.value for f in new_fields}["pic_width_in_mbs_minus1"] == 65535
        assert rec[8 + sps_len :] == b"\x01" + struct.pack(">H", len(PPS)) + PPS

    def test_ps_edit_needs_avcc(self):
        assert mutate_mp4_cfg(AAC, rng=ScriptedRng(randints=[Op.PS_EDIT, 0, 0, 0])) is None

    # Falsification: no config box at all.
    def test_plain_mp4_declines(self):
        assert mutate_mp4_cfg(mp4(box(b"stsd", bytes(8))), rng=ScriptedRng()) is None

    def test_truncations_never_raise(self):
        for cut in range(0, len(AVC) + 1, 3):
            for op in Op:
                rng = ScriptedRng(randints=[op, 0, 0, 0], choice_idxs=[0, 0])
                out = mutate_mp4_cfg(AVC[:cut], rng=rng)
                assert out is None or isinstance(out, bytes)


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("mp4_codec_cfg_mutate", AVC)
        assert format_gate_matches("mp4_codec_cfg_mutate", AAC)
        assert not format_gate_matches("mp4_codec_cfg_mutate", mp4(box(b"stsd", bytes(8))))


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="needs ffmpeg")
def test_ps_edit_reaches_ffprobe(tmp_path):
    src = tmp_path / "in.mp4"
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc=d=0.2:s=64x48"]
    subprocess.run([*cmd, "-pix_fmt", "yuv420p", "-c:v", "libx264", str(src)], check=True)
    data = src.read_bytes()

    rec = _payload(data, b"avcC")
    sps_len = struct.unpack_from(">H", rec, 6)[0]
    fields, _ = parse_ps(7, unescape(rec[9 : 8 + sps_len]))
    idx = [f.name for f in fields].index("pic_width_in_mbs_minus1")
    rng = ScriptedRng(randints=[Op.PS_EDIT, 0, 0, idx], choice_idxs=[5])
    out_path = tmp_path / "out.mp4"
    out_path.write_bytes(mutate_mp4_cfg(data, rng=rng))

    probe = [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(out_path)]
    info = json.loads(subprocess.run(probe, capture_output=True, text=True).stdout)
    assert info["streams"][0]["coded_width"] == (UE_EDGES[5] + 1) * 16
