"""Tests for the AAC AudioSpecificConfig mutator (`aac_asc_mutate`)."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from fuzzer_tool.core.mutations.aac_asc import (
    AOT_EDGES,
    find_asc,
    mutate_aac_asc,
    parse_asc,
)
from fuzzer_tool.core.mutations.bitrec import Tail, pack
from fuzzer_tool.core.mutations.mp4_path import find_boxes, parse_esds
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.mp4_build import box, esds, mp4, stsd
from .support.scripted_rng import ScriptedRng

FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
LC_44K_STEREO = bytes([0x12, 0x10])  # aot 2, sfi 4, ch 2
SFI_48K = 3  # ISO/IEC 14496-3 Table 1.18
AAC = mp4(stsd(b"mp4a", esds(LC_44K_STEREO)))


def _bits(fields: list[tuple[int, int]]) -> bytes:
    acc, n = 0, 0
    for value, width in fields:
        acc, n = (acc << width) | value, n + width
    pad = -n % 8
    return (acc << pad).to_bytes((n + pad) // 8, "big")


def _asc(data: bytes) -> bytes:
    (hit,) = find_boxes(data, frozenset({b"esds"}))
    payload = data[hit.start + 8 : hit.start + hit.size]
    dsi = parse_esds(payload).dsi
    return payload[dsi.payload_off : dsi.end]


def _values(asc: bytes) -> dict[str, int]:
    return {f.name: f.value for f in parse_asc(asc)[0]}


def _index(asc: bytes, name: str) -> int:
    return [f.name for f in parse_asc(asc)[0]].index(name)


class TestParse:
    def test_aac_lc(self):
        values = _values(LC_44K_STEREO)
        assert values["audio_object_type"] == 2
        assert values["sampling_frequency_index"] == 4
        assert values["channel_configuration"] == 2
        assert values["frame_length_flag"] == 0

    def test_he_aac_extension(self):
        # aot 5 (SBR), sfi 6, ch 2, ext sfi 3, core aot 2, GA flags.
        asc = _bits([(5, 5), (6, 4), (2, 4), (3, 4), (2, 5), (0, 3)])
        values = _values(asc)
        assert values["extension_sampling_frequency_index"] == 3
        assert values["ext_audio_object_type"] == 2
        assert "frame_length_flag" in values

    def test_escaped_aot_and_explicit_rate(self):
        asc = _bits([(31, 5), (10, 6), (15, 4), (48000, 24), (1, 4)])
        values = _values(asc)
        assert values["aot_ext"] == 10
        assert values["sampling_frequency"] == 48000

    def test_round_trip(self):
        for asc in (LC_44K_STEREO, _bits([(5, 5), (6, 4), (2, 4), (3, 4), (2, 5), (0, 3)])):
            fields, rest = parse_asc(asc)
            assert pack(fields, rest, Tail.NONE) == asc

    # Adversarial: too short to hold the fixed fields.
    def test_truncated(self):
        assert parse_asc(b"\x12") is None


class TestMutate:
    def test_channel_configuration(self):
        idx = _index(LC_44K_STEREO, "channel_configuration")
        out = mutate_aac_asc(AAC, rng=ScriptedRng(randints=[0, idx, 7]))
        assert _values(_asc(out))["channel_configuration"] == 7
        assert len(out) == len(AAC)

    def test_object_type_edge(self):
        # ER AAC LD (23) keeps the GA flags, so the 2-byte ASC still parses.
        idx = _index(LC_44K_STEREO, "audio_object_type")
        rng = ScriptedRng(randints=[0, idx], choice_idxs=[AOT_EDGES.index(23)])
        out = mutate_aac_asc(AAC, rng=rng)
        assert _values(_asc(out))["audio_object_type"] == 23

    # Adversarial: SBR (5) needs 4 more ext bits than a 2-byte ASC has.
    def test_sbr_edge_leaves_truncated_asc(self):
        idx = _index(LC_44K_STEREO, "audio_object_type")
        rng = ScriptedRng(randints=[0, idx], choice_idxs=[AOT_EDGES.index(5)])
        assert parse_asc(_asc(mutate_aac_asc(AAC, rng=rng))) is None

    def test_flag_flip(self):
        idx = _index(LC_44K_STEREO, "frame_length_flag")
        out = mutate_aac_asc(AAC, rng=ScriptedRng(randints=[0, idx]))
        assert _values(_asc(out))["frame_length_flag"] == 1

    def test_find_asc(self):
        ((hit, layout),) = find_asc(AAC)
        assert layout.dsi is not None

    # Falsification: esds without DecoderSpecificInfo has nothing to edit.
    def test_no_dsi_declines(self):
        es = b"\x03\x10\x00\x01\x00\x04\x0d\x40\x15" + bytes(11)
        data = mp4(stsd(b"mp4a", box(b"esds", bytes(4) + es)))
        assert mutate_aac_asc(data, rng=ScriptedRng()) is None

    def test_truncations_never_raise(self):
        for cut in range(len(AAC) + 1):
            out = mutate_aac_asc(AAC[:cut], rng=ScriptedRng(randints=[0, 0], choice_idxs=[0]))
            assert out is None or isinstance(out, bytes)


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("aac_asc_mutate", AAC)
        assert not format_gate_matches("aac_asc_mutate", mp4(box(b"stsd", bytes(8))))


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="needs ffmpeg")
def test_rate_edit_reaches_ffprobe(tmp_path):
    # Channels are re-derived from decoded frames; the rate comes from the ASC.
    src = tmp_path / "in.mp4"
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=d=0.2:sample_rate=44100"]
    subprocess.run([*cmd, "-c:a", "aac", str(src)], check=True)
    data = src.read_bytes()

    idx = _index(_asc(data), "sampling_frequency_index")
    out = tmp_path / "out.mp4"
    out.write_bytes(mutate_aac_asc(data, rng=ScriptedRng(randints=[0, idx, SFI_48K])))
    probe = [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(out)]
    info = json.loads(subprocess.run(probe, capture_output=True, text=True).stdout)
    assert info["streams"][0]["sample_rate"] == "48000"
