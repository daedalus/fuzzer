"""Tests for the H.264 SPS/PPS field mutator (`h264_ps_mutate`)."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from fuzzer_tool.core.mutations.bitrec import Field, Kind, Tail, pack
from fuzzer_tool.core.mutations.h264_ps import (
    SE_EDGES,
    UE_EDGES,
    escape,
    find_ps_units,
    mutate_h264_ps,
    parse_ps,
    unescape,
)
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
SPS, PPS = 7, 8
SC = b"\x00\x00\x00\x01"


def _f(name, kind, value, width=0):
    return Field(name, kind, width, value)


U, UE, SE = Kind.FIXED, Kind.UE, Kind.SE

# Baseline SPS, 4x3 macroblocks (64x48), POC type 0, no VUI.
BASELINE = [
    _f("profile_idc", U, 66, 8),
    _f("constraint_flags", U, 0xC0, 8),
    _f("level_idc", U, 30, 8),
    _f("seq_parameter_set_id", UE, 0),
    _f("log2_max_frame_num_minus4", UE, 0),
    _f("pic_order_cnt_type", UE, 0),
    _f("log2_max_pic_order_cnt_lsb_minus4", UE, 2),
    _f("max_num_ref_frames", UE, 1),
    _f("gaps_in_frame_num_allowed_flag", U, 0, 1),
    _f("pic_width_in_mbs_minus1", UE, 3),
    _f("pic_height_in_map_units_minus1", UE, 2),
    _f("frame_mbs_only_flag", U, 1, 1),
    _f("direct_8x8_inference_flag", U, 1, 1),
    _f("frame_cropping_flag", U, 0, 1),
    _f("vui_parameters_present_flag", U, 0, 1),
]
PPS_FIELDS = [
    _f("pic_parameter_set_id", UE, 0),
    _f("seq_parameter_set_id", UE, 0),
    _f("entropy_coding_mode_flag", U, 0, 1),
    _f("bottom_field_pic_order_in_frame_present_flag", U, 0, 1),
    _f("num_slice_groups_minus1", UE, 0),
    _f("num_ref_idx_l0_default_active_minus1", UE, 0),
    _f("num_ref_idx_l1_default_active_minus1", UE, 0),
    _f("weighted_pred_flag", U, 0, 1),
    _f("weighted_bipred_idc", U, 0, 2),
    _f("pic_init_qp_minus26", SE, -3),
    _f("pic_init_qs_minus26", SE, 0),
    _f("chroma_qp_index_offset", SE, 0),
    _f("deblocking_filter_control_present_flag", U, 1, 1),
    _f("constrained_intra_pred_flag", U, 0, 1),
    _f("redundant_pic_cnt_present_flag", U, 0, 1),
]
SPS_RBSP = pack(BASELINE, "", Tail.RBSP)
PPS_RBSP = pack(PPS_FIELDS, "", Tail.RBSP)
STREAM = SC + b"\x67" + escape(SPS_RBSP) + SC + b"\x68" + escape(PPS_RBSP) + SC + b"\x65\x88\x84"


def _names(fields):
    return [f.name for f in fields]


def _value(data: bytes, unit: int, name: str) -> int:
    nal_type, start, end = find_ps_units(data)[unit]
    fields, _ = parse_ps(nal_type, unescape(data[start:end]))
    return {f.name: f.value for f in fields}[name]


class TestEmulationPrevention:
    def test_unescape(self):
        assert unescape(b"\x00\x00\x03\x01\x00\x00\x03\x00") == b"\x00\x00\x01\x00\x00\x00"

    def test_escape(self):
        # The trailing zero also gets a 03 (H.264 7.4.1).
        assert escape(b"\x00\x00\x01\x00\x00\x00") == b"\x00\x00\x03\x01\x00\x00\x03\x00\x03"

    def test_escape_trailing_zero(self):
        assert escape(b"\x05\x00") == b"\x05\x00\x03"

    def test_round_trip(self):
        raw = bytes([0, 0, 0, 0, 0, 3, 0, 0, 2, 7])
        assert unescape(escape(raw)) == raw


class TestParse:
    def test_baseline_sps(self):
        fields, rest = parse_ps(SPS, SPS_RBSP)
        assert fields == BASELINE
        assert rest == ""

    def test_pps(self):
        assert parse_ps(PPS, PPS_RBSP) == (PPS_FIELDS, "")

    def test_high_profile_adds_chroma_fields(self):
        high = [_f("profile_idc", U, 100, 8), *BASELINE[1:4]]
        high += [
            _f("chroma_format_idc", UE, 1),
            _f("bit_depth_luma_minus8", UE, 0),
            _f("bit_depth_chroma_minus8", UE, 0),
            _f("qpprime_y_zero_transform_bypass_flag", U, 0, 1),
            _f("seq_scaling_matrix_present_flag", U, 0, 1),
            *BASELINE[4:],
        ]
        assert parse_ps(SPS, pack(high, "", Tail.RBSP))[0] == high

    def test_scaling_matrix_stops_parse(self):
        # Scaling lists are kept verbatim as the rest.
        head = [_f("profile_idc", U, 100, 8), *BASELINE[1:4]]
        head += [_f("chroma_format_idc", UE, 1), _f("bit_depth_luma_minus8", UE, 0)]
        head += [_f("bit_depth_chroma_minus8", UE, 0)]
        head += [_f("qpprime_y_zero_transform_bypass_flag", U, 0, 1)]
        head += [_f("seq_scaling_matrix_present_flag", U, 1, 1)]
        fields, rest = parse_ps(SPS, pack(head, "1011", Tail.RBSP))
        assert fields == head
        assert rest == "1011"

    def test_crop_and_vui_rest(self):
        sps = [*BASELINE[:-2], _f("frame_cropping_flag", U, 1, 1)]
        sps += [_f(f"frame_crop_{s}_offset", UE, i) for i, s in enumerate("lrtb")]
        sps += [_f("vui_parameters_present_flag", U, 1, 1)]
        assert parse_ps(SPS, pack(sps, "0110", Tail.RBSP)) == (sps, "0110")

    # Adversarial: truncated RBSP and an exp-Golomb with 32+ zeros.
    def test_truncated(self):
        assert parse_ps(SPS, SPS_RBSP[:4]) is None

    def test_huge_codeword(self):
        assert parse_ps(PPS, bytes(6)) is None

    def test_find_units(self):
        units = find_ps_units(STREAM)
        assert [u[0] for u in units] == [SPS, PPS]
        _, start, end = units[0]
        assert unescape(STREAM[start:end]) == SPS_RBSP


class TestMutate:
    def test_sps_width(self):
        idx = _names(BASELINE).index("pic_width_in_mbs_minus1")
        out = mutate_h264_ps(STREAM, rng=ScriptedRng(randints=[0, idx], choice_idxs=[5]))
        assert _value(out, 0, "pic_width_in_mbs_minus1") == UE_EDGES[5]
        assert _value(out, 0, "pic_height_in_map_units_minus1") == 2
        assert out.endswith(SC + b"\x68" + escape(PPS_RBSP) + SC + b"\x65\x88\x84")

    def test_pps_se_field(self):
        idx = _names(PPS_FIELDS).index("pic_init_qp_minus26")
        out = mutate_h264_ps(STREAM, rng=ScriptedRng(randints=[1, idx], choice_idxs=[6]))
        assert _value(out, 1, "pic_init_qp_minus26") == SE_EDGES[6]

    def test_flag_flip(self):
        idx = _names(BASELINE).index("frame_mbs_only_flag")
        out = mutate_h264_ps(STREAM, rng=ScriptedRng(randints=[0, idx]))
        assert _value(out, 0, "frame_mbs_only_flag") == 0

    def test_huge_ue_is_escaped(self):
        # 2^32 - 2 has 31 leading zeros: zero runs must be emulation-escaped.
        idx = _names(BASELINE).index("seq_parameter_set_id")
        big = UE_EDGES.index(2**32 - 2)
        out = mutate_h264_ps(STREAM, rng=ScriptedRng(randints=[0, idx], choice_idxs=[big]))
        assert [u[0] for u in find_ps_units(out)] == [SPS, PPS]  # no false start code
        assert _value(out, 0, "seq_parameter_set_id") == 2**32 - 2

    # Falsification: slices only, nothing to edit.
    def test_no_parameter_sets_declines(self):
        assert mutate_h264_ps(SC + b"\x65\x88\x84", rng=ScriptedRng()) is None

    def test_unparsable_unit_declines(self):
        data = SC + b"\x67\x00\x00\x03\x00\x00\x03\x00"
        assert mutate_h264_ps(data, rng=ScriptedRng(randints=[0, 0])) is None

    # Adversarial: every truncation never raises.
    def test_truncations_never_raise(self):
        for cut in range(len(STREAM) + 1):
            out = mutate_h264_ps(STREAM[:cut], rng=ScriptedRng(randints=[0, 0], choice_idxs=[0]))
            assert out is None or isinstance(out, bytes)


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("h264_ps_mutate", STREAM)
        assert not format_gate_matches("h264_ps_mutate", SC + b"\x65\x88\x84")
        assert not format_gate_matches("h264_ps_mutate", b"\x00\x00\x01\xba" + STREAM)


@pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="needs ffmpeg")
class TestFfmpeg:
    def _encode(self, tmp_path, pix_fmt: str) -> bytes:
        path = tmp_path / f"{pix_fmt}.h264"
        cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc=d=0.2:s=64x48"]
        cmd += ["-pix_fmt", pix_fmt, "-c:v", "libx264", "-bsf:v", "h264_mp4toannexb", str(path)]
        subprocess.run(cmd, check=True)
        return path.read_bytes()

    @pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuv444p"])
    def test_libx264_round_trip(self, tmp_path, pix_fmt):
        data = self._encode(tmp_path, pix_fmt)
        units = find_ps_units(data)
        assert {u[0] for u in units} == {SPS, PPS}
        for nal_type, start, end in units:
            rbsp = unescape(data[start:end])
            fields, rest = parse_ps(nal_type, rbsp)
            assert pack(fields, rest, Tail.RBSP) == rbsp

    def test_width_edit_reaches_decoder(self, tmp_path):
        data = self._encode(tmp_path, "yuv420p")
        sps = next(i for i, u in enumerate(find_ps_units(data)) if u[0] == SPS)
        _, start, end = find_ps_units(data)[sps]
        names = _names(parse_ps(SPS, unescape(data[start:end]))[0])
        idx = names.index("pic_width_in_mbs_minus1")
        out = mutate_h264_ps(data, rng=ScriptedRng(randints=[sps, idx], choice_idxs=[5]))
        path = tmp_path / "out.h264"
        path.write_bytes(out)
        cmd = [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)]
        info = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
        assert info["streams"][0]["coded_width"] == (UE_EDGES[5] + 1) * 16
