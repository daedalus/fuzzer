"""Tests for the HEVC VPS/SPS/PPS field mutator (`hevc_ps_mutate`)."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from fuzzer_tool.core.mutations.bitrec import Field, Kind, Tail, pack
from fuzzer_tool.core.mutations.h264_ps import SE_EDGES, UE_EDGES, escape
from fuzzer_tool.core.mutations.hevc_ps import (
    find_ps_units,
    mutate_hevc_ps,
    parse_ps,
    unescape,
)
from fuzzer_tool.core.operator_categories import OPERATOR_CATEGORIES
from fuzzer_tool.core.operator_registry import format_gate_matches

from .support.scripted_rng import ScriptedRng

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _has_libx265() -> bool:
    if FFMPEG is None:
        return False
    out = subprocess.run(
        [FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    return "libx265" in out


LIBX265 = _has_libx265()
VPS, SPS, PPS = 32, 33, 34
SC = b"\x00\x00\x00\x01"


def _f(name, kind, value, width=0):
    return Field(name, kind, width, value)


U, UE, SE = Kind.FIXED, Kind.UE, Kind.SE

# profile_tier_level(1, 0), Main profile level 1 (see the libx265 round-trip).
PTL = [
    _f("general_profile_space", U, 0, 2),
    _f("general_tier_flag", U, 0, 1),
    _f("general_profile_idc", U, 1, 5),
    _f("general_profile_compatibility_flags", U, 0x60000000, 32),
    _f("general_progressive_source_flag", U, 1, 1),
    _f("general_interlaced_source_flag", U, 0, 1),
    _f("general_non_packed_constraint_flag", U, 0, 1),
    _f("general_frame_only_constraint_flag", U, 1, 1),
    _f("general_reserved_44bits", U, 0, 44),
    _f("general_level_idc", U, 30, 8),
]
# Baseline VPS, single sub-layer, no timing info.
VPS_FIELDS = [
    _f("vps_video_parameter_set_id", U, 0, 4),
    _f("vps_base_layer_internal_flag", U, 1, 1),
    _f("vps_base_layer_available_flag", U, 1, 1),
    _f("vps_max_layers_minus1", U, 0, 6),
    _f("vps_max_sub_layers_minus1", U, 0, 3),
    _f("vps_temporal_id_nesting_flag", U, 1, 1),
    _f("vps_reserved_0xffff_16bits", U, 0xFFFF, 16),
    *PTL,
    _f("vps_sub_layer_ordering_info_present_flag", U, 1, 1),
    _f("vps_max_dec_pic_buffering_minus1", UE, 4),
    _f("vps_max_num_reorder_pics", UE, 2),
    _f("vps_max_latency_increase_plus1", UE, 4),
    _f("vps_max_layer_id", U, 0, 6),
    _f("vps_num_layer_sets_minus1", UE, 0),
    _f("vps_timing_info_present_flag", U, 0, 1),
    _f("vps_extension_flag", U, 0, 1),
]
# Baseline SPS, 64x48 4:2:0, no window/scaling lists/RPS/VUI.
SPS_FIELDS = [
    _f("sps_video_parameter_set_id", U, 0, 4),
    _f("sps_max_sub_layers_minus1", U, 0, 3),
    _f("sps_temporal_id_nesting_flag", U, 1, 1),
    *PTL,
    _f("sps_seq_parameter_set_id", UE, 0),
    _f("chroma_format_idc", UE, 1),
    _f("pic_width_in_luma_samples", UE, 64),
    _f("pic_height_in_luma_samples", UE, 48),
    _f("conformance_window_flag", U, 0, 1),
    _f("bit_depth_luma_minus8", UE, 0),
    _f("bit_depth_chroma_minus8", UE, 0),
    _f("log2_max_pic_order_cnt_lsb_minus4", UE, 4),
    _f("sps_sub_layer_ordering_info_present_flag", U, 1, 1),
    _f("sps_max_dec_pic_buffering_minus1", UE, 4),
    _f("sps_max_num_reorder_pics", UE, 2),
    _f("sps_max_latency_increase_plus1", UE, 4),
    _f("log2_min_luma_coding_block_size_minus3", UE, 1),
    _f("log2_diff_max_min_luma_coding_block_size", UE, 1),
    _f("log2_min_luma_transform_block_size_minus2", UE, 0),
    _f("log2_diff_max_min_luma_transform_block_size", UE, 3),
    _f("max_transform_hierarchy_depth_inter", UE, 0),
    _f("max_transform_hierarchy_depth_intra", UE, 0),
    _f("scaling_list_enabled_flag", U, 0, 1),
    _f("amp_enabled_flag", U, 0, 1),
    _f("sample_adaptive_offset_enabled_flag", U, 0, 1),
    _f("pcm_enabled_flag", U, 0, 1),
    _f("num_short_term_ref_pic_sets", UE, 0),
    _f("long_term_ref_pics_present_flag", U, 0, 1),
    _f("sps_temporal_mvp_enabled_flag", U, 1, 1),
    _f("strong_intra_smoothing_enabled_flag", U, 1, 1),
    _f("vui_parameters_present_flag", U, 0, 1),
    _f("sps_extension_present_flag", U, 0, 1),
]
# Baseline PPS: CABAC, cu_qp_delta on, no tiles/WPP/deblocking control.
PPS_FIELDS = [
    _f("pps_pic_parameter_set_id", UE, 0),
    _f("pps_seq_parameter_set_id", UE, 0),
    _f("dependent_slice_segments_enabled_flag", U, 0, 1),
    _f("output_flag_present_flag", U, 0, 1),
    _f("num_extra_slice_header_flags", U, 0, 3),
    _f("sign_data_hiding_enabled_flag", U, 0, 1),
    _f("cabac_init_present_flag", U, 0, 1),
    _f("num_ref_idx_l0_default_active_minus1", UE, 0),
    _f("num_ref_idx_l1_default_active_minus1", UE, 0),
    _f("init_qp_minus26", SE, 0),
    _f("constrained_intra_pred_flag", U, 0, 1),
    _f("transform_skip_enabled_flag", U, 0, 1),
    _f("cu_qp_delta_enabled_flag", U, 1, 1),
    _f("diff_cu_qp_delta_depth", UE, 0),
    _f("pps_cb_qp_offset", SE, 0),
    _f("pps_cr_qp_offset", SE, 0),
    _f("pps_slice_chroma_qp_offsets_present_flag", U, 0, 1),
    _f("weighted_pred_flag", U, 0, 1),
    _f("weighted_bipred_flag", U, 0, 1),
    _f("transquant_bypass_enabled_flag", U, 0, 1),
    _f("tiles_enabled_flag", U, 0, 1),
    _f("entropy_coding_sync_enabled_flag", U, 0, 1),
    _f("pps_loop_filter_across_slices_enabled_flag", U, 1, 1),
    _f("deblocking_filter_control_present_flag", U, 0, 1),
    _f("pps_scaling_list_data_present_flag", U, 0, 1),
    _f("lists_modification_present_flag", U, 0, 1),
    _f("log2_parallel_merge_level_minus2", UE, 0),
    _f("slice_segment_header_extension_present_flag", U, 0, 1),
    _f("pps_extension_present_flag", U, 0, 1),
]
VPS_RBSP = pack(VPS_FIELDS, "", Tail.RBSP)
SPS_RBSP = pack(SPS_FIELDS, "", Tail.RBSP)
PPS_RBSP = pack(PPS_FIELDS, "", Tail.RBSP)
STREAM = (
    SC
    + b"\x40\x01"
    + escape(VPS_RBSP)
    + SC
    + b"\x42\x01"
    + escape(SPS_RBSP)
    + SC
    + b"\x44\x01"
    + escape(PPS_RBSP)
    + SC
    + b"\x26\x01\xaf"  # IDR_W_RADL slice
)


def _names(fields):
    return [f.name for f in fields]


def _value(data: bytes, unit: int, name: str) -> int:
    nal_type, start, end = find_ps_units(data)[unit]
    fields, _ = parse_ps(nal_type, unescape(data[start:end]))
    return {f.name: f.value for f in fields}[name]


class TestParse:
    def test_vps(self):
        assert parse_ps(VPS, VPS_RBSP) == (VPS_FIELDS, "")

    def test_sps(self):
        assert parse_ps(SPS, SPS_RBSP) == (SPS_FIELDS, "")

    def test_pps(self):
        assert parse_ps(PPS, PPS_RBSP) == (PPS_FIELDS, "")

    def test_chroma_444_adds_plane_flag(self):
        idx = _names(SPS_FIELDS).index("chroma_format_idc")
        sps = SPS_FIELDS[:idx] + [_f("chroma_format_idc", UE, 3)]
        sps += [_f("separate_colour_plane_flag", U, 1, 1), *SPS_FIELDS[idx + 1 :]]
        assert parse_ps(SPS, pack(sps, "", Tail.RBSP))[0] == sps

    def test_vps_sub_layers_stop(self):
        vps = [*VPS_FIELDS[:4], _f("vps_max_sub_layers_minus1", U, 1, 3)]
        fields, rest = parse_ps(VPS, pack(vps, "1011", Tail.RBSP))
        assert fields == vps
        assert rest == "1011"

    def test_scaling_list_stops_parse(self):
        idx = _names(SPS_FIELDS).index("scaling_list_enabled_flag")
        sps = [*SPS_FIELDS[:idx], _f("scaling_list_enabled_flag", U, 1, 1)]
        fields, rest = parse_ps(SPS, pack(sps, "0110", Tail.RBSP))
        assert fields == sps
        assert rest == "0110"

    def test_conformance_window(self):
        idx = _names(SPS_FIELDS).index("conformance_window_flag")
        sps = SPS_FIELDS[:idx] + [_f("conformance_window_flag", U, 1, 1)]
        sps += [_f(f"conf_win_{s}_offset", UE, i) for i, s in enumerate("lrtb")]
        sps += SPS_FIELDS[idx + 1 :]
        assert parse_ps(SPS, pack(sps, "", Tail.RBSP))[0] == sps

    # Adversarial: truncated RBSP, a >32-bit codeword and an unknown NAL type.
    def test_truncated(self):
        assert parse_ps(VPS, VPS_RBSP[:4]) is None

    def test_huge_codeword(self):
        assert parse_ps(PPS, bytes(6)) is None

    def test_unknown_type(self):
        assert parse_ps(19, SPS_RBSP) is None

    def test_find_units(self):
        units = find_ps_units(STREAM)
        assert [u[0] for u in units] == [VPS, SPS, PPS]
        assert unescape(STREAM[units[1][1] : units[1][2]]) == SPS_RBSP


class TestMutate:
    def test_sps_width(self):
        idx = _names(SPS_FIELDS).index("pic_width_in_luma_samples")
        out = mutate_hevc_ps(STREAM, rng=ScriptedRng(randints=[1, idx, 5]))
        assert _value(out, 1, "pic_width_in_luma_samples") == UE_EDGES[5]
        assert _value(out, 1, "pic_height_in_luma_samples") == 48
        assert out.endswith(SC + b"\x44\x01" + escape(PPS_RBSP) + SC + b"\x26\x01\xaf")

    def test_pps_se_field(self):
        idx = _names(PPS_FIELDS).index("init_qp_minus26")
        out = mutate_hevc_ps(STREAM, rng=ScriptedRng(randints=[2, idx, 6]))
        assert _value(out, 2, "init_qp_minus26") == SE_EDGES[6]

    def test_vps_fixed_field(self):
        idx = _names(VPS_FIELDS).index("vps_reserved_0xffff_16bits")
        out = mutate_hevc_ps(STREAM, rng=ScriptedRng(randints=[0, idx, 0x1234]))
        assert _value(out, 0, "vps_reserved_0xffff_16bits") == 0x1234

    def test_flag_flip(self):
        idx = _names(PPS_FIELDS).index("pps_loop_filter_across_slices_enabled_flag")
        out = mutate_hevc_ps(STREAM, rng=ScriptedRng(randints=[2, idx]))
        assert _value(out, 2, "pps_loop_filter_across_slices_enabled_flag") == 0

    def test_huge_ue_is_escaped(self):
        # 2^32 - 2 has 31 leading zeros: zero runs must be emulation-escaped.
        idx = _names(SPS_FIELDS).index("sps_seq_parameter_set_id")
        big = UE_EDGES.index(2**32 - 2)
        out = mutate_hevc_ps(STREAM, rng=ScriptedRng(randints=[1, idx, big]))
        assert [u[0] for u in find_ps_units(out)] == [VPS, SPS, PPS]  # no false start code
        assert _value(out, 1, "sps_seq_parameter_set_id") == 2**32 - 2

    # Falsification: slices only, nothing to edit.
    def test_no_parameter_sets_declines(self):
        assert mutate_hevc_ps(SC + b"\x26\x01\xaf", rng=ScriptedRng()) is None

    def test_unparsable_unit_declines(self):
        data = SC + b"\x40\x01\x00\x00\x03\x00\x00\x03\x00"
        assert mutate_hevc_ps(data, rng=ScriptedRng(randints=[0, 0])) is None

    # Adversarial: every truncation never raises.
    def test_truncations_never_raise(self):
        rng = ScriptedRng(randints=[0] * 200, choice_idxs=[0] * 200)
        for cut in range(len(STREAM) + 1):
            out = mutate_hevc_ps(STREAM[:cut], rng=rng)
            assert out is None or isinstance(out, bytes)


class TestRegistry:
    def test_sniffer(self):
        assert format_gate_matches("hevc_ps_mutate", STREAM)
        assert not format_gate_matches("hevc_ps_mutate", SC + b"\x26\x01\xaf")
        assert not format_gate_matches("hevc_ps_mutate", b"\x00\x00\x01\xba" + STREAM)

    def test_category(self):
        assert {"h264_ps_mutate", "hevc_ps_mutate"} <= OPERATOR_CATEGORIES["format"]


@pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None or not LIBX265, reason="needs ffmpeg + libx265"
)
class TestFfmpeg:
    def _encode(self, tmp_path, pix_fmt: str) -> bytes:
        path = tmp_path / f"{pix_fmt}.h265"
        cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc=d=0.2:s=64x48"]
        cmd += [
            "-pix_fmt",
            pix_fmt,
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-f",
            "hevc",
            str(path),
        ]
        subprocess.run(cmd, check=True)
        return path.read_bytes()

    @pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuv444p"])
    def test_libx265_round_trip(self, tmp_path, pix_fmt):
        data = self._encode(tmp_path, pix_fmt)
        units = find_ps_units(data)
        assert {u[0] for u in units} == {VPS, SPS, PPS}
        for nal_type, start, end in units:
            rbsp = unescape(data[start:end])
            fields, rest = parse_ps(nal_type, rbsp)
            assert pack(fields, rest, Tail.RBSP) == rbsp

    def test_width_edit_reaches_decoder(self, tmp_path):
        data = self._encode(tmp_path, "yuv420p")
        sps = next(i for i, u in enumerate(find_ps_units(data)) if u[0] == SPS)
        _, start, end = find_ps_units(data)[sps]
        names = _names(parse_ps(SPS, unescape(data[start:end]))[0])
        idx = names.index("pic_width_in_luma_samples")
        edge = UE_EDGES.index(16)  # a min-CU-aligned edge width
        out = mutate_hevc_ps(data, rng=ScriptedRng(randints=[sps, idx, edge]))
        path = tmp_path / "out.h265"
        path.write_bytes(out)
        cmd = [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)]
        info = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
        assert info["streams"][0]["coded_width"] == 16
