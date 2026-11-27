"""HEVC (H.265) VPS/SPS/PPS field mutator (FFmpeg ``hevc_ps.c``).

``nal_chunk_mutate`` flips random bits in parameter sets; a flip inside an
exp-Golomb codeword re-frames every later field, so most edits land on
garbage. This op parses the RBSP into fields and rewrites one:

  Annex B unit ──unescape──► RBSP ──parse──► [ptl][dims][depths] ... | rest
                                              edit one field (edge value)
  spliced back ◄──escape──── re-pack (fields + rest + stop bit)

VPS stops at extra sub-layers / layer sets / timing / extension, SPS at a
scaling list, PCM, RPS set, long-term refs, VUI or extension, PPS at tiles /
WPP / deblocking control / scaling lists / extension; the rest is kept
verbatim. Edges: ids past 31/255, 2^32-2 sizes, QP past +-26.
``unescape``/``escape``/edges are shared with ``h264_ps`` (also used by avcC).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from fuzzer_tool.core.mutations.bitrec import Field, FieldParser, Kind, Tail, pack
from fuzzer_tool.core.mutations.h264_ps import (
    SE_EDGES,
    U8_EDGES,
    UE_EDGES,
    escape,
    unescape,
)
from fuzzer_tool.core.rand_pool import RandPool

VPS_TYPE, SPS_TYPE, PPS_TYPE = 32, 33, 34
NAL_TYPE_MASK = 0x3F
FORBIDDEN_BIT = 0x80
START_CODE = b"\x00\x00\x01"
CHROMA_444 = 3

# profile_tier_level(1, 0): profile + compatibility + constraints + level.
# The 44-bit constraint block is one field in both branches of the spec
# (10 flags + 34 reserved, or all reserved), so its width is constant.
_PTL_BITS = 44

# PS unit in an Annex B stream: start code + a 0x40..0x45 header byte.
# That covers VPS/SPS/PPS (types 32/33/34, forbidden bit clear) with the
# high bit of nuh_layer_id (0 in every real stream) either way.
_PS_UNIT = re.compile(rb"\x00\x00\x01[\x40-\x45]")


def _ptl(p: FieldParser) -> None:
    """profile_tier_level(1, 0) general fields, fixed-width."""
    p.u("general_profile_space", 2)
    p.u("general_tier_flag", 1)
    p.u("general_profile_idc", 5)
    p.u("general_profile_compatibility_flags", 32)
    p.u("general_progressive_source_flag", 1)
    p.u("general_interlaced_source_flag", 1)
    p.u("general_non_packed_constraint_flag", 1)
    p.u("general_frame_only_constraint_flag", 1)
    p.u("general_reserved_44bits", _PTL_BITS)
    p.u("general_level_idc", 8)


def _order_info(p: FieldParser, prefix: str) -> None:
    """The one sub-layer ordering triple parsed with max_sub_layers 0."""
    p.ue(f"{prefix}_max_dec_pic_buffering_minus1")
    p.ue(f"{prefix}_max_num_reorder_pics")
    p.ue(f"{prefix}_max_latency_increase_plus1")


def _vps(p: FieldParser) -> None:
    p.u("vps_video_parameter_set_id", 4)
    p.u("vps_base_layer_internal_flag", 1)
    p.u("vps_base_layer_available_flag", 1)
    p.u("vps_max_layers_minus1", 6)
    if p.u("vps_max_sub_layers_minus1", 3):
        return  # sub-layer PTL/ordering: deeper than the parse goes
    p.u("vps_temporal_id_nesting_flag", 1)
    p.u("vps_reserved_0xffff_16bits", 16)
    _ptl(p)
    if p.u("vps_sub_layer_ordering_info_present_flag", 1):
        _order_info(p, "vps")
    p.u("vps_max_layer_id", 6)
    if p.ue("vps_num_layer_sets_minus1"):
        return
    if p.u("vps_timing_info_present_flag", 1):
        return  # timing + optional extension: rest kept verbatim
    p.u("vps_extension_flag", 1)


def _sps(p: FieldParser) -> None:
    p.u("sps_video_parameter_set_id", 4)
    if p.u("sps_max_sub_layers_minus1", 3):
        return  # sub-layer PTL/ordering: deeper than the parse goes
    p.u("sps_temporal_id_nesting_flag", 1)
    _ptl(p)
    p.ue("sps_seq_parameter_set_id")
    if p.ue("chroma_format_idc") == CHROMA_444:
        p.u("separate_colour_plane_flag", 1)
    p.ue("pic_width_in_luma_samples")
    p.ue("pic_height_in_luma_samples")
    if p.u("conformance_window_flag", 1):
        for side in "lrtb":
            p.ue(f"conf_win_{side}_offset")
    p.ue("bit_depth_luma_minus8")
    p.ue("bit_depth_chroma_minus8")
    p.ue("log2_max_pic_order_cnt_lsb_minus4")
    if p.u("sps_sub_layer_ordering_info_present_flag", 1):
        _order_info(p, "sps")
    p.ue("log2_min_luma_coding_block_size_minus3")
    p.ue("log2_diff_max_min_luma_coding_block_size")
    p.ue("log2_min_luma_transform_block_size_minus2")
    p.ue("log2_diff_max_min_luma_transform_block_size")
    p.ue("max_transform_hierarchy_depth_inter")
    p.ue("max_transform_hierarchy_depth_intra")
    if p.u("scaling_list_enabled_flag", 1):
        return  # scaling lists: rest kept verbatim
    p.u("amp_enabled_flag", 1)
    p.u("sample_adaptive_offset_enabled_flag", 1)
    if p.u("pcm_enabled_flag", 1):
        return
    if p.ue("num_short_term_ref_pic_sets"):
        return  # st_ref_pic_set(): rest kept verbatim
    if p.u("long_term_ref_pics_present_flag", 1):
        return
    p.u("sps_temporal_mvp_enabled_flag", 1)
    p.u("strong_intra_smoothing_enabled_flag", 1)
    if p.u("vui_parameters_present_flag", 1):
        return
    p.u("sps_extension_present_flag", 1)


def _pps_prefix(p: FieldParser) -> int:
    """PPS fields up to and including tiles_enabled_flag; returns its value."""
    p.ue("pps_pic_parameter_set_id")
    p.ue("pps_seq_parameter_set_id")
    p.u("dependent_slice_segments_enabled_flag", 1)
    p.u("output_flag_present_flag", 1)
    p.u("num_extra_slice_header_flags", 3)
    p.u("sign_data_hiding_enabled_flag", 1)
    p.u("cabac_init_present_flag", 1)
    p.ue("num_ref_idx_l0_default_active_minus1")
    p.ue("num_ref_idx_l1_default_active_minus1")
    p.se("init_qp_minus26")
    p.u("constrained_intra_pred_flag", 1)
    p.u("transform_skip_enabled_flag", 1)
    if p.u("cu_qp_delta_enabled_flag", 1):
        p.ue("diff_cu_qp_delta_depth")
    p.se("pps_cb_qp_offset")
    p.se("pps_cr_qp_offset")
    p.u("pps_slice_chroma_qp_offsets_present_flag", 1)
    p.u("weighted_pred_flag", 1)
    p.u("weighted_bipred_flag", 1)
    p.u("transquant_bypass_enabled_flag", 1)
    return p.u("tiles_enabled_flag", 1)


def _pps_tail(p: FieldParser) -> None:
    """PPS fields after the tiles section (WPP, deblocking, merge, extensions)."""
    if p.u("entropy_coding_sync_enabled_flag", 1):
        return
    p.u("pps_loop_filter_across_slices_enabled_flag", 1)
    if p.u("deblocking_filter_control_present_flag", 1):
        return
    if p.u("pps_scaling_list_data_present_flag", 1):
        return
    p.u("lists_modification_present_flag", 1)
    p.ue("log2_parallel_merge_level_minus2")
    p.u("slice_segment_header_extension_present_flag", 1)
    p.u("pps_extension_present_flag", 1)


def _pps(p: FieldParser) -> None:
    if _pps_prefix(p):
        return  # tiles section: rest kept verbatim (hevc_tiles edits it)
    _pps_tail(p)


_PARSERS: dict[int, Callable[[FieldParser], None]] = {
    VPS_TYPE: _vps,
    SPS_TYPE: _sps,
    PPS_TYPE: _pps,
}


def parse_ps(nal_type: int, rbsp: bytes) -> tuple[list[Field], str] | None:
    """(fields, rest bits) of a VPS/SPS/PPS RBSP, or None if it does not parse."""
    p = FieldParser(rbsp)
    try:
        _PARSERS[nal_type](p)
    except (KeyError, ValueError):
        return None  # unknown type, truncated or a >32-bit codeword
    return p.fields, p.reader.rest(Tail.RBSP)


def _edge(field: Field, rng: RandPool) -> int:
    """New value for *field*: flag flip, or a draw from its kind's edges."""
    if field.kind is Kind.UE:
        return UE_EDGES[rng.randint(0, len(UE_EDGES) - 1)]
    if field.kind is Kind.SE:
        return SE_EDGES[rng.randint(0, len(SE_EDGES) - 1)]
    if field.width == 1:
        return field.value ^ 1
    if field.width == 8:
        return U8_EDGES[rng.randint(0, len(U8_EDGES) - 1)]
    return rng.randint(0, (1 << field.width) - 1)


def edit_ps(nal_type: int, rbsp: bytes, rng: RandPool) -> bytes | None:
    """Rewrite one field of a VPS/SPS/PPS RBSP; None if it does not parse.

    Draw order: field index, then the value.
    """
    parsed = parse_ps(nal_type, rbsp)
    if parsed is None:
        return None
    fields, rest = parsed

    field = fields[rng.randint(0, len(fields) - 1)]
    field.value = _edge(field, rng)
    return pack(fields, rest, Tail.RBSP)


def find_ps_units(data: bytes) -> list[tuple[int, int, int]]:
    """(nal_type, payload start, payload end) of every VPS/SPS/PPS unit.

    Payload excludes the 2-byte NAL header and trailing zero bytes (part of
    a 4-byte start code or trailing_zero_8bits).
    """
    units: list[tuple[int, int, int]] = []
    for m in _PS_UNIT.finditer(data):
        start = m.end() + 1  # second NAL header byte
        nxt = data.find(START_CODE, start)
        end = len(data) if nxt < 0 else nxt
        while end > start and data[end - 1] == 0:
            end -= 1
        units.append(((data[m.end() - 1] >> 1) & NAL_TYPE_MASK, start, end))
    return units


def sniff_hevc_ps(data: bytes) -> bool:
    """Annex B stream (start code at 0, forbidden bit clear) holding a VPS/SPS/PPS."""
    head = 4 if data[:4] == b"\x00" + START_CODE else 3
    if data[head - 3 : head] != START_CODE or len(data) <= head:
        return False
    return not data[head] & FORBIDDEN_BIT and _PS_UNIT.search(data) is not None


def mutate_hevc_ps(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Rewrite one VPS/SPS/PPS field in an Annex B stream; None if declined.

    Draw order: unit index, field index, value.
    """
    if not sniff_hevc_ps(data):
        return None
    units = find_ps_units(data)
    nal_type, start, end = units[rng.randint(0, len(units) - 1)]

    rbsp = edit_ps(nal_type, unescape(data[start:end]), rng)
    if rbsp is None:
        return None
    out = data[:start] + escape(rbsp) + data[end:]
    return out[:max_len]
