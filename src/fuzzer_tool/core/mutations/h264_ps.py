"""H.264 SPS/PPS field mutator (FFmpeg ``h264_ps.c``).

``nal_chunk_mutate`` flips random bits in parameter sets; a flip inside an
exp-Golomb codeword re-frames every later field, so most edits land on
garbage. This op parses the RBSP into fields and rewrites one:

  Annex B unit ──unescape──► RBSP ──parse──► [profile][level][sps_id] ... | rest
                                              edit one field (edge value)
  spliced back ◄──escape──── re-pack (fields + rest + stop bit)

SPS stops at a present scaling matrix or VUI, PPS at slice groups; the
rest is kept verbatim. Edges: ids past 31/255, 2^32-2 sizes, QP past +-26.
``unescape``/``escape``/``parse_ps``/``edit_ps`` are reused for avcC.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from fuzzer_tool.core.mutations.bitrec import Field, FieldParser, Kind, Tail, pack
from fuzzer_tool.core.rand_pool import RandPool

SPS_TYPE, PPS_TYPE = 7, 8
NAL_TYPE_MASK = 0x1F
FORBIDDEN_BIT = 0x80
START_CODE = b"\x00\x00\x01"
MAX_POC_CYCLE = 255  # FFmpeg h264_ps.c limit

# Profiles carrying chroma_format_idc / bit depth (H.264 §7.3.2.1.1).
HIGH_PROFILES = frozenset({100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135})
CHROMA_444 = 3

UE_EDGES = (0, 1, 2, 3, 4, 8, 12, 15, 16, 31, 32, 255, 1023, 65535, 2**31 - 1, 2**32 - 2)
SE_EDGES = (0, 1, -1, 25, -26, 51, -52, 127, -128, 2**31 - 1, -(2**31))
U8_EDGES = (0, 44, 66, 77, 88, 100, 110, 122, 144, 244, 255)

# PS unit in the first bytes of an Annex B stream: start code + type 7/8.
_PS_UNIT = re.compile(rb"\x00\x00\x01[\x07\x27\x47\x67\x08\x28\x48\x68]")
_EP_SEQ = re.compile(rb"\x00\x00\x03")
_EP_NEEDED = re.compile(rb"\x00\x00(?=[\x00-\x03])")


def unescape(ebsp: bytes) -> bytes:
    """Drop emulation-prevention bytes: ``00 00 03`` -> ``00 00``."""
    return _EP_SEQ.sub(b"\x00\x00", ebsp)


def escape(rbsp: bytes) -> bytes:
    """Insert ``03`` after any two zeros followed by a byte <= 3, and after
    a trailing zero (H.264 §7.4.1)."""
    out = _EP_NEEDED.sub(b"\x00\x00\x03", rbsp)
    return out + b"\x03" if out[-1:] == b"\x00" else out


def _high_profile(p: FieldParser) -> bool:
    """Chroma / bit-depth block; False when a scaling matrix ends the parse."""
    if p.ue("chroma_format_idc") == CHROMA_444:
        p.u("separate_colour_plane_flag", 1)
    p.ue("bit_depth_luma_minus8")
    p.ue("bit_depth_chroma_minus8")
    p.u("qpprime_y_zero_transform_bypass_flag", 1)
    return not p.u("seq_scaling_matrix_present_flag", 1)


def _poc(p: FieldParser) -> bool:
    """POC type fields; False when the cycle is too long to parse."""
    poc_type = p.ue("pic_order_cnt_type")
    if poc_type == 0:
        p.ue("log2_max_pic_order_cnt_lsb_minus4")
        return True
    if poc_type != 1:
        return True

    p.u("delta_pic_order_always_zero_flag", 1)
    p.se("offset_for_non_ref_pic")
    p.se("offset_for_top_to_bottom_field")
    cycle = p.ue("num_ref_frames_in_pic_order_cnt_cycle")
    if cycle > MAX_POC_CYCLE:
        return False
    for i in range(cycle):
        p.se(f"offset_for_ref_frame_{i}")
    return True


def _sps(p: FieldParser) -> None:
    profile = p.u("profile_idc", 8)
    p.u("constraint_flags", 8)
    p.u("level_idc", 8)
    p.ue("seq_parameter_set_id")
    if profile in HIGH_PROFILES and not _high_profile(p):
        return

    p.ue("log2_max_frame_num_minus4")
    if not _poc(p):
        return

    p.ue("max_num_ref_frames")
    p.u("gaps_in_frame_num_allowed_flag", 1)
    p.ue("pic_width_in_mbs_minus1")
    p.ue("pic_height_in_map_units_minus1")
    if not p.u("frame_mbs_only_flag", 1):
        p.u("mb_adaptive_frame_field_flag", 1)
    p.u("direct_8x8_inference_flag", 1)
    if p.u("frame_cropping_flag", 1):
        for side in "lrtb":
            p.ue(f"frame_crop_{side}_offset")
    p.u("vui_parameters_present_flag", 1)


def _pps(p: FieldParser) -> None:
    p.ue("pic_parameter_set_id")
    p.ue("seq_parameter_set_id")
    p.u("entropy_coding_mode_flag", 1)
    p.u("bottom_field_pic_order_in_frame_present_flag", 1)
    if p.ue("num_slice_groups_minus1"):
        return

    p.ue("num_ref_idx_l0_default_active_minus1")
    p.ue("num_ref_idx_l1_default_active_minus1")
    p.u("weighted_pred_flag", 1)
    p.u("weighted_bipred_idc", 2)
    p.se("pic_init_qp_minus26")
    p.se("pic_init_qs_minus26")
    p.se("chroma_qp_index_offset")
    p.u("deblocking_filter_control_present_flag", 1)
    p.u("constrained_intra_pred_flag", 1)
    p.u("redundant_pic_cnt_present_flag", 1)


_PARSERS: dict[int, Callable[[FieldParser], None]] = {SPS_TYPE: _sps, PPS_TYPE: _pps}


def parse_ps(nal_type: int, rbsp: bytes) -> tuple[list[Field], str] | None:
    """(fields, rest bits) of an SPS/PPS RBSP, or None if it does not parse."""
    p = FieldParser(rbsp)
    try:
        _PARSERS[nal_type](p)
    except ValueError:
        return None  # truncated or a >32-bit codeword: not a usable PS
    return p.fields, p.reader.rest(Tail.RBSP)


def _edge(field: Field, rng: RandPool) -> int:
    """New value for *field*: flag flip, or a draw from its kind's edges."""
    if field.kind is Kind.UE:
        return rng.choice(UE_EDGES)
    if field.kind is Kind.SE:
        return rng.choice(SE_EDGES)
    if field.width == 1:
        return field.value ^ 1
    if field.width == 8:
        return rng.choice(U8_EDGES)
    return rng.randint(0, (1 << field.width) - 1)


def edit_ps(nal_type: int, rbsp: bytes, rng: RandPool) -> bytes | None:
    """Rewrite one field of an SPS/PPS RBSP; None if it does not parse.

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
    """(nal_type, payload start, payload end) of every SPS/PPS unit.

    Payload excludes the 1-byte NAL header and trailing zero bytes (part of
    a 4-byte start code or trailing_zero_8bits).
    """
    units: list[tuple[int, int, int]] = []
    for m in _PS_UNIT.finditer(data):
        start = m.end()
        nxt = data.find(START_CODE, start)
        end = len(data) if nxt < 0 else nxt
        while end > start and data[end - 1] == 0:
            end -= 1
        units.append((data[m.end() - 1] & NAL_TYPE_MASK, start, end))
    return units


def sniff_h264_ps(data: bytes) -> bool:
    """Annex B stream (start code at 0, forbidden bit clear) holding an SPS/PPS."""
    head = 4 if data[:4] == b"\x00" + START_CODE else 3
    if data[head - 3 : head] != START_CODE or len(data) <= head:
        return False
    return not data[head] & FORBIDDEN_BIT and _PS_UNIT.search(data) is not None


def mutate_h264_ps(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Rewrite one SPS/PPS field in an Annex B stream; None if declined.

    Draw order: unit index, field index, value.
    """
    if not sniff_h264_ps(data):
        return None
    units = find_ps_units(data)
    nal_type, start, end = units[rng.randint(0, len(units) - 1)]

    rbsp = edit_ps(nal_type, unescape(data[start:end]), rng)
    if rbsp is None:
        return None
    out = data[:start] + escape(rbsp) + data[end:]
    return out[:max_len]
