"""Tests for the HEVC PPS tile-grid mutator (`hevc_tile_mutate`)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fuzzer_tool.core.mutations.bitrec import Field, Kind, Tail, pack
from fuzzer_tool.core.mutations.h264_ps import escape, unescape
from fuzzer_tool.core.mutations.hevc_ps import find_ps_units
from fuzzer_tool.core.mutations.hevc_tiles import (
    COUNT_EDGES,
    WIDTH_EDGES,
    mutate_hevc_tiles,
    parse_tiles,
    sniff_hevc_tiles,
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
PPS, SPS, VPS = 34, 33, 32
SC = b"\x00\x00\x00\x01"


def _f(name, kind, value, width=0):
    return Field(name, kind, width, value)


U, UE, SE = Kind.FIXED, Kind.UE, Kind.SE

# PPS prefix/tail shared with the hevc_ps baseline (cu_qp_delta off, no
# WPP/deblocking/scaling lists), around one of two tile grids.
_PREFIX = [
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
    _f("cu_qp_delta_enabled_flag", U, 0, 1),
    _f("pps_cb_qp_offset", SE, 0),
    _f("pps_cr_qp_offset", SE, 0),
    _f("pps_slice_chroma_qp_offsets_present_flag", U, 0, 1),
    _f("weighted_pred_flag", U, 0, 1),
    _f("weighted_bipred_flag", U, 0, 1),
    _f("transquant_bypass_enabled_flag", U, 0, 1),
    _f("tiles_enabled_flag", U, 1, 1),
]
_TAIL = [
    _f("loop_filter_across_tiles_enabled_flag", U, 1, 1),
    _f("entropy_coding_sync_enabled_flag", U, 0, 1),
    _f("pps_loop_filter_across_slices_enabled_flag", U, 1, 1),
    _f("deblocking_filter_control_present_flag", U, 0, 1),
    _f("pps_scaling_list_data_present_flag", U, 0, 1),
    _f("lists_modification_present_flag", U, 0, 1),
    _f("log2_parallel_merge_level_minus2", UE, 0),
    _f("slice_segment_header_extension_present_flag", U, 0, 1),
    _f("pps_extension_present_flag", U, 0, 1),
]


def _grid(cols_m1, rows_m1, uniform, width=8):
    grid = [
        _f("num_tile_columns_minus1", UE, cols_m1),
        _f("num_tile_rows_minus1", UE, rows_m1),
        _f("uniform_spacing_flag", U, uniform, 1),
    ]
    if not uniform:
        grid += [_f(f"column_width_minus1_{i}", UE, width) for i in range(cols_m1)]
        grid += [_f(f"row_height_minus1_{i}", UE, width) for i in range(rows_m1)]
    return grid


def _pps_tiles(cols_m1, rows_m1, uniform, width=8):
    return _PREFIX + _grid(cols_m1, rows_m1, uniform, width) + _TAIL


UNIFORM_RBSP = pack(_pps_tiles(1, 1, 1), "", Tail.RBSP)  # 2x2 uniform grid
EXPLICIT_RBSP = pack(_pps_tiles(2, 1, 0), "", Tail.RBSP)  # 3x2 grid, widths 8
UNIFORM_STREAM = SC + b"\x44\x01" + escape(UNIFORM_RBSP) + SC + b"\x26\x01\xaf"
EXPLICIT_STREAM = SC + b"\x44\x01" + escape(EXPLICIT_RBSP) + SC + b"\x26\x01\xaf"


def _names(fields):
    return [f.name for f in fields]


def _tiles_value(data: bytes, name: str) -> int:
    """One field value from the first PPS of *data*."""
    pps = next(u for u in find_ps_units(data) if u[0] == PPS)
    fields, _ = parse_tiles(unescape(data[pps[1] : pps[2]]))
    return {f.name: f.value for f in fields}[name]


class TestParse:
    def test_uniform_grid(self):
        assert parse_tiles(UNIFORM_RBSP) == (_pps_tiles(1, 1, 1), "")

    def test_explicit_grid(self):
        assert parse_tiles(EXPLICIT_RBSP) == (_pps_tiles(2, 1, 0), "")

    def test_no_tiles_declines(self):
        rbsp = pack(_PREFIX[:-1] + _TAIL[1:], "", Tail.RBSP)  # tiles flag 0
        assert parse_tiles(rbsp) is None

    def test_oversized_grid_declines(self):
        rbsp = pack(_pps_tiles(64, 1, 1), "", Tail.RBSP)  # 65 columns
        assert parse_tiles(rbsp) is None

    def test_wpp_tail_stops_parse(self):
        fields = _pps_tiles(1, 1, 1)
        idx = _names(fields).index("entropy_coding_sync_enabled_flag")
        wpp = fields[:idx] + [_f("entropy_coding_sync_enabled_flag", U, 1, 1)]
        parsed = parse_tiles(pack(wpp, "0110", Tail.RBSP))
        assert parsed is not None
        fields_out, rest = parsed
        assert fields_out == wpp  # WPP keeps the rest verbatim
        assert rest == "0110"


class TestMutate:
    def test_count_edit_uniform(self):
        # unit 0, op 0 (count), target 0 (columns), edge 4 (18)
        out = mutate_hevc_tiles(UNIFORM_STREAM, rng=ScriptedRng(randints=[0, 0, 0, 4]))
        assert _tiles_value(out, "num_tile_columns_minus1") == COUNT_EDGES[4]
        assert _tiles_value(out, "num_tile_rows_minus1") == 1
        assert _tiles_value(out, "uniform_spacing_flag") == 1

    def test_count_edit_explicit_forces_uniform(self):
        # unit 0, op 0 (count), target 1 (rows), edge 3 (21): the explicit
        # width/height lists cannot be resized, so spacing becomes uniform.
        out = mutate_hevc_tiles(EXPLICIT_STREAM, rng=ScriptedRng(randints=[0, 0, 1, 3]))
        assert _tiles_value(out, "num_tile_rows_minus1") == COUNT_EDGES[3]
        assert _tiles_value(out, "uniform_spacing_flag") == 1
        pps = next(u for u in find_ps_units(out) if u[0] == PPS)
        fields, _ = parse_tiles(unescape(out[pps[1] : pps[2]]))
        assert not any(n.startswith(("column_width", "row_height")) for n in _names(fields))

    def test_lf_flip(self):
        # unit 0, op 1 (lf)
        out = mutate_hevc_tiles(UNIFORM_STREAM, rng=ScriptedRng(randints=[0, 1]))
        assert _tiles_value(out, "loop_filter_across_tiles_enabled_flag") == 0

    def test_spacing_uniform_to_explicit(self):
        # unit 0, op 2 (spacing): lists rebuilt for a 2x2 grid = 1+1 entries.
        out = mutate_hevc_tiles(UNIFORM_STREAM, rng=ScriptedRng(randints=[0, 2, 1, 2]))
        assert _tiles_value(out, "uniform_spacing_flag") == 0
        assert _tiles_value(out, "column_width_minus1_0") == WIDTH_EDGES[1]
        assert _tiles_value(out, "row_height_minus1_0") == WIDTH_EDGES[2]

    def test_spacing_explicit_to_uniform(self):
        # unit 0, op 2 (spacing): the 2+1 explicit entries are dropped.
        out = mutate_hevc_tiles(EXPLICIT_STREAM, rng=ScriptedRng(randints=[0, 2]))
        assert _tiles_value(out, "uniform_spacing_flag") == 1
        pps = find_ps_units(out)[0]
        fields, _ = parse_tiles(unescape(out[pps[1] : pps[2]]))
        assert not any(n.startswith(("column_width", "row_height")) for n in _names(fields))

    def test_width_edit(self):
        # unit 0, op 3 (width), entry 2 of widths+heights (row_height), edge 2
        out = mutate_hevc_tiles(EXPLICIT_STREAM, rng=ScriptedRng(randints=[0, 3, 2, 2]))
        assert _tiles_value(out, "row_height_minus1_0") == WIDTH_EDGES[2]

    # Falsification: a PPS without tiles has nothing this op reaches.
    def test_no_tiles_declines(self):
        assert mutate_hevc_tiles(SC + b"\x44\x01\xc0\x73\xc0\x89", rng=ScriptedRng()) is None

    def test_non_hevc_declines(self):
        assert mutate_hevc_tiles(b"\x00\x00\x01\xba" + UNIFORM_STREAM, rng=ScriptedRng()) is None

    # Adversarial: every truncation never raises; a huge-count edit is allowed
    # to produce a PPS whose grid no longer reparses (decoders must reject it).
    def test_truncations_never_raise(self):
        rng = ScriptedRng(randints=[0] * 600)
        for cut in range(len(EXPLICIT_STREAM) + 1):
            out = mutate_hevc_tiles(EXPLICIT_STREAM[:cut], rng=rng)
            assert out is None or isinstance(out, bytes)


class TestRegistry:
    def test_sniffer(self):
        assert sniff_hevc_tiles(UNIFORM_STREAM)
        assert sniff_hevc_tiles(EXPLICIT_STREAM)
        assert not sniff_hevc_tiles(SC + b"\x44\x01\xc0\x73\xc0\x89")
        assert not sniff_hevc_tiles(b"\x00\x00\x01\xba" + UNIFORM_STREAM)

    def test_gate(self):
        assert format_gate_matches("hevc_tile_mutate", UNIFORM_STREAM)
        assert not format_gate_matches("hevc_tile_mutate", SC + b"\x44\x01\xc0\x73\xc0\x89")

    def test_category(self):
        assert "hevc_tile_mutate" in OPERATOR_CATEGORIES["format"]


@pytest.mark.skipif(FFMPEG is None or not LIBX265, reason="needs ffmpeg + libx265")
class TestFfmpeg:
    # x265 4.1 dropped tile encoding, so a real libx265 stream carries no
    # tiles; the fixture splices a hand-built 2x2-grid PPS in place of the
    # encoder's own (same ids) so the decoder still consumes the stream.
    def _encode(self, tmp_path) -> bytes:
        path = tmp_path / "base.h265"
        cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc=d=0.2:s=64x48"]
        cmd += ["-c:v", "libx265", "-preset", "ultrafast", "-f", "hevc", str(path)]
        subprocess.run(cmd, check=True)
        data = path.read_bytes()
        _, start, end = next(u for u in find_ps_units(data) if u[0] == PPS)
        return data[:start] + escape(UNIFORM_RBSP) + data[end:]

    def test_tiles_pps_round_trip(self, tmp_path):
        data = self._encode(tmp_path)
        _, start, end = next(u for u in find_ps_units(data) if u[0] == PPS)
        rbsp = unescape(data[start:end])
        parsed = parse_tiles(rbsp)
        assert parsed is not None
        fields, rest = parsed
        assert pack(fields, rest, Tail.RBSP) == rbsp
        assert _tiles_value(data, "tiles_enabled_flag") == 1

    def test_tile_count_edit_reaches_decoder(self, tmp_path):
        data = self._encode(tmp_path)
        # unit 0 (the one tiles PPS), op 0 (count), target 0 (columns), edge 7
        out = mutate_hevc_tiles(data, rng=ScriptedRng(randints=[0, 0, 0, 7]))
        path = tmp_path / "out.h265"
        path.write_bytes(out)
        cmd = [FFMPEG, "-v", "error", "-i", str(path), "-f", "null", "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert "tile" in proc.stderr.lower()  # num_tile_columns out of range
