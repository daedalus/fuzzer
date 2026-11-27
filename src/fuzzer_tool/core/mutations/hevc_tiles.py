"""HEVC (H.265) PPS tile-grid mutator (FFmpeg ``dxva2_hevc.c``).

``hevc_ps_mutate`` stops where the tiles section starts (tiles/WPP keep the
rest verbatim), so the tile grid — the fields FFmpeg's
``ff_dxva2_hevc_fill_picture_parameters()`` copies into fixed-size DXVA
arrays (CVE-2026-125239: crafted bitstreams with many non-uniformly spaced
tiles write controlled 16-bit values past those arrays during DXVA2/D3D11VA/
D3D12VA decode) — only ever sees flat bit flips. This op parses the whole
grid and rewrites it:

  Annex B PPS ──unescape──► RBSP ──parse──► [prefix][counts][spacing][lists]
                                              [loop_filter][tail] | rest
  one edit: count edge / spacing flip (lists rebuilt) / list edge / LF flip
  spliced back ◄──escape──── re-pack (fields + rest + stop bit)

Edges: tile counts at 0/1/2 and past the PicWidthInCtbsY bounds (18, 19,
21, 255, 2^32-2 — one huge count forces uniform spacing so the re-pack
stays small), column/row sizes 0 (1 luma sample: the max tile count for the
picture) to 2^32-2 (widths that cannot sum to the picture). Grids wider
than 64 columns/rows decline: the width list is that many ue codewords and
anything past the spec bound of 20 is already unreachable structure.
The prefix/tail parsers are shared with ``hevc_ps`` (one source of truth).
"""

from __future__ import annotations

from fuzzer_tool.core.mutations.bitrec import Field, FieldParser, Kind, Tail, pack
from fuzzer_tool.core.mutations.h264_ps import escape, unescape
from fuzzer_tool.core.mutations.hevc_ps import (
    PPS_TYPE,
    _pps_prefix,
    _pps_tail,
    find_ps_units,
    sniff_hevc_ps,
)
from fuzzer_tool.core.rand_pool import RandPool

# Spec bound is 20 columns / 22 rows; 64 caps the width/height lists a re-
# pack would have to rebuild. Above that the PPS is unrecoverable garbage.
MAX_TILE_GRID = 64

COUNT_EDGES = (0, 1, 2, 18, 19, 21, 255, 2**32 - 2)
WIDTH_EDGES = (0, 1, 65, 255, 2**32 - 2)


def _pps_tiles(p: FieldParser) -> None:
    """PPS parse including the tile grid; raises ValueError past its edges."""
    if not _pps_prefix(p):
        raise ValueError  # no tiles: not this op's structure
    cols_minus1 = p.ue("num_tile_columns_minus1")
    rows_minus1 = p.ue("num_tile_rows_minus1")
    if cols_minus1 >= MAX_TILE_GRID or rows_minus1 >= MAX_TILE_GRID:
        raise ValueError  # width/height lists would be unbounded
    if not p.u("uniform_spacing_flag", 1):
        for i in range(cols_minus1):
            p.ue(f"column_width_minus1_{i}")
        for i in range(rows_minus1):
            p.ue(f"row_height_minus1_{i}")
    p.u("loop_filter_across_tiles_enabled_flag", 1)
    _pps_tail(p)


def parse_tiles(rbsp: bytes) -> tuple[list[Field], str] | None:
    """(fields, rest bits) of a PPS RBSP with a tile grid, or None."""
    p = FieldParser(rbsp)
    try:
        _pps_tiles(p)
    except ValueError:
        return None  # no tiles, truncated grid, or an oversized count
    return p.fields, p.reader.rest(Tail.RBSP)


def _count(names: list[str], prefix: str) -> list[int]:
    """Field indices of the tile width/height list entries (document order)."""
    return [i for i, n in enumerate(names) if n.startswith(prefix)]


def _drop_lists(fields: list[Field], i_uniform: int, i_lf: int) -> list[Field]:
    """Set uniform spacing, removing the explicit width/height lists."""
    fields[i_uniform].value = 1
    del fields[i_uniform + 1 : i_lf]
    return fields


def edit_tiles(rbsp: bytes, rng: RandPool) -> bytes | None:
    """Rewrite one tile-grid field of a PPS RBSP; None if it does not parse.

    Draw order: op (among the feasible ones), then value(s).
    """
    parsed = parse_tiles(rbsp)
    if parsed is None:
        return None
    fields, rest = parsed
    names = [f.name for f in fields]
    i_cols = names.index("num_tile_columns_minus1")
    i_rows = names.index("num_tile_rows_minus1")
    i_uniform = names.index("uniform_spacing_flag")
    i_lf = names.index("loop_filter_across_tiles_enabled_flag")
    widths = _count(names, "column_width_minus1_")
    heights = _count(names, "row_height_minus1_")
    uniform = fields[i_uniform].value == 1

    ops = ["count", "lf"]
    if (uniform and (fields[i_cols].value or fields[i_rows].value)) or not uniform:
        ops.append("spacing")
    if widths or heights:
        ops.append("width")
    op = ops[rng.randint(0, len(ops) - 1)]

    if op == "count":
        # A count edit on an explicit grid rebuilds it as uniform: the new
        # count's width list is not the old one's, and 2^32-2 entries cannot
        # be packed at all.
        if not uniform:
            fields = _drop_lists(fields, i_uniform, i_lf)
        target = (fields[i_cols], fields[i_rows])[rng.randint(0, 1)]
        target.value = COUNT_EDGES[rng.randint(0, len(COUNT_EDGES) - 1)]
    elif op == "spacing":
        if uniform:
            fields[i_uniform].value = 0
            built: list[Field] = []
            for kind, n in (
                ("column_width_minus1", fields[i_cols].value),
                ("row_height_minus1", fields[i_rows].value),
            ):
                for i in range(n):
                    built.append(
                        Field(
                            f"{kind}_{i}",
                            Kind.UE,
                            0,
                            WIDTH_EDGES[rng.randint(0, len(WIDTH_EDGES) - 1)],
                        )
                    )
            fields[i_lf:i_lf] = built
        else:
            fields = _drop_lists(fields, i_uniform, i_lf)
    elif op == "width":
        idx = (widths + heights)[rng.randint(0, len(widths + heights) - 1)]
        fields[idx].value = WIDTH_EDGES[rng.randint(0, len(WIDTH_EDGES) - 1)]
    else:
        fields[i_lf].value ^= 1
    out = pack(fields, rest, Tail.RBSP)
    return None if out == rbsp else out  # an unchanged grid declines


def _has_grid(rbsp: bytes) -> bool:
    """A PPS RBSP whose tiles_enabled_flag is set (hevc_ps stops there)."""
    p = FieldParser(rbsp)
    try:
        return _pps_prefix(p) == 1
    except ValueError:
        return False


def sniff_hevc_tiles(data: bytes) -> bool:
    """Annex B HEVC stream holding at least one PPS with tiles enabled."""
    if not sniff_hevc_ps(data):
        return False
    return any(
        nal_type == PPS_TYPE and _has_grid(unescape(data[start:end]))
        for nal_type, start, end in find_ps_units(data)
    )


def mutate_hevc_tiles(data: bytes, max_len: int = 65536, *, rng: RandPool) -> bytes | None:
    """Rewrite one tile-grid field in an Annex B stream; None if declined.

    Draw order: tile-grid PPS unit, op, value(s).
    """
    if not sniff_hevc_ps(data):
        return None
    units = [
        (start, end)
        for nal_type, start, end in find_ps_units(data)
        if nal_type == PPS_TYPE and _has_grid(unescape(data[start:end]))
    ]
    if not units:
        return None
    start, end = units[rng.randint(0, len(units) - 1)]

    rbsp = edit_tiles(unescape(data[start:end]), rng)
    if rbsp is None:
        return None
    out = data[:start] + escape(rbsp) + data[end:]
    return out[:max_len]
