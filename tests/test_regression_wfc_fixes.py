"""Regression tests for four WFC defects, plus the JPEG WFC wiring.

Each class records the broken behaviour it pins, so a refactor that
reintroduces it fails with an explanation rather than a bare assertion.
"""

import random
import struct

import numpy as np
import pytest

from fuzzer_tool.core.mutations.jpeg import (
    EOI,
    SOI,
    JpegMutator,
    _marker_name,
    parse_jpeg_markers,
)
from fuzzer_tool.core.wfc import (
    AdjacencyTable,
    ConstraintSet,
    Tile,
    WaveGrid,
)


def _png_wave(width=6):
    tiles = [Tile(name=n) for n in (b"IHDR", b"IDAT", b"IEND")]
    return WaveGrid(tiles, ConstraintSet.png_chunks(), width=width, height=1)


def _seg(marker, payload=b""):
    if marker in (SOI, EOI):
        return b"\xff" + bytes([marker])
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload


SAMPLE_JPEG = (
    _seg(SOI)
    + _seg(0xE0, b"JFIF\x00")
    + _seg(0xDB, b"\x00" * 10)
    + _seg(0xC4, b"\x00" * 8)
    + _seg(0xC0, b"\x08\x00\x10\x00\x10\x01\x01\x11\x00")
    + _seg(0xDA, b"\x01\x01\x00\x00\x3f\x00")
    + b"\x12\x34"
    + _seg(EOI)
)


# ── Bug 1: run() reseeded the global RNG ───────────────────────────────


class TestRunDoesNotTouchGlobalRng:
    """``run()`` called ``random.seed(seed)``, resetting the process-wide
    stream that the fuzzer's operators and schedulers draw from. WFC runs per
    PNG reorder and per BMP row, so the global stream was being reset
    constantly and its continuation became a function of the WFC seed alone
    rather than of fuzzing history."""

    def test_global_stream_is_unchanged_by_a_seeded_run(self):
        random.seed(42)
        expected = [random.random() for _ in range(5)]

        random.seed(42)
        _png_wave().run(seed=12345)
        actual = [random.random() for _ in range(5)]

        assert actual == expected, "WFC hijacked the global random stream"

    def test_global_stream_survives_restarts(self):
        """Restart paths reseeded too; force several and re-check."""
        random.seed(7)
        expected = [random.random() for _ in range(3)]
        random.seed(7)
        _png_wave(width=12).run(seed=99, max_restarts=3, ac3_budget=1)
        assert [random.random() for _ in range(3)] == expected

    def test_run_is_still_deterministic_for_a_fixed_seed(self):
        assert _png_wave().run(seed=7) == _png_wave().run(seed=7)

    def test_different_seeds_still_differ(self):
        results = {tuple(_png_wave(width=8).run(seed=s)[0]) for s in range(12)}
        assert len(results) > 1

    def test_grid_seed_is_independent_of_global_state(self):
        """Output must depend on the passed seed, not on ambient global state."""
        random.seed(1)
        a = _png_wave(width=8).run(seed=555)
        random.seed(999999)
        b = _png_wave(width=8).run(seed=555)
        assert a == b


# ── Bug 2: _direction_to misread single-column grids ───────────────────


class TestDirectionIsWidthAware:
    """``_direction_to`` inferred the axis from ``to_idx - from_idx``. A
    vertical step is ``±w``, which equals ``±1`` when ``w == 1``, so a
    single-column grid resolved its vertical neighbours as left/right and
    checked them against horizontal adjacency rules."""

    def test_single_column_neighbours_are_vertical(self):
        grid = WaveGrid([Tile(name=b"A")], AdjacencyTable(), width=1, height=4)
        assert grid._direction_to(1, 0) == "up"
        assert grid._direction_to(1, 2) == "down"

    def test_single_row_neighbours_are_horizontal(self):
        grid = WaveGrid([Tile(name=b"A")], AdjacencyTable(), width=4, height=1)
        assert grid._direction_to(1, 0) == "left"
        assert grid._direction_to(1, 2) == "right"

    def test_square_grid_directions(self):
        grid = WaveGrid([Tile(name=b"A")], AdjacencyTable(), width=3, height=3)
        assert grid._direction_to(4, 3) == "left"
        assert grid._direction_to(4, 5) == "right"
        assert grid._direction_to(4, 1) == "up"
        assert grid._direction_to(4, 7) == "down"

    @pytest.mark.parametrize(("w", "h"), [(1, 5), (5, 1), (2, 3), (3, 2), (4, 4)])
    def test_every_neighbour_resolves_consistently(self, w, h):
        """Direction must be the exact inverse when read from either end."""
        opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}
        grid = WaveGrid([Tile(name=b"A")], AdjacencyTable(), width=w, height=h)
        for idx in range(grid.n):
            for nidx in grid._neighbors(idx):
                forward = grid._direction_to(idx, nidx)
                assert grid._direction_to(nidx, idx) == opposite[forward]


# ── Bug 3: _reset() discarded caller constraints ───────────────────────


class TestResetPreservesCallerConstraints:
    """Callers pin cells by writing into ``superpositions`` before ``run()``
    (IHDR first, IEND last, SOI first...). ``_reset()`` set the whole array
    back to True, so any wave that contradicted once restarted completely
    unconstrained and could place an anchor anywhere."""

    def _pinned_wave(self, width=6):
        wave = _png_wave(width=width)
        wave.superpositions[0][:] = False
        wave.superpositions[0][0] = True  # IHDR pinned first
        wave.superpositions[-1][:] = False
        wave.superpositions[-1][2] = True  # IEND pinned last
        return wave

    def test_pins_survive_a_forced_restart(self):
        wave = self._pinned_wave()
        wave.run(seed=3, max_restarts=2)
        wave._reset()
        assert wave.superpositions[0].sum() == 1
        assert bool(wave.superpositions[0][0])
        assert wave.superpositions[-1].sum() == 1
        assert bool(wave.superpositions[-1][2])

    def test_anchors_hold_in_the_output(self):
        for seed in range(25):
            row = self._pinned_wave(width=7).run(seed=seed, max_restarts=3)[0]
            assert row[0] == b"IHDR"
            assert row[-1] == b"IEND"

    def test_unconstrained_wave_still_resets_to_all_true(self):
        """Grids with no caller pins must keep the original reset behaviour."""
        wave = _png_wave()
        wave.run(seed=1)
        wave._reset()
        assert wave.superpositions.all()

    def test_initial_snapshot_is_taken_per_run(self):
        """A second run must snapshot the state it is actually given."""
        wave = _png_wave()
        wave.run(seed=1)
        wave._reset()
        wave.superpositions[0][:] = False
        wave.superpositions[0][1] = True
        wave.run(seed=2)
        wave._reset()
        assert wave.superpositions[0].sum() == 1
        assert bool(wave.superpositions[0][1])


# ── Bug 4 / wiring: jpeg_markers() was defined but never used ──────────


class TestJpegConstraintSet:
    def test_covers_every_marker_the_parser_emits(self):
        """Adjacency is closed-world: a marker with no rules is pruned from
        every cell and forces a contradiction, so the table must cover the
        full parsed marker set."""
        table = ConstraintSet.jpeg_markers()
        parsed = parse_jpeg_markers(SAMPLE_JPEG)
        for m in parsed:
            name = _marker_name(m.marker).encode()
            assert table.has_tile(name), f"{name!r} has no adjacency rules"

    def test_dri_is_present(self):
        assert ConstraintSet.jpeg_markers().has_tile(b"DRI")

    def test_sos_is_followed_only_by_eoi(self):
        table = ConstraintSet.jpeg_markers()
        assert table.compatible(b"SOS", b"EOI", "right")
        assert not table.compatible(b"SOS", b"DQT", "right")


class TestJpegWfcReorder:
    def _mutator(self, seed=3, use_wfc=True):
        mut = JpegMutator()
        mut.use_wfc = use_wfc
        mut._rng = random.Random(seed)
        return mut

    def test_wfc_is_off_by_default(self):
        assert JpegMutator.use_wfc is False

    def test_preserves_every_segment(self):
        """Reordering must not drop or duplicate segments."""
        mut = self._mutator()
        original = len(parse_jpeg_markers(SAMPLE_JPEG))
        for _ in range(100):
            result = mut._reorder_markers(parse_jpeg_markers(SAMPLE_JPEG), 4096)
            assert len(result) == original

    def test_soi_first_and_eoi_last(self):
        """WFC emits a sequence of tile *types*, not a permutation, so
        unplaced segments are appended — anchors must be re-pinned after
        reassembly or a segment strands after EOI."""
        mut = self._mutator()
        for _ in range(200):
            result = mut._reorder_markers(parse_jpeg_markers(SAMPLE_JPEG), 4096)
            assert result[0].marker == SOI
            assert result[-1].marker == EOI

    def test_produces_varied_orderings(self):
        mut = self._mutator()
        orders = {
            tuple(
                _marker_name(m.marker)
                for m in mut._reorder_markers(parse_jpeg_markers(SAMPLE_JPEG), 4096)
            )
            for _ in range(200)
        }
        assert len(orders) > 5, "WFC reorder produced almost no variety"

    def test_falls_back_when_a_marker_has_no_rules(self):
        """An unknown marker must route to the random swap, not contradict."""
        exotic = _seg(SOI) + _seg(0xE5, b"\x00" * 4) + _seg(0xDB, b"\x00" * 8) + _seg(EOI)
        parsed = parse_jpeg_markers(exotic)
        assert parsed is not None
        result = self._mutator()._reorder_markers(parsed, 4096)
        assert len(result) == len(parsed)

    def test_non_wfc_path_is_unchanged(self):
        mut = self._mutator(use_wfc=False)
        original = len(parse_jpeg_markers(SAMPLE_JPEG))
        result = mut._reorder_markers(parse_jpeg_markers(SAMPLE_JPEG), 4096)
        assert len(result) == original
        assert result[0].marker == SOI

    def test_short_marker_lists_pass_through(self):
        mut = self._mutator()
        parsed = parse_jpeg_markers(_seg(SOI) + _seg(EOI))
        if parsed is not None:
            assert len(mut._reorder_markers(parsed, 4096)) == len(parsed)

    def test_does_not_disturb_the_global_rng(self):
        """The whole chain, not just WaveGrid, must leave globals alone."""
        random.seed(11)
        expected = [random.random() for _ in range(4)]
        random.seed(11)
        self._mutator()._reorder_markers(parse_jpeg_markers(SAMPLE_JPEG), 4096)
        assert [random.random() for _ in range(4)] == expected


class TestWfcFlagPropagation:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_operator_sets_use_wfc_from_fuzzer_flag(self, enabled):
        """_op_jpeg_chunk_mutate must mirror the fuzzer's --wfc setting."""
        from fuzzer_tool.services.operators import OperatorEngine

        class _Fuzzer:
            _wfc_enabled = enabled
            max_len = 4096
            _rand_pool = random.Random(1)

        engine = OperatorEngine.__new__(OperatorEngine)
        engine.f = _Fuzzer()
        engine._op_jpeg_chunk_mutate(bytearray(SAMPLE_JPEG), 0, SAMPLE_JPEG)
        assert engine.f._jpeg_mutator.use_wfc is enabled


class TestWfcOutputIntegrity:
    def test_collapsed_grid_has_no_none_when_solvable(self):
        row = _png_wave(width=5).run(seed=4)[0]
        assert all(t is not None for t in row)

    def test_superposition_dtype_stays_bool(self):
        """The bool dtype is the memory optimisation the module docstring
        claims; an accidental widening would silently cost ~8x."""
        wave = _png_wave()
        wave.run(seed=1)
        assert wave.superpositions.dtype == np.bool_
