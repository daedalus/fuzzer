"""Regression: WFC pixel mutation must not stall the fuzz loop.

The wave-collapse generator's propagation cost is ~O(n^2 * n_tiles^2) per
row, with the BMP mutator feeding it real image widths (thousands of
cells) and up to 256 distinct byte tiles. A single pixel mutation then
took from ~9 s (512 cells / 128 tiles) to minutes at BMP widths — a hang
from the fuzzer's point of view. Two layers pin the fix:

- ``WaveGrid.run`` carries a hard work budget; exhausting it stops the
  collapse and returns a partial grid instead of spinning (the budget is
  prune-call bounded, so an adversarial wave terminates in bounded work).
- ``BmpMutator._wfc_pixels`` refuses rows whose width or alphabet makes
  collapse unaffordable and falls back to plain byte flips.
"""

import random
import time

from fuzzer_tool.core.mutations.bmp import BmpInfo, BmpMutator
from fuzzer_tool.core.wfc import AdjacencyTable, Tile, WaveGrid


def _dense_wave(n_cells: int, n_tiles: int) -> WaveGrid:
    """A satisfiable-but-churning wave: near-dense adjacency over a closed
    world, wide enough that the pre-fix algorithm ran for minutes."""
    tiles = [Tile(name=bytes([i]), weight=1.0) for i in range(n_tiles)]
    adj = AdjacencyTable()
    for i in range(n_tiles):
        for j in range(n_tiles):
            if (i + j) % 3 != 0:
                adj.add_forward(tiles[i].name, tiles[j].name)
    return WaveGrid(tiles, adj, width=n_cells, height=1)


class TestWaveWorkBudget:
    def test_run_terminates_on_a_wide_wave(self):
        """The old code took ~100 s on 4096 x 256; the budget must bound it."""
        wave = _dense_wave(1024, 256)
        t0 = time.perf_counter()
        grid = wave.run(seed=1, max_restarts=2, ac3_budget=2000, work_budget=20_000)
        dt = time.perf_counter() - t0
        assert len(grid[0]) == 1024, "grid shape preserved"
        assert wave._work_used <= 20_000, "work budget honoured"
        assert dt < 30, f"contrary to the budget, run took {dt:.1f}s"

    def test_budget_exhaustion_returns_a_partial_grid_not_a_hang(self):
        """A tight budget must stop the collapse and still produce output."""
        wave = _dense_wave(4096, 256)
        t0 = time.perf_counter()
        grid = wave.run(seed=1, max_restarts=2, ac3_budget=2000, work_budget=500)
        dt = time.perf_counter() - t0
        assert dt < 30, f"wall-clock guard: {dt:.1f}s"
        assert wave._budget_exhausted, "tight budget should have fired"
        assert len(grid[0]) == 4096


class TestBmpWfcAffordabilityGuard:
    def _info(self, *, width: int, height: int, bit_count: int, pixels: bytes) -> BmpInfo:
        return BmpInfo(
            file_size=0,
            pixel_offset=0,
            dib_size=40,
            width=width,
            height=height,
            planes=1,
            bit_count=bit_count,
            compression=0,
            image_size=0,
            x_ppm=0,
            y_ppm=0,
            colors_used=0,
            colors_important=0,
            header=bytearray(b"\x00" * 40),
            color_table=b"",
            pixel_data=pixels,
        )

    def test_wide_row_falls_back_to_byte_flips(self):
        """A 24-bit row with an all-distinct alphabet (> 64 tiles) must not
        enter WFC: the mutator returns the cheap flip mutation instead."""
        width, height, bpp = 200, 2, 3
        stride = ((width * bpp + 3) // 4) * 4
        pixels = bytearray(stride * height)
        # Distinct 3-byte samples for the whole first row: 200 unique tiles.
        for x in range(width):
            pixels[x * bpp : (x + 1) * bpp] = (x + 1).to_bytes(3, "big")
        info = self._info(width=width, height=height, bit_count=24, pixels=bytes(pixels))

        mut = BmpMutator()
        mut.use_wfc = True
        mut._rng = random.Random(7)
        out = mut._wfc_pixels(info, 4096, None)

        assert len(out.pixel_data) == len(pixels), "length preserved by fallback"
        diff = sum(a != b for a, b in zip(out.pixel_data, pixels, strict=False))
        assert diff <= 8, f"fallback flips at most 8 bytes, got {diff}"

    def test_affordable_row_still_runs_wfc(self):
        """A small alphabet row keeps the structural generator (no fallback)."""
        width, height = 16, 2
        pixels = bytes(i % 4 for i in range(width * height))
        info = self._info(width=width, height=height, bit_count=8, pixels=pixels)

        mut = BmpMutator()
        mut.use_wfc = True
        mut._rng = random.Random(3)
        t0 = time.perf_counter()
        out = mut._wfc_pixels(info, 4096, None)
        assert time.perf_counter() - t0 < 5
        assert len(out.pixel_data) == width * height

    def test_regression_all_distinct_24bit_row_does_not_hang(self):
        """Wall-clock guard for the exact reported hang shape (wide BMP row,
        large alphabet): a mutation must return in bounded time."""
        width, height, bpp = 1024, 4, 3
        stride = ((width * bpp + 3) // 4) * 4
        pixels = bytearray(stride * height)
        for y in range(height):
            for x in range(width):
                v = (x * 251 + y * 61 + 7) % 256
                off = y * stride + x * bpp
                pixels[off] = v
                pixels[off + 1] = (v + 1) % 256
                pixels[off + 2] = (v + 2) % 256
        info = self._info(width=width, height=height, bit_count=24, pixels=bytes(pixels))

        mut = BmpMutator()
        mut.use_wfc = True
        mut._rng = random.Random(5)
        t0 = time.perf_counter()
        out = mut._wfc_pixels(info, 4096, None)
        dt = time.perf_counter() - t0
        assert dt < 5, f"BMP WFC mutation took {dt:.1f}s (pre-fix: minutes)"
        assert len(out.pixel_data) == stride * height
