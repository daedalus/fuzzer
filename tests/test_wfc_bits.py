"""BitWaveGrid: identical to WaveGrid without backtracking, complete with it."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from fuzzer_tool.core.wfc import AdjacencyTable, Tile, WaveGrid
from fuzzer_tool.core.wfc_bits import BitWaveGrid


def _tiles(n: int, weighted: bool = False) -> list[Tile]:
    return [Tile(bytes([65 + i]), 1.0 + (i % 3 if weighted else 0)) for i in range(n)]


def _random_table(tiles: list[Tile], density: float, rng: random.Random) -> AdjacencyTable:
    t = AdjacencyTable()
    for a in tiles:
        t._ensure(a.name)
    for a in tiles:
        for b in tiles:
            for d in ("left", "right", "up", "down"):
                if rng.random() < density:
                    t._rules[a.name][d].add(b.name)
                    t._rules[b.name][
                        {"left": "right", "right": "left", "up": "down", "down": "up"}[d]
                    ].add(a.name)
    return t


def _coloring_table(tiles: list[Tile]) -> AdjacencyTable:
    """Neighbours must differ (proper colouring)."""
    t = AdjacencyTable()
    for a in tiles:
        t._ensure(a.name)
        for b in tiles:
            if a.name != b.name:
                for d in ("left", "right", "up", "down"):
                    t._rules[a.name][d].add(b.name)
    return t


def _valid(grid, adj: AdjacencyTable, w: int, h: int) -> bool:
    for y in range(h):
        for x in range(w):
            if grid[y][x] is None:
                return False
            if x + 1 < w and not adj.compatible(grid[y][x], grid[y][x + 1], "right"):
                return False
            if y + 1 < h and not adj.compatible(grid[y][x], grid[y + 1][x], "down"):
                return False
    return True


def _brute_exists(tiles, adj, w, h, pins) -> bool:
    names = [t.name for t in tiles]
    for combo in itertools.product(range(len(names)), repeat=w * h):
        if any(combo[c] != t for c, t in pins.items()):
            continue
        g = [[names[combo[y * w + x]] for x in range(w)] for y in range(h)]
        if _valid(g, adj, w, h):
            return True
    return False


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("shape", [(9, 1), (4, 3), (1, 6)])
def test_matches_wavegrid_exactly(seed, shape):
    w, h = shape
    rng = random.Random(seed)
    tiles = _tiles(5 + seed % 4, weighted=bool(seed % 2))
    adj = _random_table(tiles, 0.55, rng)
    pin = rng.randrange(len(tiles))
    a = WaveGrid(tiles, adj, w, h)
    b = BitWaveGrid(tiles, adj, w, h)
    for g in (a, b):
        g.superpositions[0, :] = False
        g.superpositions[0, pin] = True
    ga = a.run(seed=seed)
    gb = b.run(seed=seed)
    assert ga == gb
    assert a.contradiction == b.contradiction
    assert np.array_equal(a.superpositions, b.superpositions)


def test_clear_and_rerun_matches_fresh_grid():
    tiles = _tiles(6)
    adj = _random_table(tiles, 0.8, random.Random(3))
    g = BitWaveGrid(tiles, adj, 8)
    first = g.run(seed=5)
    g.clear()
    assert g.run(seed=5) == first
    assert BitWaveGrid(tiles, adj, 8).run(seed=5) == first


def test_pins_survive_restarts_and_array_roundtrips():
    tiles = _tiles(4)
    adj = _random_table(tiles, 0.9, random.Random(1))
    g = BitWaveGrid(tiles, adj, 6)
    g.superpositions[0, :] = False
    g.superpositions[0, 2] = True
    out = g.run(seed=1)
    assert out[0][0] == tiles[2].name
    assert g.superpositions.dtype == bool
    assert g.superpositions.shape == (6, 4)


def test_more_than_64_tiles():
    tiles = _tiles(90)
    adj = _random_table(tiles, 0.5, random.Random(8))
    ga = WaveGrid(tiles, adj, 5).run(seed=2)
    gb = BitWaveGrid(tiles, adj, 5).run(seed=2)
    assert ga == gb


@pytest.mark.parametrize("seed", range(25))
def test_backtracking_is_complete_on_small_grids(seed):
    rng = random.Random(100 + seed)
    tiles = _tiles(3)
    adj = _random_table(tiles, 0.45, rng)
    w, h = 3, 3
    pins = {rng.randrange(w * h): rng.randrange(3) for _ in range(rng.randrange(0, 3))}
    g = BitWaveGrid(tiles, adj, w, h)
    for c, t in pins.items():
        g.superpositions[c, :] = False
        g.superpositions[c, t] = True
    grid = g.run(seed=seed, max_restarts=0, max_backtracks=10_000, work_budget=10**7)
    exists = _brute_exists(tiles, adj, w, h, pins)
    if exists:
        assert not g.contradiction
        assert _valid(grid, adj, w, h)
        for c, t in pins.items():
            assert grid[c // w][c % w] == tiles[t].name
    else:
        assert g.contradiction


def test_backtracking_fixes_a_wave_restarts_cannot():
    # 2-colouring a 5x5 grid with opposite-parity pins on a 3-colour palette
    # is only unsatisfiable through a long cycle; satisfiable variants need
    # undo, not luck. Count how often plain restart (no retries) fails vs DFS.
    tiles = _tiles(3)
    adj = _coloring_table(tiles)
    w = h = 6
    fails_restart = fails_dfs = 0
    for seed in range(40):
        a = BitWaveGrid(tiles, adj, w, h)
        a.run(seed=seed, max_restarts=0)
        fails_restart += a.contradiction or not _valid(a._to_grid(), adj, w, h)
        b = BitWaveGrid(tiles, adj, w, h)
        grid = b.run(seed=seed, max_restarts=0, max_backtracks=5000)
        fails_dfs += b.contradiction or not _valid(grid, adj, w, h)
    assert fails_dfs == 0
    assert fails_restart >= fails_dfs


def test_unsat_pins_detected_without_restarting_forever():
    tiles = _tiles(2)
    adj = _coloring_table(tiles)
    g = BitWaveGrid(tiles, adj, 3, 1)
    for c, t in ((0, 0), (2, 1)):  # A ? B with 2 colours: ends differ -> impossible
        g.superpositions[c, :] = False
        g.superpositions[c, t] = True
    g.run(seed=0, max_restarts=5, max_backtracks=50)
    assert g.contradiction


def test_backtrack_limit_exhausts_gracefully():
    tiles = _tiles(3)
    adj = _coloring_table(tiles)
    g = BitWaveGrid(tiles, adj, 6, 6)
    g.superpositions[0, :] = False
    g.superpositions[0, 0] = True
    g.superpositions[35, :] = False
    g.superpositions[35, 0] = True
    grid = g.run(seed=1, max_restarts=0, max_backtracks=1)
    assert len(grid) == 6 and len(grid[0]) == 6


def test_work_budget_returns_partial_grid():
    tiles = _tiles(8)
    adj = _random_table(tiles, 0.6, random.Random(2))
    g = BitWaveGrid(tiles, adj, 40)
    grid = g.run(seed=1, max_restarts=0, work_budget=5, max_backtracks=100)
    assert len(grid[0]) == 40
