"""Regression tests for the fractal Voronoi geometry caches.

The operator's partition is a function of the grid geometry alone. These
tests pin the three properties that make caching it legitimate, plus the
exactness of the 3x3-first nearest-site search, against a verbatim copy of
the pre-optimisation implementation used as an oracle.
"""

from __future__ import annotations

import hashlib
import math
import os

import pytest

from fuzzer_tool.core.mutations.fractal_voronoi import FractalVoronoiMutator

# ---------------------------------------------------------------------------
# Oracle: the pre-optimisation implementation, verbatim.
# ---------------------------------------------------------------------------


def _old_nearest_site(m: FractalVoronoiMutator, layer: int, p: tuple[float, float]):
    """Full 5x5 sweep, the original search."""
    s = 2.0 ** (-layer)
    cx = int(p[0] / s)
    cy = int(p[1] / s)
    best_cell = None
    best_site = None
    best_distance = float("inf")
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            cell = (cx + dx, cy + dy)
            q = m._site(layer, cell)
            d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d < best_distance:
                best_distance = d
                best_cell = cell
                best_site = q
    return best_cell, best_site


def _old_mutate(m: FractalVoronoiMutator, data: bytes, max_len: int = 0):
    """Per-byte recomputation, the original mutate body."""
    if len(data) < 16:
        return None
    side = max(1, int(math.sqrt(len(data))))
    out = bytearray(data)
    for idx in range(len(data)):
        y, x = divmod(idx, side)
        px = (x + 0.5) / side
        py = (y + 0.5) / side
        cell, _ = _old_nearest_site(m, m.max_depth, (px, py))
        root = m._root(m.max_depth, cell)
        root_hash = int(hashlib.sha256(f"root:{root}".encode()).hexdigest(), 16)
        if m.cell_ops:
            op = m.cell_ops[root_hash % len(m.cell_ops)]
            if (root_hash + idx) % 7 == 0:
                mutated_byte = op(bytes([data[idx]]))
                if mutated_byte and len(mutated_byte) == 1:
                    out[idx] = mutated_byte[0]
        else:
            if (root_hash + idx) % 5 == 0:
                out[idx] ^= root_hash & 0xFF
        # Original boundary test: recomputes the cell from the point.
        bcell, _ = _old_nearest_site(m, m.max_depth, (px, py))
        broot = m._root(m.max_depth, bcell)
        on_boundary = False
        for dx, dy in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            if m._root(m.max_depth, (bcell[0] + dx, bcell[1] + dy)) != broot:
                on_boundary = True
                break
        if on_boundary:
            boundary_hash = int(
                hashlib.sha256(
                    f"boundary:{root}:{px:.6f}:{py:.6f}".encode()
                ).hexdigest(),
                16,
            )
            if (boundary_hash + idx) % 3 == 0:
                out[idx] ^= (boundary_hash >> 8) & 0xFF
    result = bytes(out)
    if max_len and len(result) > max_len:
        result = result[:max_len]
    return result


# ---------------------------------------------------------------------------


_SIZES = [16, 17, 31, 63, 64, 100, 255, 256, 257, 999, 1024, 2048]


@pytest.mark.parametrize("n", _SIZES)
def test_mutate_matches_per_byte_oracle(n):
    """The cached plan reproduces the per-byte implementation byte for byte."""
    data = os.urandom(n)
    m = FractalVoronoiMutator()
    assert m.mutate(data, None) == _old_mutate(FractalVoronoiMutator(), data)


@pytest.mark.parametrize("n", [64, 256, 1024])
def test_mutate_matches_oracle_with_cell_ops(n):
    """The sub-operator path is unaffected by the plan cache."""
    ops = [
        lambda b: bytes([(b[0] + 1) & 0xFF]),
        lambda b: bytes([b[0] ^ 0x5A]),
        lambda b: bytes([255 - b[0]]),
    ]
    data = os.urandom(n)
    assert FractalVoronoiMutator(cell_ops=ops).mutate(data, None) == _old_mutate(
        FractalVoronoiMutator(cell_ops=ops), data
    )


@pytest.mark.parametrize("layer", [0, 1, 2, 3, 4])
def test_nearest_site_3x3_expansion_is_exact(layer):
    """The 3x3-first search returns what the full 5x5 sweep returns."""
    m = FractalVoronoiMutator()
    n = 40
    for i in range(n):
        for j in range(n):
            p = ((i + 0.5) / n, (j + 0.5) / n)
            assert m._nearest_site(layer, p) == _old_nearest_site(m, layer, p)


def test_boundary_is_a_property_of_the_cell_not_the_point():
    """Every point inside one cell shares one boundary answer.

    This is what licenses caching the test on the cell. If it ever stops
    holding, _boundary_cell is returning a stale answer for some points.
    """
    m = FractalVoronoiMutator()
    seen: dict[tuple[int, int], bool] = {}
    n = 64
    for i in range(n):
        for j in range(n):
            px, py = (i + 0.5) / n, (j + 0.5) / n
            cell, _ = m._nearest_site(m.max_depth, (px, py))
            answer = m._is_boundary(m.max_depth, px, py)
            if cell in seen:
                assert seen[cell] == answer, f"cell {cell} gave two answers"
            seen[cell] = answer
    assert len(seen) > 1, "test degenerate: only one cell sampled"


def test_plan_is_independent_of_buffer_contents():
    """Two different buffers of the same length share one plan."""
    m = FractalVoronoiMutator()
    a = m._plan(16, 256)
    b = FractalVoronoiMutator()._plan(16, 256)
    assert a == b
    # And the mutation differs only where the input differs in a way the
    # plan reacts to -- i.e. the plan itself did not change.
    m.mutate(os.urandom(256), None)
    assert m._plan(16, 256) == a


def test_plan_is_reused_across_calls_of_the_same_length():
    """Repeat calls at one length must not rebuild the geometry."""
    m = FractalVoronoiMutator()
    m.mutate(os.urandom(400), None)
    assert len(m._plan_cache) == 1
    cached = next(iter(m._plan_cache.values()))
    for _ in range(5):
        m.mutate(os.urandom(400), None)
    assert len(m._plan_cache) == 1
    assert next(iter(m._plan_cache.values())) is cached


def test_plan_cache_is_bounded_and_per_instance():
    m = FractalVoronoiMutator()
    for n in range(64, 64 + m._PLAN_CACHE_MAX + 4):
        m.mutate(os.urandom(n), None)
    assert len(m._plan_cache) <= m._PLAN_CACHE_MAX
    assert FractalVoronoiMutator()._plan_cache == {}


def test_max_len_still_truncates():
    data = os.urandom(512)
    out = FractalVoronoiMutator().mutate(data, None, max_len=64)
    assert len(out) == 64


def test_short_input_returns_none():
    assert FractalVoronoiMutator().mutate(b"x" * 15, None) is None


def test_nearest_site_full_sweep_for_negative_coordinates():
    """Negative points must not take the 3x3 fast path.

    ``int()`` truncates toward zero, so the cell index is not a floor for
    negative coordinates and the "two cells out is farther than s" bound
    does not hold there. Measured over 200k signed probes, the 3x3 core
    disagrees with the full sweep 10.5% of the time -- and in 97% of those
    the distance guard alone would not have triggered the expansion.
    ``_root`` reaches negative coordinates via parent sites, so this path
    is live.
    """
    m = FractalVoronoiMutator()
    import random

    rnd = random.Random(0)
    for _ in range(4000):
        layer = rnd.randrange(0, 7)
        p = (rnd.uniform(-2.0, 2.0), rnd.uniform(-2.0, 2.0))
        assert m._nearest_site(layer, p) == _old_nearest_site(m, layer, p)

