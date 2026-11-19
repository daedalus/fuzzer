"""Tests for the Kohonen seed-coverage map (`core/seed_som.py`), report only."""

from __future__ import annotations

import math
import types

import pytest

from fuzzer_tool.core import seed_som
from fuzzer_tool.core.edge_set import EdgeSet
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.seed_som import SeedSom, coverage_matrix, fit_tracker
from fuzzer_tool.services.report import _seed_som


def _two_clusters(per: int = 10) -> dict[str, set[int]]:
    """Two disjoint coverage blobs; one private edge per seed keeps rows distinct."""
    seeds = {f"a{i:02d}": set(range(20)) | {200 + i} for i in range(per)}
    seeds |= {f"b{i:02d}": set(range(100, 120)) | {300 + i} for i in range(per)}
    return seeds


def _chain(n: int = 30, width: int = 10) -> dict[str, set[int]]:
    """Seed i covers edges [i, i + width): neighbours in i overlap most."""
    return {f"s{i:02d}": set(range(i, i + width)) for i in range(n)}


def _grid_dist(som: SeedSom, u: int, v: int) -> float:
    cols = som.shape[1]
    return math.dist(divmod(u, cols), divmod(v, cols))


class TestCoverageMatrix:
    def test_drops_edges_every_seed_hits(self):
        keys, edges, x = coverage_matrix({"a": {1, 2}, "b": {1, 3}})
        assert keys == ["a", "b"]
        assert edges.tolist() == [2, 3]
        assert x.tolist() == [[1, 0], [0, 1]]

    def test_column_cap_keeps_highest_variance(self, monkeypatch):
        # 4 seeds: edge 10 in 2/4 (p(1-p) = .25), edge 11 in 1/4 (.1875).
        monkeypatch.setattr(seed_som, "MAX_CELLS", 4)
        seeds = {"a": {10, 11}, "b": {10}, "c": set(), "d": set()}
        _, edges, _ = coverage_matrix(seeds)
        assert edges.tolist() == [10]

    def test_accepts_edge_set(self):
        top = 2**32 - 1  # EdgeSet is uint32
        _, edges, _ = coverage_matrix({"a": EdgeSet([top]), "b": EdgeSet([0])})
        assert edges.tolist() == [0, top]

    def test_huge_edge_ids(self):
        big = 2**64 - 1
        _, edges, x = coverage_matrix({"a": {big}, "b": {0}})
        assert edges.tolist() == [0, big]
        assert x.tolist() == [[0, 1], [1, 0]]

    # Adversarial: degenerate inputs give no map, not a crash.
    @pytest.mark.parametrize(
        "seeds",
        [{}, {"a": {1, 2}}, {"a": {1}, "b": {1}}, {"a": set(), "b": set()}],
        ids=["empty", "one-seed", "identical", "no-edges"],
    )
    def test_degenerate_none(self, seeds):
        assert coverage_matrix(seeds) is None


class TestSeedSom:
    def test_separates_disjoint_clusters(self):
        # Falsification: a map ignoring coverage would mix the blobs.
        som = SeedSom(coverage_matrix(_two_clusters())[2])
        a, b = set(som.bmu[:10].tolist()), set(som.bmu[10:].tolist())
        assert a.isdisjoint(b)

    def test_chain_keeps_topology(self):
        som = SeedSom(coverage_matrix(_chain())[2])
        steps = [_grid_dist(som, som.bmu[i], som.bmu[i + 1]) for i in range(len(som.bmu) - 1)]
        assert max(steps) <= math.sqrt(2)  # chain neighbours land on adjacent units
        assert len(set(som.bmu.tolist())) >= som.hits.size // 2  # chain spread, not collapsed

    def test_hits_count_every_seed(self):
        som = SeedSom(coverage_matrix(_chain())[2])
        assert int(som.hits.sum()) == len(_chain())

    def test_quantization_error_brute_force(self):
        # Independent loop over rows and units, no vectorized distance.
        x = coverage_matrix(_two_clusters())[2]
        som = SeedSom(x)
        w = som.weights
        best = [min(math.dist(row, unit) for unit in w.tolist()) for row in x.tolist()]
        assert som.quantization_error == pytest.approx(sum(best) / len(best), rel=1e-4)

    def test_topographic_error_brute_force(self):
        x = coverage_matrix(_chain())[2]
        som = SeedSom(x)
        w = som.weights.tolist()
        misses = 0
        for row in x.tolist():
            first, second = sorted(range(len(w)), key=lambda u: math.dist(row, w[u]))[:2]
            misses += _grid_dist(som, first, second) > math.sqrt(2)
        assert som.topographic_error == pytest.approx(misses / len(x))

    def test_deterministic(self):
        x = coverage_matrix(_two_clusters())[2]
        assert SeedSom(x).bmu.tolist() == SeedSom(x).bmu.tolist()

    def test_umatrix_shape_and_sign(self):
        som = SeedSom(coverage_matrix(_two_clusters())[2])
        assert som.umatrix.shape == som.shape
        assert (som.umatrix >= 0).all()

    @pytest.mark.parametrize(("n", "side"), [(1, 2), (100, 7), (10**6, seed_som.MAX_SIDE)])
    def test_grid_side(self, n, side):
        # 5 * sqrt(n) units (Vesanto), square, clamped to [2, MAX_SIDE].
        assert seed_som.grid_side(n) == side


class TestFitTracker:
    def test_none_without_tracker(self):
        assert fit_tracker(None) is None
        assert fit_tracker(EdgeTracker()) is None

    def test_over_budget(self, monkeypatch):
        monkeypatch.setattr(seed_som, "REPORT_NNZ_BUDGET", 5)
        assert fit_tracker(types.SimpleNamespace(seed_edges=_two_clusters())) is None

    def test_fits(self):
        som = fit_tracker(types.SimpleNamespace(seed_edges=_two_clusters()))
        assert som is not None and int(som.hits.sum()) == 20


class TestReportSection:
    def test_section(self):
        f = types.SimpleNamespace(_edge_tracker=types.SimpleNamespace(seed_edges=_two_clusters()))
        text = _seed_som(f)
        assert "--- Seed Coverage Map (SOM) ---" in text
        assert "20 seeds x " in text
        side = seed_som.grid_side(20)
        assert len([ln for ln in text.splitlines() if ln.startswith("    ")]) == side

    def test_silent_without_seeds(self):
        assert _seed_som(types.SimpleNamespace(_edge_tracker=EdgeTracker())) == ""
        assert _seed_som(types.SimpleNamespace()) == ""
