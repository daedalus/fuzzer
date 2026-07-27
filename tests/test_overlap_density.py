"""Tests for FMM-clustered pairwise overlap density (overlap_density.py)."""

import math
import random

from fuzzer_tool.core.edge_tracker import EdgeTracker, MinHashLSH
from fuzzer_tool.core.overlap_density import (
    compute_corpus_overlap_density,
)


def _make_synthetic_seeds(
    num_clusters: int = 3,
    seeds_per_cluster: int = 5,
    core_edges: int = 40,
    individual_edges: int = 10,
) -> dict[str, set[int]]:
    """Generate seed edge sets with controlled cluster structure."""
    seeds: dict[str, set[int]] = {}
    edge_offset = 0
    rng = random.Random(42)

    for c in range(num_clusters):
        core = set(range(edge_offset, edge_offset + core_edges))
        edge_offset += core_edges + 30
        for s in range(seeds_per_cluster):
            sk = f"c{c}_s{s}"
            indiv = set(rng.sample(range(edge_offset, edge_offset + 500), k=individual_edges))
            seeds[sk] = core | indiv
    return seeds


def _register_seeds(et: EdgeTracker, seeds: dict[str, set[int]]):
    for sk, edges in seeds.items():
        et.record_edges(sk, edges)


# ── Naive O(N²) reference ──────────────────────────────────────────────

def _naive_densities(seed_keys: list[str], minhash: MinHashLSH) -> dict[str, float]:
    n = len(seed_keys)
    densities: dict[str, float] = {}
    for i in range(n):
        total = 0.0
        for j in range(n):
            if i == j:
                continue
            total += minhash.approximate_jaccard(seed_keys[i], seed_keys[j])
        densities[seed_keys[i]] = total / (n - 1) if n > 1 else 0.0
    return densities


# ── Tests ───────────────────────────────────────────────────────────────

class TestFMMVsNaive:
    def test_small_corpus(self):
        """N=15, 3 clusters. FMM should be close to naive."""
        seeds = _make_synthetic_seeds(num_clusters=3, seeds_per_cluster=5)
        et = EdgeTracker(map_size=65536, max_tracked_seeds=1000)
        _register_seeds(et, seeds)
        seed_keys = list(seeds.keys())

        naive = _naive_densities(seed_keys, et._minhash)
        fmm, clusters, _stc = compute_corpus_overlap_density(
            seed_keys, et._minhash, min_jaccard=0.25
        )

        # FMM vs naive: MAE should be low
        errors = [abs(naive[sk] - fmm[sk]) for sk in seed_keys if sk in fmm]
        mae = sum(errors) / len(errors) if errors else 0.0
        assert mae < 0.05, f"MAE too high: {mae:.6f}"

        # Clusters should exist (we have 3 clusters)
        assert len(clusters) >= 2, f"Expected >=2 clusters, got {len(clusters)}"

    def test_large_corpus(self):
        """N=60, 5 clusters. Accuracy should hold at scale."""
        seeds = _make_synthetic_seeds(num_clusters=5, seeds_per_cluster=12)
        et = EdgeTracker(map_size=65536, max_tracked_seeds=1000)
        _register_seeds(et, seeds)
        seed_keys = list(seeds.keys())

        naive = _naive_densities(seed_keys, et._minhash)
        fmm, clusters, _stc = compute_corpus_overlap_density(
            seed_keys, et._minhash, min_jaccard=0.25
        )

        errors = [abs(naive[sk] - fmm[sk]) for sk in seed_keys if sk in fmm]
        mae = sum(errors) / len(errors) if errors else 0.0
        assert mae < 0.10, f"MAE too high: {mae:.6f}"

    def test_identical_sets(self):
        """All seeds have identical edge sets → densities ≈ 1.0."""
        seeds = {}
        for i in range(10):
            seeds[f"s{i}"] = {1, 2, 3, 4, 5}
        et = EdgeTracker(map_size=65536, max_tracked_seeds=1000)
        _register_seeds(et, seeds)
        seed_keys = list(seeds.keys())

        fmm, _clusters, _stc = compute_corpus_overlap_density(
            seed_keys, et._minhash, min_jaccard=0.25
        )
        for sk in seed_keys:
            assert fmm.get(sk, 0.0) > 0.95, f"{sk} density {fmm.get(sk)} not close to 1.0"

    def test_disjoint_sets(self):
        """All seeds have disjoint edge sets → densities ≈ 0.0."""
        seeds = {}
        for i in range(10):
            seeds[f"s{i}"] = {i * 100, i * 100 + 1, i * 100 + 2}
        et = EdgeTracker(map_size=65536, max_tracked_seeds=1000)
        _register_seeds(et, seeds)
        seed_keys = list(seeds.keys())

        fmm, _clusters, _stc = compute_corpus_overlap_density(
            seed_keys, et._minhash, min_jaccard=0.25
        )
        for sk in seed_keys:
            assert fmm.get(sk, 1.0) < 0.05, f"{sk} density {fmm.get(sk)} not close to 0.0"

    def test_single_seed(self):
        """Single seed → density 0.0."""
        seeds = {"s0": {1, 2, 3}}
        et = EdgeTracker(map_size=65536, max_tracked_seeds=1000)
        _register_seeds(et, seeds)
        seed_keys = list(seeds.keys())

        fmm, _clusters, _stc = compute_corpus_overlap_density(
            seed_keys, et._minhash, min_jaccard=0.25
        )
        assert fmm.get("s0", 1.0) == 0.0

    def test_empty_corpus(self):
        """Empty seed list → empty dict."""
        et = EdgeTracker(map_size=65536, max_tracked_seeds=1000)
        fmm, _clusters, _stc = compute_corpus_overlap_density(
            [], et._minhash, min_jaccard=0.25
        )
        assert fmm == {}

    def test_two_seeds_same_cluster(self):
        """Two seeds with identical edge sets → both density 1.0."""
        seeds = {"a": {1, 2, 3}, "b": {1, 2, 3}}
        et = EdgeTracker(map_size=65536, max_tracked_seeds=1000)
        _register_seeds(et, seeds)
        seed_keys = list(seeds.keys())

        fmm, _clusters, _stc = compute_corpus_overlap_density(
            seed_keys, et._minhash, min_jaccard=0.25
        )
        # Both should be near 1.0 (identical)
        assert abs(fmm.get("a", 0.0) - 1.0) < 0.05
        assert abs(fmm.get("b", 0.0) - 1.0) < 0.05

    def test_two_seeds_disjoint(self):
        """Two seeds with disjoint edge sets → both density 0.0."""
        seeds = {"a": {1, 2, 3}, "b": {100, 101, 102}}
        et = EdgeTracker(map_size=65536, max_tracked_seeds=1000)
        _register_seeds(et, seeds)
        seed_keys = list(seeds.keys())

        fmm, _clusters, _stc = compute_corpus_overlap_density(
            seed_keys, et._minhash, min_jaccard=0.25
        )
        assert fmm.get("a", 1.0) < 0.05
        assert fmm.get("b", 1.0) < 0.05

    def test_corpus_versions_match(self):
        """FMM's seed_to_cluster should be consistent with clusters."""
        seeds = _make_synthetic_seeds(num_clusters=2, seeds_per_cluster=4)
        et = EdgeTracker(map_size=65536, max_tracked_seeds=1000)
        _register_seeds(et, seeds)
        seed_keys = list(seeds.keys())

        _fmm, clusters, seed_to_cluster = compute_corpus_overlap_density(
            seed_keys, et._minhash, min_jaccard=0.25
        )

        # Every seed index should map to a cluster
        for i in range(len(seed_keys)):
            assert i in seed_to_cluster, f"Seed {i} not in seed_to_cluster"

        # Cluster membership should be consistent
        for cidx, members in enumerate(clusters):
            for m in members:
                assert seed_to_cluster[m] == cidx, f"Seed {m} maps to {seed_to_cluster[m]} not {cidx}"
