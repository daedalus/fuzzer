"""Tests for FMM-clustered pairwise overlap density (overlap_density.py)."""

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
        fmm, _clusters, _stc = compute_corpus_overlap_density([], et._minhash, min_jaccard=0.25)
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
                assert seed_to_cluster[m] == cidx, (
                    f"Seed {m} maps to {seed_to_cluster[m]} not {cidx}"
                )


class TestFMMCohesionGate:
    """Regression test: the far-field cohesion gate bounds centroid-approximation error."""

    def _make_diverse_overlap_seeds(self) -> dict[str, set[int]]:
        """Generate a pathological case for the far-field centroid approximation.

        Creates a query seed and a separate cluster whose members each overlap
        with a *different* slice of the query seed's edges.  The centroid of
        the far cluster (element-wise min = union) accumulates all slices,
        making J(query, centroid) much larger than J(query, any single member).

        Without the cohesion gate, this produces a ~4-5x density overestimate
        for the query seed.  The cohesion gate detects that the far cluster
        members are diverse (low min-J-to-centroid) and falls back to exact
        pairwise computation.

        Edge layout:
          s0 (query):  edges 0-99                          → 100 edges
          far_i:       1000-1029 (base)                     → 30 edges shared
                       i*5..(i+1)*5-1 (query slice)        → 5 edges overlapping query
                       2000+i*100..2000+i*100+14 (private)  → 15 unique edges
                                                              50 total per member

        J(far_i, far_j) ≈ 30/70 ≈ 0.43 (high enough for LSH with fine bands).
        J(far_i, centroid) ≈ 50/190 ≈ 0.26 < 0.3 (low cohesion → gate triggers).
        J(s0, centroid) ≈ 40/250 = 0.16 vs true J(s0, far_i) ≈ 5/145 ≈ 0.034.
        Centroid estimate over 8 members: 0.16×8 vs true 0.034×8 ≈ 4.7x overestimate.
        """
        seeds: dict[str, set[int]] = {}
        query_edges = set(range(100))
        seeds["s0"] = query_edges
        base = set(range(1000, 1030))
        for i in range(8):
            slice_ = set(range(i * 5, (i + 1) * 5))
            private = set(range(2000 + i * 100, 2000 + i * 100 + 15))
            seeds[f"far_{i}"] = base | slice_ | private
        return seeds

    def _register_seeds_to_minhash(
        self, seeds: dict[str, set[int]], num_bands: int = 16
    ) -> MinHashLSH:
        """Register seeds to a MinHashLSH instance with specified band count.

        Uses more bands (= smaller bands = more sensitive LSH) so that
        the far-cluster members (J ~0.45-0.50) actually collide in LSH buckets
        and get clustered together.
        """
        mh = MinHashLSH(num_perm=64, num_bands=num_bands, seed=42)
        for sk, edges in seeds.items():
            sig = mh.compute_signature(edges)
            mh.add(sk, sig)
        return mh

    def test_regression_diverse_overlap_farfield(self):
        """Cohesion gate bounds the far-field overestimate from diverse-overlap clusters.

        A far cluster whose members each overlap with a different slice of the
        query seed's edges causes the centroid approximation to overestimate
        density.  The cohesion gate detects this (low min-J-to-centroid) and
        falls back to exact computation, bounding the error.
        """
        seeds = self._make_diverse_overlap_seeds()
        seed_keys = list(seeds.keys())
        mh = self._register_seeds_to_minhash(seeds, num_bands=16)

        naive = _naive_densities(seed_keys, mh)

        # FMM with cohesion gate enabled (default threshold 0.3)
        fmm_gated, clusters_gated, _stc = compute_corpus_overlap_density(
            seed_keys, mh, min_jaccard=0.25, cohesion_threshold=0.3
        )
        errors_gated = [abs(naive[sk] - fmm_gated[sk]) for sk in seed_keys]
        max_err_gated = max(errors_gated)

        # Verify the diverse-overlap cluster is actually formed (precondition)
        # We need at least 2 clusters and the far cluster should have >1 member
        far_cluster_found = any(len(m) > 1 for m in clusters_gated)
        assert far_cluster_found, "Test precondition failed: far members did not cluster via LSH"

        # FMM with cohesion gate disabled (threshold=0 = unconditional approximation)
        fmm_ungated, _clusters2, _stc2 = compute_corpus_overlap_density(
            seed_keys, mh, min_jaccard=0.25, cohesion_threshold=0.0
        )
        errors_ungated = [abs(naive[sk] - fmm_ungated[sk]) for sk in seed_keys]
        max_err_ungated = max(errors_ungated)

        # Without the cohesion gate, the far-field overestimate dominates
        assert max_err_ungated > 0.10, (
            f"Without cohesion gate, max error should be > 0.10 "
            f"but got {max_err_ungated:.6f}. The test construction may no "
            f"longer trigger the diverse-overlap pathology."
        )
        # With the gate, the worst-case error must be bounded
        assert max_err_gated < 0.10, (
            f"With cohesion gate, max error should be < 0.10 but got {max_err_gated:.6f}"
        )
        # The gate must improve the worst case
        assert max_err_gated < max_err_ungated, (
            f"Cohesion gate increased worst-case error "
            f"({max_err_gated:.6f} >= {max_err_ungated:.6f})"
        )

    def test_regression_high_cohesion_gate_does_not_trigger(self):
        """When far-cluster cohesion is high, the gate must not trigger.

        A far cluster whose members share a large common base plus small
        per-member query-slices has high cohesion (0.86-0.90), so the
        cohesion gate (threshold=0.3) never engages.  The centroid
        approximation overestimates the true pairwise density by ~2.4x
        because the union of far members has more overlap with the query
        seed than any individual member.

        Edge layout:
          s0 (query):   edges 0-99                        → 100 edges
          far_i:        common_base 1000-1079              → 80 edges (20 overlap s0)
                        query_slice_i  (i*5..i*5+4)        →  5 edges (overlap s0, disjoint)
                        private_i      (2000+i*10..+14)    →  5 edges (no overlap)
                        total per member = 90 edges

        J(far_i, far_j) = 80 / (90+90-80) = 0.80
        J(far_i, union)  = 90 / (80+8*5+8*5) = 90/160 = 0.562
        Cohesion ≈ 0.562 > 0.3  → gate does not trigger.

        True J(s0, far_i) = 25 / (100+90-25) = 25/165 ≈ 0.152
        Approx J(s0, union) = (20+40) / (100+160-60) = 60/200 = 0.300
        Overestimate factor ≈ 0.300 / 0.152 ≈ 1.97x
        """
        seeds: dict[str, set[int]] = {}
        seeds["s0"] = set(range(100))
        common_base = set(range(1000, 1080))
        for i in range(8):
            query_slice = set(range(i * 5, (i + 1) * 5))
            private = set(range(2000 + i * 10, 2000 + i * 10 + 5))
            seeds[f"far_{i}"] = common_base | query_slice | private

        seed_keys = list(seeds.keys())
        mh = self._register_seeds_to_minhash(seeds, num_bands=16)

        naive = _naive_densities(seed_keys, mh)

        fmm_gated, clusters, _stc = compute_corpus_overlap_density(
            seed_keys, mh, min_jaccard=0.25, cohesion_threshold=0.3
        )

        fmm_ungated, _clusters2, _stc2 = compute_corpus_overlap_density(
            seed_keys, mh, min_jaccard=0.25, cohesion_threshold=0.0
        )

        for sk in seed_keys:
            assert abs(fmm_gated[sk] - fmm_ungated[sk]) < 0.05, (
                f"Cohesion gate appears to have triggered for {sk}: "
                f"gated={fmm_gated[sk]:.4f}, ungated={fmm_ungated[sk]:.4f}"
            )

        true_s0 = naive["s0"]
        approx_s0 = fmm_gated["s0"]

        assert true_s0 > 0.01, (
            f"Query seed has near-zero true density ({true_s0:.4f}); "
            f"test construction may not reproduce the overestimate pathology."
        )
        overestimate_factor = approx_s0 / true_s0
        assert overestimate_factor > 1.5, (
            f"Expected significant overestimate (>1.5x) but got {overestimate_factor:.2f}x. "
            f"True={true_s0:.4f}, Approx={approx_s0:.4f}"
        )
