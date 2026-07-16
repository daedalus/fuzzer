"""Tests for edge tracker metrics: Jaccard, JS divergence, Wasserstein."""

import math

from fuzzer_tool.core.edge_tracker import EdgeTracker, _js_divergence


class TestJsDivergence:
    def test_identical_distributions(self):
        p = {0: 0.5, 1: 0.5}
        assert _js_divergence(p, p) == 0.0

    def test_disjoint_distributions(self):
        p = {0: 1.0}
        q = {1: 1.0}
        js = _js_divergence(p, q)
        assert js > 0.0
        assert js <= math.log(2)

    def test_symmetric(self):
        p = {0: 0.3, 1: 0.7}
        q = {0: 0.7, 1: 0.3}
        assert abs(_js_divergence(p, q) - _js_divergence(q, p)) < 1e-10

    def test_empty_distributions(self):
        assert _js_divergence({}, {}) == 0.0

    def test_partial_overlap(self):
        p = {0: 0.5, 1: 0.5}
        q = {0: 0.5, 2: 0.5}
        js = _js_divergence(p, q)
        assert 0.0 < js < math.log(2)


class TestJaccardSubsumption:
    def test_no_edges_returns_one(self):
        et = EdgeTracker()
        assert et.compute_subsumption_weight("missing") == 1.0

    def test_empty_edges_returns_half(self):
        et = EdgeTracker()
        # record_edges with empty bitmap → empty edge set
        et.record_edges("a", b"\x00\x00")
        assert et.compute_subsumption_weight("a") == 0.5

    def test_only_seed_novel(self):
        et = EdgeTracker()
        et.record_edges("a", b"\x01\x02\x03")
        assert et.compute_subsumption_weight("a") == 1.0

    def test_identical_edges_low_weight(self):
        et = EdgeTracker()
        et.record_edges("a", b"\x01\x02\x03\x00")
        et.record_edges("b", b"\x01\x02\x03\x00")
        w = et.compute_subsumption_weight("a")
        # MinHash approximates Jaccard; identical sets → high jaccard → low weight
        assert w < 0.5

    def test_partial_overlap(self):
        et = EdgeTracker()
        et.record_edges("a", b"\x01\x02\x03\x00")
        et.record_edges("b", b"\x00\x02\x03\x04")
        w = et.compute_subsumption_weight("a")
        # Some overlap → medium weight
        assert 0.1 <= w <= 1.0

    def test_disjoint_edges_high_weight(self):
        et = EdgeTracker()
        # Edge 1-2 vs edge 100-101 (far apart → low Jaccard)
        et.record_edges("a", b"\x01\x02" + b"\x00" * 98)
        et.record_edges("b", b"\x00" * 100 + b"\x01\x02")
        w = et.compute_subsumption_weight("a")
        assert w > 0.5


class TestRecordEdges:
    def test_records_hit_counts(self):
        et = EdgeTracker()
        bitmap = bytes([0, 5, 0, 3, 0])
        et.record_edges("a", bitmap)
        assert et.seed_hit_counts["a"] == {1: 5, 3: 3}

    def test_returns_new_edges(self):
        et = EdgeTracker()
        new = et.record_edges("a", bytes([0, 1, 0, 1, 0]))
        assert new == {1, 3}
        new2 = et.record_edges("b", bytes([0, 1, 0, 0, 1]))
        assert new2 == {4}  # edge 1 already seen

    def test_invalidates_aggregate_cache(self):
        et = EdgeTracker()
        et.record_edges("a", bytes([0, 1, 0]))
        _ = et._build_aggregate_distribution()
        et.record_edges("b", bytes([0, 0, 1]))
        assert et._aggregate_cache is None


class TestHitcountDiversity:
    def test_no_data_returns_one(self):
        et = EdgeTracker()
        assert et.compute_hitcount_diversity_weight("missing") == 1.0

    def test_single_seed_returns_one(self):
        et = EdgeTracker()
        # Two seeds with identical hit counts → low JS → weight near 0.5
        et.record_edges("a", b"\x0a\x05\x00")
        et.record_edges("b", b"\x0a\x05\x00")
        w = et.compute_hitcount_diversity_weight("a")
        assert 0.5 <= w <= 2.0

    def test_unusual_profile_gets_boost(self):
        et = EdgeTracker()
        # Most seeds hit edge 0 heavily
        for i in range(10):
            et.record_edges(f"normal_{i}", b"\x64\x01")  # edge0=100, edge1=1
        # One seed hits edge 1 heavily (unusual)
        et.record_edges("unusual", b"\x01\x64")  # edge0=1, edge1=100
        w = et.compute_hitcount_diversity_weight("unusual")
        w_normal = et.compute_hitcount_diversity_weight("normal_0")
        assert w > w_normal


class TestWassersteinDistance:
    def test_identical_profiles_zero(self):
        et = EdgeTracker()
        et.seed_hit_counts["a"] = {10: 5, 20: 3}
        et.seed_hit_counts["b"] = {10: 5, 20: 3}
        assert et.compute_wasserstein_distance("a", "b") == 0.0

    def test_adjacent_profiles_small(self):
        et = EdgeTracker()
        et.seed_hit_counts["a"] = {10: 5}
        et.seed_hit_counts["b"] = {11: 5}
        w = et.compute_wasserstein_distance("a", "b")
        assert 0.0 < w < 5.0

    def test_distant_profiles_large(self):
        et = EdgeTracker()
        et.seed_hit_counts["a"] = {0: 5}
        et.seed_hit_counts["b"] = {1000: 5}
        w = et.compute_wasserstein_distance("a", "b")
        assert w > 900.0

    def test_missing_data_returns_map_size(self):
        et = EdgeTracker()
        et.seed_hit_counts["a"] = {0: 1}
        assert et.compute_wasserstein_distance("a", "missing") == 65536.0

    def test_symmetric(self):
        et = EdgeTracker()
        et.seed_hit_counts["a"] = {10: 3, 20: 7}
        et.seed_hit_counts["b"] = {15: 5, 25: 5}
        assert (
            abs(
                et.compute_wasserstein_distance("a", "b")
                - et.compute_wasserstein_distance("b", "a")
            )
            < 1e-6
        )


class TestCorpusDiversity:
    def test_single_seed_zero(self):
        et = EdgeTracker()
        et.record_edges("a", b"\x01\x00")
        assert et.compute_corpus_diversity() == 0.0

    def test_two_identical_seeds_zero(self):
        et = EdgeTracker()
        et.record_edges("a", b"\x01\x02\x03\x00")
        et.record_edges("b", b"\x01\x02\x03\x00")
        assert et.compute_corpus_diversity() == 0.0

    def test_two_distant_seeds_positive(self):
        et = EdgeTracker()
        # Edge 0 vs edge 100 → different edge sets
        et.record_edges("a", b"\x05" + b"\x00" * 99)
        et.record_edges("b", b"\x00" * 100 + b"\x05")
        assert et.compute_corpus_diversity() > 0.0


class TestWassersteinWeight:
    def test_no_data_returns_one(self):
        et = EdgeTracker()
        assert et.compute_wasserstein_weight("missing") == 1.0

    def test_at_centroid_returns_low(self):
        et = EdgeTracker()
        # All seeds identical → centroid same as seed → low weight
        for i in range(5):
            et.record_edges(f"s{i}", b"\x0a\x05\x03\x00")
        w = et.compute_wasserstein_weight("s0")
        assert 0.5 <= w <= 1.0

    def test_far_from_centroid_returns_high(self):
        et = EdgeTracker()
        # Most seeds at edge 0
        for i in range(10):
            et.record_edges(f"s{i}", b"\x0a\x00\x00")
        # One seed far away at edge 50
        et.record_edges("far", b"\x00" * 50 + b"\x0a")
        w = et.compute_wasserstein_weight("far")
        w_centroid = et.compute_wasserstein_weight("s0")
        assert w > w_centroid


class TestShannonEntropyGlobal:
    def test_empty_returns_zero(self):
        et = EdgeTracker()
        assert et.shannon_entropy_global() == 0.0

    def test_uniform_distribution(self):
        et = EdgeTracker()
        # 4 edges each hit once → uniform → entropy = log2(4) = 2.0
        bitmap = bytes([1, 1, 1, 1])
        et.record_edges("a", bitmap)
        h = et.shannon_entropy_global()
        assert abs(h - 2.0) < 1e-6

    def test_single_edge_zero(self):
        et = EdgeTracker()
        bitmap = bytes([5, 0, 0, 0])
        et.record_edges("a", bitmap)
        h = et.shannon_entropy_global()
        assert h == 0.0

    def test_skewed_distribution_lower(self):
        et = EdgeTracker()
        # Skewed: one edge dominates
        et.record_edges("a", bytes([100, 1, 1, 1]))
        h_skewed = et.shannon_entropy_global()
        # Uniform: all edges equal
        et2 = EdgeTracker()
        et2.record_edges("a", bytes([25, 25, 25, 25]))
        h_uniform = et2.shannon_entropy_global()
        assert h_skewed < h_uniform

    def test_multiple_seeds_accumulate(self):
        et = EdgeTracker()
        et.record_edges("a", bytes([1, 0, 1, 0]))
        et.record_edges("b", bytes([0, 1, 0, 1]))
        h = et.shannon_entropy_global()
        # All 4 edges hit once → uniform → log2(4) = 2.0
        assert abs(h - 2.0) < 1e-6

    def test_monotonically_increases_with_diversity(self):
        et = EdgeTracker()
        et.record_edges("a", bytes([10, 0, 0, 0]))
        h1 = et.shannon_entropy_global()
        et.record_edges("b", bytes([0, 10, 0, 0]))
        h2 = et.shannon_entropy_global()
        et.record_edges("c", bytes([0, 0, 10, 0]))
        h3 = et.shannon_entropy_global()
        assert h1 < h2 < h3


class TestSimpsonDiversityGlobal:
    def test_empty_returns_zero(self):
        et = EdgeTracker()
        assert et.simpson_diversity_global() == 0.0

    def test_single_edge_zero(self):
        et = EdgeTracker()
        et.record_edges("a", bytes([5, 0, 0]))
        assert et.simpson_diversity_global() == 0.0

    def test_uniform_distribution_high(self):
        et = EdgeTracker()
        # 4 edges each hit once → D = 1 - 4*(0.25²) = 1 - 0.25 = 0.75
        et.record_edges("a", bytes([1, 1, 1, 1]))
        d = et.simpson_diversity_global()
        assert abs(d - 0.75) < 1e-6

    def test_two_edges_equal(self):
        et = EdgeTracker()
        # 2 edges each hit once → D = 1 - 2*(0.5²) = 0.5
        et.record_edges("a", bytes([1, 1]))
        d = et.simpson_diversity_global()
        assert abs(d - 0.5) < 1e-6

    def test_skewed_lower_than_uniform(self):
        et = EdgeTracker()
        et.record_edges("a", bytes([100, 1, 1, 1]))
        d_skewed = et.simpson_diversity_global()
        et2 = EdgeTracker()
        et2.record_edges("a", bytes([25, 25, 25, 25]))
        d_uniform = et2.simpson_diversity_global()
        assert d_skewed < d_uniform

    def test_value_in_range(self):
        et = EdgeTracker()
        et.record_edges("a", bytes([3, 7, 2, 5, 1]))
        d = et.simpson_diversity_global()
        assert 0.0 <= d <= 1.0


class TestShannonEntropySeed:
    def test_missing_seed_returns_zero(self):
        et = EdgeTracker()
        assert et.shannon_entropy_seed("missing") == 0.0

    def test_uniform_seed_high_entropy(self):
        et = EdgeTracker()
        et.seed_hit_counts["a"] = {0: 5, 1: 5, 2: 5, 3: 5}
        h = et.shannon_entropy_seed("a")
        assert abs(h - 2.0) < 1e-6

    def test_single_edge_zero(self):
        et = EdgeTracker()
        et.seed_hit_counts["a"] = {0: 10}
        assert et.shannon_entropy_seed("a") == 0.0

    def test_two_edges(self):
        et = EdgeTracker()
        et.seed_hit_counts["a"] = {0: 1, 1: 1}
        h = et.shannon_entropy_seed("a")
        assert abs(h - 1.0) < 1e-6

    def test_empty_hit_counts(self):
        et = EdgeTracker()
        et.seed_hit_counts["a"] = {}
        assert et.shannon_entropy_seed("a") == 0.0
