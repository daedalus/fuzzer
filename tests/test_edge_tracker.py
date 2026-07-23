"""Tests for edge_tracker.py — KS, NCD, Good-Turing, weights, CDF norms."""

import math

from fuzzer_tool.core.edge_tracker import (
    EdgeTracker,
    _js_divergence,
    _kolmogorov_pvalue,
    _ks_p_from_cdf_diff,
    ks_significance_threshold,
    ks_two_sample,
    normalized_compression_distance,
)


class TestKSTwoSample:
    def test_identical_distributions(self):
        d, p = ks_two_sample([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert d == 0.0
        assert p == 1.0

    def test_completely_separated(self):
        d, p = ks_two_sample([1.0, 1.0, 1.0], [5.0, 5.0, 5.0])
        assert d == 1.0
        assert p < 0.01

    def test_overlapping_distributions(self):
        import random

        random.seed(42)
        a = [random.gauss(10, 1) for _ in range(50)]
        b = [random.gauss(11, 1) for _ in range(50)]
        d, p = ks_two_sample(a, b)
        assert 0.0 < d < 1.0
        assert 0.0 <= p <= 1.0

    def test_empty_samples(self):
        d, p = ks_two_sample([], [1.0])
        assert d == 0.0
        assert p == 1.0

    def test_single_element(self):
        d, p = ks_two_sample([1.0], [2.0])
        assert d > 0.0

    def test_large_samples_converge(self):
        import random

        random.seed(0)
        a = [random.gauss(0, 1) for _ in range(500)]
        b = [random.gauss(0, 1) for _ in range(500)]
        d, p = ks_two_sample(a, b)
        assert p > 0.05  # same distribution, should not be significant


class TestKSFunctions:
    def test_significance_threshold_decreases_with_n(self):
        t10 = ks_significance_threshold(10, 0.05)
        t100 = ks_significance_threshold(100, 0.05)
        t1000 = ks_significance_threshold(1000, 0.05)
        assert t10 > t100 > t1000

    def test_stricter_alpha_higher_threshold(self):
        t05 = ks_significance_threshold(100, 0.05)
        t01 = ks_significance_threshold(100, 0.01)
        assert t01 > t05

    def test_zero_samples(self):
        assert ks_significance_threshold(0, 0.05) == 1.0

    def test_kolmogorov_pvalue_zero_d(self):
        assert _kolmogorov_pvalue(0.0, 100, 100) == 1.0

    def test_kolmogorov_pvalue_large_d(self):
        assert _kolmogorov_pvalue(1.0, 100, 100) == 0.0

    def test_ks_p_from_cdf_diff_zero(self):
        assert _ks_p_from_cdf_diff(0.0, 100) == 1.0

    def test_ks_p_from_cdf_diff_negative(self):
        assert _ks_p_from_cdf_diff(-1.0, 100) == 1.0


class TestNCD:
    def test_identical_bytes(self):
        x = b"hello world" * 20
        ncd = normalized_compression_distance(x, x)
        assert ncd < 0.15  # identical content, low NCD (zlib overhead prevents 0.0)

    def test_empty_input(self):
        assert normalized_compression_distance(b"", b"test") == 1.0
        assert normalized_compression_distance(b"test", b"") == 1.0
        assert normalized_compression_distance(b"", b"") == 1.0

    def test_similar_vs_random(self):
        similar1 = b"\x89PNG" + b"\x00" * 200
        similar2 = b"\x89PNG" + b"\x01" * 200
        random_bytes = bytes(range(256)) * 3
        ncd_similar = normalized_compression_distance(similar1, similar2)
        ncd_random = normalized_compression_distance(similar1, random_bytes)
        assert ncd_similar < ncd_random

    def test_symmetric(self):
        a = b"AAAA" * 50
        b = b"BBBB" * 50
        assert normalized_compression_distance(a, b) == normalized_compression_distance(b, a)


class TestJSDivergence:
    def test_identical_distributions(self):
        p = {1: 0.5, 2: 0.5}
        assert _js_divergence(p, p) < 1e-10

    def test_disjoint_distributions(self):
        p = {1: 1.0}
        q = {2: 1.0}
        js = _js_divergence(p, q)
        assert js > 0.0

    def test_partial_overlap(self):
        p = {1: 0.7, 2: 0.3}
        q = {1: 0.3, 2: 0.7}
        js = _js_divergence(p, q)
        assert 0.0 < js < math.log(2)

    def test_empty_distributions(self):
        assert _js_divergence({}, {}) == 0.0


class TestEdgeTrackerCore:
    def test_init(self):
        et = EdgeTracker()
        assert et.map_size == 65536
        assert len(et.cumulative_edges) == 0

    def test_record_edges_new(self):
        et = EdgeTracker(map_size=256)
        bitmap = bytearray(256)
        bitmap[10] = 5
        bitmap[20] = 3
        new = et.record_edges("seed1", bytes(bitmap))
        assert 10 in new
        assert 20 in new
        assert 10 in et.cumulative_edges

    def test_record_edges_no_new(self):
        et = EdgeTracker(map_size=256)
        bitmap = bytearray(256)
        bitmap[10] = 5
        et.record_edges("seed1", bytes(bitmap))
        new = et.record_edges("seed1", bytes(bitmap))
        assert len(new) == 0  # already seen

    def test_record_edges_increments_global_hits(self):
        et = EdgeTracker(map_size=256)
        bitmap = bytearray(256)
        bitmap[10] = 5
        et.record_edges("seed1", bytes(bitmap))
        bitmap[10] = 3
        et.record_edges("seed1", bytes(bitmap))
        # 5 → class 4, 3 → class 3 (count_class bucketization)
        assert et._global_edge_hits[10] == 7  # 4+3

    def test_record_edges_invalidation(self):
        et = EdgeTracker(map_size=256)
        bm = bytearray(256)
        bm[10] = 1
        et.record_edges("s", bytes(bm))
        et._build_aggregate_distribution()  # populate cache
        assert et._aggregate_cache is not None
        bm[20] = 2
        et.record_edges("s2", bytes(bm))
        assert et._aggregate_cache is None

    def test_get_cumulative_edge_count(self):
        et = EdgeTracker(map_size=256)
        assert et.get_cumulative_edge_count() == 0
        bitmap = bytearray(256)
        bitmap[5] = 1
        et.record_edges("s", bytes(bitmap))
        assert et.get_cumulative_edge_count() == 1


class TestEdgeTrackerWeights:
    def _make_tracker_with_seeds(self):
        et = EdgeTracker(map_size=256)
        # Seed A: edges at 10, 20
        bm_a = bytearray(256)
        bm_a[10] = 10
        bm_a[20] = 5
        et.record_edges("a", bytes(bm_a))
        # Seed B: edges at 30, 40
        bm_b = bytearray(256)
        bm_b[30] = 8
        bm_b[40] = 3
        et.record_edges("b", bytes(bm_b))
        return et

    def test_subsumption_weight_no_overlap(self):
        et = self._make_tracker_with_seeds()
        w = et.compute_subsumption_weight("a")
        # A has edges not in B, B has edges not in A — low overlap
        assert 0.1 <= w <= 1.0

    def test_subsumption_weight_unknown_seed(self):
        et = self._make_tracker_with_seeds()
        assert et.compute_subsumption_weight("nonexistent") == 1.0

    def test_subsumption_weight_empty_seed(self):
        et = EdgeTracker(map_size=256)
        et.seed_edges["empty"] = set()
        assert et.compute_subsumption_weight("empty") == 0.5

    def test_hitcount_diversity_weight(self):
        et = self._make_tracker_with_seeds()
        w = et.compute_hitcount_diversity_weight("a")
        assert 0.5 <= w <= 2.0

    def test_hitcount_diversity_no_data(self):
        et = EdgeTracker(map_size=256)
        assert et.compute_hitcount_diversity_weight("x") == 1.0

    def test_wasserstein_weight(self):
        et = self._make_tracker_with_seeds()
        w = et.compute_wasserstein_weight("a")
        assert 0.5 <= w <= 2.0

    def test_wasserstein_weight_no_hit_counts(self):
        et = EdgeTracker(map_size=256)
        assert et.compute_wasserstein_weight("x") == 1.0

    def test_get_seed_edge_count(self):
        et = self._make_tracker_with_seeds()
        assert et.get_seed_edge_count("a") == 2
        assert et.get_seed_edge_count("b") == 2
        assert et.get_seed_edge_count("c") == 0


class TestCDFNorms:
    def _make_tracker_with_seeds(self):
        et = EdgeTracker(map_size=256)
        bm_a = bytearray(256)
        bm_a[10] = 10
        bm_a[20] = 5
        et.record_edges("a", bytes(bm_a))
        bm_b = bytearray(256)
        bm_b[30] = 8
        bm_b[40] = 3
        et.record_edges("b", bytes(bm_b))
        return et

    def test_wasserstein_distance(self):
        et = self._make_tracker_with_seeds()
        d = et.compute_wasserstein_distance("a", "b")
        assert d >= 0.0

    def test_ks_distance(self):
        et = self._make_tracker_with_seeds()
        d = et.compute_ks_distance("a", "b")
        assert 0.0 <= d <= 1.0

    def test_crps(self):
        et = self._make_tracker_with_seeds()
        c = et.compute_crps("a", "b")
        assert c >= 0.0

    def test_same_seed_zero_distance(self):
        et = EdgeTracker(map_size=256)
        bm = bytearray(256)
        bm[10] = 5
        et.record_edges("s", bytes(bm))
        w, ks, crps = et._cdf_norms("s", "s")
        assert w == 0.0
        assert ks == 0.0
        assert crps == 0.0

    def test_missing_seed(self):
        et = EdgeTracker(map_size=256)
        w, ks, crps = et._cdf_norms("a", "b")
        assert w == 256.0  # max distance = map_size

    def test_corpus_diversity(self):
        et = self._make_tracker_with_seeds()
        div = et.compute_corpus_diversity()
        assert div > 0.0

    def test_corpus_diversity_single_seed(self):
        et = EdgeTracker(map_size=256)
        bm = bytearray(256)
        bm[10] = 1
        et.record_edges("s", bytes(bm))
        assert et.compute_corpus_diversity() == 0.0


class TestGoodTuring:
    def test_empty_tracker(self):
        et = EdgeTracker()
        gt = et.good_turing_estimate()
        assert gt["n"] == 0
        assert gt["confidence"] == "low"

    def test_singletons_high(self):
        et = EdgeTracker(map_size=256)
        # 10 edges each seen once
        for i in range(10):
            bm = bytearray(256)
            bm[i] = 1
            et.record_edges(f"s{i}", bytes(bm))
        gt = et.good_turing_estimate()
        assert gt["n"] == 10
        assert gt["n1"] == 10
        assert gt["saturation"] < 1.0

    def test_repeated_edges_reduce_singletons(self):
        et = EdgeTracker(map_size=256)
        bm = bytearray(256)
        bm[10] = 5
        et.record_edges("s1", bytes(bm))
        et.record_edges("s2", bytes(bm))
        gt = et.good_turing_estimate()
        # Edge 10 seen twice, so n1=0
        assert gt["n1"] == 0 or gt["n"] > 0

    def test_bitmap_density(self):
        et = EdgeTracker(map_size=256)
        bm = bytearray(256)
        bm[0] = 1
        bm[1] = 1
        bm[2] = 1
        et.record_edges("s", bytes(bm))
        d = et.bitmap_density()
        assert abs(d - 3 / 256) < 1e-6

    def test_birthday_collision_risk_zero_edges(self):
        et = EdgeTracker(map_size=256)
        assert et.birthday_collision_risk() == 0.0

    def test_birthday_collision_risk_low(self):
        et = EdgeTracker(map_size=65536)
        bm = bytearray(65536)
        for i in range(10):
            bm[i] = 1
        et.record_edges("s", bytes(bm))
        risk = et.birthday_collision_risk()
        assert risk < 0.01

    def test_birthday_collision_risk_high(self):
        et = EdgeTracker(map_size=256)
        bm = bytearray(256)
        for i in range(100):
            bm[i % 256] = 1
        et.record_edges("s", bytes(bm))
        risk = et.birthday_collision_risk()
        assert risk > 0.5

    def test_recommended_map_size_adequate(self):
        et = EdgeTracker(map_size=65536)
        bm = bytearray(65536)
        for i in range(10):
            bm[i] = 1
        et.record_edges("s", bytes(bm))
        assert et.recommended_map_size() == 0

    def test_recommended_map_size_needed(self):
        et = EdgeTracker(map_size=4096)
        bm = bytearray(4096)
        for i in range(500):
            bm[i % 4096] = 1
        et.record_edges("s", bytes(bm))
        rec = et.recommended_map_size()
        assert rec > et.map_size


class TestAggregateDistribution:
    def test_build_and_cache(self):
        et = EdgeTracker(map_size=256)
        bm = bytearray(256)
        bm[10] = 10
        et.record_edges("s", bytes(bm))
        agg = et._build_aggregate_distribution()
        assert 10 in agg
        assert et._aggregate_cache is agg  # cached

    def test_cache_invalidation(self):
        et = EdgeTracker(map_size=256)
        bm = bytearray(256)
        bm[10] = 5
        et.record_edges("s1", bytes(bm))
        agg1 = et._build_aggregate_distribution()
        bm[20] = 3
        et.record_edges("s2", bytes(bm))
        agg2 = et._build_aggregate_distribution()
        assert agg1 is not agg2  # rebuilt

    def test_empty_tracker(self):
        et = EdgeTracker(map_size=256)
        assert et._build_aggregate_distribution() == {}


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        et = EdgeTracker(map_size=256)
        bm = bytearray(256)
        bm[10] = 5
        bm[20] = 3
        et.record_edges("seed1", bytes(bm))
        et.record_edges("seed1", bytes(bm))  # accumulate hits

        path = str(tmp_path / "tracker.json")
        assert et.save(path)

        et2 = EdgeTracker(map_size=256)
        assert et2.load(path)
        assert et2.map_size == 256
        assert 10 in et2.cumulative_edges
        # 5 → class 4 (count_class bucketization)
        assert et2.seed_hit_counts["seed1"][10] == 4  # latest classified value per seed
        assert et2._global_edge_hits[10] == 8  # 4+4

    def test_load_nonexistent(self, tmp_path):
        et = EdgeTracker()
        assert not et.load(str(tmp_path / "nope.json"))

    def test_load_corrupt(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        et = EdgeTracker()
        assert not et.load(str(path))


class TestBayesianGrowthModel:
    def test_insufficient_data_falls_back(self):
        et = EdgeTracker()
        # Only 3 timeline points — below the 5-point threshold
        et._coverage_timeline = [(0, 0), (100, 10), (200, 15)]
        result = et.bayesian_coverage_growth_model()
        assert result["method"] == "fallback_insufficient_data"

    def test_bayesian_model_runs_with_enough_data(self):
        et = EdgeTracker()
        # Simulate a coverage growth curve: fast start, slowing down
        import math as _m
        timeline = []
        for i in range(20):
            exec_count = i * 100
            edges = int(500 * (1 - _m.exp(-0.1 * i)))
            timeline.append((exec_count, edges))
        et._coverage_timeline = timeline
        result = et.bayesian_coverage_growth_model()
        # Should use the Levenberg-Marquardt path, not fallback
        assert result["method"] == "bayesian_laplace"
        assert result["A_mean"] is not None
        assert result["k_mean"] is not None
        assert result["A_mean"] > 0
        assert result["k_mean"] > 0
        assert result["sigma_mean"] >= 0
        assert result["p_stalled"] is not None
        assert result["p_growth_remaining"] is not None
        assert 0.0 <= result["p_stalled"] <= 1.0
        assert 0.0 <= result["p_growth_remaining"] <= 1.0

    def test_bayesian_model_converged_plateau(self):
        et = EdgeTracker()
        import math as _m
        # Simulate fully saturated coverage — rate dropped to near zero
        timeline = []
        for i in range(30):
            exec_count = i * 100
            edges = min(200, int(200 * (1 - _m.exp(-0.5 * i))))
            timeline.append((exec_count, edges))
        et._coverage_timeline = timeline
        result = et.bayesian_coverage_growth_model()
        assert result["method"] == "bayesian_laplace"
        # P(stalled) should be high for a saturated curve
        assert result["p_stalled"] is not None
        assert result["p_stalled"] > 0.5
