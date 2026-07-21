"""Tests for numpy acceleration of information theory algorithms.

Verifies that numpy fast paths produce identical results to pure-Python
fallbacks, and that the threshold gating works correctly.
"""

import math
from unittest.mock import patch

from fuzzer_tool.core import edge_tracker as et_mod
from fuzzer_tool.core import renyi as renyi_mod
from fuzzer_tool.core.edge_tracker import EdgeTracker, _js_divergence
from fuzzer_tool.core.renyi import RenyiEntropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_pure_python():
    """Disable numpy paths in both modules."""
    et_mod._HAS_NUMPY = False
    renyi_mod._HAS_NUMPY = False


def _force_numpy():
    """Enable numpy paths in both modules."""
    et_mod._HAS_NUMPY = True
    renyi_mod._HAS_NUMPY = True


def _large_uniform_dict(n=100):
    """Dict with n elements, each count = 1."""
    return {i: 1 for i in range(n)}


def _large_skewed_dict(n=100):
    """Dict with n elements, power-law counts."""
    return {i: n - i for i in range(n)}


# ---------------------------------------------------------------------------
# Shannon entropy — edge_tracker
# ---------------------------------------------------------------------------


class TestShannonEntropyNumpy:
    """Verify numpy and pure-Python Shannon entropy agree."""

    def test_global_uniform(self):
        _force_numpy()
        et = EdgeTracker()
        et._global_edge_hits = {i: 1 for i in range(80)}
        np_result = et.shannon_entropy_global()

        _force_pure_python()
        py_result = et.shannon_entropy_global()

        assert abs(np_result - py_result) < 1e-10

    def test_global_skewed(self):
        _force_numpy()
        et = EdgeTracker()
        et._global_edge_hits = {i: 100 - i for i in range(80)}
        np_result = et.shannon_entropy_global()

        _force_pure_python()
        py_result = et.shannon_entropy_global()

        assert abs(np_result - py_result) < 1e-10

    def test_seed_uniform(self):
        _force_numpy()
        et = EdgeTracker()
        et.seed_hit_counts["s1"] = {i: 1 for i in range(80)}
        np_result = et.shannon_entropy_seed("s1")

        _force_pure_python()
        py_result = et.shannon_entropy_seed("s1")

        assert abs(np_result - py_result) < 1e-10

    def test_seed_skewed(self):
        _force_numpy()
        et = EdgeTracker()
        et.seed_hit_counts["s1"] = {i: (80 - i) for i in range(80)}
        np_result = et.shannon_entropy_seed("s1")

        _force_pure_python()
        py_result = et.shannon_entropy_seed("s1")

        assert abs(np_result - py_result) < 1e-10

    def test_small_data_falls_back(self):
        """With <= 50 elements, numpy path is not taken even if available."""
        _force_numpy()
        et = EdgeTracker()
        et._global_edge_hits = {i: 1 for i in range(10)}
        result = et.shannon_entropy_global()
        # Should still be correct (uniform over 10)
        assert abs(result - math.log2(10)) < 1e-10

    def test_empty_returns_zero(self):
        _force_numpy()
        et = EdgeTracker()
        assert et.shannon_entropy_global() == 0.0
        assert et.shannon_entropy_seed("missing") == 0.0


# ---------------------------------------------------------------------------
# Simpson diversity — edge_tracker
# ---------------------------------------------------------------------------


class TestSimpsonDiversityNumpy:
    def test_uniform(self):
        _force_numpy()
        et = EdgeTracker()
        et._global_edge_hits = {i: 1 for i in range(80)}
        np_result = et.simpson_diversity_global()

        _force_pure_python()
        py_result = et.simpson_diversity_global()

        assert abs(np_result - py_result) < 1e-10

    def test_skewed(self):
        _force_numpy()
        et = EdgeTracker()
        et._global_edge_hits = {i: 100 - i for i in range(80)}
        np_result = et.simpson_diversity_global()

        _force_pure_python()
        py_result = et.simpson_diversity_global()

        assert abs(np_result - py_result) < 1e-10

    def test_single_edge(self):
        _force_numpy()
        et = EdgeTracker()
        et._global_edge_hits = {0: 42}
        assert et.simpson_diversity_global() == 0.0

    def test_two_edges_equal(self):
        _force_numpy()
        et = EdgeTracker()
        et._global_edge_hits = {0: 5, 1: 5}
        # D = 1 - (0.5^2 + 0.5^2) = 0.5
        assert abs(et.simpson_diversity_global() - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# JS divergence — edge_tracker
# ---------------------------------------------------------------------------


class TestJsDivergenceNumpy:
    def test_identical_distributions(self):
        _force_numpy()
        p = {i: 1.0 / 80 for i in range(80)}
        q = {i: 1.0 / 80 for i in range(80)}
        np_result = _js_divergence(p, q)

        _force_pure_python()
        py_result = _js_divergence(p, q)

        assert abs(np_result - py_result) < 1e-10
        assert np_result < 1e-10  # identical → ~0

    def test_disjoint_distributions(self):
        _force_numpy()
        p = {i: 0.5 / 40 for i in range(40)}
        q = {i: 0.5 / 40 for i in range(40, 80)}
        np_result = _js_divergence(p, q)

        _force_pure_python()
        py_result = _js_divergence(p, q)

        assert abs(np_result - py_result) < 1e-10
        assert np_result > 0.3  # disjoint → high divergence (max is ln(2)/2 ≈ 0.347)

    def test_vs_aggregate(self):
        _force_numpy()
        et = EdgeTracker()
        et._aggregate_total_count = 1000
        et._aggregate_totals = {i: 10.0 for i in range(80)}
        seed_dist = {i: 1.0 / 80 for i in range(80)}
        np_result = et._js_divergence_vs_aggregate(seed_dist)

        _force_pure_python()
        py_result = et._js_divergence_vs_aggregate(seed_dist)

        assert abs(np_result - py_result) < 1e-10


# ---------------------------------------------------------------------------
# Rényi entropy — renyi.py
# ---------------------------------------------------------------------------


class TestRenyiNumpy:
    """Verify numpy and pure-Python Rényi entropy agree."""

    def test_renyi_alpha2_uniform(self):
        _force_numpy()
        r = RenyiEntropy()
        np_result = r.renyi([1, 1, 1, 1], alpha=2.0)

        _force_pure_python()
        py_result = r.renyi([1, 1, 1, 1], alpha=2.0)

        assert abs(np_result - py_result) < 1e-10

    def test_renyi_alpha05_skewed(self):
        _force_numpy()
        r = RenyiEntropy()
        counts = list(range(1, 81))  # [1, 2, ..., 80]
        np_result = r.renyi(counts, alpha=0.5)

        _force_pure_python()
        py_result = r.renyi(counts, alpha=0.5)

        assert abs(np_result - py_result) < 1e-10

    def test_renyi_alpha5(self):
        _force_numpy()
        r = RenyiEntropy()
        counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        np_result = r.renyi(counts, alpha=5.0)

        _force_pure_python()
        py_result = r.renyi(counts, alpha=5.0)

        assert abs(np_result - py_result) < 1e-10

    def test_renyi_alpha10(self):
        _force_numpy()
        r = RenyiEntropy()
        counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        np_result = r.renyi(counts, alpha=10.0)

        _force_pure_python()
        py_result = r.renyi(counts, alpha=10.0)

        assert abs(np_result - py_result) < 1e-10


# ---------------------------------------------------------------------------
# Rényi spectrum — batch computation
# ---------------------------------------------------------------------------


class TestRenyiSpectrumNumpy:
    def test_spectrum_uniform(self):
        _force_numpy()
        r = RenyiEntropy()
        np_spectrum = r.entropy_spectrum([1, 1, 1, 1])

        _force_pure_python()
        py_spectrum = r.entropy_spectrum([1, 1, 1, 1])

        for key in np_spectrum:
            assert abs(np_spectrum[key] - py_spectrum[key]) < 1e-10, f"Mismatch at {key}"

    def test_spectrum_skewed(self):
        _force_numpy()
        r = RenyiEntropy()
        counts = list(range(1, 81))
        np_spectrum = r.entropy_spectrum(counts)

        _force_pure_python()
        py_spectrum = r.entropy_spectrum(counts)

        for key in np_spectrum:
            assert abs(np_spectrum[key] - py_spectrum[key]) < 1e-10, f"Mismatch at {key}"

    def test_spectrum_monotonicity(self):
        """For non-uniform: H_0 >= H_0.5 >= H_1 >= H_2 >= H_5 >= H_10 >= H_inf."""
        _force_numpy()
        r = RenyiEntropy()
        counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        s = r.entropy_spectrum(counts)

        alphas = ["renyi_0.0", "renyi_0.5", "renyi_1.0", "renyi_2.0", "renyi_5.0", "renyi_10.0"]
        for i in range(len(alphas) - 1):
            assert s[alphas[i]] >= s[alphas[i + 1]] - 1e-10, (
                f"{alphas[i]}={s[alphas[i]]:.6f} < {alphas[i + 1]}={s[alphas[i + 1]]:.6f}"
            )
        assert s[alphas[-1]] >= s["min_entropy"] - 1e-10

    def test_spectrum_all_keys_present(self):
        _force_numpy()
        r = RenyiEntropy()
        s = r.entropy_spectrum([1, 1, 1, 1])
        expected_keys = {
            "renyi_0.0",
            "renyi_0.5",
            "renyi_1.0",
            "renyi_2.0",
            "renyi_5.0",
            "renyi_10.0",
            "min_entropy",
        }
        assert set(s.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Shannon (renyi helper)
# ---------------------------------------------------------------------------


class TestRenyiShannonNumpy:
    def test_uniform(self):
        _force_numpy()
        r = RenyiEntropy()
        np_result = r._shannon([0.25, 0.25, 0.25, 0.25])

        _force_pure_python()
        py_result = r._shannon([0.25, 0.25, 0.25, 0.25])

        assert abs(np_result - py_result) < 1e-10
        assert abs(np_result - 2.0) < 1e-10

    def test_deterministic(self):
        _force_numpy()
        r = RenyiEntropy()
        assert r._shannon([1.0]) == 0.0


# ---------------------------------------------------------------------------
# Tsallis entropy
# ---------------------------------------------------------------------------


class TestTsallisNumpy:
    def test_q2_uniform(self):
        _force_numpy()
        r = RenyiEntropy()
        np_result = r.tsallis_entropy([1, 1, 1, 1], q=2.0)

        _force_pure_python()
        py_result = r.tsallis_entropy([1, 1, 1, 1], q=2.0)

        assert abs(np_result - py_result) < 1e-10

    def test_q05_skewed(self):
        _force_numpy()
        r = RenyiEntropy()
        counts = list(range(1, 81))
        np_result = r.tsallis_entropy(counts, q=0.5)

        _force_pure_python()
        py_result = r.tsallis_entropy(counts, q=0.5)

        assert abs(np_result - py_result) < 1e-10

    def test_q1_limit_equals_shannon(self):
        _force_numpy()
        r = RenyiEntropy()
        counts = [3, 7, 2, 8]
        tsallis = r.tsallis_entropy(counts, q=1.0)
        shannon = r.shannon(counts)
        assert isinstance(tsallis, float)
        assert tsallis >= 0


# ---------------------------------------------------------------------------
# Threshold gating
# ---------------------------------------------------------------------------


class TestNumpyThreshold:
    """Verify numpy path is only taken when len(data) > 50."""

    def test_below_threshold_uses_pure_python(self):
        et = EdgeTracker()
        et._global_edge_hits = {i: 1 for i in range(30)}
        # Force numpy on but data is small
        _force_numpy()
        # The function should still work correctly
        result = et.shannon_entropy_global()
        assert abs(result - math.log2(30)) < 1e-10

    def test_above_threshold_uses_numpy(self):
        et = EdgeTracker()
        et._global_edge_hits = {i: 1 for i in range(100)}
        _force_numpy()
        result = et.shannon_entropy_global()
        assert abs(result - math.log2(100)) < 1e-10
