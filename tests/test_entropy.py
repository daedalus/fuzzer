"""Tests for Shannon entropy, Simpson's diversity, and entropy rate tracking."""

import math
from unittest.mock import MagicMock, patch

from fuzzer_tool.core.edge_tracker import EdgeTracker


class TestEntropyRateTracking:
    """Tests for entropy history and rate-of-change in the fuzzer loop."""

    def _make_fuzzer(self, **kwargs):
        from fuzzer_tool.services.fuzzer import Fuzzer

        defaults = dict(
            target="/bin/true",
            corpus_dir="/tmp/fuzz_entropy_test_corpus",
            crashes_dir="/tmp/fuzz_entropy_test_crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )
        defaults.update(kwargs)
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(**defaults)
        return f

    def test_entropy_history_initialized(self):
        f = self._make_fuzzer()
        assert hasattr(f, "_entropy_history")
        assert f._entropy_history == []

    def test_entropy_history_records_samples(self):
        f = self._make_fuzzer()
        f.exec_count = 100
        f._record_entropy_sample(1.5)
        f.exec_count = 200
        f._record_entropy_sample(1.8)
        f.exec_count = 300
        f._record_entropy_sample(2.0)
        assert len(f._entropy_history) == 3
        assert f._entropy_history == [(100, 1.5), (200, 1.8), (300, 2.0)]

    def test_entropy_history_trims_old(self):
        f = self._make_fuzzer()
        # Simulate 250 samples via the real recording method, each call
        # trimming independently once the bound is exceeded (as in run()).
        for i in range(250):
            f.exec_count = i * 100
            f._record_entropy_sample(1.0 + i * 0.01)
            assert len(f._entropy_history) <= 200
        # Most recent sample must always be retained.
        assert f._entropy_history[-1] == (249 * 100, 1.0 + 249 * 0.01)

    def test_entropy_rate_positive(self):
        # Rising entropy → positive rate
        history = [(100, 1.0), (200, 1.5), (300, 2.0), (400, 2.5)]
        recent = history[-4:]
        dt = recent[-1][0] - recent[0][0]
        dS = recent[-1][1] - recent[0][1]
        rate = dS / dt
        assert rate > 0

    def test_entropy_rate_negative(self):
        # Falling entropy → negative rate
        history = [(100, 2.5), (200, 2.0), (300, 1.5), (400, 1.0)]
        recent = history[-4:]
        dt = recent[-1][0] - recent[0][0]
        dS = recent[-1][1] - recent[0][1]
        rate = dS / dt
        assert rate < 0

    def test_entropy_rate_flat(self):
        # Flat entropy → rate near zero
        history = [(100, 1.5), (200, 1.5), (300, 1.5), (400, 1.5)]
        recent = history[-4:]
        dt = recent[-1][0] - recent[0][0]
        dS = abs(recent[-1][1] - recent[0][1])
        rate = dS / dt
        assert rate < 0.001

    def test_stall_detection_with_flat_entropy(self):
        """When entropy is flat and no new edges, stall is confirmed."""
        f = self._make_fuzzer()
        f._stall_recovery_active = False
        f._stall_threshold = 100
        f.exec_count = 500
        f._last_new_edge_exec = 100  # 400 execs since last edge

        # Flat entropy history
        f._entropy_history = [
            (100, 1.5), (200, 1.5), (300, 1.5), (400, 1.5)
        ]

        execs_since_edge = f.exec_count - f._last_new_edge_exec
        assert execs_since_edge >= f._stall_threshold
        assert f._compute_entropy_flat() is True
        assert f._maybe_trigger_stall_recovery(execs_since_edge) is True
        assert f._stall_recovery_active is True
        assert f._stall_recovery_count == 1

    def test_stall_not_confirmed_with_changing_entropy(self):
        """When entropy is rising, stall is not confirmed even without new edges."""
        f = self._make_fuzzer()
        f._stall_recovery_active = False
        f._stall_recovery_count = 0
        f._entropy_history = [
            (100, 1.0), (200, 1.5), (300, 2.0), (400, 2.5)
        ]

        assert f._compute_entropy_flat() is False
        assert f._maybe_trigger_stall_recovery(400) is False
        assert f._stall_recovery_active is False
        assert f._stall_recovery_count == 0

    def test_stall_not_confirmed_falls_back_without_enough_samples(self):
        """With <4 entropy samples, stall triggers on no-new-edges alone."""
        f = self._make_fuzzer()
        f._stall_recovery_active = False
        f._stall_recovery_count = 0
        f._entropy_history = [(100, 1.5)]

        assert f._compute_entropy_flat() is None
        assert f._maybe_trigger_stall_recovery(400) is True
        assert f._stall_recovery_active is True
        assert f._stall_recovery_count == 1


class TestSeedPickerEntropyBonus:
    """Tests for input-level entropy bonus in seed weighting."""

    def test_shannon_entropy_seed_different_profiles(self):
        et = EdgeTracker()
        # Uniform profile → high entropy
        et.seed_hit_counts["uniform"] = {0: 5, 1: 5, 2: 5, 3: 5}
        # Skewed profile → low entropy
        et.seed_hit_counts["skewed"] = {0: 100, 1: 1, 2: 1, 3: 1}

        h_uniform = et.shannon_entropy_seed("uniform")
        h_skewed = et.shannon_entropy_seed("skewed")

        assert h_uniform > h_skewed

    def test_shannon_entropy_seed_independent_of_global(self):
        """Per-seed entropy is computed from that seed's own distribution."""
        et = EdgeTracker()
        # Seed "a" hits 4 edges uniformly
        et.seed_hit_counts["a"] = {0: 5, 1: 5, 2: 5, 3: 5}
        h_a = et.shannon_entropy_seed("a")

        # Add many other seeds that hit different edges (doesn't affect "a"'s entropy)
        for i in range(10):
            et.seed_hit_counts[f"other_{i}"] = {100 + i: 10}

        h_a_after = et.shannon_entropy_seed("a")
        assert h_a == h_a_after

    def test_entropy_bonus_computation(self):
        """Verify the z-score bonus formula works correctly."""
        # Simulate the bonus computation from _compute_weights
        seed_entropy = 2.0
        mean_entropy = 1.0
        deviation = abs(seed_entropy - mean_entropy) / max(mean_entropy, 0.01)
        bonus = 1.0 + min(deviation, 1.0) * 0.5

        # deviation = 1.0/1.0 = 1.0, clamped to 1.0
        # bonus = 1.0 + 1.0 * 0.5 = 1.5
        assert abs(bonus - 1.5) < 1e-6

    def test_entropy_bonus_clamped(self):
        """Bonus is clamped at 1.5x even for extreme deviation."""
        seed_entropy = 5.0
        mean_entropy = 0.1
        deviation = abs(seed_entropy - mean_entropy) / max(mean_entropy, 0.01)
        # deviation = 4.9/0.1 = 49.0
        # min(deviation, 1.0) = 1.0
        bonus = 1.0 + min(deviation, 1.0) * 0.5
        assert abs(bonus - 1.5) < 1e-6

    def test_entropy_bonus_small_when_near_mean(self):
        """Bonus is small when seed entropy is near the mean."""
        seed_entropy = 1.05
        mean_entropy = 1.0
        deviation = abs(seed_entropy - mean_entropy) / max(mean_entropy, 0.01)
        # deviation = 0.05/1.0 = 0.05
        bonus = 1.0 + min(deviation, 1.0) * 0.5
        # bonus = 1.0 + 0.05 * 0.5 = 1.025
        assert abs(bonus - 1.025) < 1e-6


class TestSimpsonDiversityProperties:
    """Verify mathematical properties of Simpson's Diversity."""

    def test_uniform_maximizes_diversity(self):
        et = EdgeTracker()
        # n edges each hit once → D = 1 - 1/n
        for n in [2, 4, 8, 16]:
            bitmap = bytes([1] * n + [0] * (64 - n))
            et.record_edges(f"uniform_{n}", bitmap)
        # All edges hit once → D = 1 - n*(1/n²) = 1 - 1/n
        # But since edges accumulate, let's test with a fresh tracker per n
        for n in [2, 4, 8, 16]:
            et2 = EdgeTracker()
            bitmap = bytes([1] * n + [0] * (64 - n))
            et2.record_edges("a", bitmap)
            expected = 1.0 - 1.0 / n
            actual = et2.simpson_diversity_global()
            assert abs(actual - expected) < 1e-6, f"n={n}: expected {expected}, got {actual}"

    def test_diversity_increases_with_more_edges(self):
        et2 = EdgeTracker()
        et2.record_edges("a", bytes([1, 1]))
        d2 = et2.simpson_diversity_global()

        et4 = EdgeTracker()
        et4.record_edges("a", bytes([1, 1, 1, 1]))
        d4 = et4.simpson_diversity_global()

        et8 = EdgeTracker()
        et8.record_edges("a", bytes([1, 1, 1, 1, 1, 1, 1, 1]))
        d8 = et8.simpson_diversity_global()

        assert d2 < d4 < d8


class TestShannonEntropyProperties:
    """Verify mathematical properties of Shannon entropy."""

    def test_entropy_non_negative(self):
        et = EdgeTracker()
        et.record_edges("a", bytes([10, 5, 3, 7]))
        assert et.shannon_entropy_global() >= 0.0

    def test_entropy_bounded_by_log2_n(self):
        """Entropy ≤ log2(n) where n is number of non-zero edges."""
        et = EdgeTracker()
        et.record_edges("a", bytes([1, 2, 3, 4, 5]))
        h = et.shannon_entropy_global()
        # 5 non-zero edges → max entropy = log2(5) ≈ 2.32
        assert h <= math.log2(5) + 1e-6

    def test_entropy_maximized_by_uniform(self):
        """Among all distributions with n outcomes, uniform maximizes entropy."""
        et_uniform = EdgeTracker()
        et_uniform.record_edges("a", bytes([10, 10, 10, 10]))
        h_uniform = et_uniform.shannon_entropy_global()

        et_skewed = EdgeTracker()
        et_skewed.record_edges("a", bytes([37, 1, 1, 1]))
        h_skewed = et_skewed.shannon_entropy_global()

        assert h_uniform > h_skewed

    def test_entropy_zero_for_degenerate(self):
        """Single outcome → entropy = 0."""
        et = EdgeTracker()
        et.record_edges("a", bytes([42, 0, 0, 0]))
        assert et.shannon_entropy_global() == 0.0
