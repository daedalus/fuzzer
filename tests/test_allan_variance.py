"""Tests for core/allan_variance.py — Allan variance stall detection and DispersionIndex."""

import math
import random

import pytest

from fuzzer_tool.core.allan_variance import (
    AllanVarianceDetector,
    DispersionIndex,
    chi2_cdf,
    chi2_sf,
)
from fuzzer_tool.core.chi_squared import chi_squared_pvalue


def _white_noise(n: int, scale: float = 1.0, seed: int = 42) -> list[float]:
    """Generate n samples of Gaussian white noise."""
    rng = random.Random(seed)
    return [rng.gauss(0, scale) for _ in range(n)]


class TestAllanVarianceDetector:
    def test_init(self):
        d = AllanVarianceDetector()
        assert d.n_samples == 0
        assert not d.buffer_full
        assert d.noise_type() == "unknown"
        assert d.noise_slope() is None

    def test_adev_constant_series(self):
        """Allan deviation of a constant series should be ~0."""
        d = AllanVarianceDetector(max_buffer_pow=4, min_samples=4)
        for _ in range(32):
            d.update(1.0)
        dev = d.adev(1)
        assert dev < 1e-12, f"expected ~0, got {dev}"

    def test_adev_requires_samples(self):
        """adev(tau) returns NaN when n < 2*tau+1."""
        d = AllanVarianceDetector(max_buffer_pow=4, min_samples=4)
        d.update(1.0)
        d.update(1.0)
        d.update(1.0)
        assert math.isnan(d.adev(2))  # n=3, tau=2 → need 5

    def test_adev_power_of_two(self):
        """adev works for various tau values."""
        d = AllanVarianceDetector(max_buffer_pow=6, min_samples=4)
        for v in _white_noise(64, scale=2.0):
            d.update(v)
        for p in range(1, 5):
            tau = 2**p
            dev = d.adev(tau)
            assert math.isfinite(dev), f"adev({tau}) should be finite"
            assert dev > 0, f"adev({tau}) should be > 0 for noise"

    def test_noise_type_active(self):
        """Sustained random exploration classifies as 'active'."""
        d = AllanVarianceDetector(max_buffer_pow=8, min_samples=8)
        rng = random.Random(42)
        for _ in range(200):
            d.update(rng.gauss(5, 2))
        assert d.noise_type() == "active", f"expected active, got {d.noise_type()}"

    def test_noise_type_stalled(self):
        """Zero-variance signal classifies as 'stalled'."""
        d = AllanVarianceDetector(max_buffer_pow=8, min_samples=8)
        for _ in range(200):
            d.update(0.0)
        assert d.noise_type() == "stalled", f"expected stalled, got {d.noise_type()}"
        assert d.adev(2) < 0.01

    def test_stalled_detects_zero_variance(self):
        """All-zero signal has adev(2) below stall threshold."""
        d = AllanVarianceDetector(max_buffer_pow=8, min_samples=8)
        for _ in range(200):
            d.update(0.0)
        assert d.noise_type() == "stalled"
        assert d.adev(2) < 0.01

    def test_noise_type_fatiguing(self):
        """Decaying discovery rate classifies as 'fatiguing'."""
        d = AllanVarianceDetector(max_buffer_pow=8, min_samples=8)
        rng = random.Random(42)
        for i in range(200):
            rate = max(0, 10.0 - i * 10.0 / 200)
            d.update(rng.gauss(rate, max(rate * 0.3, 0.5)))
        assert d.noise_type() == "fatiguing", f"expected fatiguing, got {d.noise_type()}"
        slope = d.noise_slope()
        assert slope is not None and slope > 0.1, f"expected slope > 0.1, got {slope}"

    def test_noise_type_insufficient_data(self):
        """Returns 'unknown' with fewer than min_samples."""
        d = AllanVarianceDetector(max_buffer_pow=4, min_samples=20)
        for v in _white_noise(10):
            d.update(v)
        assert d.noise_type() == "unknown"

    def test_buffer_full(self):
        """buffer_full returns True when buffer is at capacity."""
        d = AllanVarianceDetector(max_buffer_pow=3, min_samples=2)
        assert not d.buffer_full
        for _ in range(8):
            d.update(1.0)
        assert d.buffer_full

    def test_reset(self):
        """reset() clears all samples."""
        d = AllanVarianceDetector(max_buffer_pow=4, min_samples=4)
        for v in _white_noise(16):
            d.update(v)
        assert d.n_samples == 16
        d.reset()
        assert d.n_samples == 0

    def test_save_load_roundtrip(self):
        """save() → load() preserves state, including the dispersion index
        reconstructed from the restored buffer."""
        d1 = AllanVarianceDetector(max_buffer_pow=6, min_samples=4)
        for v in _white_noise(32, scale=2.0):
            d1.update(v)
        saved = d1.save()

        d2 = AllanVarianceDetector()
        d2.load(saved)
        assert d2.n_samples == d1.n_samples
        assert d2.noise_type() == d1.noise_type()
        assert d2.dispersion() == pytest.approx(d1.dispersion(), abs=1e-12)
        assert d2.dispersion_pvalue() == pytest.approx(d1.dispersion_pvalue(), abs=1e-12)

    def test_update_buffer_wraps(self):
        """Buffer correctly evicts oldest when full."""
        d = AllanVarianceDetector(max_buffer_pow=3, min_samples=2)
        for i in range(16):
            d.update(float(i))
        assert d.n_samples == 8
        data = list(d._buf)
        assert data[0] == 8.0
        assert data[-1] == 15.0


class TestAllanVarianceDispersion:
    """Tests for AllanVarianceDetector.dispersion()."""

    def test_dispersion_constant(self):
        """Constant signal → variance ≈ 0 → D ≈ 0."""
        d = AllanVarianceDetector(max_buffer_pow=4, min_samples=4)
        for _ in range(16):
            d.update(5.0)
        d_val = d.dispersion()
        assert d_val is not None and d_val < 0.01, f"expected ~0, got {d_val}"

    def test_dispersion_poisson_like(self):
        """Poisson-like signal → D ≈ 1 (variance ≈ mean)."""
        d = AllanVarianceDetector(max_buffer_pow=6, min_samples=4)
        rng = random.Random(42)
        for _ in range(64):
            d.update(rng.expovariate(1.0))
        d_val = d.dispersion()
        assert d_val is not None, "dispersion should not be None"
        # Exponential has mean=variance → D=1, allow ±0.5 for sample noise
        assert 0.3 <= d_val <= 1.8, f"expected ≈1.0, got {d_val}"

    def test_dispersion_bursty(self):
        """Bursty signal (clusters of high values in zeros) → D > 1.5."""
        d = AllanVarianceDetector(max_buffer_pow=6, min_samples=4)
        for _ in range(10):
            for _ in range(3):
                d.update(0.0)
            for _ in range(2):
                d.update(10.0)
        d_val = d.dispersion()
        assert d_val is not None and d_val > 1.5, f"expected >1.5, got {d_val}"

    def test_dispersion_zero_signal(self):
        """All zeros → mean=0 → None (can't compute D)."""
        d = AllanVarianceDetector(max_buffer_pow=4, min_samples=4)
        for _ in range(16):
            d.update(0.0)
        assert d.dispersion() is None

    def test_dispersion_insufficient_data(self):
        """Fewer than 2 samples → None."""
        d = AllanVarianceDetector(max_buffer_pow=4, min_samples=4)
        d.update(1.0)
        assert d.dispersion() is None


class TestDispersionIndex:
    """Tests for standalone DispersionIndex class."""

    def test_init(self):
        di = DispersionIndex(window=100)
        assert di.count == 0
        assert di.value is None

    def test_constant_signal(self):
        """All identical values → variance=0 → D=0."""
        di = DispersionIndex(window=100)
        for _ in range(50):
            di.update(3.0)
        d_val = di.value
        assert d_val is not None and d_val < 0.01

    def test_bursty_signal(self):
        """Bursty pattern (occasional large values in zeros) → D > 1.5."""
        di = DispersionIndex(window=200)
        # Most values near zero, occasional bursts of 10.0
        for _ in range(5):
            for _ in range(95):
                di.update(0.1)
            for _ in range(5):
                di.update(10.0)
        d_val = di.value
        assert d_val is not None and d_val > 1.5, f"expected >1.5, got {d_val}"

    def test_binary_bernoulli(self):
        """Fair coin flips (Bernoulli p=0.5) → D = 1-p ≈ 0.5."""
        di = DispersionIndex(window=200)
        rng = random.Random(42)
        for _ in range(200):
            di.update(1.0 if rng.random() < 0.5 else 0.0)
        d_val = di.value
        assert d_val is not None and 0.3 <= d_val <= 0.7, f"expected ≈0.5, got {d_val}"

    def test_nearly_constant_signal(self):
        """Nearly constant signal (tight around mean) → D < 0.3."""
        di = DispersionIndex(window=200)
        for _ in range(200):
            di.update(1.0 + random.uniform(-0.05, 0.05))
        d_val = di.value
        assert d_val is not None and d_val < 0.3, f"expected <0.3, got {d_val}"

    def test_insufficient_data(self):
        """Single observation → None."""
        di = DispersionIndex(window=100)
        di.update(1.0)
        assert di.value is None

    def test_save_load_roundtrip(self):
        """save() → load() preserves dispersion value."""
        di1 = DispersionIndex(window=200)
        rng = random.Random(42)
        for _ in range(100):
            di1.update(1.0 if rng.random() < 0.3 else 0.0)
        saved = di1.save()
        expected = di1.value

        di2 = DispersionIndex(window=200)
        di2.load(saved)
        assert di2.count == di1.count
        assert (
            di2.value == pytest.approx(expected, abs=1e-10)
            if expected is not None
            else di2.value is None
        )


class TestChi2Distribution:
    """Validate the pure-Python chi-squared implementation against known
    reference critical values (standard statistical tables)."""

    @pytest.mark.parametrize(
        "x,k,expected_alpha",
        [
            (18.307, 10, 0.05),
            (3.841, 1, 0.05),
            (124.342, 100, 0.05),
            (9.488, 4, 0.05),
            (6.635, 1, 0.01),
            (23.209, 10, 0.01),
        ],
    )
    def test_sf_matches_known_critical_values(self, x, k, expected_alpha):
        got = chi2_sf(x, k)
        assert got == pytest.approx(expected_alpha, abs=1e-3)

    def test_cdf_plus_sf_equals_one(self):
        for x, k in [(0.5, 3), (10.0, 5), (50.0, 20), (200.0, 100)]:
            assert chi2_cdf(x, k) + chi2_sf(x, k) == pytest.approx(1.0, abs=1e-9)

    def test_sf_at_zero_is_one(self):
        assert chi2_sf(0.0, 5) == 1.0
        assert chi2_sf(-1.0, 5) == 1.0

    def test_invalid_dof_raises(self):
        with pytest.raises(ValueError):
            chi2_sf(1.0, 0)

    def test_regression_chi2_sf_matches_canonical_chi_squared_pvalue(self):
        """chi2_sf must delegate to chi_squared_pvalue (canonical home of
        the survival function) — regression guard for the dedup that moved
        the implementation out of this module."""
        for k in (1, 2, 10, 100):
            for i in range(0, 101):
                x = 5 * k * i / 100
                assert chi2_sf(x, k) == pytest.approx(chi_squared_pvalue(x, k), abs=1e-12), (
                    f"mismatch at x={x}, k={k}"
                )
        with pytest.raises(ValueError):
            chi2_sf(1.0, 0)


class TestDispersionSignificance:
    """Chi-squared Poisson dispersion test: the core behavior this
    replaces the fixed D>1.5 / D<0.3 thresholds with — the effective
    threshold should adapt to sample count instead of firing/missing
    based on a single magic number regardless of how much data backs it.
    """

    def test_small_sample_does_not_overfire(self):
        """A handful of samples with D just above the old fixed 1.5
        threshold should NOT be flagged significant — too little evidence."""
        di = DispersionIndex(window=200)
        # D ≈ 1.71 with only 8 samples (see reproduction in review)
        for v in [1, 1, 1, 1, 1, 5, 5, 1]:
            di.update(float(v))
        assert di.value is not None and di.value > 1.5
        assert di.is_overdispersed() is False

    def test_large_sample_detects_mild_but_real_overdispersion(self):
        """The same underlying mild-overdispersion process, given enough
        samples, should be detected even though D itself settles below
        the old fixed 1.5 threshold."""
        rng = random.Random(7)
        di = DispersionIndex(window=300)
        for _ in range(250):
            v = 5 if rng.random() < 0.15 else 1
            di.update(float(v))
        assert di.value is not None and di.value < 1.5
        assert di.is_overdispersed() is True

    def test_insufficient_data_returns_false_not_none(self):
        """is_overdispersed/is_underdispersed must be directly usable in
        boolean logic without a None-check, unlike value/dispersion_pvalue."""
        di = DispersionIndex(window=100)
        di.update(1.0)
        assert di.is_overdispersed() is False
        assert di.is_underdispersed() is False
        assert di.dispersion_pvalue() is None

    def test_underdispersion_significance(self):
        """A near-constant signal should be flagged underdispersed once
        enough samples accumulate, matching the old D<<0.3 intuition but
        via a calibrated test rather than a fixed cutoff."""
        di = DispersionIndex(window=300)
        for _ in range(200):
            di.update(1.0 + random.uniform(-0.01, 0.01))
        assert di.is_underdispersed() is True
        assert di.is_overdispersed() is False

    def test_allan_variance_detector_matches_dispersion_index(self):
        """AllanVarianceDetector's significance methods delegate to
        DispersionIndex — keep as a behavioral guard that the delegation
        stays in sync (it is now the same code path, not two independent
        implementations)."""
        rng = random.Random(7)
        values = [5.0 if rng.random() < 0.15 else 1.0 for _ in range(250)]

        avd = AllanVarianceDetector(max_buffer_pow=10, min_samples=2)
        di = DispersionIndex(window=300)
        for v in values:
            avd.update(v)
            di.update(v)

        assert avd.is_overdispersed() == di.is_overdispersed()
        assert avd.dispersion_pvalue() == pytest.approx(di.dispersion_pvalue(), abs=1e-9)
