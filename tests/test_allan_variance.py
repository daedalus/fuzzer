"""Tests for core/allan_variance.py — Allan variance stall detection and DispersionIndex."""

import math
import random

import pytest

from fuzzer_tool.core.allan_variance import AllanVarianceDetector, DispersionIndex


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
        """save() → load() preserves state."""
        d1 = AllanVarianceDetector(max_buffer_pow=6, min_samples=4)
        for v in _white_noise(32, scale=2.0):
            d1.update(v)
        saved = d1.save()

        d2 = AllanVarianceDetector()
        d2.load(saved)
        assert d2.n_samples == d1.n_samples
        assert d2.noise_type() == d1.noise_type()

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
