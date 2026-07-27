"""Tests for core/allan_variance.py — Allan variance stall detection."""

import math
import random

from fuzzer_tool.core.allan_variance import AllanVarianceDetector


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
