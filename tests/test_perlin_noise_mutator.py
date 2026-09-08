"""Tests for the Perlin noise coherent-perturbation mutator."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.mutations.perlin_noise import (
    PerlinNoise1D,
    PerlinNoiseMutator,
)
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.operator_registry import REGISTRY


class _Rng:
    """Minimal RNG stand-in for tests."""

    def __init__(self, seed: int = 42):
        self._r = random.Random(seed)

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def choice(self, seq):
        return self._r.choice(seq)


class TestPerlinNoise1D:
    """Unit tests for the underlying noise primitive."""

    def test_deterministic_same_seed(self):
        n1 = PerlinNoise1D(seed=1)
        n2 = PerlinNoise1D(seed=1)
        assert n1(3.7) == n2(3.7)

    def test_different_seeds_diverge(self):
        n1 = PerlinNoise1D(seed=1)
        n2 = PerlinNoise1D(seed=2)
        # Extremely unlikely to collide across many sample points.
        assert any(n1(x / 10) != n2(x / 10) for x in range(50))

    def test_zero_at_integer_lattice_points(self):
        """Perlin noise passes through 0 at every integer grid node."""
        n = PerlinNoise1D(seed=5)
        for node in range(10):
            assert n(float(node)) == pytest.approx(0.0, abs=1e-9)

    def test_bounded_output(self):
        n = PerlinNoise1D(seed=9)
        for i in range(500):
            v = n(i / 7.0)
            assert -1.0 <= v <= 1.0

    def test_octaves_bounded(self):
        n = PerlinNoise1D(seed=9)
        for i in range(200):
            v = n.octaves(i / 5.0, n_octaves=6, persistence=0.5)
            assert -1.0 <= v <= 1.0

    def test_continuity_no_large_jumps(self):
        """Adjacent samples should be close -- the whole point of coherent noise."""
        n = PerlinNoise1D(seed=3)
        prev = n(0.0)
        for i in range(1, 200):
            cur = n(i * 0.01)
            assert abs(cur - prev) < 0.1
            prev = cur


class TestPerlinNoiseMutator:
    """Unit tests for PerlinNoiseMutator."""

    def test_name_and_category(self):
        m = PerlinNoiseMutator()
        assert m.name == "perlin_noise"
        assert m.category == "structural"

    def test_registered_in_registry(self):
        """The mutator self-registers on module import."""
        assert "perlin_noise" in REGISTRY.names()

    def test_mutate_changes_data(self):
        m = PerlinNoiseMutator(seed=7, strength=40)
        data = bytes(range(256))
        rng = _Rng(seed=1)
        out = m.mutate(data, rng)
        assert out is not None
        assert out != data

    def test_declines_small_input(self):
        m = PerlinNoiseMutator()
        rng = _Rng(seed=1)
        assert m.mutate(b"short", rng) is None
        assert m.mutate(b"x" * 7, rng) is None
        assert m.mutate(b"x" * 8, rng) is not None

    def test_respects_max_len(self):
        m = PerlinNoiseMutator(seed=7)
        data = b"A" * 256
        rng = _Rng(seed=1)
        out = m.mutate(data, rng, max_len=64)
        assert out is not None
        assert len(out) <= 64

    def test_is_available_always(self):
        m = PerlinNoiseMutator()
        ctx = MutationContext(max_len=64)
        assert m.is_available(ctx, b"anything") is True

    def test_returns_bytes_not_bytearray(self):
        m = PerlinNoiseMutator(seed=7)
        data = b"C" * 64
        rng = _Rng(seed=1)
        out = m.mutate(data, rng)
        assert isinstance(out, bytes)

    def test_fixed_seed_is_reproducible_given_same_rng_draws(self):
        """Same fixed noise seed + same rng sequence -> same output."""
        data = b"A" * 128
        m1 = PerlinNoiseMutator(seed=99, strength=30)
        out1 = m1.mutate(data, _Rng(seed=5))
        m2 = PerlinNoiseMutator(seed=99, strength=30)
        out2 = m2.mutate(data, _Rng(seed=5))
        assert out1 == out2

    def test_smoothness_neighbouring_deltas_are_correlated(self):
        """Coherent noise: adjacent byte deltas should mostly move together,
        unlike independent per-byte random mutation."""
        data = bytes([128] * 512)
        m = PerlinNoiseMutator(seed=1, scale=32.0, strength=50, octaves=1)
        rng = _Rng(seed=1)
        out = m.mutate(data, rng)
        assert out is not None
        deltas = [out[i] - data[i] for i in range(len(data))]
        # Average absolute difference between adjacent deltas should be
        # small relative to the strength -- a smooth field, not white noise.
        adjacent_diffs = [abs(deltas[i + 1] - deltas[i]) for i in range(len(deltas) - 1)]
        assert sum(adjacent_diffs) / len(adjacent_diffs) < 10

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError, match="scale must be > 0"):
            PerlinNoiseMutator(scale=0)
        with pytest.raises(ValueError, match="strength must be"):
            PerlinNoiseMutator(strength=0)
        with pytest.raises(ValueError, match="strength must be"):
            PerlinNoiseMutator(strength=300)
        with pytest.raises(ValueError, match="octaves must be"):
            PerlinNoiseMutator(octaves=0)
