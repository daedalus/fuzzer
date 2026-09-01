"""Tests for the fractal jittered Voronoi spatial meta-mutator."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.mutations.fractal_voronoi import FractalVoronoiMutator
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


class TestFractalVoronoiMutator:
    """Unit tests for FractalVoronoiMutator."""

    def test_name_and_category(self):
        m = FractalVoronoiMutator()
        assert m.name == "fractal_voronoi"
        assert m.category == "structural"

    def test_registered_in_registry(self):
        """The mutator self-registers on module import."""
        assert "fractal_voronoi" in REGISTRY.names()

    def test_mutate_determinism(self):
        """Same input + same state → same output."""
        m = FractalVoronoiMutator(max_depth=3)
        data = b"A" * 256
        rng = _Rng(seed=7)
        out1 = m.mutate(data, rng)
        # Re-instantiate to clear caches but same parameters
        m2 = FractalVoronoiMutator(max_depth=3)
        rng2 = _Rng(seed=7)
        out2 = m2.mutate(data, rng2)
        assert out1 == out2

    def test_mutate_changes_data(self):
        """A non-trivial input should be mutated."""
        m = FractalVoronoiMutator(max_depth=3)
        data = bytes(range(256))
        rng = _Rng(seed=7)
        out = m.mutate(data, rng)
        assert out is not None
        assert out != data

    def test_declines_small_input(self):
        """Inputs < 16 bytes are too small to partition."""
        m = FractalVoronoiMutator(max_depth=3)
        rng = _Rng(seed=7)
        assert m.mutate(b"short", rng) is None
        assert m.mutate(b"x" * 15, rng) is None
        assert m.mutate(b"x" * 16, rng) is not None

    def test_respects_max_len(self):
        """Output should not exceed max_len."""
        m = FractalVoronoiMutator(max_depth=3)
        data = b"A" * 256
        rng = _Rng(seed=7)
        out = m.mutate(data, rng, max_len=128)
        assert out is not None
        assert len(out) <= 128

    def test_is_available_always(self):
        """No external dependencies required."""
        m = FractalVoronoiMutator()
        ctx = MutationContext(max_len=64)
        assert m.is_available(ctx, b"anything") is True

    def test_different_depths_produce_different_outputs(self):
        """max_depth parameter changes the fractal partition."""
        data = b"B" * 256
        rng = _Rng(seed=7)
        m3 = FractalVoronoiMutator(max_depth=3)
        m5 = FractalVoronoiMutator(max_depth=5)
        out3 = m3.mutate(data, rng)
        out5 = m5.mutate(data, rng)
        assert out3 is not None
        assert out5 is not None
        assert out3 != out5

    def test_boundary_detection(self):
        """Boundary check should identify some boundary bytes in a large buffer."""
        m = FractalVoronoiMutator(max_depth=4)
        side = 64
        boundary_count = 0
        for y in range(side):
            for x in range(side):
                px = (x + 0.5) / side
                py = (y + 0.5) / side
                if m._is_boundary(4, px, py):
                    boundary_count += 1
        # With a 64×64 grid and fractal depth 4, we expect *some* boundaries
        assert boundary_count > 0
        assert boundary_count < side * side  # not everything is a boundary

    def test_cell_ops_integration(self):
        """Custom cell_ops are applied deterministically."""
        def invert_byte(b: bytes) -> bytes:
            return bytes([b[0] ^ 0xFF])

        m = FractalVoronoiMutator(max_depth=3, cell_ops=[invert_byte])
        data = b"A" * 256
        rng = _Rng(seed=7)
        out = m.mutate(data, rng)
        assert out is not None
        # Some bytes should have been inverted (0x41 ^ 0xFF = 0xBE)
        assert 0xBE in out

    def test_invalid_depth_raises(self):
        """max_depth < 1 is rejected at construction."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            FractalVoronoiMutator(max_depth=0)

    def test_mutate_returns_bytes_not_bytearray(self):
        """Contract: return bytes, not a mutable type."""
        m = FractalVoronoiMutator(max_depth=3)
        data = b"C" * 128
        rng = _Rng(seed=7)
        out = m.mutate(data, rng)
        assert isinstance(out, bytes)
