"""Regression tests: randomness.profile_buffer wired into position selection."""

import os
import zlib

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import (
    _REGION_CACHE_MAX,
    _REGION_MIN_LEN,
    OperatorEngine,
)


class _MockFuzzer:
    def __init__(self, region_profile=True):
        self.max_len = 1 << 20
        self._rand_pool = RandPool()
        self._use_region_profile = region_profile
        self._use_transfer_entropy = False
        self._te = None
        self._use_mi = False
        self._mi = None
        self._use_sensitivity = False
        self._sensitivity = None
        self._crash_mi = None


def _mixed_seed() -> bytes:
    """Half incompressible (deflated noise), half tabular (u32 offsets)."""
    compressed = zlib.compress(os.urandom(16384), 9)[:8192]
    table = b"".join((i * 64).to_bytes(4, "little") for i in range(2048))
    return compressed + table


class TestRegionWeights:
    def test_short_seeds_have_no_profile(self):
        engine = OperatorEngine(_MockFuzzer())
        assert engine.region_weights(os.urandom(_REGION_MIN_LEN - 1)) is None

    def test_profile_is_cached_per_seed(self):
        """profile_buffer costs ~1 ms per window; it must not run per mutation."""
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        first = engine.region_weights(seed)
        assert first is not None
        assert engine.region_weights(seed) is first

    def test_cache_is_bounded(self):
        engine = OperatorEngine(_MockFuzzer())
        for _ in range(_REGION_CACHE_MAX + 5):
            engine.region_weights(os.urandom(8192))
        assert len(engine._region_cache) <= _REGION_CACHE_MAX

    def test_weights_are_cumulative_and_cover_the_seed(self):
        engine = OperatorEngine(_MockFuzzer())
        cumulative, bounds, total = engine.region_weights(_mixed_seed())
        assert cumulative == sorted(cumulative)
        assert cumulative[-1] == pytest.approx(total)
        assert bounds[0][0] == 0
        assert all(lo < hi for lo, hi in bounds)


class TestRegionWeightedPosition:
    def test_position_lands_inside_the_buffer(self):
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        for _ in range(200):
            pos = engine._region_weighted_position(seed, len(seed))
            assert pos is None or 0 <= pos < len(seed)

    def test_position_clamped_to_a_shrunken_buffer(self):
        """Bounds come from the seed, but the buffer may already be shorter.

        Earlier operators in the same mutation round resize the buffer, so an
        unclamped offset from the seed's profile would index past its end.
        """
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        for _ in range(200):
            pos = engine._region_weighted_position(seed, 600)
            assert pos is None or 0 <= pos < 600

    def test_disabled_by_default_in_select_position(self):
        """The flag gates the cost: no profiling unless asked for."""
        fuzzer = _MockFuzzer(region_profile=False)
        engine = OperatorEngine(fuzzer)
        seed = _mixed_seed()
        engine.select_position(bytearray(seed), seed)
        assert engine._region_cache == {}

    def test_enabled_flag_populates_the_cache(self):
        fuzzer = _MockFuzzer(region_profile=True)
        engine = OperatorEngine(fuzzer)
        seed = _mixed_seed()
        pos = engine.select_position(bytearray(seed), seed)
        assert 0 <= pos < len(seed)
        assert engine._region_cache

    def test_weighting_favours_the_tabular_half(self):
        """A deflate payload weighs 0.15, an offset table 1.6 — a >10x ratio.

        Asserted as a majority rather than a precise share: the split point
        between the two halves is a window boundary, not a byte boundary.
        """
        engine = OperatorEngine(_MockFuzzer())
        seed = _mixed_seed()
        split = 8192
        positions = [engine._region_weighted_position(seed, len(seed)) for _ in range(2000)]
        drawn = [p for p in positions if p is not None]
        assert len(drawn) > 1000
        assert sum(p >= split for p in drawn) / len(drawn) > 0.6
