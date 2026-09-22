"""Tests for core/badness_floor.py (P1: badness-indexed exploration floor)."""

import pytest

from fuzzer_tool.core.badness_floor import (
    DEFAULT_MAX_EXPLORE_FLOOR,
    badness_from_regime,
    floor_for_badness,
)
from fuzzer_tool.core.percolation import CoverageRegime


class TestBadnessFromRegime:
    def test_subcritical_is_worst_case(self):
        assert badness_from_regime(CoverageRegime.SUBCRITICAL) == 1.0

    def test_supercritical_is_best_case(self):
        assert badness_from_regime(CoverageRegime.SUPERCRITICAL) == 0.0

    def test_critical_is_midpoint(self):
        assert badness_from_regime(CoverageRegime.CRITICAL) == 0.5

    def test_none_is_neutral_not_best_case(self):
        # An absent signal must not silently behave like a healthy regime.
        assert badness_from_regime(None) == 0.5


class TestFloorForBadness:
    def test_badness_zero_returns_base_floor(self):
        assert floor_for_badness(0.06, 0.0, 0.25) == pytest.approx(0.06)

    def test_badness_one_returns_max_floor(self):
        assert floor_for_badness(0.06, 1.0, 0.25) == pytest.approx(0.25)

    def test_linear_interpolation_midpoint(self):
        assert floor_for_badness(0.06, 0.5, 0.26) == pytest.approx(0.16)

    def test_badness_clamped_above_one(self):
        assert floor_for_badness(0.06, 5.0, 0.25) == pytest.approx(0.25)

    def test_badness_clamped_below_zero(self):
        assert floor_for_badness(0.06, -5.0, 0.25) == pytest.approx(0.06)

    def test_default_max_floor(self):
        assert floor_for_badness(0.0, 1.0) == pytest.approx(DEFAULT_MAX_EXPLORE_FLOOR)

    def test_max_floor_below_base_floor_rejected(self):
        with pytest.raises(ValueError, match="max_floor"):
            floor_for_badness(0.3, 0.5, max_floor=0.1)

    def test_max_floor_equal_to_one_rejected(self):
        with pytest.raises(ValueError, match="max_floor"):
            floor_for_badness(0.06, 0.5, max_floor=1.0)

    def test_negative_base_floor_rejected(self):
        with pytest.raises(ValueError, match="max_floor"):
            floor_for_badness(-0.1, 0.5, max_floor=0.25)

    def test_max_floor_equal_to_base_floor_is_a_constant_family(self):
        # Degenerate but valid: every badness maps to the same floor.
        assert floor_for_badness(0.1, 0.0, max_floor=0.1) == pytest.approx(0.1)
        assert floor_for_badness(0.1, 1.0, max_floor=0.1) == pytest.approx(0.1)
