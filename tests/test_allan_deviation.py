"""Tests for core/allan_deviation.py.

Expectations for the noise-classification tests are derived from the
textbook power-law-noise exponents (white FM -> slope -1/2, random-walk
FM -> slope +1/2), verified against a fixed-seed synthetic series, not
asserted against the implementation's own output -- same discipline as
test_pll.py and test_kuramoto.py.
"""

from __future__ import annotations

import math
import random

import pytest

from fuzzer_tool.core.allan_deviation import (
    NoiseRegime,
    allan_deviation,
    classify_segments,
)


def _white_fm(n: int, seed: int = 1234) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def _random_walk_fm(n: int, seed: int = 1234) -> list[float]:
    rng = random.Random(seed)
    out = []
    acc = 0.0
    for _ in range(n):
        acc += rng.gauss(0.0, 1.0)
        out.append(acc)
    return out


class TestValidation:
    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError):
            allan_deviation([1.0, 2.0, 3.0])

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError):
            allan_deviation([1.0, float("nan"), 3.0, 4.0, 5.0])

    def test_rejects_infinite(self):
        with pytest.raises(ValueError):
            allan_deviation([1.0, float("inf"), 3.0, 4.0, 5.0])

    def test_rejects_nonpositive_tau0(self):
        with pytest.raises(ValueError):
            allan_deviation([1.0, 2.0, 3.0, 4.0, 5.0], tau0=0.0)

    def test_rejects_negative_tau0(self):
        with pytest.raises(ValueError):
            allan_deviation([1.0, 2.0, 3.0, 4.0, 5.0], tau0=-1.0)

    def test_rejects_non_positive_int_m(self):
        with pytest.raises(ValueError):
            allan_deviation([1.0] * 20, m_values=[0])

    def test_rejects_float_m(self):
        with pytest.raises(ValueError):
            allan_deviation([1.0] * 20, m_values=[1.5])

    def test_minimum_length_accepted(self):
        # N=4 with default m_values: only m=1 leaves n_pairs=2 >= 1.
        pts = allan_deviation([1.0, 2.0, 1.0, 2.0])
        assert len(pts) >= 1


class TestAllanPointBookkeeping:
    def test_ascending_by_tau(self):
        pts = allan_deviation(_white_fm(256), tau0=1.0)
        taus = [p.tau for p in pts]
        assert taus == sorted(taus)

    def test_tau_equals_m_times_tau0(self):
        pts = allan_deviation(_white_fm(256), tau0=2.5, m_values=[1, 4, 16])
        for p in pts:
            assert p.tau == pytest.approx(p.m * 2.5)

    def test_n_pairs_matches_formula(self):
        n = 100
        pts = allan_deviation(_white_fm(n), m_values=[1, 10, 40])
        by_m = {p.m: p.n_pairs for p in pts}
        assert by_m[1] == n - 2
        assert by_m[10] == n - 20
        assert by_m[40] == n - 80

    def test_m_too_large_is_dropped_not_raised(self):
        n = 20
        # m=10 leaves n_pairs = 0, must be silently dropped.
        pts = allan_deviation(_white_fm(n), m_values=[1, 10])
        assert {p.m for p in pts} == {1}

    def test_explicit_m_values_all_too_large_gives_empty_list(self):
        pts = allan_deviation(_white_fm(10), m_values=[5, 6])
        assert pts == []

    def test_constant_series_gives_zero_adev(self):
        pts = allan_deviation([5.0] * 40, m_values=[1, 2, 4])
        assert all(p.adev == 0.0 for p in pts)

    def test_duplicate_m_values_deduplicated(self):
        pts = allan_deviation(_white_fm(100), m_values=[4, 4, 4, 8])
        assert [p.m for p in pts] == [4, 8]

    def test_default_m_values_are_powers_of_two_up_to_n_over_4(self):
        pts = allan_deviation(_white_fm(64))
        assert [p.m for p in pts] == [1, 2, 4, 8, 16]


class TestClassifySegments:
    def test_constant_series_gives_no_segments(self):
        pts = allan_deviation([3.0] * 40, m_values=[1, 2, 4])
        assert classify_segments(pts) == []

    def test_white_fm_classified_as_white_fm(self):
        # Textbook exponent: white frequency noise -> ADEV ~ tau^(-1/2).
        # Use the low-m range where the overlapping-window count is
        # largest and the slope estimate is least noisy.
        pts = allan_deviation(_white_fm(8192), m_values=[1, 2, 4, 8, 16, 32, 64])
        segments = classify_segments(pts)
        assert len(segments) == len(pts) - 1
        regimes = [s.regime for s in segments]
        # Not every adjacent pair need land inside the classification
        # band (estimator noise), but the large majority should, and
        # none should invert to the opposite-signed regime.
        assert regimes.count(NoiseRegime.WHITE_FM) >= len(regimes) - 1
        assert NoiseRegime.RANDOM_WALK_FM not in regimes
        assert NoiseRegime.DRIFT not in regimes

    def test_random_walk_fm_classified_as_random_walk_fm(self):
        # Textbook exponent: random-walk frequency noise -> ADEV ~ tau^(+1/2).
        pts = allan_deviation(_random_walk_fm(8192), m_values=[1, 2, 4, 8, 16, 32, 64])
        segments = classify_segments(pts)
        regimes = [s.regime for s in segments]
        assert regimes.count(NoiseRegime.RANDOM_WALK_FM) >= len(regimes) - 1
        assert NoiseRegime.WHITE_FM not in regimes
        assert NoiseRegime.PHASE_NOISE not in regimes

    def test_white_fm_and_random_walk_fm_slopes_have_opposite_sign(self):
        white_pts = allan_deviation(_white_fm(4096), m_values=[1, 2, 4, 8, 16])
        rw_pts = allan_deviation(_random_walk_fm(4096), m_values=[1, 2, 4, 8, 16])
        white_slope = classify_segments(white_pts)[0].slope
        rw_slope = classify_segments(rw_pts)[0].slope
        assert white_slope < 0.0
        assert rw_slope > 0.0

    def test_segment_bounds_match_points(self):
        pts = allan_deviation(_white_fm(256), m_values=[1, 2, 4])
        segments = classify_segments(pts)
        assert segments[0].tau_lo == pts[0].tau
        assert segments[0].tau_hi == pts[1].tau
        assert segments[1].tau_lo == pts[1].tau
        assert segments[1].tau_hi == pts[2].tau

    def test_canonical_slopes_classify_exactly(self):
        # Bypass allan_deviation entirely: fabricate points whose adev
        # follows an exact power law at the canonical exponent for each
        # regime, and confirm classify_segments recovers that regime.
        from fuzzer_tool.core.allan_deviation import AllanPoint

        for regime, mu in [
            (NoiseRegime.PHASE_NOISE, -1.0),
            (NoiseRegime.WHITE_FM, -0.5),
            (NoiseRegime.FLICKER_FM, 0.0),
            (NoiseRegime.RANDOM_WALK_FM, 0.5),
            (NoiseRegime.DRIFT, 1.0),
        ]:
            taus = [1.0, 2.0]
            adevs = [1.0, math.pow(2.0, mu)]
            pts = [
                AllanPoint(tau=t, m=int(t), adev=a, n_pairs=100)
                for t, a in zip(taus, adevs, strict=True)
            ]
            segments = classify_segments(pts)
            assert len(segments) == 1
            assert segments[0].regime is regime

    def test_slope_far_from_any_canonical_exponent_is_unknown(self):
        from fuzzer_tool.core.allan_deviation import AllanPoint

        pts = [
            AllanPoint(tau=1.0, m=1, adev=1.0, n_pairs=100),
            AllanPoint(tau=2.0, m=2, adev=8.0, n_pairs=100),  # slope = 3.0
        ]
        segments = classify_segments(pts)
        assert segments[0].regime is NoiseRegime.UNKNOWN
