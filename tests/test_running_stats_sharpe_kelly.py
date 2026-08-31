"""Unit tests for sharpe_ratio and kelly_fraction in running_stats."""

import pytest

from fuzzer_tool.core.running_stats import kelly_fraction, sharpe_ratio


class TestSharpeRatio:
    def test_zero_stddev_positive_mean_returns_inf(self):
        assert sharpe_ratio(1.0, 0.0) == float("inf")

    def test_zero_stddev_zero_mean_returns_zero(self):
        assert sharpe_ratio(0.0, 0.0) == 0.0

    def test_negative_stddev_returns_zero(self):
        assert sharpe_ratio(1.0, -0.1) == 0.0

    def test_basic_calculation(self):
        assert sharpe_ratio(2.0, 1.0) == pytest.approx(2.0)

    def test_less_than_one_sharpe(self):
        assert sharpe_ratio(0.5, 2.0) == pytest.approx(0.25)

    def test_zero_mean_positive_stddev(self):
        assert sharpe_ratio(0.0, 1.0) == 0.0

    def test_symmetric_positive_negative(self):
        # Sharpe is signed: negative mean / positive stddev = negative ratio
        assert sharpe_ratio(-1.0, 2.0) == pytest.approx(-0.5)

    def test_negative_mean_zero_stddev_returns_neg_inf(self):
        assert sharpe_ratio(-1.0, 0.0) == float("-inf")


class TestKellyFraction:
    def test_zero_variance_returns_zero(self):
        assert kelly_fraction(1.0, 0.0) == 0.0

    def test_negative_variance_returns_zero(self):
        assert kelly_fraction(1.0, -0.1) == 0.0

    def test_basic_calculation(self):
        # f* = mean / variance = 4.0 / 2.0 = 2.0, clamped to 1.0
        assert kelly_fraction(4.0, 2.0) == pytest.approx(1.0)

    def test_fractional_kelly(self):
        # f* = 0.5 / 0.5 = 1.0
        assert kelly_fraction(0.5, 0.5) == pytest.approx(1.0)

    def test_small_fraction(self):
        # f* = 0.1 / 1.0 = 0.1
        assert kelly_fraction(0.1, 1.0) == pytest.approx(0.1)

    def test_negative_kelly_clamped_to_zero(self):
        # mean < 0 → negative Kelly → clamp to 0.0
        assert kelly_fraction(-1.0, 1.0) == pytest.approx(0.0)

    def test_zero_mean_zero_variance(self):
        assert kelly_fraction(0.0, 0.0) == 0.0

    def test_high_variance_small_kelly(self):
        # f* = 0.01 / 4.0 = 0.0025
        assert kelly_fraction(0.01, 4.0) == pytest.approx(0.0025)
