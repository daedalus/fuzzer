"""Coupon-collector budget helpers (tests/support/coupon.py)."""

import math

import pytest

from tests.support.coupon import DELTA, coverage_budget, skewed_budget


def _miss_bound(n: int, p_min: float, draws: int) -> float:
    """Union bound on P(rarest-or-any coupon unseen), from the definition."""
    return n * (1.0 - p_min) ** draws


@pytest.mark.parametrize(("n", "p_min"), [(2, 0.5), (16, 1 / 16), (270, 1e-4), (3, 0.01)])
def test_falsify_skewed_budget_is_tight(n, p_min):
    """Budget meets delta; one draw fewer does not."""
    t = skewed_budget(n, p_min)
    assert _miss_bound(n, p_min, t) <= DELTA
    assert _miss_bound(n, p_min, t - 1) > DELTA


def test_uniform_budget_covers_skewed_at_one_over_n():
    """-ln(1 - 1/n) > 1/n, so the exact budget never exceeds the n(ln n + ln 1/d) form."""
    for n in (2, 16, 270):
        assert skewed_budget(n, 1 / n) <= coverage_budget(n)
        assert _miss_bound(n, 1 / n, coverage_budget(n)) <= DELTA


@pytest.mark.parametrize(
    ("n", "p_min"), [(0, 0.5), (2, 0.0), (2, 1.0), (4, 0.5), (2, -0.1), (2, math.nan)]
)
def test_adversarial_impossible_inputs_rejected(n, p_min):
    """p_min above 1/n is impossible for the rarest of n; zero never covers."""
    with pytest.raises(ValueError):
        skewed_budget(n, p_min)
