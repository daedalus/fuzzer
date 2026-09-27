"""Tests for core/zipf.py -- discrete power-law tail fit and Heaps' law fit.

Fixtures are deterministic multisets built from the model's own PMF
(``n_k = round(N * pmf(k))``), so no RNG is involved (Hard Rules 16, 39).
"""

import math
import time
from statistics import NormalDist

import pytest

from fuzzer_tool.core.zipf import (
    ALPHA_HI,
    MIN_TAIL,
    TailLaw,
    fit_heaps,
    fit_zipf,
    hurwitz,
)

SUPPORT = 2000  # largest value a PMF fixture places mass on
TOTAL = 20_000  # fixture size before rounding
ALPHA_TOL = 0.02


def _zipf_counts(alpha: float, support: int = SUPPORT, total: int = TOTAL) -> list[int]:
    """Values 1..support, each repeated round(total * pmf(k)) times (truncated pmf)."""
    norm = sum(k**-alpha for k in range(1, support + 1))
    out: list[int] = []
    for k in range(1, support + 1):
        out.extend([k] * round(total * k**-alpha / norm))
    return out


def _geometric_counts(p: float, total: int = TOTAL) -> list[int]:
    out: list[int] = []
    k = 1
    while True:
        n = round(total * p * (1 - p) ** (k - 1))
        if n == 0 and k > 1:
            return out
        out.extend([k] * n)
        k += 1


def _lognormal_counts(mu: float, sigma: float, total: int = TOTAL) -> list[int]:
    """Deterministic quantile grid of a lognormal, rounded up to integers."""
    nd = NormalDist(mu, sigma)
    return [math.ceil(math.exp(nd.inv_cdf((i + 0.5) / total))) for i in range(total)]


class TestHurwitz:
    def test_control_basel(self):
        # zeta(2, 1) = pi^2 / 6 (Euler).
        assert hurwitz(2.0, 1.0) == pytest.approx(math.pi**2 / 6, rel=1e-12)

    def test_control_direct_sum(self):
        # Partial sum to K plus the integral tail bound, independent of Euler-Maclaurin.
        s, q, k = 1.5, 3.0, 200_000
        direct = sum((q + i) ** -s for i in range(k))
        tail = (q + k) ** (1 - s) / (s - 1) + 0.5 * (q + k) ** -s
        assert hurwitz(s, q) == pytest.approx(direct + tail, rel=1e-9)

    def test_shift_identity(self):
        # zeta(s, q) = q^-s + zeta(s, q + 1).
        for s, q in ((1.2, 1.0), (2.7, 5.0), (4.0, 40.0)):
            assert hurwitz(s, q) == pytest.approx(q**-s + hurwitz(s, q + 1), rel=1e-12)

    def test_rejects_divergent_order(self):
        with pytest.raises(ValueError):
            hurwitz(1.0, 1.0)


class TestFitZipfControl:
    def test_interleaved_halves_agree(self):
        """Hard Rule 46: the estimator must agree with itself on two halves of one law."""
        data = _zipf_counts(2.0)
        a = fit_zipf(data[0::2], xmax=max(data))
        b = fit_zipf(data[1::2], xmax=max(data))
        assert a.law is TailLaw.POWER_LAW
        assert b.law is TailLaw.POWER_LAW
        assert a.alpha == pytest.approx(b.alpha, abs=ALPHA_TOL)


class TestFitZipfRecovers:
    @pytest.mark.parametrize("alpha", [1.5, 2.0, 2.5, 3.0])
    def test_recovers_alpha(self, alpha):
        # Rounding drops every k with total * pmf(k) < 0.5, so the fixture's
        # real support ends at max(data), not SUPPORT.
        data = _zipf_counts(alpha)
        fit = fit_zipf(data, xmax=max(data))
        assert fit.law is TailLaw.POWER_LAW
        assert fit.alpha == pytest.approx(alpha, abs=ALPHA_TOL)
        assert fit.s == pytest.approx(1.0 / (fit.alpha - 1.0))

    def test_order_independent(self):
        data = _zipf_counts(2.0)
        assert fit_zipf(data, xmax=SUPPORT) == fit_zipf(data[::-1], xmax=SUPPORT)

    def test_cap_reduces_truncation_bias(self):
        # Truncating the support steepens the empirical tail; ignoring the cap
        # must read a larger alpha than honouring it.
        alpha, cap = 1.8, 200
        data = _zipf_counts(alpha, support=cap, total=200_000)
        capped = fit_zipf(data, xmax=cap)
        uncapped = fit_zipf(data)
        assert abs(capped.alpha - alpha) < abs(uncapped.alpha - alpha)
        assert capped.alpha == pytest.approx(alpha, abs=ALPHA_TOL)


class TestFitZipfFalsification:
    def test_geometric_is_not_power_law(self):
        fit = fit_zipf(_geometric_counts(0.3))
        assert fit.law is TailLaw.NOT_POWER_LAW

    def test_lognormal_small_tail_is_not_power_law(self):
        fit = fit_zipf(_lognormal_counts(1.0, 1.2))
        assert fit.law is TailLaw.NOT_POWER_LAW


class TestFitZipfAdversarial:
    def test_empty(self):
        assert fit_zipf([]).law is TailLaw.INSUFFICIENT

    def test_all_ones(self):
        # Every edge owned by one seed: one distinct value, no slope to fit.
        assert fit_zipf([1] * 5000).law is TailLaw.INSUFFICIENT

    def test_too_few_points(self):
        assert fit_zipf(_zipf_counts(2.0)[: MIN_TAIL - 1]).law is TailLaw.INSUFFICIENT

    def test_non_positive_values_ignored(self):
        # _edge_owner_count is a defaultdict: stray zero entries must not count.
        data = _zipf_counts(2.0)
        assert fit_zipf(data + [0] * 1000, xmax=SUPPORT) == fit_zipf(data, xmax=SUPPORT)

    def test_huge_values_bounded_time(self):
        data = _zipf_counts(1.5) + [10**12, 10**11, 10**10]
        t0 = time.perf_counter()
        fit = fit_zipf(data)
        assert time.perf_counter() - t0 < 1.0
        assert math.isfinite(fit.alpha)

    def test_alpha_stays_in_bracket(self):
        fit = fit_zipf(_geometric_counts(0.9))
        assert fit.alpha <= ALPHA_HI


class TestFitHeaps:
    def test_recovers_exact_power(self):
        execs = [int(100 * 1.1**i) for i in range(80)]
        edges = [round(3.0 * n**0.6) for n in execs]
        fit = fit_heaps(execs, edges)
        assert fit is not None
        assert fit.beta == pytest.approx(0.6, abs=0.01)
        assert fit.r2 > 0.99
        assert fit.project(execs[-1]) == pytest.approx(edges[-1], rel=0.02)

    def test_doubling_gain(self):
        # D(2N) / D(N) = 2^beta, independent of k.
        fit = fit_heaps([10, 100, 1000, 10000], [5 * n**0.5 for n in (10, 100, 1000, 10000)])
        assert fit is not None
        assert fit.doubling_gain == pytest.approx(fit.project(2000) / fit.project(1000) - 1.0)

    def test_plateau_has_zero_beta(self):
        execs = list(range(1000, 100_000, 1000))
        fit = fit_heaps(execs, [500] * len(execs))
        assert fit is not None
        assert fit.beta == pytest.approx(0.0, abs=1e-9)

    def test_too_few_points(self):
        assert fit_heaps([1, 2], [1, 2]) is None

    def test_zero_points_dropped(self):
        # exec 0 / edges 0 have no logarithm; they must be skipped, not crash.
        execs = [0] + [int(100 * 1.1**i) for i in range(40)]
        edges = [0] + [round(2.0 * n**0.5) for n in execs[1:]]
        fit = fit_heaps(execs, edges)
        assert fit is not None
        assert fit.beta == pytest.approx(0.5, abs=0.02)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            fit_heaps([1, 2, 3, 4], [1, 2, 3])
