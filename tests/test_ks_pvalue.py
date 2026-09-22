"""Tests for the shared KS p-value helpers (math-port P4)."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.ks_pvalue import (
    kolmogorov_pvalue_one_sample,
    kolmogorov_pvalue_two_sample,
    ks_exact_cdf,
)


class TestTwoSampleAsymptotic:
    def test_zero_d_is_one(self):
        assert kolmogorov_pvalue_two_sample(0.0, 10, 10) == 1.0

    def test_unit_d_is_zero(self):
        assert kolmogorov_pvalue_two_sample(1.0, 10, 10) == 0.0

    def test_monotonic_in_d(self):
        n = m = 50
        ps = [kolmogorov_pvalue_two_sample(d, n, m) for d in (0.05, 0.1, 0.2, 0.4)]
        assert all(p1 >= p2 for p1, p2 in zip(ps, ps[1:], strict=False))

    def test_matches_edge_tracker_alias(self):
        """Control: shared helper matches the backward-compat alias."""
        from fuzzer_tool.core.edge_tracker import _kolmogorov_pvalue

        for d, n, m in [(0.1, 20, 30), (0.3, 50, 50), (0.05, 100, 80)]:
            a = kolmogorov_pvalue_two_sample(d, n, m)
            b = _kolmogorov_pvalue(d, n, m)
            assert a == b


class TestOneSampleAsymptotic:
    def test_zero_d_is_one(self):
        assert kolmogorov_pvalue_one_sample(0.0, 20) == 1.0

    def test_matches_randomness_path(self):
        """Control: shared helper matches what ks_uniform uses for large n."""
        from fuzzer_tool.core.randomness import ks_uniform

        # Force asymptotic path (n > 140)
        pvals = [i / 200.0 for i in range(1, 200)]
        p_shared_path = ks_uniform(pvals, exact=False)
        # Direct computation of D then shared helper
        import numpy as np

        p = np.sort(np.asarray(pvals, dtype=np.float64))
        n = p.size
        i = np.arange(1, n + 1)
        d = max(float(np.max(i / n - p)), float(np.max(p - (i - 1) / n)))
        assert kolmogorov_pvalue_one_sample(d, n) == pytest.approx(p_shared_path, abs=1e-12)


class TestExactCdf:
    def test_boundary(self):
        assert ks_exact_cdf(10, 0.0) == 0.0
        assert ks_exact_cdf(10, 1.0) == 1.0

    def test_falsification_small_n_uniform_like(self):
        """Falsification: for uniform-ish D the CDF is in (0, 1)."""
        # Typical D under null is O(1/sqrt(n))
        n = 20
        d = 1.36 / math.sqrt(n)  # ~5% critical value
        cdf = ks_exact_cdf(n, d)
        assert 0.0 < cdf < 1.0

    def test_adversarial_tiny_n(self):
        """Adversarial: n=3 does not crash."""
        assert 0.0 <= ks_exact_cdf(3, 0.5) <= 1.0
