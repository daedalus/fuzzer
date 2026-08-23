"""Regression: Lanczos ``_log_gamma`` fallback carried a spurious log(2**3.5).

The Lanczos form for g=7, n=9 is::

    log(sqrt(2*pi)) + (x + 0.5)*log(t) - t + log(y)

and the literal 0.9189385332046727 already IS log(sqrt(2*pi)). A leading
``0.5 * _LOG_2 * 7.0`` term was added on top of it, so every return value was
inflated by exactly log(2**3.5) = 2.4260151319598084.

The path is dead on CPython -- ``math.lgamma`` always exists and is preferred
at import time -- so nothing in the suite exercised it and the defect sat
undetected. It is the documented fallback for a build without ``lgamma``,
where it would have silently produced garbage p-values for every chi-squared
and NIST-battery test in the tree.

Expected values here are derived from ``math.lgamma`` and from closed-form
identities (Gamma(n) = (n-1)!, Gamma(1/2) = sqrt(pi)) -- NOT by recording what
the implementation happens to return, which is what would have pinned the
defect in place.
"""

import math

import pytest

from fuzzer_tool.core.chi_squared import _log_gamma

# log(2**3.5) -- the exact offset the spurious term introduced.
SPURIOUS_OFFSET = 0.5 * math.log(2.0) * 7.0


class TestLogGammaAgainstReference:
    @pytest.mark.parametrize(
        "x",
        [0.1, 0.5, 1.0, 1.5, 2.0, 3.7, 7.5, 10.0, 50.0, 123.456, 1000.0],
    )
    def test_matches_math_lgamma(self, x):
        assert _log_gamma(x) == pytest.approx(math.lgamma(x), rel=1e-12, abs=1e-9)

    def test_matches_factorial_identity(self):
        # Gamma(n) == (n-1)! for positive integers, computed independently of
        # any gamma implementation.
        for n in range(1, 15):
            expected = math.log(math.factorial(n - 1))
            assert _log_gamma(float(n)) == pytest.approx(expected, rel=1e-12, abs=1e-9)

    def test_matches_half_integer_identity(self):
        # Gamma(1/2) == sqrt(pi).
        assert _log_gamma(0.5) == pytest.approx(0.5 * math.log(math.pi), rel=1e-12)


class TestFalsification:
    """The bug, stated as a test that must now fail to reproduce."""

    def test_no_longer_offset_by_log_2_pow_3_5(self):
        # Falsification: if the spurious term were still present, _log_gamma
        # would sit exactly SPURIOUS_OFFSET above lgamma at every point.
        for x in (0.5, 1.0, 2.0, 10.0, 100.0):
            buggy = math.lgamma(x) + SPURIOUS_OFFSET
            assert _log_gamma(x) != pytest.approx(buggy, rel=1e-9), (
                f"_log_gamma({x}) still carries the log(2**3.5) offset"
            )

    def test_gamma_one_is_zero(self):
        # Gamma(1) == 1, so log Gamma(1) == 0. Under the bug this returned
        # 2.426 -- a value that is not merely imprecise but the wrong sign
        # for a quantity that must vanish here.
        assert _log_gamma(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_gamma_two_is_zero(self):
        # Gamma(2) == 1 likewise. Two independent zeros pin the constant.
        assert _log_gamma(2.0) == pytest.approx(0.0, abs=1e-6)


class TestAdversarial:
    """Inputs chosen to stress the branch, not to confirm it."""

    def test_monotonic_beyond_the_minimum(self):
        # log Gamma is convex with its minimum near x = 1.4616; beyond that it
        # increases without bound. A constant offset preserves monotonicity, so
        # this cannot catch the original bug on its own -- it guards the
        # rewrite against having broken the shape of the curve.
        xs = [1.5, 2.0, 5.0, 20.0, 200.0, 5000.0]
        vals = [_log_gamma(x) for x in xs]
        assert vals == sorted(vals)

    def test_large_argument_no_overflow(self):
        # t = x + 7.5 and log(t) stay finite far out; a naive Gamma would have
        # overflowed long before here.
        for x in (1e4, 1e6, 1e8):
            got = _log_gamma(x)
            assert math.isfinite(got)
            assert got == pytest.approx(math.lgamma(x), rel=1e-12)

    def test_near_pole_at_zero(self):
        # Gamma has a pole at 0, so log Gamma -> +inf as x -> 0+. Values stay
        # finite and large-positive rather than going NaN.
        for x in (1e-3, 1e-5, 1e-7):
            got = _log_gamma(x)
            assert math.isfinite(got)
            assert got > 0
            assert got == pytest.approx(math.lgamma(x), rel=1e-9)

    def test_reflection_formula(self):
        # Euler reflection: Gamma(x)*Gamma(1-x) == pi/sin(pi*x), an identity
        # the implementation never references, so it cannot be satisfied by
        # construction. A constant offset breaks it by 2*SPURIOUS_OFFSET.
        for x in (0.1, 0.25, 0.3, 0.45):
            lhs = _log_gamma(x) + _log_gamma(1.0 - x)
            rhs = math.log(math.pi / math.sin(math.pi * x))
            assert lhs == pytest.approx(rhs, rel=1e-10)
