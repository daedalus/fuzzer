"""RandPool discrete samplers: ``poisson_array`` and ``geometric``.

Each sampler is checked three ways: moments against the closed form,
goodness-of-fit against the exact pmf (in-house chi-squared, no scipy), and
a contingency test against an independent oracle sampler -- with the oracle
first compared against a second run of itself (Hard Rule 46).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.chi_squared import chi_squared_goodness_of_fit, chi_squared_test
from fuzzer_tool.core.exhaustive_pool import ContinuousDrawError, ExhaustivePool
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.scripted_rng import ScriptedRng

_N = 20_000
_ALPHA = 0.001  # fixed seeds: the p-values are deterministic, alpha is a margin
_LAM = 4.0
_P = 0.3
_MOMENT_SDS = 5.0  # moment tolerance in standard errors


def _hist(samples, k: int) -> list[float]:
    """Counts of 0..k-1 plus one tail bin for >= k."""
    a = np.minimum(np.asarray(samples, dtype=np.int64), k)
    return np.bincount(a, minlength=k + 1).astype(float).tolist()


def _poisson_pmf(lam: float, k: int) -> list[float]:
    """P(X=0..k-1) and the tail P(X>=k)."""
    head = [math.exp(-lam) * lam**i / math.factorial(i) for i in range(k)]
    return [*head, 1.0 - sum(head)]


def _geometric_pmf(p: float, k: int) -> list[float]:
    """Failures before first success: P(X=i) = p (1-p)^i, plus the tail."""
    head = [p * (1.0 - p) ** i for i in range(k)]
    return [*head, (1.0 - p) ** k]


def _knuth(rng: RandPool, lam: float, n: int) -> list[int]:
    """Independent oracle: Knuth's multiplicative Poisson sampler."""
    limit = math.exp(-lam)
    out = []
    for _ in range(n):
        k, prod = 0, rng.random()
        while prod > limit:
            k += 1
            prod *= rng.random()
        out.append(k)
    return out


def _trials(rng: RandPool, p: float, n: int) -> list[int]:
    """Independent oracle: count Bernoulli failures until the first success."""
    out = []
    for _ in range(n):
        k = 0
        while rng.random() >= p:
            k += 1
        out.append(k)
    return out


class _FixedU(RandPool):
    """RandPool whose ``random()`` replays a script and counts draws."""

    def __init__(self, us):
        super().__init__(seed=0)
        self.us = iter(us)
        self.draws = 0

    def random(self):
        self.draws += 1
        return next(self.us)


class TestPoissonArray:
    def test_moments_match_lambda(self):
        x = RandPool(seed=1).poisson_array(np.full(_N, _LAM))
        se = math.sqrt(_LAM / _N)
        assert abs(x.mean() - _LAM) < _MOMENT_SDS * se
        assert abs(x.var() - _LAM) / _LAM < 0.05

    def test_fits_the_exact_pmf(self):
        k = 10
        x = RandPool(seed=2).poisson_array(np.full(_N, _LAM))
        _, p, _ = chi_squared_goodness_of_fit(_hist(x, k), [_N * q for q in _poisson_pmf(_LAM, k)])
        assert p > _ALPHA

    def test_control_oracle_against_itself(self):
        """Hard Rule 46: two Knuth runs must be homogeneous first."""
        a = _hist(_knuth(RandPool(seed=3), _LAM, _N), 10)
        b = _hist(_knuth(RandPool(seed=4), _LAM, _N), 10)
        _, p, _ = chi_squared_test([a, b])
        assert p > _ALPHA

    def test_matches_knuth_oracle(self):
        a = _hist(RandPool(seed=5).poisson_array(np.full(_N, _LAM)), 10)
        b = _hist(_knuth(RandPool(seed=6), _LAM, _N), 10)
        _, p, _ = chi_squared_test([a, b])
        assert p > _ALPHA

    def test_falsification_wrong_lambda_is_rejected(self):
        """The GOF harness can fail: a 10% lambda error is caught."""
        k = 10
        x = RandPool(seed=2).poisson_array(np.full(_N, _LAM))
        exp = [_N * q for q in _poisson_pmf(1.1 * _LAM, k)]
        _, p, _ = chi_squared_goodness_of_fit(_hist(x, k), exp)
        assert p < 1e-6

    def test_per_element_rates_and_shape(self):
        lams = np.array([[0.0, 1.0], [50.0, 0.0]])
        x = RandPool(seed=7).poisson_array(lams)
        assert x.shape == lams.shape
        assert x[0, 0] == 0 and x[1, 1] == 0
        assert x[1, 0] > 0

    def test_adversarial_empty_and_negative(self):
        assert RandPool(seed=8).poisson_array(np.array([])).size == 0
        with pytest.raises(ValueError):
            RandPool(seed=8).poisson_array(np.array([1.0, -0.5]))

    def test_does_not_touch_the_pool_words(self):
        """Pool draws already fetched are not consumed by the numpy call."""
        a, b = RandPool(seed=9), RandPool(seed=9)
        a.random(), b.random()
        a.poisson_array(np.ones(3))
        assert a._idx == b._idx


class TestGeometric:
    def test_moments_match_closed_form(self):
        rng = RandPool(seed=11)
        x = np.array([rng.geometric(_P) for _ in range(_N)])
        mean, var = (1 - _P) / _P, (1 - _P) / _P**2
        assert abs(x.mean() - mean) < _MOMENT_SDS * math.sqrt(var / _N)
        assert abs(x.var() - var) / var < 0.06

    def test_fits_the_exact_pmf(self):
        k = 14
        rng = RandPool(seed=12)
        x = [rng.geometric(_P) for _ in range(_N)]
        exp = [_N * q for q in _geometric_pmf(_P, k)]
        _, p, _ = chi_squared_goodness_of_fit(_hist(x, k), exp)
        assert p > _ALPHA

    def test_control_oracle_against_itself(self):
        """Hard Rule 46: two Bernoulli-trial runs must be homogeneous first."""
        a = _hist(_trials(RandPool(seed=13), _P, _N), 14)
        b = _hist(_trials(RandPool(seed=14), _P, _N), 14)
        _, p, _ = chi_squared_test([a, b])
        assert p > _ALPHA

    def test_matches_trial_oracle(self):
        rng = RandPool(seed=15)
        a = _hist([rng.geometric(_P) for _ in range(_N)], 14)
        b = _hist(_trials(RandPool(seed=16), _P, _N), 14)
        _, p, _ = chi_squared_test([a, b])
        assert p > _ALPHA

    def test_falsification_wrong_p_is_rejected(self):
        k = 14
        rng = RandPool(seed=12)
        x = [rng.geometric(_P) for _ in range(_N)]
        exp = [_N * q for q in _geometric_pmf(1.1 * _P, k)]
        _, p, _ = chi_squared_goodness_of_fit(_hist(x, k), exp)
        assert p < 1e-6

    @pytest.mark.parametrize("u", [0.0, 0.25, 0.5, 0.75, 1.0 - 2**-32])
    def test_exact_inversion_with_scripted_uniform(self, u):
        """k is the unique integer with (1-p)^(k+1) < 1-u <= (1-p)^k."""
        rng = _FixedU([u])
        k = rng.geometric(_P)
        q = 1.0 - _P
        assert q ** (k + 1) < 1.0 - u <= q**k
        assert rng.draws == 1

    def test_adversarial_zero_uniform_is_finite(self):
        """U=0 must not reach log(0): it maps to the most likely outcome."""
        assert _FixedU([0.0]).geometric(0.5) == 0

    def test_p_one_is_zero_without_drawing(self):
        rng = _FixedU([])
        assert rng.geometric(1.0) == 0
        assert rng.draws == 0

    def test_consumes_exactly_one_pool_word(self):
        a, b = RandPool(seed=17), RandPool(seed=17)
        a.geometric(0.2)
        b.random()
        assert a.random() == b.random()

    def test_tiny_p_stays_accurate(self):
        """log1p keeps tiny p from collapsing log(1-p) to 0."""
        k = _FixedU([0.5]).geometric(1e-12)
        assert k == math.floor(math.log(0.5) / math.log1p(-1e-12))

    @pytest.mark.parametrize("p", [0.0, -0.1, 1.5, float("nan")])
    def test_adversarial_invalid_p_raises(self, p):
        with pytest.raises(ValueError):
            RandPool(seed=18).geometric(p)


class TestScriptedSamplers:
    def test_scripted_rng_replays_samplers(self):
        arr = np.array([3, 1])
        s = ScriptedRng(poisson_arrays=[arr], geometrics=[4, 0])
        assert s.poisson_array(np.ones(2)) is arr
        assert [s.geometric(0.5), s.geometric(0.9)] == [4, 0]
        with pytest.raises(StopIteration):
            s.geometric(0.5)


class TestExhaustiveRefusal:
    def test_adversarial_unbounded_samplers_are_refused(self):
        """Unbounded support has no finite enumeration: refuse, not truncate."""
        pool = ExhaustivePool()
        with pytest.raises(ContinuousDrawError):
            pool.poisson_array(np.ones(2))
        with pytest.raises(ContinuousDrawError):
            pool.geometric(0.5)
