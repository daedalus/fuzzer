"""quick_health_check: Poisson goodness-of-fit measure (``poisson``)."""

import numpy as np

from fuzzer_tool.core.exhaustive_pool import ExhaustivePool
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.rng_health import _P_THRESHOLD, _poisson_pvalue, quick_health_check


class _Overdispersed(RandPool):
    """Mean 4 like Poisson(4), but variance 20: geometric counts."""

    def poisson_array(self, lams):
        return np.random.default_rng(0).geometric(0.2, size=len(lams)) - 1


class _Shifted(RandPool):
    def poisson_array(self, lams):
        return np.random.default_rng(0).poisson(4.6, size=len(lams))


class _NoPoisson:
    def randint_list(self, a, b, count):
        return RandPool(seed=1).randint_list(a, b, count)


def test_reported_in_summary_and_pvalues():
    result = quick_health_check(RandPool(seed=1))
    assert 0.0 <= result.pvalues["poisson"] <= 1.0
    assert "poisson=" in result.summary()
    assert result.ok


def test_healthy_pools_are_not_flagged_and_p_is_roughly_uniform():
    ps = np.array([_poisson_pvalue(RandPool(seed=s)) for s in range(300)])
    assert (ps < _P_THRESHOLD).mean() < 0.03
    assert 0.4 < ps.mean() < 0.6


def test_overdispersed_and_shifted_streams_are_caught():
    assert _poisson_pvalue(_Overdispersed(seed=1)) < 1e-6
    assert _poisson_pvalue(_Shifted(seed=1)) < 1e-6
    assert not quick_health_check(_Overdispersed(seed=1)).ok


def test_skipped_when_unavailable():
    assert "poisson" not in quick_health_check(_NoPoisson()).pvalues
    assert _poisson_pvalue(ExhaustivePool()) is None  # poisson_array raises ContinuousDrawError


def test_does_not_advance_live_streams():
    a, b = RandPool(seed=9), RandPool(seed=9)
    quick_health_check(b)
    # quick_health_check draws bytes from the main stream either way; compare
    # the aux stream, which must be untouched by the Poisson check.
    assert a.poisson_array(np.full(50, 3.0)).tolist() == b.poisson_array(np.full(50, 3.0)).tolist()


def test_deterministic():
    assert _poisson_pvalue(RandPool(seed=4)) == _poisson_pvalue(RandPool(seed=4))
