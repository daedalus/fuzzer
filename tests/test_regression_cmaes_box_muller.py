"""Box-Muller yields two normals per uniform pair; _randn kept one.

The transform maps (U1, U2) to r*cos(theta) AND r*sin(theta), both standard
normal and independent. _randn computed only the cosine branch and dropped
the sine, so it drew 2n uniforms from the pool to produce n normals. CPython's
own random.gauss caches the sine branch in gauss_next for exactly this reason.

The log(0) guard is also wrong in a way the pool makes reachable: RandPool
uniforms are raw_uint32 / 2**32, so exact 0.0 occurs with probability 2**-32,
and clamping it to 1e-300 yields sqrt(-2*log(1e-300)) = 37.2 sigma -- it
manufactures an outlier rather than truncating. A uint32 uniform supports at
most 6.66 sigma. log1p(-u) on [0, 1) removes the special case: u = 0 maps to
0.0, which is in range and needs no clamp.
"""

import math

import numpy as np
import pytest
from scipy import stats

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.cmaes import CMAESScheduler


def _sched(seed=5):
    return CMAESScheduler(rng=RandPool(seed=seed))


@pytest.mark.parametrize("n", [1, 2, 3, 8, 155, 500])
def test_returns_requested_length(n):
    assert _sched()._randn(n).shape == (n,)


@pytest.mark.parametrize("n", [2, 8, 154, 155])
def test_consumes_one_uniform_per_normal(n):
    """The whole point of the transform: two uniforms, two normals."""
    sched = _sched()
    pool = sched._rng
    pool.random_list(4)  # land mid-pool so _idx is meaningful
    before = pool._idx
    sched._randn(n)
    used = pool._idx - before

    assert used <= n + 1, f"{used} uniforms for {n} normals"


def test_both_branches_are_used():
    """Falsification: dropping sin() halves throughput and this catches it."""
    sched = _sched()
    pool = sched._rng
    pool.random_list(4)
    before = pool._idx
    sched._randn(64)

    assert pool._idx - before != 128


def test_is_standard_normal():
    sched = _sched(seed=17)
    draws = np.concatenate([sched._randn(1000) for _ in range(300)])

    assert draws.mean() == pytest.approx(0.0, abs=0.01)
    assert draws.std() == pytest.approx(1.0, abs=0.01)
    assert stats.kurtosis(draws) == pytest.approx(0.0, abs=0.05)


def test_matches_a_reference_normal_with_a_control():
    """Two-sample KS, with the reference against itself as the control."""
    sched = _sched(seed=23)
    sample = np.concatenate([sched._randn(1000) for _ in range(200)])
    gen = np.random.default_rng(4)
    ref = gen.standard_normal(sample.size)
    control = gen.standard_normal(sample.size)

    _, p_control = stats.ks_2samp(ref, control)
    _, p_sample = stats.ks_2samp(ref, sample)

    assert p_control > 0.001, "control failed — the oracle itself is broken"
    assert p_sample > 0.001


def test_halves_are_independent():
    """Adversarial: returning the cosine branch twice would pass the KS test."""
    sched = _sched(seed=29)
    draws = np.concatenate([sched._randn(1000) for _ in range(100)]).reshape(-1, 1000)
    first, second = draws[:, :500].ravel(), draws[:, 500:].ravel()

    assert abs(np.corrcoef(first, second)[0, 1]) < 0.02


def test_zero_uniform_does_not_produce_an_absurd_outlier():
    """u1 == 0.0 is reachable: raw_uint32 / 2**32 hits it with p = 2**-32."""

    class _ZeroPool:
        def random_list(self, n):
            return [0.0] * n

    sched = CMAESScheduler(rng=_ZeroPool())

    out = sched._randn(8)

    assert np.all(np.abs(out) <= 6.7), f"max |z| = {np.abs(out).max()}"
    assert np.all(np.isfinite(out))


def test_stays_within_the_uint32_precision_limit():
    """A uint32 uniform cannot support more than this many sigma."""
    sched = _sched(seed=31)
    draws = np.concatenate([sched._randn(2000) for _ in range(200)])

    assert np.abs(draws).max() < math.sqrt(-2.0 * math.log(2.0**-32))


def test_is_seed_reproducible():
    assert np.array_equal(_sched(seed=41)._randn(64), _sched(seed=41)._randn(64))
