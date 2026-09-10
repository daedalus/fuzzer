"""Unit tests for the streaming F0 estimator (CVM).

Tests the :class:`F0Estimator` from ``fuzzer_tool.core.cvm`` directly,
mirroring the style of ``tests/test_feistel.py``: module-level functions,
deterministic where possible, bare ``assert`` and ``pytest.approx``.
"""

import math

import pytest

from fuzzer_tool.core.cvm import F0Estimator
from fuzzer_tool.core.rand_pool import RandPool

# ── construction / validation ──────────────────────────────────────────────


def test_defaults_construct():
    f0 = F0Estimator()
    assert f0.eps == 0.1
    assert f0.delta == 1e-6
    assert f0.m == 1_000_000
    assert f0.p == 1.0
    assert f0.size == 0


def test_bad_eps_raises():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            F0Estimator(eps=bad)


def test_bad_delta_raises():
    for bad in (0.0, 1.0, -1e-9):
        with pytest.raises(ValueError):
            F0Estimator(delta=bad)


def test_bad_m_raises():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            F0Estimator(m=bad)


def test_thresh_formula():
    f0 = F0Estimator(eps=0.1, delta=1e-6, m=1000)
    expected = math.ceil((2.0 / (0.1**2)) * math.log((8.0 * 1000) / 1e-6))
    assert f0.thresh == expected


# ── update / estimate ──────────────────────────────────────────────────────


def test_update_returns_true_for_each_element():
    f0 = F0Estimator(eps=0.5, delta=0.25, m=100)
    for i in range(5):
        assert f0.update(i) is True


def test_estimate_empty_stream_is_zero():
    f0 = F0Estimator(eps=0.5, delta=0.25, m=100)
    assert f0.estimate() == 0.0


def test_estimate_matches_exact_count_when_p_is_one():
    f0 = F0Estimator(eps=0.5, delta=0.25, m=100)
    for i in range(20):
        f0.update(i)
    assert f0.estimate() == 20.0


def test_duplicates_do_not_change_estimate():
    f0 = F0Estimator(eps=0.5, delta=0.25, m=100)
    for i in range(10):
        f0.update(i)
    before = f0.estimate()
    for i in range(10):
        f0.update(i)  # re-insert duplicates
    assert f0.estimate() == before


def test_estimate_is_inf_when_p_zero():
    f0 = F0Estimator(eps=0.5, delta=0.25, m=100)
    f0.p = 0.0
    assert f0.estimate() == float("inf")


# ── clear ───────────────────────────────────────────────────────────────────


def test_clear_resets_state():
    f0 = F0Estimator(eps=0.5, delta=0.25, m=100)
    for i in range(10):
        f0.update(i)
    assert f0.size > 0
    f0.clear()
    assert f0.size == 0
    assert f0.p == 1.0
    assert f0.estimate() == 0.0


def test_clear_keeps_parameters():
    f0 = F0Estimator(eps=0.2, delta=1e-4, m=5000)
    f0.clear()
    assert f0.eps == 0.2
    assert f0.delta == 1e-4
    assert f0.m == 5000


# ── down-sampling ──────────────────────────────────────────────────────────


def test_down_sample_halves_p_at_threshold():
    # With p=1.0 every admission draw succeeds, so the first thresh
    # inserts fill X and trigger exactly one down-sample.
    f0 = F0Estimator(eps=0.5, delta=0.25, m=100)
    f0.thresh = 4
    for i in range(4):
        f0.update(i)
    assert f0.p == 0.5


def test_down_sample_first_thresh_minus_one_inserts_all_succeed():
    # Before the down-sample fires, every update() must return True.
    f0 = F0Estimator(eps=0.5, delta=0.25, m=100)
    f0.thresh = 4
    results = [f0.update(i) for i in range(3)]
    assert all(r is True for r in results)


def test_down_sample_can_return_none_when_still_full():
    # Force both the admission draw and the down-sample draws to keep
    # every element: update() must surface the algorithm's perp (None)
    # rather than silently succeeding.
    # Injected, not monkeypatched onto the module: since 8312b15 the
    # estimator holds its own pool and `cvm.random` does not exist, so
    # swapping it patched nothing and the draws came from the real pool.
    class KeepAll:
        def random(self):
            return 0.0  # always admit AND always keep in down-sample

    f0 = F0Estimator(eps=0.5, delta=0.25, m=100, rng=KeepAll())
    f0.thresh = 2

    assert f0.update("a") is True  # X={a}, len=1
    assert f0.update("b") is None  # X={a,b} -> down-sample keeps both
    # -> still full -> perp


# ── (eps, delta) guarantee (property test) ─────────────────────────────────


def test_estimate_exact_when_no_downsample_fires():
    """When the stream is shorter than thresh the estimate is exact."""
    f0 = F0Estimator(eps=0.5, delta=0.25, m=100)
    for i in range(10):
        f0.update(i)
    assert f0.estimate() == pytest.approx(10.0)


def test_estimate_within_eps_delta_after_downsample():
    """Property test: over many seeded runs the F0 estimate stays within
    (1 +/- eps) of the true distinct count with probability >= 1 - delta.

    Uses a stream long enough to trigger the down-sample at the formula's
    own threshold, and gives the estimator a per-run seeded pool for
    reproducibility.
    """
    eps, delta = 0.5, 0.25
    n_runs = 200
    true_count = 200
    bad = 0
    for seed in range(n_runs):
        f0 = F0Estimator(eps=eps, delta=delta, m=10_000, rng=RandPool(seed=seed))
        for i in range(true_count):
            f0.update(i)
        est = f0.estimate()
        if not (1 - eps <= est / true_count <= 1 + eps):
            bad += 1
    # The CVM guarantee bounds the violation rate by delta; allow a small
    # margin for finite-sample noise so the test is stable across platforms.
    assert bad / n_runs <= delta + 0.05
