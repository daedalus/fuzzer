"""The cached-CDF pick must draw exactly what the pool's own weighted pick drew.

``_cdf_pick`` replaces ``rng.weighted_choice(pop, weights)`` with a cached
prefix sum plus a bisect. That is only worth doing if it is a substitution
and not a re-derivation: it has to consume the same number of RNG values,
scale them the same way and land on the same index, or a fixed ``--seed``
stops reproducing the same campaign.

These tests therefore compare against ``RandPool.weighted_choice`` under a
shared RNG state rather than against a distribution.

The three degenerate inputs are checked separately. ``_cdf_pick`` raises
them itself rather than deferring: ``weighted_choice`` turns a zero or
non-finite total into an IndexError off the end of the prefix sum, and does
not raise at all on a length mismatch -- it returns a wrong element. The
ValueError type and message are the ones ``random.choices`` raised before
the Hard Rule 16 migration, because the callers were written against those.
"""

import itertools
import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.seed_picker import _cdf_pick


def _pairs(population, weights, n=500, seed=1234):
    """Draw n picks each way from the same starting RNG state."""
    return _pairs_shared(population, weights, n=n, seed=seed)


def _pairs_shared(population, weights, n=500, seed=1234):
    """Same, but one pool per side so the streams start aligned."""
    store = {}
    a = RandPool(seed=seed)
    fast = [_cdf_pick(population, weights, store, "s", a) for _ in range(n)]
    b = RandPool(seed=seed)
    ref = [b.weighted_choice(population, weights) for _ in range(n)]
    return fast, ref


def test_draws_are_identical_to_weighted_choice():
    rng = RandPool(seed=0)
    population = [f"seed{i}".encode() for i in range(200)]
    weights = [rng.random() * 10 + 0.01 for _ in range(200)]
    fast, ref = _pairs_shared(population, weights)
    assert fast == ref


def test_identical_under_degenerate_weights():
    """One dominant weight and a long tail of near-zeros."""
    population = list(range(50))
    weights = [1e-9] * 49 + [1.0]
    fast, ref = _pairs(population, weights, n=300)
    assert fast == ref


def test_identical_with_uniform_weights():
    population = list(range(37))
    weights = [1.0] * 37
    fast, ref = _pairs(population, weights, n=400)
    assert fast == ref


def test_consumes_exactly_one_random_per_pick():
    """RNG advance must match, or a seeded run diverges after the first pick."""
    population = list(range(20))
    weights = [float(i + 1) for i in range(20)]

    store = {}
    a = RandPool(seed=99)
    for _ in range(10):
        _cdf_pick(population, weights, store, "s", a)
    after_fast = a.random()

    b = RandPool(seed=99)
    for _ in range(10):
        b.weighted_choice(population, weights)
    after_ref = b.random()

    assert after_fast == after_ref


def test_cache_is_keyed_on_the_weight_list_identity():
    population = list(range(10))
    weights = [1.0] * 10
    store = {}
    rng = RandPool(seed=0)
    _cdf_pick(population, weights, store, "s", rng)
    first = store["s"]
    _cdf_pick(population, weights, store, "s", rng)
    assert store["s"] is first, "same list must reuse the cached prefix sum"

    replacement = [2.0] * 10
    _cdf_pick(population, replacement, store, "s", rng)
    assert store["s"] is not first
    assert store["s"][0] is replacement


def test_new_weights_take_effect_immediately():
    """A stale CDF would keep sampling the old distribution."""
    population = ["a", "b"]
    store = {}
    rng = RandPool(seed=5)
    assert {_cdf_pick(population, [1.0, 0.0], store, "s", rng) for _ in range(50)} == {"a"}
    assert {_cdf_pick(population, [0.0, 1.0], store, "s", rng) for _ in range(50)} == {"b"}


def test_cached_prefix_sum_matches_accumulate_exactly():
    rng = RandPool(seed=3)
    weights = [rng.random() for _ in range(64)]
    store = {}
    _cdf_pick(list(range(64)), weights, store, "s", rng)
    assert store["s"][1] == list(itertools.accumulate(weights))


def test_length_mismatch_still_raises():
    with pytest.raises(ValueError):
        _cdf_pick([1, 2, 3], [1.0, 1.0], {}, "s", RandPool(seed=0))


def test_zero_total_still_raises():
    with pytest.raises(ValueError):
        _cdf_pick([1, 2, 3], [0.0, 0.0, 0.0], {}, "s", RandPool(seed=0))


def test_non_finite_total_still_raises():
    with pytest.raises(ValueError):
        _cdf_pick([1, 2, 3], [math.inf, 1.0, 1.0], {}, "s", RandPool(seed=0))


def test_empty_population_defers():
    with pytest.raises(IndexError):
        _cdf_pick([], [], {}, "s", RandPool(seed=0))


def test_slots_do_not_collide():
    """The corpus vector and the Pareto-front slice share one store."""
    store = {}
    rng = RandPool(seed=7)
    corpus = list(range(30))
    corpus_w = [1.0] * 30
    front = [100, 200]
    front_w = [1.0, 3.0]
    for _ in range(20):
        assert _cdf_pick(corpus, corpus_w, store, "corpus", rng) in corpus
        assert _cdf_pick(front, front_w, store, "front", rng) in front
    assert store["corpus"][0] is corpus_w
    assert store["front"][0] is front_w
