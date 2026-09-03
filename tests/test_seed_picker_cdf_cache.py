"""The cached-CDF pick must draw exactly what random.choices drew.

``_cdf_pick`` replaces ``random.choices(pop, weights=w, k=1)[0]`` with a
cached prefix sum plus a bisect. That is only worth doing if it is a
substitution and not a re-derivation: it has to consume the same number of
RNG values, scale them the same way and land on the same index, or a fixed
``--seed`` stops reproducing the same campaign.

These tests therefore compare against ``random.choices`` under a shared RNG
state rather than against a distribution. Anything the fast path declines
(length mismatch, zero total, non-finite total) must still raise what
``random.choices`` raised.
"""

import itertools
import math
import random

import pytest

from fuzzer_tool.services.seed_picker import _cdf_pick


def _pairs(population, weights, n=500, seed=1234):
    """Draw n picks each way from the same starting RNG state."""
    store = {}
    random.seed(seed)
    fast = [_cdf_pick(population, weights, store, "s") for _ in range(n)]
    random.seed(seed)
    ref = [random.choices(population, weights=weights, k=1)[0] for _ in range(n)]
    return fast, ref


def test_draws_are_identical_to_random_choices():
    rng = random.Random(0)
    population = [f"seed{i}".encode() for i in range(200)]
    weights = [rng.random() * 10 + 0.01 for _ in range(200)]
    fast, ref = _pairs(population, weights)
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
    random.seed(99)
    for _ in range(10):
        _cdf_pick(population, weights, store, "s")
    after_fast = random.random()

    random.seed(99)
    for _ in range(10):
        random.choices(population, weights=weights, k=1)
    after_ref = random.random()

    assert after_fast == after_ref


def test_cache_is_keyed_on_the_weight_list_identity():
    population = list(range(10))
    weights = [1.0] * 10
    store = {}
    _cdf_pick(population, weights, store, "s")
    first = store["s"]
    _cdf_pick(population, weights, store, "s")
    assert store["s"] is first, "same list must reuse the cached prefix sum"

    replacement = [2.0] * 10
    _cdf_pick(population, replacement, store, "s")
    assert store["s"] is not first
    assert store["s"][0] is replacement


def test_new_weights_take_effect_immediately():
    """A stale CDF would keep sampling the old distribution."""
    population = ["a", "b"]
    store = {}
    random.seed(5)
    assert {_cdf_pick(population, [1.0, 0.0], store, "s") for _ in range(50)} == {"a"}
    assert {_cdf_pick(population, [0.0, 1.0], store, "s") for _ in range(50)} == {"b"}


def test_cached_prefix_sum_matches_accumulate_exactly():
    rng = random.Random(3)
    weights = [rng.random() for _ in range(64)]
    store = {}
    _cdf_pick(list(range(64)), weights, store, "s")
    assert store["s"][1] == list(itertools.accumulate(weights))


def test_length_mismatch_still_raises():
    with pytest.raises(ValueError):
        _cdf_pick([1, 2, 3], [1.0, 1.0], {}, "s")


def test_zero_total_still_raises():
    with pytest.raises(ValueError):
        _cdf_pick([1, 2, 3], [0.0, 0.0, 0.0], {}, "s")


def test_non_finite_total_still_raises():
    with pytest.raises(ValueError):
        _cdf_pick([1, 2, 3], [math.inf, 1.0, 1.0], {}, "s")


def test_empty_population_defers():
    with pytest.raises(IndexError):
        _cdf_pick([], [], {}, "s")


def test_slots_do_not_collide():
    """The corpus vector and the Pareto-front slice share one store."""
    store = {}
    random.seed(7)
    corpus = list(range(30))
    corpus_w = [1.0] * 30
    front = [100, 200]
    front_w = [1.0, 3.0]
    for _ in range(20):
        assert _cdf_pick(corpus, corpus_w, store, "corpus") in corpus
        assert _cdf_pick(front, front_w, store, "front") in front
    assert store["corpus"][0] is corpus_w
    assert store["front"][0] is front_w
