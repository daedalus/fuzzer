"""Regression tests for ``core.mutations.generic._swap_pair``.

Covers the centralization proposed in the combinatorics handover
(``docs/handover/handover_combinatorics_permutations_2026-09-02.md``,
§10a): the C(n,2) swap-pair primitive that was previously inlined in 15
per-format mutators (``avif``, ``isobmff``, ``mpegts``, ``nal``, ``pgs``,
``protobuf``, ``webm``, ``webp``, ``x86``, ``zip``, ``asf``, ``riff``,
``adts``, ``sqlite``, ``arm``) now lives in one place. Three call shapes
existed at those sites and all three must keep working:

1. plain range over a length — ``_swap_pair(len(seq), rng)``
2. offset range over a length — ``_swap_pair(len(seq), rng, start=1)``
   (``sqlite`` keeps page 0, the file header, fixed)
3. an explicit candidate sequence instead of a length —
   ``_swap_pair(candidates, rng)`` (``arm`` swaps only over indices whose
   word kind is not ``"raw"``)
"""

import random

import pytest

from fuzzer_tool.core.exhaustive_pool import ExhaustivePool
from fuzzer_tool.core.mutations.generic import _swap_pair
from fuzzer_tool.core.rand_pool import RandPool


@pytest.fixture(params=[random, RandPool(seed=1234)], ids=["stdlib_random", "rand_pool"])
def rng(request):
    return request.param


# ── Shape 1: plain length domain ────────────────────────────────────────


def test_plain_length_returns_two_distinct_indices_in_range(rng):
    for _ in range(200):
        pair = _swap_pair(10, rng)
        assert pair is not None
        i, j = pair
        assert i != j
        assert 0 <= i < 10
        assert 0 <= j < 10


def test_plain_length_degenerate_cases_return_none(rng):
    assert _swap_pair(0, rng) is None
    assert _swap_pair(1, rng) is None


def test_plain_length_minimum_viable_domain(rng):
    # n == 2 is the smallest domain that can produce a swap.
    for _ in range(50):
        pair = _swap_pair(2, rng)
        assert pair == (0, 1) or pair == (1, 0)


# ── Shape 2: offset range domain (sqlite's start=1) ─────────────────────


def test_offset_range_never_returns_the_excluded_index(rng):
    # Mirrors sqlite._swap_pages: index 0 (page 1 / the file header) must
    # never be selected.
    for _ in range(200):
        pair = _swap_pair(5, rng, start=1)
        assert pair is not None
        i, j = pair
        assert i != 0 and j != 0
        assert i != j
        assert 1 <= i < 5
        assert 1 <= j < 5


def test_offset_range_degenerate_cases_return_none(rng):
    # len(doc.pages) < 3 is exactly the pre-existing sqlite guard
    # (len(pages) >= 3 before start=1 narrows it to >= 2 eligible pages).
    assert _swap_pair(2, rng, start=1) is None  # only page 0 excluded, 1 left
    assert _swap_pair(1, rng, start=1) is None
    assert _swap_pair(0, rng, start=1) is None


def test_offset_range_minimum_viable_domain(rng):
    for _ in range(50):
        pair = _swap_pair(3, rng, start=1)
        assert pair in ((1, 2), (2, 1))


# ── Shape 3: explicit candidate sequence (arm's filtered swapable) ──────


def test_candidate_sequence_only_returns_candidates(rng):
    candidates = [1, 3, 4, 7]
    for _ in range(200):
        pair = _swap_pair(candidates, rng)
        assert pair is not None
        i, j = pair
        assert i != j
        assert i in candidates
        assert j in candidates


def test_candidate_sequence_degenerate_cases_return_none(rng):
    assert _swap_pair([], rng) is None
    assert _swap_pair([5], rng) is None


def test_candidate_sequence_minimum_viable_domain(rng):
    for _ in range(50):
        pair = _swap_pair([2, 9], rng)
        assert pair in ((2, 9), (9, 2))


# ── Non-degenerate distribution sanity (not a rigorous uniformity test) ──


def test_plain_length_pair_selection_is_not_always_the_same(rng):
    # A regression guard against an implementation that always returns
    # the same pair regardless of the rng's draws.
    seen = {_swap_pair(20, rng) for _ in range(100)}
    assert len(seen) > 1


# ── ExhaustivePool compatibility (regression: bare range broke it) ──────
#
# ExhaustivePool.sample() (core/exhaustive_pool.py) only recognizes
# list | tuple | bytes as a sequence population -- unlike RandPool.sample,
# which deliberately special-cases range (see its docstring). A first
# version of _swap_pair passed a bare range(start, domain) straight
# through and crashed the first time it ran under exhaustive enumeration
# (TypeError: '>' not supported between instances of 'int' and 'range').
# These tests pin the fix (wrapping in list(...)) directly, independent
# of whichever operator sweep happens to exercise it.


def test_plain_length_domain_works_under_exhaustive_pool():
    pool = ExhaustivePool()
    for _ in pool.runs():
        pair = _swap_pair(5, pool)
        assert pair is not None
        i, j = pair
        assert i != j
        assert 0 <= i < 5
        assert 0 <= j < 5


def test_offset_range_domain_works_under_exhaustive_pool():
    pool = ExhaustivePool()
    for _ in pool.runs():
        pair = _swap_pair(5, pool, start=1)
        assert pair is not None
        i, j = pair
        assert i != 0 and j != 0
        assert i != j


def test_candidate_sequence_domain_works_under_exhaustive_pool():
    pool = ExhaustivePool()
    for _ in pool.runs():
        pair = _swap_pair([1, 3, 4, 7], pool)
        assert pair is not None
        i, j = pair
        assert i != j
        assert i in (1, 3, 4, 7)
        assert j in (1, 3, 4, 7)


def test_exhaustive_pool_enumerates_every_pair_over_a_full_sweep():
    # ExhaustivePool's whole purpose is falsification-by-enumeration: over
    # the full run it must produce every reachable ordered (i, j) pair
    # for a small domain, not just avoid crashing on one call.
    pool = ExhaustivePool()
    seen = {_swap_pair(4, pool) for _ in pool.runs()}
    assert pool.exhausted
    assert seen == {
        (i, j) for i in range(4) for j in range(4) if i != j
    }
