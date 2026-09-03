"""Regression tests for ``core.mutations.generic._swap_tuple``.

Covers the first implementation of the C(n,m) generalization of
``_swap_pair`` proposed (analysis + empirical validation only, not
implemented there) in the combinatorics handover
(``docs/handover/handover_combinatorics_permutations_2026-09-02.md``,
§10a.1, "Recommended form, revised"): ``_swap_tuple(domain, rng, m)``.

The one finding that has to hold here is the parity trap: an m-cycle is
an *even* permutation whenever m is odd (any odd-length cycle is even),
so a rotation-only generator over odd m can never reach a transposition
and is confined to A_n. §10a.1 found this recurs at every odd m, not
just m=3. ``_swap_tuple`` therefore must rotate for even m and fall back
to an explicit non-identity permutation for odd m.
"""

import random

import pytest

from fuzzer_tool.core.exhaustive_pool import ExhaustivePool
from fuzzer_tool.core.mutations.generic import _swap_tuple
from fuzzer_tool.core.rand_pool import RandPool


@pytest.fixture(params=[random, RandPool(seed=1234)], ids=["stdlib_random", "rand_pool"])
def rng(request):
    return request.param


def _permutation_sign(mapping: dict[int, int]) -> int:
    """Sign of the permutation described by ``mapping`` (identity elsewhere)."""
    visited = set()
    sign = 1
    for start in mapping:
        if start in visited:
            continue
        cycle_len = 0
        node = start
        while node not in visited:
            visited.add(node)
            node = mapping.get(node, node)
            cycle_len += 1
        if cycle_len > 0:
            sign *= (-1) ** (cycle_len - 1)
    return sign


# ── Basic shape: plain length domain ─────────────────────────────────────


def test_plain_length_returns_m_distinct_indices_in_range(rng):
    for _ in range(200):
        result = _swap_tuple(10, rng, 4)
        assert result is not None
        picked, permuted = result
        assert len(picked) == len(permuted) == 4
        assert len(set(picked)) == 4
        assert set(picked) == set(permuted)
        assert all(0 <= v < 10 for v in picked)


def test_permuted_is_never_the_identity_arrangement(rng):
    for m in (2, 3, 4, 5, 6, 7):
        for _ in range(100):
            picked, permuted = _swap_tuple(10, rng, m)
            assert permuted != picked


# ── Degenerate cases ──────────────────────────────────────────────────────


def test_m_less_than_two_returns_none(rng):
    assert _swap_tuple(10, rng, 1) is None
    assert _swap_tuple(10, rng, 0) is None
    assert _swap_tuple(10, rng, -1) is None


def test_domain_smaller_than_m_returns_none(rng):
    assert _swap_tuple(3, rng, 4) is None
    assert _swap_tuple(0, rng, 2) is None


def test_offset_start_narrows_domain(rng):
    for _ in range(100):
        picked, permuted = _swap_tuple(5, rng, 3, start=1)
        assert all(1 <= v < 5 for v in picked)
    assert _swap_tuple(3, rng, 3, start=1) is None  # only {1, 2} eligible


# ── Candidate-sequence domain (mirrors _swap_pair's third shape) ─────────


def test_candidate_sequence_only_returns_candidates(rng):
    candidates = [1, 3, 4, 7, 9]
    for _ in range(100):
        picked, permuted = _swap_tuple(candidates, rng, 3)
        assert all(v in candidates for v in picked)
        assert set(picked) == set(permuted)


# ── m=2 matches _swap_pair's swap shape ──────────────────────────────────


def test_m_equals_two_is_a_plain_swap(rng):
    for _ in range(100):
        picked, permuted = _swap_tuple(10, rng, 2)
        assert permuted == (picked[1], picked[0])


# ── The parity trap: even m must stay reachable outside A_n ─────────────


def test_even_m_rotation_is_an_odd_permutation(rng):
    # An even-length cycle is an odd permutation -- the safe case the
    # handover's §10a.1 recommends rotation for.
    for m in (2, 4, 6, 8):
        picked, permuted = _swap_tuple(10, rng, m)
        mapping = dict(zip(picked, permuted))
        assert _permutation_sign(mapping) == -1


def test_odd_m_fallback_is_not_confined_to_rotation_parity(rng):
    # An odd-length rotation would be an even permutation (the parity
    # trap). _swap_tuple must not just silently produce rotations for
    # odd m -- across enough draws it must surface odd permutations too,
    # i.e. it is not confined to A_n the way a rotation-only generator
    # would be.
    signs = set()
    for _ in range(300):
        picked, permuted = _swap_tuple(10, rng, 3)
        mapping = dict(zip(picked, permuted))
        signs.add(_permutation_sign(mapping))
    assert signs == {1, -1}


def test_odd_m_result_is_never_a_pure_rotation_of_picked(rng):
    # More direct check that odd m does not just apply the same rotation
    # shape used for even m (which is what the parity trap warns
    # against): the m-cycle rotation should not be the *only* shape
    # produced.
    for m in (3, 5, 7):
        rotations_only = True
        for _ in range(200):
            picked, permuted = _swap_tuple(10, rng, m)
            rotation = picked[1:] + picked[:1]
            if permuted != rotation:
                rotations_only = False
                break
        assert not rotations_only, f"m={m} fallback degenerated into pure rotation"


# ── Non-degenerate distribution sanity ────────────────────────────────────


def test_plain_length_tuple_selection_is_not_always_the_same(rng):
    seen = {_swap_tuple(20, rng, 3) for _ in range(100)}
    assert len(seen) > 1


# ── ExhaustivePool compatibility ─────────────────────────────────────────


def test_even_m_domain_works_under_exhaustive_pool():
    pool = ExhaustivePool()
    for _ in pool.runs():
        result = _swap_tuple(5, pool, 4)
        assert result is not None
        picked, permuted = result
        assert len(set(picked)) == 4
        assert set(picked) == set(permuted)
        assert permuted != picked


def test_odd_m_domain_works_under_exhaustive_pool():
    pool = ExhaustivePool()
    for _ in pool.runs():
        result = _swap_tuple(5, pool, 3)
        assert result is not None
        picked, permuted = result
        assert len(set(picked)) == 3
        assert set(picked) == set(permuted)
        assert permuted != picked


def test_candidate_sequence_domain_works_under_exhaustive_pool():
    pool = ExhaustivePool()
    for _ in pool.runs():
        result = _swap_tuple([1, 3, 4, 7], pool, 3)
        assert result is not None
        picked, permuted = result
        assert all(v in (1, 3, 4, 7) for v in picked)
        assert set(picked) == set(permuted)
