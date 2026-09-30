"""Max-weight b-matching by auction, ``core/assignment.py``."""

from __future__ import annotations

import itertools
import math

from fuzzer_tool.core.assignment import QUANT, UNMATCHED, auction
from fuzzer_tool.core.rand_pool import RandPool


def _total(weights, match):
    return sum(weights[p][t] for p, t in enumerate(match) if t != UNMATCHED)


def _brute(weights, quotas):
    """Best total over every quota-respecting assignment; independent of the auction."""
    n = len(quotas)
    best = 0.0
    for match in itertools.product([*range(n), UNMATCHED], repeat=len(weights)):
        if all(match.count(t) <= quotas[t] for t in range(n)):
            best = max(best, _total(weights, match))
    return best


def _fits(match, quotas):
    return all(match.count(t) <= q for t, q in enumerate(quotas))


def _instance(rng, k, n):
    weights = [[rng.random() for _ in range(n)] for _ in range(k)]
    quotas = [rng.randint(0, k) for _ in range(n)]
    return weights, quotas


def test_brute_force_control_is_order_free():
    """Control (HR46): the oracle must agree with itself on a reordered instance."""
    rng = RandPool(seed=7)
    for _ in range(20):
        weights, quotas = _instance(rng, 4, 3)

        assert math.isclose(_brute(weights, quotas), _brute(weights[::-1], quotas))


def test_matches_brute_force():
    """Falsification: auction total within the documented (m + k) / QUANT of the optimum."""
    rng = RandPool(seed=11)
    for _ in range(60):
        k, n = rng.randint(1, 5), rng.randint(1, 3)
        weights, quotas = _instance(rng, k, n)
        m = max(k, sum(quotas))

        match = auction(weights, quotas)

        assert _fits(match, quotas)
        assert _total(weights, match) >= _brute(weights, quotas) - (m + k) / QUANT


def test_beats_seed_greedy():
    """Both seeds like t0 best; the optimum sends A to its runner-up.

    Greedy (A first): A->t0 0.9 + B->t1 0.1 = 1.0.  Optimum: A->t1 0.8 + B->t0 0.85 = 1.65.
    """
    weights = [[0.9, 0.8], [0.85, 0.1]]

    assert auction(weights, [1, 1]) == [1, 0]


def test_short_seats_leave_the_cheapest_unmatched():
    """Adversarial: 3 seeds, 1 seat -> only the best-paying seed is seated."""
    assert auction([[0.2], [0.9], [0.5]], [1]) == [UNMATCHED, 0, UNMATCHED]


def test_ties_terminate_and_fill_quotas():
    """Adversarial: all-equal weights (the bidding-war case) still ends, quotas held."""
    quotas = [3, 2, 5]
    match = auction([[0.5] * 3 for _ in range(10)], quotas)

    assert UNMATCHED not in match
    assert _fits(match, quotas)


def test_zero_quota_target_is_never_used():
    match = auction([[1.0, 0.0], [1.0, 0.0]], [0, 2])

    assert match == [1, 1]


def test_non_finite_weights_are_worth_nothing():
    """Adversarial: NaN / inf never win a seat over a finite positive weight."""
    assert auction([[math.nan, 0.3], [math.inf, 0.2]], [1, 1]) in ([1, 0], [0, 1])
    assert auction([[math.inf, 0.3]], [1, 1]) == [1]


def test_degenerate_shapes():
    assert auction([], [1, 2]) == []
    assert auction([[], []], []) == [UNMATCHED, UNMATCHED]
    assert auction([[0.4]], [1]) == [0]
