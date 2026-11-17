"""Max-weight b-matching by auction, ``core/assignment.py``."""

from __future__ import annotations

import itertools
import math

from fuzzer_tool.core.assignment import QUANT, UNMATCHED, auction, auction_warm
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


# -- warm start (soksolv incremental_match.go) ---------------------------------


def _bound(weights, quotas):
    k = len(weights)
    return (max(k, sum(q for q in quotas if q > 0)) + k) / QUANT


def _stale(rng, quotas, spread):
    """Random seat prices per target (and the spare group): stale or garbage."""
    keys = [*range(len(quotas)), UNMATCHED]
    return {t: [rng.randint(0, spread) for _ in range(rng.randint(0, 4))] for t in keys}


def test_warm_control_cold_matches_itself():
    """Control (HR46): cold vs a second cold run passes the bound used below."""
    rng = RandPool(seed=21)
    for _ in range(40):
        weights, quotas = _instance(rng, rng.randint(1, 5), rng.randint(1, 3))
        a, b = auction(weights, quotas), auction(weights, quotas)

        assert _total(weights, b) >= _total(weights, a) - _bound(weights, quotas)
        assert _total(weights, a) >= _brute(weights, quotas) - _bound(weights, quotas)


def test_warm_is_eps_optimal_from_stale_prices():
    """Falsification: any start prices still end within (m + k) / QUANT of cold and of optimum."""
    rng = RandPool(seed=23)
    for _ in range(80):
        k, n = rng.randint(1, 5), rng.randint(1, 3)
        weights, quotas = _instance(rng, k, n)
        prices = _stale(rng, quotas, 4 * QUANT)

        match, _ = auction_warm(weights, quotas, prices)

        assert _fits(match, quotas)
        best = _brute(weights, quotas)
        assert _total(weights, match) >= best - _bound(weights, quotas)
        cold = _total(weights, auction(weights, quotas))
        assert _total(weights, match) >= cold - _bound(weights, quotas)


def test_warm_chain_over_perturbed_epochs():
    """Prices carried across epochs of small weight changes stay eps-optimal."""
    rng = RandPool(seed=29)
    weights, quotas = _instance(rng, 5, 3)
    prices = None
    for _ in range(30):
        weights = [[min(1.0, max(0.0, w + (rng.random() - 0.5) * 0.1)) for w in r] for r in weights]
        match, prices = auction_warm(weights, quotas, prices)

        assert _fits(match, quotas)
        assert _total(weights, match) >= _brute(weights, quotas) - _bound(weights, quotas)


def test_warm_returns_normalized_prices_for_live_items_only():
    """Vanished items are dropped; prices are shifted so the cheapest seat is 0."""
    _, prices = auction_warm([[0.9, 0.1], [0.8, 0.2]], [2, 0], {0: [5, 7], 1: [3], 9: [1]})

    assert set(prices) == {0}
    assert len(prices[0]) == 2
    assert min(min(p) for p in prices.values()) == 0


def test_warm_without_prices_equals_cold():
    rng = RandPool(seed=31)
    for _ in range(20):
        weights, quotas = _instance(rng, rng.randint(1, 6), rng.randint(1, 4))

        assert auction_warm(weights, quotas)[0] == auction(weights, quotas)


def test_warm_adversarial_empty_and_rows_removed():
    assert auction_warm([], [1, 2], {0: [3]}) == ([], {})
    weights = [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]
    _, prices = auction_warm(weights, [2, 1])
    match, _ = auction_warm(weights[:1], [2, 1], prices)  # two rows gone

    assert match == [0]


def test_warm_adversarial_duplicate_weights_terminate():
    """All-equal weights from wildly stale prices: still ends, quotas held."""
    quotas = [3, 2, 5]
    match, _ = auction_warm([[0.5] * 3 for _ in range(10)], quotas, {0: [10**6], 2: [0, 0]})

    assert UNMATCHED not in match
    assert _fits(match, quotas)
