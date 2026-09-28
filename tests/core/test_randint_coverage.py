"""RandPool.randint / randint_list: range, endpoint coverage, uniformity.

Coverage is a coupon-collector property, so it is probabilistic: no finite
number of draws T guarantees every value in [a, b] appears. The draw budget
is therefore derived from a target failure probability delta,

    T = ceil(n * (ln n + ln(1/delta))),   n = b - a + 1,

which gives P(some value missed) <= n * (1 - 1/n)**T <= delta (union bound).
A fixed "(b - a)**2 draws" bound is NOT used: for b = a + 1 it allows 1 draw
to see 2 values, and for n = 3 it passes only ~44% of the time.

Seeds are fixed, so every run is deterministic; delta only bounds how likely
a *fixed* seed is to be an unlucky one when the test is written or re-seeded.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from fuzzer_tool.core.rand_pool import RandPool

DELTA = 1e-9


def coverage_budget(n: int, delta: float = DELTA) -> int:
    """Draws needed so P(any of n values unseen) <= delta."""
    return math.ceil(n * (math.log(n) + math.log(1.0 / delta)))


def miss_probability_bound(n: int, draws: int) -> float:
    """Union bound on P(some value unseen after `draws` uniform draws)."""
    return min(1.0, n * (1.0 - 1.0 / n) ** draws)


RANGES = [
    (0, 1),  # n = 2: the case where (b-a)**2 draws is impossible
    (0, 2),  # n = 3
    (5, 9),
    (-3, 3),  # negative lower bound
    (-10, -1),  # entirely negative
    (0, 9),
    (0, 254),  # n = 255, just under the fast path
    (0, 255),  # n = 256: the pre-computed %256 fast path
    (10, 265),  # n = 256 with a != 0 (offset applied to fast path)
    (0, 256),  # n = 257, just over the fast path
    (-1000, 1000),
]


@pytest.mark.parametrize("a,b", RANGES)
def test_budget_meets_target_failure_probability(a, b):
    """The budget formula itself must achieve the claimed delta."""
    n = b - a + 1
    assert miss_probability_bound(n, coverage_budget(n)) <= DELTA


@pytest.mark.parametrize("a,b", RANGES)
def test_randint_covers_full_range_within_budget(a, b):
    """Every value in [a, b], both endpoints included, none outside."""
    n = b - a + 1
    pool = RandPool(seed=12345)
    seen = {pool.randint(a, b) for _ in range(coverage_budget(n))}

    assert seen == set(range(a, b + 1))


@pytest.mark.parametrize("a,b", RANGES)
def test_randint_list_covers_full_range_within_budget(a, b):
    """Vectorized path has the same contract (spans pool refills if needed)."""
    n = b - a + 1
    pool = RandPool(seed=54321)
    vals = pool.randint_list(a, b, coverage_budget(n))

    assert len(vals) == coverage_budget(n)
    assert set(vals) == set(range(a, b + 1))


@pytest.mark.parametrize("a,b", RANGES)
def test_randint_never_leaves_bounds(a, b):
    """Endpoints inclusive: min >= a and max <= b for scalar and list paths."""
    pool = RandPool(seed=7)
    scalar = [pool.randint(a, b) for _ in range(5000)]
    vector = pool.randint_list(a, b, 5000)

    assert a <= min(scalar) and max(scalar) <= b
    assert a <= min(vector) and max(vector) <= b


@pytest.mark.parametrize("seed", range(20))
def test_two_value_range_is_reachable_across_seeds(seed):
    """n = 2 must show both values well within budget for every seed.

    P(one fixed seed fails) <= 2 * 2**-T; T = coverage_budget(2) = 30.
    """
    pool = RandPool(seed=seed)
    seen = {pool.randint(0, 1) for _ in range(coverage_budget(2))}

    assert seen == {0, 1}


def test_square_bound_is_not_a_valid_guarantee():
    """Documents why the (b-a)**2 sanity check was rejected.

    Analytic, not sampled: P(all n values in (n-1)**2 draws), by inclusion-
    exclusion. n = 2 is 0 (1 draw, 2 values); n = 3 is 4/9 (36 surjections
    of 81 sequences).
    """

    def p_full(n: int, t: int) -> float:
        return sum((-1) ** k * math.comb(n, k) * ((n - k) / n) ** t for k in range(n + 1))

    assert p_full(2, 1) == 0.0
    assert p_full(3, 4) == pytest.approx(36 / 81)
    # ... but it does become very likely as n grows.
    assert p_full(10, 81) > 0.99


@pytest.mark.parametrize("a,b,draws_per_value", [(0, 9, 4000), (3, 20, 2000), (0, 255, 500)])
def test_randint_is_uniform_chi_square(a, b, draws_per_value):
    """Coverage alone passes cyclic / biased generators; check the counts.

    Pearson chi-square vs uniform, df = n - 1. Threshold df + 6*sqrt(2*df)
    (~6 sigma under the normal approximation) keeps false alarms negligible
    while still catching modulo-style skew or an off-by-one endpoint.
    """
    n = b - a + 1
    total = n * draws_per_value
    pool = RandPool(seed=2024)
    counts = Counter(pool.randint(a, b) for _ in range(total))

    expected = total / n
    chi2 = sum((counts.get(v, 0) - expected) ** 2 / expected for v in range(a, b + 1))
    df = n - 1

    assert chi2 < df + 6 * math.sqrt(2 * df)


def test_chi_square_detects_a_biased_sampler():
    """Sanity-check the uniformity check: a sampler that draws the top value
    half as often as the rest has full coverage (so it passes the budget
    test) but must fail the chi-square threshold. A deterministic a..b cycle
    is perfectly uniform and is not caught by either test; that limitation
    is inherent to counting-based checks."""
    a, b, total = 0, 9, 40000
    n = b - a + 1

    # Biased: value b is drawn half as often as the others.
    weights = [2] * (n - 1) + [1]
    seq = [v for v, w in enumerate(weights) for _ in range(w)]
    counts = Counter(seq[i % len(seq)] for i in range(total))
    expected = total / n
    chi2 = sum((counts[v] - expected) ** 2 / expected for v in range(a, b + 1))
    df = n - 1

    assert chi2 > df + 6 * math.sqrt(2 * df)


def test_empty_range_returns_lower_bound():
    """Documented degenerate behaviour: b < a yields a (no exception)."""
    pool = RandPool(seed=1)

    assert pool.randint(5, 4) == 5
    assert pool.randint_list(5, 4, 10) == []
