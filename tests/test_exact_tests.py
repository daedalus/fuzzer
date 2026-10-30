"""``core/exact_tests.py``: exact paired tests checked against brute-force enumeration.

Each oracle enumerates the null distribution outright (every sign vector, every
row assignment) with ``Fraction`` arithmetic, so it shares no formula with the
code under test. Rule 46: every oracle is first run against itself on an input
it must treat identically (shuffled order, flipped labels).
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest

from fuzzer_tool.core.exact_tests import (
    _NUMPY_RANK_MIN,
    EXACT_MAX_N,
    fisher_exact,
    mcnemar_exact,
    wilcoxon_signed_rank,
)
from fuzzer_tool.core.rand_pool import RandPool

# ── oracles ───────────────────────────────────────────────────────────


def _brute_mcnemar(b: int, c: int) -> float:
    """2 * P(X <= min(b, c)), X counted over every 0/1 vector of length b + c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    low = sum(1 for v in itertools.product((0, 1), repeat=n) if sum(v) <= k)
    return float(min(Fraction(1), Fraction(2 * low, 2**n)))


def _brute_fisher(a: int, b: int, c: int, d: int) -> float:
    """Sum of P(table) <= P(observed), counting every labelled row-1 draw."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    hits = set(range(a + c))  # labelled column-1 items
    counts: dict[int, int] = {}
    for row in itertools.combinations(range(n), a + b):
        x = len(hits.intersection(row))
        counts[x] = counts.get(x, 0) + 1
    total = sum(counts.values())
    obs = counts[a]
    tail = sum(m for m in counts.values() if m <= obs)
    return float(min(Fraction(1), Fraction(tail, total)))


def _naive_ranks(mags: list[float]) -> list[Fraction]:
    """1-based mid-ranks by definition: 1 + #smaller + (#equal - 1) / 2."""
    return [1 + sum(m < x for m in mags) + Fraction(sum(m == x for m in mags) - 1, 2) for x in mags]


def _brute_wilcoxon(deltas: list[float]) -> tuple[float, float]:
    """Smaller rank sum and its two-sided p over all 2^n sign vectors."""
    nz = [d for d in deltas if d != 0]
    if not nz:
        return 0.0, 1.0
    ranks = _naive_ranks([abs(d) for d in nz])
    w_pos = sum((r for r, d in zip(ranks, nz, strict=True) if d > 0), Fraction(0))
    total = sum(ranks, Fraction(0))
    stat = min(w_pos, total - w_pos)

    n = len(nz)
    le = ge = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = sum((r for r, s in zip(ranks, signs, strict=True) if s), Fraction(0))
        le += w <= stat
        ge += w >= total - stat
    return float(stat), float(min(Fraction(1), Fraction(2 * min(le, ge), 2**n)))


def _legacy_ranks(ranked: list[tuple[float, int]]) -> list[float]:
    """Verbatim pre-move ``_tied_ranks``."""
    ranks = [0.0] * len(ranked)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _legacy_exact_p(ranks: list[float], stat: float, total: float) -> float:
    """Verbatim pre-move ``_exact_signed_p``: all 2^n sign vectors."""
    n = len(ranks)
    le = ge = 0
    for sv in itertools.product((-1, 1), repeat=n):
        w = sum(r for r, s in zip(ranks, sv, strict=False) if s > 0)
        le += w <= stat + 1e-12
        ge += w >= total - stat - 1e-12
    return min(1.0, 2.0 * min(le, ge) / (2**n))


def _legacy_wilcoxon(deltas: list[float]) -> tuple[float, float]:
    """Verbatim pre-move ``tools/lib/bench_paired.py`` code (bit-identity oracle)."""
    ranked = sorted((abs(d), i) for i, d in enumerate(deltas) if d != 0.0)
    if not ranked:
        return 0.0, 1.0
    ranks = _legacy_ranks(ranked)
    signs = [1.0 if deltas[idx] > 0 else -1.0 for _, idx in ranked]
    w_pos = sum(r for r, s in zip(ranks, signs, strict=False) if s > 0)
    w_neg = sum(r for r, s in zip(ranks, signs, strict=False) if s < 0)
    stat = min(w_pos, w_neg)
    n = len(ranked)
    if n <= 20:
        return stat, _legacy_exact_p(ranks, stat, sum(ranks))
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (abs(w_pos - mean) - 0.5) / math.sqrt(var)
    if z < 0:
        return stat, 1.0
    return stat, min(1.0, math.erfc(z / math.sqrt(2.0)))


def _deltas(pool: RandPool, n: int) -> list[float]:
    """Small-magnitude integer deltas: ties and zeros are common."""
    return [float(v) for v in pool.randint_list(-3, 3, n)]


# ── McNemar ───────────────────────────────────────────────────────────


class TestMcnemar:
    def test_oracle_control_is_label_symmetric(self):
        for b, c in itertools.product(range(6), repeat=2):
            assert _brute_mcnemar(b, c) == _brute_mcnemar(c, b)

    def test_matches_enumeration(self):
        for b, c in itertools.product(range(8), repeat=2):
            assert mcnemar_exact(b, c) == pytest.approx(_brute_mcnemar(b, c), rel=1e-12)

    def test_no_discordant_pairs(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_adversarial_negative_count_rejected(self):
        with pytest.raises(ValueError):
            mcnemar_exact(-1, 3)

    def test_adversarial_large_counts_stay_finite(self):
        # 2^n overflows float at n > 1023; the exact integer tail must not.
        p = mcnemar_exact(0, 1100)
        assert 0.0 <= p < 1e-300


# ── Fisher ────────────────────────────────────────────────────────────


class TestFisher:
    def test_oracle_control_is_transpose_symmetric(self):
        for a, b, c, d in [(3, 1, 1, 3), (2, 0, 1, 4), (0, 3, 2, 1)]:
            assert _brute_fisher(a, b, c, d) == _brute_fisher(a, c, b, d)

    def test_matches_enumeration(self):
        for a, b, c, d in itertools.product(range(4), repeat=4):
            assert fisher_exact(a, b, c, d) == pytest.approx(_brute_fisher(a, b, c, d), rel=1e-9), (
                a,
                b,
                c,
                d,
            )

    def test_empty_table(self):
        assert fisher_exact(0, 0, 0, 0) == 1.0

    def test_adversarial_negative_cell_rejected(self):
        with pytest.raises(ValueError):
            fisher_exact(2, -1, 0, 3)


# ── Wilcoxon ──────────────────────────────────────────────────────────


class TestWilcoxon:
    def test_oracle_control_is_order_and_sign_invariant(self):
        pool = RandPool(5)
        for n in range(1, 9):
            d = _deltas(pool, n)
            assert _brute_wilcoxon(d) == _brute_wilcoxon(d[::-1])
            assert _brute_wilcoxon(d) == _brute_wilcoxon([-x for x in d])

    def test_matches_enumeration_with_ties_and_zeros(self):
        pool = RandPool(7)
        for n in range(1, 11):
            for _ in range(3):
                d = _deltas(pool, n)
                stat, p = wilcoxon_signed_rank(d)
                want_stat, want_p = _brute_wilcoxon(d)
                assert stat == want_stat
                assert p == pytest.approx(want_p, rel=1e-12), d

    def test_bit_identical_to_legacy_tool(self):
        pool = RandPool(9)
        sizes = list(range(0, 15)) + [EXACT_MAX_N + 1, 60, _NUMPY_RANK_MIN, 3 * _NUMPY_RANK_MIN]
        for n in sizes:
            d = [float(v) for v in pool.randint_list(-8, 8, n)]
            assert wilcoxon_signed_rank(d) == _legacy_wilcoxon(d)

    def test_empty_and_all_zero(self):
        assert wilcoxon_signed_rank([]) == (0.0, 1.0)
        assert wilcoxon_signed_rank([0.0, 0.0, -0.0]) == (0.0, 1.0)

    def test_all_ties_one_sign(self):
        n = 6
        # Every rank is (n + 1) / 2; only the all-+ / all-- vectors are extreme.
        assert wilcoxon_signed_rank([2.0] * n) == (0.0, 2 / 2**n)

    def test_numpy_ranking_with_few_nonzero_takes_exact_path(self):
        # Long input, short nonzero tail: numpy ranks feed the exact null.
        tail = [3.0, -1.0, 2.0, 2.0, 5.0, -4.0, 1.0, 6.0, 2.0, -2.0]
        d = [0.0] * _NUMPY_RANK_MIN + tail
        assert wilcoxon_signed_rank(d) == wilcoxon_signed_rank(tail) == _legacy_wilcoxon(tail)

    @pytest.mark.parametrize("pad", [0, _NUMPY_RANK_MIN])
    def test_adversarial_nan_rejected(self, pad):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([0.0] * pad + [1.0, float("nan"), 2.0])

    def test_adversarial_exact_limit_is_fast(self):
        # n = EXACT_MAX_N used to enumerate 2^20 sign vectors in Python.
        d = [float(i % 7 - 3) or 1.0 for i in range(EXACT_MAX_N)]
        assert wilcoxon_signed_rank(d) == _legacy_wilcoxon(d)
