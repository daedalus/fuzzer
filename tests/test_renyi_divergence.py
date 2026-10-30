"""renyi_divergence: limits, closed forms, monotonicity, adversarial input.

D_a(P||Q) = log2(sum p^a q^(1-a)) / (a - 1), in bits. Inputs are counts or
probabilities (normalized internally); negative / NaN / zero-mass raise.
"""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.renyi import renyi_divergence

P = [0.5, 0.3, 0.2]
Q = [0.2, 0.5, 0.3]
EPS = 1e-6
ALPHAS = [0.0, 0.25, 0.5, 0.9, 1.0 - EPS, 1.0, 1.0 + EPS, 1.5, 2.0, 5.0, math.inf]


def _kl_bits(p: list[float], q: list[float]) -> float:
    """Independent oracle: plain KL(P||Q) in bits, no shared code."""
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0) / math.log(2)


def _random_dist(rng: RandPool, size: int) -> list[float]:
    raw = [0.01 + v for v in rng.random_list(size)]
    total = sum(raw)
    return [v / total for v in raw]


def test_control_oracle_against_itself():
    """Rule 46: the oracle must read 0 on identical inputs before it judges."""
    assert _kl_bits(P, P) == 0.0
    assert _kl_bits(P, Q) == _kl_bits(P, Q)


@pytest.mark.parametrize("alpha", [1.0 - EPS, 1.0, 1.0 + EPS])
def test_alpha_one_matches_kl(alpha):
    assert math.isclose(renyi_divergence(P, Q, alpha), _kl_bits(P, Q), rel_tol=1e-5)


def test_half_is_bhattacharyya():
    """p=(1/2,1/2), q=(9/10,1/10): BC = (3+1)/sqrt(20) = 2/sqrt(5),
    D_0.5 = -2 log2 BC = log2(5/4)."""
    got = renyi_divergence([0.5, 0.5], [0.9, 0.1], 0.5)
    assert math.isclose(got, math.log2(5 / 4))


def test_alpha_zero_and_inf_closed_forms():
    p = [0.5, 0.5, 0.0]
    q = [0.25, 0.25, 0.5]

    # -log2 Q(supp P) = -log2(1/2); log2 max p/q = log2 2.
    assert math.isclose(renyi_divergence(p, q, 0.0), 1.0)
    assert math.isclose(renyi_divergence(p, q, math.inf), 1.0)


def test_monotone_non_decreasing_in_alpha():
    rng = RandPool(seed=42)
    for size in (2, 5, 17):
        p = _random_dist(rng, size)
        q = _random_dist(rng, size)
        values = [renyi_divergence(p, q, a) for a in ALPHAS]
        for lo, hi in zip(values, values[1:], strict=False):
            assert hi >= lo - 1e-9, (size, values)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_self_divergence_is_zero(alpha):
    assert renyi_divergence(P, P, alpha) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0, math.inf])
def test_adversarial_disjoint_support_is_inf(alpha):
    assert renyi_divergence({1: 1.0}, {2: 1.0}, alpha) == math.inf


def test_adversarial_q_misses_p_support():
    """supp P not in supp Q: inf for a >= 1, finite below 1."""
    p, q = [0.5, 0.5], [1.0, 0.0]

    assert renyi_divergence(p, q, 1.0) == math.inf
    assert renyi_divergence(p, q, 2.0) == math.inf
    assert renyi_divergence(p, q, math.inf) == math.inf
    # Overlap sum = sqrt(1/2) * sqrt(1) -> log2(2^-0.5) / (0.5 - 1) = 1.
    assert math.isclose(renyi_divergence(p, q, 0.5), 1.0)


def test_adversarial_zeros_in_p_are_finite():
    p, q = [1.0, 0.0], [0.5, 0.5]

    for alpha in ALPHAS:
        assert math.isclose(renyi_divergence(p, q, alpha), 1.0)


def test_adversarial_counts_are_normalized():
    counts_p, counts_q = [50, 30, 20], [2, 5, 3]

    for alpha in ALPHAS:
        assert math.isclose(
            renyi_divergence(counts_p, counts_q, alpha), renyi_divergence(P, Q, alpha)
        )


def test_adversarial_mapping_keys_align():
    p = {"a": 0.5, "b": 0.3, "c": 0.2}
    q = {"c": 0.3, "a": 0.2, "b": 0.5}

    assert math.isclose(renyi_divergence(p, q, 2.0), renyi_divergence(P, Q, 2.0))


@pytest.mark.parametrize(
    ("p", "q", "alpha"),
    [
        ([0.5, -0.5], [0.5, 0.5], 1.0),
        ([0.0, 0.0], [0.5, 0.5], 1.0),
        ([], [1.0], 1.0),
        ([float("nan"), 1.0], [0.5, 0.5], 1.0),
        ([0.5, 0.5], [0.5, 0.5], -1.0),
        ([0.5, 0.5], [0.5, 0.5], float("nan")),
    ],
)
def test_adversarial_invalid_raises(p, q, alpha):
    with pytest.raises(ValueError):
        renyi_divergence(p, q, alpha)
