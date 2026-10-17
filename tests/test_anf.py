"""Algebraic normal form over GF(2) (core/anf.py): Möbius transform, low-degree fit.

Reference ANF coefficients are computed by the defining subset sum
``a_S = XOR_{T ⊆ S} f(T)``, independent of the butterfly under test.
"""

import itertools
from math import comb

import numpy as np
import pytest

from fuzzer_tool.core import anf

N = 6


def _table(fn, n):
    return np.array([fn(x) for x in range(1 << n)], dtype=np.uint64)


def _reference_anf(fn, n):
    """O(3^n) subset sum straight from the definition."""
    out = []
    for s in range(1 << n):
        a, t = 0, s
        while True:
            a ^= fn(t)
            if t == 0:
                break
            t = (t - 1) & s
        out.append(a)
    return out


def _quad16(x):
    """16-bit output, every bit of degree <= 2 in the 16 input bits."""
    r = (x << 3 | x >> 13) & 0xFFFF
    return (x & r) ^ (x >> 5) ^ 0xA5C3


def _cubic16(x):
    return x & (x >> 1) & (x >> 2) ^ 0x1234


# ── transform ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x & 1 & (x >> 3),
        lambda x: bin(x).count("1") & 1,
        lambda x: int(x == 0b101101),
        lambda x: (x * 37 + 11) & 0xFF,
    ],
    ids=["monomial", "parity", "equality_gate", "packed_bytes"],
)
def test_moebius_matches_subset_sum(fn):
    got = anf.moebius(_table(fn, N)).tolist()
    want = _reference_anf(fn, N)
    # Control: the reference agrees with a second run of itself.
    assert want == _reference_anf(fn, N)
    assert got == want


def test_moebius_is_an_involution():
    table = _table(lambda x: (x * 0x9E37) & 0xFFFF, N)
    assert np.array_equal(anf.moebius(anf.moebius(table)), table)


def test_moebius_rejects_non_power_of_two_length():
    with pytest.raises(ValueError):
        anf.moebius(np.zeros(12, dtype=np.uint64))


@pytest.mark.parametrize(
    ("fn", "deg"),
    [
        (lambda x: 0, 0),
        (lambda x: 1, 0),
        (lambda x: (x ^ (x >> 2)) & 1, 1),
        (lambda x: x & (x >> 1) & 1, 2),
        (lambda x: int(x == 0b110011), N),
    ],
    ids=["zero", "one", "affine", "quadratic", "equality_gate"],
)
def test_degree_is_exact(fn, deg):
    """An equality gate is degree n exactly; random probing could not tell."""
    assert anf.degree(_table(fn, N)) == deg


# ── layer ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("n", "d"), [(1, 2), (8, 0), (8, 2), (16, 2), (10, 3)])
def test_layer_holds_every_point_of_low_weight_once(n, d):
    pts = anf.layer(n, d)
    assert len(pts) == sum(comb(n, j) for j in range(min(d, n) + 1))
    assert len(set(pts)) == len(pts)
    assert all(bin(x).count("1") <= d and x < 1 << n for x in pts)
    # Lowest weight first: the scan finds the smallest flip.
    weights = [bin(x).count("1") for x in pts]
    assert weights == sorted(weights)


# ── low-degree fit ───────────────────────────────────────────────────


def test_fit_low_predicts_the_whole_quadratic_map():
    n = 16
    calls = []

    def f(x):
        calls.append(x)
        return _quad16(x)

    table = anf.fit_low(f, n, 2, confirms=[0xFFFF, 0x5555, 0x3333])

    assert table is not None
    assert len(set(calls)) == 1 + n + comb(n, 2) + 3
    full = _table(_quad16, n)
    assert np.array_equal(table, full)


def test_fit_low_rejects_a_cubic_map():
    """Falsification: a degree-3 map does not survive its confirm points."""
    assert anf.fit_low(_cubic16, 16, 2, confirms=[0xFFFF, 0x7777]) is None


def test_fit_low_blind_point_fails_closed():
    assert anf.fit_low(lambda x: None if x == 3 else 0, 8, 2, confirms=[]) is None


def test_fit_low_refuses_fields_beyond_the_memory_bound():
    """Adversarial: 2^n tables must stay bounded (Hard Rule 54)."""
    with pytest.raises(ValueError):
        anf.fit_low(lambda x: 0, anf.MAX_BITS + 1, 2, confirms=[])


@pytest.mark.parametrize("seed", range(4))
def test_nonconstant_low_degree_flips_inside_its_layer(seed):
    """Minimal monomial S: f(1_S) = f(0) ^ 1, and |S| <= deg f."""
    terms = list(itertools.combinations(range(8), 2))
    picks = terms[seed * 5 : seed * 5 + 3]

    def f(x):
        v = 0
        for i, j in picks:
            v ^= (x >> i) & (x >> j) & 1
        return v

    hit = [x for x in anf.layer(8, 2) if f(x) != f(0)]
    assert hit
