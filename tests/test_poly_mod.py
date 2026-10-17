"""Polynomial compares mod 2^k (core/poly_mod.py): Newton fit + Hensel roots.

Each fake target is an integer model of the operand cmplog logs, truncated to
the operand width; expected values come from the model or brute force, never
from the solver.
"""

import pytest

from fuzzer_tool.core import poly_mod
from fuzzer_tool.core.poly_mod import PolyMod, fit, roots, solve

M32 = 0xFFFFFFFF


def _article(a: int) -> int:
    """f(a) from the Leviathan article: a degree-4 polynomial over Z."""
    a -= 53
    for i in range(1, 17):
        if not i % 7:
            a *= i * a
        elif i == 5:
            a += 20
        else:
            a += i + 1
        a -= 5
    return a - 107


def _observer(fn, bits):
    mask = (1 << bits) - 1
    calls = []

    def observe(x):
        calls.append(x)
        return fn(x) & mask

    return observe, calls


def _brute_roots(fn, target, width_bits, bits):
    """Roots below 2^min(width, bits): wider fields repeat them every 2^bits."""
    mask = (1 << bits) - 1
    return sorted(x for x in range(1 << min(width_bits, bits)) if fn(x) & mask == target)


# ── fit ──────────────────────────────────────────────────────────────


def test_article_function_is_fitted_and_both_roots_found():
    observe, _ = _observer(_article, 32)
    model = fit(observe, 1000, 32, 32)

    assert model is not None
    assert len(model.coeffs) - 1 == 4
    target = _article(81) & M32
    assert target == 1597172565
    assert {3, 81} <= set(roots(model, target, 32))


def test_model_predicts_unseen_points():
    def fn(x):
        return 5 * x**3 + 3 * x * x + 7

    observe, _ = _observer(fn, 32)
    model = fit(observe, 77, 32, 32)

    assert model is not None
    for x in (0, 1, 0xDEADBEEF, M32, 12345678):
        assert model(x) == fn(x) & M32


def test_fit_cost_is_samples_plus_confirms():
    observe, calls = _observer(lambda x: x * x, 32)
    assert fit(observe, 10, 32, 32) is not None
    assert len(calls) <= poly_mod.SAMPLES + poly_mod.CONFIRMS


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: x ^ 0x5A5A5A5A,
        lambda x: (x * 0x85EBCA6B & M32) ^ (x >> 13),
        lambda x: x**12 + x,
    ],
    ids=["xor", "hash_round", "degree_12"],
)
def test_fit_rejects_non_polynomial_or_too_high_degree(fn):
    """Falsification: maps outside the family fail closed."""
    observe, _ = _observer(fn, 32)
    assert fit(observe, 999, 32, 32) is None


def test_fit_rejects_a_flat_map():
    observe, _ = _observer(lambda x: 42, 32)
    assert fit(observe, 0, 32, 32) is None


def test_blind_observe_fits_nothing():
    assert fit(lambda x: None, 0, 32, 32) is None


def test_confirm_reaching_nothing_fails_closed():
    """A compare unreached at a confirm spot is not evidence for the fit."""

    def observe(x):
        return None if x > 1 << 20 else (x * x) & M32

    assert fit(observe, 0, 32, 32) is None


# ── roots / solve ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fn", "width_bits", "bits"),
    [
        (lambda x: x * x + 3 * x + 7, 12, 12),
        (lambda x: 4 * x**3 + 2 * x, 12, 12),
        (lambda x: x**3 + 1, 8, 16),
        (lambda x: 3 * x * x + x, 12, 8),
    ],
    ids=["quadratic", "even_coeffs", "field_narrower", "field_wider"],
)
def test_roots_match_brute_force(fn, width_bits, bits):
    observe, _ = _observer(fn, bits)
    model = fit(observe, 5, width_bits, bits)
    assert model is not None

    mask = (1 << bits) - 1
    for x in (7, 100, 255):
        target = fn(x) & mask
        brute = _brute_roots(fn, target, width_bits, bits)
        # Control: the oracle agrees with itself on a second run.
        assert brute == _brute_roots(fn, target, width_bits, bits)
        assert sorted(roots(model, target, width_bits)) == brute


def test_unreachable_residue_has_no_root():
    """Squares mod 8 are 0, 1, 4: x*x == 3 (mod 2^k) has no solution."""
    model = PolyMod(0, (0, 1, 2), 32)  # Newton form of x*x
    assert roots(model, 3, 32) == []


def test_target_wider_than_operand_has_no_root():
    model = PolyMod(0, (0, 1, 2), 16)
    assert roots(model, 1 << 16, 16) == []


def test_regression_thinning_keeps_odd_roots():
    """Adversarial: x*x << 30 keeps 2^30 roots per level; in-order truncation
    kept only even x and lost every root at bit 30. Every returned root is real."""

    def fn(x):
        return (x * x) << 30

    observe, _ = _observer(fn, 32)
    model = fit(observe, 3, 32, 32)
    assert model is not None

    found = roots(model, 1 << 30, 32)
    assert 0 < len(found) <= poly_mod.MAX_FRONTIER
    assert all(fn(x) & M32 == 1 << 30 for x in found)


def test_solve_returns_valid_solutions_nearest_first():
    def fn(x):
        return x * x + 3 * x + 7

    observe, _ = _observer(fn, 32)
    model = fit(observe, 0, 32, 32)
    target = fn(0x1234) & M32

    sols = solve(model, target, 32, near=0x1200)
    assert sols
    assert len(sols) <= poly_mod.MAX_SOLUTIONS
    assert all(fn(x) & M32 == target for x in sols)
    assert sols[0] == 0x1234


def test_solve_wide_field_lifts_roots_into_range():
    """16-bit operand of a 32-bit field: every member of a root class works."""

    def fn(x):
        return x * x + 1

    observe, _ = _observer(fn, 16)
    model = fit(observe, 9, 32, 16)
    target = fn(0x3_0011) & 0xFFFF

    sols = solve(model, target, 32, near=0x3_0000)
    assert sols
    assert all(0 <= x <= M32 and fn(x) & 0xFFFF == target for x in sols)
    assert min(abs(x - 0x3_0000) for x in sols) < 1 << 16
