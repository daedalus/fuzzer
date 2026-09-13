"""Regression tests for the Cook-Mertz GF(2^q) field layer."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.cook_mertz import CookMertzField

# ---------------------------------------------------------------------------
# Field construction
# ---------------------------------------------------------------------------


def test_field_construction_default():
    F = CookMertzField.default(8)
    assert F.q == 8
    assert F.order == 256
    assert F.m == 255


def test_field_construction_bad_modulus():
    with pytest.raises(ValueError):
        CookMertzField(8, modulus=0b11)  # degree 1, not 8


# ---------------------------------------------------------------------------
# omega powers
# ---------------------------------------------------------------------------


def test_omega_powers_cycle():
    F = CookMertzField.default(8)
    p = F.powers()
    assert len(p) == F.m + 1
    assert p[F.m] == 1
    assert p[0] == 1


def test_omega_enumerates_field():
    F = CookMertzField.default(8)
    p = F.powers()
    non_zero = set(range(1, F.order))
    assert set(p[i] for i in range(1, F.m + 1)) == non_zero


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def test_interpolate_empty_raises():
    F = CookMertzField.default(4)
    with pytest.raises(ValueError):
        F.interpolate({})


def test_interpolate_constant():
    F = CookMertzField.default(4)
    p = F.powers()
    # Use a constant already reduced in GF(2^4) with modulus 0x13.
    c = 0x6  # reduced form of 0xAB
    evals = {p[1]: c, p[2]: c, p[3]: c}
    interp = F.interpolate(evals)
    for j in range(1, F.m + 1):
        assert interp[p[j]] == c


def test_interpolate_linear():
    # f(x) = x should round-trip through interpolation
    F = CookMertzField.default(4)
    p = F.powers()
    evals = {p[i]: p[i] for i in range(1, F.m + 1)}
    interp = F.interpolate(evals)
    for j in range(1, F.m + 1):
        assert interp[p[j]] == p[j]


def test_interpolate_invalid_point_raises():
    F = CookMertzField.default(4)
    with pytest.raises(ValueError):
        F.interpolate({0xFFFF: 1})


# ---------------------------------------------------------------------------
# MLE evaluation
# ---------------------------------------------------------------------------


def test_evaluate_mle_constant():
    F = CookMertzField.default(4)
    p = F.powers()
    # All-zero coefficients -> always 0
    assert F.evaluate_mle([0] * F.order, p[1]) == 0
    assert F.evaluate_mle([0] * F.order, p[7]) == 0


def test_evaluate_mle_single_monomial():
    F = CookMertzField.default(4)
    p = F.powers()
    # f(x) = x_0 (coefficient at index 1 = 0b0001)
    coeffs = [0] * F.order
    coeffs[1] = 1
    for j in range(1, F.m + 1):
        # x_0 evaluated at omega^j is omega^j itself (since e_idx=1 means bit 0 set)
        assert F.evaluate_mle(coeffs, p[j]) == p[j]


# ---------------------------------------------------------------------------
# The actual efficiency claim: evaluate_mle's polynomial has degree <= q in
# the field variable (coefficient of monomial e_idx contributes
# point**popcount(e_idx), and popcount ranges over [0, q]). A degree-<=q
# polynomial is uniquely determined by q+1 points -- that sparse
# reconstruction, not "give interpolate() every point", is what would let a
# real Cook-Mertz tree-evaluation caller avoid storing all 2**q-1 values.
# These tests exercise that claim directly, with both a positive and a
# forced negative case (rather than only checking the trivial all-points
# round-trip, which is exact regardless of the underlying function).
# ---------------------------------------------------------------------------


def test_interpolate_recovers_degree_q_polynomial_from_q_plus_1_points():
    q = 4
    F = CookMertzField.default(q)
    p = F.powers()
    # Force exact degree q: the top monomial (all q bits set) has
    # popcount == q, so giving it a nonzero coefficient guarantees the
    # polynomial's degree is exactly q, not accidentally lower.
    coeffs = [0] * F.order
    coeffs[F.order - 1] = 0x7
    coeffs[3] = 0x5
    coeffs[0] = 0x2
    truth = {p[j]: F.evaluate_mle(coeffs, p[j]) for j in range(1, F.m + 1)}

    subset = {p[i]: truth[p[i]] for i in range(1, q + 2)}  # q+1 points
    recovered = F.interpolate(subset)
    assert recovered == truth


def test_interpolate_with_one_fewer_than_q_plus_1_points_diverges():
    """Negative control: an under-determined degree-q polynomial must NOT
    reconstruct correctly from only q points, or the positive test above
    would be vacuous (interpolate() being exact regardless of subset size,
    rather than because q+1 points are sufficient and necessary)."""
    q = 4
    F = CookMertzField.default(q)
    p = F.powers()
    coeffs = [0] * F.order
    coeffs[F.order - 1] = 0x7  # forces exact degree q, as above
    coeffs[3] = 0x5
    coeffs[0] = 0x2
    truth = {p[j]: F.evaluate_mle(coeffs, p[j]) for j in range(1, F.m + 1)}

    subset = {p[i]: truth[p[i]] for i in range(1, q + 1)}  # only q points
    recovered = F.interpolate(subset)
    assert any(recovered[p[j]] != truth[p[j]] for j in range(1, F.m + 1))
