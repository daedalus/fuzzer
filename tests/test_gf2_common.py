"""Regression tests for the shared GF(2) polynomial utilities in gf2_common."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.gf2_common import (
    GF2n,
    _prime_factors,
    find_irreducible,
    find_primitive_root,
    is_irreducible,
    poly_deg,
    poly_divmod,
    poly_gcd,
    poly_mod,
    poly_mul,
    poly_powmod,
)

# ---------------------------------------------------------------------------
# poly_deg
# ---------------------------------------------------------------------------


def test_poly_deg_zero():
    assert poly_deg(0) == -1


def test_poly_deg_monomials():
    assert poly_deg(1) == 0
    assert poly_deg(2) == 1
    assert poly_deg(1 << 17) == 17


def test_poly_deg_sparse():
    assert poly_deg(0b10101) == 4


# ---------------------------------------------------------------------------
# poly_mul
# ---------------------------------------------------------------------------


def test_poly_mul_zero():
    assert poly_mul(0, 0b111) == 0
    assert poly_mul(0b101, 0) == 0


def test_poly_mul_identity():
    x = 0b10  # x
    assert poly_mul(x, 1) == x
    assert poly_mul(1, x) == x


def test_poly_mul_commutative():
    a = 0b1101
    b = 0b1011
    assert poly_mul(a, b) == poly_mul(b, a)


def test_poly_mul_known():
    # (x + 1)(x + 1) = x^2 + 1 over GF(2)
    assert poly_mul(0b11, 0b11) == 0b101


# ---------------------------------------------------------------------------
# poly_divmod / poly_mod
# ---------------------------------------------------------------------------


def test_poly_divmod_identity():
    a = 0b11011  # x^4 + x^3 + x + 1
    b = 0b1011  # x^3 + x + 1
    q, r = poly_divmod(a, b)
    assert poly_deg(r) < poly_deg(b)
    # a = q*b + r  (mod 2)
    assert poly_mod(poly_mul(q, b), a | b) == r or a == poly_mod(poly_mul(q, b) ^ r, a | b)


def test_poly_mod_remainder():
    a = 0b11101
    b = 0b1011
    r = poly_mod(a, b)
    assert poly_deg(r) < poly_deg(b)


# ---------------------------------------------------------------------------
# poly_gcd
# ---------------------------------------------------------------------------


def test_poly_gcd_zero():
    assert poly_gcd(0, 0b101) == poly_gcd(0b101, 0)


def test_poly_gcd_identity():
    a = 0b11001
    assert poly_gcd(a, 0) == a
    assert poly_gcd(0, a) == a


def test_poly_gcd_symmetric():
    a = 0b11101
    b = 0b1011
    assert poly_gcd(a, b) == poly_gcd(b, a)


def test_poly_gcd_divides():
    a = 0b11101
    b = 0b1011
    g = poly_gcd(a, b)
    assert poly_mod(a, g) == 0 or g == 1
    assert poly_mod(b, g) == 0 or g == 1


# ---------------------------------------------------------------------------
# poly_powmod
# ---------------------------------------------------------------------------


def test_poly_powmod_zero_exp():
    assert poly_powmod(0b10, 0, 0b111) == 1


def test_poly_powmod_identity():
    mod = 0b111  # x^2 + x + 1 (irreducible)
    assert poly_powmod(0b10, 1, mod) == 0b10
    assert poly_powmod(0b10, 0, mod) == 1


# ---------------------------------------------------------------------------
# _prime_factors
# ---------------------------------------------------------------------------


def test_prime_factors_small():
    assert _prime_factors(1) == set()
    assert _prime_factors(2) == {2}
    assert _prime_factors(12) == {2, 3}
    assert _prime_factors(17) == {17}


def test_prime_factors_prime_power():
    assert _prime_factors(27) == {3}


# ---------------------------------------------------------------------------
# is_irreducible
# ---------------------------------------------------------------------------


def test_is_irreducible_degree_1():
    # x + 1 is the only degree-1 irreducible over GF(2)
    assert is_irreducible(0b11, 1)


def test_is_irreducible_degree_2():
    # x^2 + x + 1 is irreducible
    assert is_irreducible(0b111, 2)


def test_is_irreducible_degree_3():
    # x^3 + x + 1 is irreducible
    assert is_irreducible(0b1011, 3)


def test_is_irreducible_reducible():
    # x^2 is reducible
    assert not is_irreducible(0b100, 2)
    # (x + 1)^2 = x^2 + 1 is reducible
    assert not is_irreducible(0b101, 2)


def test_is_irreducible_degree_8():
    # find_irreducible(8) returns a valid one
    p = find_irreducible(8)
    assert is_irreducible(p, 8)
    assert p.bit_length() - 1 == 8


# ---------------------------------------------------------------------------
# find_irreducible
# ---------------------------------------------------------------------------


def test_find_irreducible_degree():
    for q in (1, 2, 3, 4, 8):
        p = find_irreducible(q)
        assert p.bit_length() - 1 == q
        assert is_irreducible(p, q)


# ---------------------------------------------------------------------------
# find_primitive_root
# ---------------------------------------------------------------------------


def test_find_primitive_root_small():
    # For order 7 (prime), every non-1 element is primitive, but we still
    # want a generator that covers all 1..6.
    def is_prim(a):
        # order 7: check a^k != 1 for k < 6
        cur = a
        for _ in range(5):
            cur = poly_mul(cur, a)
            if cur == 1:
                return False
        return True

    rng = pytest.importorskip("random").Random(0)
    a = find_primitive_root(7, is_prim, rng)
    assert 1 <= a < 7


def test_find_primitive_root_order_1():
    rng = pytest.importorskip("random").Random(0)
    assert find_primitive_root(1, lambda a: True, rng) == 1


# ---------------------------------------------------------------------------
# GF2n smoke tests
# ---------------------------------------------------------------------------


def test_gf2n_construction():
    F = GF2n(8)
    assert F.order == 256
    assert F.m == 255
    assert is_irreducible(F.mod, 8)


def test_gf2n_primitive_element_enumerates_field():
    F = GF2n(8)
    powers = F.omega_powers()
    seen = {powers[i] for i in range(1, F.m + 1)}
    assert len(seen) == F.m
    assert seen == set(range(1, F.order))


def test_gf2n_inverse():
    F = GF2n(8)
    for a in (1, 2, 3, 5, 17, 42, 127, 255):
        if a == 0:
            continue
        assert F.mul(F.inv(a), a) == 1


def test_gf2n_add_is_xor():
    F = GF2n(8)
    assert F.add(0b101, 0b110) == 0b011


def test_gf2n_pow_identity():
    F = GF2n(8)
    gen = F.gen
    assert F.pow(gen, 0) == 1
    assert F.pow(gen, 1) == gen
    assert F.pow(gen, F.m) == 1
