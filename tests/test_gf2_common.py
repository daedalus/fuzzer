"""Regression tests for the shared GF(2) utilities in gf2_common: the
polynomial/field layer, and the bitmask-vector layer merged in from the
former ``gf2_linalg.py`` (see that module's docstring, now folded into
``gf2_common.py``, for provenance)."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.gf2_common import (
    GF2n,
    _prime_factors,
    apply_bitmask_map,
    bitmask_from_indices,
    compose_bitmask_maps,
    find_irreducible,
    find_primitive_root,
    indices_from_bitmask,
    invert_bitmask_map,
    is_irreducible,
    poly_deg,
    poly_divmod,
    poly_gcd,
    poly_mod,
    poly_mul,
    poly_powmod,
    verified_apply_inverse,
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


# ---------------------------------------------------------------------------
# Bitmask-vector layer (formerly tests/test_gf2_linalg.py)
# ---------------------------------------------------------------------------


def _random_invertible(n_bits: int, rng: random.Random) -> list[int]:
    """Build a random invertible n_bits x n_bits GF(2) map by composing
    random row swaps and row-XOR-additions onto the identity -- both are
    elementary operations that preserve invertibility, so the result is
    guaranteed full rank without needing a rejection loop."""
    rows = [1 << i for i in range(n_bits)]
    for _ in range(n_bits * 4):
        a, b = rng.randrange(n_bits), rng.randrange(n_bits)
        if a != b:
            rows[a] ^= rows[b]
    return rows


class TestIndexBitmaskConversion:
    def test_roundtrip(self):
        for indices in [(), (0,), (1, 3, 7), tuple(range(20))]:
            assert indices_from_bitmask(bitmask_from_indices(indices)) == tuple(sorted(indices))

    def test_bitmask_from_indices_value(self):
        assert bitmask_from_indices([0, 2, 3]) == 0b1101

    def test_indices_from_bitmask_value(self):
        assert indices_from_bitmask(0b1101) == (0, 2, 3)

    def test_empty(self):
        assert bitmask_from_indices([]) == 0
        assert indices_from_bitmask(0) == ()


class TestApplyBitmaskMap:
    def test_identity(self):
        n = 8
        identity = [1 << i for i in range(n)]
        for v in (0, 1, 0xFF, 0b10110):
            assert apply_bitmask_map(identity, v) == v

    def test_known_map(self):
        # output_0 = in0 ^ in1, output_1 = in1
        masks = [0b011, 0b010]
        assert apply_bitmask_map(masks, 0b00) == 0b00
        assert apply_bitmask_map(masks, 0b01) == 0b01
        assert apply_bitmask_map(masks, 0b10) == 0b11
        assert apply_bitmask_map(masks, 0b11) == 0b10

    def test_zero_map_always_zero(self):
        masks = [0, 0, 0]
        for v in range(8):
            assert apply_bitmask_map(masks, v) == 0

    def test_linearity(self):
        """f(a) ^ f(b) == f(a ^ b) must hold for any linear map."""
        rng = random.Random(1)
        n = 6
        masks = _random_invertible(n, rng)
        for _ in range(20):
            a, b = rng.randrange(1 << n), rng.randrange(1 << n)
            assert apply_bitmask_map(masks, a) ^ apply_bitmask_map(masks, b) == apply_bitmask_map(
                masks, a ^ b
            )


class TestInvertBitmaskMap:
    def test_identity_self_inverse(self):
        n = 5
        identity = [1 << i for i in range(n)]
        assert invert_bitmask_map(identity, n) == identity

    def test_singular_returns_none(self):
        # row 1 == row 0 -> rank-deficient, not invertible
        masks = [0b001, 0b001, 0b100]
        assert invert_bitmask_map(masks, 3) is None

    def test_all_zero_row_singular(self):
        masks = [0b001, 0b000, 0b100]
        assert invert_bitmask_map(masks, 3) is None

    def test_round_trip_random(self):
        rng = random.Random(42)
        for n in (1, 2, 4, 8, 16, 33):
            masks = _random_invertible(n, rng)
            inv = invert_bitmask_map(masks, n)
            assert inv is not None
            for _ in range(10):
                v = rng.randrange(1 << n)
                assert apply_bitmask_map(inv, apply_bitmask_map(masks, v)) == v
                assert apply_bitmask_map(masks, apply_bitmask_map(inv, v)) == v

    def test_double_inverse_is_original(self):
        rng = random.Random(7)
        n = 10
        masks = _random_invertible(n, rng)
        inv = invert_bitmask_map(masks, n)
        inv2 = invert_bitmask_map(inv, n)
        assert inv2 == masks

    def test_wrong_row_count_raises(self):
        with pytest.raises(ValueError):
            invert_bitmask_map([0b01, 0b10], 3)

    def test_out_of_range_bits_raises(self):
        with pytest.raises(ValueError):
            invert_bitmask_map([0b100, 0b010], 2)


class TestVerifiedApplyInverse:
    def test_accepts_true_inverse(self):
        rng = random.Random(3)
        n = 12
        masks = _random_invertible(n, rng)
        inv = invert_bitmask_map(masks, n)
        for _ in range(10):
            v = rng.randrange(1 << n)
            fwd = apply_bitmask_map(masks, v)
            assert verified_apply_inverse(masks, inv, fwd) == v

    def test_rejects_placeholder_row_corruption(self):
        """A structurally full-rank 'inverse' derived from a map with a
        placeholder (wrong) row is still full rank, so
        invert_bitmask_map succeeds -- but verified_apply_inverse must
        catch the semantic mismatch that a naive apply would miss."""
        rng = random.Random(9)
        n = 6
        true_fwd = _random_invertible(n, rng)
        corrupted_fwd = list(true_fwd)
        corrupted_fwd[2] ^= 1  # simulate a wrong/placeholder row
        inv_of_corrupted = invert_bitmask_map(corrupted_fwd, n)
        assert inv_of_corrupted is not None  # still structurally invertible

        # Round-trip against the corrupted map itself always succeeds by
        # construction -- the guard only matters when checked against the
        # *true* forward map, which is exactly what callers care about.
        mismatches = 0
        for v in range(1 << n):
            fwd_true = apply_bitmask_map(true_fwd, v)
            result = verified_apply_inverse(true_fwd, inv_of_corrupted, fwd_true)
            if result is None:
                mismatches += 1
            else:
                assert result == v
        assert mismatches > 0, "corruption should be observable for at least one input"


class TestComposeBitmaskMaps:
    def test_compose_matches_sequential_apply(self):
        rng = random.Random(5)
        n = 8
        inner = _random_invertible(n, rng)
        outer = _random_invertible(n, rng)
        composed = compose_bitmask_maps(inner, outer)
        for _ in range(15):
            v = rng.randrange(1 << n)
            assert apply_bitmask_map(composed, v) == apply_bitmask_map(
                outer, apply_bitmask_map(inner, v)
            )

    def test_compose_with_identity_is_noop(self):
        n = 6
        rng = random.Random(11)
        m = _random_invertible(n, rng)
        identity = [1 << i for i in range(n)]
        assert compose_bitmask_maps(identity, m) == m
        assert compose_bitmask_maps(m, identity) == m

    def test_compose_inverse_is_identity(self):
        rng = random.Random(13)
        n = 7
        m = _random_invertible(n, rng)
        inv = invert_bitmask_map(m, n)
        composed = compose_bitmask_maps(m, inv)
        identity = [1 << i for i in range(n)]
        assert composed == identity

    def test_out_of_range_reference_raises(self):
        inner = [0b01, 0b10]  # only 2 rows
        outer = [0b100]  # references bit 2, which doesn't exist in inner
        with pytest.raises(ValueError):
            compose_bitmask_maps(inner, outer)
