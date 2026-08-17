"""Regression + falsification tests for ``core/gf2_linalg.py``."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.gf2_linalg import (
    apply_bitmask_map,
    bitmask_from_indices,
    compose_bitmask_maps,
    indices_from_bitmask,
    invert_bitmask_map,
    verified_apply_inverse,
)


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
