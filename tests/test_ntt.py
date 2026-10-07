"""Unit tests for the NTT primitive in core/ntt.py."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.ntt import DEFAULT_MOD, DEFAULT_ROOT, ntt, poly_mul_ntt


def test_poly_mul_example():
    # (1 + 2x + 3x²) * (4 + 5x) = 4 + 13x + 22x² + 15x³
    assert poly_mul_ntt([1, 2, 3], [4, 5]) == [4, 13, 22, 15]


def test_poly_mul_ones():
    assert poly_mul_ntt([1, 1], [1, 1, 1]) == [1, 2, 2, 1]


def test_ntt_roundtrip():
    a = [3, 1, 4, 1, 5, 9, 2, 6]
    original = a[:]
    ntt(a, invert=False)
    ntt(a, invert=True)
    assert a == [x % DEFAULT_MOD for x in original]


def test_ntt_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        ntt([1, 2, 3])


def test_poly_mul_zero():
    assert poly_mul_ntt([], [1, 2]) == [0]
    assert poly_mul_ntt([1, 2], []) == [0]


def test_poly_mul_constant():
    assert poly_mul_ntt([7], [11]) == [77 % DEFAULT_MOD]


def test_default_params_are_ntt_friendly():
    # p-1 must be divisible by a large power of two; root must be primitive.
    assert (DEFAULT_MOD - 1) % (1 << 23) == 0
    # g^{(p-1)/q} ≠ 1 for every prime factor q of p-1 (spot-check 2 and 7, 17).
    for q in (2, 7, 17):
        assert pow(DEFAULT_ROOT, (DEFAULT_MOD - 1) // q, DEFAULT_MOD) != 1
