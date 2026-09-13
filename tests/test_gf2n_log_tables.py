"""GF2n.mul/pow/inv now route through Zech (log/antilog) tables built from
the same omega-cycle enumeration ``omega_powers()`` already computes,
instead of doing a carryless multiply + polynomial long division (or
repeated-squaring exponentiation) on every call.

These tests keep the polynomial-path methods (``_mul_poly``/``_pow_poly``,
retained on the class for exactly this purpose) as the reference
implementation, and check the table path reproduces them exactly -- not
just on a handful of samples, but exhaustively over the whole field for
the two operations (``mul``, ``inv``) that are cheap enough to check that
way at ``q=8``.
"""

from __future__ import annotations

import random

from fuzzer_tool.core.gf2_common import GF2n


def test_tables_are_built_for_default_field_size():
    F = GF2n(8, seed=0)
    assert F._log is not None
    assert F._exp is not None
    assert len(F._exp) == F.m
    assert len(F._log) == F.order


def test_mul_matches_polynomial_path_exhaustively():
    F = GF2n(8, seed=0)
    for a in range(F.order):
        for b in range(F.order):
            assert F.mul(a, b) == F._mul_poly(a, b), (a, b)


def test_inv_matches_polynomial_path_exhaustively():
    F = GF2n(8, seed=0)
    for a in range(1, F.order):
        assert F.inv(a) == F._pow_poly(a, F.m - 1), a


def test_inv_satisfies_field_axiom_for_every_nonzero_element():
    """a * inv(a) == 1 for every a != 0 -- a correctness property of the
    field itself, not just agreement with the old implementation."""
    F = GF2n(8, seed=0)
    for a in range(1, F.order):
        assert F.mul(a, F.inv(a)) == 1, a


def test_pow_matches_polynomial_path_randomized():
    F = GF2n(8, seed=0)
    rng = random.Random(1)
    for _ in range(2000):
        a = rng.randint(0, F.order - 1)
        e = rng.randint(0, 500)
        assert F.pow(a, e) == F._pow_poly(a, e), (a, e)


def test_inv_zero_still_raises():
    F = GF2n(8, seed=0)
    try:
        F.inv(0)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("expected ZeroDivisionError")


def test_large_q_skips_table_construction_gracefully():
    """Above _MAX_TABLE_ORDER the field must still work correctly via the
    polynomial path, just without the table speedup."""
    F = GF2n(21, seed=0)  # order = 2**21 > _MAX_TABLE_ORDER (2**20)
    assert F._log is None
    assert F._exp is None
    # Still correct: mul/inv fall back to the polynomial path transparently.
    a, b = F.gen, F.mul(F.gen, F.gen)
    assert F.mul(a, F.inv(a)) == 1
    assert F.mul(a, b) == F._mul_poly(a, b)
