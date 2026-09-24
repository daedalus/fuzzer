"""Tests for core/lattice.py: LLL reduction and Babai rounding (P4-1)."""

from fractions import Fraction

import pytest

from fuzzer_tool.core.lattice import babai_round, lll_reduce, solve_rows


def _det(rows):
    """Exact determinant by fraction elimination (independent of solve_rows)."""
    m = [[Fraction(v) for v in r] for r in rows]
    n, det = len(m), Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if m[r][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            m[c], m[p] = m[p], m[c]
            det = -det
        det *= m[c][c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            m[r] = [a - f * b for a, b in zip(m[r], m[c], strict=True)]
    return det


class TestLLL:
    def test_preserves_lattice_volume(self):
        basis = [[1, 0, 0, 12345], [0, 1, 0, 23456], [0, 0, 1, 34567], [0, 0, 0, 99991]]
        red = lll_reduce(basis)
        assert abs(_det(red)) == abs(_det(basis))

    def test_finds_known_short_vector(self):
        """Falsification: rows 1 and 2 differ by (1, 1, 0); LLL must expose a vector that short."""
        basis = [[1000, 1, 0], [1001, 2, 0], [0, 0, 1000]]
        red = lll_reduce(basis)
        assert min(sum(v * v for v in r) for r in red) <= 2

    def test_degenerate_inputs(self):
        assert lll_reduce([]) == []
        assert lll_reduce([[5, 7]]) == [[5, 7]]

    def test_control_is_deterministic(self):
        basis = [[3, 1, 4], [1, 5, 9], [2, 6, 5]]
        assert lll_reduce(basis) == lll_reduce(basis)


class TestBabai:
    def test_solve_rows_exact(self):
        rows = [[2, 0], [1, 3]]
        c = solve_rows(rows, [5, 6])
        assert [sum(ci * r[i] for ci, r in zip(c, rows, strict=True)) for i in range(2)] == [5, 6]

    def test_lattice_point_is_returned_unchanged(self):
        rows = [[4, 1], [1, 3]]
        point = [2 * 4 - 1, 2 * 1 - 3]
        assert babai_round(rows, point) == point

    def test_singular_basis_raises(self):
        """Adversarial: a dependent basis has no unique coordinates."""
        with pytest.raises(ValueError):
            solve_rows([[1, 2], [2, 4]], [1, 1])
