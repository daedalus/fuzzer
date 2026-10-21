"""Tests for core/lattice.py: LLL reduction and Babai rounding (P4-1)."""

import random
from fractions import Fraction

import pytest

from fuzzer_tool.core.lattice import babai_round, lll_integral, lll_reduce, solve_rows


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


def _gso(rows):
    """Exact Gram-Schmidt (mu, |b*|^2) over Fractions: oracle independent of the module."""
    star, mu, norms = [], {}, []
    for i, row in enumerate(rows):
        v = [Fraction(x) for x in row]
        for j in range(i):
            mu[i, j] = sum(a * b for a, b in zip(row, star[j], strict=True)) / norms[j]
            v = [a - mu[i, j] * b for a, b in zip(v, star[j], strict=True)]
        star.append(v)
        norms.append(sum(a * a for a in v))
    return mu, norms


def _is_reduced(rows, delta=Fraction(3, 4)):
    mu, norms = _gso(rows)
    size = all(abs(m) <= Fraction(1, 2) for m in mu.values())
    lovasz = all(
        norms[i] >= (delta - mu[i, i - 1] ** 2) * norms[i - 1] for i in range(1, len(rows))
    )
    return size and lovasz


def _gram_det(rows):
    return _det([[sum(a * b for a, b in zip(r, q, strict=True)) for q in rows] for r in rows])


def _random_basis(rng, n, dim, bound):
    while True:
        rows = [[rng.randint(-bound, bound) for _ in range(dim)] for _ in range(n)]
        if _gram_det(rows) != 0:
            return rows


class TestLLLIntegral:
    def test_textbook_example(self):
        """Wikipedia's LLL example."""
        assert lll_integral([[1, 1, 1], [-1, 0, 2], [3, 5, 6]]) == [
            [0, 1, 0],
            [1, 0, 1],
            [-1, 0, 2],
        ]

    @pytest.mark.parametrize("seed", range(10))
    def test_result_is_exactly_reduced(self, seed):
        rng = random.Random(seed)
        basis = _random_basis(rng, rng.randint(2, 7), 8, 10 ** rng.randint(1, 6))
        assert _is_reduced(lll_integral(basis))

    @pytest.mark.parametrize("seed", range(10))
    def test_preserves_the_lattice(self, seed):
        """Same Gram determinant: the output basis spans the same lattice."""
        rng = random.Random(50 + seed)
        basis = _random_basis(rng, rng.randint(2, 6), 7, 10 ** rng.randint(1, 5))
        assert _gram_det(lll_integral(basis)) == _gram_det(basis)

    def test_planted_short_vector_is_first(self):
        """Knapsack embedding with a hidden short relation (the null_lattice shape)."""
        scale, weights = 1 << 40, [3, -5, 2]
        basis = [
            [1 if i == j else 0 for j in range(3)] + [scale * w] for i, w in enumerate(weights)
        ]
        basis += [[0, 0, 0, scale * 7]]
        assert lll_integral(basis)[0][3] == 0

    def test_matches_float_lll_on_the_lattice(self):
        basis = [[1, 0, 0, 12345], [0, 1, 0, 23456], [0, 0, 1, 34567], [0, 0, 0, 99991]]
        assert _gram_det(lll_integral(basis)) == _gram_det(lll_reduce(basis))

    def test_other_delta_is_reduced_for_that_delta(self):
        rng = random.Random(3)
        basis = _random_basis(rng, 5, 6, 10**4)
        out = lll_integral(basis, Fraction(99, 100))
        assert _is_reduced(out, Fraction(99, 100))

    # falsification

    def test_dependent_rows_rejected(self):
        with pytest.raises(ValueError):
            lll_integral([[1, 2, 3], [2, 4, 6]])

    def test_zero_row_rejected(self):
        with pytest.raises(ValueError):
            lll_integral([[0, 0]])
        with pytest.raises(ValueError):
            lll_integral([[1, 0], [0, 0]])

    @pytest.mark.parametrize("delta", [Fraction(1, 4), Fraction(1), Fraction(0), Fraction(2)])
    def test_delta_out_of_range_rejected(self, delta):
        with pytest.raises(ValueError):
            lll_integral([[1, 0], [0, 1]], delta)

    def test_unreduced_basis_is_not_reported_reduced(self):
        """The oracle can fail: the input of the textbook example is not reduced."""
        assert not _is_reduced([[1, 1, 1], [-1, 0, 2], [3, 5, 6]])

    # adversarial

    def test_empty_and_single(self):
        assert lll_integral([]) == []
        assert lll_integral([[3, 4]]) == [[3, 4]]

    def test_ragged_rejected(self):
        with pytest.raises(ValueError):
            lll_integral([[1, 2, 3], [4, 5]])

    def test_entries_far_beyond_float_range_stay_exact(self):
        big = 1 << 400
        basis = [[1, 0, big], [0, 1, 2 * big + 1], [0, 0, big * big]]
        out = lll_integral(basis)
        assert _is_reduced(out)
        assert _gram_det(out) == _gram_det(basis)

    def test_long_vectors_next_to_short_ones(self):
        out = lll_integral([[10**30, 1], [10**30 + 1, 2]])
        assert out[0] in ([1, 1], [-1, -1])
        assert _is_reduced(out)

    def test_does_not_mutate_input(self):
        basis = [[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
        keep = [list(r) for r in basis]
        lll_integral(basis)
        assert basis == keep

    # control: the reference against itself (Hard Rule 46)

    @pytest.mark.parametrize("seed", range(5))
    def test_control_deterministic_and_idempotent(self, seed):
        rng = random.Random(70 + seed)
        basis = _random_basis(rng, 5, 6, 10**5)
        once = lll_integral(basis)
        assert once == lll_integral(basis)
        assert lll_integral(once) == once
