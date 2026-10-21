"""Tests for core/null_lattice.py: exact integer kernel lattice, LLL-reduced."""

from __future__ import annotations

import itertools
import random
from fractions import Fraction

import numpy as np
import pytest

from fuzzer_tool.core import int_checksum_solver as solver
from fuzzer_tool.core.null_lattice import null_relations

HUGE = 1 << 200


def _dot(row, z) -> int:
    return sum(a * b for a, b in zip(row, z, strict=True))


def _residual_zero(matrix, vectors) -> bool:
    return all(_dot(r, z) == 0 for r in matrix for z in vectors)


def _rank(matrix) -> int:
    """Exact rank by Fraction elimination: an oracle independent of the module."""
    m = [[Fraction(x) for x in r] for r in matrix]
    rank = 0
    for col in range(len(m[0])):
        piv = next((r for r in range(rank, len(m)) if m[r][col]), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for r in range(len(m)):
            if r != rank and m[r][col]:
                f = m[r][col] / m[rank][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[rank], strict=True)]
        rank += 1
    return rank


def _in_span(basis, z) -> bool:
    """True when *z* is an integer combination of *basis* (Fraction Gauss-Jordan)."""
    k, n = len(basis), len(z)
    aug = [[Fraction(basis[j][i]) for j in range(k)] + [Fraction(z[i])] for i in range(n)]
    row = 0
    pivots = []
    for col in range(k):
        piv = next((r for r in range(row, n) if aug[r][col]), None)
        if piv is None:
            return False
        aug[row], aug[piv] = aug[piv], aug[row]
        aug[row] = [x / aug[row][col] for x in aug[row]]
        for r in range(n):
            if r != row and aug[r][col]:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[row], strict=True)]
        pivots.append(row)
        row += 1
    if any(aug[r][k] for r in range(row, n)):
        return False
    return all(aug[p][k].denominator == 1 for p in pivots)


def _random_matrix(rng: random.Random, rows: int, cols: int, bound: int):
    return [[rng.randint(-bound, bound) for _ in range(cols)] for _ in range(rows)]


# ── the demo case ──────────────────────────────────────────────────────


def test_sgld_demo_relation():
    """sgld_demo.py's matrix: column1 - 2*column2 + column3 = 0."""
    assert null_relations([[1, 2, 3], [4, 5, 6]]) == [(1, -2, 1)]


def test_first_nonzero_entry_is_positive():
    for z in null_relations([[3, 1, -4, 1, -5]]):
        assert next(x for x in z if x) > 0


# ── exactness and completeness ─────────────────────────────────────────


@pytest.mark.parametrize("seed", range(12))
def test_every_vector_is_an_exact_kernel_vector(seed):
    rng = random.Random(seed)
    matrix = _random_matrix(rng, rng.randint(1, 4), rng.randint(2, 8), 40)
    assert _residual_zero(matrix, null_relations(matrix))


@pytest.mark.parametrize("seed", range(12))
def test_count_is_n_minus_exact_rank(seed):
    rng = random.Random(100 + seed)
    rows, cols = rng.randint(1, 5), rng.randint(2, 8)
    matrix = _random_matrix(rng, rows, cols, 9)
    # Force dependent rows so rank < rows in some trials.
    matrix.append([a + b for a, b in zip(matrix[0], matrix[-1], strict=True)])
    assert len(null_relations(matrix)) == cols - _rank(matrix)


def test_basis_generates_the_whole_kernel_lattice():
    """No short kernel vector is missed: all of {-3..3}^n with A z = 0 are in the span."""
    matrix = [[2, 4, -2, 6], [1, -1, 3, 0]]
    basis = null_relations(matrix)
    found = 0
    for z in itertools.product(range(-3, 4), repeat=4):
        if not any(z) or _dot(matrix[0], z) or _dot(matrix[1], z):
            continue
        found += 1
        assert _in_span(basis, z)
    assert found


def test_kernel_vectors_are_primitive():
    """The kernel lattice is saturated: no basis vector is a multiple of another vector."""
    from math import gcd

    for z in null_relations([[6, 10, 15, 0], [1, 2, 3, 4]]):
        assert gcd(*z) == 1


def test_sorted_shortest_first():
    out = null_relations([[1, 1, 1, 1, 1, 1]])
    norms = [sum(x * x for x in z) for z in out]
    assert norms == sorted(norms)


# ── falsification ──────────────────────────────────────────────────────


def test_independent_columns_have_no_relation():
    """Falsification: full column rank means nothing to find."""
    assert null_relations([[1, 0], [0, 1]]) == []
    assert null_relations([[1, 2], [3, 5], [7, 11]]) == []


def test_zero_matrix_kernel_is_everything():
    assert sorted(null_relations([[0, 0, 0]])) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]


def test_rational_but_not_integer_relation_is_found_as_primitive():
    """x1 = x2 / 2 over the rows still yields the primitive (2, -1), not (1, -0.5)."""
    assert null_relations([[2, 4], [3, 6]]) == [(2, -1)]


def test_wrong_target_is_not_a_relation():
    matrix = [[1, 2, 3], [4, 5, 6]]
    assert _dot(matrix[1], (1, -2, 2)) != 0
    assert (1, -2, 2) not in null_relations(matrix)


# ── adversarial ────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [[], [[]]])
def test_adversarial_empty_matrix_rejected(bad):
    with pytest.raises(ValueError):
        null_relations(bad)


def test_adversarial_ragged_rows_rejected():
    with pytest.raises(ValueError):
        null_relations([[1, 2, 3], [4, 5]])


@pytest.mark.parametrize("entry", [1.5, "3", None, 2.0])
def test_adversarial_non_integer_entry_rejected(entry):
    with pytest.raises(TypeError):
        null_relations([[1, entry]])


def test_adversarial_numpy_ints_accepted_and_floats_still_rejected():
    assert null_relations(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)) == [(1, -2, 1)]
    with pytest.raises(TypeError):
        null_relations(np.array([[1.0, 2.0]]))


def test_adversarial_entries_beyond_int64_stay_exact():
    matrix = [[HUGE, HUGE, 0], [0, 1, 1]]
    out = null_relations(matrix)
    assert out and _residual_zero(matrix, out)
    assert out == [(1, -1, 1)]


def test_adversarial_duplicate_and_zero_rows_and_columns():
    matrix = [[1, 2, 0, 3], [1, 2, 0, 3], [0, 0, 0, 0]]
    out = null_relations(matrix)
    assert len(out) == 4 - _rank(matrix) == 3
    assert _residual_zero(matrix, out)
    assert (0, 0, 1, 0) in out


def test_adversarial_wide_modulus_system_mirrors_the_solver_shape():
    """n = 19 columns, 10 equations, N = 2^32: the dimension the solver feeds it."""
    rng = random.Random(7)
    width, eqs, mod = 8, 10, 1 << 32
    weights = [rng.randint(-9, 9) or 1 for _ in range(width)]
    rows = []
    for _ in range(eqs + 1):
        words = [rng.randrange(1 << 16) for _ in range(width)]
        rows.append((words, sum(w * x for w, x in zip(weights, words, strict=True)) % mod))
    matrix = solver._relation_matrix(rows, mod)
    out = null_relations(matrix)
    assert _residual_zero(matrix, out)
    assert out[0][-1] == 1 and list(out[0][:width]) == weights


# ── control: the reference against itself (Hard Rule 46) ───────────────


@pytest.mark.parametrize("seed", range(6))
def test_control_deterministic(seed):
    rng = random.Random(200 + seed)
    matrix = _random_matrix(rng, 3, 7, 30)
    assert null_relations(matrix) == null_relations(matrix)


@pytest.mark.parametrize("seed", range(6))
def test_control_row_order_does_not_change_the_lattice(seed):
    """Same equations in another order: each basis spans the other."""
    rng = random.Random(300 + seed)
    matrix = _random_matrix(rng, 3, 7, 30)
    shuffled = matrix[::-1]
    a, b = null_relations(matrix), null_relations(shuffled)
    assert len(a) == len(b)
    assert all(_in_span(b, z) for z in a)
    assert all(_in_span(a, z) for z in b)


# ── wiring: the linear solver goes through it ──────────────────────────


def test_linear_weights_recovers_planted_weights():
    rng = random.Random(11)
    weights, mod = [3, -1, 7, 2], 1 << 16
    rows = []
    for _ in range(6):
        words = [rng.randrange(1 << 8) for _ in weights]
        rows.append((words, (sum(w * x for w, x in zip(weights, words, strict=True)) + 5) % mod))
    assert solver._linear_weights(rows, mod) == tuple(weights)


def test_linear_weights_calls_null_relations(monkeypatch):
    calls = []
    real = solver.null_relations
    monkeypatch.setattr(solver, "null_relations", lambda *a, **k: calls.append(1) or real(*a, **k))
    rng = random.Random(12)
    rows = [([rng.randrange(256) for _ in range(2)], rng.randrange(1 << 16)) for _ in range(4)]
    solver._linear_weights(rows, 1 << 16)
    assert calls


def test_linear_weights_rejects_noise():
    """Falsification: random checksums have no small weight relation."""
    rng = random.Random(13)
    for _ in range(10):
        rows = [
            ([rng.randrange(1 << 16) for _ in range(4)], rng.randrange(1 << 16)) for _ in range(6)
        ]
        assert solver._linear_weights(rows, 1 << 16) is None


def test_linear_weights_constant_words_have_no_relation():
    rows = [([5, 6], c) for c in (1, 2, 3, 4)]
    assert solver._linear_weights(rows, 1 << 16) is None
