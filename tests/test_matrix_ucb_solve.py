"""matrix_ucb_select must score identically without forming the inverse.

The old route factored the covariance, discarded the factor, re-formed
``L @ L.T``, ran a general LU against the whole identity to get ``C^-1``,
and then read one vector out of it -- computing ``mu^T C^-1 mu`` and
``C^-1 mu`` as two separate products of the same quantity.

The scores only ever read ``x = C^-1 mu`` and ``mu . x``, so the rewrite
solves that single right-hand side against the factor already in hand. The
oracle below is the old computation written out verbatim; agreement has to
hold to floating-point tolerance, since the two orderings are algebraically
equal but not bit-identical.
"""

import math

import pytest

from fuzzer_tool.core.schedulers.monte_carlo import MonteCarloScheduler

np = pytest.importorskip("numpy")


def _old_way(C, mu):
    """Inverse-then-multiply, as matrix_ucb_select used to do it."""
    L = np.linalg.cholesky(C)
    inv = np.linalg.solve(L @ L.T, np.eye(len(C)))
    return float(mu @ inv @ mu), inv @ mu


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T / n + np.eye(n) * 0.5, rng.random(n)


@pytest.mark.parametrize("n", [3, 8, 32, 155])
def test_solve_matches_inverse_then_multiply(n):
    C, mu = _spd(n, seed=n)
    base_old, x_old = _old_way(C, mu)

    L = np.linalg.cholesky(C)
    x_new = MonteCarloScheduler._solve_cholesky_vec(L, mu, n)
    base_new = MonteCarloScheduler._dot(mu, x_new, n)

    assert np.allclose(x_new, x_old, rtol=1e-9, atol=1e-11)
    assert base_new == pytest.approx(base_old, rel=1e-9, abs=1e-11)


def test_quadratic_form_is_the_dot_of_mu_with_the_solve():
    """mu^T C^-1 mu == mu . (C^-1 mu); it was computed twice before."""
    C, mu = _spd(24, seed=5)
    L = np.linalg.cholesky(C)
    x = MonteCarloScheduler._solve_cholesky_vec(L, mu, 24)
    assert float(mu @ x) == pytest.approx(float(mu @ np.linalg.inv(C) @ mu), rel=1e-9)


def test_solve_actually_inverts():
    """C @ x must return mu."""
    C, mu = _spd(16, seed=11)
    L = np.linalg.cholesky(C)
    x = MonteCarloScheduler._solve_cholesky_vec(L, mu, 16)
    assert np.allclose(C @ x, mu, rtol=1e-9, atol=1e-11)


def test_pure_python_solve_matches_numpy_solve():
    """The no-numpy fallback path must agree with the vectorized one."""
    C, mu = _spd(12, seed=21)
    L = np.linalg.cholesky(C)
    x_np = MonteCarloScheduler._solve_cholesky_vec(L, mu, 12)
    x_py = MonteCarloScheduler._solve_cholesky_vec(L.tolist(), mu.tolist(), 12)
    assert np.allclose(np.asarray(x_py), x_np, rtol=1e-9, atol=1e-11)


def test_singular_factor_returns_none():
    """A zero on the diagonal must fall back, not divide by zero."""
    L = np.eye(4)
    L[2, 2] = 0.0
    assert MonteCarloScheduler._solve_cholesky_vec(L, np.ones(4), 4) is None
    assert MonteCarloScheduler._solve_cholesky_vec(L.tolist(), [1.0] * 4, 4) is None


def test_empty_returns_none():
    assert MonteCarloScheduler._solve_cholesky_vec(np.zeros((0, 0)), np.zeros(0), 0) is None


def test_scores_match_the_inverse_formulation():
    """End-to-end: same UCB scores as the inverse-based helper produced."""
    n = 20
    C, mu = _spd(n, seed=42)
    ops = [f"op{i}" for i in range(n)]
    beta, t = 2.0, 500

    base_old, inv_mu_old = _old_way(C, mu)
    expected = {
        op: float(mu[i])
        + beta * math.sqrt(max(0.0, math.log(t) + base_old - 2.0 * float(inv_mu_old[i])))
        for i, op in enumerate(ops)
    }

    sched = MonteCarloScheduler()
    L = np.linalg.cholesky(C)
    x = MonteCarloScheduler._solve_cholesky_vec(L, mu, n)
    got = sched._matrix_ucb_scores(ops, mu, x, MonteCarloScheduler._dot(mu, x, n), beta, t, n)

    assert set(got) == set(expected)
    for op in ops:
        assert got[op] == pytest.approx(expected[op], rel=1e-9, abs=1e-11)
    # The decision, not just the numbers.
    assert max(ops, key=lambda o: got[o]) == max(ops, key=lambda o: expected[o])


def test_matrix_ucb_select_still_returns_a_listed_op():
    sched = MonteCarloScheduler()
    ops = ["a", "b", "c", "d"]
    for i, op in enumerate(ops):
        sched.arm_alpha[op] = 1.0 + i
        sched.arm_beta[op] = 2.0
    assert sched.matrix_ucb_select(ops) in ops
