"""Matrix-free Gram operators vs dense references built independently."""

import numpy as np
import pytest

from fuzzer_tool.core.gram_matrixfree import (
    Incidence,
    column_weights,
    compact_columns,
    hub_scores,
    participation_ratio,
    top_eigenpair,
    top_k_eigenpairs,
)
from fuzzer_tool.core.rand_pool import RandPool

ROWS = [
    np.array([0, 1, 2, 5]),
    np.array([0, 1, 2]),
    np.array([0, 3, 4, 5, 7]),
    np.array([], dtype=np.int64),
    np.array([0, 6, 7]),
    np.array([0, 1, 2]),
]
N_EDGES = 9  # edge 8 is hit by nobody


def _dense(rows, n_edges, w):
    a = np.zeros((len(rows), n_edges))
    for i, r in enumerate(rows):
        a[i, r] = 1.0
    return a * w[None, :]


def _w(weighting):
    deg = np.zeros(N_EDGES)
    for r in ROWS:
        deg[r] += 1
    return column_weights(deg, len(ROWS), weighting)


@pytest.mark.parametrize("weighting", ["binary", "rarity", "core_stripped"])
def test_matvecs_match_dense(weighting):
    inc = Incidence(ROWS, N_EDGES, weighting)
    a = _dense(ROWS, N_EDGES, _w(weighting))
    b = a @ a.T
    u = np.arange(1, len(ROWS) + 1, dtype=float)
    assert np.allclose(inc.matvec_b(u), b @ u)
    assert np.allclose(inc.matvec_c(u), a @ (a.T @ (a @ (a.T @ u))))
    assert np.allclose(inc.matvec_c(u), (b @ b) @ u)
    assert np.allclose(inc.diag_b(), np.diag(b))


def test_empty_seed_row_is_zero():
    inc = Incidence(ROWS, N_EDGES)
    assert inc.matvec_b(np.ones(len(ROWS)))[3] == 0.0


def test_core_stripped_drops_hot_edge():
    deg = np.zeros(N_EDGES)
    for r in ROWS:
        deg[r] += 1
    n = len(ROWS)
    # Edge 0 is owned by 5 of 6 seeds (0.83): stripped at 0.8, kept at 0.9.
    w_low = column_weights(deg, n, "core_stripped", core_frac=0.8)
    w_high = column_weights(deg, n, "core_stripped", core_frac=0.9)
    assert w_low[0] == 0.0 and w_high[0] == 1.0
    assert w_low[8] == 0.0  # unowned edge never weighs anything
    assert np.array_equal(w_low[1:8] > 0, deg[1:8] < 0.8 * n)


def test_top_eigenpair_matches_eigh():
    inc = Incidence(ROWS, N_EDGES, "rarity")
    a = _dense(ROWS, N_EDGES, _w("rarity"))
    b = a @ a.T
    evals, evecs = np.linalg.eigh(b)
    lam, v, _ = top_eigenpair(inc)
    assert lam == pytest.approx(evals[-1], rel=1e-8)
    assert abs(float(v @ evecs[:, -1])) == pytest.approx(1.0, abs=1e-6)
    lam_c, _, _ = top_eigenpair(inc, power=2)
    assert lam_c == pytest.approx(evals[-1] ** 2, rel=1e-8)


def test_top_k_matches_eigh():
    inc = Incidence(ROWS, N_EDGES)
    a = _dense(ROWS, N_EDGES, _w("binary"))
    evals = np.linalg.eigvalsh(a @ a.T)[::-1]
    vals, q = top_k_eigenpairs(inc, 3)
    assert np.allclose(vals, evals[:3], rtol=1e-6)
    assert np.allclose(q.T @ q, np.eye(3), atol=1e-8)


def test_hub_scores_are_row_sums():
    inc = Incidence(ROWS, N_EDGES)
    a = _dense(ROWS, N_EDGES, _w("binary"))
    b1, c1 = hub_scores(inc)
    assert np.allclose(b1, (a @ a.T).sum(axis=1))
    assert np.allclose(c1, (a @ a.T @ a @ a.T).sum(axis=1))


def test_participation_ratio_exact_matches_singular_values():
    inc = Incidence(ROWS, N_EDGES)
    a = _dense(ROWS, N_EDGES, _w("binary"))
    s2 = np.linalg.svd(a, compute_uv=False) ** 2
    assert participation_ratio(inc) == pytest.approx(s2.sum() ** 2 / (s2**2).sum())


def test_duplicates_lower_pr_and_disjoint_raise_it():
    base = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    pr_base = participation_ratio(Incidence(base, 8))
    pr_dup = participation_ratio(Incidence(base + [base[0]] * 3, 8))
    pr_new = participation_ratio(Incidence(base + [np.array([6, 7])], 8))
    assert pr_dup < pr_base < pr_new
    assert pr_base == pytest.approx(3.0)  # three orthogonal equal seeds


def test_hutchinson_close_to_exact_and_requires_rand():
    inc = Incidence(ROWS, N_EDGES)
    exact = participation_ratio(inc)
    est = participation_ratio(inc, rand=RandPool(seed=7), probes=4000)
    assert est == pytest.approx(exact, rel=0.1)
    with pytest.raises(ValueError):
        participation_ratio(inc, probes=3)


def test_rejects_small_n_edges_and_bad_weighting():
    with pytest.raises(ValueError):
        Incidence(ROWS, 3)
    with pytest.raises(ValueError):
        Incidence(ROWS, N_EDGES, "nope")


def test_no_dense_gram_allocated_at_scale():
    # 20k seeds: a dense B would be 3.2 GB; this must run in well under that.
    n = 20000
    rows = [np.array([i % 997, (i * 7) % 997 + 997, 2000 + i % 13]) for i in range(n)]
    inc = Incidence(rows, 2100)
    lam, v, _ = top_eigenpair(inc)
    assert lam > 0 and v.shape == (n,)


def test_compact_columns_keeps_the_matrix_and_shrinks_the_arrays():
    sparse = [np.array([5, 10**7, 16_700_000]), np.array([5, 99]), np.array([], dtype=np.int64)]
    rows, k = compact_columns(sparse)
    assert k == 4 and max(int(r.max()) for r in rows if len(r)) == 3
    assert [len(r) for r in rows] == [3, 2, 0]
    big = Incidence(sparse, weighting="rarity")
    small = Incidence(rows, n_edges=k, weighting="rarity")
    u = np.array([1.0, 2.0, 3.0])
    assert np.allclose(big.matvec_b(u), small.matvec_b(u))
    assert big.n_edges > 10**7 > small.n_edges
    empty, k0 = compact_columns([np.array([], dtype=np.int64)])
    assert k0 == 0 and len(empty[0]) == 0
