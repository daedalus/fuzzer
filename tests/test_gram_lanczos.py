"""Lanczos eigensolvers on the matrix-free Gram operator (B = A Aᵀ)."""

import numpy as np
import pytest

from fuzzer_tool.core.gram_matrixfree import Incidence, top_eigenpair, top_k_eigenpairs
from fuzzer_tool.core.rand_pool import RandPool

REGIONS = 12
REGION_EDGES = 300
CORE = 40


class _Counting(Incidence):
    """Incidence that counts ``matvec_b`` calls."""

    calls = 0

    def matvec_b(self, u):
        type(self).calls += 1
        return super().matvec_b(u)


def _rows(n_seeds, seed, overlap=3):
    """Seeds in regions sharing a full core: top eigenvalues nearly tied."""
    rp = RandPool(seed=seed)
    rows = []
    for _ in range(n_seeds):
        raw = np.frombuffer(rp.randbytes(160), dtype=np.uint8).astype(np.int64)
        reg = int(raw[0]) % REGIONS
        own = CORE + reg * REGION_EDGES + raw[1:60]
        oth = CORE + (int(raw[60]) % REGIONS) * REGION_EDGES + raw[61 : 61 + overlap]
        rows.append(np.unique(np.concatenate([np.arange(CORE), own, oth])))
    return rows


def _dense_b(rows, n_edges, w):
    a = np.zeros((len(rows), n_edges))
    for i, r in enumerate(rows):
        a[i, r] = 1.0
    a *= w[None, :]
    return a @ a.T


def test_top_eigenpair_few_matvecs_on_small_gap():
    rows = _rows(600, seed=3)
    _Counting.calls = 0
    inc = _Counting(rows, weighting="core_stripped")
    lam, v, steps = top_eigenpair(inc)
    ref = np.linalg.eigvalsh(_dense_b(rows, inc.n_edges, inc.w))[-1]
    assert lam == pytest.approx(ref, rel=1e-8)
    assert _Counting.calls <= 30  # power iteration needs ~80 here
    assert steps <= 30


def test_top_eigenpair_vector_matches_eigh_and_is_nonnegative():
    rows = _rows(300, seed=5)
    inc = Incidence(rows, weighting="rarity")
    evals, evecs = np.linalg.eigh(_dense_b(rows, inc.n_edges, inc.w))
    lam, v, _ = top_eigenpair(inc)
    assert lam == pytest.approx(evals[-1], rel=1e-9)
    assert abs(float(v @ evecs[:, -1])) == pytest.approx(1.0, abs=1e-6)
    assert float(v.min()) >= 0.0 and np.linalg.norm(v) == pytest.approx(1.0)


def test_top_eigenpair_is_deterministic():
    inc = Incidence(_rows(200, seed=7))
    a, b = top_eigenpair(inc), top_eigenpair(inc)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


def test_top_eigenpair_empty_zero_and_single():
    assert top_eigenpair(Incidence([]))[0] == 0.0
    lam, v, _ = top_eigenpair(Incidence([np.array([], dtype=np.int64)] * 4, 3))
    assert lam == 0.0 and not v.any()
    lam, v, _ = top_eigenpair(Incidence([np.array([0, 1, 2])], 3))
    assert lam == pytest.approx(3.0) and v.tolist() == [1.0]


def test_top_eigenpair_step_cap_bounds_memory():
    inc = Incidence(_rows(300, seed=9), weighting="core_stripped")
    _, _, steps = top_eigenpair(inc, max_iter=3)
    assert steps <= 3


def test_top_k_matches_eigh_on_clustered_spectrum():
    rows = _rows(500, seed=11)
    inc = Incidence(rows)
    evals = np.linalg.eigvalsh(_dense_b(rows, inc.n_edges, inc.w))[::-1]
    vals, q = top_k_eigenpairs(inc, 8)
    assert np.allclose(vals, evals[:8], rtol=1e-6)
    assert np.allclose(q.T @ q, np.eye(8), atol=1e-8)
    res = np.linalg.norm(inc.matvec_b(q[:, 0]) - vals[0] * q[:, 0])
    assert res <= 1e-6 * vals[0]


def test_top_k_counts_repeated_eigenvalues():
    # Four disjoint, identical 5-seed blocks: B = 4 copies of 3*J_5, so
    # the top eigenvalue 15 has multiplicity 4. A single Krylov space sees it once.
    rows = [np.array([3 * b, 3 * b + 1, 3 * b + 2]) for b in range(4) for _ in range(5)]
    vals, q = top_k_eigenpairs(Incidence(rows), 3)
    assert np.allclose(vals, 15.0)
    assert np.allclose(q.T @ q, np.eye(3), atol=1e-8)


def test_top_k_clips_k_and_handles_rank_deficiency():
    rows = [np.array([0, 1])] * 3
    vals, q = top_k_eigenpairs(Incidence(rows, 2), 10)
    assert vals.shape == (3,) and q.shape == (3, 3)
    assert vals[0] == pytest.approx(6.0) and np.allclose(vals[1:], 0.0, atol=1e-9)
    assert top_k_eigenpairs(Incidence(rows, 2), 0)[0].shape == (0,)


def test_top_k_is_much_faster_than_subspace_iteration_matvecs():
    rows = _rows(500, seed=13)
    _Counting.calls = 0
    top_k_eigenpairs(_Counting(rows), 8)
    assert _Counting.calls <= 400  # block subspace iteration used 300 iterations x 16 matvecs
