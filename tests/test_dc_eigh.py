"""Divide-and-conquer symmetric eigensolver (core/dc_eigh.py)."""

import math

import pytest

from fuzzer_tool.core.dc_eigh import eigh
from fuzzer_tool.core.shapley import ShapleyAttribution

np = pytest.importorskip("numpy")

RESID_TOL = 1e-9


def _check(a, w, v):
    """Oracle: ascending w, A v = v diag(w), V orthonormal."""
    a, w, v = np.asarray(a, float), np.asarray(w, float), np.asarray(v, float)
    n = len(a)
    scale = max(1.0, np.abs(a).max())
    assert np.all(np.diff(w) >= -RESID_TOL * scale)
    assert np.abs(a @ v - v * w).max() <= RESID_TOL * scale
    assert np.abs(v.T @ v - np.eye(n)).max() <= RESID_TOL


def _sym(n, seed):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return (m + m.T) / 2.0


def test_oracle_control():
    """Hard Rule 46: oracle passes LAPACK's own output, rejects a broken one."""
    a = _sym(12, 0)
    w, v = np.linalg.eigh(a)
    _check(a, w, v)

    bad = v.copy()
    bad[:, 0] = bad[:, 1]
    with pytest.raises(AssertionError):
        _check(a, w, bad)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 33, 64])
def test_matches_lapack(n):
    a = _sym(n, n)
    w, v = eigh(a.tolist())
    _check(a, w, v)
    assert np.allclose(w, np.linalg.eigvalsh(a), atol=1e-10)


def test_falsify_wrong_eigenvalue_detected():
    """Falsification: perturbing an output eigenvalue must fail the oracle."""
    a = _sym(9, 4)
    w, v = eigh(a.tolist())
    w[3] += 1e-3
    with pytest.raises(AssertionError):
        _check(a, w, v)


@pytest.mark.parametrize(
    "make",
    [
        lambda n: np.eye(n),  # fully degenerate
        lambda n: np.zeros((n, n)),
        lambda n: np.diag(np.arange(n, dtype=float)),  # already diagonal
        lambda n: np.ones((n, n)),  # rank one, n-1 zeros
        lambda n: np.diag([1.0] * (n // 2) + [1.0 + 1e-15] * (n - n // 2)),
        lambda n: np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1),  # path graph
        lambda n: 1e150 * _sym(n, 7),
        lambda n: 1e-150 * _sym(n, 8),
    ],
)
def test_adversarial_spectra(make):
    """Adversarial: clustered/degenerate eigenvalues and extreme scales."""
    a = make(20)
    w, v = eigh(a.tolist())
    scale = max(1.0, float(np.abs(a).max()))
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(v))
    assert np.abs(np.asarray(v).T @ np.asarray(v) - np.eye(20)).max() <= RESID_TOL
    assert np.abs(a @ np.asarray(v) - np.asarray(v) * w).max() <= RESID_TOL * scale


def test_empty():
    assert eigh([]) == ([], [])


def test_rejects_non_square():
    with pytest.raises(ValueError):
        eigh([[1.0, 2.0]])


def test_regression_py_spectral_distinct_vectors():
    """Pure-Python spectral fallback returned k copies of one eigenvector."""
    ops = [f"op{i}" for i in range(6)]
    raw = np.abs(_sym(6, 3))
    kernel = {a: {b: float(raw[i, j]) for j, b in enumerate(ops)} for i, a in enumerate(ops)}

    got = ShapleyAttribution()._spectral_embedding_py(ops, kernel, k=3)
    lap = ShapleyAttribution._build_laplacian_py(kernel, ops, len(ops))
    r = np.linalg.eigh(np.array(lap))[1][:, :3]

    g = np.array([got[op] for op in ops])
    assert np.abs(g.T @ g - np.eye(3)).max() <= RESID_TOL
    # Eigenvectors are unique up to sign for a simple spectrum.
    for d in range(3):
        assert math.isclose(abs(float(g[:, d] @ r[:, d])), 1.0, abs_tol=1e-9)
