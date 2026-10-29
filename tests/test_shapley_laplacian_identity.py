"""The normalized Laplacian must equal the dense-diagonal construction.

``D^-1/2 K D^-1/2`` with D diagonal is a row-and-column rescale. The
spectral embedding used to build ``D^-1/2`` as a dense n-by-n and multiply
through, which is 2*O(n^3) to apply n numbers. The outer-product form is
O(n^2) and, because each entry is a scalar product rather than a summed
row, it is bit-identical rather than equal to tolerance -- these tests
assert exact equality so a future rewrite that reintroduces summation is
visible.
"""

import pytest

from fuzzer_tool.core.shapley import ShapleyAttribution

np = pytest.importorskip("numpy")


def _dense_reference(K):
    """The construction this replaced, written out verbatim."""
    n = len(K)
    degrees = K.sum(axis=1)
    d_inv_sqrt = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(d_inv_sqrt, 1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
    return np.eye(n, dtype=np.float64) - d_inv_sqrt @ K @ d_inv_sqrt


def _kernel(K):
    """Dict-of-dicts kernel that ShapleyAttribution consumes, from a matrix."""
    ops = [f"op{i}" for i in range(len(K))]
    return ops, {a: {b: float(K[i, j]) for j, b in enumerate(ops)} for i, a in enumerate(ops)}


def _src_vectors(K):
    """All n eigenvectors from the real spectral embedding, as an n-by-n matrix."""
    ops, kernel = _kernel(K)
    emb = ShapleyAttribution()._spectral_embedding_numpy(ops, kernel, k=len(ops))
    return np.array([emb[op] for op in ops])


def _ref_vectors(K):
    return np.linalg.eigh(_dense_reference(K))[1]


@pytest.mark.parametrize("n", [2, 5, 40, 155])
def test_outer_form_is_bit_identical(n):
    """Identical Laplacians give identical eigh output, so any drift in src shows."""
    rng = np.random.default_rng(n)
    K = np.abs(rng.standard_normal((n, n)))
    K = (K + K.T) / 2.0

    # Control (Hard Rule 46): the reference against a second run of itself.
    assert np.array_equal(_ref_vectors(K), _ref_vectors(K))
    assert np.array_equal(_src_vectors(K), _ref_vectors(K))


def test_zero_degree_row_uses_the_same_floor():
    """An isolated operator must not divide by zero, in either form."""
    K = np.zeros((4, 4))
    K[1, 2] = K[2, 1] = 1.0
    got = _src_vectors(K)
    assert np.array_equal(got, _ref_vectors(K))
    assert np.isfinite(got).all()


def test_spectral_embedding_still_produces_k_dimensions():
    ops = [f"op{i}" for i in range(6)]
    rng = np.random.default_rng(3)
    raw = np.abs(rng.standard_normal((6, 6)))
    raw = (raw + raw.T) / 2.0
    kernel = {a: {b: float(raw[i, j]) for j, b in enumerate(ops)} for i, a in enumerate(ops)}

    emb = ShapleyAttribution()._spectral_embedding_numpy(ops, kernel, k=3)
    assert set(emb) == set(ops)
    assert all(len(v) == 3 for v in emb.values())
    assert all(np.isfinite(v).all() for v in emb.values())


def test_laplacian_diagonal_is_the_expected_one_minus_self_similarity():
    """Sanity anchor on src itself, not just on two paths agreeing.

    K = [[2, 1], [1, 2]]: degrees 3, so L = I - K/3 = [[1/3, -1/3], [-1/3, 1/3]].
    Eigenvalues 0 and 2/3; eigenvectors (1, 1)/sqrt2 and (1, -1)/sqrt2.
    """
    vecs = _src_vectors(np.array([[2.0, 1.0], [1.0, 2.0]]))
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    assert np.allclose(np.abs(vecs), inv_sqrt2)
    # Column 0 (eigenvalue 0) is constant-sign, column 1 alternates.
    assert vecs[0, 0] * vecs[1, 0] > 0
    assert vecs[0, 1] * vecs[1, 1] < 0
