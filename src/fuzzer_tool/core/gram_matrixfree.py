"""Matrix-free Gram operators on the seed x edge incidence matrix A.

``B = A Aᵀ`` (seed overlap) and ``C = B² = A(AᵀA)Aᵀ`` are S x S. Building
them costs O(S²) memory, which is the wrong trade for a corpus that grows
without bound. Every product here is instead a chain of sparse products with
``A`` and ``Aᵀ``, O(nnz(A)) time and O(S + E) extra memory per matvec, and
no S x S (or E x E) array is ever allocated. This is the "never materialize
C" rule from ``docs/handover/handover_coverage_gram_matrix_2026-10-26.md``
(§4, §6 items 1 and 4).

Weighting is decided once, in :func:`column_weights`, and applied as a column
scale ``A·diag(w)``: ``"binary"`` (every hit counts 1), ``"rarity"`` (``w =
d⁻¹ᐟ²`` with ``d`` the edge's seed count, so ``B = A D⁻¹ Aᵀ`` and the shared
parser core stops dominating) or ``"core_stripped"`` (edges owned by at least
``core_frac`` of seeds get weight 0).

Provided: :meth:`Incidence.matvec_b` / :meth:`~Incidence.matvec_c`,
:func:`top_eigenpair` (eigenvector centrality), :func:`top_k_eigenpairs`,
:func:`hub_scores`, and :func:`participation_ratio` (exact, or a Hutchinson
estimate when a ``RandPool`` is supplied). Not wired into the CLI or report.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

WEIGHTINGS = ("binary", "rarity", "core_stripped")
DEFAULT_CORE_FRAC = 0.9
DEFAULT_TOL = 1e-10
DEFAULT_MAX_ITER = 200  # Lanczos steps; basis memory is steps x n_seeds floats
LANCZOS_CHECK_EVERY = 3
LANCZOS_CHECK_EARLY = 8  # steps checked every time: big-gap operators stop here
BREAKDOWN_REL = 1e-12
MAX_ROUNDS = 16
GOLDEN = 0.6180339887498949


def compact_columns(rows: Sequence[np.ndarray]) -> tuple[list[np.ndarray], int]:
    """Renumber edge ids to ``0..k-1`` over the ids actually present.

    ``Incidence`` sizes its column arrays by the largest id. Hashed or
    context-sensitive edge ids are sparse in a huge range (measured on an
    ffmpeg corpus: max id 16.7M, 113k distinct), so uncompacted a refit took
    4.7 s and ~520 MB against 0.05 s compacted, with the same eigenvalue. The
    matrix is unchanged: columns are only relabelled.
    """
    lens = [len(r) for r in rows]
    if not sum(lens):
        return [np.zeros(0, dtype=np.int64) for _ in rows], 0
    flat = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
    uniq, inv = np.unique(flat, return_inverse=True)
    off = np.concatenate(([0], np.cumsum(lens)))
    return [inv[off[i] : off[i + 1]] for i in range(len(rows))], int(uniq.shape[0])


class Incidence:
    """Sparse seed x edge incidence in CSR form with a column scale.

    Args:
        rows: Per-seed arrays of distinct edge ids (any integer dtype).
        n_edges: Number of columns; inferred from the largest id when None.
        weighting: One of :data:`WEIGHTINGS`.
        core_frac: Ownership threshold for ``"core_stripped"``.
    """

    def __init__(
        self,
        rows: Sequence[np.ndarray],
        n_edges: int | None = None,
        weighting: str = "binary",
        core_frac: float = DEFAULT_CORE_FRAC,
    ) -> None:
        self.n_seeds = len(rows)
        lens = np.fromiter((len(r) for r in rows), dtype=np.int64, count=self.n_seeds)
        self.indptr = np.zeros(self.n_seeds + 1, dtype=np.int64)
        np.cumsum(lens, out=self.indptr[1:])
        nnz = int(self.indptr[-1])
        self.cols = (
            np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
            if nnz
            else np.zeros(0, dtype=np.int64)
        )
        inferred = int(self.cols.max()) + 1 if nnz else 0
        self.n_edges = inferred if n_edges is None else int(n_edges)
        if self.n_edges < inferred:
            raise ValueError("n_edges smaller than the largest edge id + 1")
        # Row index of every nnz entry, for the Aᵀ scatter.
        self._row_of = np.repeat(np.arange(self.n_seeds, dtype=np.int64), lens)
        self._nonempty = lens > 0
        self._starts = self.indptr[:-1][self._nonempty]
        self.degree = np.bincount(self.cols, minlength=self.n_edges)
        self.w = column_weights(self.degree, self.n_seeds, weighting, core_frac)
        self._wcols = self.w[self.cols]

    @property
    def nnz(self) -> int:
        return int(self.cols.shape[0])

    def matvec_a(self, v: np.ndarray) -> np.ndarray:
        """``A·diag(w)·v`` for ``v`` of length ``n_edges`` -> length ``n_seeds``."""
        out = np.zeros(self.n_seeds, dtype=np.float64)
        if self.nnz:
            out[self._nonempty] = np.add.reduceat(v[self.cols] * self._wcols, self._starts)
        return out

    def matvec_at(self, u: np.ndarray) -> np.ndarray:
        """``diag(w)·Aᵀ·u`` for ``u`` of length ``n_seeds`` -> length ``n_edges``."""
        out = np.bincount(self.cols, weights=u[self._row_of], minlength=self.n_edges)
        return out * self.w

    def matvec_b(self, u: np.ndarray) -> np.ndarray:
        """``B u`` with ``B = A diag(w²) Aᵀ``, in two sparse products."""
        return self.matvec_a(self.matvec_at(u))

    def matvec_c(self, u: np.ndarray) -> np.ndarray:
        """``C u = A(AᵀA)Aᵀ u = B(B u)``, four sparse products, no C."""
        return self.matvec_b(self.matvec_b(u))

    def diag_b(self) -> np.ndarray:
        """``diag(B)``: weighted coverage size per seed, O(nnz)."""
        out = np.zeros(self.n_seeds, dtype=np.float64)
        if self.nnz:
            out[self._nonempty] = np.add.reduceat(self._wcols**2, self._starts)
        return out


def column_weights(
    degree: np.ndarray,
    n_seeds: int,
    weighting: str = "binary",
    core_frac: float = DEFAULT_CORE_FRAC,
) -> np.ndarray:
    """Per-edge column scale ``w`` such that ``B = A diag(w²) Aᵀ``."""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {weighting!r}; expected one of {WEIGHTINGS}")
    deg = degree.astype(np.float64)
    if weighting == "binary":
        return (deg > 0).astype(np.float64)
    if weighting == "rarity":
        w = np.zeros_like(deg)
        live = deg > 0
        w[live] = deg[live] ** -0.5
        return w
    w = (deg > 0).astype(np.float64)
    w[deg >= core_frac * n_seeds] = 0.0
    return w


def _start_vec(n: int, rnd: int, locked: np.ndarray) -> np.ndarray:
    """Deterministic generic start, deflated against ``locked`` (n x p)."""
    v = np.cos(np.pi * (np.arange(n) + 0.5) * (rnd + 1) * GOLDEN)
    v -= locked @ (locked.T @ v)
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > BREAKDOWN_REL else np.zeros(n)


def _lanczos(
    mv,
    start: np.ndarray,
    locked: np.ndarray,
    want: int,
    tol: float,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Top ``want`` Ritz pairs of symmetric ``mv`` on span(start) \\ locked.

    One-pass Lanczos with full reorthogonalisation: the basis lives in one
    ``(steps+1) x n`` array (steps <= ``max_steps``), so memory stays
    bounded. Stops when the top ``want`` Ritz residuals ``|β_j s_last|`` are
    <= ``tol`` x the largest Ritz value, or on breakdown (invariant
    subspace, Ritz pairs exact). Returns ``(values desc, vectors as
    columns, steps)``; ``steps`` is the matvec count.
    """
    n = start.shape[0]
    cap = max(1, min(max_steps, n - locked.shape[1]))
    basis = np.empty((cap + 1, n))
    basis[0] = start
    alpha = np.zeros(cap)
    beta = np.zeros(cap)
    for j in range(cap):
        q = basis[j]
        w = mv(q)
        alpha[j] = float(q @ w)
        w -= alpha[j] * q
        if j:
            w -= beta[j - 1] * basis[j - 1]
        w -= basis[: j + 1].T @ (basis[: j + 1] @ w)
        w -= locked @ (locked.T @ w)
        beta[j] = float(np.linalg.norm(w))
        steps = j + 1
        scale = float(np.abs(alpha[:steps]).max())
        broke = beta[j] <= BREAKDOWN_REL * max(scale, 1e-300)
        due = steps <= LANCZOS_CHECK_EARLY or steps % LANCZOS_CHECK_EVERY == 0
        if not (broke or steps == cap or due):
            basis[j + 1] = w / beta[j]
            continue
        t = np.diag(alpha[:steps])
        if steps > 1:
            off = beta[: steps - 1]
            t += np.diag(off, 1) + np.diag(off, -1)
        theta, s = np.linalg.eigh(t)
        top = np.argsort(theta)[::-1][:want]
        theta, s = theta[top], s[:, top]
        res = beta[j] * np.abs(s[-1])
        if broke or steps == cap or bool(np.all(res <= tol * max(abs(theta[0]), 1e-300))):
            return theta, basis[:steps].T @ s, steps
        basis[j + 1] = w / beta[j]
    raise AssertionError("unreachable: the last step always returns")


def top_eigenpair(
    inc: Incidence,
    power: int = 1,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[float, np.ndarray, int]:
    """Dominant eigenpair of ``B`` (``power=1``) or ``C`` (``power=2``).

    Lanczos from the all-ones vector (no RNG: ``B`` is non-negative, so the
    start has a component along the Perron vector unless ``B`` is zero).
    ``max_iter`` caps the Lanczos steps (= matvecs). Returns ``(eigenvalue,
    unit eigenvector >= 0, steps)``; the eigenvector is the
    eigenvector-centrality score of each seed.
    """
    mv = inc.matvec_c if power == 2 else inc.matvec_b
    n = inc.n_seeds
    if n == 0:
        return 0.0, np.zeros(0), 0
    none = np.zeros((n, 0))
    theta, vecs, steps = _lanczos(mv, np.full(n, 1.0 / np.sqrt(n)), none, 1, tol, max_iter)
    if theta[0] <= 0.0:
        return 0.0, np.zeros(n), steps
    v = np.maximum(vecs[:, 0] * (1.0 if vecs[:, 0].sum() >= 0.0 else -1.0), 0.0)
    return float(theta[0]), v / np.linalg.norm(v), steps


def top_k_eigenpairs(
    inc: Incidence,
    k: int,
    tol: float = 1e-9,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` eigenpairs of ``B`` by Lanczos with deflation rounds.

    One Krylov space sees each distinct eigenvalue once, so round ``r`` runs
    Lanczos from a fresh deterministic start orthogonal to every pair found
    so far; the loop ends when a round's best value no longer beats the
    current ``k``-th (repeated eigenvalues are caught this way, up to
    ``max(MAX_ROUNDS, k)`` rounds). Memory is O(S·steps + S·found). Returns
    ``(eigenvalues descending, eigenvectors as columns)``; ``k`` is clipped
    to ``n_seeds``.
    """
    n = inc.n_seeds
    k = max(0, min(int(k), n))
    if k == 0:
        return np.zeros(0), np.zeros((n, 0))
    locked = np.zeros((n, 0))
    found = np.zeros(0)
    for rnd in range(max(MAX_ROUNDS, k)):
        start = _start_vec(n, rnd, locked)
        if not start.any():
            break
        vals, vecs, _ = _lanczos(inc.matvec_b, start, locked, k, tol, max_iter)
        kth = np.sort(found)[::-1][k - 1] if found.size >= k else -np.inf
        beaten = float(vals[0]) > kth + tol * max(abs(float(vals[0])), 1e-300)
        locked = np.column_stack([locked, vecs])
        found = np.concatenate([found, vals])
        if not beaten or locked.shape[1] >= n:
            break
    order = np.argsort(found)[::-1][:k]
    return found[order], locked[:, order]


def hub_scores(inc: Incidence) -> tuple[np.ndarray, np.ndarray]:
    """Row sums of ``B`` and ``C`` (``B·1``, ``C·1``) without forming either."""
    ones = np.ones(inc.n_seeds)
    b1 = inc.matvec_b(ones)
    return b1, inc.matvec_b(b1)


def participation_ratio(inc: Incidence, rand=None, probes: int = 0) -> float:
    """``PR = (tr B)² / tr(B²)`` = ``(Σσ²)² / Σσ⁴`` of ``A diag(w)``.

    ``tr B = Σ diag(B)`` is exact in O(nnz). ``tr(B²) = ‖B‖_F²`` is exact
    with ``probes=0`` (one ``B e_i`` per seed: O(S·nnz), still no S x S
    array) or a Hutchinson estimate ``mean ‖B z‖²`` over ``probes``
    Rademacher vectors drawn from ``rand`` (a ``RandPool``; Hard Rule 16).
    """
    n = inc.n_seeds
    tr_b = float(inc.diag_b().sum())
    if n == 0 or tr_b == 0.0:
        return 0.0
    if probes <= 0:
        tr_b2 = 0.0
        e = np.zeros(n)
        for i in range(n):
            e[i] = 1.0
            col = inc.matvec_b(e)
            tr_b2 += float(col @ col)
            e[i] = 0.0
    else:
        if rand is None:
            raise ValueError("a RandPool is required for the Hutchinson estimate")
        acc = 0.0
        for _ in range(probes):
            bits = np.frombuffer(rand.randbytes(n), dtype=np.uint8) & 1
            z = bits.astype(np.float64) * 2.0 - 1.0
            bz = inc.matvec_b(z)
            acc += float(bz @ bz)
        tr_b2 = acc / probes
    return tr_b * tr_b / tr_b2 if tr_b2 > 0 else 0.0
