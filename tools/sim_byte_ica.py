#!/usr/bin/env python3
"""Does an ICA rotation group byteflip rows better than byte_pca's PCA argmax?

byte_pca keeps up to MAX_COMPONENTS principal axes and puts each byte on the
axis (and pole) of its largest score. PCA axes are orthogonal; field edge
signatures that share branches are not, so PCA blends them. FastICA rotates the
same whitened subspace to maximally non-Gaussian axes, which sparse field
membership is. Logcosh ICA loses (ledger #182); three fixes recover it, and
varimax does better still (``ica_fix_labels``, ``vmax_labels``)::

    planted fields --sketch--> rows --centre, eigh, drop PC1, keep k-->  Z (white)
                                                        |                |
                                          PCA arm: argmax|score|   ICA arm: Z @ W.T,
                                                        |          argmax|source|
                                                        +-- same members --+
                                                        |
                                     adjusted Rand vs planted truth, paired per cell

Why logcosh ICA lost, and the fixes:
  - 8 kept axes, ~n_fields real: whitening lifts the noise axes to unit
    variance -> keep the rank at the largest eigengap (``gap_rank``).
  - +-poles split each axis in two; k fields after centring are k vertices of a
    simplex in k-1 dims -> positive tail names the field, no positive score is
    the remaining vertex (``signed_labels``).
  - 4 equal fields form a tetrahedron whose 2-vs-2 splits are 3 orthogonal,
    bimodal (very non-Gaussian) axes; logcosh picks them. 1-vs-rest axes are
    the only skewed ones -> skewness contrast (``Contrast.SKEW``).
  - byte_pca drops PC1 unconditionally; without a reject block PC1 is a field
    -> ``vmax_labels`` keeps it and starts the gap search at rank 2.

Control (Hard Rule 46): PCA on a second, independent noise draw of the same
cell vs PCA on the first. It must come out non-significant.

Usage:
    python tools/sim_byte_ica.py [--reps 10] [--seed 1]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from fuzzer_tool.core.byte_pca import (
    DROP_LEADING,
    EIG_FLOOR,
    MAX_COMPONENTS,
    MAX_ROWS,
    MIN_KEPT_ENERGY,
    MIN_ROWS,
    adjusted_rand,
    byte_groups,
    sketch_row,
)
from fuzzer_tool.core.rand_pool import RandPool

ICA_MAX_ITER = 400
ICA_TOL = 1e-6
VARIMAX_ITER = 200
VARIMAX_TOL = 1e-8
MAX_RANK = 16  # eigenvalues searched for the gap in vmax_labels
FIX_MIN_RANK = 1  # ica_fix: PC1 already dropped
VMAX_MIN_RANK = 2  # vmax: skip the reject-vs-fields gap after PC1
MIN_GAP_RATIO = 2.0  # a gap past min_rank below this is noise (tuned on seed 1)
MIN_FIELD_BYTES = 2  # floor of a tapered field's size
EDGES_PER_FIELD = 40  # own edges per field, and size of the shared pool
FIELD_STRIDE = 10_000  # edge-id spacing between fields' own edges
BASE = np.arange(1, 1500)  # static trace: cancels at centring
SHARED = np.arange(90_000, 90_000 + EDGES_PER_FIELD)
REJECT = np.arange(50_000, 50_300)  # shared reject path of clutter bytes
REJECT_BASE = 50  # base edges a reject byte still hits

GRID_FIELDS = (2, 3, 4, 6)
GRID_OVERLAP = (0.0, 0.25, 0.5, 0.75)
GRID_PER_FIELD = 30
GRID_REJECT = (24, 0)
GRID_TAPER = (1.0, 0.5)


class Converge(Enum):
    YES = "yes"
    NO = "no"


class Contrast(Enum):
    """FastICA nonlinearity: LOGCOSH (symmetric negentropy) or SKEW (u^3/3)."""

    LOGCOSH = "logcosh"
    SKEW = "skew"


@dataclass(frozen=True)
class Spec:
    """One planted cell: `n_fields` fields of `per_field` bytes, then clutter.

    Each field moves its own edges plus a field-specific subset of a shared
    pool; `overlap` is the chance a pool edge belongs to a field, so two fields
    share about overlap**2 of the pool. `keep` is the chance one byte's flip
    actually moves a given edge of its field (per-byte noise). Field g has
    ``per_field * taper**g`` bytes (taper 0.5: 30, 15, 8, ...).
    """

    n_fields: int
    per_field: int
    overlap: float
    reject: int
    keep: float = 0.8
    taper: float = 1.0


def planted(rand: RandPool, spec: Spec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offsets, sketch rows and truth labels (-1 = reject clutter) of one cell."""
    pool_mask = rand.random_array((spec.n_fields, SHARED.size)) < spec.overlap
    rows, truth = [], []

    # Field bytes: base + noisy own edges + noisy share of the pool.
    for g in range(spec.n_fields):
        own = np.arange(FIELD_STRIDE * (g + 1), FIELD_STRIDE * (g + 1) + EDGES_PER_FIELD)
        sig = np.concatenate([own, SHARED[pool_mask[g]]])
        size = max(MIN_FIELD_BYTES, round(spec.per_field * spec.taper**g))
        hit = rand.random_array((size, sig.size)) < spec.keep
        for h in hit:
            rows.append(sketch_row(np.concatenate([BASE, sig[h]])))
        truth += [g] * size

    # Clutter: every flip trips the same reject path.
    reject_row = sketch_row(np.concatenate([BASE[:REJECT_BASE], REJECT]))
    rows += [reject_row] * spec.reject
    truth += [-1] * spec.reject

    return np.arange(len(rows)), np.stack(rows), np.array(truth)


def _sym_decorr(w: np.ndarray) -> np.ndarray:
    """(W W^T)^{-1/2} W: the nearest orthogonal matrix."""
    s, u = np.linalg.eigh(w @ w.T)
    return (u / np.sqrt(np.maximum(s, 1e-300))) @ u.T @ w


def _g_logcosh(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.tanh(u)
    return g, 1.0 - g * g


def _g_skew(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return u * u, 2.0 * u


_NONLIN = {Contrast.LOGCOSH: _g_logcosh, Contrast.SKEW: _g_skew}


def fast_ica(
    z: np.ndarray,
    rand: RandPool,
    max_iter: int = ICA_MAX_ITER,
    tol: float = ICA_TOL,
    contrast: Contrast = Contrast.LOGCOSH,
) -> tuple[np.ndarray, Converge]:
    """Symmetric FastICA on whitened `z` (n x k).

    Returns the orthogonal unmixing W (k x k); sources are ``z @ W.T``.
    """
    n, k = z.shape
    if k == 1:
        return np.eye(1), Converge.YES

    nonlin = _NONLIN[contrast]
    w = _sym_decorr(rand.normal_array((k, k)))
    for _ in range(max_iter):
        g, gp = nonlin(z @ w.T)
        w_new = _sym_decorr(g.T @ z / n - gp.mean(axis=0)[:, None] * w)
        drift = np.abs(np.abs(np.einsum("ij,ij->i", w_new, w)) - 1.0).max()
        w = w_new
        if drift < tol:
            return w, Converge.YES
    return w, Converge.NO


def pca_labels(offsets: np.ndarray, rows: np.ndarray) -> np.ndarray | None:
    """byte_groups as one label per row (-1 = not a member)."""
    g = byte_groups(offsets, rows)
    if g is None:
        return None
    labels = np.full(offsets.size, -1, dtype=np.int64)
    labels[np.searchsorted(offsets, g.offsets)] = g.comp
    return labels


def varimax(loadings: np.ndarray) -> np.ndarray:
    """Orthogonal R maximising the variance of squared ``loadings @ R`` (Kaiser).

    Sparse rows (a byte loading on one field) are its optimum.
    """
    p, k = loadings.shape
    rot = np.eye(k)
    if k == 1:
        return rot

    crit = 0.0
    for _ in range(VARIMAX_ITER):
        lr = loadings @ rot
        u, s, vt = np.linalg.svd(loadings.T @ (lr**3 - lr * ((lr**2).sum(axis=0) / p)))
        rot = u @ vt
        if s.sum() < crit * (1.0 + VARIMAX_TOL):
            break
        crit = s.sum()
    return rot


def gap_rank(eig: np.ndarray, min_rank: int, min_gap: float = 0.0) -> int:
    """Rank at the largest relative eigengap, searched from `min_rank` on.

    Example: [100, 50, 40, 1, 0.9] -> 3 (40 / 1 is the largest ratio). When
    that gap is below `min_gap` (noise only past min_rank), the largest gap
    before min_rank decides: [500, 10, 9.5, 9] with min_rank 2 -> 1.
    """
    if eig.size <= min_rank:
        return max(1, eig.size)
    ratio = eig[:-1] / np.maximum(eig[1:], 1e-300)
    rank = int(np.argmax(ratio[min_rank - 1 :])) + min_rank
    if ratio[rank - 1] >= min_gap or min_rank == 1:
        return rank
    return int(np.argmax(ratio[: min_rank - 1])) + 1


def signed_labels(src: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Vertex assignment: the source with the largest positive score names the
    field; a byte with no positive score is the remaining vertex (label k)."""
    src = src * np.where((src**3).mean(axis=0) < 0, -1.0, 1.0)  # heavy tail up
    axis = src.argmax(axis=1)
    top = src[np.arange(src.shape[0]), axis]
    labels = np.where(top > 0, axis, src.shape[1])
    return np.where(member, labels, -1).astype(np.int64)


def _spectrum(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Descending eigenvalues and eigenvectors of the centred row Gram."""
    if not MIN_ROWS <= rows.shape[0] <= MAX_ROWS:
        return None
    y = rows.astype(np.float64)
    y -= y.mean(axis=0)
    if not y.any():
        return None
    w, u = np.linalg.eigh(y @ y.T)
    order = np.argsort(w)[::-1]
    return np.maximum(w[order], 0.0), u[:, order]


def _subspace(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """byte_pca's kept subspace, whitened, its eigenvalues and member mask.

    Mirrors byte_pca._fit up to the axis choice so both arms share members.
    """
    spec = _spectrum(rows)
    if spec is None:
        return None
    w, u = spec
    n = rows.shape[0]
    live = int(np.count_nonzero(w > w[0] * EIG_FLOOR))
    if live <= DROP_LEADING:
        return None
    lo = DROP_LEADING if live - DROP_LEADING >= 2 else 0
    keep = slice(lo, min(live, lo + MAX_COMPONENTS))

    # Membership: same energy-share rule as byte_pca (rotation-invariant).
    k2 = u[:, keep] ** 2 * w[keep]
    total = (u[:, :live] ** 2 * w[:live]).sum(axis=1)
    share = np.divide(k2.sum(axis=1), total, out=np.zeros(n), where=total > 0)
    member = (share >= MIN_KEPT_ENERGY) & (k2.max(axis=1) > 0)

    return u[:, keep] * math.sqrt(n), w[keep], member


def ica_labels(offsets: np.ndarray, rows: np.ndarray, rand: RandPool) -> np.ndarray | None:
    """Same members as pca_labels; axis and pole from an ICA rotation."""
    if rows.shape[0] != offsets.size:
        return None
    fit = _subspace(rows)
    if fit is None:
        return None
    z, eig, member = fit
    if not member.any():
        return None

    w, _ = fast_ica(z, rand)
    src = z @ w.T

    # Whitening lifts noise axes to unit variance; weigh each source by its
    # mixing-column norm (its share of the raw energy) so they stay small.
    src *= np.sqrt((w * w) @ eig)

    # Sign is arbitrary in ICA: orient each source so its heavy tail is positive.
    src *= np.where((src**3).mean(axis=0) < 0, -1.0, 1.0)

    axis = np.abs(src).argmax(axis=1)
    pole = src[np.arange(src.shape[0]), axis] < 0
    return np.where(member, 2 * axis + pole, -1).astype(np.int64)


def ica_fix_labels(offsets: np.ndarray, rows: np.ndarray, rand: RandPool) -> np.ndarray | None:
    """ICA with the three fixes: gap rank, skew contrast, vertex assignment.

    Same members as pca_labels (PC1 still dropped).
    """
    if rows.shape[0] != offsets.size:
        return None
    fit = _subspace(rows)
    if fit is None or not fit[2].any():
        return None
    z, eig, member = fit

    r = gap_rank(eig, FIX_MIN_RANK)
    z, eig = z[:, :r], eig[:r]
    w, _ = fast_ica(z, rand, contrast=Contrast.SKEW)
    src = z @ w.T * np.sqrt((w * w) @ eig)
    return signed_labels(src, member)


def vmax_labels(offsets: np.ndarray, rows: np.ndarray) -> np.ndarray | None:
    """Varimax on the gap-rank scores, PC1 kept, vertex assignment."""
    spec = _spectrum(rows) if rows.shape[0] == offsets.size else None
    if spec is None:
        return None
    w, u = spec
    live = int(np.count_nonzero(w > w[0] * EIG_FLOOR))
    r = gap_rank(w[: min(live, MAX_RANK + 1)], VMAX_MIN_RANK, MIN_GAP_RATIO)

    # Membership: energy share inside the kept rank, as byte_pca does.
    scores = u[:, :r] * np.sqrt(w[:r])
    total = (u[:, :live] ** 2 * w[:live]).sum(axis=1)
    kept = (scores**2).sum(axis=1)
    member = np.divide(kept, total, out=np.zeros(kept.size), where=total > 0) >= MIN_KEPT_ENERGY
    if not member.any():
        return None

    return signed_labels(scores @ varimax(scores), member)


def sign_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test p (ties already dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, j) for j in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


ARMS = ("PCA", "ICA", "ICA-fix", "VMAX", "PCA-b")


def _score(rand: RandPool, spec: Spec) -> tuple[float, ...]:
    """ARI of each arm on one draw (ARMS order); PCA-b on an independent draw."""
    offs, rows, truth = planted(rand, spec)
    labels = (
        pca_labels(offs, rows),
        ica_labels(offs, rows, rand),
        ica_fix_labels(offs, rows, rand),
        vmax_labels(offs, rows),
    )
    offs_b, rows_b, truth_b = planted(rand, spec)
    ari = lambda t, lab: 0.0 if lab is None else adjusted_rand(t, lab)  # noqa: E731
    return (*(ari(truth, lab) for lab in labels), ari(truth_b, pca_labels(offs_b, rows_b)))


def _paired(a: np.ndarray, b: np.ndarray, label: str) -> None:
    d = b - a
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    print(
        f"{label:<22} W/L {wins}/{losses} ties {d.size - wins - losses}  "
        f"sign p={sign_p(wins, losses):.3g}  median d={np.median(d):+.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rand = RandPool(seed=args.seed)

    cells = []
    print(
        f"{'fields':>6} {'overlap':>7} {'reject':>6} {'taper':>5}  "
        + " ".join(f"{a:>7}" for a in ARMS)
    )
    for reject in GRID_REJECT:
        for taper in GRID_TAPER:
            for n_fields in GRID_FIELDS:
                for overlap in GRID_OVERLAP:
                    spec = Spec(n_fields, GRID_PER_FIELD, overlap, reject, taper=taper)
                    res = np.array([_score(rand, spec) for _ in range(args.reps)])
                    cells.append(res)
                    m = " ".join(f"{x:7.3f}" for x in res.mean(axis=0))
                    print(f"{n_fields:>6} {overlap:>7.2f} {reject:>6} {taper:>5}  {m}")

    allr = np.concatenate(cells)
    print("\nmean ARI  " + "  ".join(f"{a} {allr[:, j].mean():.3f}" for j, a in enumerate(ARMS)))
    _paired(allr[:, 4], allr[:, 0], "A/A PCA vs PCA-b")
    for j in (1, 2, 3):
        _paired(allr[:, 0], allr[:, j], f"{ARMS[j]} vs PCA")


if __name__ == "__main__":
    main()
