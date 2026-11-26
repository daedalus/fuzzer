#!/usr/bin/env python3
"""Does an ICA rotation group byteflip rows better than byte_pca's PCA argmax?

byte_pca keeps up to MAX_COMPONENTS principal axes and puts each byte on the
axis (and pole) of its largest score. PCA axes are orthogonal; field edge
signatures that share branches are not, so PCA blends them. FastICA rotates the
same whitened subspace to maximally non-Gaussian axes, which sparse field
membership is::

    planted fields --sketch--> rows --centre, eigh, drop PC1, keep k-->  Z (white)
                                                        |                |
                                          PCA arm: argmax|score|   ICA arm: Z @ W.T,
                                                        |          argmax|source|
                                                        +-- same members --+
                                                        |
                                     adjusted Rand vs planted truth, paired per cell

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
EDGES_PER_FIELD = 40  # own edges per field, and size of the shared pool
FIELD_STRIDE = 10_000  # edge-id spacing between fields' own edges
BASE = np.arange(1, 1500)  # static trace: cancels at centring
SHARED = np.arange(90_000, 90_000 + EDGES_PER_FIELD)
REJECT = np.arange(50_000, 50_300)  # shared reject path of clutter bytes
REJECT_BASE = 50  # base edges a reject byte still hits

GRID_FIELDS = (2, 3, 4, 6)
GRID_OVERLAP = (0.0, 0.25, 0.5, 0.75)
GRID_PER_FIELD = 30
GRID_REJECT = 24


class Converge(Enum):
    YES = "yes"
    NO = "no"


@dataclass(frozen=True)
class Spec:
    """One planted cell: `n_fields` fields of `per_field` bytes, then clutter.

    Each field moves its own edges plus a field-specific subset of a shared
    pool; `overlap` is the chance a pool edge belongs to a field, so two fields
    share about overlap**2 of the pool. `keep` is the chance one byte's flip
    actually moves a given edge of its field (per-byte noise).
    """

    n_fields: int
    per_field: int
    overlap: float
    reject: int
    keep: float = 0.8


def planted(rand: RandPool, spec: Spec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offsets, sketch rows and truth labels (-1 = reject clutter) of one cell."""
    pool_mask = rand.random_array((spec.n_fields, SHARED.size)) < spec.overlap
    rows, truth = [], []

    # Field bytes: base + noisy own edges + noisy share of the pool.
    for g in range(spec.n_fields):
        own = np.arange(FIELD_STRIDE * (g + 1), FIELD_STRIDE * (g + 1) + EDGES_PER_FIELD)
        sig = np.concatenate([own, SHARED[pool_mask[g]]])
        hit = rand.random_array((spec.per_field, sig.size)) < spec.keep
        for h in hit:
            rows.append(sketch_row(np.concatenate([BASE, sig[h]])))
        truth += [g] * spec.per_field

    # Clutter: every flip trips the same reject path.
    reject_row = sketch_row(np.concatenate([BASE[:REJECT_BASE], REJECT]))
    rows += [reject_row] * spec.reject
    truth += [-1] * spec.reject

    return np.arange(len(rows)), np.stack(rows), np.array(truth)


def _sym_decorr(w: np.ndarray) -> np.ndarray:
    """(W W^T)^{-1/2} W: the nearest orthogonal matrix."""
    s, u = np.linalg.eigh(w @ w.T)
    return (u / np.sqrt(np.maximum(s, 1e-300))) @ u.T @ w


def fast_ica(
    z: np.ndarray, rand: RandPool, max_iter: int = ICA_MAX_ITER, tol: float = ICA_TOL
) -> tuple[np.ndarray, Converge]:
    """Symmetric FastICA (logcosh) on whitened `z` (n x k).

    Returns the orthogonal unmixing W (k x k); sources are ``z @ W.T``.
    """
    n, k = z.shape
    if k == 1:
        return np.eye(1), Converge.YES

    w = _sym_decorr(rand.normal_array((k, k)))
    for _ in range(max_iter):
        g = np.tanh(z @ w.T)
        w_new = _sym_decorr(g.T @ z / n - (1.0 - g * g).mean(axis=0)[:, None] * w)
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


def _subspace(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """byte_pca's kept subspace, whitened, its eigenvalues and member mask.

    Mirrors byte_pca._fit up to the axis choice so both arms share members.
    """
    n = rows.shape[0]
    if not MIN_ROWS <= n <= MAX_ROWS:
        return None
    y = rows.astype(np.float64)
    y -= y.mean(axis=0)
    if not y.any():
        return None

    w, u = np.linalg.eigh(y @ y.T)
    order = np.argsort(w)[::-1]
    w, u = np.maximum(w[order], 0.0), u[:, order]
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


def sign_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test p (ties already dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, j) for j in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def _score(rand: RandPool, spec: Spec) -> tuple[float, float, float]:
    """ARI of PCA and ICA on one draw, and of PCA on an independent draw."""
    offs, rows, truth = planted(rand, spec)
    p, i = pca_labels(offs, rows), ica_labels(offs, rows, rand)
    offs_b, rows_b, truth_b = planted(rand, spec)
    pb = pca_labels(offs_b, rows_b)
    ari = lambda t, lab: 0.0 if lab is None else adjusted_rand(t, lab)  # noqa: E731
    return ari(truth, p), ari(truth, i), ari(truth_b, pb)


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
    print(f"{'fields':>6} {'overlap':>7}  {'PCA':>6} {'ICA':>6} {'PCA-b':>6}")
    for n_fields in GRID_FIELDS:
        for overlap in GRID_OVERLAP:
            spec = Spec(n_fields, GRID_PER_FIELD, overlap, GRID_REJECT)
            res = np.array([_score(rand, spec) for _ in range(args.reps)])
            cells.append(res)
            m = res.mean(axis=0)
            print(f"{n_fields:>6} {overlap:>7.2f}  {m[0]:6.3f} {m[1]:6.3f} {m[2]:6.3f}")

    allr = np.concatenate(cells)
    print(f"\nmean ARI  PCA {allr[:, 0].mean():.3f}  ICA {allr[:, 1].mean():.3f}")
    _paired(allr[:, 2], allr[:, 0], "A/A PCA vs PCA-b")
    _paired(allr[:, 0], allr[:, 1], "ICA vs PCA")


if __name__ == "__main__":
    main()
