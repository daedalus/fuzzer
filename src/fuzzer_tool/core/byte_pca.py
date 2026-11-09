"""Byte groups from a PCA of the byteflip trace matrix.

The deterministic byteflip pass executes one mutant per byte. Each changed
trace is a row; PCA over the rows says which bytes move the same edges::

    byte   edges moved (count-sketched into SKETCH_COLS columns)
    ----   ----------------------------------------------------
    0..3   [reject path ......................]   all alike  -> PC1 (clutter)
    4..7   [reject path][len-field branches ..]               -> PC2
    8      [reject path][checksum branches ...]               -> PC3

    centre columns (drop the static trace) -> eigh of the smaller Gram
    -> drop DROP_LEADING components (shared reject path / volume)
    -> byte i joins (component, sign) of its largest kept |score|; the sign
       is the pole, so one axis splits two groups

PC1 of a seed x edge matrix is volume (rho = +0.989, ``handover_edge_id_axis``
F13), and the same holds here: bytes whose flip trips the shared reject path
load on one component whatever field they belong to. Dropping it (Doppler's
wall filter, ``core/power_doppler.py``) leaves the groups that differ.

A component's axis is only defined when its eigenvalue is separated from its
neighbours: near-equal eigenvalues span a plane in which the eigenvectors are an
arbitrary rotation, so which group a byte lands in (the argmax) can flip with a
little noise. ``group_stability`` reports the relative eigengaps and the
agreement of the grouping under a column bootstrap (the same bytes, a resampled
set of sketch columns); ``byte_groups`` itself is unchanged. Same physics as the
"principal-axis swing" of a sample covariance when its top eigenvalues
approach each other: the orientation, not the spectrum, is what is unstable.

Rows are count-sketched (signed hash into SKETCH_COLS) so a trace of any size
costs a fixed row; a signed sketch keeps inner products unbiased, and the
static trace cancels at centring, so no baseline run is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SKETCH_COLS = 2048  # power of two: column = top bits of the hash
SKETCH_BITS = SKETCH_COLS.bit_length() - 1
MAX_ROWS = 256  # live bytes kept per seed (the pass is in offset order)
MIN_ROWS = 4  # centring costs one dof, the dropped component another
DROP_LEADING = 1  # components removed as clutter
MAX_COMPONENTS = 8  # kept after the drop
EIG_FLOOR = 1e-9  # eigenvalues below this fraction of the largest are noise
MIN_KEPT_ENERGY = 0.05  # row energy share outside the dropped components
MIN_REL_GAP = 0.05  # (w_k - w_k+1) / w_k below this: component k is degenerate
BOOT_RESAMPLES = 16  # column bootstraps in group_stability
MIN_AGREEMENT = 0.8  # median adjusted Rand vs the original grouping to call it stable

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SIGN_BIT = np.uint64(1) << np.uint64(20)


@dataclass(frozen=True)
class ByteGroups:
    """Grouped bytes of one seed: parallel arrays, offsets ascending.

    ``comp`` is the group id ``2 * component + pole`` (component 0-based after
    the drop, pole 1 for a negative score); ``weight`` the squared score on
    it, so a byte that is the group's best exemplar weighs most. ``count`` is
    the number of group ids (2 per kept component).
    """

    offsets: np.ndarray
    comp: np.ndarray
    weight: np.ndarray
    count: int


def sketch_row(edge_ids: np.ndarray) -> np.ndarray:
    """One trace as a signed count-sketch row (int16, SKETCH_COLS)."""
    h = edge_ids.astype(np.uint64) * _GOLDEN
    col = (h >> np.uint64(64 - SKETCH_BITS)).astype(np.intp)
    sign = np.where(h & _SIGN_BIT, 1.0, -1.0)
    row = np.bincount(col, weights=sign, minlength=SKETCH_COLS)
    return row.astype(np.int16)


def byte_groups(offsets: np.ndarray, rows: np.ndarray) -> ByteGroups | None:
    """Group *offsets* by the PCA components of their sketch *rows*.

    Args:
        offsets: byte offset of each row, ascending.
        rows: ``(n, SKETCH_COLS)`` sketch rows.

    Returns:
        ``None`` when the matrix cannot support a decomposition (too few or
        more than MAX_ROWS rows, no variance after centring, or nothing left
        after the drop).
    """
    fit = _fit(offsets, rows)
    return None if fit is None else fit[0]


def _fit(
    offsets: np.ndarray, rows: np.ndarray
) -> tuple[ByteGroups, np.ndarray, slice, np.ndarray] | None:
    """byte_groups plus what group_stability needs: the grouping, the sorted
    eigenvalues, the kept slice, and a per-row group label (-1 = not a member)."""
    n = rows.shape[0]
    if not MIN_ROWS <= n <= MAX_ROWS or n != offsets.size:
        return None

    y = rows.astype(np.float64)
    y -= y.mean(axis=0)
    if not y.any():
        return None

    # n <= MAX_ROWS << SKETCH_COLS, so the n x n Gram is the small side.
    w, u = np.linalg.eigh(y @ y.T)
    scores = u * np.sqrt(np.maximum(w, 0.0))
    order = np.argsort(w)[::-1]
    w, scores = np.maximum(w[order], 0.0), scores[:, order]

    live = int(np.count_nonzero(w > w[0] * EIG_FLOOR))
    # Only drop the leading component if there are enough remaining
    # components to preserve grouping resolution. If dropping would leave
    # only 1 component (which can split by at most pole sign, not distinct
    # fields), keep all components instead.
    if live - DROP_LEADING >= 2:
        keep = slice(DROP_LEADING, min(live, DROP_LEADING + MAX_COMPONENTS))
    else:
        keep = slice(0, live)

    if live <= DROP_LEADING:
        return None

    kept = scores[:, keep]
    k2 = kept * kept
    total = (scores[:, :live] ** 2).sum(axis=1)
    share = np.divide(k2.sum(axis=1), total, out=np.zeros(n), where=total > 0)

    # Largest kept |score| names the axis; its sign names the pole.
    rows_ix = np.arange(n)
    axis = k2.argmax(axis=1)
    pole = kept[rows_ix, axis] < 0
    comp = (2 * axis + pole).astype(np.int8)
    weight = k2[rows_ix, axis].astype(np.float32)

    member = (share >= MIN_KEPT_ENERGY) & (weight > 0)
    if not member.any():
        return None
    groups = ByteGroups(
        offsets=offsets[member].astype(np.uint32),
        comp=comp[member],
        weight=weight[member],
        count=2 * kept.shape[1],
    )
    labels = np.where(member, comp, -1).astype(np.int64)
    return groups, w, keep, labels


@dataclass(frozen=True)
class GroupStability:
    """How far the grouping of one seed can be trusted.

    ``gaps[j]`` is the relative gap ``(w_k - w_k+1) / w_k`` after kept component
    ``k = keep.start + j``; the last entry is the gap to the first component
    outside the kept block (the truncation edge). ``degenerate`` lists the j
    with a gap below MIN_REL_GAP among components that actually hold bytes (a
    noise tail is always near-degenerate and says nothing). ``agreement`` is
    the median adjusted Rand index between the original grouping and the
    groupings of ``resamples`` column bootstraps (a bootstrap that declines
    scores 0). ``stable`` follows ``agreement`` alone: the bootstrap is the
    test, the eigengap says where an instability would come from (planted
    fields of equal size are degenerate yet regroup identically, so a
    degenerate flag is a warning, not a veto).
    """

    gaps: np.ndarray
    degenerate: tuple[int, ...]
    agreement: float
    resamples: int

    @property
    def stable(self) -> bool:
        return self.agreement >= MIN_AGREEMENT


def _comb2(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float((x * (x - 1.0) / 2.0).sum())


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two label vectors; label values are arbitrary
    (so a pole flip or a component renumbering does not count as disagreement).
    Identical partitions score 1.0, including the trivial one-cluster case."""
    n = a.size
    if n != b.size:
        raise ValueError("label vectors differ in length")
    if n < 2:
        return 1.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(cont, (ai, bi), 1)
    sum_ij = _comb2(cont)
    sum_a = _comb2(cont.sum(axis=1))
    sum_b = _comb2(cont.sum(axis=0))
    total = n * (n - 1) / 2.0
    expected = sum_a * sum_b / total
    top = 0.5 * (sum_a + sum_b)
    if top == expected:  # both trivial (all together or all apart)
        return 1.0 if sum_ij == top else 0.0
    return (sum_ij - expected) / (top - expected)


def group_stability(
    offsets: np.ndarray, rows: np.ndarray, rand, resamples: int = BOOT_RESAMPLES
) -> GroupStability | None:
    """Eigengap and column-bootstrap check for ``byte_groups(offsets, rows)``.

    Args:
        offsets, rows: as for ``byte_groups``.
        rand: a ``RandPool`` (Hard Rule 16) drawing the bootstrap columns.
        resamples: number of column bootstraps.

    Returns:
        ``None`` exactly when ``byte_groups`` declines.
    """
    fit = _fit(offsets, rows)
    if fit is None:
        return None
    _, w, keep, labels = fit
    stop = keep.stop if keep.stop is not None else w.size
    hi = np.maximum(w[keep.start : stop], 0.0)
    nxt = w[keep.start + 1 : stop + 1]
    nxt = np.concatenate([nxt, np.zeros(hi.size - nxt.size)])  # past the end: 0
    gaps = np.divide(hi - nxt, hi, out=np.zeros(hi.size), where=hi > 0)
    used = np.unique(labels[labels >= 0] // 2)  # kept-component index of each member
    degenerate = tuple(int(j) for j in used if gaps[j] < MIN_REL_GAP)

    scores = np.zeros(max(resamples, 0))
    for r in range(scores.size):
        raw = np.frombuffer(rand.randbytes(2 * SKETCH_COLS), dtype=np.uint16)
        cols = (raw % SKETCH_COLS).astype(np.intp)  # SKETCH_COLS | 65536: unbiased
        boot = _fit(offsets, rows[:, cols])
        scores[r] = 0.0 if boot is None else adjusted_rand(labels, boot[3])
    agreement = float(np.median(scores)) if scores.size else 1.0
    return GroupStability(
        gaps=gaps, degenerate=degenerate, agreement=agreement, resamples=scores.size
    )
