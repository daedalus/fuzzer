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
    keep = slice(DROP_LEADING, min(live, DROP_LEADING + MAX_COMPONENTS))
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
    return ByteGroups(
        offsets=offsets[member].astype(np.uint32),
        comp=comp[member],
        weight=weight[member],
        count=2 * kept.shape[1],
    )
