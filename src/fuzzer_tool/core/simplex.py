"""Batched projection of swarm positions onto the operator simplex.

Shared by ``MOptScheduler`` (PSO) and ``FireflyScheduler``: one row per
particle/firefly, one column per operator. Same sequence the per-row list
code used to run::

    clip negatives -> divide by row sum -> floor at frac/n -> renormalize

    e.g. n=4, frac=0.1: [0.9, 0.1, 0.0, 0.0] -> floor 0.025 -> [0.88, 0.1, 0.024, 0.024]
"""

import numpy as np


def project_rows(pos: np.ndarray, min_prob_frac: float) -> np.ndarray:
    """Project each row of ``pos`` (shape ``(rows, n)``) onto the simplex."""
    n = pos.shape[1]
    if n == 0:
        return pos

    clipped = np.maximum(pos, 0.0)
    totals = clipped.sum(axis=1, keepdims=True)

    # Rows with no positive mass fall back to uniform.
    positive = totals > 0.0
    out = np.where(positive, clipped / np.where(positive, totals, 1.0), 1.0 / n)

    floor = min_prob_frac / n
    if floor <= 0.0:
        return out

    out = np.maximum(out, floor)
    return out / out.sum(axis=1, keepdims=True)
