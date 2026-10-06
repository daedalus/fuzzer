"""Maximum-sum contiguous run (Kadane's problem), vectorized.

Given signed scores ``x[0..n)``, find the half-open window ``[i, j)`` whose sum
is largest. The scan form (keep a running sum, restart when it goes negative)
is a Python loop; the prefix-sum form is the same answer in three numpy calls::

    S[k]   = x[0] + ... + x[k-1]            S[0] = 0
    best_j = max over j of S[j] - min(S[0..j-1])      (window must be non-empty)

Measured against the loop (4096 elements): ~43 us vs ~450 us; at 64 elements the
two are within 1.4x of each other, so there is no small-input case for the loop.

Used by ``core/schedulers/pos_kadane.py`` to turn per-bin excess-gain scores
into a contiguous byte window.
"""

from __future__ import annotations

import numpy as np


def max_subarray(x: np.ndarray) -> tuple[int, int, float] | None:
    """``(start, end, total)`` of the maximum-sum non-empty window of *x*.

    ``end`` is exclusive. Ties go to the earliest end, then the *latest* start,
    i.e. the tightest of the equal-sum windows: zero-score elements at either
    edge are left out, so a gap of exactly-zero scores is bridged only when it
    sits between two scoring elements. ``None`` for an empty input. NaN is not supported: it would propagate through ``cumsum`` and make
    ``argmax`` return the first NaN, so callers must pass finite scores.
    """
    n = len(x)
    if n == 0:
        return None

    s = np.empty(n + 1, dtype=np.float64)
    s[0] = 0.0
    np.cumsum(x, out=s[1:])

    lo = np.minimum.accumulate(s[:-1])  # lo[j-1] = min(S[0..j-1])
    gain = s[1:] - lo
    j = int(gain.argmax())
    head = s[: j + 1]
    i = j - int(head[::-1].argmin())  # last occurrence of the minimum
    return i, j + 1, float(gain[j])
