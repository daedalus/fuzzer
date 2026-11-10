"""Ratio-driven bit flips (port of zzuf's ``-r`` mutation density).

zzuf flips each bit with probability ``r``. The flip count scales with
input length, unlike havoc's fixed stack depth. Per call here: draw ``r``
log-uniform in a bucket, flip ``Binomial(8n, r)`` bits.

    r:      1e-5     1e-4     1e-3     1e-2     1e-1
            |  e5    |  e4    |  e3    |  e2    |      one bandit arm each
    512 B:  0.04     0.4      4        41       410     expected flips

Positions are drawn with replacement, as zzuf does: a bit hit twice is not
flipped. ``MIN_FLIPS`` replaces zzuf's no-op draws (its dither can return 0)
with one flip, since an unchanged input wastes an exec.
"""

from __future__ import annotations

import math

import numpy as np

MIN_FLIPS = 1
BITS_PER_BYTE = 8
BIT_SHIFT_BYTE = 3
BIT_MASK_IN_BYTE = 7

# One decade per arm, so the scheduler learns the density per target.
RATIO_BUCKETS: dict[str, tuple[float, float]] = {
    "ratio_flip_e5": (1e-5, 1e-4),
    "ratio_flip_e4": (1e-4, 1e-3),
    "ratio_flip_e3": (1e-3, 1e-2),
    "ratio_flip_e2": (1e-2, 1e-1),
}


def ratio_flip(data: bytes, rng, lo: float, hi: float) -> bytes | None:
    """Flip ``Binomial(8n, r)`` bits, ``r`` log-uniform in ``[lo, hi)``."""
    if not data:
        return None

    nbits = len(data) * BITS_PER_BYTE
    log_lo = math.log(lo)
    ratio = math.exp(log_lo + (math.log(hi) - log_lo) * rng.random())

    drawn = int(rng.binomial_array([nbits], ratio)[0])
    count = max(MIN_FLIPS, drawn)
    pos = np.asarray(rng.randint_list(0, nbits - 1, count), dtype=np.int64)

    # ufunc.at applies repeats one by one, so duplicate bits cancel.
    out = np.frombuffer(data, dtype=np.uint8).copy()
    mask = (1 << (pos & BIT_MASK_IN_BYTE)).astype(np.uint8)
    np.bitwise_xor.at(out, pos >> BIT_SHIFT_BYTE, mask)
    return out.tobytes()
