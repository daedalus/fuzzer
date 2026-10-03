"""Sort-based set operations on 1-D numpy arrays.

numpy 2.4's ``np.unique`` takes a hash path that costs 6.1 s on 4.9M int64
keys; sort + adjacent-diff gives the same result in 0.09 s. The ICFG build
and the Katz recompute run it on millions of packed edge keys.
"""

import numpy as np


def sorted_unique(a: np.ndarray) -> np.ndarray:
    """Same as ``np.unique(a)`` for a 1-D array: sorted, deduplicated, same dtype."""
    s = np.sort(a)
    if len(s) == 0:
        return s

    keep = np.empty(len(s), dtype=bool)
    keep[0] = True
    np.not_equal(s[1:], s[:-1], out=keep[1:])
    return s[keep]


def sorted_isin(keys: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``np.isin(keys, ref)`` for a sorted, deduplicated *ref*."""
    if len(ref) == 0:
        return np.zeros(len(keys), dtype=bool)

    pos = np.minimum(np.searchsorted(ref, keys), len(ref) - 1)
    return ref[pos] == keys
