"""(edge, hit-count) items for count-aware corpus minimization.

Set cover over bare edge ids keeps one seed per edge set, but the fuzzer's
virgin map treats a new AFL count bucket on a known edge as new coverage.
``minimize -c --count-aware`` covers these items instead:

    BUCKET  (edge, AFL bucket): one item per edge, matching what the virgin
            map distinguishes (1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+).
    THERMO  (edge, level k):    one item per k with count >= 2^k, k <= 7.
            Nested, so a seed with a higher count covers every lower level
            of that edge; keeps fewer seeds than BUCKET.

    edge 10 hit 5 times   BUCKET: {(10, 4-7)}   THERMO: {(10,0) (10,1) (10,2)}

Items are uint64 (edge id << 8 | tag), sorted and unique, so equal coverage
gives equal ``tobytes()``. Edge ids may use all 32 bits.
"""

from enum import Enum

import numpy as np

from fuzzer_tool.core.count_class import bucket_bits

THERMO_LEVELS = 8
_TAG_BITS = 8
_BITLEN = np.array([int(v).bit_length() for v in range(256)], dtype=np.int64)


class CountItems(Enum):
    OFF = "off"
    BUCKET = "bucket"
    THERMO = "thermo"


def _bucket_items(ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return (ids << np.uint64(_TAG_BITS)) | bucket_bits(counts).astype(np.uint64)


def _thermo_items(ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
    levels = np.minimum(_BITLEN[np.minimum(counts, 255)], THERMO_LEVELS)
    starts = np.cumsum(levels) - levels
    level_idx = np.arange(int(levels.sum()), dtype=np.int64) - np.repeat(starts, levels)
    return (np.repeat(ids, levels) << np.uint64(_TAG_BITS)) | level_idx.astype(np.uint64)


_BUILDERS = {CountItems.BUCKET: _bucket_items, CountItems.THERMO: _thermo_items}


def count_items(ids: np.ndarray, counts: np.ndarray, mode: CountItems) -> np.ndarray:
    """Sorted unique items for one seed; ``OFF`` returns *ids* unchanged."""
    if mode is CountItems.OFF:
        return ids
    if ids.shape != counts.shape:
        raise ValueError("ids and counts differ in length")

    live = counts > 0
    items = _BUILDERS[mode](ids[live].astype(np.uint64), counts[live].astype(np.int64))
    return np.unique(items)
