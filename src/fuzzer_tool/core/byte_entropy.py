"""Shannon entropy of an input's byte distribution.

Separate from ``EdgeTracker.shannon_entropy_seed``, which measures the
entropy of a seed's *coverage hit-count* distribution. This module measures
the entropy of the seed's *bytes*, which is what the honggfuzz energy
factor in :mod:`core.schedules` is written against: it separates random or
already-compressed blobs (near 8 bits/byte) from text and structured
formats (~4-5) and from sparse, mostly-zero inputs (~0-2).

The percentage scale is fixed by the thresholds in
``SeedScorer._honggfuzz_factors``: 93, 62 and 25 correspond to 7.44, 4.96
and 2.00 bits/byte under ``pct = bits / 8 * 100``, which is where random,
structured-text and near-zero inputs actually land. Any other scale puts
those three cut points somewhere the comments there do not describe.

The 4096-byte cap matches ``report._corpus_byte_entropy``. Byte entropy
converges long before that (a 4 KiB sample of a 1 MiB input is within
noise of the full scan), and the cap is what keeps the cost flat in seed
size rather than linear.
"""

from __future__ import annotations

import math
from collections import Counter

try:  # pragma: no cover - exercised by whichever branch is installed
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

#: Bits per byte at maximum entropy; the divisor taking bits to percent.
MAX_BITS_PER_BYTE = 8.0

#: Bytes sampled from the head of an input. Matches report._corpus_byte_entropy.
ENTROPY_SAMPLE_CAP = 4096

# Entropy is computed in its counts form rather than its probability form:
#
#     H = -sum_i (c_i/N) log2(c_i/N)
#       = -(1/N) sum_i c_i (log2 c_i - log2 N)
#       = log2(N) - (1/N) sum_i c_i log2(c_i)
#
# The rearrangement keeps the counts as integers all the way through. That
# removes the probability array (an allocation and a division per distinct
# byte value) and, more usefully, makes the only transcendental term a
# function of an integer bounded by the sample cap -- so it is a table
# lookup. c log2 c is 0 at c = 0, which is the same convention the masked
# probability form encoded by dropping zero counts, so no mask is needed
# either.
if _HAS_NUMPY:

    def _build_c_log2_c(size: int) -> np.ndarray:
        c = np.arange(size + 1, dtype=np.float64)
        table = np.zeros(size + 1, dtype=np.float64)
        np.log2(c[1:], out=table[1:])
        table[1:] *= c[1:]
        return table

    #: c * log2(c) for c in [0, cap]. Sized for the default cap; callers
    #: passing a larger one grow it once rather than falling off the end.
    _C_LOG2_C = _build_c_log2_c(ENTROPY_SAMPLE_CAP)


def byte_entropy_bits(data: bytes, cap: int = ENTROPY_SAMPLE_CAP) -> float:
    """Shannon entropy of ``data``'s byte distribution, in bits/byte.

    Returns 0.0 for empty input. Only the first ``cap`` bytes are read.
    """
    if not data:
        return 0.0
    chunk = bytes(data[:cap])
    if not chunk:
        return 0.0
    if _HAS_NUMPY:
        global _C_LOG2_C
        arr = np.frombuffer(chunk, dtype=np.uint8)
        if arr.size >= _C_LOG2_C.size:
            _C_LOG2_C = _build_c_log2_c(arr.size)
        counts = np.bincount(arr, minlength=256)
        ent = math.log2(arr.size) - float(_C_LOG2_C[counts].sum()) / arr.size
    else:
        total = len(chunk)
        acc = 0.0
        for count in Counter(chunk).values():
            acc += count * math.log2(count)
        ent = math.log2(total) - acc / total
    # A single-symbol input sums to -0.0; clamp so callers comparing against
    # zero and formatting the value never see a negative zero.
    return ent if ent > 0.0 else 0.0


def byte_entropy_pct(data: bytes, cap: int = ENTROPY_SAMPLE_CAP) -> float:
    """Byte entropy of ``data`` on the 0-100 scale ``SeedScorer`` expects.

    Empty input scores 0.0, not the -1.0 "unknown" sentinel: an empty seed
    genuinely has no entropy, and 0.0 lands it in the sparse-input branch
    where it belongs. Callers with no data at all should pass -1.0
    themselves rather than routing through here.
    """
    return byte_entropy_bits(data, cap) / MAX_BITS_PER_BYTE * 100.0
