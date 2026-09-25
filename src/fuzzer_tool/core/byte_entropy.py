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
from typing import Any, cast

try:  # pragma: no cover - exercised by whichever branch is installed
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

#: Bits per byte at maximum entropy; the divisor taking bits to percent.
MAX_BITS_PER_BYTE = 8.0

#: Bytes sampled from the head of an input. Matches report._corpus_byte_entropy.
ENTROPY_SAMPLE_CAP = 4096

#: Pseudo-counts per bin in CumulativeByteEntropy.freq_dist. 1/256 puts one
#: pseudo-byte in total across the alphabet, so the floor that keeps a KL
#: finite is worth a single byte of the pool and vanishes as it grows.
POOL_SMOOTHING = 1.0 / 256

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


#: A 256-bin byte tally: a numpy integer array when numpy is installed, a
#: list of ints otherwise. Both index by byte value; ``_as_bins`` is the
#: one place that has to care which.
ByteCounts = Any


def _as_bins(counts: ByteCounts) -> list[int]:
    """Byte counts as a plain list, whichever branch produced them."""
    return cast(list[int], counts.tolist() if hasattr(counts, "tolist") else counts)


def byte_histogram(data: bytes, cap: int = ENTROPY_SAMPLE_CAP) -> tuple[ByteCounts, int]:
    """256-bin byte counts over ``data[:cap]``, and the bytes counted.

    A numpy array when numpy is installed, a list otherwise; both index by
    byte value. Split out of :func:`byte_entropy_bits` so a caller that
    needs the distribution itself -- a KL against a pooled corpus
    distribution, the cumulative tracker's fold -- gets it from the same
    single scan that yields the entropy instead of scanning twice.
    """
    chunk = bytes(data[:cap])
    if not chunk:
        return (np.zeros(256, dtype=np.int64) if _HAS_NUMPY else [0] * 256), 0
    if _HAS_NUMPY:
        return np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256), len(chunk)

    counts = [0] * 256
    for value, count in Counter(chunk).items():
        counts[value] = count
    return counts, len(chunk)


def entropy_bits_from_counts(counts: ByteCounts, total: int) -> float:
    """Shannon entropy in bits/byte of a distribution given as byte counts.

    ``total`` is the number of bytes tallied, which bounds every count --
    that is what keeps ``c log2 c`` a table lookup. Callers pooling a whole
    corpus want :meth:`CumulativeByteEntropy.bits` instead, whose total is
    unbounded. 0.0 on an empty distribution.
    """
    if total <= 0:
        return 0.0
    if _HAS_NUMPY:
        global _C_LOG2_C
        if total >= _C_LOG2_C.size:
            _C_LOG2_C = _build_c_log2_c(total)
        acc = float(_C_LOG2_C[counts].sum())
    else:
        acc = 0.0
        for count in counts:
            if count:
                acc += count * math.log2(count)
    ent = math.log2(total) - acc / total
    # A single-symbol input sums to -0.0; clamp so callers comparing against
    # zero and formatting the value never see a negative zero.
    return ent if ent > 0.0 else 0.0


def byte_entropy_bits(data: bytes, cap: int = ENTROPY_SAMPLE_CAP) -> float:
    """Shannon entropy of ``data``'s byte distribution, in bits/byte.

    Returns 0.0 for empty input. Only the first ``cap`` bytes are read.
    """
    if not data:
        return 0.0
    return entropy_bits_from_counts(*byte_histogram(data, cap))


def byte_entropy_pct(data: bytes, cap: int = ENTROPY_SAMPLE_CAP) -> float:
    """Byte entropy of ``data`` on the 0-100 scale ``SeedScorer`` expects.

    Empty input scores 0.0, not the -1.0 "unknown" sentinel: an empty seed
    genuinely has no entropy, and 0.0 lands it in the sparse-input branch
    where it belongs. Callers with no data at all should pass -1.0
    themselves rather than routing through here.
    """
    return byte_entropy_bits(data, cap) / MAX_BITS_PER_BYTE * 100.0


def _h_bits(probs: Any) -> float:
    """Shannon entropy in bits of a numpy probability vector."""
    nz = probs[probs > 0]
    return -float(np.dot(nz, np.log2(nz)))


class CumulativeByteEntropy:
    """Running (aggregate) Shannon entropy of every seed folded in so far.

    ``report._corpus_byte_entropy`` computes this same pooled-distribution
    metric (treat every seed's bytes as one shared alphabet, not an
    average of per-seed entropies) but rescans the whole corpus from
    scratch each time it's called -- O(total corpus bytes) on every status
    tick or report. This keeps 256 running byte-frequency counts instead,
    so ``add()`` is O(cap) in the one seed just read and ``bits()`` is
    O(1), the same incremental-over-rescan trade this codebase already
    makes elsewhere (EdgeTracker, RunningMoments, the Kalman EPS filter).

    Intended to be fed once per seed at the point its bytes are actually
    read from disk (corpus load, resume, delta reconstruction) rather than
    recomputed later from the in-memory corpus list.
    """

    __slots__ = ("_freq", "_total")

    def __init__(self) -> None:
        self._freq = [0] * 256
        self._total = 0

    def add(self, data: bytes, cap: int = ENTROPY_SAMPLE_CAP) -> float:
        """Fold one seed's (capped) bytes into the running totals.

        Returns that seed's own Shannon entropy in bits/byte, so a caller
        reading a seed from disk gets the per-seed figure for free instead
        of scanning the same bytes twice.
        """
        counts, total = byte_histogram(data, cap)
        if not total:
            return 0.0
        self._fold(counts, total)
        return entropy_bits_from_counts(counts, total)

    def remove(self, data: bytes, cap: int = ENTROPY_SAMPLE_CAP) -> None:
        """Unfold one seed's (capped) bytes from the running totals.

        The inverse of :meth:`add`, for a caller whose pool is a *live* set
        rather than an append-only stream: a seed the corpus prunes has to
        leave the pooled distribution too, or the counts only ever climb
        and keep crediting bytes to seeds that no longer exist -- what
        ``_edge_owner_count`` did before ``_maybe_prune`` rebuilt it from
        the survivors.

        Removing bytes that were never added is a caller bug; counts floor
        at zero rather than going negative, so one bad call cannot poison
        every later read.
        """
        counts, total = byte_histogram(data, cap)
        if total:
            self._unfold(counts, total)

    def _fold(self, counts: ByteCounts, total: int) -> None:
        """Add one histogram to the running totals."""
        freq = self._freq
        # One .tolist() beats 256 numpy-scalar unboxings, and the clamp
        # lives in _unfold rather than here: measured together, 167us ->
        # 29us per 4 KiB fold.
        for value, count in enumerate(_as_bins(counts)):
            freq[value] += count
        self._total += total

    def _unfold(self, counts: ByteCounts, total: int) -> None:
        """Subtract one histogram, flooring at zero."""
        freq = self._freq
        for value, count in enumerate(_as_bins(counts)):
            if count:
                freq[value] = max(0, freq[value] - count)
        self._total = max(0, self._total - total)

    def freq_dist(self, smoothing: float = POOL_SMOOTHING) -> tuple[float, ...]:
        """The pooled byte distribution as 256 probabilities.

        ``smoothing`` pseudo-counts go into every bin before normalising,
        so a byte value the pool has never seen keeps positive probability
        and a KL or cross-entropy against this distribution stays finite.
        The default spreads exactly one pseudo-byte across the 256 bins
        rather than Laplace's add-one, which injects 256 pseudo-bytes and
        would dominate any pool smaller than a few kilobytes.

        Uniform before anything has been added.
        """
        denom = self._total + smoothing * 256
        if denom <= 0:
            return tuple([1.0 / 256] * 256)
        return tuple((count + smoothing) / denom for count in self._freq)

    def bits(self) -> float:
        """Aggregate Shannon entropy, in bits/byte, of every seed added so far.

        0.0 before anything has been added -- same empty-input convention
        as ``byte_entropy_bits``.
        """
        if self._total <= 0:
            return 0.0
        total = self._total
        ent = 0.0
        for count in self._freq:
            if count:
                pr = count / total
                ent -= pr * math.log2(pr)
        return ent if ent > 0.0 else 0.0

    def copy(self) -> CumulativeByteEntropy:
        """Independent snapshot of the running totals."""
        clone = CumulativeByteEntropy()
        clone._freq = list(self._freq)
        clone._total = self._total
        return clone

    def js_bits(self, other: CumulativeByteEntropy) -> float:
        """Jensen-Shannon divergence to ``other``, in bits, bounded to [0, 1].

        Chosen over KL because the midpoint M = (P+Q)/2 covers both
        supports: a byte value only one side has stays finite without
        smoothing. 0.0 when either side is empty.
        """
        n_p, n_q = self._total, other._total
        if n_p <= 0 or n_q <= 0:
            return 0.0

        # JS = H(M) - (H(P) + H(Q)) / 2, with M = (P + Q) / 2.
        # numpy: 28 us vs 51 us for the loop below.
        if _HAS_NUMPY:
            p = np.asarray(self._freq, dtype=np.float64) / n_p
            q = np.asarray(other._freq, dtype=np.float64) / n_q
            js = _h_bits((p + q) / 2) - (_h_bits(p) + _h_bits(q)) / 2
            return js if js > 0.0 else 0.0

        h_m = 0.0
        for c_p, c_q in zip(self._freq, other._freq, strict=True):
            m = (c_p / n_p + c_q / n_q) / 2
            if m > 0.0:
                h_m -= m * math.log2(m)
        js = h_m - (self.bits() + other.bits()) / 2
        return js if js > 0.0 else 0.0

    def novel_mass(self, other: CumulativeByteEntropy) -> float:
        """Fraction of this pool's bytes on values ``other`` never holds.

        The rare-new-value signal JS dilutes by its 1/2 weighting.
        0.0 when this pool is empty.
        """
        if self._total <= 0:
            return 0.0
        novel = sum(c for c, o in zip(self._freq, other._freq, strict=True) if not o)
        return novel / self._total

    def __len__(self) -> int:
        """Total (capped) bytes folded in so far, across all seeds added."""
        return self._total
