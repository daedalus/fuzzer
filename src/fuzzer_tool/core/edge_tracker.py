"""Edge tracker for per-seed coverage tracking.

Tracks which coverage edges each seed contributes, enabling the fuzzer
to deprioritize seeds whose coverage is fully subsumed by others.
Also tracks per-seed hit-count distributions for JS divergence and
Wasserstein distance-based diversity scoring.

Wasserstein distance on edge indices treats the edge map as a 1D metric
space — two seeds hitting adjacent edges are "close" even if they share
no edges, while two seeds hitting the same number of edges at opposite
ends of the map are "far". This captures coverage spatial diversity that
Jaccard (set overlap) and JS (frequency divergence) miss.
"""

import heapq
import logging
import math
import random
import struct
import time
import zlib
from array import array
from collections import defaultdict

from fuzzer_tool.core import fast_json as json
from fuzzer_tool.core.crc32 import crc32_ieee

# ── Memory bounds ────────────────────────────────────────────────────
# Below this many doubleton edges the classic Chao2 ratio Q1^2/(2*Q2) swings on
# a single edge changing owner count, so use the bias-corrected form instead.
_CHAO_BIAS_CORRECT_BELOW = 10

# Fraction of the coverage clock that counts as the discovery frontier in
# compute_coverage_proximity().
_FRONTIER_FRACTION = 0.25

CORRELATION_MATRIX_MAX = 10_000  # max edge-pair entries in branch correlation
COVERAGE_TIMELINE_MAX = 1_000  # max snapshots in coverage timeline

# Tracked-seed ceiling, and the fraction of it a prune drops back to.
#
# The ceiling exists to bound memory: seed_edges plus its eight companion maps
# cost, measured, 95 KiB per tracked seed on a png_read-shaped target (~500
# live edges) and 592 KiB on an ffmpeg_read-shaped one (8,189). At 1,000 seeds
# that is 93 MiB and 578 MiB respectively. The 200,000 this replaces was not a
# bound at all -- it works out to 113 GiB on the ffmpeg figure, which is to say
# the tracker grew without limit and _maybe_prune never fired.
#
# The low-water mark is what makes the ceiling affordable. Pruning back to the
# ceiling exactly leaves the tracker one insertion over it again immediately,
# so the O(seeds x edges) owner-count pass ran on *every* record_edges call to
# evict a single seed; measured at 18x the cost of not pruning at all. Dropping
# to 90% amortises that pass over the next ~10% of the ceiling in insertions.
#
# The batch is floored by truncation, not by max(1, ...), so a ceiling under 10
# gets a batch of zero and keeps the exact prune-to-the-ceiling behaviour.
# Hysteresis engages only where it buys something, and the small ceilings that
# tests use to force pruning in a handful of insertions keep their semantics
# rather than being pruned to half.
MAX_TRACKED_SEEDS = 1_000
PRUNE_BATCH_FRAC = 0.1

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from fuzzer_tool.core.count_class import classify_counts  # noqa: E402
from fuzzer_tool.core.elf import (  # noqa: E402
    MAP_SIZE_DEFAULT,
    _map_size_max,
)


def _clamp_entropy(value: float) -> float:
    """Pin a counts-form entropy to its non-negative range.

    ``log2(T) - (1/T) sum c log2 c`` is exact in exact arithmetic, but it
    is a difference of two quantities of size log2(T), so the absolute
    error is eps*log2(T) rather than eps*H. Everywhere but H = 0 that is
    far below anything a caller can see; at H = 0 it is the whole value,
    and a seed covering one edge can come back at -4e-16. Entropy is
    non-negative by definition, so clamping restores a property the
    probability form got for free rather than hiding an error.
    """
    return value if value > 0.0 else 0.0


def _sig_np(sig):
    """Zero-copy uint64 view of a MinHash signature (array('Q') or list)."""
    if isinstance(sig, array) and array("Q").itemsize == 8:
        return np.frombuffer(sig, dtype=np.uint64)
    return np.asarray(sig, dtype=np.uint64)


def _sig_matches(sig_a, sig_b) -> int:
    """Count equal positions (zip(strict=False) semantics), vectorized.

    Legacy truncates to the shorter signature; numpy would raise on a
    mismatch, so truncate explicitly and never raise.
    """
    if not _HAS_NUMPY:
        return sum(1 for a, b in zip(sig_a, sig_b, strict=False) if a == b)
    a = _sig_np(sig_a)
    b = _sig_np(sig_b)
    k = min(len(a), len(b))
    return int(np.count_nonzero(a[:k] == b[:k]))


log = logging.getLogger(__name__)

MORRIS_A = 30


# ── Pure-Python replacements for scipy ────────────────────────────────


def _norm_cdf(x: float, loc: float = 0.0, scale: float = 1.0) -> float:
    """Normal cumulative distribution function (replaces scipy.stats.norm.cdf).

    Uses the math.erf approximation: Φ(x) = 0.5 * (1 + erf((x - loc) / (scale * √2))).
    Max error ≈ 1.5e-7, more than sufficient for probability estimates.
    """
    if scale <= 0:
        return 1.0 if x >= loc else 0.0
    z = (x - loc) / (scale * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


class _LevenbergMarquardtResult:
    """Minimal result container matching scipy.optimize.least_squares interface."""

    __slots__ = ("x", "cost", "jac")

    def __init__(self, x, cost, jac):
        self.x = x
        self.cost = cost
        self.jac = jac


def _levenberg_marquardt(
    residuals,
    p0,
    bounds,
    max_nfev: int = 200,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
):
    """Levenberg-Marquardt nonlinear least squares with box constraints.

    Replaces scipy.optimize.least_squares for the specific use case in
    bayesian_coverage_growth_model(). Returns a result with .x, .cost, .jac.

    Args:
        residuals: Callable(p) -> 1D array of residuals.
        p0: Initial parameter guess (numpy array).
        bounds: (lower_bounds, upper_bounds) as two numpy arrays.
        max_nfev: Maximum function evaluations.
        xtol: Parameter convergence tolerance.
        ftol: Cost convergence tolerance.
    """
    lo, hi = bounds
    p = np.array(p0, dtype=np.float64)
    lo = np.array(lo, dtype=np.float64)
    hi = np.array(hi, dtype=np.float64)

    # Clamp initial guess
    p = np.clip(p, lo, hi)

    lam = 1e-3
    nfev = 0

    def _fd_jacobian(func, p, h=1e-8):
        """Central-difference Jacobian."""
        n = len(p)
        r0 = np.asarray(func(p), dtype=np.float64)
        m = len(r0)
        J = np.empty((m, n), dtype=np.float64)
        for j in range(n):
            p_hi = p.copy()
            p_lo = p.copy()
            p_hi[j] += h
            p_lo[j] -= h
            p_hi = np.clip(p_hi, lo, hi)
            p_lo = np.clip(p_lo, lo, hi)
            J[:, j] = (
                np.asarray(func(p_hi), dtype=np.float64) - np.asarray(func(p_lo), dtype=np.float64)
            ) / max(p_hi[j] - p_lo[j], h)
        return J, r0

    J, r = _fd_jacobian(residuals, p)
    nfev += 1 + 2 * len(p)
    cost = float(r @ r)

    for _iter in range(max_nfev - nfev):
        # Normal equations: (J^T J + λ diag(J^T J)) Δp = -J^T r
        JtJ = J.T @ J
        diagJtJ = np.diag(JtJ) + 1e-12
        A = JtJ + lam * np.diag(diagJtJ)
        b = -(J.T @ r)

        try:
            dp = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        # Step with gain ratio
        p_new = np.clip(p + dp, lo, hi)
        r_new = np.asarray(residuals(p_new), dtype=np.float64)
        nfev += 1
        cost_new = float(r_new @ r_new)

        actual_loss = cost - cost_new
        pred_loss = float(dp @ (lam * np.diag(JtJ) * dp - J.T @ r))
        # Clamp predicted gain to avoid division issues
        if abs(pred_loss) < 1e-15:
            gain_ratio = 1.0 if actual_loss >= 0 else 0.0
        else:
            gain_ratio = actual_loss / pred_loss

        if gain_ratio > 0:
            p = p_new
            r = r_new
            cost = cost_new
            lam *= max(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3)
            lam = max(lam, 1e-15)
            # Recheck Jacobian periodically
            if _iter % 5 == 0:
                J, _ = _fd_jacobian(residuals, p)
                nfev += 2 * len(p)
        else:
            lam *= 2.0
            lam = min(lam, 1e10)

        # Convergence checks
        if np.max(np.abs(dp)) < xtol:
            break
        if actual_loss > 0 and abs(actual_loss) < ftol * cost:
            break

    # Final Jacobian for covariance estimation
    J, _ = _fd_jacobian(residuals, p)
    return _LevenbergMarquardtResult(x=p, cost=cost, jac=J)


def morris_estimate(v: int) -> float:
    """Convert Morris counter value to approximate count.

    estimate(v) = a * ((1 + 1/a)^v - 1)
    """
    if v == 0:
        return 0.0
    return MORRIS_A * ((1.0 + 1.0 / MORRIS_A) ** v - 1.0)


def ks_two_sample(samples_a: list[float], samples_b: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov test.

    Computes the KS statistic D and its p-value using the asymptotic
    Kolmogorov distribution. Works for any sample sizes — the p-value
    naturally tightens as more data accumulates.

    Args:
        samples_a: Observations from sample A.
        samples_b: Observations from sample B.

    Returns:
        (D, p_value) where D ∈ [0,1] and p_value ∈ [0,1].
        Low p_value (< 0.05) indicates the samples come from different distributions.
    """
    if not samples_a or not samples_b:
        return 0.0, 1.0

    a = sorted(samples_a)
    b = sorted(samples_b)
    n, m = len(a), len(b)

    # Walk merged sorted values, tracking empirical CDF jumps
    i = j = 0
    d = 0.0
    fi = fj = 0.0

    while i < n and j < m:
        if a[i] < b[j]:
            fi = (i + 1) / n
            d = max(d, abs(fi - fj))
            i += 1
        elif a[i] > b[j]:
            fj = (j + 1) / m
            d = max(d, abs(fi - fj))
            j += 1
        else:
            # Tie: advance both (CDFs jump at the same point)
            fi = (i + 1) / n
            fj = (j + 1) / m
            d = max(d, abs(fi - fj))
            i += 1
            j += 1

    # Check remaining elements
    while i < n:
        fi = (i + 1) / n
        d = max(d, abs(fi - fj))
        i += 1
    while j < m:
        fj = (j + 1) / m
        d = max(d, abs(fi - fj))
        j += 1

    # P-value from asymptotic Kolmogorov distribution
    p = _kolmogorov_pvalue(d, n, m)
    return d, p


def _kolmogorov_pvalue(d: float, n: int, m: int) -> float:
    """P-value for two-sample KS test using asymptotic Kolmogorov distribution.

    Uses the series: P(D >= d) = 2 * sum_{k=1}^{inf} (-1)^{k-1} exp(-2 k^2 lambda^2)
    where lambda = d * sqrt(n*m / (n+m)).
    Converges rapidly — 20 terms suffice for all practical values.
    """
    if d <= 0:
        return 1.0
    if d >= 1.0:
        return 0.0

    # Effective sample size
    nm = n * m / (n + m)
    lam = d * math.sqrt(nm)
    lam2 = lam * lam

    # Series converges fast; 20 terms covers everything
    p = 0.0
    for k in range(1, 21):
        term = ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam2)
        p += term
    p = max(0.0, min(1.0, 2.0 * p))
    return p


def _ks_p_from_cdf_diff(max_cdf_diff: float, n_samples: int) -> float:
    """P-value for one-sample KS test against a fully specified distribution.

    Uses Kolmogorov distribution directly: P(D >= d) = 2 * sum exp(-2 k^2 n d^2).
    """
    if max_cdf_diff <= 0 or n_samples <= 0:
        return 1.0
    nd = n_samples * max_cdf_diff * max_cdf_diff * 2.0
    p = 0.0
    for k in range(1, 21):
        p += ((-1) ** (k - 1)) * math.exp(-k * k * nd)
    return max(0.0, min(1.0, 2.0 * p))


def ks_significance_threshold(n_samples: int, alpha: float = 0.05) -> float:
    """Minimum KS D-statistic needed for significance at level alpha with n samples.

    Inverts the Kolmogorov distribution to find the critical value.
    For small n, the threshold is high (need large D to be significant).
    For large n, it drops (subtle differences become detectable).

    This replaces fixed magnitude thresholds with sample-size-aware ones:
    instead of "JS < 0.01 → plateau", use "JS-equivalent D < threshold(n) → plateau".
    """
    if n_samples <= 0:
        return 1.0
    # Asymptotic: D_crit ≈ c(alpha) / sqrt(n), where c(alpha) is the Kolmogorov critical value
    # c(0.05) ≈ 1.358, c(0.01) ≈ 1.628, c(0.10) ≈ 1.224
    _crit_values = {0.01: 1.628, 0.05: 1.358, 0.10: 1.224, 0.20: 1.073}
    c = _crit_values.get(alpha, 1.358)
    return c / math.sqrt(n_samples)


def normalized_compression_distance(x: bytes, y: bytes) -> float:
    """Normalized Compression Distance — proxy for Kolmogorov similarity.

    NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))

    Where C is compressed size via zlib. Values near 0 mean x and y are
    algorithmically similar (share structure). Values near 1 mean unrelated.

    Note: noisy on small inputs (< ~200 bytes) due to zlib header overhead.
    Gate calls with a minimum-size check for reliable results.
    """
    if not x or not y:
        return 1.0
    cx = len(zlib.compress(x, 9))
    cy = len(zlib.compress(y, 9))
    cxy = len(zlib.compress(x + y, 9))
    denom = max(cx, cy)
    if denom == 0:
        return 0.0
    return max(0.0, (cxy - min(cx, cy)) / denom)


def _js_divergence(p: dict[int, float], q: dict[int, float]) -> float:
    """Compute Jensen-Shannon divergence between two discrete distributions.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q).

    Both p and q are sparse dicts mapping event -> probability.
    Returns a value in [0, ln(2)] where 0 means identical distributions.
    """
    all_keys = set(p) | set(q)
    if _HAS_NUMPY and len(all_keys) > 50:
        keys = np.array(sorted(all_keys), dtype=np.int64)
        p_arr = np.array([p.get(int(k), 0.0) for k in keys], dtype=np.float64)
        q_arr = np.array([q.get(int(k), 0.0) for k in keys], dtype=np.float64)
        m = 0.5 * (p_arr + q_arr)
        valid_p = (p_arr > 0.0) & (m > 0.0)
        valid_q = (q_arr > 0.0) & (m > 0.0)
        kl_pm = float(np.sum(p_arr[valid_p] * np.log(p_arr[valid_p] / m[valid_p])))
        kl_qm = float(np.sum(q_arr[valid_q] * np.log(q_arr[valid_q] / m[valid_q])))
        return 0.5 * kl_pm + 0.5 * kl_qm

    m: dict[int, float] = {}
    for k in all_keys:
        m[k] = 0.5 * (p.get(k, 0.0) + q.get(k, 0.0))

    def _kl(a: dict[int, float], b: dict[int, float]) -> float:
        kl = 0.0
        for k, pa in a.items():
            mb = b.get(k, 0.0)
            if pa > 0.0 and mb > 0.0:
                kl += pa * math.log(pa / mb)
        return kl

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


class MinHashLSH:
    """MinHash + Locality-Sensitive Hashing for approximate Jaccard similarity.

    Computes a fixed-size signature (num_perm hash values) per seed's edge
    set, then hashes signatures into LSH buckets so "find similar seeds"
    becomes a bucket lookup instead of a full corpus scan.

    Approximate Jaccard: matching positions / total positions in two signatures.
    LSH: signatures are split into bands; two seeds collide if ANY band matches,
    giving sub-linear "find similar" queries.
    """

    def __init__(self, num_perm: int = 64, num_bands: int = 8, seed: int = 42):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.band_size = num_perm // num_bands
        # Per-seed MinHash signatures: seed_key -> array('Q') of length num_perm
        self.signatures: dict[str, array] = {}
        # LSH buckets: (band_idx, band_hash) -> set of seed_keys
        self.buckets: dict[tuple[int, int], set[str]] = {}
        # Precomputed hash function coefficients: h(x) = (a*x + b) & mask
        # Uses uint64 wrapping — consistent, no prime needed, fully vectorizable.
        rng = random.Random(seed)
        self._mask = (1 << 64) - 1
        self._coeffs = [
            (rng.randint(1, self._mask), rng.randint(0, self._mask)) for _ in range(num_perm)
        ]
        # Numpy arrays for vectorized compute_signature (~22x faster)
        self._coeffs_a_np = None
        self._coeffs_b_np = None
        try:
            import numpy as _np

            self._coeffs_a_np = _np.array([a for a, _ in self._coeffs], dtype=_np.uint64)
            self._coeffs_b_np = _np.array([b for _, b in self._coeffs], dtype=_np.uint64)
        except ImportError:
            pass

    def compute_signature(self, edge_set: set[int]) -> array:
        """Compute MinHash signature for a set of edge indices.

        Uses k independent hash functions of the form h(x) = (a*x + b) & mask,
        taking the minimum hash value across all elements in the set.
        Vectorized with numpy when available (~22x faster).
        """
        if self._coeffs_a_np is not None and len(edge_set) > 0:
            import numpy as _np

            edges = _np.fromiter(edge_set, dtype=_np.uint64, count=len(edge_set))
            # Materializing the whole (num_perm, n_edges) product costs
            # num_perm*n*8 bytes twice over (product, then sum) -- 4.2MB per
            # call at n=4000, and it was among the largest allocation sites in
            # a profiled run. The row-min is independent per permutation, so
            # the same result comes from a band of rows at a time at a
            # fraction of the footprint.
            out = _np.empty(self.num_perm, dtype=_np.uint64)
            band = 8
            for lo in range(0, self.num_perm, band):
                hi = lo + band
                blk = (
                    self._coeffs_a_np[lo:hi, None] * edges[None, :] + self._coeffs_b_np[lo:hi, None]
                )
                out[lo:hi] = blk.min(axis=1)
            return array("Q", out)

        # Python fallback
        sig = array("Q", [self._mask]) * self.num_perm
        for edge in edge_set:
            for i, (a, b) in enumerate(self._coeffs):
                h = (a * edge + b) & self._mask
                if h < sig[i]:
                    sig[i] = h
        return sig

    def add(self, seed_key: str, sig: list[int] | array):
        """Add a seed's signature to the index."""
        self.signatures[seed_key] = sig if isinstance(sig, array) else array("Q", sig)
        # Insert into LSH buckets
        for band_idx in range(self.num_bands):
            start = band_idx * self.band_size
            end = start + self.band_size
            band_bytes = struct.pack(f"<{end - start}Q", *sig[start:end])
            band_hash = crc32_ieee(band_bytes)
            bucket_key = (band_idx, band_hash)
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = set()
            self.buckets[bucket_key].add(seed_key)

    def remove(self, seed_key: str):
        """Remove a seed from the index."""
        sig = self.signatures.pop(seed_key, None)
        if sig is None:
            return
        for band_idx in range(self.num_bands):
            start = band_idx * self.band_size
            end = start + self.band_size
            band_bytes = struct.pack(f"<{end - start}Q", *sig[start:end])
            band_hash = crc32_ieee(band_bytes)
            bucket_key = (band_idx, band_hash)
            bucket = self.buckets.get(bucket_key)
            if bucket:
                bucket.discard(seed_key)
                if not bucket:
                    del self.buckets[bucket_key]

    def approximate_jaccard(self, key_a: str, key_b: str) -> float:
        """Estimate Jaccard similarity between two seeds via MinHash signatures.

        Returns value in [0, 1] where 1 = identical edge sets.
        """
        sig_a = self.signatures.get(key_a)
        sig_b = self.signatures.get(key_b)
        if sig_a is None or sig_b is None:
            return 0.0
        return _sig_matches(sig_a, sig_b) / self.num_perm

    def find_similar(self, seed_key: str, min_jaccard: float = 0.3) -> set[str]:
        """Find seeds with approximate Jaccard >= min_jaccard via LSH buckets.

        Returns set of similar seed_keys (excluding the query seed itself).
        Uses band-based LSH: two seeds are candidates if they share ANY band.
        Then filters by full-signature Jaccard threshold.
        """
        sig = self.signatures.get(seed_key)
        if sig is None:
            return set()

        # Collect all candidates from LSH buckets
        candidates: set[str] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.band_size
            end = start + self.band_size
            band_bytes = struct.pack(f"<{end - start}Q", *sig[start:end])
            band_hash = crc32_ieee(band_bytes)
            bucket_key = (band_idx, band_hash)
            bucket = self.buckets.get(bucket_key)
            if bucket:
                candidates.update(bucket)
        candidates.discard(seed_key)

        # Filter by full Jaccard threshold
        if min_jaccard <= 0:
            return candidates
        return {k for k in candidates if self.approximate_jaccard(seed_key, k) >= min_jaccard}

    def corpus_minhash(self, seed_keys: set[str] | None = None) -> array:
        """Compute MinHash of the union of all seeds' edge sets.

        The union's MinHash is the element-wise minimum of individual
        signatures — this is a property of MinHash, not an approximation.
        """
        if seed_keys is None:
            seed_keys = set(self.signatures.keys())
        if not seed_keys:
            return array("Q", [self._mask]) * self.num_perm
        if _HAS_NUMPY:
            present = [self.signatures[k] for k in seed_keys if k in self.signatures]
            if not present:
                return array("Q", [self._mask]) * self.num_perm
            sigs = [_sig_np(s) for s in present]
            return array("Q", np.minimum.reduce(sigs).tolist())
        # Legacy elementwise-min loop (no-numpy fallback).
        first = next(iter(seed_keys))
        result = list(self.signatures.get(first, [self._mask] * self.num_perm))
        for key in seed_keys:
            sig = self.signatures.get(key)
            if sig:
                for i in range(self.num_perm):
                    if sig[i] < result[i]:
                        result[i] = sig[i]
        return array("Q", result)

    def approximate_union_jaccard(self, seed_key: str, corpus_sig: list[int]) -> float:
        """Estimate Jaccard(seed, corpus_union) using precomputed corpus signature.

        This replaces the O(n) other_edges union scan with an O(k) signature
        comparison, where k = num_perm (typically 64).
        """
        sig = self.signatures.get(seed_key)
        if sig is None:
            return 0.0
        return _sig_matches(sig, corpus_sig) / self.num_perm


class EdgeTracker:
    """Track coverage edges per seed for smarter scheduling.

    After each execution that produces new coverage, records which
    edges are now hit. Seeds that contribute unique edges get higher
    priority; seeds fully subsumed by others get deprioritized.
    """

    def __init__(
        self,
        map_size: int = 65536,
        max_tracked_seeds: int = MAX_TRACKED_SEEDS,
        morris_mode: bool = False,
    ):
        self.map_size = map_size
        self.max_tracked_seeds = max_tracked_seeds
        self._morris_mode = morris_mode
        # Per-seed edge sets: seed_key -> set of edge indices
        self.seed_edges: dict[str, set[int]] = {}
        # Per-seed hit counts: seed_key -> {edge_index: hit_count} (sparse)
        self.seed_hit_counts: dict[str, dict[int, int]] = {}
        # Global cumulative edge set (all edges ever seen)
        self.cumulative_edges: set[int] = set()
        # Total edges ever (preserved across bitmap resize — monotonically increasing)
        self._cumulative_edges_total: int = 0
        # Aggregate distribution: maintained incrementally, not rebuilt from scratch
        self._aggregate_totals: dict[int, int] = {}
        self._aggregate_total_count: int = 0
        self._aggregate_cache: dict[int, float] | None = None
        # Good-Turing: global cumulative hit count per edge (across all seeds)
        self._global_edge_hits: dict[int, int] = {}

        # Which key space the tracked dicts are in. record_edges() accepts two
        # kinds of input that produce DIFFERENT and incompatible keys:
        #
        #   bytes bitmap -> keys are SLOT INDICES (np.flatnonzero positions).
        #                   Meaningless after a resize: the same logical edge
        #                   lands somewhere else. Used by ptrace coverage.
        #   set of ints  -> keys are EDGE IDs (ctx ^ prev_loc ^ cur_loc).
        #                   Carry no map_size term at all, so a resize cannot
        #                   invalidate them. Used by the SHM path.
        #
        # on_resize() needs to tell these apart, and nothing else recorded it.
        # Note both spaces write into the SAME dicts, so a run that mixed
        # ptrace and SHM coverage would silently collide slot indices with
        # edge IDs — nothing does today, which is why making this explicit is
        # worth more than the one branch it enables.
        self._key_space: str | None = None
        self._spectrum_dirty = True
        self._frequency_spectrum: dict[int, int] = {}
        self.max_hit_count: int = 0
        # MinHash/LSH for approximate Jaccard and subsumption
        self._minhash = MinHashLSH(num_perm=64, num_bands=8)
        self._corpus_sig: array | None = None
        self._corpus_profile_cache: dict[float, float] | None = None
        # Per-seed edge traces for directed distance: seed_key -> set of (prev, curr) edges
        self.seed_edge_traces: dict[str, set[tuple[int, int]]] = {}
        # Per-target cumulative edge sets: target_name -> set of edge indices
        self.target_cumulative_edges: dict[str, set[int]] = {}
        # Per-seed per-target edge sets: seed_key -> {target_name: set of edges}
        self.seed_target_edges: dict[str, dict[str, set[int]]] = {}
        # ── Temporal coverage tracking ──────────────────────────────────────
        # Per-edge lifetime: exec count when edge was first/last seen
        self._edge_first_seen: dict[int, int] = {}
        self._edge_last_seen: dict[int, int] = {}
        # Edge discovery time-series: list of (exec_count, cumulative_edge_count)
        self._coverage_execs: array = array("Q")  # exec_count per coverage snapshot
        self._coverage_edges: array = array("Q")  # cumulative edges per snapshot
        self._coverage_timestamps: array = array("d")  # wall-clock timestamp per snapshot
        # Branch correlation matrix: {(edge_a, edge_b): co_occurrence_count}
        self._correlation_matrix: dict[tuple[int, int], int] = {}
        self._correlation_total: int = 0
        # ── Stack depth + path hash per seed ─────────────────────────────
        self.seed_stack_depth: dict[str, int] = {}
        self.seed_path_hash: dict[str, int] = {}
        # ── Rare edge tracking ──────────────────────────────────────────
        # Per-edge owner count: how many distinct seeds hit each edge
        self._edge_owner_count: defaultdict[int, int] = defaultdict(int)
        # ── Hardware perf metrics per seed ──────────────────────────────
        self.seed_hw_instructions: dict[str, int] = {}
        self.seed_hw_branches: dict[str, int] = {}
        self.seed_hw_branch_misses: dict[str, int] = {}

    def record_edges(
        self,
        seed_key: str,
        hit_edges: set[int] | bytes,
        target_name: str = "",
        hit_counts: dict[int, int] | None = None,
        morris_mode: bool = False,
        stack_depth: int = 0,
        path_hash: int = 0,
        hw_instructions: int = 0,
        hw_branches: int = 0,
        hw_branch_misses: int = 0,
    ) -> set[int]:
        """Record edges hit by a seed execution.

        Args:
            seed_key: Hash of the seed input.
            hit_edges: Set of edge IDs that were hit.
            target_name: Name of the target binary (for multi-target tracking).
            hit_counts: Optional {edge_id: count} map. When provided (e.g. sparse
                SHM entries with 32-bit saturating counters) these are used for
                hit-count diversity scoring. Defaults to count=1 per edge.
            stack_depth: Max stack depth in bytes (from __sancov_lowest_stack).
            path_hash: Rolling 64-bit path hash from edge IDs.
            hw_instructions: Hardware instruction count delta (from perf_event_open).
            hw_branches: Hardware branch count delta (from perf_event_open).
            hw_branch_misses: Hardware branch miss count delta (from perf_event_open).

        Returns:
            Set of NEW edge indices not previously seen.
        """
        if stack_depth > 0:
            self.seed_stack_depth[seed_key] = stack_depth
        if path_hash != 0:
            self.seed_path_hash[seed_key] = path_hash
        if hw_instructions > 0:
            self.seed_hw_instructions[seed_key] = hw_instructions
        if hw_branches > 0:
            self.seed_hw_branches[seed_key] = hw_branches
        if hw_branch_misses > 0:
            self.seed_hw_branch_misses[seed_key] = hw_branch_misses
        new_edges = set()
        if seed_key not in self.seed_edges:
            self.seed_edges[seed_key] = set()
        if seed_key not in self.seed_hit_counts:
            self.seed_hit_counts[seed_key] = {}
        hc = self.seed_hit_counts[seed_key]

        # Backward compat: bytes input treated as byte bitmap where
        # non-zero byte positions = edge indices (for ptrace + tests).
        if isinstance(hit_edges, bytes):
            self._note_key_space("position")
            bitmap = hit_edges
            if not morris_mode:
                bitmap = classify_counts(bitmap)
            arr = np.frombuffer(bitmap, dtype=np.uint8, count=min(len(bitmap), self.map_size))
            for i in np.flatnonzero(arr):
                i = int(i)
                raw_val = int(arr[i])
                val = int(round(morris_estimate(raw_val))) if morris_mode else raw_val
                new_edges.add(i)
                hc[i] = val
                self._aggregate_totals[i] = self._aggregate_totals.get(i, 0) + val
                self._aggregate_total_count += val
                old_gh = self._global_edge_hits.get(i, 0)
                self._global_edge_hits[i] = old_gh + val
                self._spectrum_dirty = True
                if self._global_edge_hits[i] > self.max_hit_count:
                    self.max_hit_count = self._global_edge_hits[i]

        else:
            # New sparse path: hit_edges is a set of edge IDs
            self._note_key_space("edge_id")
            for edge_id in hit_edges:
                val = hit_counts.get(edge_id, 1) if hit_counts else 1
                new_edges.add(edge_id)
                hc[edge_id] = val
                self._aggregate_totals[edge_id] = self._aggregate_totals.get(edge_id, 0) + val
                self._aggregate_total_count += val
                old_gh = self._global_edge_hits.get(edge_id, 0)
                self._global_edge_hits[edge_id] = old_gh + val
                self._spectrum_dirty = True
                if self._global_edge_hits[edge_id] > self.max_hit_count:
                    self.max_hit_count = self._global_edge_hits[edge_id]

        new_contributions = new_edges - self.cumulative_edges
        self.cumulative_edges.update(new_edges)

        # Rare edge tracking: count how many distinct seeds cover each edge.
        #
        # This only ran when hit_edges was a set (the sparse edge-ID path). On
        # the bitmap path -- which is what SHM coverage uses -- the guard
        # evaluated to an empty set and _edge_owner_count was never populated
        # at all, so count_rare_edges() returned 0 for every seed and the
        # honggfuzz rare-edge energy boost was uniformly inert. It also left
        # edge_rarity_stats() with nothing to read, which is why that function
        # had been pointed at _global_edge_hits (execution hit volume) instead.
        #
        # Counting new_edges minus the seed's already-known edges keeps this
        # idempotent: re-executing the same seed must not inflate its
        # ownership share.
        already_owned = self.seed_edges[seed_key]
        for edge_id in new_edges - already_owned:
            self._edge_owner_count[edge_id] = self._edge_owner_count[edge_id] + 1

        self.seed_edges[seed_key].update(new_edges)

        # Per-target tracking
        if target_name:
            if target_name not in self.target_cumulative_edges:
                self.target_cumulative_edges[target_name] = set()
            self.target_cumulative_edges[target_name].update(new_edges)
            if seed_key not in self.seed_target_edges:
                self.seed_target_edges[seed_key] = {}
            if target_name not in self.seed_target_edges[seed_key]:
                self.seed_target_edges[seed_key][target_name] = set()
            self.seed_target_edges[seed_key][target_name].update(new_edges)

        # Update MinHash signature and LSH index
        sig = self._minhash.compute_signature(self.seed_edges[seed_key])
        self._minhash.add(seed_key, sig)

        # Invalidate caches
        self._aggregate_cache = None
        self._corpus_sig = None
        self._corpus_profile_cache = None

        # Update temporal tracking
        self.record_edge_lifetimes(new_edges, len(self.cumulative_edges))
        self.update_correlation(new_edges)

        # Prune old seeds if over limit
        self._maybe_prune()

        return new_contributions

    def _note_key_space(self, space: str) -> None:
        """Record which key space record_edges() is populating."""
        if self._key_space is None:
            self._key_space = space
        elif self._key_space != space:
            # Both spaces share the same dicts, so mixing them corrupts every
            # per-edge statistic. Warn rather than raise: the tracker is not
            # the right place to abort a fuzzing run.
            log.warning(
                "EdgeTracker fed %s keys after %s keys — slot indices and edge "
                "IDs are now mixed in the same tables; per-edge statistics are "
                "unreliable for the rest of this run",
                space,
                self._key_space,
            )
            self._key_space = "mixed"

    def on_resize(self, new_map_size: int) -> None:
        """Adapt tracked state to a resized coverage map.

        Only slot-indexed state is invalidated by a resize. Edge-ID-keyed
        state is not, and clearing it is destructive:

        The previous implementation wiped everything unconditionally, on the
        reasoning that "AFL's hash (edge_id = hash(src,dst) % map_size) maps
        the same logical edge to a different position". That is correct for
        AFL's classic bitmap, where the INDEX IS the edge identity. It is
        false for this design: entries carry an explicit 8-byte
        {edge_id, count} pair, and edge_id = ctx ^ prev_loc ^ cur_loc has no
        map_size term. Only the starting probe position edge_id % map_size
        changes, and no position is ever persisted. reset_edge_map() now
        bumps a generation counter instead of memsetting the table, and stale
        entries are filtered by the slow path on the next read.

        The cost of getting this wrong was not subtle. A stall-triggered
        resize wiped cumulative_edges, _global_edge_hits, seed_edges,
        seed_hit_counts, the MinHash signatures and the rarity spectrum, so
        the next execution reported every already-known edge as new. That
        reset execs_since_edge — defeating the stall detector that triggered
        the resize in the first place — produced a burst of spurious
        "interesting" saves, and corrupted the rarity schedule and the
        Elo/Shapley operator attribution. It also zeroed the edge count in
        the UI, which is why _cumulative_edges_total existed at all.

        The one case where the old reasoning holds is a tracker fed byte
        bitmaps (ptrace coverage), whose keys really are slot indices. Resize
        only fires on the SHM path today, which only ever passes edge-ID sets,
        so this branch is currently unreachable — kept so the invariant is
        enforced rather than assumed.

        Args:
            new_map_size: Entry count of the resized table.
        """
        self.map_size = new_map_size

        if self._key_space != "position":
            # edge_id keys survive a resize unchanged. Nothing to do.
            return

        log.info("Resize invalidates slot-indexed coverage state; clearing")
        self._cumulative_edges_total = max(self._cumulative_edges_total, len(self.cumulative_edges))
        self.cumulative_edges.clear()
        self._global_edge_hits.clear()
        self.seed_edges.clear()
        self.seed_hit_counts.clear()
        self.seed_edge_traces.clear()
        self._aggregate_totals.clear()
        self._aggregate_total_count = 0
        self._aggregate_cache = None
        self._spectrum_dirty = True
        self.max_hit_count = 0

    def reset_after_resize(self):
        """Deprecated alias for on_resize(); preserves the current map_size."""
        self.on_resize(self.map_size)

    def _maybe_prune(self):
        """Prune tracked seeds when count exceeds max_tracked_seeds.

        Evicts cheapest-first by how much unique coverage is lost with the
        seed, so fully-subsumed seeds go before any seed that owns an edge no
        other tracked seed covers; ties keep insertion order.

        Overflowing the ceiling prunes back below it by PRUNE_BATCH_FRAC, not
        to the ceiling itself.  Without that margin ``excess`` is 1 in steady
        state, so the owner-count pass below -- O(tracked seeds x edges per
        seed) -- ran on every single record_edges call to evict one seed.
        Measured on 600 insertions of 800 edges each: 14.78s with the pass
        firing every call against 0.81s with it never firing, which is why the
        ceiling was raised to 200,000 in fe8fd42 rather than the cost being
        paid.  Pruning in batches amortises the pass and keeps the bound.
        """
        if len(self.seed_edges) <= self.max_tracked_seeds:
            return

        batch = int(self.max_tracked_seeds * PRUNE_BATCH_FRAC)
        low_water = max(1, self.max_tracked_seeds - batch)
        excess = len(self.seed_edges) - low_water

        # Count how many currently-tracked seeds own each edge.
        edge_owners: dict[int, int] = {}
        for edges in self.seed_edges.values():
            for e in edges:
                edge_owners[e] = edge_owners.get(e, 0) + 1

        # One revalidating pass, cheapest-first by how much unique coverage
        # goes with the seed.  A seed whose every edge is held by another
        # still-tracked seed has loss 0, so fully-subsumed seeds drain first
        # and the old two-phase behaviour falls out of the ordering; ties keep
        # insertion order, so age remains the tiebreak it always was.
        #
        # Two phases against a snapshot cannot get this right, and both ways
        # it fails lose coverage silently:
        #
        #   * Two seeds jointly owning one edge each see an owner count of 2,
        #     so a snapshot marks neither protected and evicting both drops
        #     the edge.  Unreachable while ``excess`` was always 1; routine
        #     once batches are pruned.
        #   * Evicting a subsumed seed can make another seed's edges unique.
        #     A subsumption phase would evict A because B covered for it, then
        #     an age phase would evict B, dropping an edge that neither
        #     eviction loses on its own.
        #
        # Hence the loss is recomputed at the moment of eviction and the entry
        # re-queued if it went stale, rather than being ordered once up front.
        #
        # In real campaigns the ordering is decided almost entirely by the tie.
        # Instrumented over png_read and gzip_read, 99.9% of tracked candidates
        # had a loss of zero: a seed is admitted for coverage it alone had, but
        # later seeds subsume it and the edge space saturates, so by the time
        # the ceiling binds nearly everything is redundant.  Which seed goes is
        # therefore settled by the tiebreak, not by the loss figure.
        keys_to_prune: list[str] = []
        evicted: set[str] = set()

        def _unique_loss(key: str) -> int:
            return sum(1 for e in self.seed_edges[key] if edge_owners.get(e, 0) <= 1)

        heap = [(_unique_loss(k), i, k) for i, k in enumerate(self.seed_edges)]
        heapq.heapify(heap)
        while len(keys_to_prune) < excess and heap:
            loss, order, key = heapq.heappop(heap)
            if key in evicted:
                continue
            current = _unique_loss(key)
            if current != loss:
                heapq.heappush(heap, (current, order, key))
                continue
            keys_to_prune.append(key)
            evicted.add(key)
            for e in self.seed_edges[key]:
                edge_owners[e] -= 1

        for key in keys_to_prune:
            self.seed_edges.pop(key, None)
            self.seed_hit_counts.pop(key, None)
            self.seed_edge_traces.pop(key, None)
            self.seed_target_edges.pop(key, None)
            self.seed_stack_depth.pop(key, None)
            self.seed_path_hash.pop(key, None)
            self.seed_hw_instructions.pop(key, None)
            self.seed_hw_branches.pop(key, None)
            self.seed_hw_branch_misses.pop(key, None)
            self._minhash.remove(key)

        # Owner counts are incremented in record_edges but were never adjusted
        # here, so every prune left _edge_owner_count crediting edges to seeds
        # that no longer exist. The counts only ever rose, which inflates
        # mean_owners (suppressing the crowding bonus) and pushes edges above
        # RARE_EDGE_OWNERS so they stop reading as rare -- a slow, silent decay
        # of the rarity signal the whole schedule is steered by.
        #
        # edge_owners was decremented as each seed was selected above, so it is
        # already the survivor tally exactly -- no second O(seeds x edges) pass
        # is needed to rebuild it. Zero and negative entries are dropped rather
        # than stored: _edge_owner_count is read by bare subscript on a
        # defaultdict, so keeping them would grow the map with edges no seed
        # owns.
        if keys_to_prune:
            rebuilt: defaultdict[int, int] = defaultdict(int)
            for e, n in edge_owners.items():
                if n > 0:
                    rebuilt[e] = n
            self._edge_owner_count = rebuilt

        self._aggregate_cache = None
        self._corpus_sig = None
        self._corpus_profile_cache = None

    # ── Temporal coverage tracking methods ─────────────────────────────────

    def record_edge_lifetimes(self, edge_set: set[int], exec_count: int):
        """Update first/last seen timestamps for edges.

        Args:
            edge_set: Set of edge indices hit in this execution.
            exec_count: Current execution count.
        """
        for edge in edge_set:
            if edge not in self._edge_first_seen:
                self._edge_first_seen[edge] = exec_count
            self._edge_last_seen[edge] = exec_count

    def record_coverage_snapshot(self, exec_count: int):
        """Record a point-in-time snapshot of cumulative edge count.

        Args:
            exec_count: Current execution count.
        """
        self._coverage_execs.append(exec_count)
        self._coverage_edges.append(len(self.cumulative_edges))
        self._coverage_timestamps.append(time.time())
        if len(self._coverage_execs) > COVERAGE_TIMELINE_MAX:
            del self._coverage_execs[:250]
            del self._coverage_edges[:250]

    def update_correlation(self, edge_set: set[int]):
        """Update branch correlation matrix with co-occurring edges.

        Tracks which edges fire together in the same execution.
        Uses sampling to bound O(n²) cost: samples up to 20 pairs per call.

        Args:
            edge_set: Set of edge indices hit in this execution.
        """
        if len(edge_set) < 2:
            return
        self._correlation_total += 1
        edges = sorted(edge_set)[:50]
        # Sample pairs instead of all O(n²) — bounds to O(1) per call
        n = len(edges)
        max_pairs = min(20, n * (n - 1) // 2)
        if n <= 8:
            # Small set: track all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    key = (edges[i], edges[j])
                    self._correlation_matrix[key] = self._correlation_matrix.get(key, 0) + 1
        else:
            # Large set: sample random pairs
            import random as _rand

            for _ in range(max_pairs):
                i = _rand.randint(0, n - 2)
                j = _rand.randint(i + 1, n - 1)
                key = (edges[i], edges[j])
                self._correlation_matrix[key] = self._correlation_matrix.get(key, 0) + 1
        if len(self._correlation_matrix) > CORRELATION_MATRIX_MAX:
            self._prune_correlation()

    def _prune_correlation(self):
        """Keep the top 50% of correlation pairs by count."""
        items = sorted(self._correlation_matrix.items(), key=lambda kv: kv[1], reverse=True)
        keep = len(items) // 2
        self._correlation_matrix = dict(items[:keep])
        self._correlation_total = sum(c for _, c in items[:keep])

    def branch_correlation(self, edge_a: int, edge_b: int) -> float:
        """Get correlation strength between two edges.

        Returns:
            Float in [0, 1] where 1 = always co-occur.
        """
        if self._correlation_total == 0:
            return 0.0
        key = (min(edge_a, edge_b), max(edge_a, edge_b))
        count = self._correlation_matrix.get(key, 0)
        return count / self._correlation_total

    def top_correlated_pairs(self, k: int = 20) -> list[tuple[int, int, float]]:
        """Get the top-k most correlated edge pairs.

        Returns:
            List of (edge_a, edge_b, correlation) sorted by correlation desc.
        """
        if not self._correlation_matrix:
            return []
        # Sort by count, take top k
        sorted_pairs = sorted(
            self._correlation_matrix.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]
        return [(a, b, count / self._correlation_total) for (a, b), count in sorted_pairs]

    def edge_lifetime_stats(self) -> dict:
        """Compute statistics on edge lifetimes.

        Returns:
            Dict with median, mean, max lifetime (in execs).
        """
        if not self._edge_first_seen or not self._edge_last_seen:
            return {"median": 0, "mean": 0.0, "max": 0}
        lifetimes = []
        for edge in self._edge_last_seen:
            first = self._edge_first_seen.get(edge, 0)
            last = self._edge_last_seen[edge]
            lifetimes.append(last - first)
        if not lifetimes:
            return {"median": 0, "mean": 0.0, "max": 0}
        lifetimes.sort()
        n = len(lifetimes)
        median = lifetimes[n // 2]
        mean = sum(lifetimes) / n
        return {"median": median, "mean": mean, "max": lifetimes[-1]}

    def coverage_growth_rate(self) -> float:
        """Compute edges-per-exec from coverage timeline.

        Returns:
            Average edges discovered per execution.
        """
        if len(self._coverage_execs) < 2:
            return 0.0
        exec_diff = self._coverage_execs[-1] - self._coverage_execs[0]
        if exec_diff <= 0:
            return 0.0
        return (self._coverage_edges[-1] - self._coverage_edges[0]) / exec_diff

    def edge_age_distribution(self) -> dict[str, int]:
        """Classify edges by age (how recently they were first seen).

        Returns:
            Dict with counts: new (last 10%), mature, old (first 10%).
        """
        if not self._edge_first_seen:
            return {"new": 0, "mature": 0, "old": 0}
        first_vals = sorted(self._edge_first_seen.values())
        n = len(first_vals)
        if n < 10:
            return {"new": n, "mature": 0, "old": 0}
        p10 = first_vals[n // 10]
        p90 = first_vals[9 * n // 10]
        counts = {"new": 0, "mature": 0, "old": 0}
        for first in self._edge_first_seen.values():
            if first >= p90:
                counts["new"] += 1
            elif first <= p10:
                counts["old"] += 1
            else:
                counts["mature"] += 1
        return counts

    def coverage_growth_model(self) -> dict:
        """Fit a coverage growth model to the edge discovery curve.

        Uses the coverage timeline to estimate:
        - current_rate: edges per exec (recent)
        - projected_total: estimated total edges at saturation
        - time_to_plateau: estimated execs until marginal gain < 1 edge per 1000 execs
        - confidence: based on timeline length and fit quality

        Returns:
            Dict with growth model parameters.
        """
        if len(self._coverage_execs) < 3:
            return {
                "current_rate": 0.0,
                "projected_total": 0,
                "time_to_plateau": 0,
                "confidence": 0.0,
                "plateau_detected": False,
            }

        # Extract time series
        execs = self._coverage_execs.tolist()
        edges = self._coverage_edges.tolist()

        # Simple exponential saturation model: E(t) = A * (1 - exp(-k*t))
        # Linearize: ln(A - E(t)) = ln(A) - k*t
        # Use last 10 points for recent rate
        n = len(execs)
        recent_n = min(10, n)
        if recent_n < 2:
            return {
                "current_rate": 0.0,
                "projected_total": edges[-1],
                "time_to_plateau": 0,
                "confidence": 0.1,
                "plateau_detected": False,
            }

        # Recent rate: edges per exec
        recent_execs = execs[-recent_n:]
        recent_edges = edges[-recent_n:]
        exec_diff = recent_execs[-1] - recent_execs[0]
        edge_diff = recent_edges[-1] - recent_edges[0]
        current_rate = edge_diff / exec_diff if exec_diff > 0 else 0.0

        # Projected total: use linear extrapolation from recent trend
        # If rate is declining, project saturation point
        if n >= 4:
            mid = n // 2
            early_rate = (edges[mid] - edges[0]) / max(1, execs[mid] - execs[0])
            late_rate = (edges[-1] - edges[mid]) / max(1, execs[-1] - execs[mid])
            if late_rate < early_rate * 0.5 and late_rate > 0:
                # Rate is declining — estimate saturation
                plateau_detected = True
                if late_rate > 0.001:
                    execs_to_plateau = int((edges[-1] - edges[0]) / max(0.001, late_rate))
                else:
                    execs_to_plateau = 0
                projected_total = edges[-1] + execs_to_plateau * late_rate
            else:
                # Rate stable or increasing — no saturation detected yet
                plateau_detected = False
                execs_to_plateau = 0
                projected_total = 0
        else:
            plateau_detected = False
            execs_to_plateau = 0
            projected_total = 0

        # Confidence based on timeline length
        confidence = min(1.0, n / 5)

        return {
            "current_rate": current_rate,
            "projected_total": int(projected_total),
            "time_to_plateau": execs_to_plateau,
            "plateau_detected": plateau_detected,
            "confidence": confidence,
        }

    def bayesian_coverage_growth_model(self) -> dict:
        """Bayesian coverage growth model with posterior uncertainty.

        Fits E(t) = A * (1 - exp(-k * t)) via nonlinear least squares and
        computes approximate posterior distributions via the Laplace
        approximation (Hessian-based). Returns:
        - posterior means and 95% credible intervals for A and k
        - P(stalled): posterior probability the growth rate has dropped below
          threshold (1 edge per 1000 execs)
        - P(growth remaining): posterior probability that asymptote > current
        - Proper uncertainty quantification vs. the heuristic confidence field

        Falls back to the frequentist model when data is insufficient (< 5
        timeline points) or the nonlinear fit fails to converge.
        """
        result: dict = {
            "A_mean": None,
            "A_ci95": None,
            "k_mean": None,
            "k_ci95": None,
            "sigma_mean": None,
            "p_stalled": None,
            "p_growth_remaining": None,
            "current_rate_posterior_mean": None,
            "method": "bayesian_laplace",
        }

        n_points = len(self._coverage_execs)
        if n_points < 5:
            fallback = self.coverage_growth_model()
            result.update(fallback)
            result["method"] = "fallback_insufficient_data"
            return result

        execs = self._coverage_execs.tolist()
        edges = self._coverage_edges.tolist()
        t_arr = np.array(execs, dtype=np.float64)
        y_arr = np.array(edges, dtype=np.float64)

        # Initial guess: A ≈ max(edges) * 1.2, k ≈ ln(2) / (t_mid)
        A0 = float(max(y_arr)) * 1.2 + 100.0
        t_mid = float(t_arr[len(t_arr) // 2] - t_arr[0]) if len(t_arr) > 1 else 1.0
        k0 = max(0.693 / max(t_mid, 1.0), 1e-10)
        p0 = np.array([A0, k0], dtype=np.float64)

        # Weight: more recent points get slightly higher weight
        weights = np.linspace(0.5, 1.0, len(t_arr))

        def residuals(p):
            A, k = p
            return weights * (y_arr - A * (1.0 - np.exp(-k * t_arr)))

        try:
            fit = _levenberg_marquardt(
                residuals, p0, bounds=([10.0, 1e-10], [1e8, 1.0]), max_nfev=200
            )
            A_hat, k_hat = fit.x
            sigma_hat = np.sqrt(fit.cost / max(1, n_points - 2))

            # Laplace approximation: posterior covariance ≈ inv(J^T J) * sigma^2
            J = fit.jac
            try:
                cov = np.linalg.inv(J.T @ J) * sigma_hat**2
            except np.linalg.LinAlgError:
                cov = None

            result["A_mean"] = float(A_hat)
            result["k_mean"] = float(k_hat)
            result["sigma_mean"] = float(sigma_hat)
            rate = A_hat * k_hat * np.exp(-k_hat * float(t_arr[-1]))
            result["current_rate_posterior_mean"] = float(rate)

            if cov is not None:
                se_A = float(np.sqrt(max(cov[0, 0], 0)))
                se_k = float(np.sqrt(max(cov[1, 1], 0)))
                z = 1.96
                result["A_ci95"] = (float(A_hat - z * se_A), float(A_hat + z * se_A))
                result["k_ci95"] = (float(k_hat - z * se_k), float(k_hat + z * se_k))

                # P(stalled) via delta method on current rate
                t_last = float(t_arr[-1])
                grad_rate = np.array(
                    [
                        k_hat * np.exp(-k_hat * t_last),
                        A_hat * np.exp(-k_hat * t_last) * (1 - k_hat * t_last),
                    ]
                )
                var_rate = float(grad_rate @ cov @ grad_rate)
                se_rate = float(np.sqrt(max(var_rate, 0)))
                if se_rate > 1e-12:
                    p_stalled = _norm_cdf(0.001, loc=rate, scale=se_rate)
                else:
                    p_stalled = 1.0 if rate < 0.001 else 0.0
                result["p_stalled"] = float(p_stalled)

                # P(growth remaining): posterior that A > current edges
                current_edges = int(y_arr[-1])
                if se_A > 1e-12:
                    p_growth = 1.0 - _norm_cdf(current_edges, loc=A_hat, scale=se_A)
                else:
                    p_growth = 1.0 if A_hat > current_edges else 0.0
                result["p_growth_remaining"] = float(p_growth)
            else:
                result["p_stalled"] = 1.0 if rate < 0.001 else 0.0
                result["p_growth_remaining"] = 1.0 if A_hat > y_arr[-1] else 0.0

            result["method"] = "bayesian_laplace"

        except Exception:
            fallback = self.coverage_growth_model()
            result.update(fallback)
            result["method"] = "fallback_fit_failed"

        return result

    def record_edge_trace(self, seed_key: str, edges: set[tuple[int, int]]):
        """Record the (prev, curr) edge trace for a seed.

        Used by directed distance computation to know which basic blocks
        a seed's execution passed through.

        Args:
            seed_key: Hash of the seed input.
            edges: Set of (prev_edge_index, curr_edge_index) pairs.
        """
        if edges:
            if seed_key in self.seed_edge_traces:
                self.seed_edge_traces[seed_key].update(edges)
            else:
                self.seed_edge_traces[seed_key] = set(edges)

    def compute_subsumption_weight(self, seed_key: str) -> float:
        """Compute a weight multiplier based on Jaccard similarity of edge sets.

        Uses MinHash to approximate Jaccard(seed, corpus_union) in O(k) time
        instead of O(n) full-set union. The corpus MinHash signature (element-wise
        minimum of all individual signatures) is precomputed and cached.

        Returns a continuous weight in [0.1, 1.0] based on how much this
        seed's coverage overlaps with other seeds.
        """
        if seed_key not in self.seed_edges:
            return 1.0

        seed_edges = self.seed_edges[seed_key]
        if not seed_edges:
            return 0.5  # no coverage data → slightly deprioritize

        if len(self.seed_edges) <= 1:
            return 1.0  # only seed — all edges are novel

        # Use MinHash: Jaccard ≈ matching positions / num_perm
        # corpus_sig is the element-wise min of all signatures (= union MinHash)
        if self._corpus_sig is None:
            self._corpus_sig = self._minhash.corpus_minhash()

        jaccard = self._minhash.approximate_union_jaccard(seed_key, self._corpus_sig)

        # Scale: high overlap (jaccard → 1.0) → low weight, novel → high weight
        return max(0.1, 1.0 - jaccard)

    def _build_aggregate_distribution(self) -> dict[int, float]:
        """Build the corpus-wide aggregate hit-count distribution.

        Uses precomputed running totals maintained incrementally by
        record_edges — only normalizes to probabilities, no iteration
        over all seeds. O(k) where k = number of distinct edges.
        """
        if self._aggregate_cache is not None:
            return self._aggregate_cache

        if self._aggregate_total_count == 0:
            return {}

        self._aggregate_cache = {
            e: c / self._aggregate_total_count for e, c in self._aggregate_totals.items()
        }
        return self._aggregate_cache

    def _js_divergence_vs_aggregate(self, seed_dist: dict[int, float]) -> float:
        """JS divergence between a seed's profile and the corpus's, on the
        edges the seed actually exercises.

        The aggregate is renormalised over the seed's support, so both arguments
        are probability distributions on the same set and the result is a proper
        divergence in [0, ln 2]. Cost stays O(|seed_edges|).

        This used to iterate only the seed's edges while comparing against the
        *unconditioned* aggregate, justified as "edges where the seed is zero
        contribute 0 to KL(P || M)". That is true of the KL(P || M) term and
        false of KL(Q || M): where p = 0 the mixture is m = q/2, so the
        contribution is q*ln2, not zero. The omitted mass was exactly
        0.5 * ln2 * (1 - Q_seed) -- measured 0.204 as-coded against a true JS of
        0.506 on a 40-seed, 2000-edge corpus.

        Adding the missing term back was measured and rejected. The dropped
        quantity is a monotone function of how much of the corpus the seed does
        *not* cover, so restoring it makes this a breadth proxy: across a
        60-seed corpus the full JS correlates -0.988 with edge count and only
        -0.161 with whether the seed actually has an unusual hit-count profile.
        The as-coded version was already mostly that (-0.965 and -0.129). Both
        rank a narrow seed with a perfectly flat profile above a seed with three
        edges hit 500 times, which inverts what this weight is for, and breadth
        is already carried by compute_subsumption_weight.

        Conditioning on the seed's support measures what the docstring on
        compute_hitcount_diversity_weight actually describes -- same edges,
        different frequency profile -- and measures nothing else: -0.026 with
        edge count, +0.789 with loopiness on the same corpus.
        """
        if self._aggregate_total_count == 0:
            return 0.0

        # Aggregate mass on the seed's own edges; the conditioning constant.
        mass = 0.0
        for e in seed_dist:
            mass += self._aggregate_totals.get(e, 0.0)
        if mass <= 0.0:
            return 0.0

        js = 0.0
        for e, p in seed_dist.items():
            if p <= 0.0:
                continue
            q = self._aggregate_totals.get(e, 0.0) / mass
            m = 0.5 * (p + q)
            if m > 0.0:
                js += p * math.log(p / m)
                if q > 0.0:
                    js += q * math.log(q / m)
        return 0.5 * js

    def _wasserstein_vs_aggregate(self, hc: dict[int, int]) -> float:
        """Wasserstein-1 between a seed's hit-count profile and the corpus's.

        Takes the seed's raw {edge: count} map. Normalising to a probability
        mass first would lose the absolute counts, and the absolute counts are
        the coordinate -- a seed that hits a loop 500 times is what this is
        meant to separate from one that hits it 5 times.

        The distance is computed on the log2 hit-count axis; the edge index
        carries no metric (see _hitcount_profile).
        """
        if not hc:
            return 0.0
        corpus = self._corpus_hitcount_profile()
        if not corpus:
            return 0.0
        wasserstein, _ks, _crps = self._cdf_walk(self._hitcount_profile(hc), corpus)
        return wasserstein

    def compute_hitcount_diversity_weight(self, seed_key: str) -> float:
        """Compute weight based on JS divergence of hit-count distribution.

        A seed that exercises the same edges as the corpus but with a very
        different frequency profile (e.g. hits a loop 500x instead of 5x)
        is behaviorally distinct even with zero new edges.

        Returns a weight in [0.5, 2.0]:
        - 2.0 = unusual profile (high JS — exercises its edges very differently)
        - 0.5 = profile matching the corpus on those edges (redundant)

        The comparison is made on the seed's own edges: the aggregate is
        conditioned on that support, so this measures frequency profile and not
        coverage breadth, which compute_subsumption_weight already covers. See
        _js_divergence_vs_aggregate for why restoring the unconditioned form was
        measured and rejected.

        The full band is reachable -- a seed with one edge hit 5000x scores
        1.48, and the ceiling needs a profile more extreme still -- but ordinary
        seeds sit low: on a 60-seed corpus with a third of the seeds carrying
        hot loops the observed range was [0.54, 0.90]. The mapping below is left
        as it was rather than recentred, because rescaling changes which seeds
        get energy and deserves to be decided on its own evidence instead of
        riding along inside a correctness fix.

        JS divergence is bounded in [0, ln(2)] ≈ [0, 0.693].
        We normalize to [0, 1] and scale to [0.5, 2.0].
        """
        hc = self.seed_hit_counts.get(seed_key)
        if not hc:
            return 1.0

        if self._aggregate_total_count == 0:
            return 1.0

        # Build normalized distribution for this seed
        total = sum(hc.values())
        if total == 0:
            return 1.0
        seed_dist = {e: c / total for e, c in hc.items()}

        # JS divergence computed directly against aggregate (no dict materialization)
        js = self._js_divergence_vs_aggregate(seed_dist)
        # Normalize: max JS is ln(2) ≈ 0.693
        normalized = min(js / math.log(2), 1.0)
        # Scale to [0.5, 2.0]: low divergence → 0.5, high → 2.0
        return 0.5 + 1.5 * normalized

    def _hitcount_profile(self, hc: dict[int, int]) -> dict[float, float]:
        """Empirical distribution of a seed's per-edge hit counts.

        The coordinate is log2(1 + count) and each of the seed's edges
        contributes equal mass. This is the axis the Wasserstein/KS/CRPS
        family is defined on.

        It used to be the edge index. That axis has no metric structure in
        this fuzzer: afl_shim.c folds every site through
        ``edge_id = caller_ctx ^ prev_loc ^ cur_loc`` (and the ptrace backend
        hashes addresses the same way), so |e_a - e_b| is a property of the
        XOR, not of the program. Two "adjacent" edge ids are unrelated code,
        and renaming one location reshuffles every distance. Measured on a
        30-seed corpus with realistic XOR ids, the resulting weight spanned
        only [1.10, 1.34] and what spread it had tracked the hash layout.

        Hit counts are genuinely ordered -- 500 hits is further from 5 than
        50 is -- and separating those cases is what the docstring on
        compute_hitcount_diversity_weight says this family is for. log2
        matches the scale AFL's counter bucketing already imposes.
        """
        n = len(hc)
        if not n:
            return {}
        prof: dict[float, float] = {}
        w = 1.0 / n
        for count in hc.values():
            x = math.log2(1.0 + max(0, count))
            prof[x] = prof.get(x, 0.0) + w
        return prof

    def _profile_axis_span(self) -> float:
        """Width of the hit-count axis, for normalising distances onto [0, 1]."""
        return max(1.0, math.log2(1.0 + max(1, self.max_hit_count)))

    @staticmethod
    def _cdf_walk(prof_a: dict[float, float], prof_b: dict[float, float]) -> tuple[float, ...]:
        """L1, Linf and L2 norms of the CDF difference between two profiles.

        One pass over the merged support. Returns (wasserstein, ks, crps).
        """
        xs = sorted(set(prof_a) | set(prof_b))
        if not xs:
            return 0.0, 0.0, 0.0
        cdf_diff = 0.0
        wasserstein = 0.0
        ks = 0.0
        crps = 0.0
        prev = xs[0]
        for x in xs:
            gap = x - prev
            abs_diff = abs(cdf_diff)
            wasserstein += abs_diff * gap
            ks = max(ks, abs_diff)
            crps += cdf_diff * cdf_diff * gap
            cdf_diff += prof_a.get(x, 0.0) - prof_b.get(x, 0.0)
            prev = x
        ks = max(ks, abs(cdf_diff))
        return wasserstein, ks, crps

    def _corpus_hitcount_profile(self) -> dict[float, float]:
        """Corpus-wide hit-count profile, one observation per discovered edge.

        Each edge's coordinate is its mean count over the seeds that reach it,
        so a hot edge covered by many seeds is not counted once per seed.
        """
        if self._corpus_profile_cache is not None:
            return self._corpus_profile_cache
        totals = self._aggregate_totals
        if not totals:
            self._corpus_profile_cache = {}
            return self._corpus_profile_cache
        prof: dict[float, float] = {}
        w = 1.0 / len(totals)
        for edge, total in totals.items():
            # .get, not a subscript: totals is keyed by execution hit volume,
            # which includes edges hit by runs that were never admitted as
            # seeds. Those have no owner entry, and subscripting a defaultdict
            # would mint one per profile pass.
            owners = max(1, self._edge_owner_count.get(edge, 1))
            x = math.log2(1.0 + total / owners)
            prof[x] = prof.get(x, 0.0) + w
        self._corpus_profile_cache = prof
        return prof

    def compute_wasserstein_distance(self, seed_key_a: str, seed_key_b: str) -> float:
        """Wasserstein-1 distance between two seeds' hit-count profiles.

        Measured on the log2 hit-count axis (see _hitcount_profile), so two
        seeds that drive the same edges at very different intensities are far
        apart even though Jaccard calls them identical.
        """
        wasserstein, _ks, _crps = self._cdf_norms(seed_key_a, seed_key_b)
        return wasserstein

    def compute_ks_distance(self, seed_key_a: str, seed_key_b: str) -> float:
        """Kolmogorov-Smirnov statistic between two seeds' hit-count profiles.

        Maximum absolute CDF difference — L∞ norm of the same quantity
        Wasserstein measures in L¹. KS ∈ [0, 1].
        """
        _w, ks, _crps = self._cdf_norms(seed_key_a, seed_key_b)
        return ks

    def compute_crps(self, seed_key_a: str, seed_key_b: str) -> float:
        """CRPS (Continuous Ranked Probability Score) between two profiles.

        L² integral of the CDF difference. Measured in log2-hit-count units,
        so it is bounded by the width of that axis.
        """
        _w, _ks, crps = self._cdf_norms(seed_key_a, seed_key_b)
        return crps

    def _cdf_norms(self, seed_key_a: str, seed_key_b: str) -> tuple[float, float, float]:
        """Compute Wasserstein-1, KS, and CRPS from a single CDF walk.

        Returns (wasserstein, ks, crps) — L¹, L∞, and L² norms of the same
        CDF difference over the log2 hit-count axis.
        """
        hc_a = self.seed_hit_counts.get(seed_key_a, {})
        hc_b = self.seed_hit_counts.get(seed_key_b, {})
        span = self._profile_axis_span()
        if not hc_a or not hc_b:
            return span, 1.0, span

        w, ks, crps = self._cdf_walk(self._hitcount_profile(hc_a), self._hitcount_profile(hc_b))
        return w, ks, crps

    def compute_coverage_proximity(self, seed_key: str, radius: int = 5) -> float:
        """How much of a seed's coverage sits on the discovery frontier.

        Returns the fraction of the seed's edges that were first discovered in
        the most recent quarter of the coverage clock -- edges the fuzzer only
        just reached. Seeds anchored there are the ones adjacent to whatever
        the search is currently opening up, which is what this weight is for.

        This previously counted, for each edge, whether any *uncovered* edge id
        lay within ``radius`` positions. Edge ids come out of
        ``caller_ctx ^ prev_loc ^ cur_loc``, so id adjacency is a property of
        the XOR rather than of the program, and the result carried no signal:
        measured over a 30-seed corpus with realistic ids it returned exactly
        1.0 for every seed, making ``w *= 0.5 + cov`` a uniform 1.5x. It also
        cost O(|seed_edges| * radius) per call plus a scan of the whole id
        space to build the uncovered set.

        Returns a weight in [0.0, 1.0]:
        - 0.0 = every edge is long-established code
        - 1.0 = the seed lives entirely on newly discovered edges

        ``radius`` is accepted and ignored; it described the old id-space
        window and has no meaning on the coverage clock.
        """
        seed_edges = self.seed_edges.get(seed_key)
        if not seed_edges or not self._edge_first_seen:
            return 0.5

        first_seen = self._edge_first_seen
        clock = max(first_seen.values())
        if clock <= 0:
            return 0.5
        # The clock is len(cumulative_edges) at discovery time, so this is the
        # last quarter of coverage growth rather than the last quarter of wall
        # time -- a long plateau does not age the frontier out.
        cutoff = clock * (1.0 - _FRONTIER_FRACTION)

        recent = sum(1 for e in seed_edges if first_seen.get(e, 0) >= cutoff)
        return recent / len(seed_edges)

    def compute_corpus_diversity(self) -> float:
        """Estimate corpus diversity using MinHash signatures.

        Instead of O(n²) pairwise Wasserstein, computes average MinHash
        distance from each seed to the corpus centroid. This is O(n) and
        gives a diversity estimate correlated with the true pairwise metric.

        Returns a value in [0, 1] where:
        - 0 = all seeds have identical edge sets
        - 1 = seeds are maximally diverse
        """
        keys = list(self.seed_hit_counts.keys())
        if len(keys) < 2:
            return 0.0

        corpus_sig = self._minhash.corpus_minhash()
        total_dist = 0.0
        for key in keys:
            sig = self._minhash.signatures.get(key)
            if sig:
                # Jaccard distance = 1 - Jaccard similarity
                jaccard = _sig_matches(sig, corpus_sig) / self._minhash.num_perm
                total_dist += 1.0 - jaccard

        return total_dist / len(keys) if keys else 0.0

    def compute_byte_level_diversity(self) -> float:
        """Compute average pairwise Hamming distance across seed bytes.

        Uses hamming_distance_padded to handle seeds of different lengths.
        Returns value in [0, 1] where 0 = all seeds identical at byte level.
        """
        from fuzzer_tool.core.similarity import hamming_distance_padded

        keys = list(self.seed_hit_counts.keys())
        if len(keys) < 2:
            return 0.0
        n = len(keys)
        total_dist = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                a = self.seed_hit_counts[keys[i]].get("seed_bytes", b"")
                b = self.seed_hit_counts[keys[j]].get("seed_bytes", b"")
                if isinstance(a, str):
                    a = a.encode()
                if isinstance(b, str):
                    b = b.encode()
                total_dist += hamming_distance_padded(a, b)
                count += 1
        return total_dist / max(1, count) if count else 0.0

    def compute_average_jaccard(self) -> float:
        """Compute average pairwise Jaccard similarity across all seeds.

        Uses MinHash signatures for O(1) per pair. Returns value in [0, 1]
        where 0 = all seeds have disjoint edge sets, 1 = all identical.

        This is the Jaccard index exposed as a metric — useful for monitoring
        corpus redundancy over time. High average Jaccard means the corpus
        is heavily redundant; low means seeds cover diverse code regions.

        Computed by the pair-counting identity rather than pairwise. The
        quantity wanted is

            mean_{i<j} (1/p) * sum_k [s_ik == s_jk]

        Swapping the two sums moves the work from "for every pair, walk the
        signature" to "for every signature column, count how the pairs fall
        out of the value multiplicities":

            = (1 / (p * C(n,2))) * sum_k sum_v C(m_kv, 2)

        where m_kv is how many seeds carry value v in column k. Same number
        (this is an identity, not an approximation — the two paths agree
        bit-for-bit), but O(n*p) instead of O(n^2 * p), and no (n, n, p)
        intermediate. That intermediate was the real problem: at n=2000
        seeds and p=128 the broadcast allocated 512 MB to produce one float,
        and stats.py calls this on the display path.
        """
        keys = list(self._minhash.signatures.keys())
        if len(keys) < 2:
            return 0.0

        n = len(keys)
        num_perm = self._minhash.num_perm

        if _HAS_NUMPY and n > 20:
            sigs = np.array([self._minhash.signatures[k] for k in keys], dtype=np.uint64)
            matching_pairs = 0
            for col in range(sigs.shape[1]):
                # Multiplicities of each distinct value in this column; a value
                # shared by m seeds contributes C(m,2) agreeing pairs.
                mult = np.unique(sigs[:, col], return_counts=True)[1].astype(np.int64)
                matching_pairs += int((mult * (mult - 1) // 2).sum())
            total_pairs = n * (n - 1) // 2
            return matching_pairs / (num_perm * total_pairs)

        # Pure-Python path
        total_jaccard = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_jaccard += self._minhash.approximate_jaccard(keys[i], keys[j])
                count += 1

        return total_jaccard / count if count else 0.0

    def shannon_entropy_global(self) -> float:
        """Shannon entropy of the global edge hit distribution in bits.

        H = -Σ(p_i * log2(p_i)) where p_i = hit_count_i / total_hits.
        High entropy → edges hit uniformly (good exploration).
        Low entropy → a few edges dominate (stuck in loops/hot paths).
        """
        hits = self._global_edge_hits
        if not hits:
            return 0.0
        total = sum(hits.values())
        if total == 0:
            return 0.0
        if _HAS_NUMPY and len(hits) > 50:
            arr = np.fromiter(hits.values(), dtype=np.float64)
            arr = arr[arr > 0] / total
            return -float(np.sum(arr * np.log2(arr)))
        h = 0.0
        for count in hits.values():
            if count > 0:
                p = count / total
                h -= p * math.log2(p)
        return h

    def simpson_diversity_global(self) -> float:
        """Simpson's Diversity Index of global edge hits.

        D = 1 - Σ(p_i²) where p_i = hit_count_i / total_hits.
        Value in [0, 1]:
        - 0.0 = all hits on one edge (monoculture)
        - 1.0 = perfectly uniform distribution

        Interpretable as: probability that two random hits land on
        different edges.
        """
        hits = self._global_edge_hits
        if not hits:
            return 0.0
        total = sum(hits.values())
        if total == 0:
            return 0.0
        if _HAS_NUMPY and len(hits) > 50:
            arr = np.fromiter(hits.values(), dtype=np.float64) / total
            return 1.0 - float(np.sum(arr * arr))
        sum_p_sq = sum((count / total) ** 2 for count in hits.values())
        return 1.0 - sum_p_sq

    def shannon_entropy_seed(self, seed_key: str) -> float:
        """Shannon entropy of a single seed's hit-count distribution.

        Seeds with unusual entropy profiles (very high or very low relative
        to the corpus average) are behaviorally distinct.

        Computed in counts form,

            H = log2(T) - (1/T) * sum_i c_i log2(c_i)

        which is the same number without the intermediate probability array.
        The rearrangement pays off more here than it does for byte entropy:
        hit counts arrive already bucketed by ``count_class``, so a seed
        covering thousands of edges still carries only a handful of distinct
        count values. Binning first collapses one log2 per edge into one
        log2 per distinct bucket.

        The counts form trades one rounding property for the speed: it is a
        difference of two quantities of size log2(T), so its absolute error
        is eps*log2(T) rather than eps*H. That is invisible everywhere
        except at H = 0, where the probability form returned an exact zero
        and this returns a few ulps either side of it. ``_clamp_entropy``
        pins that boundary, which is the same thing ``byte_entropy_bits``
        does and for the same reason.
        """
        hc = self.seed_hit_counts.get(seed_key)
        if not hc:
            return 0.0
        total = sum(hc.values())
        if total == 0:
            return 0.0
        if _HAS_NUMPY and len(hc) > 50:
            arr = np.fromiter(hc.values(), dtype=np.int64, count=len(hc))
            arr = arr[arr > 0]
            if arr.size == 0:
                return 0.0
            # bincount over the bucketed values, then one log2 per distinct
            # bucket rather than one per edge.
            mult = np.bincount(arr)
            nz = np.flatnonzero(mult)
            acc = float(np.sum(mult[nz] * nz * np.log2(nz)))
            return _clamp_entropy(math.log2(total) - acc / total)
        acc = 0.0
        for count in hc.values():
            if count > 0:
                acc += count * math.log2(count)
        return _clamp_entropy(math.log2(total) - acc / total)

    def compute_wasserstein_weight(self, seed_key: str) -> float:
        """Compute scheduling weight based on Wasserstein distance to corpus centroid.

        Seeds whose coverage profile is far from the corpus average (high
        Wasserstein distance to the aggregate) are spatially diverse and
        should be explored more. Seeds clustered near the centroid are
        redundant in terms of coverage location.

        Returns a weight in [0.5, 2.0]:
        - 0.5 = profile is at the centroid (spatially redundant)
        - 2.0 = profile is far from centroid (spatially novel)
        """
        hc = self.seed_hit_counts.get(seed_key)
        if not hc:
            return 1.0

        if self._aggregate_total_count == 0:
            return 1.0

        # Raw counts, not a normalised mass: the count is the coordinate.
        wasserstein = self._wasserstein_vs_aggregate(hc)

        # Normalize against the width of the hit-count axis, not map_size:
        # the distance is no longer measured in edge-index units.
        normalized = min(wasserstein / self._profile_axis_span(), 1.0)
        # Scale to [0.5, 2.0]
        return 0.5 + 1.5 * normalized

    def compute_hamming_bitmap_distance(self, seed_key_a: str, seed_key_b: str) -> float:
        """Bit-level Hamming distance between two seeds' edge bitmaps.

        The bitmaps are indicator vectors over the edge universe, so their
        Hamming distance is the size of the symmetric difference:

            |A XOR B| = |A| + |B| - 2|A AND B|

        which ``len(edges_a ^ edges_b)`` evaluates directly. No bitmap is
        built at all, and the answer is exact rather than an approximation
        of one.

        This used to pack both sets into byte bitmaps and hand them to
        ``similarity.hamming_distance``, which counts differing *byte*
        positions -- while the divisor below counts *bits*. Numerator in
        bytes over a denominator in bits undercounts by up to 8x, and it
        collapses distinctions: eight edges differing inside one byte
        scored exactly the same as one edge differing (0.0096 in both
        cases on a 13-byte map), and a 64-edge difference scored 0.0769
        where the true distance is 0.6154.

        Note on the divisor, which is unchanged here and is a separate
        question: it is the bitmap width, so it grows with the largest
        edge *id* while the numerator grows with the number of *live*
        edges. On a 256 KiB map two seeds carrying ~1800 edges each cannot
        exceed ~0.014 however different they are, so the 0.05 default in
        find_near_duplicate_seeds is not a binding threshold on a real
        target -- the MinHash LSH pre-filter is what actually selects the
        pairs. Making it binding means normalising by |A OR B| instead
        (i.e. Jaccard distance), which changes what the number means and
        wants its own calibration rather than riding along with a bug fix.

        Args:
            seed_key_a: First seed key.
            seed_key_b: Second seed key.

        Returns:
            Normalized Hamming distance in [0.0, 1.0].
                0.0 = identical bitmaps, 1.0 = all bits differ.
        """
        edges_a = self.seed_edges.get(seed_key_a, set())
        edges_b = self.seed_edges.get(seed_key_b, set())
        if not edges_a and not edges_b:
            return 0.0
        if not edges_a or not edges_b:
            return 1.0

        # Width of the bitmap the two sets would have occupied, kept
        # byte-aligned so the scale matches what callers were calibrated on.
        max_edge = max(max(edges_a), max(edges_b)) + 1
        bits = ((max_edge + 7) // 8) * 8
        return len(edges_a ^ edges_b) / bits if bits > 0 else 0.0

    def find_near_duplicate_seeds(self, max_hamming: float = 0.05) -> list[tuple[str, str, float]]:
        """Find pairs of seeds with near-identical edge bitmaps.

        Uses Hamming distance on edge bitmaps. Only checks seed pairs that
        share a MinHash LSH bucket for sub-linear performance.

        Args:
            max_hamming: Maximum normalized Hamming distance to report.

        Returns:
            List of (seed_key_a, seed_key_b, hamming_distance) tuples.
        """
        candidates = set()
        keys = list(self.seed_edges.keys())
        for key in keys:
            similar = self._minhash.find_similar(key, min_jaccard=0.5)
            for other in similar:
                pair = tuple(sorted((key, other)))
                candidates.add(pair)

        results = []
        for a, b in candidates:
            hdist = self.compute_hamming_bitmap_distance(a, b)
            if 0 < hdist <= max_hamming:
                results.append((a, b, hdist))

        results.sort(key=lambda x: x[2])
        return results

    def find_similar_seeds(self, seed_key: str, min_jaccard: float = 0.3) -> set[str]:
        """Find seeds with approximate Jaccard >= min_jaccard via LSH buckets.

        Thin wrapper around MinHashLSH.find_similar() for use by GA speciation.

        Args:
            seed_key: Seed to find similar ones for.
            min_jaccard: Minimum Jaccard similarity threshold.

        Returns:
            Set of similar seed_keys (excluding the query seed itself).
        """
        return self._minhash.find_similar(seed_key, min_jaccard=min_jaccard)

    def compute_overlap_density(
        self,
        seed_keys: list[str],
        min_jaccard: float = 0.25,
        cohesion_threshold: float = 0.3,
    ) -> tuple[dict[str, float], list[list[int]], dict[int, int]]:
        """FMM-clustered pairwise overlap density for the given seeds.

        Delegates to :func:`overlap_density.compute_corpus_overlap_density`.

        Args:
            seed_keys: Seed content hashes (must be in the MinHash index).
            min_jaccard: Jaccard threshold for LSH clustering (default: 0.25).
            cohesion_threshold: Minimum cluster cohesion to use the centroid
                approximation for a far cluster (default: 0.3).

        Returns:
            (densities_dict, clusters, seed_to_cluster) — see the delegated
            function for details.
        """
        from fuzzer_tool.core.overlap_density import compute_corpus_overlap_density

        return compute_corpus_overlap_density(
            seed_keys, self._minhash, min_jaccard, cohesion_threshold
        )

    def get_cumulative_edge_count(self) -> int:
        """Get total unique edges seen across all seeds.

        Returns the max of the current set and the preserved total,
        so the count survives bitmap resizes.
        """
        return max(len(self.cumulative_edges), self._cumulative_edges_total)

    def _rebuild_frequency_spectrum(self):
        """Rebuild the *abundance* frequency spectrum from global edge hits (lazy).

        This is a spectrum over bucketed execution hit volume. It no longer
        feeds good_turing_estimate(), which is an incidence estimator and reads
        _edge_owner_count instead; nothing else consumes it yet. Kept because
        the hit-volume spectrum is the right input for Zipf-slope analysis of
        loop structure, which is a separate question from richness.
        """
        if not self._spectrum_dirty:
            return
        if _HAS_NUMPY:
            vals, cnts = np.unique(
                np.fromiter(
                    self._global_edge_hits.values(),
                    np.uint64,
                    len(self._global_edge_hits),
                ),
                return_counts=True,
            )
            self._frequency_spectrum = dict(zip(map(int, vals), map(int, cnts), strict=True))
        else:
            self._frequency_spectrum.clear()
            for count in self._global_edge_hits.values():
                self._frequency_spectrum[count] = self._frequency_spectrum.get(count, 0) + 1
        self._spectrum_dirty = False

    def good_turing_estimate(self) -> dict:
        """Estimate undiscovered edges from the *incidence* frequency spectrum.

        The sampling unit is the corpus seed, and Q_k is the number of edges
        reached by exactly k distinct seeds. That is the model Chao2 is built
        for, and it is the one the coverage data actually supports.

        This used to read ``_global_edge_hits``, which sums AFL-bucketed hit
        counters (1, 2, 4, 8, 16, 32, 128) across every execution. Under that
        map "seen exactly twice" means the bucket values happened to add to
        two -- a lattice artifact of the bucketing and of how often the seed
        was re-run, not a doubleton in any species-sampling sense. The damping
        factor and the 5*N cap the old code needed were compensating for the
        wrong input rather than for anything about fuzzing.

        Estimators (Chao 1987; Chao & Chiu 2016), with m sampling units and
        A = (m - 1) / m:

          Chao2   = S_obs + A * Q1^2 / (2*Q2)                     (Q2 >= 10)
          Chao2bc = S_obs + A * Q1*(Q1 - 1) / (2*(Q2 + 1))        (Q2 < 10)

        The bias-corrected form is the principled version of what the damping
        hack was approximating: it stays finite at Q2 = 0 and is insensitive
        to a single doubleton flipping.

        Sample coverage (Chao & Jost 2012), with U = sum of incidences:

          C = 1 - (Q1/U) * (m-1)*Q1 / ((m-1)*Q1 + 2*Q2)

        1 - C estimates the probability that the next seed reaches code
        nothing in the corpus reaches yet -- a direct, unbiased discovery-rate
        signal rather than a proxy for one.

        Returns dict with:
          - n: total distinct edges observed (S_obs)
          - n1, n2: Q1 (edges reached by one seed), Q2 (by exactly two)
          - m: number of sampling units (seeds) the spectrum is built from
          - estimated_undiscovered: Chao2 richness minus S_obs
          - saturation: S_obs / Chao2
          - sample_coverage: C above
          - discovery_probability: 1 - C
          - ci_low, ci_high: log-transformed 95% interval on total richness
          - confidence: low/medium/high from the relative width of that interval
        """
        n = len(self.cumulative_edges)
        owners = self._edge_owner_count or {}
        m = len(self.seed_edges)
        empty = {
            "n": n,
            "n1": 0,
            "n2": 0,
            "m": m,
            "estimated_undiscovered": 0,
            "chao2": float(n),
            "saturation": 1.0 if n else 0.0,
            "sample_coverage": 1.0 if n else 0.0,
            "discovery_probability": 0.0,
            "ci_low": float(n),
            "ci_high": float(n),
            "confidence": "low",
        }
        if n == 0 or not owners:
            empty["saturation"] = 0.0
            return empty
        # A single sampling unit carries no information about what a second one
        # would add: every edge is a singleton by construction.
        if m < 2:
            return empty

        counts = list(owners.values())
        q1 = sum(1 for c in counts if c == 1)
        q2 = sum(1 for c in counts if c == 2)
        incidences = sum(counts)
        s_obs = len(counts)

        a = (m - 1) / m
        if q2 >= _CHAO_BIAS_CORRECT_BELOW:
            f0 = a * (q1 * q1) / (2.0 * q2)
        else:
            f0 = a * (q1 * (q1 - 1)) / (2.0 * (q2 + 1))
        chao2 = s_obs + f0

        # Chao (1987) variance of the richness estimator.
        if q2 > 0:
            r = q1 / q2
            var = q2 * ((a / 2.0) * r**2 + (a**2) * r**3 + (a**2 / 4.0) * r**4)
        elif q1 > 0 and chao2 > 0:
            var = (
                a * q1 * (q1 - 1) / 2.0
                + (a**2) * q1 * (2 * q1 - 1) ** 2 / 4.0
                - (a**2) * q1**4 / (4.0 * chao2)
            )
            var = max(var, 0.0)
        else:
            var = 0.0

        # Log transform (Chao 1987): asymmetric, and never puts the lower bound
        # below the number of edges actually observed.
        if f0 > 0 and var > 0:
            k = math.exp(1.96 * math.sqrt(math.log(1.0 + var / (f0 * f0))))
            ci_low = s_obs + f0 / k
            ci_high = s_obs + f0 * k
        else:
            ci_low = ci_high = float(chao2)

        # Chao & Jost incidence-based sample coverage.
        denom = (m - 1) * q1 + 2 * q2
        if incidences > 0 and denom > 0:
            coverage = 1.0 - (q1 / incidences) * ((m - 1) * q1 / denom)
        else:
            coverage = 1.0
        coverage = min(max(coverage, 0.0), 1.0)

        saturation = s_obs / chao2 if chao2 > 0 else 1.0
        # Confidence now reflects how tightly the data pins the total, which is
        # what the word meant all along; the old N1/N ratio was a proxy.
        rel_width = (ci_high - ci_low) / chao2 if chao2 > 0 else float("inf")
        if rel_width < 0.25:
            confidence = "high"
        elif rel_width < 1.0:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "n": n,
            "n1": q1,
            "n2": q2,
            "m": m,
            "estimated_undiscovered": int(f0),
            # Exact point estimate. estimated_undiscovered is truncated to an
            # int for the existing display callers, so n + that is a slightly
            # low reading of the same quantity.
            "chao2": chao2,
            "saturation": saturation,
            "sample_coverage": coverage,
            "discovery_probability": 1.0 - coverage,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence": confidence,
        }

    def bitmap_density(self) -> float:
        """Fraction of the hash table entries that have been hit (0.0 to 1.0).

        For the sparse hash table this is the load factor — entries with
        non-zero edge_id divided by total entries.  High load factor
        (> 0.7) means longer probe chains.
        """
        return len(self._global_edge_hits) / self.map_size if self.map_size else 0.0

    def birthday_collision_risk(self) -> float:
        """Collision probability for the current edge count.

        With the sparse hash table (open-addressing with unique edge_ids),
        there are NO silent collisions — two edges always have different
        edge_ids.  Returns 0.  This method exists for backward compat
        with the old byte-bitmap approach.
        """
        return 0.0

    def recommended_map_size(self, dropped_edges: int = 0) -> int:
        """Recommend a larger map_size if the hash table is too full.

        Args:
            dropped_edges: Edges the shim had to discard because the probe
                found no free slot (ShmCoverage.read_dropped_edges()). Any
                non-zero value is proof of saturation.

        Returns:
            Recommended map_size (entries), or 0 if current size is adequate.

        Two independent triggers, because the cheap one is blind in the case
        that matters most.

        Observed load factor is computed from _global_edge_hits, which only
        ever contains edges that made it into the table. A table full enough
        to be DISCARDING edges therefore reports a load factor that stops
        rising -- it looks healthy precisely when it is worst, so the 0.7
        threshold alone can never fire on a saturated map. That is why the
        stall-triggered resize never helped the targets it was written for.

        dropped_edges comes from the shim, counted where the loss actually
        happens, and needs no threshold: one dropped edge is one edge of
        coverage the run will never see.
        """
        n = len(self._global_edge_hits)
        if not self.map_size:
            return 0

        load_factor = n / self.map_size
        saturated = dropped_edges > 0
        if not saturated and (n < 100 or load_factor < 0.7):
            return 0

        # Under saturation the observed count is a floor, not a measurement:
        # the true edge count is n + (at least) dropped_edges, and drops are
        # only counted once the table is already full. Size from the floor
        # and expect to be called again -- converging upward over a few
        # resizes beats one speculative jump to the cap.
        target_edges = n + dropped_edges if saturated else n
        needed = int(target_edges * 2)  # ~2x headroom -> load factor < 0.5

        recommended = 1
        while recommended < needed:
            recommended *= 2
        recommended = max(MAP_SIZE_DEFAULT, min(_map_size_max(), recommended))
        if recommended <= self.map_size:
            return 0
        return recommended

    def get_seed_edge_count(self, seed_key: str) -> int:
        """Get number of edges a specific seed covers."""
        return len(self.seed_edges.get(seed_key, set()))

    def get_seed_stack_depth(self, seed_key: str) -> int:
        """Get the max stack depth (bytes) for a seed."""
        return self.seed_stack_depth.get(seed_key, 0)

    def get_seed_path_hash(self, seed_key: str) -> int:
        """Get the rolling path hash for a seed."""
        return self.seed_path_hash.get(seed_key, 0)

    def get_seed_hw_instructions(self, seed_key: str) -> int:
        """Get hardware instruction count for a seed."""
        return self.seed_hw_instructions.get(seed_key, 0)

    def get_seed_hw_branches(self, seed_key: str) -> int:
        """Get hardware branch count for a seed."""
        return self.seed_hw_branches.get(seed_key, 0)

    def get_seed_hw_branch_misses(self, seed_key: str) -> int:
        """Get hardware branch miss count for a seed."""
        return self.seed_hw_branch_misses.get(seed_key, 0)

    def rare_edge_count(self, seed_key: str, threshold: int = 4) -> int:
        """Count how many edges hit by this seed are 'rare' (seen by <threshold seeds).

        Rare edges are those that few corpus entries hit — inputs that
        exercise rare edges get an energy boost in the honggfuzz power schedule.

        Args:
            seed_key: Hash of the seed input.
            threshold: Edges hit by fewer seeds than this are considered rare.

        Returns:
            Number of rare edges hit by this seed.
        """
        edges = self.seed_edges.get(seed_key, set())
        count = 0
        for eid in edges:
            if self._edge_owner_count[eid] < threshold:
                count += 1
        return count

    def edge_owner_count(self, edge_id: int) -> int:
        """How many distinct corpus seeds cover ``edge_id``.

        This is the per-*edge* companion to :meth:`rare_edge_count`, which is
        keyed by *seed*. The two take different key spaces and are easy to
        confuse; callers ordering edges by rarity want this one.

        Returns 0 for an edge no seed has covered.

        Read with ``.get`` rather than a bare subscript: the backing map is a
        defaultdict, so subscripting *inserts* a zero entry for every edge
        queried. That turns a read-only accessor into a mutating one and grows
        the map without bound -- fuzzer.py sorts candidate edges through here,
        and most of those edges are unowned by construction.
        """
        return self._edge_owner_count.get(edge_id, 0)

    def edge_rarity_stats(self) -> dict:
        """Compute per-edge rarity statistics, in units of *seeds*.

        Rarity here means "how many corpus seeds reach this edge" -- that is
        what makes an edge lossy to prune. The counts come from
        _edge_owner_count, which is incremented once per seed that covers an
        edge. This previously read _global_edge_hits, which accumulates the
        bucketed SHM hit *counters* (a single execution can add hundreds for a
        hot edge), so the buckets described execution volume while claiming to
        describe seeds and avg_seeds_per_edge could exceed the corpus size --
        the ffmpeg_read_nosan run reported 2502.4 seeds per edge from a
        489-seed corpus, which is over five times the arithmetic ceiling.

        Returns dict with:
          - total: number of discovered edges
          - singleton: edges covered by exactly 1 seed (lossy if pruned)
          - cold: edges covered by 2-3 seeds (fragile coverage)
          - warm: edges covered by 4-10 seeds
          - hot: edges covered by >10 seeds (redundant -- safe to prune)
          - avg_seeds_per_edge: mean number of seeds covering an edge
          - bounds: (cold_hi, warm_hi) thresholds actually applied, so the
            report can label the buckets instead of hardcoding them
        """
        owners = self._edge_owner_count or {}
        if not owners:
            return {
                "total": 0,
                "singleton": 0,
                "cold": 0,
                "warm": 0,
                "hot": 0,
                "avg_seeds_per_edge": 0.0,
                "bounds": (3, 10),
            }

        counts = list(owners.values())
        total = len(counts)
        # Owner counts are exact integers regardless of Morris mode -- Morris
        # approximation applies to hit counters, not to per-seed ownership --
        # so a single set of thresholds is correct here.
        cold_hi, warm_hi = 3, 10
        if _HAS_NUMPY:
            arr = np.fromiter(counts, np.int64)
            singleton = int(np.count_nonzero(arr == 1))
            cold = int(np.count_nonzero((arr >= 2) & (arr <= cold_hi)))
            warm = int(np.count_nonzero((arr > cold_hi) & (arr <= warm_hi)))
            hot = int(np.count_nonzero(arr > warm_hi))
            avg = float(arr.sum()) / total if total else 0.0
        else:
            singleton = sum(1 for c in counts if c == 1)
            cold = sum(1 for c in counts if 2 <= c <= cold_hi)
            warm = sum(1 for c in counts if cold_hi < c <= warm_hi)
            hot = sum(1 for c in counts if c > warm_hi)
            avg = sum(counts) / total if total else 0.0

        return {
            "total": total,
            "singleton": singleton,
            "cold": cold,
            "warm": warm,
            "hot": hot,
            "avg_seeds_per_edge": avg,
            "bounds": (cold_hi, warm_hi),
        }

    def edge_hit_distribution(self) -> dict[int, dict]:
        """Per-edge hit statistics across all seeds.

        Returns dict mapping edge_index -> {
            "hit_count": total hit count (sum of all seed hit counts),
            "seed_count": number of distinct seeds hitting this edge,
            "mean_hit_per_seed": average hits per seed,
        }
        """
        result = {}
        for edge, total_hits in self._global_edge_hits.items():
            # Count distinct seeds hitting this edge
            seed_count = 0
            for _seed_key, hc in self.seed_hit_counts.items():
                if edge in hc:
                    seed_count += 1
            mean_per_seed = total_hits / seed_count if seed_count > 0 else 0.0
            result[edge] = {
                "hit_count": total_hits,
                "seed_count": seed_count,
                "mean_hit_per_seed": mean_per_seed,
            }
        return result

    def edge_cooccurrence(self, top_k: int = 10) -> list[tuple[int, int, float]]:
        """Find edges that co-occur most frequently in seeds.

        Returns list of (edge_a, edge_b, jaccard_similarity) sorted
        by similarity descending. Only considers edges hit by >= 2 seeds.
        """
        # Build edge -> seed set mapping
        edge_to_seeds: dict[int, set[str]] = {}
        for seed_key, edges in self.seed_edges.items():
            for e in edges:
                if e not in edge_to_seeds:
                    edge_to_seeds[e] = set()
                edge_to_seeds[e].add(seed_key)

        # Only consider edges with >= 2 seeds
        common = {e: s for e, s in edge_to_seeds.items() if len(s) >= 2}
        edges = list(common.keys())

        pairs = []
        for i in range(min(len(edges), 200)):  # cap for performance
            for j in range(i + 1, min(len(edges), 200)):
                a, b = edges[i], edges[j]
                intersection = len(common[a] & common[b])
                union = len(common[a] | common[b])
                if union > 0:
                    jaccard = intersection / union
                    if jaccard > 0.1:  # only meaningful co-occurrences
                        pairs.append((a, b, jaccard))

        pairs.sort(key=lambda x: x[2], reverse=True)
        del edge_to_seeds, common  # free bipartite map before return
        return pairs[:top_k]

    def seed_uniqueness(self) -> dict[str, int]:
        """For each seed, count how many edges ONLY it covers.

        Returns dict mapping seed_key -> number of singleton edges.
        Seeds with high uniqueness are irreplaceable.
        """
        # Build edge -> [seeds] mapping
        edge_seeds: dict[int, list[str]] = {}
        for seed_key, edges in self.seed_edges.items():
            for e in edges:
                if e not in edge_seeds:
                    edge_seeds[e] = []
                edge_seeds[e].append(seed_key)

        # Singleton edges (hit by exactly 1 seed)
        singletons = {e: seeds[0] for e, seeds in edge_seeds.items() if len(seeds) == 1}

        # Count singletons per seed
        result = defaultdict(int)
        for _edge, seed_key in singletons.items():
            result[seed_key] += 1
        del edge_seeds, singletons  # free bipartite map + singletons dict
        return dict(result)

    def classify_seeds(self) -> dict[str, dict]:
        """Classify seeds as keystone, useful, parasitic, or redundant.

        Classification:
        - keystone: covers edges no other seed covers (singleton edges > 0)
        - useful: contributes edges shared with others but not fully subsumed
        - parasitic: all edges covered by other seeds (subsumption weight < 0.1)
        - redundant: similar to parasitic but edge count is very low

        Returns:
            Dict mapping seed_key -> {classification, singleton_edges, edge_count, subsumption_weight}
        """
        uniqueness = self.seed_uniqueness()
        result = {}

        for seed_key, edges in self.seed_edges.items():
            singleton_count = uniqueness.get(seed_key, 0)
            edge_count = len(edges)

            # Compute subsumption weight
            weight = self.compute_subsumption_weight(seed_key) if self._corpus_sig else 1.0

            # Classify
            if singleton_count > 0:
                classification = "keystone"
            elif weight < 0.1:
                classification = "parasitic"
            elif edge_count < 5:
                classification = "redundant"
            else:
                classification = "useful"

            result[seed_key] = {
                "classification": classification,
                "singleton_edges": singleton_count,
                "edge_count": edge_count,
                "subsumption_weight": weight,
            }

        del uniqueness  # free singleton-count dict after loop
        return result

    def seed_contribution_graph(self) -> dict[str, dict]:
        """Build a bipartite seed↔edge contribution graph.

        Returns:
            Dict with:
            - seed_to_edges: {seed_key: [edge_indices]}
            - edge_to_seeds: {edge_index: [seed_keys]}
            - keystone_seeds: [seed_keys with singleton edges]
            - parasitic_seeds: [seed_keys fully subsumed]
        """
        # Build edge -> seeds mapping
        edge_to_seeds: dict[int, list[str]] = {}
        for seed_key, edges in self.seed_edges.items():
            for e in edges:
                if e not in edge_to_seeds:
                    edge_to_seeds[e] = []
                edge_to_seeds[e].append(seed_key)

        # Classify seeds
        classifications = self.classify_seeds()
        keystone = [k for k, v in classifications.items() if v["classification"] == "keystone"]
        parasitic = [k for k, v in classifications.items() if v["classification"] == "parasitic"]

        result = {
            "seed_to_edges": {k: sorted(v) for k, v in self.seed_edges.items()},
            "edge_to_seeds": {e: s for e, s in edge_to_seeds.items()},
            "keystone_seeds": keystone,
            "parasitic_seeds": parasitic,
        }
        del edge_to_seeds  # free bipartite map
        return result

    def coverage_dominance_tree(self) -> dict[str, list[str]]:
        """Build a coverage dominance tree.

        Seed A dominates seed B if edge(A) is a strict subset of edge(B).
        Returns dict mapping seed_key -> list of seeds it dominates.

        Uses MinHash for approximate subset checks on large edge sets,
        exact checks for small sets (< 100 edges).

        Returns:
            Dict mapping dominator -> list of dominated seeds.
        """
        dominance: dict[str, list[str]] = {k: [] for k in self.seed_edges}

        # Sort seeds by edge count (ascending) for efficiency
        sorted_seeds = sorted(
            self.seed_edges.items(),
            key=lambda x: len(x[1]),
        )

        for i, (key_a, edges_a) in enumerate(sorted_seeds):
            if not edges_a:
                continue
            for j in range(i + 1, len(sorted_seeds)):
                key_b, edges_b = sorted_seeds[j]
                if not edges_b:
                    continue

                # Check if edges_a ⊂ edges_b (A dominated by B)
                if len(edges_a) > len(edges_b):
                    continue

                # For small sets, use exact check
                if len(edges_a) <= 100 and len(edges_b) <= 100:
                    is_subset = edges_a.issubset(edges_b)
                else:
                    # Use MinHash approximation
                    if self._corpus_sig is None:
                        self._corpus_sig = self._minhash.corpus_minhash()
                    jaccard_ab = self._minhash.approximate_jaccard(key_a, key_b)
                    # If Jaccard is high and |A| <= |B|, likely subset
                    is_subset = jaccard_ab > 0.8 and len(edges_a) <= len(edges_b)

                if is_subset:
                    # A is dominated by B
                    dominance[key_b].append(key_a)

        # Remove empty entries
        return {k: v for k, v in dominance.items() if v}

    def find_redundant_seeds(self) -> list[str]:
        """Find seeds that are fully dominated by other seeds.

        Returns:
            List of seed_keys that are redundant (can be removed).
        """
        tree = self.coverage_dominance_tree()
        redundant = set()
        for dominated_list in tree.values():
            redundant.update(dominated_list)
        return sorted(redundant)

    def to_dict(self) -> dict:
        """Serialize tracker state to a dict (for StateStore pickle)."""
        return {
            "map_size": self.map_size,
            "morris_mode": self._morris_mode,
            "cumulative_edges": sorted(self.cumulative_edges),
            "seed_edges": {k: sorted(v) for k, v in self.seed_edges.items()},
            "seed_hit_counts": {
                k: {str(e): c for e, c in hc.items()} for k, hc in self.seed_hit_counts.items()
            },
            "global_edge_hits": {str(e): c for e, c in self._global_edge_hits.items()},
            "minhash_sigs": {k: sig.tolist() for k, sig in self._minhash.signatures.items()},
            "aggregate_totals": {str(e): c for e, c in self._aggregate_totals.items()},
            "aggregate_total_count": self._aggregate_total_count,
            "edge_traces": {k: [list(e) for e in v] for k, v in self.seed_edge_traces.items()},
            "edge_first_seen": {str(e): c for e, c in self._edge_first_seen.items()},
            "edge_last_seen": {str(e): c for e, c in self._edge_last_seen.items()},
            "coverage_timeline": [
                list(p) for p in zip(self._coverage_execs, self._coverage_edges, strict=True)
            ],
            "correlation_matrix": {f"{a},{b}": c for (a, b), c in self._correlation_matrix.items()},
            "correlation_total": self._correlation_total,
            "seed_stack_depth": self.seed_stack_depth,
            "seed_path_hash": self.seed_path_hash,
            "edge_owner_count": {str(e): c for e, c in self._edge_owner_count.items()},
            "seed_hw_instructions": self.seed_hw_instructions,
            "seed_hw_branches": self.seed_hw_branches,
            "seed_hw_branch_misses": self.seed_hw_branch_misses,
        }

    def from_dict(self, data: dict) -> None:
        """Restore tracker state from a serialized dict."""
        self.map_size = data.get("map_size", self.map_size)
        self._morris_mode = data.get("morris_mode", self._morris_mode)
        self.cumulative_edges = set(data.get("cumulative_edges", []))
        self.seed_edges = {k: set(v) for k, v in data.get("seed_edges", {}).items()}
        self.seed_hit_counts = {
            k: {int(e): c for e, c in hc.items()}
            for k, hc in data.get("seed_hit_counts", {}).items()
        }
        self._global_edge_hits = {int(e): c for e, c in data.get("global_edge_hits", {}).items()}
        self._spectrum_dirty = True
        self._aggregate_cache = None
        self._aggregate_totals = {int(e): c for e, c in data.get("aggregate_totals", {}).items()}
        self._aggregate_total_count = data.get("aggregate_total_count", 0)
        if not self._aggregate_totals and self.seed_hit_counts:
            for hc in self.seed_hit_counts.values():
                for edge, count in hc.items():
                    self._aggregate_totals[edge] = self._aggregate_totals.get(edge, 0) + count
                    self._aggregate_total_count += count
        self.seed_edge_traces = {
            k: {(e[0], e[1]) for e in v} for k, v in data.get("edge_traces", {}).items()
        }
        self._edge_first_seen = {int(e): c for e, c in data.get("edge_first_seen", {}).items()}
        self._edge_last_seen = {int(e): c for e, c in data.get("edge_last_seen", {}).items()}
        tl = data.get("coverage_timeline", [])
        self._coverage_execs = array("Q", (t[0] for t in tl))
        self._coverage_edges = array("Q", (t[1] for t in tl))
        corr_data = data.get("correlation_matrix", {})
        self._correlation_matrix = {
            (int(k.split(",")[0]), int(k.split(",")[1])): v for k, v in corr_data.items()
        }
        self._correlation_total = data.get("correlation_total", 0)
        self.seed_stack_depth = data.get("seed_stack_depth", {})
        self.seed_path_hash = data.get("seed_path_hash", {})
        # defaultdict, not dict: __init__ establishes _edge_owner_count as a
        # defaultdict(int) and the read sites subscript it bare (owners[e]) on
        # that guarantee. Rebuilding it as a plain dict here dropped the
        # invariant at the first state restore, so every bare subscript raised
        # KeyError afterwards -- worst on a snapshot written before this field
        # existed, where seed_edges restores fully against an empty owner map
        # and rare_edge_count() dies on its first edge.
        self._edge_owner_count = defaultdict(
            int, {int(e): c for e, c in data.get("edge_owner_count", {}).items()}
        )
        self.seed_hw_instructions = data.get("seed_hw_instructions", {})
        self.seed_hw_branches = data.get("seed_hw_branches", {})
        self.seed_hw_branch_misses = data.get("seed_hw_branch_misses", {})
        # Restore MinHash signatures and rebuild LSH index
        self._minhash = MinHashLSH(num_perm=64, num_bands=8)
        self._corpus_sig = None
        saved_sigs = data.get("minhash_sigs", {})
        if saved_sigs:
            for k, sig in saved_sigs.items():
                self._minhash.add(k, sig)
        else:
            for k, edges in self.seed_edges.items():
                sig = self._minhash.compute_signature(edges)
                self._minhash.add(k, sig)

    def save(self, path: str) -> bool:
        """Save tracker state to JSON (legacy interface)."""
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"))
            log.info(
                "Edge tracker saved: %s (%d seeds, %d edges)",
                path,
                len(self.seed_edges),
                len(self.cumulative_edges),
            )
            return True
        except OSError as e:
            log.warning("Failed to save edge tracker: %s", e)
            return False

    def load(self, path: str) -> bool:
        """Load tracker state from JSON (legacy interface)."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.debug("Failed to load edge tracker: %s", e)
            return False
        self.from_dict(data)
        log.info(
            "Edge tracker loaded: %s (%d seeds, %d edges)",
            path,
            len(self.seed_edges),
            len(self.cumulative_edges),
        )
        return True
