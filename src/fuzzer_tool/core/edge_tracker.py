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

import logging
import math
import random
import struct
import zlib
from collections import defaultdict

from fuzzer_tool.core import fast_json as json

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from fuzzer_tool.core.count_class import classify_counts
from fuzzer_tool.core.similarity import hamming_distance

log = logging.getLogger(__name__)

MORRIS_A = 30


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
        # Per-seed MinHash signatures: seed_key -> list[int] of length num_perm
        self.signatures: dict[str, list[int]] = {}
        # LSH buckets: (band_idx, band_hash) -> set of seed_keys
        self.buckets: dict[tuple[int, int], set[str]] = {}
        # Precomputed hash function coefficients: a*x + b (mod large prime)
        rng = random.Random(seed)
        self._prime = (1 << 61) - 1  # Mersenne prime
        self._coeffs = [
            (rng.randint(1, self._prime - 1), rng.randint(0, self._prime - 1))
            for _ in range(num_perm)
        ]

    def compute_signature(self, edge_set: set[int]) -> list[int]:
        """Compute MinHash signature for a set of edge indices.

        Uses k independent hash functions of the form h(x) = (a*x + b) mod p,
        taking the minimum hash value across all elements in the set.
        """
        sig = [self._prime] * self.num_perm
        for edge in edge_set:
            for i, (a, b) in enumerate(self._coeffs):
                h = (int(a) * int(edge) + int(b)) % self._prime
                if h < sig[i]:
                    sig[i] = h
        return sig

    def add(self, seed_key: str, sig: list[int]):
        """Add a seed's signature to the index."""
        self.signatures[seed_key] = sig
        # Insert into LSH buckets
        for band_idx in range(self.num_bands):
            start = band_idx * self.band_size
            end = start + self.band_size
            band_bytes = struct.pack(f"<{end - start}Q", *sig[start:end])
            band_hash = zlib.crc32(band_bytes)
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
            band_hash = zlib.crc32(band_bytes)
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
        matches = sum(1 for a, b in zip(sig_a, sig_b, strict=False) if a == b)
        return matches / self.num_perm

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
            band_hash = zlib.crc32(band_bytes)
            bucket_key = (band_idx, band_hash)
            bucket = self.buckets.get(bucket_key)
            if bucket:
                candidates.update(bucket)
        candidates.discard(seed_key)

        # Filter by full Jaccard threshold
        if min_jaccard <= 0:
            return candidates
        return {k for k in candidates if self.approximate_jaccard(seed_key, k) >= min_jaccard}

    def corpus_minhash(self, seed_keys: set[str] | None = None) -> list[int]:
        """Compute MinHash of the union of all seeds' edge sets.

        The union's MinHash is the element-wise minimum of individual
        signatures — this is a property of MinHash, not an approximation.
        """
        if seed_keys is None:
            seed_keys = set(self.signatures.keys())
        if not seed_keys:
            return [self._prime] * self.num_perm
        # Start with first seed's signature as baseline
        first = next(iter(seed_keys))
        result = list(self.signatures.get(first, [self._prime] * self.num_perm))
        for key in seed_keys:
            sig = self.signatures.get(key)
            if sig:
                for i in range(self.num_perm):
                    if sig[i] < result[i]:
                        result[i] = sig[i]
        return result

    def approximate_union_jaccard(self, seed_key: str, corpus_sig: list[int]) -> float:
        """Estimate Jaccard(seed, corpus_union) using precomputed corpus signature.

        This replaces the O(n) other_edges union scan with an O(k) signature
        comparison, where k = num_perm (typically 64).
        """
        sig = self.signatures.get(seed_key)
        if sig is None:
            return 0.0
        matches = sum(1 for a, b in zip(sig, corpus_sig, strict=False) if a == b)
        return matches / self.num_perm


class EdgeTracker:
    """Track coverage edges per seed for smarter scheduling.

    After each execution that produces new coverage, records which
    edges are now hit. Seeds that contribute unique edges get higher
    priority; seeds fully subsumed by others get deprioritized.
    """

    def __init__(
        self, map_size: int = 65536, max_tracked_seeds: int = 200, morris_mode: bool = False
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
        # Aggregate distribution: maintained incrementally, not rebuilt from scratch
        self._aggregate_totals: dict[int, int] = {}
        self._aggregate_total_count: int = 0
        self._aggregate_cache: dict[int, float] | None = None
        # Good-Turing: global cumulative hit count per edge (across all seeds)
        self._global_edge_hits: dict[int, int] = {}
        self._spectrum_dirty = True
        self._frequency_spectrum: dict[int, int] = {}
        self.max_hit_count: int = 0
        # MinHash/LSH for approximate Jaccard and subsumption
        self._minhash = MinHashLSH(num_perm=64, num_bands=8)
        self._corpus_sig: list[int] | None = None
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
        self._coverage_timeline: list[tuple[int, int]] = []
        # Branch correlation matrix: {(edge_a, edge_b): co_occurrence_count}
        self._correlation_matrix: dict[tuple[int, int], int] = {}
        self._correlation_total: int = 0

    def record_edges(
        self, seed_key: str, edge_bitmap: bytes, target_name: str = "", morris_mode: bool = False
    ) -> set[int]:
        """Record edges hit by a seed execution.

        Args:
            seed_key: Hash of the seed input.
            edge_bitmap: Raw edge bitmap (bytes where > 0 = edge hit).
            target_name: Name of the target binary (for multi-target tracking).
            morris_mode: If True, convert Morris counter values to approximate counts.

        Returns:
            Set of NEW edge indices not previously seen.
        """
        new_edges = set()
        if seed_key not in self.seed_edges:
            self.seed_edges[seed_key] = set()
        if seed_key not in self.seed_hit_counts:
            self.seed_hit_counts[seed_key] = {}
        hc = self.seed_hit_counts[seed_key]

        # Classify hit counts before recording (non-Morris only) —
        # collapses noise like 47 vs 52 into the same bucket (32-127),
        # so diversity scoring (JS divergence, Wasserstein) uses
        # bucketized magnitude rather than raw jittery counts.
        # Morris mode skips classification because its own
        # morris_estimate() already handles approximate counting.
        if not morris_mode:
            edge_bitmap = classify_counts(edge_bitmap)

        # Scan only non-zero bytes via numpy flatnonzero.
        # For a 256KB bitmap with ~100 active edges this is ~2600x fewer
        # Python iterations than iterating every byte with enumerate.
        arr = np.frombuffer(edge_bitmap, dtype=np.uint8, count=min(len(edge_bitmap), self.map_size))
        for i in np.flatnonzero(arr):
            i = int(i)
            raw_val = int(arr[i])
            if morris_mode:
                val = int(round(morris_estimate(raw_val)))
            else:
                val = raw_val
            new_edges.add(i)
            hc[i] = val
            # Aggregate totals
            old_agg = self._aggregate_totals.get(i, 0)
            self._aggregate_totals[i] = old_agg + val
            self._aggregate_total_count += val
            # Global edge hits
            old_gh = self._global_edge_hits.get(i, 0)
            new_gh = old_gh + val
            self._global_edge_hits[i] = new_gh
            self._spectrum_dirty = True
            if new_gh > self.max_hit_count:
                self.max_hit_count = new_gh

        new_contributions = new_edges - self.cumulative_edges
        self.cumulative_edges.update(new_edges)
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

        # Update temporal tracking
        self.record_edge_lifetimes(new_edges, len(self.cumulative_edges))
        self.update_correlation(new_edges)

        # Prune old seeds if over limit
        self._maybe_prune()

        return new_contributions

    def reset_after_resize(self):
        """Clear all position-based state after bitmap resize.

        When the bitmap resizes, AFL's hash (edge_id = hash(src,dst) % map_size)
        maps the same logical edge to a different position. All position-based
        tracking must be cleared to avoid stale entries.
        """
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

    def _maybe_prune(self):
        """Prune oldest seeds when tracked count exceeds max_tracked_seeds.

        Keeps the most recent seeds and removes their edge data.
        This bounds memory usage while preserving recent coverage information.
        """
        if len(self.seed_edges) <= self.max_tracked_seeds:
            return

        # Find excess seeds to prune (oldest first by insertion order)
        excess = len(self.seed_edges) - self.max_tracked_seeds
        keys_to_prune = list(self.seed_edges.keys())[:excess]

        for key in keys_to_prune:
            self.seed_edges.pop(key, None)
            self.seed_hit_counts.pop(key, None)
            self.seed_edge_traces.pop(key, None)
            self._minhash.remove(key)

        self._aggregate_cache = None
        self._corpus_sig = None

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
        self._coverage_timeline.append((exec_count, len(self.cumulative_edges)))

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
        if len(self._coverage_timeline) < 2:
            return 0.0
        first_exec, first_edges = self._coverage_timeline[0]
        last_exec, last_edges = self._coverage_timeline[-1]
        exec_diff = last_exec - first_exec
        if exec_diff <= 0:
            return 0.0
        return (last_edges - first_edges) / exec_diff

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
        if len(self._coverage_timeline) < 3:
            return {
                "current_rate": 0.0,
                "projected_total": 0,
                "time_to_plateau": 0,
                "confidence": 0.0,
                "plateau_detected": False,
            }

        # Extract time series
        execs = [t[0] for t in self._coverage_timeline]
        edges = [t[1] for t in self._coverage_timeline]

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
        """Compute JS divergence between a seed's distribution and the aggregate.

        Only iterates edges where the seed has non-zero probability —
        edges where the seed is zero contribute 0 to KL(P || M).
        O(|seed_edges|) instead of O(|all_edges|).
        """
        total = self._aggregate_total_count
        if total == 0:
            return 0.0

        js = 0.0
        for e, p in seed_dist.items():
            if p <= 0.0:
                continue
            q = self._aggregate_totals.get(e, 0.0) / total
            m = 0.5 * (p + q)
            if m > 0.0:
                js += p * math.log(p / m)
                if q > 0.0:
                    js += q * math.log(q / m)
        return 0.5 * js

    def _wasserstein_vs_aggregate(self, seed_dist: dict[int, float]) -> float:
        """Compute Wasserstein-1 distance between a seed and the aggregate centroid.

        Only iterates edges present in the seed (aggregate-only edges
        contribute zero CDF difference since seed CDF is flat there).
        O(|seed_edges|) instead of O(|all_edges|).
        """
        total = self._aggregate_total_count
        if total == 0:
            return float(self.map_size)

        all_edges = sorted(seed_dist.keys())
        if not all_edges:
            return 0.0

        cdf_diff = 0.0
        wasserstein = 0.0
        prev_edge = all_edges[0]

        for edge in all_edges:
            gap = edge - prev_edge
            wasserstein += abs(cdf_diff) * gap
            p = seed_dist.get(edge, 0.0)
            q = self._aggregate_totals.get(edge, 0.0) / total
            cdf_diff += p - q
            prev_edge = edge

        # Account for remaining aggregate mass after last seed edge
        if cdf_diff != 0.0:
            wasserstein += abs(cdf_diff) * (self.map_size - prev_edge)

        return wasserstein

    def compute_hitcount_diversity_weight(self, seed_key: str) -> float:
        """Compute weight based on JS divergence of hit-count distribution.

        A seed that exercises the same edges as the corpus but with a very
        different frequency profile (e.g. hits a loop 500x instead of 5x)
        is behaviorally distinct even with zero new edges.

        Returns a weight in [0.5, 2.0]:
        - 1.0 = typical profile (JS divergence near corpus average)
        - 2.0 = unusual profile (high JS — exercises edges differently)
        - 0.5 = near-identical profile to aggregate (redundant)

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

    def compute_wasserstein_distance(self, seed_key_a: str, seed_key_b: str) -> float:
        """Compute 1D Wasserstein-1 distance between two seeds' edge profiles.

        Treats edge indices as positions on a line, so adjacent edges
        are "close" even with no overlap. This captures coverage spatial
        diversity that Jaccard and JS divergence miss — two seeds hitting
        different but nearby edges are more similar than two seeds hitting
        the same number of edges at opposite ends of the map.

        Uses CDF-based algorithm: W = integral of |F_p(x) - F_q(x)| dx
        over sorted edge positions. O(n log n) where n = |keys_a| + |keys_b|.
        """
        wasserstein, _ks, _crps = self._cdf_norms(seed_key_a, seed_key_b)
        return wasserstein

    def compute_ks_distance(self, seed_key_a: str, seed_key_b: str) -> float:
        """Kolmogorov-Smirnov statistic between two seeds' edge profiles.

        Maximum absolute CDF difference — L∞ norm of the same quantity
        Wasserstein measures in L¹. KS ∈ [0, 1].
        """
        _w, ks, _crps = self._cdf_norms(seed_key_a, seed_key_b)
        return ks

    def compute_crps(self, seed_key_a: str, seed_key_b: str) -> float:
        """CRPS (Continuous Ranked Probability Score) between two edge profiles.

        L² integral of the CDF difference: ∫(F_a - F_b)² dy.
        CRPS ∈ [0, map_size]. Measured in the same units as the edge index
        space, so interpretable directly.
        """
        _w, _ks, crps = self._cdf_norms(seed_key_a, seed_key_b)
        return crps

    def _cdf_norms(self, seed_key_a: str, seed_key_b: str) -> tuple[float, float, float]:
        """Compute Wasserstein-1, KS, and CRPS from a single CDF walk.

        Returns (wasserstein, ks, crps) — L¹, L∞, and L² norms of the
        same CDF difference, computed in one pass over sorted edge positions.
        """
        hc_a = self.seed_hit_counts.get(seed_key_a, {})
        hc_b = self.seed_hit_counts.get(seed_key_b, {})
        if not hc_a or not hc_b:
            return float(self.map_size), 1.0, float(self.map_size)

        total_a = sum(hc_a.values())
        total_b = sum(hc_b.values())
        if total_a == 0 or total_b == 0:
            return float(self.map_size), 1.0, float(self.map_size)

        all_edges = sorted(set(hc_a) | set(hc_b))
        n = len(all_edges)
        if n == 0:
            return 0.0, 0.0, 0.0

        # Vectorized path for larger edge sets
        if _HAS_NUMPY and n > 20:
            p = np.array([hc_a.get(e, 0) / total_a for e in all_edges], dtype=np.float64)
            q = np.array([hc_b.get(e, 0) / total_b for e in all_edges], dtype=np.float64)
            gaps = np.diff(all_edges).astype(np.float64)
            cdf_diff = np.cumsum(p - q)
            wasserstein = float(np.sum(np.abs(cdf_diff[:-1]) * gaps))
            ks = float(np.max(np.abs(cdf_diff)))
            crps = float(np.sum(cdf_diff[:-1] ** 2 * gaps))
            return wasserstein, ks, crps

        # Pure-Python path for small edge sets or numpy unavailable
        cdf_diff = 0.0
        wasserstein = 0.0
        ks = 0.0
        crps = 0.0
        prev_edge = all_edges[0]

        for edge in all_edges:
            gap = edge - prev_edge
            abs_diff = abs(cdf_diff)
            wasserstein += abs_diff * gap
            ks = max(ks, abs_diff)
            crps += cdf_diff * cdf_diff * gap
            cdf_diff += hc_a.get(edge, 0) / total_a - hc_b.get(edge, 0) / total_b
            prev_edge = edge

        return wasserstein, ks, crps

    def compute_coverage_proximity(self, seed_key: str, radius: int = 5) -> float:
        """Compute how close a seed's edges are to uncovered edges.

        For each edge the seed hits, check if any uncovered edge is within
        `radius` positions in the bitmap. Seeds close to uncovered edges
        are more likely to reveal new code paths when mutated.

        Returns a weight in [0.0, 1.0]:
        - 0.0 = seed is far from any uncovered edge
        - 1.0 = seed is adjacent to uncovered edges
        """
        seed_edges = self.seed_edges.get(seed_key)
        if not seed_edges or not self.cumulative_edges:
            return 0.5

        # Compute uncovered edge positions
        max_edge = max(self.cumulative_edges) + radius + 1
        all_edges = set(range(min(self.map_size, max_edge)))
        uncovered = all_edges - self.cumulative_edges
        if not uncovered:
            return 0.0

        # Count how many of this seed's edges are within radius of an uncovered edge
        close_count = 0
        for edge in seed_edges:
            for offset in range(-radius, radius + 1):
                neighbor = edge + offset
                if 0 <= neighbor < self.map_size and neighbor in uncovered:
                    close_count += 1
                    break

        return close_count / len(seed_edges) if seed_edges else 0.0

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
                matches = sum(1 for a, b in zip(sig, corpus_sig, strict=False) if a == b)
                jaccard = matches / self._minhash.num_perm
                total_dist += 1.0 - jaccard

        return total_dist / len(keys) if keys else 0.0

    def compute_average_jaccard(self) -> float:
        """Compute average pairwise Jaccard similarity across all seeds.

        Uses MinHash signatures for O(1) per pair. Returns value in [0, 1]
        where 0 = all seeds have disjoint edge sets, 1 = all identical.

        This is the Jaccard index exposed as a metric — useful for monitoring
        corpus redundancy over time. High average Jaccard means the corpus
        is heavily redundant; low means seeds cover diverse code regions.
        """
        keys = list(self._minhash.signatures.keys())
        if len(keys) < 2:
            return 0.0

        n = len(keys)
        num_perm = self._minhash.num_perm

        # Vectorized path: broadcasting (n, 1, p) == (1, n, p) → (n, n, p)
        if _HAS_NUMPY and n > 20:
            sigs = np.array([self._minhash.signatures[k] for k in keys], dtype=np.int64)
            matches = np.sum(sigs[:, None, :] == sigs[None, :, :], axis=2)
            jaccard_matrix = matches / num_perm
            triu = np.triu_indices(n, k=1)
            return float(np.mean(jaccard_matrix[triu]))

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
        """
        hc = self.seed_hit_counts.get(seed_key)
        if not hc:
            return 0.0
        total = sum(hc.values())
        if total == 0:
            return 0.0
        if _HAS_NUMPY and len(hc) > 50:
            arr = np.fromiter(hc.values(), dtype=np.float64)
            arr = arr[arr > 0] / total
            return -float(np.sum(arr * np.log2(arr)))
        h = 0.0
        for count in hc.values():
            if count > 0:
                p = count / total
                h -= p * math.log2(p)
        return h

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

        # Build seed distribution
        seed_total = sum(hc.values())
        if seed_total == 0:
            return 1.0
        seed_dist = {e: c / seed_total for e, c in hc.items()}

        # Wasserstein computed directly against aggregate (no dict materialization)
        wasserstein = self._wasserstein_vs_aggregate(seed_dist)

        # Normalize: max possible Wasserstein is map_size
        normalized = min(wasserstein / self.map_size, 1.0)
        # Scale to [0.5, 2.0]
        return 0.5 + 1.5 * normalized

    def compute_hamming_bitmap_distance(self, seed_key_a: str, seed_key_b: str) -> float:
        """Compute Hamming distance between two seeds' edge bitmaps.

        Converts each seed's edge set to a fixed-length bitmap and counts
        differing positions. Faster than Jaccard/Wasserstein for detecting
        seeds that are byte-level near-duplicates.

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

        # Build compact bitmaps
        max_edge = max(max(edges_a), max(edges_b)) + 1
        size = (max_edge + 7) // 8
        bm_a = bytearray(size)
        bm_b = bytearray(size)
        for e in edges_a:
            bm_a[e >> 3] |= 1 << (e & 7)
        for e in edges_b:
            bm_b[e >> 3] |= 1 << (e & 7)

        dist = hamming_distance(bytes(bm_a), bytes(bm_b))
        return dist / (size * 8) if size > 0 else 0.0

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

    def get_cumulative_edge_count(self) -> int:
        """Get total unique edges seen across all seeds."""
        return len(self.cumulative_edges)

    def _rebuild_frequency_spectrum(self):
        """Rebuild frequency spectrum from global edge hits (lazy)."""
        if not self._spectrum_dirty:
            return
        self._frequency_spectrum.clear()
        for count in self._global_edge_hits.values():
            self._frequency_spectrum[count] = self._frequency_spectrum.get(count, 0) + 1
        self._spectrum_dirty = False

    def good_turing_estimate(self) -> dict:
        """Estimate undiscovered edges using Good-Turing frequency analysis.

        The raw Good-Turing formula N1²/(2*N2) assumes random sampling.
        Coverage-guided fuzzing samples non-randomly, producing more
        singletons than random sampling would. This inflates the estimate.

        We apply two corrections:
        1. Damping: when N2 < 10, scale the estimate by N2/10. This
           prevents a single doubleton change from causing a 4x swing.
        2. Cap: limit estimate to 5*N to avoid absurd extrapolations
           from sparse frequency data.

        Returns dict with:
          - n: total distinct edges observed
          - n1: edges seen exactly once (singletons)
          - n2: edges seen exactly twice
          - estimated_undiscovered: damped estimate
          - saturation: 1.0 - (estimated_undiscovered / total)
          - confidence: low/medium/high based on N1/N ratio
        """
        self._rebuild_frequency_spectrum()
        n = len(self.cumulative_edges)
        if n == 0:
            return {
                "n": 0,
                "n1": 0,
                "n2": 0,
                "estimated_undiscovered": 0,
                "saturation": 0.0,
                "confidence": "low",
            }
        if self._morris_mode:
            # In Morris mode, frequency spectrum is coarser.
            # Use approximate thresholds: n1=edges with count≤1, n2=count 2-3.
            counts = list(self._global_edge_hits.values())
            n1 = sum(1 for c in counts if c <= 1)
            n2 = sum(1 for c in counts if 2 <= c <= 3)
        else:
            n1 = self._frequency_spectrum.get(1, 0)
            n2 = self._frequency_spectrum.get(2, 0)
        if n2 > 0:
            raw_est = (n1 * n1) / (2 * n2)
        elif n1 > 0:
            raw_est = float(n1)
        else:
            raw_est = 0.0

        # Damping: when N2 is small, the estimate is unstable.
        # Scale by N2/10 to reduce sensitivity to single doubleton changes.
        damping = min(1.0, n2 / 10.0) if n2 > 0 else 0.5
        est_undiscovered = raw_est * damping

        # Cap: don't extrapolate more than 5x what we've already found
        est_undiscovered = min(est_undiscovered, n * 5)

        total = n + est_undiscovered
        saturation = 1.0 - (est_undiscovered / total) if total > 0 else 1.0
        ratio = n1 / n if n > 0 else 1.0
        if ratio < 0.05:
            confidence = "high"
        elif ratio < 0.20:
            confidence = "medium"
        else:
            confidence = "low"
        return {
            "n": n,
            "n1": n1,
            "n2": n2,
            "estimated_undiscovered": int(est_undiscovered),
            "saturation": saturation,
            "confidence": confidence,
        }

    def bitmap_density(self) -> float:
        """Fraction of the edge map that has been hit (0.0 to 1.0)."""
        return len(self._global_edge_hits) / self.map_size if self.map_size else 0.0

    def birthday_collision_risk(self) -> float:
        """Estimate birthday-paradox collision probability for current edge count.

        Uses the standard birthday bound: P(collision) ≈ 1 - exp(-n²/(2m))
        where n = number of distinct edges, m = map_size.

        Returns:
            Collision probability as a fraction (0.0 to 1.0).
        """
        n = len(self._global_edge_hits)
        m = self.map_size
        if m == 0 or n == 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - math.exp(-(n * n) / (2.0 * m))))

    def recommended_map_size(self) -> int:
        """Recommend a larger map_size if collision risk is high.

        Based on birthday bound: to keep collision probability < 1%,
        need map_size >= n² / (2 * ln(0.99)) ≈ n² / 0.02.

        Returns:
            Recommended map_size (next power of 2), or 0 if current size is adequate.
        """
        n = len(self._global_edge_hits)
        if n < 100:
            return 0
        needed = int(n * n / 0.02)
        recommended = 1
        while recommended < needed:
            recommended *= 2
        recommended = max(4096, min(1048576, recommended))
        if recommended <= self.map_size:
            return 0
        return recommended

    def get_seed_edge_count(self, seed_key: str) -> int:
        """Get number of edges a specific seed covers."""
        return len(self.seed_edges.get(seed_key, set()))

    def edge_rarity_stats(self) -> dict:
        """Compute per-edge rarity statistics.

        Returns dict with:
          - total: number of discovered edges
          - singleton: edges hit by exactly 1 seed (rare — lossy if pruned)
          - cold: edges hit by 2-3 seeds (fragile coverage)
          - warm: edges hit by 4-10 seeds
          - hot: edges hit by >10 seeds (redundant — safe to prune)
          - avg_seeds_per_edge: average number of seeds hitting each edge
        """
        if not self._global_edge_hits:
            return {
                "total": 0,
                "singleton": 0,
                "cold": 0,
                "warm": 0,
                "hot": 0,
                "avg_seeds_per_edge": 0.0,
            }

        counts = list(self._global_edge_hits.values())
        total = len(counts)
        if self._morris_mode:
            # Approximate counts from Morris: thresholds adjusted for log-scale
            singleton = sum(1 for c in counts if c <= 1)
            cold = sum(1 for c in counts if 2 <= c <= 5)
            warm = sum(1 for c in counts if 6 <= c <= 20)
            hot = sum(1 for c in counts if c > 20)
        else:
            singleton = sum(1 for c in counts if c == 1)
            cold = sum(1 for c in counts if 2 <= c <= 3)
            warm = sum(1 for c in counts if 4 <= c <= 10)
            hot = sum(1 for c in counts if c > 10)
        avg = sum(counts) / total if total else 0.0

        return {
            "total": total,
            "singleton": singleton,
            "cold": cold,
            "warm": warm,
            "hot": hot,
            "avg_seeds_per_edge": avg,
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
            if self._corpus_sig:
                weight = self.compute_subsumption_weight(seed_key)
            else:
                weight = 1.0

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

    def save(self, path: str) -> bool:
        """Save tracker state to JSON."""
        data = {
            "map_size": self.map_size,
            "morris_mode": self._morris_mode,
            "cumulative_edges": sorted(self.cumulative_edges),
            "seed_edges": {k: sorted(v) for k, v in self.seed_edges.items()},
            "seed_hit_counts": {
                k: {str(e): c for e, c in hc.items()} for k, hc in self.seed_hit_counts.items()
            },
            "global_edge_hits": {str(e): c for e, c in self._global_edge_hits.items()},
            "minhash_sigs": {k: sig for k, sig in self._minhash.signatures.items()},
            "aggregate_totals": {str(e): c for e, c in self._aggregate_totals.items()},
            "aggregate_total_count": self._aggregate_total_count,
            "edge_traces": {k: [list(e) for e in v] for k, v in self.seed_edge_traces.items()},
            "edge_first_seen": {str(e): c for e, c in self._edge_first_seen.items()},
            "edge_last_seen": {str(e): c for e, c in self._edge_last_seen.items()},
            "coverage_timeline": self._coverage_timeline,
            "correlation_matrix": {f"{a},{b}": c for (a, b), c in self._correlation_matrix.items()},
            "correlation_total": self._correlation_total,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
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
        """Load tracker state from JSON."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.debug("Failed to load edge tracker: %s", e)
            return False
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
        # Restore incremental aggregate totals
        self._aggregate_totals = {int(e): c for e, c in data.get("aggregate_totals", {}).items()}
        self._aggregate_total_count = data.get("aggregate_total_count", 0)
        # If no saved totals, rebuild from seed_hit_counts (legacy state files)
        if not self._aggregate_totals and self.seed_hit_counts:
            for hc in self.seed_hit_counts.values():
                for edge, count in hc.items():
                    self._aggregate_totals[edge] = self._aggregate_totals.get(edge, 0) + count
                    self._aggregate_total_count += count
        # Restore edge traces
        self.seed_edge_traces = {
            k: {(e[0], e[1]) for e in v} for k, v in data.get("edge_traces", {}).items()
        }
        # Restore temporal tracking
        self._edge_first_seen = {int(e): c for e, c in data.get("edge_first_seen", {}).items()}
        self._edge_last_seen = {int(e): c for e, c in data.get("edge_last_seen", {}).items()}
        self._coverage_timeline = [tuple(t) for t in data.get("coverage_timeline", [])]
        corr_data = data.get("correlation_matrix", {})
        self._correlation_matrix = {
            (int(k.split(",")[0]), int(k.split(",")[1])): v for k, v in corr_data.items()
        }
        self._correlation_total = data.get("correlation_total", 0)
        # Restore MinHash signatures and rebuild LSH index
        self._minhash = MinHashLSH(num_perm=64, num_bands=8)
        self._corpus_sig = None
        saved_sigs = data.get("minhash_sigs", {})
        if saved_sigs:
            for k, sig in saved_sigs.items():
                self._minhash.add(k, sig)
        else:
            # Rebuild from seed_edges for older state files
            for k, edges in self.seed_edges.items():
                sig = self._minhash.compute_signature(edges)
                self._minhash.add(k, sig)
        log.info(
            "Edge tracker loaded: %s (%d seeds, %d edges)",
            path,
            len(self.seed_edges),
            len(self.cumulative_edges),
        )
        return True
