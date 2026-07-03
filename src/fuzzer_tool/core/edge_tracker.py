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

import json
import logging
import math
import random
import zlib

log = logging.getLogger(__name__)


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
    m: dict[int, float] = {}
    all_keys = set(p) | set(q)
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
        self._coeffs = [(rng.randint(1, self._prime - 1), rng.randint(0, self._prime - 1))
                        for _ in range(num_perm)]

    def compute_signature(self, edge_set: set[int]) -> list[int]:
        """Compute MinHash signature for a set of edge indices.

        Uses k independent hash functions of the form h(x) = (a*x + b) mod p,
        taking the minimum hash value across all elements in the set.
        """
        sig = [self._prime] * self.num_perm
        for edge in edge_set:
            for i, (a, b) in enumerate(self._coeffs):
                h = (a * edge + b) % self._prime
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
            band_hash = hash(tuple(sig[start:end]))
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
            band_hash = hash(tuple(sig[start:end]))
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
            band_hash = hash(tuple(sig[start:end]))
            bucket_key = (band_idx, band_hash)
            bucket = self.buckets.get(bucket_key)
            if bucket:
                candidates.update(bucket)
        candidates.discard(seed_key)

        # Filter by full Jaccard threshold
        if min_jaccard <= 0:
            return candidates
        return {k for k in candidates
                if self.approximate_jaccard(seed_key, k) >= min_jaccard}

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

    def __init__(self, map_size: int = 65536):
        self.map_size = map_size
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

    def record_edges(self, seed_key: str, edge_bitmap: bytes) -> set[int]:
        """Record edges hit by a seed execution.

        Args:
            seed_key: Hash of the seed input.
            edge_bitmap: Raw edge bitmap (bytes where > 0 = edge hit).
                Values are hit counts (0-255), not just binary.

        Returns:
            Set of NEW edge indices not previously seen.
        """
        new_edges = set()
        for i, val in enumerate(edge_bitmap):
            if val and i < self.map_size:
                new_edges.add(i)

        new_contributions = new_edges - self.cumulative_edges
        self.cumulative_edges.update(new_edges)

        if seed_key not in self.seed_edges:
            self.seed_edges[seed_key] = set()
        self.seed_edges[seed_key].update(new_edges)

        # Store sparse hit-count vector for JS divergence scoring
        if seed_key not in self.seed_hit_counts:
            self.seed_hit_counts[seed_key] = {}
        hc = self.seed_hit_counts[seed_key]
        for i, val in enumerate(edge_bitmap):
            if val and i < self.map_size:
                hc[i] = val

        # Update MinHash signature and LSH index
        sig = self._minhash.compute_signature(self.seed_edges[seed_key])
        self._minhash.add(seed_key, sig)

        # Incrementally update aggregate distribution (avoids full O(n) rebuild)
        for i, val in enumerate(edge_bitmap):
            if val and i < self.map_size:
                old = self._aggregate_totals.get(i, 0)
                self._aggregate_totals[i] = old + val
                self._aggregate_total_count += val
        # Invalidate normalized cache (totals changed, need re-normalize)
        self._aggregate_cache = None
        self._corpus_sig = None  # invalidate MinHash corpus signature

        # Update global edge hits for Good-Turing estimation
        for i, val in enumerate(edge_bitmap):
            if val and i < self.map_size:
                old = self._global_edge_hits.get(i, 0)
                new = old + val
                self._global_edge_hits[i] = new
                self._spectrum_dirty = True
                if new > self.max_hit_count:
                    self.max_hit_count = new

        return new_contributions

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
            e: c / self._aggregate_total_count
            for e, c in self._aggregate_totals.items()
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

    def compute_wasserstein_distance(
        self, seed_key_a: str, seed_key_b: str
    ) -> float:
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

    def compute_ks_distance(
        self, seed_key_a: str, seed_key_b: str
    ) -> float:
        """Kolmogorov-Smirnov statistic between two seeds' edge profiles.

        Maximum absolute CDF difference — L∞ norm of the same quantity
        Wasserstein measures in L¹. KS ∈ [0, 1].
        """
        _w, ks, _crps = self._cdf_norms(seed_key_a, seed_key_b)
        return ks

    def compute_crps(
        self, seed_key_a: str, seed_key_b: str
    ) -> float:
        """CRPS (Continuous Ranked Probability Score) between two edge profiles.

        L² integral of the CDF difference: ∫(F_a - F_b)² dy.
        CRPS ∈ [0, map_size]. Measured in the same units as the edge index
        space, so interpretable directly.
        """
        _w, _ks, crps = self._cdf_norms(seed_key_a, seed_key_b)
        return crps

    def _cdf_norms(
        self, seed_key_a: str, seed_key_b: str
    ) -> tuple[float, float, float]:
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

        cdf_diff = 0.0
        wasserstein = 0.0
        ks = 0.0
        crps = 0.0
        prev_edge = all_edges[0] if all_edges else 0

        for edge in all_edges:
            gap = edge - prev_edge
            abs_diff = abs(cdf_diff)
            wasserstein += abs_diff * gap
            ks = max(ks, abs_diff)
            crps += cdf_diff * cdf_diff * gap
            cdf_diff += hc_a.get(edge, 0) / total_a - hc_b.get(edge, 0) / total_b
            prev_edge = edge

        return wasserstein, ks, crps

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

        Returns dict with:
          - n: total distinct edges observed
          - n1: edges seen exactly once (singletons)
          - n2: edges seen exactly twice
          - estimated_undiscovered: N1^2 / (2 * N2)
          - saturation: 1.0 - (N1^2 / (2 * N2 * N)) — how close to done
          - confidence: low/medium/high based on N1/N ratio
        """
        self._rebuild_frequency_spectrum()
        n = len(self.cumulative_edges)
        if n == 0:
            return {"n": 0, "n1": 0, "n2": 0, "estimated_undiscovered": 0,
                    "saturation": 0.0, "confidence": "low"}
        n1 = self._frequency_spectrum.get(1, 0)
        n2 = self._frequency_spectrum.get(2, 0)
        if n2 > 0:
            est_undiscovered = (n1 * n1) / (2 * n2)
        elif n1 > 0:
            est_undiscovered = float(n1)
        else:
            est_undiscovered = 0.0
        total = n + est_undiscovered
        saturation = 1.0 - (est_undiscovered / total) if total > 0 else 1.0
        ratio = n1 / n if n > 0 else 1.0
        if ratio < 0.05:
            confidence = "high"
        elif ratio < 0.20:
            confidence = "medium"
        else:
            confidence = "low"
        return {"n": n, "n1": n1, "n2": n2, "estimated_undiscovered": int(est_undiscovered),
                "saturation": saturation, "confidence": confidence}

    def bitmap_density(self) -> float:
        """Fraction of the edge map that has been hit (0.0 to 1.0)."""
        return len(self._global_edge_hits) / self.map_size if self.map_size else 0.0

    def get_seed_edge_count(self, seed_key: str) -> int:
        """Get number of edges a specific seed covers."""
        return len(self.seed_edges.get(seed_key, set()))

    def save(self, path: str) -> bool:
        """Save tracker state to JSON."""
        data = {
            "map_size": self.map_size,
            "cumulative_edges": sorted(self.cumulative_edges),
            "seed_edges": {k: sorted(v) for k, v in self.seed_edges.items()},
            "seed_hit_counts": {
                k: {str(e): c for e, c in hc.items()}
                for k, hc in self.seed_hit_counts.items()
            },
            "global_edge_hits": {str(e): c for e, c in self._global_edge_hits.items()},
            "minhash_sigs": {k: sig for k, sig in self._minhash.signatures.items()},
            "aggregate_totals": {str(e): c for e, c in self._aggregate_totals.items()},
            "aggregate_total_count": self._aggregate_total_count,
            "edge_traces": {
                k: [list(e) for e in v]
                for k, v in self.seed_edge_traces.items()
            },
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            log.info("Edge tracker saved: %s (%d seeds, %d edges)", path, len(self.seed_edges), len(self.cumulative_edges))
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
            k: {(e[0], e[1]) for e in v}
            for k, v in data.get("edge_traces", {}).items()
        }
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
        log.info("Edge tracker loaded: %s (%d seeds, %d edges)", path, len(self.seed_edges), len(self.cumulative_edges))
        return True
