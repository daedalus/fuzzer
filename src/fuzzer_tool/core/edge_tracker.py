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
        # Cached aggregate hit-count distribution (rebuilt lazily)
        self._aggregate_cache: dict[int, float] | None = None
        # Good-Turing: global cumulative hit count per edge (across all seeds)
        self._global_edge_hits: dict[int, int] = {}
        self._spectrum_dirty = True
        self._frequency_spectrum: dict[int, int] = {}
        self.max_hit_count: int = 0

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

        self._aggregate_cache = None  # invalidate

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

    def compute_subsumption_weight(self, seed_key: str) -> float:
        """Compute a weight multiplier based on Jaccard similarity of edge sets.

        Returns a continuous weight in [0.1, 1.0] based on how much this
        seed's coverage overlaps with other seeds.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B| where A = seed edges,
        B = union of all other seeds' edges. High overlap → low weight,
        novel edges → high weight.

        This replaces the previous binary check (unique / subsumed / partial)
        with a continuous score, so near-duplicate seeds that technically have
        1 unique edge among 100 shared ones get deprioritized instead of
        receiving full weight.
        """
        if seed_key not in self.seed_edges:
            return 1.0

        seed_edges = self.seed_edges[seed_key]
        if not seed_edges:
            return 0.5  # no coverage data → slightly deprioritize

        # Compute edges covered by OTHER seeds (excluding this seed)
        other_edges: set[int] = set()
        for k, edges in self.seed_edges.items():
            if k != seed_key:
                other_edges.update(edges)

        if not other_edges:
            return 1.0  # only seed — all edges are novel

        intersection = len(seed_edges & other_edges)
        union = len(seed_edges | other_edges)
        jaccard = intersection / union if union else 0.0

        # Scale: high overlap (jaccard → 1.0) → low weight, novel → high weight
        return max(0.1, 1.0 - jaccard)

    def _build_aggregate_distribution(self) -> dict[int, float]:
        """Build the corpus-wide aggregate hit-count distribution.

        Sums hit counts across all seeds for each edge, then normalizes
        to a probability distribution. Cached until a new seed is recorded.
        """
        if self._aggregate_cache is not None:
            return self._aggregate_cache

        totals: dict[int, int] = {}
        for hc in self.seed_hit_counts.values():
            for edge, count in hc.items():
                totals[edge] = totals.get(edge, 0) + count

        total_count = sum(totals.values())
        if total_count == 0:
            return {}

        self._aggregate_cache = {e: c / total_count for e, c in totals.items()}
        return self._aggregate_cache

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

        aggregate = self._build_aggregate_distribution()
        if not aggregate:
            return 1.0

        # Build normalized distribution for this seed
        total = sum(hc.values())
        if total == 0:
            return 1.0
        seed_dist = {e: c / total for e, c in hc.items()}

        js = _js_divergence(seed_dist, aggregate)
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
        """Compute average pairwise Wasserstein distance across all seeds.

        Returns a value in [0, map_size] where:
        - 0 = all seeds hit exactly the same edges with same frequencies
        - high = seeds are spread across the edge map (diverse coverage)

        This is O(n^2) in the number of tracked seeds, so it's called
        periodically (not every iteration) and cached.
        """
        keys = list(self.seed_hit_counts.keys())
        if len(keys) < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                total += self.compute_wasserstein_distance(keys[i], keys[j])
                count += 1

        return total / count if count else 0.0

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

        aggregate = self._build_aggregate_distribution()
        if not aggregate:
            return 1.0

        centroid_dist = aggregate

        # Build seed distribution
        seed_total = sum(hc.values())
        if seed_total == 0:
            return 1.0
        seed_dist = {e: c / seed_total for e, c in hc.items()}

        # Wasserstein between seed and centroid
        all_edges = sorted(set(seed_dist) | set(centroid_dist))
        cdf_diff = 0.0
        wasserstein = 0.0
        prev_edge = all_edges[0] if all_edges else 0

        for edge in all_edges:
            gap = edge - prev_edge
            wasserstein += abs(cdf_diff) * gap
            cdf_diff += seed_dist.get(edge, 0.0) - centroid_dist.get(edge, 0.0)
            prev_edge = edge

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
        log.info("Edge tracker loaded: %s (%d seeds, %d edges)", path, len(self.seed_edges), len(self.cumulative_edges))
        return True
