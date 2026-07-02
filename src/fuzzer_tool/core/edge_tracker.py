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
        """Compute 1D Wasserstein distance between two seeds' edge profiles.

        Treats edge indices as positions on a line, so adjacent edges
        are "close" even with no overlap. This captures coverage spatial
        diversity that Jaccard and JS divergence miss — two seeds hitting
        different but nearby edges are more similar than two seeds hitting
        the same number of edges at opposite ends of the map.

        Uses CDF-based algorithm: W = integral of |F_p(x) - F_q(x)| dx
        over sorted edge positions. O(n log n) where n = |keys_a| + |keys_b|.
        """
        hc_a = self.seed_hit_counts.get(seed_key_a, {})
        hc_b = self.seed_hit_counts.get(seed_key_b, {})
        if not hc_a or not hc_b:
            return float(self.map_size)  # max distance if no data

        total_a = sum(hc_a.values())
        total_b = sum(hc_b.values())
        if total_a == 0 or total_b == 0:
            return float(self.map_size)

        # Merge all edge positions and sort
        all_edges = sorted(set(hc_a) | set(hc_b))

        # Walk sorted edges, accumulating CDF difference
        cdf_diff = 0.0
        wasserstein = 0.0
        prev_edge = all_edges[0] if all_edges else 0

        for edge in all_edges:
            # Distance from previous edge position
            gap = edge - prev_edge
            wasserstein += abs(cdf_diff) * gap

            # Update CDF at this position
            cdf_diff += hc_a.get(edge, 0) / total_a - hc_b.get(edge, 0) / total_b
            prev_edge = edge

        return wasserstein

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
        self._aggregate_cache = None
        log.info("Edge tracker loaded: %s (%d seeds, %d edges)", path, len(self.seed_edges), len(self.cumulative_edges))
        return True
