"""Edge tracker for per-seed coverage tracking.

Tracks which coverage edges each seed contributes, enabling the fuzzer
to deprioritize seeds whose coverage is fully subsumed by others.
"""

import logging

log = logging.getLogger(__name__)


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
        # Global cumulative edge set (all edges ever seen)
        self.cumulative_edges: set[int] = set()

    def record_edges(self, seed_key: str, edge_bitmap: bytes) -> set[int]:
        """Record edges hit by a seed execution.

        Args:
            seed_key: Hash of the seed input.
            edge_bitmap: Raw edge bitmap (bytes where > 0 = edge hit).

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

        return new_contributions

    def compute_subsumption_weight(self, seed_key: str) -> float:
        """Compute a weight multiplier based on edge subsumption.

        Returns 1.0 if the seed has unique edges, lower if fully subsumed
        by other seeds' coverage.

        The key insight: we check if this seed's edges are ALL covered
        by the UNION of other seeds' edges (not including this seed).
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

        # Check what this seed uniquely contributes
        unique_to_seed = seed_edges - other_edges
        if unique_to_seed:
            return 1.0  # has edges no other seed covers

        # Fully subsumed — all this seed's edges are covered by others
        if seed_edges.issubset(other_edges):
            return 0.1  # heavily deprioritize

        return 0.8  # partially covered

    def get_cumulative_edge_count(self) -> int:
        """Get total unique edges seen across all seeds."""
        return len(self.cumulative_edges)

    def get_seed_edge_count(self, seed_key: str) -> int:
        """Get number of edges a specific seed covers."""
        return len(self.seed_edges.get(seed_key, set()))
