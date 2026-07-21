"""Track correlation between input length and coverage edge discovery.

Records (input_length, edge_set) pairs and computes which lengths
are most likely to reveal new coverage. Used to bias seed selection
and length-changing mutations toward productive lengths.
"""

from __future__ import annotations

import random
from collections import defaultdict

# ── Memory bounds ────────────────────────────────────────────────────
MAX_TRACKED_LENGTHS = 500  # max unique input lengths to track
MAX_EDGES_PER_LENGTH = 200  # max edge entries per input length


class LengthEdgeTracker:
    """Track which input lengths discover the most new edges.

    Maintains per-length edge discovery counts and provides
    recommendations for which lengths to try next.
    """

    def __init__(self):
        self.length_edge_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.length_total: dict[int, int] = defaultdict(int)
        self.total_execs: int = 0
        self._cached_totals: dict[int, int] = {}
        self._cached_sum: int = 0
        self._dirty = True

    def record(self, input_length: int, new_edges: set[int]) -> None:
        """Record that a specific input length produced new edges."""
        self.total_execs += 1
        self.length_total[input_length] += 1
        for edge in new_edges:
            self.length_edge_counts[input_length][edge] += 1
        self._dirty = True
        # Bound memory: prune least-productive lengths when too many tracked
        if len(self.length_edge_counts) > MAX_TRACKED_LENGTHS:
            self._prune_lengths()
        # Bound per-length edge count
        edges = self.length_edge_counts.get(input_length)
        if edges and len(edges) > MAX_EDGES_PER_LENGTH:
            # Keep only top half by count
            top = sorted(edges.items(), key=lambda kv: kv[1], reverse=True)[
                : MAX_EDGES_PER_LENGTH // 2
            ]
            self.length_edge_counts[input_length] = defaultdict(int, dict(top))

    def _prune_lengths(self):
        """Keep the top 50% of lengths by total edge count."""
        totals = {k: sum(v.values()) for k, v in self.length_edge_counts.items()}
        keep = sorted(totals, key=totals.get, reverse=True)[: len(totals) // 2]
        pruned = {k: self.length_edge_counts[k] for k in keep}
        self.length_edge_counts.clear()
        for k, v in pruned.items():
            self.length_edge_counts[k] = v
        self.length_total = defaultdict(
            int, {k: self.length_total[k] for k in keep if k in self.length_total}
        )
        self._dirty = True

    def _rebuild_cache(self):
        if not self._dirty:
            return
        self._cached_totals = {
            length: sum(edges.values()) for length, edges in self.length_edge_counts.items()
        }
        self._cached_sum = sum(self._cached_totals.values())
        self._dirty = False

    def recommended_lengths(self, k: int = 5) -> list[int]:
        """Return the k lengths that discovered the most new edges."""
        self._rebuild_cache()
        if not self._cached_totals:
            return []
        return sorted(self._cached_totals, key=self._cached_totals.get, reverse=True)[:k]

    def length_productivity(self, input_length: int) -> float:
        """Return a productivity score for a given length.

        Score is ratio of this length's edge discovery to the mean.
        Returns 1.0 when no data or when at the mean.
        """
        if not self.length_edge_counts:
            return 1.0
        self._rebuild_cache()
        this_edges = self._cached_totals.get(input_length, 0)
        n = len(self._cached_totals)
        mean_edges = self._cached_sum / max(n, 1)
        if mean_edges <= 0:
            return 1.0
        return this_edges / mean_edges

    def save(self) -> dict:
        """Serialize tracker state."""
        # Only save lengths with significant data
        counts = {
            k: {ek: ev for ek, ev in v.items() if ev > 0}
            for k, v in self.length_edge_counts.items()
            if self.length_total.get(k, 0) >= 5
        }
        return {
            "length_edge_counts": {
                str(k): {str(ek): ev for ek, ev in v.items()} for k, v in counts.items()
            },
            "length_total": {str(k): v for k, v in self.length_total.items() if v >= 5},
            "total_execs": self.total_execs,
        }

    def load(self, data: dict) -> None:
        """Restore tracker state from serialized data."""
        self.length_edge_counts = defaultdict(lambda: defaultdict(int))
        for k, v in data.get("length_edge_counts", {}).items():
            for ek, ev in v.items():
                self.length_edge_counts[int(k)][int(ek)] = ev
        self.length_total = defaultdict(
            int, {int(k): v for k, v in data.get("length_total", {}).items()}
        )
        self.total_execs = data.get("total_execs", 0)
        self._dirty = True
