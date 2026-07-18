"""Track correlation between input length and coverage edge discovery.

Records (input_length, edge_set) pairs and computes which lengths
are most likely to reveal new coverage. Used to bias seed selection
and length-changing mutations toward productive lengths.
"""

from __future__ import annotations

import random
from collections import defaultdict


class LengthEdgeTracker:
    """Track which input lengths discover the most new edges.

    Maintains per-length edge discovery counts and provides
    recommendations for which lengths to try next.
    """

    def __init__(self):
        self.length_edge_counts: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.length_total: dict[int, int] = defaultdict(int)
        self.total_execs: int = 0

    def record(self, input_length: int, new_edges: set[int]) -> None:
        """Record that a specific input length produced new edges."""
        self.total_execs += 1
        self.length_total[input_length] += 1
        for edge in new_edges:
            self.length_edge_counts[input_length][edge] += 1

    def recommended_lengths(self, k: int = 5) -> list[int]:
        """Return the k lengths that discovered the most new edges."""
        scores = {}
        for length, edges in self.length_edge_counts.items():
            scores[length] = sum(edges.values())
        if not scores:
            return []
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def length_productivity(self, input_length: int) -> float:
        """Return a productivity score for a given length.

        Score is ratio of this length's edge discovery to the mean.
        Returns 1.0 when no data or when at the mean.
        """
        if not self.length_edge_counts:
            return 1.0
        this_edges = sum(self.length_edge_counts.get(input_length, {}).values())
        all_totals = [sum(e.values()) for e in self.length_edge_counts.values()]
        mean_edges = sum(all_totals) / max(len(all_totals), 1)
        if mean_edges <= 0:
            return 1.0
        return this_edges / mean_edges

    def save(self) -> dict:
        """Serialize tracker state."""
        return {
            "length_edge_counts": {
                str(k): {str(ek): ev for ek, ev in v.items()}
                for k, v in self.length_edge_counts.items()
            },
            "length_total": dict(self.length_total),
            "total_execs": self.total_execs,
        }

    def load(self, data: dict) -> None:
        """Restore tracker state from serialized data."""
        self.length_edge_counts = defaultdict(lambda: defaultdict(int))
        for k, v in data.get("length_edge_counts", {}).items():
            for ek, ev in v.items():
                self.length_edge_counts[int(k)][int(ek)] = ev
        self.length_total = defaultdict(int, {int(k): v for k, v in data.get("length_total", {}).items()})
        self.total_execs = data.get("total_execs", 0)
