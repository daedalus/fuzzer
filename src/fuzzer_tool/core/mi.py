"""Mutual information between input bytes and coverage edges.

Computes I(X_i; Y) where X_i is byte position i and Y is the edge bitmap.
High-MI bytes are the ones that actually control which code paths execute —
mutating them is more likely to discover new coverage.

Also provides:
- Conditional MI: I(X_i; Y | X_j) — MI of byte i given byte j is fixed
- Interaction information: I(X_i, X_j; Y) — synergy between two positions
- Per-position MI profiles for scheduling decisions
"""

import math
from collections import defaultdict


class MutualInformationTracker:
    """Track mutual information between byte positions and coverage edges.

    Maintains joint distributions P(byte_val, edge_hit) incrementally.
    After sufficient observations, computes MI profiles that guide
    mutation scheduling toward high-impact byte positions.

    Args:
        max_positions: Maximum number of byte positions to track.
        min_observations: Minimum observations before computing MI.
    """

    def __init__(self, max_positions: int = 4096, min_observations: int = 50):
        self.max_positions = max_positions
        self.min_observations = min_observations
        self._max_mi_cache: dict[int, float] = {}  # input_length -> max_mi
        self._total_edges: int | None = None  # cached sum(edge_marginal.values())

        # Per-position: byte_value -> edge_index -> count
        # P(X_i = v, Y = e)
        self.joint: dict[int, dict[int, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        # Per-position: byte_value -> count
        self.byte_marginal: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Global: edge_index -> count
        self.edge_marginal: dict[int, int] = defaultdict(int)
        # Total observations per position
        self.position_counts: dict[int, int] = defaultdict(int)
        self.total_observations: int = 0

    def record(
        self, input_bytes: bytes, edge_bitmap: bytes, map_size: int = 65536
    ) -> None:
        """Record one input-coverage pair.

        Args:
            input_bytes: The input that was executed.
            edge_bitmap: Coverage bitmap (byte > 0 = edge hit).
            map_size: Maximum edge index to consider.
        """
        self.total_observations += 1
        self._total_edges = None
        self._invalidate_max_mi_cache()
        if hasattr(self, '_wp_all_positions') and self._wp_all_positions is not None:
            # Invalidate if we see a position we haven't tracked yet
            max_pos = len(input_bytes) - 1 if input_bytes else 0
            if max_pos >= self.max_positions:
                max_pos = self.max_positions - 1
            if max_pos not in {p for p in self._wp_all_positions}:
                self._wp_all_positions = None
        # Find which edges were hit
        hit_edges = {i for i, v in enumerate(edge_bitmap) if v > 0 and i < map_size}

        for pos, byte_val in enumerate(input_bytes):
            if pos >= self.max_positions:
                break
            self.position_counts[pos] += 1
            self.byte_marginal[pos][byte_val] += 1
            for edge in hit_edges:
                self.joint[pos][byte_val][edge] += 1
                self.edge_marginal[edge] += 1

    def mi(self, position: int) -> float:
        """Compute I(X_pos; Y) in bits.

        I(X; Y) = sum_{x,y} P(x,y) * log2(P(x,y) / (P(x) * P(y)))

        Returns 0.0 if insufficient data or position not observed.
        """
        n = self.position_counts.get(position, 0)
        if n < self.min_observations:
            return 0.0

        if self._total_edges is None:
            self._total_edges = sum(self.edge_marginal.values())
        total_edges = self._total_edges
        if total_edges == 0:
            return 0.0

        mi_value = 0.0
        byte_counts = self.byte_marginal.get(position, {})
        joint_pos = self.joint.get(position, {})

        for byte_val, bv_count in byte_counts.items():
            p_x = bv_count / n
            for edge, joint_count in joint_pos.get(byte_val, {}).items():
                p_xy = joint_count / n
                p_y = self.edge_marginal[edge] / total_edges
                if p_xy > 0 and p_y > 0:
                    mi_value += p_xy * math.log2(p_xy / (p_x * p_y))

        return max(0.0, mi_value)

    def mi_profile(self, input_length: int | None = None) -> dict[int, float]:
        """Compute MI for all tracked positions.

        Args:
            input_length: Only compute for positions < input_length.

        Returns:
            Dict mapping position -> MI in bits.
        """
        if input_length is None:
            input_length = max(self.position_counts.keys()) + 1 if self.position_counts else 0
        return {pos: self.mi(pos) for pos in range(input_length) if pos in self.position_counts}

    def top_positions(self, k: int = 10, input_length: int | None = None) -> list[tuple[int, float]]:
        """Return the k positions with highest MI.

        Returns:
            List of (position, mi_bits) sorted by MI descending.
        """
        profile = self.mi_profile(input_length)
        sorted_pos = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        return sorted_pos[:k]

    def _invalidate_max_mi_cache(self):
        """Invalidate max_mi cache when new observations are recorded."""
        self._max_mi_cache.clear()

    def mutation_weight(self, position: int, input_length: int) -> float:
        """Compute a mutation weight for a position based on MI.

        Returns a weight in [0.1, 5.0]:
        - High MI → weight near 5.0 (mutate this position aggressively)
        - Low MI → weight near 0.1 (skip this position)

        Normalizes MI to [0, 1] using the maximum observed MI across positions,
        then maps to the weight range.
        """
        mi_val = self.mi(position)
        if mi_val <= 0:
            return 0.1

        # Find max MI across observed positions for normalization (cached per input_length)
        if input_length not in self._max_mi_cache:
            candidates = [self.mi(pos) for pos in self.position_counts if pos < input_length]
            self._max_mi_cache[input_length] = max(candidates) if candidates else 0.0
        max_mi = self._max_mi_cache[input_length]
        if max_mi <= 0:
            return 1.0

        normalized = mi_val / max_mi
        return 0.1 + 4.9 * normalized

    def weighted_position(self, input_length: int) -> int:
        """Sample a byte position weighted by MI.

        Uses MI-weighted roulette wheel selection. Returns a position
        in [0, input_length) that is more likely to be information-rich.
        Precomputes all weights once using max_positions; subsequent calls
        filter to positions < input_length.
        """
        if not self.position_counts:
            return 0

        if not hasattr(self, '_wp_all_positions'):
            self._wp_all_positions = None
            self._wp_all_weights = None

        if self._wp_all_positions is None:
            # Precompute weights for ALL observed positions
            self._wp_all_positions = []
            self._wp_all_weights = []
            for pos in self.position_counts:
                if self.position_counts[pos] >= self.min_observations:
                    self._wp_all_positions.append(pos)
                    self._wp_all_weights.append(
                        self.mutation_weight(pos, self.max_positions)
                    )

        all_pos = self._wp_all_positions
        all_w = self._wp_all_weights
        if not all_w:
            return 0

        # Filter to positions within input_length
        if input_length >= self.max_positions:
            positions, weights = all_pos, all_w
        else:
            # Filter: keep positions < input_length
            # Weights are precomputed with max_positions as reference —
            # this is an approximation; _max_mi is the same regardless
            positions = []
            weights = []
            for i, p in enumerate(all_pos):
                if p < input_length:
                    positions.append(p)
                    weights.append(all_w[i])

        if not weights:
            return 0

        total = sum(weights)
        r = __import__("random").random() * total
        cumulative = 0.0
        for pos, w in zip(positions, weights, strict=False):
            cumulative += w
            if r <= cumulative:
                return pos
        return positions[-1]

    def conditional_mi(self, position_a: int, position_b: int) -> float:
        """Compute I(X_a; Y | X_b) — MI of position a given position b is observed.

        This captures the *additional* information position a provides
        beyond what position b already tells us about coverage.

        Uses the chain rule: I(X_a; Y | X_b) = H(X_a | X_b) + H(Y | X_b) - H(X_a, Y | X_b)

        Simplified approximation: if positions are independent given coverage,
        this equals I(X_a; Y). Significant deviation indicates correlation.
        """
        mi_a = self.mi(position_a)
        mi_b = self.mi(position_b)
        if mi_a == 0 or mi_b == 0:
            return mi_a

        # Joint MI of both positions
        joint = self._joint_mi_two(position_a, position_b)
        # I(X_a; Y | X_b) ≈ I(X_a, X_b; Y) - I(X_b; Y)
        return max(0.0, joint - mi_b)

    def _joint_mi_two(self, pos_a: int, pos_b: int) -> float:
        """Compute I(X_a, X_b; Y) — joint MI of two positions with coverage.

        Approximated by treating (byte_a, byte_b) as a combined symbol.
        Only works when both positions have sufficient observations.
        """
        n_a = self.position_counts.get(pos_a, 0)
        n_b = self.position_counts.get(pos_b, 0)
        if n_a < self.min_observations or n_b < self.min_observations:
            return 0.0

        # Build joint distribution: (byte_a, byte_b) -> edge -> count
        # This is approximate — we assume independent marginals
        # A more exact version would require tracking all position pairs
        mi_a = self.mi(pos_a)
        mi_b = self.mi(pos_b)
        # Upper bound: I(X_a,X_b;Y) <= I(X_a;Y) + I(X_b;Y)
        # Lower bound: I(X_a,X_b;Y) >= max(I(X_a;Y), I(X_b;Y))
        # Use sum as approximation (independence assumption)
        return mi_a + mi_b

    def interaction_information(self, pos_a: int, pos_b: int) -> float:
        """Compute interaction information: I(X_a, X_b; Y) - I(X_a; Y) - I(X_b; Y).

        Positive = synergy (together they explain more than sum of parts)
        Negative = redundancy (they explain overlapping coverage)
        Zero = independent
        """
        joint = self._joint_mi_two(pos_a, pos_b)
        mi_a = self.mi(pos_a)
        mi_b = self.mi(pos_b)
        return joint - mi_a - mi_b

    def save(self, path: str) -> bool:
        """Save tracker state to JSON."""
        import json

        data = {
            "max_positions": self.max_positions,
            "min_observations": self.min_observations,
            "total_observations": self.total_observations,
            "position_counts": dict(self.position_counts),
            "edge_marginal": {str(k): v for k, v in self.edge_marginal.items()},
            "byte_marginal": {
                str(pos): {str(bv): c for bv, c in counts.items()}
                for pos, counts in self.byte_marginal.items()
            },
            "joint": {
                str(pos): {
                    str(bv): {str(e): c for e, c in edges.items()}
                    for bv, edges in byte_vals.items()
                }
                for pos, byte_vals in self.joint.items()
            },
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            return True
        except OSError:
            return False

    def load(self, path: str) -> bool:
        """Load tracker state from JSON."""
        import json

        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        self.max_positions = data.get("max_positions", self.max_positions)
        self.min_observations = data.get("min_observations", self.min_observations)
        self.total_observations = data.get("total_observations", 0)
        self.position_counts = defaultdict(int, {int(k): v for k, v in data.get("position_counts", {}).items()})
        self.edge_marginal = defaultdict(int, {int(k): v for k, v in data.get("edge_marginal", {}).items()})
        self.byte_marginal = defaultdict(
            lambda: defaultdict(int),
            {
                int(pos): defaultdict(int, {int(bv): c for bv, c in counts.items()})
                for pos, counts in data.get("byte_marginal", {}).items()
            },
        )
        self.joint = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)),
            {
                int(pos): defaultdict(
                    lambda: defaultdict(int),
                    {
                        int(bv): defaultdict(int, {int(e): c for e, c in edges.items()})
                        for bv, edges in byte_vals.items()
                    },
                )
                for pos, byte_vals in data.get("joint", {}).items()
            },
        )
        return True
