"""Rate-distortion theory for optimal corpus minimization.

Rate-distortion theory (Shannon 1959) formalizes lossy compression:
what's the minimum number of bits (rate) needed to describe a source
within a given distortion level?

Applied to fuzzer corpus:
- Source = set of edge-coverage profiles across seeds
- Rate = number of seeds in the minimized corpus
- Distortion = coverage loss from removing a seed
- Goal: find the smallest subset of seeds that preserves ≥ X% of coverage

This is the information-theoretic foundation for corpus minimization —
it tells you exactly how much compression is achievable before coverage
degrades unacceptably.

Provides:
- Rate-distortion curve: corpus size vs. coverage preservation
- Optimal pruning: which seeds to remove first (highest distortion-per-bit)
- Information bottleneck: tradeoff between corpus size and coverage diversity
"""

import math


class RateDistortionCorpus:
    """Rate-distortion analysis for corpus minimization.

    Models the corpus as an information source where each seed carries
    a certain amount of coverage information. The rate-distortion curve
    shows how much coverage is lost as seeds are removed.

    Args:
        map_size: Edge bitmap size.
    """

    def __init__(self, map_size: int = 65536):
        self.map_size = map_size

    def compute_rate_distortion_curve(
        self,
        seed_edges: dict[str, set[int]],
        step_size: int = 1,
    ) -> list[tuple[int, float]]:
        """Compute the rate-distortion curve by greedy removal.

        Iteratively removes the seed with the least coverage impact,
        recording (corpus_size, coverage_fraction) at each step.

        Args:
            seed_edges: Dict mapping seed_key -> set of edge indices.
            step_size: Remove this many seeds between measurements.

        Returns:
            List of (corpus_size, coverage_fraction) pairs, sorted by
            corpus_size descending. coverage_fraction ∈ [0, 1].
        """
        if not seed_edges:
            return [(0, 0.0)]

        # Start with full corpus
        all_edges: set[int] = set()
        for edges in seed_edges.values():
            all_edges.update(edges)
        total_edges = len(all_edges)
        if total_edges == 0:
            return [(len(seed_edges), 0.0)]

        remaining_edges = set(all_edges)
        remaining_seeds = dict(seed_edges)
        curve = [(len(remaining_seeds), 1.0)]

        while remaining_seeds:
            # Find seed whose removal causes least coverage loss
            best_key = None
            best_loss = float("inf")
            for key, edges in remaining_seeds.items():
                # Loss = edges unique to this seed
                loss = len(edges - (remaining_edges - edges))
                if loss < best_loss:
                    best_loss = loss
                    best_key = key

            if best_key is None:
                break

            # Remove the seed
            removed_edges = remaining_seeds.pop(best_key)
            remaining_edges -= removed_edges

            # Record point on curve
            if len(remaining_seeds) % step_size == 0 or not remaining_seeds:
                frac = len(remaining_edges) / total_edges if total_edges > 0 else 0.0
                curve.append((len(remaining_seeds), frac))

        return curve

    def optimal_pruning(
        self,
        seed_edges: dict[str, set[int]],
        target_fraction: float = 0.95,
    ) -> tuple[list[str], float]:
        """Find the smallest corpus preserving target_fraction of coverage.

        Uses greedy set-cover: repeatedly add the seed covering the most
        uncovered edges until target is met.

        Args:
            seed_edges: Dict mapping seed_key -> set of edge indices.
            target_fraction: Minimum coverage fraction to preserve (0.0-1.0).

        Returns:
            Tuple of (selected_seed_keys, actual_coverage_fraction).
        """
        if not seed_edges:
            return [], 0.0

        # Compute total coverage
        all_edges: set[int] = set()
        for edges in seed_edges.values():
            all_edges.update(edges)
        total = len(all_edges)
        target_count = int(math.ceil(total * target_fraction))

        if target_count == 0:
            return [], 1.0

        # Greedy set cover
        covered: set[int] = set()
        selected: list[str] = []
        remaining = dict(seed_edges)

        while covered < all_edges and len(covered) < target_count and remaining:
            # Pick seed covering the most uncovered edges
            best_key = max(
                remaining,
                key=lambda k: len(remaining[k] - covered),
            )
            best_edges = remaining[best_key]
            new_edges = best_edges - covered

            if not new_edges:
                break  # diminishing returns

            covered.update(new_edges)
            selected.append(best_key)
            del remaining[best_key]

        actual_frac = len(covered) / total if total > 0 else 0.0
        return selected, actual_frac

    def seed_marginal_value(
        self,
        seed_key: str,
        seed_edges: dict[str, set[int]],
    ) -> float:
        """Compute the marginal information value of a seed.

        Value = (edges uniquely covered by this seed) / (total corpus edges).
        Seeds with high marginal value are irremovable without coverage loss.
        Seeds with low marginal value are redundant.

        Returns a value in [0, 1]:
        - 0.0 = seed is completely redundant
        - 1.0 = seed covers edges no other seed covers
        """
        if seed_key not in seed_edges:
            return 0.0

        my_edges = seed_edges[seed_key]
        if not my_edges:
            return 0.0

        # Edges covered by other seeds
        other_edges: set[int] = set()
        for key, edges in seed_edges.items():
            if key != seed_key:
                other_edges.update(edges)

        # Unique edges = my_edges not in others
        unique = my_edges - other_edges
        return len(unique) / len(my_edges) if my_edges else 0.0

    def information_bottleneck(
        self,
        seed_edges: dict[str, set[int]],
        max_seeds: int,
    ) -> list[str]:
        """Apply the information bottleneck: select max_seeds that maximize
        coverage while minimizing redundancy.

        Greedy approach: each step adds the seed with highest
        (new_edges_covered - redundancy_penalty).

        Args:
            seed_edges: Dict mapping seed_key -> set of edge indices.
            max_seeds: Maximum number of seeds to select.

        Returns:
            List of selected seed keys, ordered by selection (best first).
        """
        if not seed_edges or max_seeds <= 0:
            return []

        all_edges: set[int] = set()
        for edges in seed_edges.values():
            all_edges.update(edges)

        covered: set[int] = set()
        selected: list[str] = []
        remaining = dict(seed_edges)

        for _ in range(min(max_seeds, len(remaining))):
            if not remaining:
                break

            best_key = None
            best_score = -float("inf")

            for key, edges in remaining.items():
                new_edges = len(edges - covered)
                overlap = len(edges & covered)
                # Score: new coverage minus redundancy penalty
                score = new_edges - 0.1 * overlap
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_key is None or best_score <= 0:
                break

            covered.update(remaining[best_key])
            selected.append(best_key)
            del remaining[best_key]

        return selected

    def compression_ratio(
        self,
        seed_edges: dict[str, set[int]],
        selected: list[str],
    ) -> dict:
        """Compute compression ratio and coverage preservation.

        Returns:
            Dict with original_size, compressed_size, ratio, coverage_preserved.
        """
        all_edges: set[int] = set()
        for edges in seed_edges.values():
            all_edges.update(edges)
        total = len(all_edges)

        selected_edges: set[int] = set()
        for key in selected:
            if key in seed_edges:
                selected_edges.update(seed_edges[key])

        preserved = len(selected_edges) / total if total > 0 else 0.0
        ratio = len(seed_edges) / max(1, len(selected)) if selected else 1.0

        return {
            "original_size": len(seed_edges),
            "compressed_size": len(selected),
            "ratio": ratio,
            "coverage_preserved": preserved,
        }
