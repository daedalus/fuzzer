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
from collections.abc import Callable


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

        remaining_seeds = dict(seed_edges)
        curve = [(len(remaining_seeds), 1.0)]

        while remaining_seeds:
            # Find seed whose removal causes least coverage loss
            best_key = None
            best_loss = float("inf")
            for key, edges in remaining_seeds.items():
                other_edges = set()
                for other_key, other_set in remaining_seeds.items():
                    if other_key != key:
                        other_edges |= other_set
                loss = len(edges - other_edges)
                if loss < best_loss:
                    best_loss = loss
                    best_key = key

            if best_key is None:
                break

            # Remove the seed (its edge set is recomputed below, not reused)
            remaining_seeds.pop(best_key)

            # Recompute covered edges from remaining seeds
            remaining_edges = set()
            for edges in remaining_seeds.values():
                remaining_edges |= edges

            # Record point on curve
            if len(remaining_seeds) % step_size == 0 or not remaining_seeds:
                frac = len(remaining_edges) / total_edges if total_edges > 0 else 0.0
                curve.append((len(remaining_seeds), frac))

        return curve

    def minimax_robust_pruning(
        self,
        seed_edges: dict[str, set[int]],
        target_fraction: float = 0.95,
    ) -> tuple[list[str], float]:
        """Find the smallest corpus preserving target_fraction of coverage using minimax-robust criterion.

        This is the minimax-robust version of optimal_pruning: instead of greedy set-cover,
        we minimize the maximum coverage loss if any single seed is removed.

        Args:
            seed_edges: Dict mapping seed_key -> set of edge indices.
            target_fraction: Minimum coverage fraction to preserve (0.0-1.0).

        Returns:
            Tuple of (selected_seed_keys, actual_coverage_fraction).
        """
        if not seed_edges:
            return [], 0.0

        all_edges: set[int] = set()
        for edges in seed_edges.values():
            all_edges.update(edges)
        total = len(all_edges)
        target_count = int(math.ceil(total * target_fraction))

        if target_count == 0:
            return [], 1.0

        # Cover to target, then back up the seeds whose loss hurts most.
        cover = _RobustCover(seed_edges)
        remaining = dict(seed_edges)
        _greedy_cover(cover, remaining, lambda: len(cover.covered) >= target_count)
        _robust_fill(cover, remaining, lambda: False)

        return cover.selected, len(cover.covered) / total

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

        all_edges: set[int] = set()
        for edges in seed_edges.values():
            all_edges.update(edges)
        total = len(all_edges)
        target_count = int(math.ceil(total * target_fraction))

        if target_count == 0:
            return [], 1.0

        covered: set[int] = set()
        selected: list[str] = []
        remaining = dict(seed_edges)

        while covered < all_edges and len(covered) < target_count and remaining:
            best_key = max(
                remaining,
                key=lambda k: len(remaining[k] - covered),
            )
            best_edges = remaining[best_key]
            new_edges = best_edges - covered

            if not new_edges:
                break

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

    def minimax_robust_corpus_admission(
        self,
        seed_edges: dict[str, set[int]],
        max_seeds: int,
        preselected: list[str] | tuple[str, ...] = (),
    ) -> list[str]:
        """Select seeds to minimize the maximum coverage loss if any single seed
        is removed.

        This is the direct analog of the minimax estimator's "least favorable prior" —
        the corpus that is robust to the worst-case loss of any single seed.

        Args:
            seed_edges: Dict mapping seed_key -> set of edge indices.
            max_seeds: Maximum number of seeds to add.
            preselected: Keys already kept (e.g. set-cover mandatory seeds);
                counted for robustness, never returned.

        Returns:
            Added seed keys, ordered by selection (best first).
        """
        if not seed_edges or max_seeds <= 0:
            return []

        cover = _RobustCover(seed_edges)
        for key in preselected:
            if key in seed_edges:
                cover.add(key)
        base = len(cover.selected)
        remaining = {k: e for k, e in seed_edges.items() if k not in cover.selected}

        def full() -> bool:
            return len(cover.selected) - base >= max_seeds

        _greedy_cover(cover, remaining, full)
        _robust_fill(cover, remaining, full)
        return cover.selected[base:]

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


class _RobustCover:
    """Incremental single-seed-loss bookkeeping for minimax-robust selection.

    A seed's loss is the edges only it covers: dropping it loses exactly
    those. ``owner[e]`` is the one selected seed covering edge ``e``;
    shared edges have no owner. Risk is ``(max loss, seeds at that max)``,
    compared lexicographically, so backing up one of two tied worst seeds
    still counts as progress.

        A={1,2,5} B={3,4,5}  ->  owner {1:A,2:A,3:B,4:B}  risk (2, 2)
        add A2={1,2}         ->  owner {3:B,4:B}          risk (2, 1)
    """

    def __init__(self, seed_edges: dict[str, set[int]]):
        self._edges = seed_edges
        self.covered: set[int] = set()
        self.selected: list[str] = []
        self._owner: dict[int, str] = {}
        self._uniq: dict[str, int] = {}

    def add(self, key: str) -> None:
        """Select *key*: its fresh edges become its own, owned ones shared."""
        self._uniq[key] = 0
        for e in self._edges[key]:
            if e not in self.covered:
                self.covered.add(e)
                self._owner[e] = key
                self._uniq[key] += 1
                continue

            prev = self._owner.pop(e, None)
            if prev is not None:
                self._uniq[prev] -= 1
        self.selected.append(key)

    def risk(self) -> tuple[int, int]:
        """Current ``(max loss, count at max)``; ``(0, 0)`` when nothing is at risk."""
        top = max(self._uniq.values(), default=0)
        if top == 0:
            return 0, 0
        return top, sum(1 for u in self._uniq.values() if u == top)

    def ranked(self) -> tuple[list[tuple[str, int]], dict[int, int]]:
        """Selected seeds by loss, descending, plus a loss histogram (per round)."""
        order = sorted(self._uniq.items(), key=lambda kv: kv[1], reverse=True)
        hist: dict[int, int] = {}
        for _, u in order:
            hist[u] = hist.get(u, 0) + 1
        return order, hist

    def risk_with(
        self, key: str, order: list[tuple[str, int]], hist: dict[int, int]
    ) -> tuple[int, int]:
        """Risk after adding *key*, in O(|edges(key)|) instead of O(|selected|)."""
        dec: dict[str, int] = {}
        own = 0
        for e in self._edges[key]:
            o = self._owner.get(e)
            if o is not None:
                dec[o] = dec.get(o, 0) + 1
            elif e not in self.covered:
                own += 1

        # Untouched seeds keep their loss: the largest is the first one
        # in *order* that *key* does not overlap.
        top = next((u for k, u in order if k not in dec), 0)
        touched = [self._uniq[k] - d for k, d in dec.items()]
        touched.append(own)
        worst = max(top, max(touched))
        if worst == 0:
            return 0, 0

        count = sum(1 for v in touched if v == worst)
        if top == worst:
            count += hist[worst] - sum(1 for k in dec if self._uniq[k] == worst)
        return worst, count


def _greedy_cover(
    cover: _RobustCover, remaining: dict[str, set[int]], done: Callable[[], bool]
) -> None:
    """Greedy set-cover: add the seed with most uncovered edges until *done*."""
    while remaining and not done():
        best = max(remaining, key=lambda k: len(remaining[k] - cover.covered))
        if not remaining[best] - cover.covered:
            return
        cover.add(best)
        del remaining[best]


def _robust_fill(
    cover: _RobustCover, remaining: dict[str, set[int]], done: Callable[[], bool]
) -> None:
    """Add the seed that most lowers the risk; stop when none strictly does.

    Strict lexicographic decrease bounds the loop and keeps seeds that
    protect nothing (e.g. ones touching only shared edges) out.
    Ties go to the larger resulting coverage.
    """
    while remaining and not done():
        current = cover.risk()
        if current == (0, 0):
            return

        order, hist = cover.ranked()
        best, best_key = None, (current, 0)
        for key, edges in remaining.items():
            cand = (cover.risk_with(key, order, hist), -len(edges - cover.covered))
            if cand < best_key:
                best, best_key = key, cand

        if best is None or best_key[0] >= current:
            return
        cover.add(best)
        del remaining[best]
