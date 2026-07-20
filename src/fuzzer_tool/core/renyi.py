"""Rényi entropy for generalized information-theoretic analysis.

Rényi-α entropy generalizes Shannon entropy:
  H_α(X) = 1/(1-α) * log2(sum(p_i^α))

Special cases:
- α → 0: support size (log2 of number of non-zero elements)
- α = 1: Shannon entropy (limit)
- α = 2: collision entropy (Hartley)
- α → ∞: min-entropy (dominant element)

For fuzzing:
- High min-entropy (α→∞) means coverage is dominated by a few hot edges
  → mutation budget is wasted on cold paths
- Low min-entropy means coverage is well-spread → uniform mutation is fine
- Rényi spectrum (H_α across α values) characterizes the "shape" of coverage

Also provides:
- Rényi divergence: generalization of KL divergence for comparing distributions
- Tsallis entropy: another non-extensive entropy measure
- Coverage spectrum analysis: per-edge hit count distribution profiling
"""

import math
from collections import Counter

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class RenyiEntropy:
    """Compute Rényi entropy of order α for discrete distributions.

    Works with raw count data (normalizes internally) or pre-normalized
    probability distributions.
    """

    def __init__(self):
        pass

    def renyi(self, counts: dict | Counter | list, alpha: float) -> float:
        """Compute Rényi-α entropy in bits.

        Args:
            counts: Raw counts or probabilities. If counts are integers,
                    they're normalized to probabilities.
            alpha: Order parameter. Must be >= 0. Use alpha=0 for support size,
                   alpha=1 for Shannon entropy, alpha=2 for collision entropy.

        Returns:
            Rényi entropy in bits.
        """
        probs = self._to_probs(counts)
        if not probs:
            return 0.0

        if alpha == 0:
            # Support size: log2(number of non-zero elements)
            n_support = sum(1 for p in probs if p > 0)
            return math.log2(max(1, n_support))

        if abs(alpha - 1.0) < 1e-10:
            # Shannon entropy (limit as α→1)
            return self._shannon(probs)

        # General Rényi: H_α = 1/(1-α) * log2(sum(p_i^α))
        if _HAS_NUMPY:
            p_arr = np.array([x for x in probs if x > 0], dtype=np.float64)
            if len(p_arr) == 0:
                return 0.0
            sum_p_alpha = float(np.sum(p_arr ** alpha))
            if sum_p_alpha <= 0:
                return 0.0
            return (1.0 / (1.0 - alpha)) * math.log2(sum_p_alpha)
        sum_p_alpha = sum(p**alpha for p in probs if p > 0)
        if sum_p_alpha <= 0:
            return 0.0
        return (1.0 / (1.0 - alpha)) * math.log2(sum_p_alpha)

    def shannon(self, counts: dict | Counter | list) -> float:
        """Shannon entropy in bits (convenience method)."""
        return self.renyi(counts, alpha=1.0)

    def min_entropy(self, counts: dict | Counter | list) -> float:
        """Min-entropy: H_∞ = -log2(max(p_i)).

        Captures the worst-case predictability. Low min-entropy means
        a few edges dominate coverage.
        """
        probs = self._to_probs(counts)
        if not probs:
            return 0.0
        max_p = max(probs)
        return -math.log2(max_p) if max_p > 0 else 0.0

    def collision_entropy(self, counts: dict | Counter | list) -> float:
        """Collision entropy: H_2 = -log2(sum(p_i^2)).

        Related to the probability of collision when drawing two samples.
        """
        return self.renyi(counts, alpha=2.0)

    def entropy_spectrum(self, counts: dict | Counter | list) -> dict[str, float]:
        """Compute the Rényi spectrum: H_α for α ∈ {0, 0.5, 1, 2, 5, 10, ∞}.

        The shape of this spectrum reveals coverage distribution properties:
        - Flat spectrum → uniform coverage (good)
        - Steep decline → heavy-tailed coverage (hot edges dominate)
        - H_0 >> H_∞ → many edges but coverage concentrated
        """
        alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

        if _HAS_NUMPY:
            probs = self._to_probs(counts)
            if not probs:
                return {f"renyi_{a}": 0.0 for a in alphas} | {"min_entropy": 0.0}
            p = np.array(probs, dtype=np.float64)
            p = p[p > 0]
            if len(p) == 0:
                return {f"renyi_{a}": 0.0 for a in alphas} | {"min_entropy": 0.0}
            spectrum = {}
            a_arr = np.array(alphas, dtype=np.float64)
            # Vectorized Rényi for all alpha != 1
            p_alpha = p[:, np.newaxis] ** a_arr[np.newaxis, :]  # shape: (n_elements, n_alphas)
            sum_p_alpha = np.sum(p_alpha, axis=0)  # shape: (n_alphas,)
            # H_alpha = 1/(1-alpha) * log2(sum(p^alpha))
            with np.errstate(divide="ignore", invalid="ignore"):
                h = np.where(
                    a_arr != 1.0,
                    (1.0 / (1.0 - a_arr)) * np.log2(np.maximum(sum_p_alpha, 1e-300)),
                    0.0,  # Shannon handled separately
                )
            # Handle alpha=0 (support size)
            h[0] = np.log2(max(1, len(p)))
            for i, a in enumerate(alphas):
                spectrum[f"renyi_{a}"] = float(h[i]) if a != 1.0 else self._shannon(probs)
            spectrum["min_entropy"] = -np.log2(float(np.max(p))) if np.max(p) > 0 else 0.0
            return spectrum

        spectrum = {}
        for a in alphas:
            spectrum[f"renyi_{a}"] = self.renyi(counts, a)
        spectrum["min_entropy"] = self.min_entropy(counts)
        return spectrum

    def coverage_uniformity(self, counts: dict | Counter | list) -> float:
        """Measure coverage uniformity using Rényi spectrum.

        Returns a value in [0, 1]:
        - 1.0 = perfectly uniform (all edges hit equally)
        - 0.0 = maximally non-uniform (one edge dominates)

        Computed as: H_∞ / H_0 (min-entropy / support-size entropy).
        For uniform distribution over n elements: H_∞ = H_0 = log2(n).
        """
        spectrum = self.entropy_spectrum(counts)
        h0 = spectrum.get("renyi_0.0", 0.0)
        h_inf = spectrum.get("min_entropy", 0.0)
        if h0 <= 0:
            return 1.0  # no edges = trivially uniform
        return min(1.0, h_inf / h0)

    def tsallis_entropy(self, counts: dict | Counter | list, q: float) -> float:
        """Tsallis entropy of order q.

        S_q = 1/(q-1) * (1 - sum(p_i^q))

        Non-extensive entropy measure. Additive for independent systems
        (unlike Rényi). Useful for measuring non-extensivity of coverage.

        Args:
            counts: Raw counts or probabilities.
            q: Non-extensivity parameter. q→1 gives Shannon entropy.
        """
        probs = self._to_probs(counts)
        if not probs:
            return 0.0

        if abs(q - 1.0) < 1e-10:
            # Shannon limit: S_1 = -sum(p_i * log(p_i))
            return self._shannon(probs)

        if _HAS_NUMPY:
            p_arr = np.array([x for x in probs if x > 0], dtype=np.float64)
            if len(p_arr) == 0:
                return 0.0
            sum_p_q = float(np.sum(p_arr ** q))
            return (1.0 / (q - 1.0)) * (1.0 - sum_p_q)

        sum_p_q = sum(p**q for p in probs if p > 0)
        return (1.0 / (q - 1.0)) * (1.0 - sum_p_q)

    def _to_probs(self, counts) -> list[float]:
        """Convert counts to normalized probability distribution."""
        if isinstance(counts, (dict, Counter)):
            values = list(counts.values())
        elif isinstance(counts, (list, tuple)):
            values = list(counts)
        else:
            return []

        total = sum(values)
        if total == 0:
            return []
        return [v / total for v in values]

    def _shannon(self, probs: list[float]) -> float:
        """Shannon entropy in bits."""
        if _HAS_NUMPY:
            p_arr = np.array([x for x in probs if x > 0], dtype=np.float64)
            if len(p_arr) == 0:
                return 0.0
            return -float(np.sum(p_arr * np.log2(p_arr)))
        h = 0.0
        for p in probs:
            if p > 0:
                h -= p * math.log2(p)
        return h


class CoverageSpectrumAnalyzer:
    """Analyze the hit-count spectrum of coverage edges.

    Profiles the distribution of edge hit counts across the coverage map.
    Reveals whether coverage is dominated by hot edges (loops, common paths)
    or is well-distributed across cold edges (rare code paths).

    Args:
        max_hit_count: Maximum observed hit count for normalization.
    """

    def __init__(self, max_hit_count: int = 255):
        self.max_hit_count = max_hit_count
        self._renyi = RenyiEntropy()

    def analyze(self, hit_counts: dict[int, int]) -> dict:
        """Analyze an edge hit-count distribution.

        Args:
            hit_counts: Dict mapping edge_index -> hit_count.

        Returns:
            Dict with spectrum analysis results.
        """
        if not hit_counts:
            return {
                "n_edges": 0,
                "uniformity": 1.0,
                "dominance_ratio": 0.0,
                "hot_edge_fraction": 0.0,
                "spectrum": {},
            }

        values = list(hit_counts.values())
        total = sum(values)

        # Compute spectrum
        spectrum = self._renyi.entropy_spectrum(values)

        # Dominance ratio: fraction of total hits from top-10% edges
        sorted_vals = sorted(values, reverse=True)
        top_10_pct = max(1, len(sorted_vals) // 10)
        top_sum = sum(sorted_vals[:top_10_pct])
        dominance_ratio = top_sum / total if total > 0 else 0.0

        # Hot edge fraction: edges with hit_count > median
        median_val = sorted_vals[len(sorted_vals) // 2] if sorted_vals else 0
        if _HAS_NUMPY:
            hot_edges = int(np.count_nonzero(np.array(values, dtype=np.int64) > median_val))
        else:
            hot_edges = sum(1 for v in values if v > median_val)
        hot_fraction = hot_edges / len(values) if values else 0.0

        # Uniformity from Rényi spectrum
        uniformity = self._renyi.coverage_uniformity(values)

        return {
            "n_edges": len(values),
            "total_hits": total,
            "uniformity": uniformity,
            "dominance_ratio": dominance_ratio,
            "hot_edge_fraction": hot_fraction,
            "median_hit_count": median_val,
            "mean_hit_count": total / len(values) if values else 0,
            "spectrum": spectrum,
        }

    def mutation_budget_weight(self, hit_counts: dict[int, int], edge_index: int) -> float:
        """Compute mutation weight for an edge based on its hit frequency.

        Edges hit very often (hot) get lower weight — mutating near them
        is likely redundant. Edges hit rarely (cold) get higher weight —
        they're closer to unexplored code.

        Returns a weight in [0.1, 3.0]:
        - 0.1 = edge is very hot (hit 1000+ times)
        - 3.0 = edge is cold (hit once)
        """
        if not hit_counts:
            return 1.0

        count = hit_counts.get(edge_index, 0)
        if count == 0:
            return 2.0  # unseen edges get high weight

        # Normalize by max hit count
        normalized = min(count / self.max_hit_count, 1.0)
        # Inverse: low hits → high weight
        return 0.1 + 2.9 * (1.0 - normalized)
