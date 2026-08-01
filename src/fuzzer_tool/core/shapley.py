"""ShapleyAttribution: game-theoretic attribution of operators to edge coverage."""

import collections
import math
import random
from collections import defaultdict

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ── Memory bounds ────────────────────────────────────────────────────
SHAPLEY_EDGES_MAX = 10_000  # max edges tracked in Shapley attribution


class ShapleyAttribution:
    """Compute Shapley values for mutation operator contribution.

    Uses per-edge frequency-weighted attribution: for each edge, credit
    is distributed among operators proportional to how often each operator
    co-occurred with that edge across all executions. Operators that
    consistently appear when a specific edge is observed get more credit;
    operators that merely co-occur with productive ones get less.

    This is an improvement over naive co-occurrence attribution (giving
    every stacked operator all edges), though it still measures
    correlation, not causation. True causal attribution would require
    per-operator bitmap snapshots between mutation steps.

    Args:
        n_samples: Number of random permutations to sample.
        window_size: Number of recent outcomes to consider.
    """

    def __init__(self, n_samples: int = 100, window_size: int = 500):
        self.n_samples = n_samples
        self.window_size = window_size
        # Recent outcomes: list of (operators_used_set, discovered_edges_count)
        self._outcomes: collections.deque = collections.deque(maxlen=window_size)
        # Per-operator: set of edges this operator has co-occurred with
        self._operator_edges: dict[str, set[int]] = defaultdict(set)
        # Per-edge: total number of executions where this edge was observed
        self._edge_total: dict[int, int] = defaultdict(int)
        # Per-edge: per-operator count of executions where both co-occurred
        self._edge_op_count: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Global edge set
        self._all_edges: set[int] = set()

    def record(
        self, operators: set[str], new_edges: int, edge_indices: set[int] | None = None
    ) -> None:
        """Record an execution outcome.

        Args:
            operators: Set of mutation operators used in this execution.
            new_edges: Number of new edges discovered (0 if none).
            edge_indices: Optional set of specific new edge indices.
        """
        self._outcomes.append((operators, new_edges))
        if edge_indices:
            for op in operators:
                self._operator_edges[op].update(edge_indices)
            for edge in edge_indices:
                self._edge_total[edge] += 1
                for op in operators:
                    self._edge_op_count[edge][op] += 1
            self._all_edges.update(edge_indices)
            if len(self._all_edges) > SHAPLEY_EDGES_MAX:
                self._prune_edges()

    def _prune_edges(self):
        """Drop oldest half of tracked edges to bound memory."""
        edges = sorted(self._all_edges)
        drop = edges[: len(edges) // 2]
        for edge in drop:
            self._all_edges.discard(edge)
            self._edge_total.pop(edge, None)
            self._edge_op_count.pop(edge, None)
            for op_edges in self._operator_edges.values():
                op_edges.discard(edge)

    def _edge_attribution(self, edge: int) -> dict[str, float]:
        """Compute frequency-weighted credit for a single edge.

        Returns dict mapping operator -> credit weight. Credit is
        proportional to co-occurrence frequency, normalized to sum to 1.
        """
        op_counts = self._edge_op_count.get(edge, {})
        total = sum(op_counts.values())
        if total == 0:
            return {}
        return {op: count / total for op, count in op_counts.items()}

    def _shapley_marginal(self, op: str, prefix_edges: set[int]) -> float:
        """Compute marginal contribution of one operator given already-covered edges."""
        marginal = 0.0
        for edge in self._operator_edges.get(op, set()):
            if edge not in prefix_edges:
                attr = self._edge_attribution(edge)
                marginal += attr.get(op, 0.0)
        return marginal

    def shapley_values(self, operators: list[str] | None = None) -> dict[str, float]:
        """Compute Shapley values using per-edge frequency-weighted attribution.

        For each edge, credit is distributed among operators proportional
        to co-occurrence frequency. The Shapley computation then determines
        marginal contributions given these per-edge credits.

        Returns:
            Dict mapping operator name -> Shapley value (in [0, 1]).
            Values sum to 1.0 (or less if some operators have zero contribution).
        """
        if not self._outcomes:
            return {op: 1.0 / max(1, len(operators or [])) for op in (operators or [])}

        if operators is None:
            operators = sorted({op for ops, _ in self._outcomes for op in ops})
        if not operators:
            return {}

        n_ops = len(operators)
        shapley = {op: 0.0 for op in operators}

        for _ in range(self.n_samples):
            perm = operators[:]
            random.shuffle(perm)

            prefix_edges: set[int] = set()
            for op in perm:
                marginal = self._shapley_marginal(op, prefix_edges)
                shapley[op] += marginal
                prefix_edges.update(self._operator_edges.get(op, set()))

        total = sum(shapley.values())
        if total > 0:
            return {op: v / total for op, v in shapley.items()}
        return {op: 1.0 / n_ops for op in operators}

    def operator_synergy(self, op_a: str, op_b: str) -> float:
        """Compute synergy between two operators.

        Synergy = I(X_a, X_b; Y) - I(X_a; Y) - I(X_b; Y)
        where X_a, X_b are operator usage indicators and Y is coverage.

        Positive = operators work better together than alone.
        Negative = operators are redundant.
        """
        edges_a = self._operator_edges.get(op_a, set())
        edges_b = self._operator_edges.get(op_b, set())
        if not edges_a or not edges_b:
            return 0.0

        # Approximate: joint coverage minus individual coverages
        joint = len(edges_a | edges_b)
        individual = len(edges_a) + len(edges_b)
        return (joint - individual) / max(1, individual)

    def operator_kernel(self, operators: list[str] | None = None) -> dict[str, dict[str, float]]:
        """Build a kernel matrix measuring operator similarity via Jaccard.

        K(i,j) = |E_i ∩ E_j| / |E_i ∪ E_j|
        High K → redundant operators. Low K → complementary.

        Args:
            operators: Operators to include. If None, uses all.

        Returns:
            Nested dict: kernel[op_a][op_b] = Jaccard similarity in [0, 1].
        """
        if operators is None:
            operators = sorted(self._operator_edges.keys())
        if len(operators) < 2:
            return {op: {op: 1.0} for op in operators}

        kernel: dict[str, dict[str, float]] = {op: {} for op in operators}

        for i, op_i in enumerate(operators):
            edges_i = self._operator_edges.get(op_i, set())
            for j, op_j in enumerate(operators):
                if i == j:
                    kernel[op_i][op_j] = 1.0
                elif i < j:
                    edges_j = self._operator_edges.get(op_j, set())
                    if not edges_i and not edges_j:
                        sim = 0.0
                    else:
                        intersection = len(edges_i & edges_j)
                        union = len(edges_i | edges_j)
                        sim = intersection / union if union > 0 else 0.0
                    kernel[op_i][op_j] = sim
                    kernel[op_j][op_i] = sim

        return kernel

    def operator_similarity(self, op_a: str, op_b: str) -> float:
        """Compute Jaccard similarity between two operators."""
        edges_a = self._operator_edges.get(op_a, set())
        edges_b = self._operator_edges.get(op_b, set())
        if not edges_a and not edges_b:
            return 0.0
        intersection = len(edges_a & edges_b)
        union = len(edges_a | edges_b)
        return intersection / union if union > 0 else 0.0

    def redundant_operators(
        self, threshold: float = 0.9, operators: list[str] | None = None
    ) -> list[tuple[str, str, float]]:
        """Find pairs of operators that are near-duplicates.

        Returns pairs where K(i,j) >= threshold, sorted by similarity.

        Args:
            threshold: Minimum Jaccard similarity to consider redundant.
            operators: Operators to check. If None, uses all.

        Returns:
            List of (op_a, op_b, similarity) tuples.
        """
        kernel = self.operator_kernel(operators)
        pairs = []
        seen = set()
        for op_a in kernel:
            for op_b in kernel[op_a]:
                if op_a == op_b:
                    continue
                key = (min(op_a, op_b), max(op_a, op_b))
                if key in seen:
                    continue
                seen.add(key)
                sim = kernel[op_a][op_b]
                if sim >= threshold:
                    pairs.append((op_a, op_b, sim))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def _spectral_embedding_numpy(
        self, operators: list[str], kernel: dict, k: int
    ) -> dict[str, list[float]]:
        """Numpy path for spectral_embedding."""
        n = len(operators)
        K = np.zeros((n, n), dtype=np.float64)
        for i, op_i in enumerate(operators):
            for j, op_j in enumerate(operators):
                K[i, j] = kernel[op_i].get(op_j, 0.0)
        degrees = K.sum(axis=1)
        d_inv_sqrt = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(d_inv_sqrt, 1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
        L = np.eye(n, dtype=np.float64) - d_inv_sqrt @ K @ d_inv_sqrt
        eigvals, eigvecs = np.linalg.eigh(L)
        embedding: dict[str, list[float]] = {}
        for idx, op in enumerate(operators):
            embedding[op] = [float(eigvecs[idx, d]) for d in range(k)]
        return embedding

    @staticmethod
    def _build_laplacian_py(kernel: dict, operators: list[str], n: int) -> list[list[float]]:
        """Build normalized Laplacian matrix (pure-Python)."""
        degrees = [0.0] * n
        for i in range(n):
            for j in range(n):
                degrees[i] += kernel[operators[i]].get(operators[j], 0.0)
        laplacian: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                w_ij = kernel[operators[i]].get(operators[j], 0.0)
                d_i = math.sqrt(degrees[i]) if degrees[i] > 0 else 1.0
                d_j = math.sqrt(degrees[j]) if degrees[j] > 0 else 1.0
                laplacian[i][j] = -w_ij / (d_i * d_j)
            laplacian[i][i] = 1.0
        return laplacian

    @staticmethod
    def _inverse_iteration_py(laplacian: list[list[float]], n: int) -> list[float]:
        """Inverse power iteration for smallest eigenvector (pure-Python)."""
        w = [random.gauss(0, 1) for _ in range(n)]
        norm = math.sqrt(sum(x * x for x in w))
        w = [x / norm for x in w]
        for _ in range(100):
            lw = [0.0] * n
            for i in range(n):
                for j in range(n):
                    lw[i] += laplacian[i][j] * w[j]
            new_w = [w[i] - 0.5 * lw[i] for i in range(n)]
            norm = math.sqrt(sum(x * x for x in new_w))
            if norm < 1e-12:
                break
            new_w = [x / norm for x in new_w]
            diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(w, new_w, strict=False)))
            w = new_w
            if diff < 1e-8:
                break
        return w

    def _spectral_embedding_py(
        self, operators: list[str], kernel: dict, k: int
    ) -> dict[str, list[float]]:
        """Pure-Python fallback for spectral_embedding."""
        n = len(operators)
        laplacian = self._build_laplacian_py(kernel, operators, n)
        eigenvectors: list[list[float]] = []
        for _ in range(k):
            w = self._inverse_iteration_py(laplacian, n)
            eigenvectors.append(w)
            for i in range(n):
                for j in range(n):
                    laplacian[i][j] -= w[i] * w[j]
        return {op: [eigenvectors[d][idx] for d in range(k)] for idx, op in enumerate(operators)}

    def spectral_embedding(
        self, operators: list[str] | None = None, k: int = 2
    ) -> dict[str, list[float]]:
        """Spectral embedding of operators using Laplacian eigenmap.

        Returns low-dimensional coordinates where similar operators cluster.

        Args:
            operators: Operators to embed. If None, uses all.
            k: Number of embedding dimensions.

        Returns:
            Dict mapping operator name -> [dim_0, dim_1, ...] coordinates.
        """
        if operators is None:
            operators = sorted(self._operator_edges.keys())
        n = len(operators)
        if n < k + 1:
            return {op: [0.0] * k for op in operators}

        kernel = self.operator_kernel(operators)

        if _HAS_NUMPY:
            return self._spectral_embedding_numpy(operators, kernel, k)
        return self._spectral_embedding_py(operators, kernel, k)

    def ranking(self, operators: list[str] | None = None) -> list[tuple[str, float]]:
        """Return operators ranked by Shapley value.

        Returns:
            List of (operator, shapley_value) sorted descending.
        """
        sv = self.shapley_values(operators)
        return sorted(sv.items(), key=lambda x: x[1], reverse=True)
