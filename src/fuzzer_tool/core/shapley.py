"""ShapleyAttribution: game-theoretic attribution of operators to edge coverage."""

import collections
import math
from collections import defaultdict

from fuzzer_tool.core.dc_eigh import eigh

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ── Memory bounds ────────────────────────────────────────────────────
SHAPLEY_EDGES_MAX = 10_000  # max edges tracked in Shapley attribution
DEGREE_FLOOR = 1e-12  # Laplacian degree floor: isolated operators stay finite
FAMILY_THRESHOLD = 0.9  # Jaccard at or above which two operators are near-duplicates
FIEDLER_DIM = 1  # embedding dim 0 is the trivial sqrt-degree vector
# numpy B·Bᵀ beat the pure loop at every set size measured from 32 operators up
# (n=32: 61 vs 86 us at 1 edge each, 6.9 vs 30 ms at 2000); below, the loop wins.
KERNEL_NUMPY_MIN_OPS = 32


def _jaccard_py(sets: list[set[int]]) -> list[list[float]]:
    """Pairwise Jaccard, diagonal 1; |A ∪ B| = |A| + |B| - |A ∩ B| skips the union set."""
    n = len(sets)
    sizes = [len(s) for s in sets]
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        out[i][i] = 1.0
        a, size_a = sets[i], sizes[i]
        for j in range(i + 1, n):
            inter = len(a & sets[j])
            union = size_a + sizes[j] - inter
            out[i][j] = out[j][i] = inter / union if union else 0.0
    return out


def _jaccard_numpy(sets: list[set[int]]) -> list[list[float]]:
    """``_jaccard_py`` via an operator-by-edge 0/1 matrix: B·Bᵀ counts every intersection.

    Counts stay below 2^53, so float64 products and the final division are
    exact and match the pure loop bit for bit.
    """
    n = len(sets)
    sizes = np.fromiter(map(len, sets), dtype=np.int64, count=n)
    flat = np.fromiter((e for s in sets for e in s), dtype=np.int64, count=int(sizes.sum()))
    _, cols = np.unique(flat, return_inverse=True)

    b = np.zeros((n, int(cols.max()) + 1 if cols.size else 0))
    b[np.repeat(np.arange(n), sizes), cols] = 1.0
    inter = b @ b.T
    union = sizes[:, None] + sizes[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(sim, 1.0)
    return sim.tolist()


def _components(pairs: list[tuple[str, str, float]]) -> list[list[str]]:
    """Connected components of the graph whose edges are *pairs* (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.setdefault(root, root) != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b, _ in pairs:
        parent[find(a)] = find(b)

    groups: dict[str, list[str]] = defaultdict(list)
    for x in parent:
        groups[find(x)].append(x)
    return list(groups.values())


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
        n_samples: Unused by :meth:`shapley_values`, which is now computed
            in closed form (see its docstring) rather than by permutation
            sampling. Kept as a constructor/attribute for backward
            compatibility with existing callers and configs.
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
        """Drop the least-frequent half of tracked edges to bound memory.

        Edge ids are ``prev_loc ^ cur_loc`` hashes — numerical order is not
        age.  Sorting by id and dropping the low half systematically
        under-credits operators whose edges land in the low half of the
        map (measured 5.3× distortion under equal productivity).  Evict by
        ``_edge_total`` ascending so rare edges go first; frequency is the
        quantity Shapley already tracks.
        """
        # Sort by observed frequency (ascending); ties broken by edge id for
        # determinism.
        ranked = sorted(self._all_edges, key=lambda e: (self._edge_total.get(e, 0), e))
        drop = ranked[: len(ranked) // 2]
        for edge in drop:
            self._all_edges.discard(edge)
            self._edge_total.pop(edge, None)
            self._edge_op_count.pop(edge, None)
            for op_edges in self._operator_edges.values():
                op_edges.discard(edge)

    def shapley_values(self, operators: list[str] | None = None) -> dict[str, float]:
        """Compute exact Shapley values for the per-edge frequency-weighted game.

        This game has a closed form, so no permutation sampling is needed.
        For each edge ``e``, let ``T_e`` be the operators (restricted to
        ``operators``) that co-occurred with ``e``, and let
        ``credit(e, op) = count(e, op) / sum(count(e, ·) for · in T_e)`` be
        the fixed per-edge credit split (``_edge_op_count``). An
        operator only earns that credit in a given permutation if it is the
        *first* member of ``T_e`` to appear — and by symmetry, in a uniformly
        random permutation each of the ``|T_e|`` members is equally likely to
        be first. So the expected (= exact, since this is a linear
        functional of a uniform distribution) marginal contribution of ``op``
        from edge ``e`` is simply ``credit(e, op) / |T_e|``, with no
        dependence on the other operators' relative order.

        Summing that over every edge gives the exact Shapley value directly,
        replacing an O(n_samples · n_ops · edges) Monte-Carlo estimate with a
        single O(total edge-operator co-occurrences) pass — exact rather
        than approximate, and independent of ``n_samples``. Verified against
        the old permutation sampler: the two agree to within Monte-Carlo
        noise that shrinks as ``n_samples`` grows (see
        ``test_shapley_closed_form_identity.py``).

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
        op_set = set(operators)
        shapley = {op: 0.0 for op in operators}

        self._add_edge_credit(op_set, shapley)

        total = sum(shapley.values())
        if total > 0:
            return {op: v / total for op, v in shapley.items()}
        return {op: 1.0 / n_ops for op in operators}

    def _add_edge_credit(self, op_set: set[str], shapley: dict[str, float]) -> None:
        """Add credit(e, op) / |T_e| per edge to *shapley* in place."""
        for _edge, op_counts in self._edge_op_count.items():
            touching = [op for op in op_counts if op in op_set]
            k = len(touching)
            if k == 0:
                continue
            total = sum(op_counts[op] for op in touching)
            if total == 0:
                continue
            for op in touching:
                shapley[op] += op_counts[op] / total / k

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

        sets = [self._operator_edges.get(op, set()) for op in operators]
        use_numpy = _HAS_NUMPY and len(operators) >= KERNEL_NUMPY_MIN_OPS
        rows = _jaccard_numpy(sets) if use_numpy else _jaccard_py(sets)
        return {
            op: dict(zip(operators, row, strict=True))
            for op, row in zip(operators, rows, strict=True)
        }

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
        pairs = self._redundant_pairs(self.operator_kernel(operators), threshold)
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    @staticmethod
    def _redundant_pairs(kernel: dict, threshold: float) -> list[tuple[str, str, float]]:
        """Unordered operator pairs with K >= threshold, in kernel order."""
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
        return pairs

    def _spectral_embedding_numpy(
        self, operators: list[str], kernel: dict, k: int
    ) -> dict[str, list[float]]:
        """Numpy path for spectral_embedding."""
        n = len(operators)
        K = np.zeros((n, n), dtype=np.float64)
        for i, op_i in enumerate(operators):
            for j, op_j in enumerate(operators):
                K[i, j] = kernel[op_i].get(op_j, 0.0)
        # D^-1/2 K D^-1/2 with D diagonal is a row-and-column rescale, not a
        # pair of matrix products: (D^-1/2 K D^-1/2)_ij == K_ij * s_i * s_j
        # for s = diag(D)^-1/2. Materializing D^-1/2 as a dense n-by-n and
        # multiplying through spends 2*O(n^3) to apply n numbers; the outer
        # form is O(n^2) and, since every product is a scalar times an
        # entry rather than a summed row, bit-identical rather than merely
        # equal to tolerance.
        s = 1.0 / np.sqrt(np.maximum(K.sum(axis=1), DEGREE_FLOOR))
        L = np.eye(n, dtype=np.float64) - K * s[:, None] * s[None, :]
        eigvals, eigvecs = np.linalg.eigh(L)
        embedding: dict[str, list[float]] = {}
        for idx, op in enumerate(operators):
            embedding[op] = [float(eigvecs[idx, d]) for d in range(k)]
        return embedding

    @staticmethod
    def _build_laplacian_py(kernel: dict, operators: list[str], n: int) -> list[list[float]]:
        """Normalized Laplacian I - D^-1/2 K D^-1/2, self-loops kept (pure-Python).

        Same matrix as the numpy path: the diagonal is 1 - K_ii / deg_i.
        """
        rows = [[kernel[a].get(b, 0.0) for b in operators] for a in operators]
        s = [1.0 / math.sqrt(max(sum(row), DEGREE_FLOOR)) for row in rows]
        laplacian = [
            [-w * si * sj for w, sj in zip(row, s, strict=True)]
            for row, si in zip(rows, s, strict=True)
        ]
        for i in range(n):
            laplacian[i][i] += 1.0
        return laplacian

    def _spectral_embedding_py(
        self, operators: list[str], kernel: dict, k: int
    ) -> dict[str, list[float]]:
        """Pure-Python fallback for spectral_embedding."""
        n = len(operators)
        laplacian = self._build_laplacian_py(kernel, operators, n)
        _, vecs = eigh(laplacian, count=k)
        return {op: vecs[idx] for idx, op in enumerate(operators)}

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

        return self._embed(operators, self.operator_kernel(operators), k)

    def _embed(self, operators: list[str], kernel: dict, k: int) -> dict[str, list[float]]:
        if _HAS_NUMPY:
            return self._spectral_embedding_numpy(operators, kernel, k)
        return self._spectral_embedding_py(operators, kernel, k)

    def operator_families(self, threshold: float = FAMILY_THRESHOLD) -> list[list[str]]:
        """Near-duplicate operator families, laid out in spectral order.

        A family is a connected component of the K >= threshold graph, so it
        is transitive: a~b and b~c group a, b, c even when K(a, c) < threshold.
        Singletons are dropped. Members, and families by their mean, are
        ordered by the Fiedler coordinate, which places operators with
        similar coverage next to each other.

        Args:
            threshold: Minimum Jaccard similarity for a near-duplicate pair.

        Returns:
            Families as lists of operator names; empty when none qualify.
        """
        operators = sorted(self._operator_edges)
        if len(operators) < 2:
            return []

        # One kernel for both the pair graph and the embedding (O(n^2) set ops).
        kernel = self.operator_kernel(operators)
        pairs = self._redundant_pairs(kernel, threshold)
        if not pairs:
            return []

        fiedler = self._fiedler(operators, kernel)
        families = [sorted(fam, key=lambda op: (fiedler[op], op)) for fam in _components(pairs)]
        families.sort(key=lambda fam: (math.fsum(fiedler[op] for op in fam) / len(fam), fam[0]))
        return families

    def _fiedler(self, operators: list[str], kernel: dict) -> dict[str, float]:
        """Fiedler coordinate per operator; zeros when there are too few to embed."""
        k = FIEDLER_DIM + 1
        if len(operators) < k + 1:
            return dict.fromkeys(operators, 0.0)
        return {op: v[FIEDLER_DIM] for op, v in self._embed(operators, kernel, k).items()}

    def ranking(self, operators: list[str] | None = None) -> list[tuple[str, float]]:
        """Return operators ranked by Shapley value.

        Returns:
            List of (operator, shapley_value) sorted descending.
        """
        sv = self.shapley_values(operators)
        return sorted(sv.items(), key=lambda x: x[1], reverse=True)
