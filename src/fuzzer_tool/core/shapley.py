"""ShapleyAttribution: game-theoretic attribution of operators to edge coverage."""

import collections
import math
from collections import defaultdict
from collections.abc import Mapping

from fuzzer_tool.core.dc_eigh import eigh
from fuzzer_tool.core.marginal_coverage import DIRECT_MAX_ID, MarginalRow, marginal_table
from fuzzer_tool.core.misra_gries import reduce_counts

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


class _KeyViews(Mapping):
    """``{op: set-like of edges}`` read from ``{op: Counter}`` without copying or creating keys."""

    __slots__ = ("_counts",)

    def __init__(self, counts):
        self._counts = counts

    def __getitem__(self, op):
        if op not in self._counts:
            raise KeyError(op)
        return self._counts[op].keys()

    def __iter__(self):
        return iter(self._counts)

    def __len__(self):
        return len(self._counts)

    def __contains__(self, op):
        return op in self._counts


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
        # Per-edge: total number of executions where this edge was observed
        self._edge_total: collections.Counter = collections.Counter()
        # Per-operator: per-edge count of executions where both co-occurred. Kept
        # operator-major so record() updates it with one C-level Counter.update
        # per operator; ``_edge_op_count`` is the edge-major view, built on read.
        self._op_edge_count: dict[str, collections.Counter] = defaultdict(collections.Counter)
        # Per-operator: edges this operator has co-occurred with. A live key view of
        # ``_op_edge_count`` (the two always held identical sets), so nothing is stored twice.
        self._operator_edges = _KeyViews(self._op_edge_count)
        self._edge_view: dict[int, dict[str, int]] | None = None
        # How often the edge cap evicted the rarest half (marginal tables are then partial).
        self.prune_count = 0

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
                self._op_edge_count[op].update(edge_indices)
            self._edge_total.update(edge_indices)
            self._edge_view = None
            if len(self._edge_total) > SHAPLEY_EDGES_MAX:
                self._prune_edges()

    @property
    def _all_edges(self):
        """Tracked edges: the key view of ``_edge_total`` (one store, no per-exec set update)."""
        return self._edge_total.keys()

    @property
    def _edge_op_count(self) -> dict[int, dict[str, int]]:
        """Edge-major view ``{edge: {op: count}}``, in first-seen edge order; cached."""
        if self._edge_view is None:
            view: dict[int, dict[str, int]] = {edge: {} for edge in self._edge_total}
            for op, counts in self._op_edge_count.items():
                for edge, n in counts.items():
                    view[edge][op] = n
            self._edge_view = {edge: per_op for edge, per_op in view.items() if per_op}
        return self._edge_view

    def _prune_edges(self):
        """Bound tracked edges with batch Misra-Gries over ``_edge_total``.

        Subtracts the (k+1)-th largest total from every total and evicts the
        non-positive (``k = SHAPLEY_EDGES_MAX``): any edge seen in more than
        n/(k+1) of all (exec, edge) observations survives, and a stored total
        undercounts by at most that. Ties evict together, so edge-id order
        (``prev_loc ^ cur_loc`` hashes) never decides; sorting by id once
        distorted credit 5.3x. Replaced drop-the-rarest-half, which could evict
        edges above that bound. Per-operator co-occurrence counts stay exact for
        the edges kept.
        """
        dropped = set(reduce_counts(self._edge_total, SHAPLEY_EDGES_MAX))
        for counts in self._op_edge_count.values():
            # C-level intersection; the Python loop only visits edges the operator holds.
            hit = counts.keys() & dropped
            if hit:
                # dict.pop is C-level; Counter.__delitem__ is a Python-level override.
                collections.deque(map(counts.pop, hit), maxlen=0)
        self._edge_view = None
        self.prune_count += 1

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
        """Add credit(e, op) / |T_e| per edge to *shapley* in place.

        Works on the operator-major counts: one dense pass over every
        (operator, edge) pair, grouped by edge with ``np.unique``.
        """
        ops = sorted(op for op in op_set if self._op_edge_count.get(op))
        if not ops:
            return
        keys, vals, owner = [], [], []
        for i, op in enumerate(ops):
            counts = self._op_edge_count[op]
            n = len(counts)
            keys.append(np.fromiter(counts.keys(), dtype=np.int64, count=n))
            vals.append(np.fromiter(counts.values(), dtype=np.int64, count=n))
            owner.append(np.full(n, i, dtype=np.int64))
        ids, count, who = np.concatenate(keys), np.concatenate(vals), np.concatenate(owner)

        edge = ids
        if ids.min() < 0 or ids.max() >= DIRECT_MAX_ID:
            _, edge = np.unique(ids, return_inverse=True)
        total = np.bincount(edge, weights=count)
        touching = np.bincount(edge)
        credit = count / total[edge] / touching[edge]
        per_op = np.bincount(who, weights=credit, minlength=len(ops))
        for op, value in zip(ops, per_op.tolist(), strict=True):
            shapley[op] += value

    def operator_edge_sets(self) -> dict[str, set[int]]:
        """Copy of the per-operator edge sets."""
        return {op: set(edges) for op, edges in self._operator_edges.items()}

    def marginal_coverage(self) -> list[MarginalRow]:
        """Per-operator contribution to union coverage, in greedy cover order."""
        # marginal_table only reads; skip the per-set copy (~as costly as the table).
        return marginal_table(self._operator_edges)

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
