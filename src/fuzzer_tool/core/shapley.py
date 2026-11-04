"""ShapleyAttribution: game-theoretic attribution of operators to edge coverage."""

import collections
import math
from collections import defaultdict
from collections.abc import Mapping
from operator import itemgetter

from fuzzer_tool.core.dc_eigh import eigh
from fuzzer_tool.core.marginal_coverage import (
    DIRECT_MAX_ID,
    MarginalRow,
    marginal_rows,
    marginal_table,
)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ── Memory bounds ────────────────────────────────────────────────────
# Execs with at least this many increments (edges x (ops + 1)) take one fused
# numpy scatter; below, C-level Counter.update wins (record() per exec: 25 edges
# x 1 op 3.2 vs 7.1 us; 25 x 10 16.9 vs 9.6; 2000 x 4 740 vs 350).
FUSED_MIN_WORK = 250
# Pending small-exec increments folded into the matrix past this (~20 MB of Counters).
PENDING_MAX = 200_000
# Initial count matrix (operator lanes x edge slots); each axis doubles on demand.
COUNTS_EDGES0 = 1024
COUNTS_OPS0 = 64
TOTAL_LANE = 0  # pending-buffer index of the per-edge totals
UINT32_WIDEN_AT = 1 << 31  # widen the uint32 matrix to int64 once a total reaches this
# Counter store -> dense store, decided once: after DENSE_WARMUP records when the
# mean work per exec reaches FUSED_MIN_WORK (measured crossover: 25 edges x <=4 ops
# Counters 12.9 vs 19.8 us/exec; 60 x <=8 53.7 vs 35.9), or whenever the Counters
# hold COUNTER_PAIRS_MAX (op, edge) pairs (~100 B each, so ~100 MB).
DENSE_WARMUP = 1000
COUNTER_PAIRS_MAX = 1_000_000
PAIRS_CHECK_EVERY = 1024  # records between exact pair counts once the bound is near
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


class _CounterCounts:
    """Exact counts in one Counter per operator: fastest for narrow execs, ~100 B per pair.

    ``record()`` is one C-level ``Counter.update`` per operator. Nothing is
    evicted; ``wants_dense()`` tells the owner when to migrate to
    ``_EdgeCounts`` (wide execs, or too many pairs for Counters).
    """

    __slots__ = ("op", "total", "records", "work", "check_at")

    def __init__(self):
        self.op: dict[str, collections.Counter] = defaultdict(collections.Counter)
        self.total: collections.Counter = collections.Counter()
        self.records = 0
        self.work = 0  # increments recorded: edges x (ops + 1) per exec
        self.check_at = 1  # records at which the owner next asks wants_dense()

    def add(self, ops, edges: set[int]) -> None:
        for op in ops:
            self.op[op].update(edges)
        self.total.update(edges)
        self.records += 1
        self.work += len(edges) * (len(ops) + 1)

    def wants_dense(self) -> bool:
        """Wide execs on average past warmup, or the pair count past the memory budget.

        Called only when ``records >= check_at``, so record() pays one int compare.
        """
        r = self.records
        step = r + PAIRS_CHECK_EVERY
        self.check_at = step if r >= DENSE_WARMUP else min(step, DENSE_WARMUP)
        if r >= DENSE_WARMUP and self.work >= FUSED_MIN_WORK * r:
            return True
        # len(total) x len(op) bounds the pairs; count exactly only when it could be over.
        if len(self.total) * len(self.op) < COUNTER_PAIRS_MAX:
            return False
        return sum(map(len, self.op.values())) >= COUNTER_PAIRS_MAX

    def op_counts(self) -> dict[str, collections.Counter]:
        return self.op

    def edge_sets(self) -> Mapping:
        return _KeyViews(self.op)

    def totals(self) -> collections.Counter:
        return self.total

    def edge_keys(self):
        return self.total.keys()

    def edge_order(self):
        return self.total

    def entries(self) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, int]:
        """``(ops, op index, edge index, count, n edges)`` over every nonzero pair."""
        ops = [op for op, c in self.op.items() if c]
        if not ops:
            empty = np.zeros(0, dtype=np.int64)
            return ops, empty, empty, empty, 0
        keys, vals, owner = [], [], []
        for i, op in enumerate(ops):
            counts = self.op[op]
            n = len(counts)
            keys.append(np.fromiter(counts.keys(), dtype=np.int64, count=n))
            vals.append(np.fromiter(counts.values(), dtype=np.int64, count=n))
            owner.append(np.full(n, i, dtype=np.int64))
        ids, count, who = np.concatenate(keys), np.concatenate(vals), np.concatenate(owner)
        if ids.min() < 0 or ids.max() >= DIRECT_MAX_ID:
            _, edge = np.unique(ids, return_inverse=True)
            return ops, who, edge, count, int(edge.max()) + 1
        return ops, who, ids, count, int(ids.max()) + 1

    def marginal(self) -> list[MarginalRow]:
        # marginal_table only reads; skip the per-set copy (~as costly as the table).
        return marginal_table(_KeyViews(self.op))


class _EdgeCounts:
    """Exact (operator, edge) and per-edge co-occurrence counts; never evicts.

    Every edge seen keeps a slot (a matrix column) for the life of the run, so
    memory is ``4 B x lanes x edges`` (264 ops x 30k edges ~ 32 MB) instead of
    ~100 B per pair in per-operator Counters, which is what forced the old
    edge cap. Wide execs write the matrix with one fused numpy scatter; small
    execs keep C-level ``Counter.update`` into a pending buffer, folded into
    the matrix before any read or once it holds ``PENDING_MAX`` entries.

        mat (uint32):   slot 0   slot 1   slot 2      ids[slot] = edge id
        lane 1 "flip"   3        0        1           lanes["flip"] = 1
        lane 2 "splice" 0        2        1
        total (int64)   3        2        2           execs that saw the edge
        + pending[lane] Counters (lane 0 = total), not yet folded
    """

    __slots__ = (
        "lanes",
        "slot",
        "ids",
        "mat",
        "total",
        "pending",
        "pending_work",
        "bound",
        "_entries",
    )

    def __init__(self):
        self.lanes: dict[str, int] = {}  # op -> mat row (row 0 unused: TOTAL_LANE)
        self.slot: dict[int, int] = {}  # edge id -> mat column
        self.ids: list[int] = []  # mat column -> edge id
        self.mat = np.zeros((1 + COUNTS_OPS0, COUNTS_EDGES0), dtype=np.uint32)
        self.total = np.zeros(COUNTS_EDGES0, dtype=np.int64)
        self.pending: list[collections.Counter] = [collections.Counter()]  # by lane
        self.pending_work = 0  # increments buffered since the last fold (>= entries)
        self.bound = 0  # >= every total: an exec adds at most 1 per edge
        self._entries: tuple | None = None

    def add(self, ops, edges: set[int]) -> None:
        """Count one exec: each op in *ops* co-occurred once with each edge in *edges*."""
        self._entries = None

        # A pair count never exceeds its edge's total; widen before uint32 can wrap.
        self.bound += 1
        if self.bound == UINT32_WIDEN_AT:
            self.mat = self.mat.astype(np.int64)

        new = edges.difference(self.slot)  # exact-dict fast path; usually empty
        if new:
            self._add_slots(list(new))

        lanes = self.lanes
        idx = [lanes.get(op) or self._add_lane(op) for op in ops]  # lanes start at 1
        work = len(edges) * (len(idx) + 1)
        if work >= FUSED_MIN_WORK:
            self._add_fused(idx, edges)
            return

        pending = self.pending
        pending[TOTAL_LANE].update(edges)
        for lane in idx:
            pending[lane].update(edges)
        self.pending_work += work
        if self.pending_work > PENDING_MAX:
            self.fold()

    def _add_fused(self, idx: list[int], edges: set[int]) -> None:
        # ndmin: itemgetter returns a bare value, not a tuple, for one edge.
        cols = np.array(itemgetter(*edges)(self.slot), dtype=np.intp, ndmin=1)
        self.total[cols] += 1
        if not idx:
            return
        # Unique (lane, slot) pairs, so a plain fancy += counts each once.
        flat = (np.array(idx, dtype=np.intp)[:, None] * self.mat.shape[1] + cols).ravel()
        self.mat.reshape(-1)[flat] += 1

    def _add_slots(self, fresh: list[int]) -> None:
        start = len(self.ids)
        self.ids.extend(fresh)
        self.slot.update(zip(fresh, range(start, len(self.ids)), strict=True))
        if len(self.ids) > self.mat.shape[1]:
            self._grow(self.mat.shape[0], max(2 * self.mat.shape[1], len(self.ids)))

    def _add_lane(self, op: str) -> int:
        lane = len(self.pending)
        if lane == self.mat.shape[0]:
            self._grow(2 * self.mat.shape[0] - 1, self.mat.shape[1])
        self.lanes[op] = lane
        self.pending.append(collections.Counter())
        return lane

    def _grow(self, n_lanes: int, n_slots: int) -> None:
        mat = np.zeros((n_lanes, n_slots), dtype=self.mat.dtype)
        lanes, slots = self.mat.shape
        mat[:lanes, :slots] = self.mat
        self.mat = mat
        total = np.zeros(n_slots, dtype=np.int64)
        total[: len(self.total)] = self.total
        self.total = total

    def fold(self) -> None:
        """Move the pending Counters into the matrix and totals."""
        get = self.slot.__getitem__
        for lane, counts in enumerate(self.pending):
            if not counts:
                continue
            k = len(counts)
            cols = np.fromiter(map(get, counts), dtype=np.intp, count=k)
            vals = np.fromiter(counts.values(), dtype=np.int64, count=k)
            if lane == TOTAL_LANE:
                self.total[cols] += vals
            else:
                self.mat[lane, cols] += vals.astype(self.mat.dtype)
            counts.clear()
        self.pending_work = 0

    def pending_size(self) -> int:
        """Distinct (lane, edge) entries waiting in the pending Counters."""
        return sum(map(len, self.pending))

    def entries(self) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, int]:
        """``(ops, op index, slot, count, n slots)`` per nonzero pair, by op then slot; cached."""
        if self._entries is None:
            self.fold()
            block = self.mat[1 : len(self.pending), : len(self.ids)]
            who, slot = np.nonzero(block)  # row-major: by op, then slot
            count = block[who, slot].astype(np.int64)
            self._entries = (list(self.lanes), who, slot, count, len(self.ids))
        return self._entries

    def id_array(self) -> np.ndarray:
        """Edge id per slot; ``object`` dtype when an id does not fit int64."""
        try:
            return np.array(self.ids, dtype=np.int64)
        except OverflowError:  # ids are only hashed; keep them as Python ints
            return np.array(self.ids, dtype=object)

    def _by_op(self) -> tuple[list[str], list[int], list[int], np.ndarray]:
        ops, who, slot, count, _ = self.entries()
        ids = self.id_array()[slot].tolist()
        bounds = np.searchsorted(who, np.arange(len(ops) + 1)).tolist()
        return ops, ids, bounds, count

    def per_op(self) -> dict[str, dict[int, int]]:
        """``{op: {edge: count}}`` for every operator counted."""
        ops, ids, bounds, count = self._by_op()
        count = count.tolist()
        return {
            op: dict(zip(ids[a:b], count[a:b], strict=True))
            for op, a, b in zip(ops, bounds, bounds[1:], strict=False)
        }

    def edge_sets(self) -> dict[str, set[int]]:
        """``{op: edges}`` for every operator counted."""
        ops, ids, bounds, _ = self._by_op()
        return {op: set(ids[a:b]) for op, a, b in zip(ops, bounds, bounds[1:], strict=False)}

    def edges(self) -> dict[int, int]:
        """``{edge: execs that saw it}`` in first-seen exec order."""
        self.fold()
        return dict(zip(self.ids, self.total[: len(self.ids)].tolist(), strict=True))

    @classmethod
    def from_counters(cls, cc: "_CounterCounts") -> "_EdgeCounts":
        """Exact copy of a Counter store (one-time migration)."""
        dense = cls()
        dense._add_slots(list(cc.total))
        get = dense.slot.__getitem__
        n = len(cc.total)
        dense.total[:n] = np.fromiter(cc.total.values(), dtype=np.int64, count=n)
        for op, counts in cc.op.items():
            lane = dense._add_lane(op)
            k = len(counts)
            cols = np.fromiter(map(get, counts), dtype=np.intp, count=k)
            dense.mat[lane, cols] = np.fromiter(counts.values(), dtype=np.int64, count=k)
        dense.bound = int(dense.total.max(initial=0))
        if dense.bound >= UINT32_WIDEN_AT:
            dense.mat = dense.mat.astype(np.int64)
        return dense

    # Never migrates further: record() never reaches check_at.
    records = 0
    check_at = 1 << 62

    def op_counts(self) -> dict[str, collections.Counter]:
        return {op: collections.Counter(c) for op, c in self.per_op().items()}

    def totals(self) -> collections.Counter:
        return collections.Counter(self.edges())

    def edge_keys(self):
        return self.slot.keys()

    def edge_order(self):
        return self.ids

    def marginal(self) -> list[MarginalRow]:
        # Slots are a dense edge index already grouped by operator: no set flatten.
        ops, who, slot, _, _ = self.entries()
        return marginal_rows(ops, np.bincount(who, minlength=len(ops)), slot)


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
        # Exact per (operator, edge) and per-edge counts for every edge ever seen.
        # Starts on Counters; migrates once to the dense store (wide execs or too
        # many pairs). The views below are cached until the next write.
        self._counts: _CounterCounts | _EdgeCounts = _CounterCounts()
        self._op_view: dict[str, collections.Counter] | None = None
        self._sets_view: dict[str, set[int]] | None = None
        self._edge_view: dict[int, dict[str, int]] | None = None

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
            self._counts.add(operators, edge_indices)
            self._op_view = self._sets_view = self._edge_view = None
            counts = self._counts
            if counts.records >= counts.check_at and counts.wants_dense():
                self._counts = _EdgeCounts.from_counters(counts)

    @property
    def _op_edge_count(self) -> dict[str, collections.Counter]:
        """{op: Counter({edge: count})}; cached until the next write. Read-only."""
        if self._op_view is None:
            self._op_view = self._counts.op_counts()
        return self._op_view

    @property
    def _operator_edges(self) -> Mapping:
        """{op: edges} co-occurred with; cached until the next write. Read-only."""
        if self._sets_view is None:
            self._sets_view = self._counts.edge_sets()
        return self._sets_view

    @property
    def _edge_total(self) -> collections.Counter:
        """{edge: execs that saw it}."""
        return self._counts.totals()

    @property
    def _all_edges(self):
        """Every edge seen (set-like key view; nothing is ever evicted)."""
        return self._counts.edge_keys()

    @property
    def _edge_op_count(self) -> dict[int, dict[str, int]]:
        """Edge-major view {edge: {op: count}}, in first-seen edge order; cached."""
        if self._edge_view is None:
            view: dict[int, dict[str, int]] = {edge: {} for edge in self._counts.edge_order()}
            for op, counts in self._op_edge_count.items():
                for edge, n in counts.items():
                    view[edge][op] = n
            self._edge_view = {edge: per_op for edge, per_op in view.items() if per_op}
        return self._edge_view

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

        One pass over the nonzero (operator, slot) pairs; slots are already a
        dense edge index, so no np.unique over edge ids.
        """
        ops, who, edge, count, n = self._counts.entries()
        pick = np.fromiter((op in op_set for op in ops), dtype=bool, count=len(ops))
        if not who.size or not pick.any():
            return
        mask = pick[who]
        who, edge, count = who[mask], edge[mask], count[mask]

        total = np.bincount(edge, weights=count, minlength=n)
        touching = np.bincount(edge, minlength=n)
        credit = count / total[edge] / touching[edge]
        per_op = np.bincount(who, weights=credit, minlength=len(ops))
        for op, value, picked in zip(ops, per_op.tolist(), pick.tolist(), strict=True):
            if picked:
                shapley[op] += value

    def operator_edge_sets(self) -> dict[str, set[int]]:
        """Copy of the per-operator edge sets."""
        return {op: set(edges) for op, edges in self._operator_edges.items()}

    def marginal_coverage(self) -> list[MarginalRow]:
        """Per-operator contribution to union coverage, in greedy cover order."""
        return self._counts.marginal()

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
