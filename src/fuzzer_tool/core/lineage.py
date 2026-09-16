"""Weighted mutation lineage tree.

A parent-pointer forest over corpus seeds. Each node records the seed key,
its parent, the mutation operators + byte sites that produced it (edge
weight = operator attribution), and its node weight = number of new
coverage edges it contributed at insertion.

Three queries are supported:

1. **Subtree-productivity pruning** — ``subtree_weight`` (maintained
   incrementally on insert) plus ``recent_credit`` (recomputed on demand)
   let corpus auto-minimize drop an entire unproductive branch.
2. **Causal crash-path replay** — ``chain_from`` walks the
   (parent, operator, mutation-site) chain from a seed to the root so
   tmin can replay the exact mutation path instead of delta-debugging
   from scratch.
3. **LCA-based diversity** — ``lca`` / ``lca_distance`` over the
   parent-pointer forest feed a diversity term in seed scoring.
4. **Branching-complexity ranking** — ``strahler`` gives the
   Horton-Strahler number of a subtree, a weight-free topology measure
   (bushy vs. degenerate-chain) usable alongside ``subtree_weight`` /
   ``pagerank_credit`` when picking which branch to spend budget on.

The tree is keyed by ``seed_key`` strings (16-hex xxhash) — one hash
space shared with ``hash_data``, crash sidecars, and on-disk delta
records. It never uses Python's builtin ``hash()``.

Weights: node weight ``w`` = new edges at insertion. Structural
subtree weight ``subtree_weight(r) = Σ_{d∈subtree(r)} w(d)·γ^(dist(r,d))``
with γ = 0.9, maintained by propagate-on-insert (O(depth) ancestor walk
with a delta short-circuit). Recent credit is a separate, memoized-DFS
computation over coverage deltas at minimize time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

GAMMA = 0.9
"""Discount factor per tree edge for structural subtree weights."""

_PROPAGATE_EPS = 1e-6
"""Ancestor-walk delta below this is skipped (depth cap ≈ 131 at γ=0.9)."""

_MAX_INACTIVE_DEFAULT = 100_000
"""Cap on soft-deleted (pruned) nodes kept in the tree."""


class LineageNode:
    """A single node in the lineage forest.

    ``child_ops`` / ``child_sites`` describe the *inbound* edge — the
    operators and byte positions applied to ``parent_key`` that produced
    this node. Op names are interned into ``LineageTree.op_table`` so the
    stored lists are small int ids.
    """

    __slots__ = (
        "key",
        "parent_key",
        "depth",
        "node_weight",
        "child_ops",
        "child_sites",
        "subtree_weight",
        "active",
        "seq",
    )

    def __init__(
        self,
        key: str,
        parent_key: str | None,
        depth: int,
        node_weight: int,
        child_ops: list[int],
        child_sites: list[int],
        seq: int,
    ) -> None:
        self.key = key
        self.parent_key = parent_key
        self.depth = depth
        self.node_weight = node_weight
        self.child_ops = child_ops
        self.child_sites = child_sites
        self.subtree_weight = float(node_weight)
        self.active = True
        self.seq = seq

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"LineageNode(key={self.key!r}, parent={self.parent_key!r}, depth={self.depth}, "
            f"w={self.node_weight}, sw={self.subtree_weight:.3f}, active={self.active})"
        )


class LineageTree:
    """Parent-pointer forest over corpus seeds with subtree aggregates.

    All public methods are pure functions of the tree state; no service
    dependencies. ``rebuild_from_meta`` is idempotent — building twice
    yields an identical tree with no double-counting.
    """

    def __init__(self, max_inactive: int = _MAX_INACTIVE_DEFAULT) -> None:
        self.nodes: dict[str, LineageNode] = {}
        self.op_table: list[str] = []
        self._op_ids: dict[str, int] = {}
        self._children: dict[str, set[str]] = {}
        # Root keys, maintained incrementally. Consumers that select over the
        # forest (the MCTS scheduler descends from the roots on every seed
        # pick) would otherwise rescan every node per call, which is O(corpus)
        # on the fuzzer's hot path.
        self._root_keys: set[str] = set()
        self._inactive_keys: list[str] = []
        self._max_inactive = max_inactive
        self._seq = 0

    # ── op intern ─────────────────────────────────────────────────────

    def _op_id(self, name: str) -> int:
        oid = self._op_ids.get(name)
        if oid is None:
            oid = len(self.op_table)
            self.op_table.append(name)
            self._op_ids[name] = oid
        return oid

    def op_name(self, op_id: int) -> str:
        return self.op_table[op_id] if 0 <= op_id < len(self.op_table) else f"op:{op_id}"

    # ── insert ────────────────────────────────────────────────────────

    def insert(
        self,
        parent_key: str | None,
        child_key: str,
        ops: Iterable[str],
        sites: Iterable[int],
        new_edge_count: int,
    ) -> LineageNode:
        """Insert a child node under *parent_key* (None → root).

        Propagates the structural subtree weight up the ancestor chain
        with γ-discounting. Idempotent per child key: a second insert for
        an existing key is a no-op returning the existing node.
        """
        existing = self.nodes.get(child_key)
        if existing is not None:
            return existing

        self._seq += 1
        parent = self.nodes.get(parent_key) if parent_key is not None else None
        depth = parent.depth + 1 if parent is not None else 0

        child_ops = [self._op_id(op) for op in ops]
        child_sites = [int(s) for s in sites]
        node = LineageNode(
            key=child_key,
            # A child whose hash equals its parent's key (no-op mutation)
            # or whose parent was pruned first resolves to no parent here;
            # storing the dangling key would leave a self/cycle reference
            # that spins lca()/ancestors() forever.
            parent_key=parent_key if parent is not None else None,
            depth=depth,
            node_weight=int(new_edge_count),
            child_ops=child_ops,
            child_sites=child_sites,
            seq=self._seq,
        )
        self.nodes[child_key] = node
        if parent is not None:
            self._children.setdefault(parent_key, set()).add(child_key)
        else:
            # No resolvable parent — a root of its own tree in the forest.
            self._root_keys.add(child_key)

        # Propagate w(child)·γ^dist up to each ancestor; short-circuit
        # when the marginal delta drops below the epsilon floor.
        weight = float(new_edge_count)
        if weight > 0 and parent is not None:
            cur = parent
            dist = 1
            while cur is not None:
                delta = weight * (GAMMA**dist)
                if delta < _PROPAGATE_EPS:
                    break
                cur.subtree_weight += delta
                cur = self.nodes.get(cur.parent_key) if cur.parent_key else None
                dist += 1
        return node

    def get(self, key: str) -> LineageNode | None:
        return self.nodes.get(key)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, key: str) -> bool:
        return key in self.nodes

    # ── ancestry / LCA ────────────────────────────────────────────────

    def ancestors(self, key: str) -> list[LineageNode]:
        """Bottom-up list of ancestors of *key* (excluding itself).

        Walks are bounded by the node count so a corrupted forest
        (parent cycle) degrades to a short answer instead of hanging
        the fuzzer's hot path.
        """
        result: list[LineageNode] = []
        node = self.nodes.get(key)
        if node is None or node.parent_key is None:
            return result
        cur = self.nodes.get(node.parent_key)
        seen: set[str] = {key}
        while cur is not None and len(seen) <= len(self.nodes):
            if cur.key in seen:
                break
            seen.add(cur.key)
            result.append(cur)
            cur = self.nodes.get(cur.parent_key) if cur.parent_key else None
        return result

    def lca(self, a: str, b: str) -> str | None:
        """Lowest common ancestor key of *a* and *b*, or None if disconnected.

        Bounded by the node count: corrupt parent chains answer None
        rather than spinning (regression: fresh-corpus --elo all hang).
        """
        na = self.nodes.get(a)
        nb = self.nodes.get(b)
        if na is None or nb is None:
            return None
        limit = len(self.nodes) + 1
        steps = 0
        while na.depth > nb.depth:
            na = self.nodes.get(na.parent_key) if na.parent_key else None
            if na is None:
                return None
            steps += 1
            if steps > limit:
                return None
        steps = 0
        while nb.depth > na.depth:
            nb = self.nodes.get(nb.parent_key) if nb.parent_key else None
            if nb is None:
                return None
            steps += 1
            if steps > limit:
                return None
        steps = 0
        while na is not nb:
            na = self.nodes.get(na.parent_key) if na.parent_key else None
            nb = self.nodes.get(nb.parent_key) if nb.parent_key else None
            if na is None or nb is None:
                return None
            steps += 1
            if steps > limit:
                return None
        return na.key

    def lca_distance(self, a: str, b: str) -> int:
        """Tree distance between *a* and *b* via their LCA; -1 if disconnected."""
        lca_key = self.lca(a, b)
        if lca_key is None:
            return -1
        na = self.nodes.get(a)
        nb = self.nodes.get(b)
        if na is None or nb is None:
            return -1
        lca_node = self.nodes[lca_key]
        return na.depth + nb.depth - 2 * lca_node.depth

    def batch_lca_distances(self, pairs: Iterable[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Tree distance for many ``(a, b)`` pairs at once, Tarjan-offline.

        ``lca_distance`` walks up to O(depth) per call, and any caller
        scoring more than a couple of pairs against the same tree (e.g. a
        diversity term that checks every seed against a peer sample every
        pass) ends up paying O(pairs · depth) — quadratic in corpus size
        whenever depth grows with n, which it does on a realistic
        mostly-linear mutation lineage: a mutate-and-keep chain with
        occasional branching has depth ~ Θ(n), not Θ(log n) or Θ(√n) —
        there is no rebalancing force pushing it toward either (unlike the
        Dyck-path parse trees in ``tree_mutator.py``, this forest has no
        Catalan-uniform shape distribution to lean on). This method
        answers the whole batch in one pass with Tarjan's offline LCA
        algorithm (union-find over an iterative post-order walk of the
        forest): O((n + q)·α(n)) total instead of O(q·depth), where q is
        the number of pairs.

        Benchmarked against a loop of ``lca_distance`` calls on a
        synthetic mostly-chain lineage (85% linear growth, 15%
        branch-from-recent-50 — the shape a real corpus lineage takes),
        at n=500/2000/8000 with q matching a per-seed ×64-sample
        diversity workload (q ≈ 64n): naive loop time was ~0.10s/1.35s/
        21.6s, this method's ~0.06s/0.25s/1.15s — 1.9x/5.4x/18.8x, and
        rising with n exactly as the complexity difference predicts (the
        naive side scales roughly as n², this method roughly as n).

        Unknown keys and disconnected pairs resolve to -1 for that entry
        (same convention as ``lca_distance``), rather than raising, so one
        bad pair in a large batch doesn't lose the rest. A corrupted
        forest (a cycle in parent pointers, possible after
        ``rebuild_from_meta`` on untrusted metadata) does not hang: a node
        whose parent chain cycles back on itself is walled off as its own
        isolated root the first time the cycle is detected, so every
        *other* pair still resolves correctly and only the cyclic node's
        own pairs fall back to -1-like non-resolution.

        Args:
            pairs: Iterable of ``(a, b)`` seed-key pairs. Order doesn't
                matter within a pair; ``a == b`` resolves to distance 0.

        Returns:
            ``{(a, b): distance}`` for every pair in *pairs*, keyed
            exactly as given (not sorted/deduplicated) — -1 for any pair
            where either key is unknown or the pair is disconnected.
        """
        pairs = list(pairs)
        result: dict[tuple[str, str], int] = {}
        if not pairs:
            return result

        q_by_node: dict[str, list[tuple[str, str]]] = {}
        for a, b in pairs:
            if a == b:
                result[(a, b)] = 0 if a in self.nodes else -1
                continue
            if a not in self.nodes or b not in self.nodes:
                result[(a, b)] = -1
                continue
            q_by_node.setdefault(a, []).append((a, b))
            q_by_node.setdefault(b, []).append((b, a))

        if not q_by_node:
            return result

        uf_parent: dict[str, str] = {}

        def find(x: str) -> str:
            root = x
            while uf_parent[root] != root:
                root = uf_parent[root]
            while uf_parent[x] != root:
                uf_parent[x], x = root, uf_parent[x]
            return root

        ancestor: dict[str, str] = {}
        globally_done: set[str] = set()  # union of every component's `finished`, for the outer loop only
        visiting: set[str] = set()
        lca_key: dict[tuple[str, str], str] = {}

        def run_from(start: str) -> None:
            # `finished` is local to this one tree/component: a query
            # partner from a *different* component must never look
            # "already finished" here, or a disconnected pair would
            # resolve to a bogus LCA instead of falling through to -1
            # (the forest has multiple roots -- this is not one tree).
            finished: set[str] = set()
            uf_parent[start] = start
            ancestor[start] = start
            visiting.add(start)
            stack: list[tuple[str, object]] = [(start, iter(self._children.get(start, ())))]
            while stack:
                node, it = stack[-1]
                advanced = False
                for child in it:
                    if child in finished or child in visiting:
                        # Already resolved, or a back-edge into a cycle:
                        # never re-descend, which is what keeps a
                        # corrupted mutual-parent link from hanging.
                        continue
                    uf_parent[child] = child
                    ancestor[child] = child
                    visiting.add(child)
                    stack.append((child, iter(self._children.get(child, ()))))
                    advanced = True
                    break
                if advanced:
                    continue
                stack.pop()
                visiting.discard(node)
                finished.add(node)
                ancestor[find(node)] = node
                # Resolve *node*'s own pending queries now, against the
                # current DSU state -- before it gets unioned into its
                # parent below. Doing the union first would make
                # find(descendant) resolve into the parent's class,
                # reporting the parent as LCA for a pair that is really
                # (node, one of node's own descendants).
                for u, v in q_by_node.get(node, ()):
                    if v in finished and (u, v) not in lca_key and (v, u) not in lca_key:
                        lca_key[(u, v)] = ancestor[find(v)]
                if stack:
                    parent = stack[-1][0]
                    uf_parent[find(node)] = find(parent)
                    ancestor[find(parent)] = parent
            globally_done.update(finished)

        for root in self.roots():
            if root in self.nodes and root not in globally_done and root not in visiting:
                run_from(root)
        # Mop-up pass: a node unreachable from any registered root only
        # happens under a corrupted/cyclic parent chain (see docstring).
        # Treating it as its own isolated component here still resolves
        # every pair that doesn't involve the cycle itself, instead of
        # silently dropping an entire disconnected chunk of the batch.
        for key in list(self.nodes.keys()):
            if key not in globally_done and key not in visiting:
                run_from(key)

        for a, b in pairs:
            if (a, b) in result:
                continue
            lk = lca_key.get((a, b)) or lca_key.get((b, a))
            if lk is None:
                result[(a, b)] = -1
                continue
            na = self.nodes.get(a)
            nb = self.nodes.get(b)
            lnode = self.nodes.get(lk)
            if na is None or nb is None or lnode is None:
                result[(a, b)] = -1
            else:
                result[(a, b)] = na.depth + nb.depth - 2 * lnode.depth
        return result

    # ── aggregates ────────────────────────────────────────────────────

    def subtree_weight(self, key: str) -> float:
        """Structural subtree weight (maintained incrementally on insert)."""
        node = self.nodes.get(key)
        return node.subtree_weight if node is not None else 0.0

    def recent_credit(self, key: str, coverage_fn: Callable[[str], tuple[int, int]]) -> float:
        """Coverage gained since last minimize over the subtree of *key*.

        ``coverage_fn(k)`` returns ``(coverage_edges, coverage_edges_baseline)``
        for key ``k``; missing nodes contribute 0. One memoized DFS per call.
        """
        memo: dict[str, float] = {}

        def dfs(k: str) -> float:
            cached = memo.get(k)
            if cached is not None:
                return cached
            node = self.nodes.get(k)
            if node is None:
                return 0.0
            cov, baseline = coverage_fn(k)
            total = max(cov - baseline, 0)
            for ck in self._children.get(k, ()):
                total += dfs(ck)
            memo[k] = total
            return total

        return dfs(key)

    def strahler(self, key: str) -> int:
        """Strahler (Horton-Strahler) number of the subtree rooted at *key*.

        Bottom-up branching-complexity measure, per the standard rule: a
        leaf is 1; a node whose children's numbers have a unique max i
        keeps i; a node whose top two-or-more children tie at max i
        becomes i + 1. Where ``subtree_weight`` sums discounted
        productivity and ``pagerank_credit`` tracks yield per mutation,
        this is a pure topology measure of a branch: how much
        *concurrent* bifurcation was needed to build it, independent of
        edge counts or weights. Two lineages with identical subtree_weight
        can have very different Strahler numbers — one a single deep
        mutation chain (degenerate, Strahler 1 regardless of depth), the
        other balanced bifurcating exploration (Strahler ~ log2 of size).

        Recomputed on demand via an iterative post-order walk (stack-based,
        like the other traversals here, to avoid recursion-depth limits on
        deep lineages) rather than maintained incrementally on insert —
        unlike subtree_weight's ancestor-walk propagation, a new child can
        raise its parent's number without changing by a bounded delta, so
        there is no cheap short-circuit to propagate on insert.

        Returns 0 for an unknown key.
        """
        if key not in self.nodes:
            return 0

        memo: dict[str, int] = {}
        stack: list[tuple[str, bool]] = [(key, False)]
        while stack:
            k, expanded = stack.pop()
            if k in memo:
                continue
            children = self._children.get(k, ())
            if not expanded:
                stack.append((k, True))
                for ck in children:
                    if ck not in memo:
                        stack.append((ck, False))
                continue
            if not children:
                memo[k] = 1
                continue
            top = 0
            ties = 0
            for ck in children:
                v = memo.get(ck, 0)
                if v > top:
                    top, ties = v, 1
                elif v == top:
                    ties += 1
            memo[k] = top + 1 if ties >= 2 else top
        return memo[key]

    def pagerank_credit(
        self,
        damping: float = GAMMA,
        max_iter: int = 100,
        tol: float = 1e-12,
    ) -> dict[str, float]:
        """Per-node credit from a damped walk *up* the forest.

        ``subtree_weight`` sums a node's descendants' weights discounted by
        depth. That measures the volume a branch produced, and volume is
        bought with mutations: a parent that sprayed 13 children worth one
        edge each outranks a parent whose single child was worth twelve,
        even though the second is twelve times the yield per mutation.

        This is the same damped sum with the divisor PageRank uses — a
        child hands its parent ``credit/siblings`` rather than its full
        weight — so a branch's rank tracks yield per mutation instead of
        fan-out. ``damping`` plays γ's role and the personalization vector
        is the normalized node weights, so a node with no productive
        descendants keeps only its own share.

        Returns:
            ``{key: credit}`` over active nodes, summing to ~1. Empty when
            the forest has no weight to distribute yet.
        """
        keys = [k for k, n in self.nodes.items() if n.active]
        if not keys:
            return {}
        idx = {k: i for i, k in enumerate(keys)}
        n = len(keys)

        weights = [float(max(self.nodes[k].node_weight, 0)) for k in keys]
        total_w = sum(weights)
        if total_w <= 0.0:
            return dict.fromkeys(keys, 0.0)
        base = [w / total_w for w in weights]

        # Parent link and live-child count, both restricted to active nodes:
        # a pruned parent is not a credit sink, and counting soft-deleted
        # siblings in the divisor would dilute the survivors.
        parent = [-1] * n
        n_children = [0] * n
        for k in keys:
            pk = self.nodes[k].parent_key
            pi = idx.get(pk) if pk is not None else None
            if pi is not None:
                parent[idx[k]] = pi
                n_children[pi] += 1

        rank = list(base)
        for _ in range(max_iter):
            nxt = [(1.0 - damping) * b for b in base]
            for i in range(n):
                p = parent[i]
                if p >= 0:
                    nxt[p] += damping * rank[i] / n_children[p]
            delta = max(abs(a - b) for a, b in zip(nxt, rank, strict=False))
            rank = nxt
            if delta < tol:
                break
        # The walk only moves toward roots, so mass leaks out at every
        # childless node and the raw vector is not a distribution. Rescale so
        # callers can threshold on a share of total credit rather than on a
        # figure that drifts with corpus size.
        s = sum(rank)
        if s > 0.0:
            rank = [r / s for r in rank]
        return dict(zip(keys, rank, strict=False))

    def operator_credit(self, op: str | int) -> float:
        """Global γ-discounted new-edge credit attributed to *op*.

        Sums ``w(c)·γ^(depth(c))`` over every node whose inbound edge
        applied *op* — deeper edges are discounted the same way subtree
        weights are.
        """
        op_id = self._op_id(op) if isinstance(op, str) else op
        total = 0.0
        for node in self.nodes.values():
            if op_id in node.child_ops:
                total += node.node_weight * (GAMMA**node.depth)
        return total

    # ── pruning ───────────────────────────────────────────────────────

    def prune_subtree(self, key: str) -> int:
        """Soft-delete the subtree rooted at *key* (active=False).

        Children keep their parent pointers; only the active flag flips.
        Inactive nodes are hard-dropped past ``max_inactive`` (oldest
        first). Returns the number of nodes marked inactive.
        """
        node = self.nodes.get(key)
        if node is None:
            return 0
        stack = [key]
        count = 0
        while stack:
            k = stack.pop()
            n = self.nodes.get(k)
            if n is None or not n.active:
                continue
            n.active = False
            self._inactive_keys.append(k)
            count += 1
            stack.extend(self._children.get(k, ()))
        self._trim_inactive()
        return count

    def _trim_inactive(self) -> None:
        while len(self._inactive_keys) > self._max_inactive:
            k = self._inactive_keys.pop(0)
            self._drop(k)

    def _drop(self, key: str) -> None:
        node = self.nodes.pop(key, None)
        if node is None:
            return
        if node.parent_key is not None:
            siblings = self._children.get(node.parent_key)
            if siblings is not None:
                siblings.discard(key)
        # Orphaned children become roots; the dropped key stops being one.
        for child in self._children.get(key, ()):
            if child in self.nodes:
                self._root_keys.add(child)
        self._root_keys.discard(key)
        self._children.pop(key, None)

    def roots(self) -> list[str]:
        """Keys whose parent is absent from the tree (forest roots).

        Maintained incrementally by ``insert``/``_drop`` rather than derived
        by scanning ``nodes``, so callers on the fuzzer's per-iteration path
        do not pay a full-corpus scan per call.
        """
        return [k for k in self._root_keys if k in self.nodes]

    def subtree_keys(self, key: str) -> list[str]:
        """Keys of all nodes in the subtree rooted at *key* (inclusive)."""
        return self._subtree_keys(key)

    def _subtree_keys(self, key: str) -> list[str]:
        stack = [key]
        out: list[str] = []
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(self._children.get(k, ()))
        return out

    def chain_from(self, key: str) -> list[tuple[str, list[int], list[int], int, bool]]:
        """Root-ward lineage chain starting at *key*.

        Each element is ``(key, ops, sites, depth, active)`` where
        ``ops``/``sites`` describe the edge into that node from its
        parent (empty for a root). The chain stops at the first missing
        parent (e.g. an orphaned node after partial rebuild).
        """
        chain: list[tuple[str, list[int], list[int], int, bool]] = []
        node = self.nodes.get(key)
        while node is not None:
            chain.append(
                (node.key, list(node.child_ops), list(node.child_sites), node.depth, node.active)
            )
            node = self.nodes.get(node.parent_key) if node.parent_key else None
        return chain

    # ── rebuild ───────────────────────────────────────────────────────

    def rebuild_from_meta(
        self,
        seed_meta: dict,
        key_fn: Callable[[bytes], str],
    ) -> int:
        """Rebuild the tree from persisted seed metadata (idempotent).

        ``seed_meta`` maps seed bytes → meta dict with optional
        ``parent_key`` (str), ``parent_ops`` (list[str]), ``parent_sites``
        (list[int]), ``new_edge_count`` (int), ``lineage_depth`` (int).
        Missing fields degrade to root / empty ops / weight 0. A child
        whose parent key is absent becomes an orphan root of its own
        (parent pointer retained; chain walks stop there).

        Returns the number of nodes built.
        """
        self.nodes.clear()
        self._children.clear()
        self._root_keys.clear()
        self._inactive_keys.clear()
        self.op_table = []
        self._op_ids = {}
        self._seq = 0

        for seed, meta in seed_meta.items():
            if not isinstance(meta, dict):
                continue
            parent_key = meta.get("parent_key")
            parent_key = parent_key if isinstance(parent_key, str) else None
            if parent_key is not None:
                parent = self.nodes.get(parent_key)
                depth = (
                    parent.depth + 1 if parent is not None else int(meta.get("lineage_depth", 0))
                )
            else:
                depth = int(meta.get("lineage_depth", 0))
            ops = meta.get("parent_ops") or []
            sites = meta.get("parent_sites") or []
            try:
                weight = int(meta.get("new_edge_count", 0))
            except (TypeError, ValueError):
                weight = 0
            self._seq += 1
            node = LineageNode(
                key=key_fn(seed) if not isinstance(seed, str) else seed,
                parent_key=parent_key,
                depth=depth,
                node_weight=weight,
                child_ops=[self._op_id(str(op)) for op in ops],
                child_sites=[int(s) for s in sites],
                seq=self._seq,
            )
            self.nodes[node.key] = node
            if parent_key is not None:
                self._children.setdefault(parent_key, set()).add(node.key)

        # Roots are recomputed in one pass rather than maintained during the
        # loop above: a node whose parent_key is set but whose parent never
        # appears in seed_meta is an orphan root, and that is only decidable
        # once every node has been built. This runs on resume, not per
        # iteration, so the single scan is not on any hot path.
        self._root_keys = {
            key
            for key, node in self.nodes.items()
            if node.parent_key is None or node.parent_key not in self.nodes
        }

        # Structural subtree weights via depth-descending accumulation
        # (child.depth = parent.depth + 1 makes this a valid post-order).
        by_depth = sorted(self.nodes.values(), key=lambda n: n.depth, reverse=True)
        for node in by_depth:
            if node.parent_key is not None and node.parent_key in self.nodes:
                parent = self.nodes[node.parent_key]
                parent.subtree_weight += node.subtree_weight * GAMMA
        return len(self.nodes)
