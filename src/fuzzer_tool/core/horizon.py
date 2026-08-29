"""Edge horizon graph for K-Scheduler.

Paper: She, Shah, Jana — *Effective Seed Scheduling for Fuzzing with
Graph Centrality Analysis*, S&P'22, §4. Given per-seed visited-node sets
over the ICFG:

1. **V / U split.** V = union of seed visits, U = complement.
2. **Connectivity-preserving deletion of V.** Every U→U pair connected
   through a V-only interior gets a shortcut edge (per-U BFS through
   V-interior nodes) — the +24% ablation; skipping it silently collapses
   reachability across covered regions.
3. **DAG conversion.** Iterative Tarjan SCC over the contracted U-graph;
   intra-SCC edges are dropped so any α converges in ≤ depth iterations
   (the Katz solver's convergence argument).
4. **Seed attachment.** One node per seed with edges to every horizon
   node (unvisited, ≥1 visited parent) whose visited parent lies on that
   seed's path.

All indices into ``src``/``dst``/seed-edge sets refer to positions in
``u_nodes`` (ascending original ICFG addresses). That is a *different*
index space from the ICFG's: U is the unvisited complement, so U-index i
is ICFG node ``u_icfg_index[i]``, which is >= i and drifts further apart
the more coverage a campaign has. Any per-node array a caller holds
(hit counts, distances) is in the ICFG space and must be translated
before it is used as a per-U quantity.
"""

import numpy as np

from fuzzer_tool.core.icfg import InterproceduralCFG


class HorizonGraph:
    """Contracted, acyclic unvisited subgraph plus per-seed attachments."""

    def __init__(
        self,
        u_nodes: list[int],
        src: np.ndarray,
        dst: np.ndarray,
        horizon_set: set[int],
        seed_names: list[str],
        seed_edges: dict[str, set[int]],
        u_icfg_index: list[int] | None = None,
        visited_parents: dict[int, set[int]] | None = None,
    ):
        self.u_nodes = u_nodes
        self.node_index: dict[int, int] = {a: i for i, a in enumerate(u_nodes)}
        self.src = src
        self.dst = dst
        self.horizon_set = horizon_set  # original ICFG addresses
        self.seed_names = seed_names
        self._seed_edges = seed_edges
        # U-index -> original ICFG node index. The two spaces differ as soon
        # as anything is visited, and every array the caller holds (hit
        # counts, distances) is in the ICFG space, so the translation has to
        # be carried on the graph rather than reconstructed by each consumer.
        self.u_icfg_index: list[int] = (
            list(u_icfg_index) if u_icfg_index is not None else list(range(len(u_nodes)))
        )
        # U-index -> ICFG indices of its *visited* parents. Empty for U nodes
        # off the horizon. This is what the paper's beta is a function of:
        # R_i counts mutations reaching node i's parents, not node i, which is
        # unvisited by construction and therefore has R_i = 0.
        self.visited_parents: dict[int, set[int]] = visited_parents or {}

    @property
    def n_u(self) -> int:
        return len(self.u_nodes)

    @property
    def n_seed_edges(self) -> int:
        return sum(len(e) for e in self._seed_edges.values())

    def seed_edges_by_name(self) -> dict[str, set[int]]:
        """{seed name: set of u_node indices it attaches to}."""
        return self._seed_edges


def _unpack_mask(mask: bytes, n: int) -> np.ndarray:
    buf = np.frombuffer(mask[: (n + 7) // 8], dtype=np.uint8)
    bits = np.unpackbits(buf, bitorder="little")
    return bits[:n].astype(bool)


def _tarjan_scc(m: int, adj: list[list[int]]) -> list[int]:
    """Iterative Tarjan; returns comp id per node."""
    index = [-1] * m
    low = [0] * m
    on_stack = [False] * m
    stack: list[int] = []
    comp = [-1] * m
    counter = 0
    ncomp = 0
    for root in range(m):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work.pop()
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            neighbors = adj[v]
            for i in range(pi, len(neighbors)):
                w = neighbors[i]
                if index[w] == -1:
                    work.append((v, i + 1))
                    work.append((w, 0))
                    recurse = True
                    break
                if on_stack[w]:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp[w] = ncomp
                    if w == v:
                        break
                ncomp += 1
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return comp


def build_horizon_graph(icfg: InterproceduralCFG, visited: dict[str, bytes]) -> HorizonGraph:
    """Build the edge-horizon graph.

    Args:
        icfg: whole-program graph.
        visited: {seed name: node bitmap} — NodeBitmapShm layout, one bit
            per ICFG node, sampled every execution and OR-accumulated per
            seed by the caller.
    """
    n = icfg.n_nodes
    adj_out: list[list[int]] = [[] for _ in range(n)]
    adj_in: list[list[int]] = [[] for _ in range(n)]
    for s, d in zip(icfg.src.tolist(), icfg.dst.tolist(), strict=False):
        adj_out[s].append(d)
        adj_in[d].append(s)

    masks = [_unpack_mask(m, n) for m in visited.values()]
    v_mask = np.logical_or.reduce(masks) if masks else np.zeros(n, dtype=bool)

    # ── connectivity-preserving deletion of V ────────────────────────
    # For each surviving u, BFS through V-only interiors; every U node
    # reached becomes a shortcut successor. Direct U→U edges carry over.
    kept = [i for i in range(n) if not v_mask[i]]
    new_edges: set[tuple[int, int]] = set()
    for u in kept:
        seen_v: set[int] = set()
        stack = [w for w in adj_out[u] if v_mask[w]]
        while stack:
            w = stack.pop()
            if w in seen_v:
                continue
            seen_v.add(w)
            for x in adj_out[w]:
                if v_mask[x]:
                    stack.append(x)
                else:
                    new_edges.add((u, x))
        for d in adj_out[u]:
            if not v_mask[d] and d != u:
                new_edges.add((u, d))

    # Renumber survivors ascending → u_nodes order matches ICFG address
    # order, so node_index is monotone.
    renum = {old: new for new, old in enumerate(kept)}
    dag_src: list[int] = []
    dag_dst: list[int] = []
    for s, d in sorted(new_edges):
        dag_src.append(renum[s])
        dag_dst.append(renum[d])

    # ── DAG: drop intra-SCC edges ────────────────────────────────────
    m = len(kept)
    cadj: list[list[int]] = [[] for _ in range(m)]
    for s, d in zip(dag_src, dag_dst, strict=False):
        cadj[s].append(d)
    comp = _tarjan_scc(m, cadj)
    keep_pairs = [(s, d) for s, d in zip(dag_src, dag_dst, strict=False) if comp[s] != comp[d]]
    src_arr = (
        np.array([p[0] for p in keep_pairs], dtype=np.int64)
        if keep_pairs
        else np.zeros(0, dtype=np.int64)
    )
    dst_arr = (
        np.array([p[1] for p in keep_pairs], dtype=np.int64)
        if keep_pairs
        else np.zeros(0, dtype=np.int64)
    )

    # ── horizon + seed attachment ────────────────────────────────────
    # H = unvisited nodes with ≥1 visited parent, on the ORIGINAL graph.
    parents_of_h: dict[int, set[int]] = {}
    for hnode in kept:
        vp = {p for p in adj_in[hnode] if v_mask[p]}
        if vp:
            parents_of_h[hnode] = vp
    horizon_addrs = {icfg.node_addrs[h] for h in parents_of_h}

    seed_names = list(visited)
    seed_edges: dict[str, set[int]] = {}
    for name, mask in zip(seed_names, masks, strict=False):
        targets: set[int] = set()
        for hnode, vp in parents_of_h.items():
            if any(mask[p] for p in vp):
                targets.add(renum[hnode])
        seed_edges[name] = targets

    return HorizonGraph(
        u_nodes=[icfg.node_addrs[i] for i in kept],
        src=src_arr,
        dst=dst_arr,
        horizon_set=horizon_addrs,
        seed_names=seed_names,
        seed_edges=seed_edges,
        u_icfg_index=kept,
        visited_parents={renum[h]: vp for h, vp in parents_of_h.items()},
    )
