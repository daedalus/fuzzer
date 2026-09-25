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

from array import array
from collections.abc import Sequence
from functools import cached_property
from itertools import repeat

import numpy as np

from fuzzer_tool.core.icfg import InterproceduralCFG

# Edge (s, d) packed as s << 32 | d: one int64 that sorts as the tuple did.
_KEY_SHIFT = 32
_KEY_MASK = (1 << _KEY_SHIFT) - 1


class HorizonGraph:
    """Contracted, acyclic unvisited subgraph plus per-seed attachments."""

    def __init__(
        self,
        u_nodes: Sequence[int],
        src: np.ndarray,
        dst: np.ndarray,
        horizon_set: set[int],
        seed_names: list[str],
        seed_edges: dict[str, set[int]],
        u_icfg_index: Sequence[int] | None = None,
        visited_parents: dict[int, set[int]] | None = None,
    ):
        # Packed, 8 B per U node: as lists (plus an eager addr->index dict)
        # these were ~400 MB kept between recomputes on ffmpeg (2.7M nodes).
        self.u_nodes = array("Q", u_nodes)
        self.src = src
        self.dst = dst
        self.horizon_set = horizon_set  # original ICFG addresses
        self.seed_names = seed_names
        self._seed_edges = seed_edges
        # U-index -> original ICFG node index. The two spaces differ as soon
        # as anything is visited, and every array the caller holds (hit
        # counts, distances) is in the ICFG space, so the translation has to
        # be carried on the graph rather than reconstructed by each consumer.
        self.u_icfg_index = array(
            "q", u_icfg_index if u_icfg_index is not None else range(len(u_nodes))
        )
        # U-index -> ICFG indices of its *visited* parents. Empty for U nodes
        # off the horizon. This is what the paper's beta is a function of:
        # R_i counts mutations reaching node i's parents, not node i, which is
        # unvisited by construction and therefore has R_i = 0.
        self.visited_parents: dict[int, set[int]] = visited_parents or {}

    @cached_property
    def node_index(self) -> dict[int, int]:
        """addr -> U index. Built on first use; nothing on the fuzz path reads it."""
        return {a: i for i, a in enumerate(self.u_nodes)}

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


def _tarjan_scc(m: int, ptr: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """Iterative Tarjan over CSR; returns comp id per node.

    Packed state (8 B per node per array, 1 B on-stack flag) instead of a
    Python list per node.
    """
    ptr_a = array("q", ptr.tobytes())
    nbr_a = array("q", nbr.tobytes())
    index = array("q", [-1]) * m
    low = array("q", [0]) * m
    comp = array("q", [-1]) * m
    on_stack = bytearray(m)
    stack = array("q")
    work_v = array("q")
    work_i = array("q")  # resume position in nbr; -1 = first visit
    counter = 0
    ncomp = 0
    for root in range(m):
        if index[root] != -1:
            continue
        work_v.append(root)
        work_i.append(-1)
        while work_v:
            v = work_v.pop()
            i = work_i.pop()
            if i < 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on_stack[v] = 1
                i = ptr_a[v]
            end = ptr_a[v + 1]
            recurse = False
            while i < end:
                w = nbr_a[i]
                i += 1
                if index[w] == -1:
                    work_v.append(v)
                    work_i.append(i)
                    work_v.append(w)
                    work_i.append(-1)
                    recurse = True
                    break
                if on_stack[w] and index[w] < low[v]:
                    low[v] = index[w]
            if recurse:
                continue
            if low[v] == index[v]:
                while True:
                    w = stack.pop()
                    on_stack[w] = 0
                    comp[w] = ncomp
                    if w == v:
                        break
                ncomp += 1
            if work_v and low[v] < low[work_v[-1]]:
                low[work_v[-1]] = low[v]
    return np.frombuffer(comp, dtype=np.int64)


def _gather(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Concatenated index ranges [starts[k], ends[k]), vectorized."""
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.cumsum(counts) - counts
    return np.repeat(starts - offsets, counts) + np.arange(total, dtype=np.int64)


def _out_csr(n: int, src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ptr, nbr): successors of v are nbr[ptr[v]:ptr[v + 1]]."""
    order = np.argsort(src, kind="stable")
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=ptr[1:])
    return ptr, dst[order]


def _peel(n: int, src: np.ndarray, dst: np.ndarray, alive: np.ndarray) -> None:
    """Kill alive nodes with no alive in-edge, cascading along src -> dst.

    A node on a cycle always keeps an alive in-edge, so it is never killed.
    Level-synchronous: one numpy pass per peeled layer, no per-node lists.
    """
    ptr, nbr = _out_csr(n, src, dst)
    live_edge = alive[src] & alive[dst]
    indeg = np.bincount(dst[live_edge], minlength=n)
    frontier = np.flatnonzero(alive & (indeg == 0))
    while frontier.size:
        alive[frontier] = False
        kids = nbr[_gather(ptr[frontier], ptr[frontier + 1])]
        kids = kids[alive[kids]]
        uniq, counts = np.unique(kids, return_counts=True)
        indeg[uniq] -= counts
        frontier = uniq[indeg[uniq] == 0]


def _scc_ids(m: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Component id per node; callers only compare ids across an edge.

    Peeling both ways leaves every node that sits on a cycle (and little
    else); Tarjan runs on that core only. Peeled nodes are singletons.
    """
    alive = np.ones(m, dtype=bool)
    _peel(m, src, dst, alive)
    _peel(m, dst, src, alive)

    comp = np.arange(m, dtype=np.int64)
    core = np.flatnonzero(alive)
    if core.size == 0:
        return comp

    local = np.full(m, -1, dtype=np.int64)
    local[core] = np.arange(core.size)
    inner = alive[src] & alive[dst]
    ptr, nbr = _out_csr(core.size, local[src[inner]], local[dst[inner]])

    # Offset by m: core ids must not collide with the singleton ids.
    comp[core] = m + _tarjan_scc(core.size, ptr, nbr)
    return comp


def _v_successors(src: np.ndarray, dst: np.ndarray, in_v: np.ndarray) -> dict[int, list[int]]:
    """Successor lists for V nodes only; the shortcut walk never leaves V."""
    adj: dict[int, list[int]] = {}
    for s, d in zip(src[in_v].tolist(), dst[in_v].tolist(), strict=True):
        adj.setdefault(s, []).append(d)
    return adj


def _shortcuts(src: np.ndarray, dst: np.ndarray, v_mask: np.ndarray) -> np.ndarray:
    """Packed (u, x) keys: U nodes joined by a path whose interior is all V.

    Walks only from U nodes with a V successor; the old per-U loop also
    visited every other U node and found nothing.
    """
    in_v_src = v_mask[src]
    enter = ~in_v_src & v_mask[dst]
    if not enter.any():
        return np.zeros(0, dtype=np.int64)

    v_adj = _v_successors(src, dst, in_v_src)
    is_v = v_mask.tobytes()
    starts: dict[int, list[int]] = {}
    for u, w in zip(src[enter].tolist(), dst[enter].tolist(), strict=True):
        starts.setdefault(u, []).append(w)

    out_u, out_x = array("q"), array("q")
    for u, stack in starts.items():
        seen: set[int] = set()
        exits: set[int] = set()
        while stack:
            w = stack.pop()
            if w in seen:
                continue
            seen.add(w)
            for x in v_adj.get(w, ()):
                if is_v[x]:
                    stack.append(x)
                else:
                    exits.add(x)

        # A walk back to u is a self-loop; the DAG step drops it anyway.
        exits.discard(u)
        out_u.extend(repeat(u, len(exits)))
        out_x.extend(exits)

    u_arr = np.frombuffer(out_u, dtype=np.int64)
    return (u_arr << _KEY_SHIFT) | np.frombuffer(out_x, dtype=np.int64)


def _contracted_edges(
    src: np.ndarray, dst: np.ndarray, v_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """U -> U edges after deleting V: direct edges plus shortcuts, sorted."""
    direct = ~v_mask[src] & ~v_mask[dst] & (src != dst)
    keys = np.unique(
        np.concatenate([(src[direct] << _KEY_SHIFT) | dst[direct], _shortcuts(src, dst, v_mask)])
    )
    return keys >> _KEY_SHIFT, keys & _KEY_MASK


def _attach(
    icfg: InterproceduralCFG,
    v_mask: np.ndarray,
    renum: np.ndarray,
    visited: dict[str, bytes],
    masks: list[np.ndarray],
) -> tuple[dict[int, set[int]], set[int], dict[str, set[int]]]:
    """Horizon (unvisited nodes with a visited parent) and seed attachment.

    Works on the V -> U boundary edges only, which are few; the old code
    scanned every U node's in-list for every seed.
    """
    src, dst = icfg.src, icfg.dst
    edge = v_mask[src] & ~v_mask[dst]
    parent, child = src[edge], dst[edge]

    visited_parents: dict[int, set[int]] = {}
    for p, h in zip(parent.tolist(), renum[child].tolist(), strict=True):
        visited_parents.setdefault(h, set()).add(p)

    addrs = icfg.node_addrs
    horizon_set = {addrs[h] for h in np.unique(child).tolist()}

    seed_edges = {
        name: set(np.unique(renum[child[mask[parent]]]).tolist())
        for name, mask in zip(list(visited), masks, strict=True)
    }
    return visited_parents, horizon_set, seed_edges


def build_horizon_graph(icfg: InterproceduralCFG, visited: dict[str, bytes]) -> HorizonGraph:
    """Build the edge-horizon graph.

    Arrays throughout: the list-of-lists adjacency this used to build for
    every ICFG node peaked at ~3.9 GB on ffmpeg (2.7M nodes, ASAN).

    Args:
        icfg: whole-program graph.
        visited: {seed name: node bitmap} — NodeBitmapShm layout, one bit
            per ICFG node, sampled every execution and OR-accumulated per
            seed by the caller.
    """
    n = icfg.n_nodes
    src = np.asarray(icfg.src, dtype=np.int64)
    dst = np.asarray(icfg.dst, dtype=np.int64)
    masks = [_unpack_mask(m, n) for m in visited.values()]
    v_mask = np.logical_or.reduce(masks) if masks else np.zeros(n, dtype=bool)

    # U nodes renumbered ascending, so U order matches ICFG address order.
    kept = np.flatnonzero(~v_mask)
    renum = np.full(n, -1, dtype=np.int64)
    renum[kept] = np.arange(kept.size)

    # ── connectivity-preserving deletion of V, then DAG ─────────────
    # renum is monotone, so the (s, d) sort order survives renumbering.
    s_icfg, d_icfg = _contracted_edges(src, dst, v_mask)
    dag_src, dag_dst = renum[s_icfg], renum[d_icfg]
    comp = _scc_ids(kept.size, dag_src, dag_dst)
    across = comp[dag_src] != comp[dag_dst]

    visited_parents, horizon_set, seed_edges = _attach(icfg, v_mask, renum, visited, masks)

    u_nodes = np.frombuffer(icfg.node_addrs, dtype=np.uint64)[kept]
    return HorizonGraph(
        u_nodes=array("Q", u_nodes.tobytes()),
        src=dag_src[across],
        dst=dag_dst[across],
        horizon_set=horizon_set,
        seed_names=list(visited),
        seed_edges=seed_edges,
        u_icfg_index=array("q", kept.tobytes()),
        visited_parents=visited_parents,
    )
