"""Regression: the Katz horizon recompute peaked ~3.9 GB on ffmpeg (ASAN).

``build_horizon_graph`` held Python adjacency lists for every ICFG node plus
tuple edge sets, and ``_dag_depth`` a children dict; both now run on CSR
arrays. Output must stay identical, pinned against verbatim copies of the
old code.
"""

import tracemalloc

import numpy as np

from fuzzer_tool.core import horizon as hz
from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import seed_katz

_ADDR_STEP = 0x10
_RANDOM_GRAPHS = 40
_BIG_NODES = 60_000
_LOOP_EVERY = 50
# Old build on the chain+loop graph below measured 733 B/node; the CSR
# build must stay under a third of that.
_MAX_BYTES_PER_NODE = 220


# ── oracle: the pre-change code, verbatim ───────────────────────────────


def _old_tarjan(m, adj):
    index = [-1] * m
    low = [0] * m
    on_stack = [False] * m
    stack = []
    comp = [-1] * m
    counter = ncomp = 0
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
            for i in range(pi, len(adj[v])):
                w = adj[v][i]
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
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
    return comp


def _old_contract(icfg, visited):
    n = icfg.n_nodes
    adj_out = [[] for _ in range(n)]
    adj_in = [[] for _ in range(n)]
    for s, d in zip(icfg.src.tolist(), icfg.dst.tolist(), strict=False):
        adj_out[s].append(d)
        adj_in[d].append(s)
    masks = [hz._unpack_mask(m, n) for m in visited.values()]
    v_mask = np.logical_or.reduce(masks) if masks else np.zeros(n, dtype=bool)
    kept = [i for i in range(n) if not v_mask[i]]
    new_edges = set()
    for u in kept:
        _old_walk(u, adj_out, v_mask, new_edges)
    return adj_in, masks, v_mask, kept, new_edges


def _old_walk(u, adj_out, v_mask, new_edges):
    """Split out of _old_contract for CCN only; body unchanged."""
    seen_v = set()
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


def _old_build(icfg, visited):
    adj_in, masks, v_mask, kept, new_edges = _old_contract(icfg, visited)
    renum = {old: new for new, old in enumerate(kept)}
    dag = [(renum[s], renum[d]) for s, d in sorted(new_edges)]
    cadj = [[] for _ in range(len(kept))]
    for s, d in dag:
        cadj[s].append(d)
    comp = _old_tarjan(len(kept), cadj)
    keep = [(s, d) for s, d in dag if comp[s] != comp[d]]
    parents_of_h, seed_edges = _old_attach(kept, adj_in, v_mask, renum, visited, masks)
    return {
        "src": [s for s, _ in keep],
        "dst": [d for _, d in keep],
        "u_nodes": [icfg.node_addrs[i] for i in kept],
        "u_icfg_index": kept,
        "horizon_set": {icfg.node_addrs[h] for h in parents_of_h},
        "seed_names": list(visited),
        "seed_edges": seed_edges,
        "visited_parents": {renum[h]: vp for h, vp in parents_of_h.items()},
    }


def _old_attach(kept, adj_in, v_mask, renum, visited, masks):
    """Split out of _old_build for CCN only; body unchanged."""
    parents_of_h = {}
    for h in kept:
        vp = {p for p in adj_in[h] if v_mask[p]}
        if vp:
            parents_of_h[h] = vp
    seed_edges = {
        name: {renum[h] for h, vp in parents_of_h.items() if any(mask[p] for p in vp)}
        for name, mask in zip(list(visited), masks, strict=False)
    }
    return parents_of_h, seed_edges


def _old_dag_depth(src, dst, n):
    indeg = np.zeros(n, dtype=np.int64)
    for d in dst.tolist():
        indeg[d] += 1
    children = {}
    for s, d in zip(src.tolist(), dst.tolist(), strict=False):
        children.setdefault(s, []).append(d)
    queue = [v for v in range(n) if indeg[v] == 0]
    order = []
    while queue:
        v = queue.pop()
        order.append(v)
        for w in children.get(v, ()):
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    if len(order) != n:
        return None
    depth = [0] * n
    for u in reversed(order):
        depth[u] = max((depth[v] + 1 for v in children.get(u, ())), default=0)
    return max(depth) if depth else 0


# ── fixtures ─────────────────────────────────────────────────────────────


def _icfg(n, edges):
    src = np.array([e[0] for e in edges], dtype=np.int64)
    dst = np.array([e[1] for e in edges], dtype=np.int64)
    addrs = [0x1000 + _ADDR_STEP * i for i in range(n)]
    return InterproceduralCFG(addrs, ["f"] * n, src, dst, cfgs=None)


def _mask(n, nodes):
    bits = np.zeros(n, dtype=bool)
    bits[list(nodes)] = True
    return np.packbits(bits, bitorder="little").tobytes()


def _random_case(rng):
    n = rng.randint(2, 120)
    edges = sorted({(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 4 * n))})
    visited = {
        f"s{k}": _mask(n, {rng.randrange(n) for _ in range(rng.randint(0, n))})
        for k in range(rng.randint(0, 4))
    }
    return _icfg(n, edges), visited


def _adversarial_case():
    """V-cycle with two exits, U-cycle, self-loops, return through V to u."""
    n = 12
    edges = [
        (0, 1),
        (1, 2),
        (2, 1),
        (2, 3),
        (1, 4),  # u0 -> V{1,2} cycle -> exits 3, 4
        (3, 3),
        (3, 5),
        (5, 6),
        (6, 3),  # U self-loop and U cycle {3,5,6}
        (7, 8),
        (8, 7),  # u7 -> V8 -> back to u7
        (9, 10),
        (10, 11),
        (9, 11),
        (11, 9),  # mixed cycle through V10
    ]
    visited = {"a": _mask(n, {1, 2}), "b": _mask(n, {8, 10}), "c": _mask(n, {2})}
    return _icfg(n, edges), visited


def _assert_same(got, want):
    assert got.src.dtype == np.int64 and got.dst.dtype == np.int64
    assert got.src.tolist() == want["src"]
    assert got.dst.tolist() == want["dst"]
    assert list(got.u_nodes) == want["u_nodes"]
    assert list(got.u_icfg_index) == want["u_icfg_index"]
    assert got.horizon_set == want["horizon_set"]
    assert got.seed_names == want["seed_names"]
    assert got.seed_edges_by_name() == want["seed_edges"]
    assert got.visited_parents == want["visited_parents"]
    assert got.n_u == len(want["u_nodes"])


# ── equivalence ──────────────────────────────────────────────────────────


def test_oracle_control():
    """Hard Rule 46: the oracle agrees with a second run of itself."""
    icfg, visited = _adversarial_case()
    assert _old_build(icfg, visited) == _old_build(icfg, visited)


def test_adversarial_matches_old():
    icfg, visited = _adversarial_case()
    _assert_same(hz.build_horizon_graph(icfg, visited), _old_build(icfg, visited))


def test_random_graphs_match_old():
    rng = RandPool(seed=7)
    for _ in range(_RANDOM_GRAPHS):
        icfg, visited = _random_case(rng)
        _assert_same(hz.build_horizon_graph(icfg, visited), _old_build(icfg, visited))


def test_dag_depth_matches_old():
    """Horizon output is acyclic by construction; depth must match Kahn's."""
    rng = RandPool(seed=11)
    for _ in range(_RANDOM_GRAPHS):
        icfg, visited = _random_case(rng)
        h = hz.build_horizon_graph(icfg, visited)
        want = _old_dag_depth(h.src, h.dst, h.n_u)
        assert seed_katz._dag_depth(h.src, h.dst, h.n_u) == want


def test_node_index_still_available():
    """Adversarial: the addr -> U-index view survives being made lazy."""
    icfg, visited = _adversarial_case()
    h = hz.build_horizon_graph(icfg, visited)

    for i, addr in enumerate(h.u_nodes):
        assert h.node_index[addr] == i


# ── memory ───────────────────────────────────────────────────────────────


def _chain_with_loops(n):
    """ICFG-shaped: a long chain with a back edge every _LOOP_EVERY nodes."""
    src = np.arange(n - 1, dtype=np.int64)
    dst = src + 1
    back_to = np.arange(0, n - _LOOP_EVERY, _LOOP_EVERY, dtype=np.int64)
    src = np.concatenate([src, back_to + _LOOP_EVERY - 1])
    dst = np.concatenate([dst, back_to])
    key = np.unique((src << 32) | dst)
    addrs = [0x1000 + _ADDR_STEP * i for i in range(n)]
    return InterproceduralCFG(addrs, ["f"] * n, key >> 32, key & 0xFFFFFFFF, cfgs=None)


def test_recompute_memory_per_node():
    """Falsification: horizon + Katz peak stays under _MAX_BYTES_PER_NODE."""
    icfg = _chain_with_loops(_BIG_NODES)
    visited = {"s": _mask(_BIG_NODES, range(0, _BIG_NODES, 7))}

    tracemalloc.start()
    try:
        base, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        h = hz.build_horizon_graph(icfg, visited)
        seed_katz.katz_scores(h)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    per_node = (peak - base) / _BIG_NODES
    assert per_node < _MAX_BYTES_PER_NODE, per_node
