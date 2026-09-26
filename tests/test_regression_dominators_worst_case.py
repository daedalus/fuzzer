"""Regression: CHK ``compute_idom`` stalled on irreducible CFGs.

Georgiadis 2005 (thesis §2.4) ``idfsquad(k)``: sparse (out-degree <= 2),
irreducible, CHK needs k+1 passes. At 4096 blocks — inside
``_MAX_CFG_BLOCKS`` — CHK took ~40 s; Semi-NCA ~20 ms. See
``docs/learnings/2026-09-26-dominators-chk-worst-case.md``.

Expected idoms are hand-derived or from a brute-force oracle (remove a
block, see what becomes unreachable), never from the code under test.
"""

import time

from fuzzer_tool.core.cfg import BasicBlock, FunctionCFG
from fuzzer_tool.core.dominators import compute_idom
from fuzzer_tool.core.rand_pool import RandPool

ROOT = 0
IDFSQUAD_K = 1365  # 3k + 1 = 4096 blocks, the _MAX_CFG_BLOCKS cap
TIME_BUDGET_S = 2.0  # Semi-NCA ~0.02 s, CHK ~40 s
ORACLE_GRAPHS = 300
ORACLE_MAX_NODES = 25


def _cfg(edges: dict[int, list[int]], entry: int) -> FunctionCFG:
    """FunctionCFG from a successor map; every mentioned address gets a block."""
    addrs = set(edges) | {a for succs in edges.values() for a in succs}
    blocks = {a: BasicBlock(start=a, end=a + 1, successors=list(edges.get(a, []))) for a in addrs}
    blocks[entry].is_entry = True
    return FunctionCFG(name="f", start=min(addrs), end=max(addrs) + 1, blocks=blocks)


def _idfsquad(k: int) -> tuple[dict[int, list[int]], dict[int, int]]:
    """``idfsquad(k)`` plus its hand-derived idom map.

    r -> x1, r -> z1; x_i -> x_{i+1}, x_i -> y_i; y_i <-> z_i; y_i -> z_{i+1}.

        r ── x1 ── x2 ── x3 ...
        │    │     │
        │    y1    y2  ...        y_i reachable via x-chain or via z1,
        └─── z1 ─┘ z2 ─┘          disjoint except at r
    Every y_i, z_i: idom r (two disjoint routes). x1: r; x_i: x_{i-1}.
    """
    x = lambda i: 3 * i - 2  # noqa: E731
    y = lambda i: 3 * i - 1  # noqa: E731
    z = lambda i: 3 * i  # noqa: E731

    edges: dict[int, list[int]] = {ROOT: [x(1), z(1)]}
    for i in range(1, k + 1):
        # Successor order matters: DFS down the x-chain first is what
        # makes CHK's reverse postorder need k+1 passes.
        edges[x(i)] = ([x(i + 1)] if i < k else []) + [y(i)]
        edges[y(i)] = ([z(i + 1)] if i < k else []) + [z(i)]
        edges[z(i)] = [y(i)]

    idom = {ROOT: ROOT, x(1): ROOT}
    for i in range(1, k + 1):
        idom[y(i)] = idom[z(i)] = ROOT
        if i > 1:
            idom[x(i)] = x(i - 1)
    return edges, idom


def _reach(edges: dict[int, list[int]], skip: int | None) -> set[int]:
    """Blocks reachable from ROOT without passing through *skip*."""
    if skip == ROOT:
        return set()
    seen = {ROOT}
    stack = [ROOT]
    while stack:
        for s in edges[stack.pop()]:
            if s != skip and s not in seen:
                seen.add(s)
                stack.append(s)
    return seen


def _brute_idom(edges: dict[int, list[int]]) -> dict[int, int]:
    """idom by definition: a dominates v iff removing a cuts v off."""
    live = _reach(edges, None)
    doms = {v: {v} for v in live}
    for a in live:
        cut = live - _reach(edges, a)
        for v in cut:
            doms[v].add(a)

    # idom = the strict dominator with the most dominators (deepest).
    idom = {ROOT: ROOT}
    for v in live - {ROOT}:
        idom[v] = max(doms[v] - {v}, key=lambda d: len(doms[d]))
    return idom


def test_regression_idfsquad_is_fast():
    """Adversarial: 4096-block irreducible CFG finishes well inside budget."""
    edges, _ = _idfsquad(IDFSQUAD_K)
    cfg = _cfg(edges, ROOT)

    t0 = time.perf_counter()
    compute_idom(cfg)
    assert time.perf_counter() - t0 < TIME_BUDGET_S


def test_idfsquad_idom_hand_derived():
    edges, expected = _idfsquad(8)
    assert compute_idom(_cfg(edges, ROOT)) == expected


def test_brute_oracle_control():
    """Rule 46: the oracle must reproduce a hand-derived answer first."""
    edges, expected = _idfsquad(5)
    assert _brute_idom(edges) == expected


def test_matches_brute_oracle_on_random_graphs():
    """Falsification: seeded random graphs (self-loops, dead blocks, multi-edges)."""
    rng = RandPool(seed=1)
    for _ in range(ORACLE_GRAPHS):
        n = rng.randint(1, ORACLE_MAX_NODES)
        edges = {v: [rng.randrange(n) for _ in range(rng.randint(0, 3))] for v in range(n)}
        cfg = _cfg(edges, ROOT)
        assert compute_idom(cfg, ROOT) == _brute_idom(edges), edges
