"""Dominator-tree computation for FunctionCFG — control-dependence gating.

AFLGo-style distance (``core/distance.py``) scores every block by its
*shortest-path* distance to a target block. That conflates two different
relationships: a block can be BFS-close to a target while sitting on a
branch that never leads there, and a block can be BFS-far while sitting
on the one edge every execution reaching the target must cross.

The dominator tree recovers the second relationship. Block *a* dominates
block *b* if every path from the function's entry to *b* passes through
*a*. The chain of dominators of a target block is exactly the set of
"mandatory gates" for reaching it — the branches worth prioritizing over
and above whatever their raw BFS distance says.

Algorithm: Semi-NCA (Georgiadis, *Linear-Time Algorithms for Dominators
and Related Problems*, PhD thesis 2005, Fig. 2.8): Lengauer-Tarjan
semidominators via path-compressed eval, then idom = nearest common
ancestor of parent and semidominator in the partial dominator tree.
O(m log n) worst case. Replaced Cooper-Harvey-Kennedy's iterative fixed
point, which needs k+1 passes on irreducible CFGs: a 4096-block
``idfsquad`` took ~40 s vs ~20 ms here (see
``docs/learnings/2026-09-26-dominators-chk-worst-case.md``). Target
binaries are attacker-shaped input, so the worst case matters.

Scope: intra-procedural only, matching ``cfg.py``'s FunctionCFG. Blocks
unreachable from the chosen entry (dead code, or a block only reachable
via an indirect jump the decoder couldn't resolve) have no dominator
relationship and are simply absent from the returned map — callers must
not assume every block in ``cfg.blocks`` has an idom entry.
"""

from __future__ import annotations

from array import array

from fuzzer_tool.core.cfg import FunctionCFG

_NO_ANCESTOR = -1  # Semi-NCA forest root marker


def _dfs_preorder(cfg: FunctionCFG, entry: int) -> tuple[list[int], array]:
    """Iterative DFS preorder from *entry*; returns (order, parent).

    ``order[i]`` is the block with preorder number i; ``parent[i]`` is
    the preorder number of its DFS-tree parent (``parent[0] == 0``).
    Iterative (explicit stack): real CFGs can exceed the recursion
    limit. Unreachable blocks and successor addresses missing from
    ``cfg.blocks`` (decoder desync gap) are excluded, matching
    ``build_function_cfg``'s own successor filtering.
    """
    blocks = cfg.blocks
    num = {entry: 0}
    order = [entry]
    parent = array("i", [0])
    stack = [(0, iter(blocks[entry].successors))]

    while stack:
        u, it = stack[-1]
        for s in it:
            if s in num or s not in blocks:
                continue
            num[s] = len(order)
            order.append(s)
            parent.append(u)
            stack.append((num[s], iter(blocks[s].successors)))
            break
        else:
            stack.pop()
    return order, parent


def _compress(v: int, anc: array, label: array) -> None:
    """Path-compress *v*'s ancestor chain, pulling the minimum label down.

    Iterative form of LT's recursive ``compress``: collect the chain up
    to the forest root's child, then fold labels top-down.
    """
    path = []
    while anc[anc[v]] != _NO_ANCESTOR:
        path.append(v)
        v = anc[v]

    for w in reversed(path):
        a = anc[w]
        if label[a] < label[w]:
            label[w] = label[a]
        anc[w] = anc[a]


def _semidominators(pred_nums: list[list[int]], parent: array) -> array:
    """Semidominator preorder number of every vertex (LT step 2).

    Vertices are processed in reverse preorder; a predecessor with a
    smaller number is a tree ancestor-candidate and contributes itself,
    otherwise ``eval`` returns the minimum semi on its linked path.
    """
    n = len(parent)
    semi = array("i", range(n))
    label = array("i", range(n))
    anc = array("i", [_NO_ANCESTOR]) * n

    for w in range(n - 1, 0, -1):
        for v in pred_nums[w]:
            if v > w and anc[v] != _NO_ANCESTOR:
                _compress(v, anc, label)
            u = label[v] if v > w else v
            if u < semi[w]:
                semi[w] = u
        label[w] = semi[w]
        anc[w] = parent[w]
    return semi


def predecessors(cfg: FunctionCFG) -> dict[int, list[int]]:
    """Reverse adjacency — FunctionCFG only stores successors.

    Public: also used by ``core/distance.py`` to BFS the *reversed* CFG
    (walking backward from a target to whatever can reach it), which is
    what an AFLGo-style "distance to target" actually requires — as
    opposed to walking ``successors`` forward from the target, which
    measures the unrelated quantity of what the target can reach.
    """
    preds: dict[int, list[int]] = {b: [] for b in cfg.blocks}
    for b, blk in cfg.blocks.items():
        for s in blk.successors:
            if s in preds:
                preds[s].append(b)
    return preds


# Back-compat alias for the previous private name.
_predecessors = predecessors


def _resolve_entry(cfg: FunctionCFG, entry: int | None) -> int | None:
    """*entry*, else the ``is_entry`` block, else the lowest start; None if absent."""
    if entry is None:
        entry = next((b for b, blk in cfg.blocks.items() if blk.is_entry), None)
    if entry is None and cfg.blocks:
        entry = min(cfg.blocks)
    if entry not in cfg.blocks:
        return None
    return entry


def _pred_numbers(cfg: FunctionCFG, order: list[int], num: dict[int, int]) -> list[list[int]]:
    """Predecessors in preorder numbers; unreachable predecessors dropped."""
    pred_nums: list[list[int]] = [[] for _ in order]
    for u, b in enumerate(order):
        for s in cfg.blocks[b].successors:
            w = num.get(s)
            if w is not None:
                pred_nums[w].append(u)
    return pred_nums


def compute_idom(cfg: FunctionCFG, entry: int | None = None) -> dict[int, int]:
    """Immediate-dominator map for every block reachable from *entry*.

    ``idom[entry] == entry`` by convention (the sentinel makes
    ``dominator_chain`` terminate without a special case). *entry*
    defaults to the block flagged ``is_entry``; if none is flagged (e.g.
    a hand-built CFG in a test) it falls back to the lowest block start.

    Unreachable blocks are absent from the result.
    """
    entry = _resolve_entry(cfg, entry)
    if entry is None:
        return {}

    order, parent = _dfs_preorder(cfg, entry)
    num = {b: i for i, b in enumerate(order)}

    pred_nums = _pred_numbers(cfg, order, num)
    semi = _semidominators(pred_nums, parent)

    # NCA step: climb from the parent until at or above semi(w).
    # idom[] of every vertex numbered < w is already final.
    idom_num = array("i", parent)
    for w in range(1, len(order)):
        x = idom_num[w]
        while x > semi[w]:
            x = idom_num[x]
        idom_num[w] = x

    return {b: order[idom_num[i]] for i, b in enumerate(order)}


def dominates(idom: dict[int, int], a: int, b: int) -> bool:
    """True if block *a* dominates block *b* (every entry->b path passes a).

    Every reachable block dominates itself. Returns False if *b* is not
    in *idom* (unreachable from the entry idom was built with).
    """
    if b not in idom:
        return False
    node = b
    seen: set[int] = set()
    while True:
        if node == a:
            return True
        if node in seen:
            return False  # defensive: idom should never cycle except at the root
        seen.add(node)
        parent = idom[node]
        if parent == node:
            return False  # reached the root without seeing a
        node = parent


def dominator_chain(idom: dict[int, int], b: int) -> list[int]:
    """Blocks from *b* up to and including the entry, nearest first.

    Empty if *b* is unreachable from the entry idom was built with.
    """
    if b not in idom:
        return []
    chain = [b]
    node = b
    while idom[node] != node:
        node = idom[node]
        chain.append(node)
    return chain


def gate_blocks(cfg: FunctionCFG, targets: set[int], entry: int | None = None) -> set[int]:
    """Union of proper-dominator-chain blocks for every block in *targets*.

    These are the blocks every execution reaching *any* target in this
    function must pass through — mandatory control-flow gates, as
    opposed to blocks that are merely BFS-nearby without being on every
    path. Target blocks themselves are excluded (a target trivially
    "gates" itself; callers already special-case target blocks with
    distance 0 and don't need them relisted here).
    """
    idom = compute_idom(cfg, entry)
    gates: set[int] = set()
    for t in targets:
        for node in dominator_chain(idom, t):
            if node != t:
                gates.add(node)
    return gates
