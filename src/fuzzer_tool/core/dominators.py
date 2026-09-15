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

Algorithm: Cooper, Harvey & Kennedy, "A Simple, Fast Dominance Algorithm"
(2001) — a reverse-postorder dataflow fixed point with a finger-search
intersect, rather than Lengauer-Tarjan's link-eval forest. Near-linear in
practice on CFGs (few predecessors per block); revisit only if profiling
shows this hot on very large functions.

Scope: intra-procedural only, matching ``cfg.py``'s FunctionCFG. Blocks
unreachable from the chosen entry (dead code, or a block only reachable
via an indirect jump the decoder couldn't resolve) have no dominator
relationship and are simply absent from the returned map — callers must
not assume every block in ``cfg.blocks`` has an idom entry.
"""

from __future__ import annotations

from fuzzer_tool.core.cfg import FunctionCFG


def _reverse_postorder(cfg: FunctionCFG, entry: int) -> list[int]:
    """Iterative DFS postorder from *entry* over successors, then reversed.

    Iterative (explicit stack) rather than recursive: function CFGs from
    real binaries can be deep enough to blow the default recursion limit.
    Blocks not reachable from entry (and any successor address that
    didn't survive into ``cfg.blocks`` — a decoder desync gap) are
    silently excluded, matching ``build_function_cfg``'s own successor
    filtering.
    """
    if entry not in cfg.blocks:
        return []
    visited = {entry}
    order: list[int] = []
    stack: list[tuple[int, iter]] = [(entry, iter(cfg.blocks[entry].successors))]
    while stack:
        node, it = stack[-1]
        advanced = False
        for succ in it:
            if succ in cfg.blocks and succ not in visited:
                visited.add(succ)
                stack.append((succ, iter(cfg.blocks[succ].successors)))
                advanced = True
                break
        if not advanced:
            order.append(node)
            stack.pop()
    order.reverse()
    return order


def _predecessors(cfg: FunctionCFG) -> dict[int, list[int]]:
    """Reverse adjacency — FunctionCFG only stores successors."""
    preds: dict[int, list[int]] = {b: [] for b in cfg.blocks}
    for b, blk in cfg.blocks.items():
        for s in blk.successors:
            if s in preds:
                preds[s].append(b)
    return preds


def _intersect(a: int, b: int, idom: dict[int, int], rpo_number: dict[int, int]) -> int:
    """CHK's finger algorithm: walk both chains up to their common ancestor.

    Relies on the invariant that within one fixed-point pass, both *a*
    and *b* already have a (possibly provisional) idom, and that a
    node's reverse-postorder number is always greater than any of its
    dominators' (true for any node processed after its idom in RPO
    order, which the outer loop guarantees).
    """
    while a != b:
        while rpo_number[a] > rpo_number[b]:
            a = idom[a]
        while rpo_number[b] > rpo_number[a]:
            b = idom[b]
    return a


def compute_idom(cfg: FunctionCFG, entry: int | None = None) -> dict[int, int]:
    """Immediate-dominator map for every block reachable from *entry*.

    ``idom[entry] == entry`` by convention (matches the CHK paper — the
    entry is its own dominator, and the sentinel makes ``dominator_chain``
    terminate without a special case). *entry* defaults to the block
    flagged ``is_entry``; if none is flagged (e.g. a hand-built CFG in a
    test) it falls back to the lowest block start.

    Unreachable blocks are absent from the result.
    """
    if entry is None:
        entry = next((b for b, blk in cfg.blocks.items() if blk.is_entry), None)
        if entry is None and cfg.blocks:
            entry = min(cfg.blocks)
    if entry is None or entry not in cfg.blocks:
        return {}

    rpo = _reverse_postorder(cfg, entry)
    if not rpo:
        return {}
    rpo_number = {b: i for i, b in enumerate(rpo)}
    preds = _predecessors(cfg)

    idom: dict[int, int] = {entry: entry}
    changed = True
    while changed:
        changed = False
        for b in rpo:
            if b == entry:
                continue
            processed_preds = [p for p in preds[b] if p in idom]
            if not processed_preds:
                continue
            new_idom = processed_preds[0]
            for p in processed_preds[1:]:
                if p in idom:
                    new_idom = _intersect(new_idom, p, idom, rpo_number)
            if idom.get(b) != new_idom:
                idom[b] = new_idom
                changed = True
    return idom


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
