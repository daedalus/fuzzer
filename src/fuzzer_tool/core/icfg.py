"""Whole-program interprocedural CFG for K-Scheduler.

Lifts distance.py's ``_build_cfgs`` target-function restriction: every
function is decoded, its blocks become nodes, and each resolved direct
call adds a caller→callee-entry edge. Return edges are deliberately
absent — the paper's loop-removal step deletes them anyway (She, Shah,
Jana, *Effective Seed Scheduling for Fuzzing with Graph Centrality
Analysis*, S&P'22, arXiv:2203.12064 §4; the DAG conversion that performs
it is ``core/horizon.py``). ``indirect_call`` blocks carry
no static successor; the port surfaces them for a future β penalty.

Also emits the runtime probe-key table that fills the ``node_idx``
column of ``DistanceTableShm``: keys are the exact return addresses the
shim's ``__sanitizer_cov_trace_pc()`` observes (site − base, recovered
by the same REL32 scan as ``TargetDistance.pc_distance_table``), values
are ICFG node indices. A site whose containing block is undecodable is
omitted — the shim's bounds check would reject it anyway.
"""

import logging
import struct

import numpy as np

from fuzzer_tool.core.centrality import betweenness_centrality
from fuzzer_tool.core.cfg import FunctionCFG, build_function_cfg
from fuzzer_tool.core.analyzers.analyzer_distance import _CALL_RE, _MAX_CFG_FUNC_SIZE
from fuzzer_tool.core.mincut import min_cut

log = logging.getLogger(__name__)


class InterproceduralCFG:
    """Node-indexed whole-program ICFG.

    Nodes are basic-block start addresses in sorted order; edges are
    parallel int64 arrays suitable for ``np.bincount``-style SpMV.
    """

    def __init__(
        self,
        node_addrs: list[int],
        node_funcs: list[str],
        src: np.ndarray,
        dst: np.ndarray,
        cfgs: dict[str, FunctionCFG],
        is_call: np.ndarray | None = None,
    ):
        self.node_addrs = node_addrs
        self.node_funcs = node_funcs
        self.node_index: dict[int, int] = {a: i for i, a in enumerate(node_addrs)}
        self.src = src
        self.dst = dst
        self._cfgs = cfgs
        # Parallel bool array, same shape as src/dst: True where the edge is
        # a caller->callee call-graph edge (blk.callees) rather than a real
        # intraprocedural branch/fallthrough successor (blk.successors).
        # Defaults to all-False (every edge treated as a branch edge) for
        # callers -- mostly tests -- that build a purely intraprocedural
        # graph and never populated this distinction.
        self.is_call = (
            np.zeros(len(src), dtype=bool) if is_call is None else is_call
        )

    @property
    def branch_src(self) -> np.ndarray:
        """Source nodes of real conditional-branch/fallthrough edges only.

        Use this (with ``branch_dst``) instead of ``src``/``dst`` for
        questions about program branching -- e.g. "did the corpus split
        this fork both ways?". ``src``/``dst`` mixes those edges with
        caller->callee call-graph edges (see ``is_call``), so an
        out-degree check against them will count an ordinary function
        call as if it were a second branch target -- concretely, a call
        to a helper that's never itself flagged "visited" (it's the
        probe, not a probed site, e.g. ``__sanitizer_cov_trace_pc``)
        looks exactly like an unreached branch sibling. ``bottleneck_edges``
        and ``centrality_scores`` deliberately keep using the full
        ``src``/``dst`` -- interprocedural reachability and whole-program
        centrality are supposed to route through calls.
        """
        return self.src[~self.is_call]

    @property
    def branch_dst(self) -> np.ndarray:
        """See ``branch_src``."""
        return self.dst[~self.is_call]

    @property
    def n_nodes(self) -> int:
        return len(self.node_addrs)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def bottleneck_edges(
        self, hit_addrs: set[int], target_addrs: set[int]
    ) -> set[tuple[int, int]]:
        """Min edge cut (block-address pairs) separating *hit_addrs* from
        *target_addrs* — see ``core/mincut.py`` for the full rationale.

        *hit_addrs*/*target_addrs* are block-start addresses; any address
        this ICFG has no node for is silently ignored (matches
        ``TargetDistance``'s existing tolerance of stale/unmapped
        addresses elsewhere in this module). Returns an empty set if
        either side maps to no nodes at all, or if a node maps to both
        (nothing to separate).
        """
        sources = {self.node_index[a] for a in hit_addrs if a in self.node_index}
        sinks = {self.node_index[a] for a in target_addrs if a in self.node_index}
        sources -= sinks
        if not sources or not sinks:
            return set()
        edges = list(zip(self.src.tolist(), self.dst.tolist()))
        _, cut = min_cut(self.n_nodes, edges, sources, sinks)
        return {(self.node_addrs[u], self.node_addrs[v]) for u, v in cut}

    def centrality_scores(self, normalized: bool = True) -> dict[int, float]:
        """Betweenness centrality of every block, keyed by start address.

        See ``core/centrality.py`` for the algorithm and its relationship
        to dominance/min-cut. Unlike ``bottleneck_edges``, this needs no
        hit-set or target-set -- it is a property of the ICFG's structure
        alone, computed once over the whole graph. Nodes with no shortest
        path running through them (leaves, isolated blocks) are present
        in the result with score 0.0, not omitted.
        """
        edges = list(zip(self.src.tolist(), self.dst.tolist()))
        scores = betweenness_centrality(self.n_nodes, edges, normalized=normalized)
        return {self.node_addrs[i]: s for i, s in enumerate(scores)}


def _decode_all_cfgs(td) -> dict[str, FunctionCFG]:
    """Decode every function the symtab knows, reusing td's cache."""
    cfgs: dict[str, FunctionCFG] = dict(td._cfgs)
    total = 0
    for name, (start, end) in sorted(td.functions.items()):
        if name in cfgs:
            continue
        if end <= start or end - start > _MAX_CFG_FUNC_SIZE:
            continue
        code = td._code_slice(start, end)
        if code is None or len(code) != end - start:
            continue
        try:
            cfg = build_function_cfg(name, code, start, td._resolve_callee_name)
        except Exception:
            log.debug("CFG build failed for %s", name, exc_info=True)
            continue
        if cfg.blocks:
            cfgs[name] = cfg
            total += end - start
    log.info("icfg: %d functions decoded (%d bytes)", len(cfgs), total)
    return cfgs


def build_interprocedural_cfg(td) -> InterproceduralCFG | None:
    """Build the whole-program ICFG from a loaded TargetDistance.

    Returns None when no function could be decoded.
    """
    if not getattr(td, "_loaded", False):
        raise ValueError("call TargetDistance.load() first")
    cfgs = _decode_all_cfgs(td)
    if not cfgs:
        return None

    addrs: set[int] = set()
    func_of: dict[int, str] = {}
    for name, cfg in cfgs.items():
        for bs in cfg.blocks:
            addrs.add(bs)
            func_of[bs] = name
    node_addrs = sorted(addrs)
    idx = {a: i for i, a in enumerate(node_addrs)}
    entry_of = {name: idx[min(cfg.blocks)] for name, cfg in cfgs.items()}

    # Kept separate, not one edge_set: a call edge and a branch edge can
    # land on the same (u, v) pair (rare, but a tail-position call whose
    # fallthrough block starts exactly at the callee is possible in theory),
    # and a real branch must win that tie -- see InterproceduralCFG.is_call.
    branch_edges: set[tuple[int, int]] = set()
    call_edges: set[tuple[int, int]] = set()
    for cfg in cfgs.values():
        for blk in cfg.blocks.values():
            u = idx[blk.start]
            for succ in blk.successors:
                v = idx.get(succ)
                if v is not None:
                    branch_edges.add((u, v))
            for callee in blk.callees:
                v = entry_of.get(callee)
                # caller→callee only; a resolved callee outside the decoded
                # set (e.g. libc) has no entry node to point at.
                if v is not None and v != u:
                    call_edges.add((u, v))

    ordered = sorted(branch_edges | call_edges)
    if ordered:
        src = np.array([e[0] for e in ordered], dtype=np.int64)
        dst = np.array([e[1] for e in ordered], dtype=np.int64)
        is_call = np.array([e not in branch_edges for e in ordered], dtype=bool)
    else:
        src = np.zeros(0, dtype=np.int64)
        dst = np.zeros(0, dtype=np.int64)
        is_call = np.zeros(0, dtype=bool)
    node_funcs = [func_of[a] for a in node_addrs]
    return InterproceduralCFG(node_addrs, node_funcs, src, dst, cfgs, is_call=is_call)


def probe_key_node_table(td, icfg: InterproceduralCFG) -> dict[int, int]:
    """Runtime probe key → ICFG node index for DistanceTableShm upload.

    Same scan as ``pc_distance_table`` — keys must match what the shim
    computes byte-for-byte — but the value is the node index of the block
    containing the call site instead of an AFLGo distance. Matches calls to
    either ``__sanitizer_cov_trace_pc`` or ``__sanitizer_cov_trace_pc_guard``
    (see ``TargetDistance._trace_targets``): the shim probes the distance
    table / node bitmap from both callbacks, so this works whichever
    ``-fsanitize-coverage=`` flavor the target was built with.
    """
    targets = td._trace_targets()
    if not targets:
        return {}
    base = td._base_addr or 0
    table: dict[int, int] = {}
    for name, (start, end) in td.functions.items():
        if end <= start or start < td._text_start or end > td._text_end:
            continue
        code = td._code_slice(start, end)
        if code is None or len(code) != end - start:
            continue
        for m in _CALL_RE.finditer(code):
            offset = m.start()
            if offset + 5 > len(code):
                continue
            disp = struct.unpack_from("<i", code, offset + 1)[0]
            if start + offset + 5 + disp not in targets:
                continue
            site = start + offset + 5  # return address after the call
            cfg = icfg._cfgs.get(name)
            blk = cfg.block_containing(site) if cfg else None
            if blk is None:
                # Tail position can land past the slice's own end.
                alt = td._addr_to_function(site)
                alt_cfg = icfg._cfgs.get(alt) if alt else None
                blk = alt_cfg.block_containing(site) if alt_cfg else None
            if blk is None:
                continue
            nidx = icfg.node_index.get(blk.start)
            if nidx is not None:
                table[site - base] = nidx
    return table
