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

import bisect
import logging
import struct
from array import array
from collections.abc import Sequence
from itertools import repeat

import numpy as np

from fuzzer_tool.core.centrality import betweenness_centrality, closeness_centrality
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
        node_addrs: Sequence[int],
        node_funcs: list[str],
        src: np.ndarray,
        dst: np.ndarray,
        cfgs: dict[str, FunctionCFG] | None,
        is_call: np.ndarray | None = None,
    ):
        # Sorted block starts, 8 B each. A list of ints plus an addr->index
        # dict kept ~1.4 GB of freed block memory pinned on ffmpeg (2.7M
        # nodes); _node_at() bisects instead.
        self.node_addrs = (
            node_addrs
            if isinstance(node_addrs, array) and node_addrs.typecode == "Q"
            else array("Q", node_addrs)
        )
        self.node_funcs = node_funcs
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

    def _node_at(self, addr: int) -> int | None:
        """Index of the node starting exactly at *addr*, else None."""
        addrs = self.node_addrs
        i = bisect.bisect_left(addrs, addr)
        if i < len(addrs) and addrs[i] == addr:
            return i
        return None

    def release_cfgs(self) -> None:
        """Drop per-function CFGs; only probe_key_node_table() reads them.

        2.7M BasicBlock objects on ffmpeg, dead after the build.
        """
        self._cfgs = None

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
        sources = {i for a in hit_addrs if (i := self._node_at(a)) is not None}
        sinks = {i for a in target_addrs if (i := self._node_at(a)) is not None}
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

    def closeness_scores(self) -> dict[int, float]:
        """Out-closeness of every block, keyed by start address.

        High score: the block reaches much of the ICFG in few hops. See
        ``core/centrality.closeness_centrality``.
        """
        edges = list(zip(self.src.tolist(), self.dst.tolist()))
        scores = closeness_centrality(self.n_nodes, edges)
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


# Edge (u, v) packed as u << 32 | v: one int64 that sorts as the tuple did.
# Node counts stay far below 2**31 (ffmpeg: 2.74M).
_KEY_SHIFT = 32
_KEY_MASK = (1 << _KEY_SHIFT) - 1


def _node_table(cfgs: dict[str, FunctionCFG]) -> tuple[np.ndarray, list[str]]:
    """Sorted unique block starts, and the function each belongs to.

    A start claimed by two functions goes to the later one, as the old
    addr->func dict's last write did. Arrays, not dicts: those were ~330 B
    per block of the build peak on ffmpeg.
    """
    names = list(cfgs)
    starts = array("Q")
    owner = array("I")
    for i, cfg in enumerate(cfgs.values()):
        starts.extend(cfg.blocks)
        owner.extend(repeat(i, len(cfg.blocks)))

    flat = np.frombuffer(starts, dtype=np.uint64)

    # unique() keeps each value's first index; on the reversed array that
    # is the last occurrence.
    addrs, first = np.unique(flat[::-1], return_index=True)
    last = len(flat) - 1 - first
    owners = np.frombuffer(owner, dtype=np.uint32)[last]
    # Object-array gather: str refs only, no int object per block.
    return addrs, np.array(names, dtype=object)[owners].tolist()


def _raw_edges(cfgs: dict[str, FunctionCFG]) -> tuple[array, array, array, array]:
    """Address pairs: (block, successor) and (block, callee entry)."""
    entry = {name: min(cfg.blocks) for name, cfg in cfgs.items()}
    bu, bv, cu, cv = array("Q"), array("Q"), array("Q"), array("Q")
    for cfg in cfgs.values():
        for blk in cfg.blocks.values():
            bu.extend(repeat(blk.start, len(blk.successors)))
            bv.extend(blk.successors)
            for callee in blk.callees:
                e = entry.get(callee)
                # caller→callee only; a resolved callee outside the decoded
                # set (e.g. libc) has no entry node to point at.
                if e is None:
                    continue
                cu.append(blk.start)
                cv.append(e)
    return bu, bv, cu, cv


def _lookup(addrs: np.ndarray, query: array) -> tuple[np.ndarray, np.ndarray]:
    """Node index of each queried address, and whether it is a node at all."""
    q = np.frombuffer(query, dtype=np.uint64)
    pos = np.minimum(np.searchsorted(addrs, q), len(addrs) - 1)
    return pos.astype(np.int64, copy=False), addrs[pos] == q


def _edge_keys(addrs: np.ndarray, bu, bv, cu, cv) -> tuple[np.ndarray, np.ndarray]:
    """Unique packed branch and call edge keys.

    Branch successors that are not nodes are dropped; a call from a
    function's own entry block to itself (u == v) is not an edge.
    """
    u, _ = _lookup(addrs, bu)
    v, is_node = _lookup(addrs, bv)
    branch = np.unique((u[is_node] << _KEY_SHIFT) | v[is_node])

    u, _ = _lookup(addrs, cu)
    v, _ = _lookup(addrs, cv)
    distinct = u != v
    call = np.unique((u[distinct] << _KEY_SHIFT) | v[distinct])
    return branch, call


def build_interprocedural_cfg(td) -> InterproceduralCFG | None:
    """Build the whole-program ICFG from a loaded TargetDistance.

    Returns None when no function could be decoded.
    """
    if not getattr(td, "_loaded", False):
        raise ValueError("call TargetDistance.load() first")
    cfgs = _decode_all_cfgs(td)
    if not cfgs:
        return None

    node_addrs, node_funcs = _node_table(cfgs)
    branch, call = _edge_keys(node_addrs, *_raw_edges(cfgs))

    # A call edge and a branch edge can land on the same (u, v) pair (rare,
    # but a tail-position call whose fallthrough block starts exactly at the
    # callee is possible in theory), and a real branch must win that tie --
    # see InterproceduralCFG.is_call. Keys sort as (u, v).
    keys = np.union1d(branch, call)
    src = keys >> _KEY_SHIFT
    dst = keys & _KEY_MASK
    is_call = ~np.isin(keys, branch)

    packed = array("Q")
    packed.frombytes(node_addrs.tobytes())
    return InterproceduralCFG(packed, node_funcs, src, dst, cfgs, is_call=is_call)


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
            nidx = icfg._node_at(blk.start)
            if nidx is not None:
                table[site - base] = nidx
    return table
