"""Whole-program interprocedural CFG for K-Scheduler (W1).

Lifts distance.py's ``_build_cfgs`` target-function restriction: every
function is decoded, its blocks become nodes, and each resolved direct
call adds a caller→callee-entry edge. Return edges are deliberately
absent — the paper's loop-removal step deletes them anyway
(docs/kscheduler_centrality_port.md W1). ``indirect_call`` blocks carry
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

from fuzzer_tool.core.cfg import FunctionCFG, build_function_cfg
from fuzzer_tool.core.distance import _CALL_RE, _MAX_CFG_FUNC_SIZE

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
    ):
        self.node_addrs = node_addrs
        self.node_funcs = node_funcs
        self.node_index: dict[int, int] = {a: i for i, a in enumerate(node_addrs)}
        self.src = src
        self.dst = dst
        self._cfgs = cfgs

    @property
    def n_nodes(self) -> int:
        return len(self.node_addrs)

    @property
    def n_edges(self) -> int:
        return len(self.src)


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

    edge_set: set[tuple[int, int]] = set()
    for cfg in cfgs.values():
        for blk in cfg.blocks.values():
            u = idx[blk.start]
            for succ in blk.successors:
                v = idx.get(succ)
                if v is not None:
                    edge_set.add((u, v))
            for callee in blk.callees:
                v = entry_of.get(callee)
                # caller→callee only; a resolved callee outside the decoded
                # set (e.g. libc) has no entry node to point at.
                if v is not None and v != u:
                    edge_set.add((u, v))

    src = np.array(sorted(edge_set), dtype=np.int64)[:, 0]
    dst = np.array(sorted(edge_set), dtype=np.int64)[:, 1]
    if src.size == 0:
        src = np.zeros(0, dtype=np.int64)
        dst = np.zeros(0, dtype=np.int64)
    node_funcs = [func_of[a] for a in node_addrs]
    return InterproceduralCFG(node_addrs, node_funcs, src, dst, cfgs)


def probe_key_node_table(td, icfg: InterproceduralCFG) -> dict[int, int]:
    """Runtime probe key → ICFG node index for DistanceTableShm upload.

    Same scan as ``pc_distance_table`` — keys must match what the shim
    computes byte-for-byte — but the value is the node index of the block
    containing the call site instead of an AFLGo distance.
    """
    if td._trace_pc_addr is None:
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
            if start + offset + 5 + disp != td._trace_pc_addr:
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
