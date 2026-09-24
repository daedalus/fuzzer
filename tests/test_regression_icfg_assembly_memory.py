"""Regression: ICFG assembly peaked ~900 MB above the decoded CFGs on ffmpeg.

``build_interprocedural_cfg`` built addr->index and addr->func dicts plus sets
of (u, v) tuples over 2.74M blocks. Packed arrays and numpy lookups replace
them; the output must stay identical, pinned against a verbatim copy of the
old assembly.
"""

import tracemalloc

import numpy as np
import pytest

from fuzzer_tool.core import icfg as icfg_mod
from fuzzer_tool.core.cfg import BasicBlock, FunctionCFG

# Assembly allocations beyond the decoded CFGs, on the grid below: the old
# dict/tuple build measured 336 B per block, the packed one 133. Half the old.
_MAX_BYTES_PER_BLOCK = 170
_N_FUNCS = 400
_BLOCKS_PER_FUNC = 100
_STRIDE = 0x10


class _TD:
    _loaded = True


def _reference(cfgs):
    """The pre-change assembly, verbatim, as the oracle (split for CCN)."""
    addrs: set[int] = set()
    func_of: dict[int, str] = {}
    for name, cfg in cfgs.items():
        for bs in cfg.blocks:
            addrs.add(bs)
            func_of[bs] = name
    node_addrs = sorted(addrs)
    idx = {a: i for i, a in enumerate(node_addrs)}
    entry_of = {name: idx[min(cfg.blocks)] for name, cfg in cfgs.items()}
    src, dst, is_call = _reference_edges(cfgs, idx, entry_of)
    node_funcs = [func_of[a] for a in node_addrs]
    return node_addrs, node_funcs, src, dst, is_call


def _reference_edges(cfgs, idx, entry_of):
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
                if v is not None and v != u:
                    call_edges.add((u, v))
    ordered = sorted(branch_edges | call_edges)
    src = np.array([e[0] for e in ordered], dtype=np.int64)
    dst = np.array([e[1] for e in ordered], dtype=np.int64)
    is_call = np.array([e not in branch_edges for e in ordered], dtype=bool)
    return src, dst, is_call


def _build(monkeypatch, cfgs):
    monkeypatch.setattr(icfg_mod, "_decode_all_cfgs", lambda td: cfgs)
    return icfg_mod.build_interprocedural_cfg(_TD())


def _assert_same(got, want):
    node_addrs, node_funcs, src, dst, is_call = want
    assert list(got.node_addrs) == node_addrs
    assert got.node_funcs == node_funcs
    assert np.array_equal(got.src, src) and got.src.dtype == np.int64
    assert np.array_equal(got.dst, dst) and got.dst.dtype == np.int64
    assert np.array_equal(got.is_call, is_call)


def _adversarial():
    """Every edge case the old dicts/sets resolved implicitly."""
    blk = BasicBlock
    return {
        # Entry 0x100; branch to 0x110, dangling succ 0x999 (no node).
        "a": FunctionCFG(
            "a",
            0x100,
            0x130,
            {
                0x100: blk(0x100, 0x110, [0x110, 0x999], {"b", "nosuch"}),
                0x110: blk(0x110, 0x120, [0x200], {"b"}),  # branch AND call to b's entry
                0x120: blk(0x120, 0x130, [], {"a"}),  # call to own entry
            },
        ),
        # 0x200 is b's entry; also a block of "c" below (last writer wins).
        "b": FunctionCFG(
            "b",
            0x200,
            0x220,
            {
                0x200: blk(0x200, 0x210, [0x210], {"b"}),  # self-call: v == u, dropped
                0x210: blk(0x210, 0x220, [0x210]),  # self-loop branch kept
            },
        ),
        "c": FunctionCFG(
            "c",
            0x1F0,
            0x210,
            {
                0x1F0: blk(0x1F0, 0x200, [0x200], {"a", "b"}),
                0x200: blk(0x200, 0x210, []),
            },
        ),
    }


def test_matches_old_assembly(monkeypatch):
    cfgs = _adversarial()
    want = _reference(cfgs)

    # Control (Hard Rule 46): the oracle agrees with itself.
    again = _reference(cfgs)
    assert again[0] == want[0] and again[1] == want[1]
    assert np.array_equal(again[2], want[2]) and np.array_equal(again[4], want[4])

    _assert_same(_build(monkeypatch, cfgs), want)


def _grid():
    cfgs = {}
    for f in range(_N_FUNCS):
        base = 0x100000 + f * _BLOCKS_PER_FUNC * _STRIDE
        blocks = {}
        for b in range(_BLOCKS_PER_FUNC):
            s = base + b * _STRIDE
            succ = [s + _STRIDE] if b + 1 < _BLOCKS_PER_FUNC else []
            callees = {f"f{(f + 1) % _N_FUNCS}"} if b % 10 == 0 else frozenset()
            blocks[s] = BasicBlock(s, s + _STRIDE, succ, callees)
        cfgs[f"f{f}"] = FunctionCFG(f"f{f}", base, base + _BLOCKS_PER_FUNC * _STRIDE, blocks)
    return cfgs


def test_matches_old_on_grid(monkeypatch):
    cfgs = _grid()
    _assert_same(_build(monkeypatch, cfgs), _reference(cfgs))


@pytest.mark.filterwarnings("ignore")
def test_assembly_memory_per_block(monkeypatch):
    """Falsification: assembly peak stays under _MAX_BYTES_PER_BLOCK."""
    cfgs = _grid()
    n_blocks = _N_FUNCS * _BLOCKS_PER_FUNC
    monkeypatch.setattr(icfg_mod, "_decode_all_cfgs", lambda td: cfgs)

    tracemalloc.start()
    try:
        base, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        icfg = icfg_mod.build_interprocedural_cfg(_TD())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert icfg.n_nodes == n_blocks
    assert (peak - base) / n_blocks < _MAX_BYTES_PER_BLOCK, (peak - base) / n_blocks
