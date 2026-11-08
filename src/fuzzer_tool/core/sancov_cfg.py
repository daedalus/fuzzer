"""Compiler CFG from ``-fsanitize-coverage=control-flow`` into ``FunctionCFG``.

The table is flat, one entry per block, functions back to back::

    pc, succ..., 0, callee... (-1 = indirect), 0

An IR entry block has no predecessors, so a block no successor list names
starts a new function. Block ends are unknown (the table holds starts
only): ``end == start``, and a function ends one past its last block.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fuzzer_tool.core.cfg import BasicBlock, FunctionCFG

CFS_INDIRECT = (1 << 64) - 1  # callee slot: indirect call
_MASK64 = (1 << 64) - 1
_END = 0

# Below this many words numpy setup costs more than it saves: native
# parse. Crossover ~1000 blocks (4-5 words each); see DEEP_DIVE.
_HYBRID_MIN_WORDS = 4096

NameOf = Callable[[int], "str | None"]
# ((block, callee vaddrs) in table order, vaddrs some successor slot names)
Parsed = tuple[list[tuple[BasicBlock, list[int]]], set[int]]


def _parse_native(arr: np.ndarray, bias: int) -> Parsed:
    """Pure Python; a torn tail is dropped."""
    words = arr.tolist()
    out: list[tuple[BasicBlock, list[int]]] = []
    n = len(words)
    i = 0

    while i < n:
        try:
            s_end = words.index(_END, i + 1)
            c_end = words.index(_END, s_end + 1)
        except ValueError:
            break

        # Wrap, not negate: a callee in another module sits below the bias.
        pc = (words[i] - bias) & _MASK64
        callees = words[s_end + 1 : c_end]
        blk = BasicBlock(start=pc, end=pc)
        blk.successors = [(w - bias) & _MASK64 for w in words[i + 1 : s_end]]
        blk.indirect_call = CFS_INDIRECT in callees
        blk.is_exit = not blk.successors
        out.append((blk, [(w - bias) & _MASK64 for w in callees if w != CFS_INDIRECT]))
        i = c_end + 1

    return out, {s for blk, _ in out for s in blk.successors}


def _bounds(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-block (start, succ end, callee end) word offsets.

    Zeros pair up as terminators only while PCs are nonzero. None when a
    block starts on a zero (invalid table): the caller falls back to native.
    """
    z = np.flatnonzero(arr == _END)
    z = z[: z.size // 2 * 2].reshape(-1, 2)
    s_end, c_end = z[:, 0], z[:, 1]
    start = np.concatenate(([0], c_end[:-1] + 1)) if z.size else s_end

    if start.size and not arr[start].all():
        return None
    return start, s_end, c_end


def _parse_hybrid(arr: np.ndarray, bias: int) -> Parsed:
    """numpy for bounds, bias, indirect flags and successors; Python builds objects."""
    bounds = _bounds(arr)
    if bounds is None:
        return _parse_native(arr, bias)

    start, s_end, c_end = bounds
    nb = start.size
    if not nb:
        return [], set()

    va = arr - np.uint64(bias)  # wraps mod 2^64, as the native mask

    # Indirect markers inside a callee slot only.
    ind = np.flatnonzero(arr == np.uint64(CFS_INDIRECT))
    blk_of = np.searchsorted(c_end, ind)
    ok = blk_of < nb
    ind, blk_of = ind[ok], blk_of[ok]
    has_ind = np.zeros(nb, dtype=bool)
    has_ind[blk_of[ind > s_end[blk_of]]] = True

    # Successor slots, marked by a +1/-1 sweep over [start + 1, s_end).
    sweep = np.zeros(arr.size + 1, dtype=np.int8)
    sweep[start + 1] += 1
    sweep[s_end] -= 1
    in_succ = np.cumsum(sweep[:-1], dtype=np.int8) > 0
    reached = set(va[in_succ].tolist())

    vl = va.tolist()
    ind_va = (CFS_INDIRECT - bias) & _MASK64
    out: list[tuple[BasicBlock, list[int]]] = []
    cols = (start.tolist(), s_end.tolist(), c_end.tolist(), has_ind.tolist())
    for i, s, c, h in zip(*cols, strict=True):
        pc = vl[i]
        blk = BasicBlock(start=pc, end=pc)
        blk.successors = vl[i + 1 : s]
        blk.indirect_call = h
        blk.is_exit = s == i + 1
        callees = [v for v in vl[s + 1 : c] if v != ind_va] if h else vl[s + 1 : c]
        out.append((blk, callees))

    return out, reached


def _group(parsed: Parsed, name_of: NameOf) -> dict[str, FunctionCFG]:
    """Split parsed blocks into named functions at entry blocks."""

    def name(va: int) -> str:
        return name_of(va) or f"sub_{va:x}"

    fns: dict[str, FunctionCFG] = {}
    cur: FunctionCFG | None = None
    blocks, reached = parsed
    for blk, callees in blocks:
        if callees:
            blk.callees = {name(va) for va in callees}

        # New function: nothing branches to an IR entry block.
        if blk.start not in reached:
            blk.is_entry = True
            cur = FunctionCFG(name=name(blk.start), start=blk.start, end=blk.start + 1)
            fns[cur.name] = cur

        if cur is None:
            continue

        cur.blocks[blk.start] = blk
        cur.end = max(cur.end, blk.start + 1)

    return fns


def cfs_functions(words: np.ndarray, bias: int, name_of: NameOf) -> dict[str, FunctionCFG]:
    """Group a control-flow table into named functions.

    Args:
        words: The table as dumped (absolute uint64 values).
        bias: Module load bias; vaddr = value - bias.
        name_of: vaddr -> symbol name, or None (named ``sub_<hex>``).
    """
    parse = _parse_hybrid if words.size >= _HYBRID_MIN_WORDS else _parse_native
    return _group(parse(words, bias), name_of)
