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

NameOf = Callable[[int], "str | None"]


def _parse_blocks(words: list[int], bias: int) -> list[tuple[BasicBlock, list[int]]]:
    """(block, callee vaddrs) in table order; a torn tail is dropped."""
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

    return out


def cfs_functions(words: np.ndarray, bias: int, name_of: NameOf) -> dict[str, FunctionCFG]:
    """Group a control-flow table into named functions.

    Args:
        words: The table as dumped (absolute uint64 values).
        bias: Module load bias; vaddr = value - bias.
        name_of: vaddr -> symbol name, or None (named ``sub_<hex>``).
    """
    parsed = _parse_blocks(words.tolist(), bias)
    reached = {s for blk, _ in parsed for s in blk.successors}

    def name(va: int) -> str:
        return name_of(va) or f"sub_{va:x}"

    fns: dict[str, FunctionCFG] = {}
    cur: FunctionCFG | None = None
    for blk, callees in parsed:
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
