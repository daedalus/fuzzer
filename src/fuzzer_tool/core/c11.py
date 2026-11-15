"""C11 seed weight: locals of the deepest function an input reaches.

AFL++ C11 mode (``afl-c11-pass.so.cc``, ICSE'26 mlsec.org/docs/2026-icse.pdf):
an input that enters a function with many locals reaches more complex code,
so its seed weighs more. AFL++ counts allocas in a compiler pass; here the
count comes from DWARF and reaches the shim as a guard-indexed table:

    DWARF  low_pc -> locals          (core/dwarf.function_local_counts)
    pc-table entry guard -> low_pc   (adapters/sancov_tables)
    guard_table()  guard -> locals   (u8, filled into C11Shm)
    shim   per-exec max over entered functions
    c11_weight()   seed weight factor, AFL++ afl-fuzz-queue.c
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

C11_MAX = 255  # u8 table cell

# AFL++ afl-fuzz-queue.c: weight *= fmaf(0.9, t * t, 1.1).
_W_BASE = 1.1
_W_SPAN = 0.9


def guard_table(
    entries: np.ndarray, vaddrs: np.ndarray, locals_by_pc: dict[int, int], size: int
) -> np.ndarray:
    """``uint8[size]``: function-entry guard -> locals of that function.

    *entries* and *vaddrs* are parallel (pc-table function entries). A pc
    DWARF does not know, or a guard past *size*, leaves the cell 0.
    """
    table = np.zeros(size, dtype=np.uint8)
    if not locals_by_pc:
        return table

    pcs = np.fromiter(locals_by_pc.keys(), dtype=np.uint64, count=len(locals_by_pc))
    counts = np.fromiter(locals_by_pc.values(), dtype=np.int64, count=len(locals_by_pc))
    order = np.argsort(pcs)
    pcs, counts = pcs[order], counts[order]

    # Vectorised lookup: searchsorted, then keep exact pc hits in range.
    pos = np.minimum(np.searchsorted(pcs, vaddrs), pcs.size - 1)
    keep = (pcs[pos] == vaddrs) & (entries < size)
    table[entries[keep]] = np.minimum(counts[pos[keep]], C11_MAX)
    return table


def c11_bounds(values: Iterable[int]) -> tuple[int, int]:
    """(min, max) over measured values; 0 means unmeasured and is skipped."""
    measured = [v for v in values if v]
    if not measured:
        return 0, 0
    return min(measured), max(measured)


def c11_weight(c11: int, lo: int, hi: int) -> float:
    """Seed weight factor in [1.1, 2.0]; 1.0 when unmeasured.

    Example: lo 2, hi 10, c11 6 -> t 0.5 -> 1.1 + 0.9 * 0.25 = 1.325.
    """
    if not c11 or not hi:
        return 1.0

    if lo == hi:
        lo -= 1  # AFL++: a flat range still separates measured seeds
    t = (c11 - lo) / (hi - lo)
    return _W_BASE + _W_SPAN * t * t
