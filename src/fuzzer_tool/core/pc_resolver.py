"""Name raw crash PCs from the module's own DWARF line table and symtab.

A target without a sanitizer runtime gets no ``__sanitizer_symbolize_pc``:
the shim records ``module+offset`` only. This resolves those offsets in
process, no external symbolizer::

    frame (module, offset) ──► VA ──┬─► symtab  STT_FUNC  ─► function
                                    └─► .debug_line row   ─► file:line

VA is the offset for PIE / shared objects, the PC for ``ET_EXEC``. Callers'
PCs are return addresses: their call is at VA - 1. Under inlining the
function is the real (outer) frame and file:line the innermost inlined
source line.

Runs once per novel crash. Resolvers are cached per module file, keyed by
(size, mtime) so a rebuilt target is reparsed, and bounded in count.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import NamedTuple

import numpy as np

from fuzzer_tool.core.dwarf import DwarfSiteResolver
from fuzzer_tool.core.elf import FuncTable, detect_elf_type, func_table

_ET_EXEC = 2
_MAX_CACHED = 4  # modules: target + a few of its own .so, never libc
# DWARF walk is pure Python; past this a novel crash would stall the loop.
_MAX_MODULE_BYTES = 64 << 20  # ~20k rows per 2 MB parse in 70 ms (fuzzgoat)


class Site(NamedTuple):
    function: str
    file: str
    line: int


class PcResolver:
    """Offsets of one module → :class:`Site`."""

    def __init__(self, module: str):
        self.module = module
        with open(module, "rb") as fh:
            elf = fh.read()
        self._funcs: FuncTable | None = func_table(elf)
        del elf
        self._fixed = detect_elf_type(module) == _ET_EXEC
        self._lines = DwarfSiteResolver(module)
        self._lines.load()

    def resolve(self, pcs: list[int], offsets: list[int], callers: list[bool]) -> list[Site | None]:
        """One :class:`Site` (or None) per frame; *callers* marks return addresses."""
        base = pcs if self._fixed else offsets
        vas = [va - 1 if caller and va else va for va, caller in zip(base, callers, strict=True)]
        names = self._functions(vas)
        rows = self._lines.sites(vas)

        out: list[Site | None] = []
        for name, start, row in zip(*names, rows, strict=True):
            if not name:
                out.append(None)
                continue

            # A row below the function start belongs to code before it.
            file, line = (row[0], row[1]) if row and row[2] >= start else ("", 0)
            out.append(Site(name, file, line))
        return out

    def _functions(self, vas: list[int]) -> tuple[list[str], list[int]]:
        """Containing function name and start per VA ('' / 0 outside any)."""
        if self._funcs is None or not vas:
            return [""] * len(vas), [0] * len(vas)

        starts, ends, fnames = self._funcs
        q = np.asarray(vas, dtype=np.uint64)
        idx = np.searchsorted(starts, q, side="right") - 1
        names: list[str] = []
        begins: list[int] = []
        for i, va in zip(idx.tolist(), vas, strict=True):
            hit = i >= 0 and va < int(ends[i])
            names.append(fnames[i] if hit else "")
            begins.append(int(starts[i]) if hit else 0)
        return names, begins


_cache: OrderedDict[tuple[str, int, int], PcResolver | None] = OrderedDict()


def resolver_for(module: str) -> PcResolver | None:
    """Cached resolver for *module*; None when unreadable or too large."""
    try:
        st = os.stat(module)
    except OSError:
        return None
    key = (module, st.st_size, st.st_mtime_ns)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    res = None
    if st.st_size <= _MAX_MODULE_BYTES:
        try:
            res = PcResolver(module)
        except (OSError, ValueError):
            res = None
    _cache[key] = res
    while len(_cache) > _MAX_CACHED:
        _cache.popitem(last=False)
    return res
