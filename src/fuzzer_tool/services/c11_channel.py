"""C11 channel: the guard table segment plus its lazy fill.

    open()     allocate C11Shm, export $__AFL_C11_SHM_ID
    measure()  at admission: fill once the pc-table dump exists, then read
               the exec's max locals (0 = unmeasured)

The table needs the pc-table dump, which the shim writes during the first
exec, so it is filled lazily. A target built without pc-table or -g never
measures anything: the channel warns once and stops looking.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import numpy as np

from fuzzer_tool.adapters.sancov_tables import PC_FLAG_ENTRY, SancovTables
from fuzzer_tool.adapters.shm import C11Shm
from fuzzer_tool.core.c11 import guard_table
from fuzzer_tool.core.dwarf import function_local_counts

log = logging.getLogger(__name__)

C11_ENV = "__AFL_C11_SHM_ID"

# Admissions to wait for the pc-table dump. The shim writes it during the
# first exec, so a dump still missing after this many is never coming.
READY_ATTEMPTS = 8

_NO_PC_TABLE = "[!] --c11: no pc-table dump; rebuild with tools/build_targets.sh --sancov=trace-pc-guard,pc-table"
_NO_DWARF = "[!] --c11: no DWARF locals in the target; rebuild with -g"


class C11Channel:
    """Fills the guard table once, then reports each admitted exec's max."""

    def __init__(self, shm, tables, locals_fn: Callable[[], dict[int, int]]) -> None:
        self._shm = shm
        self._tables = tables
        self._locals_fn = locals_fn
        self._attempts = 0
        self._filled = False
        self._dead = False

    @classmethod
    def open(cls, target: str, tables: SancovTables, entries: int) -> C11Channel | None:
        """Allocate the segment and export it to targets spawned after this."""
        try:
            shm = C11Shm(entries)
        except OSError as e:
            log.warning("C11 disabled: %s", e)
            return None

        os.environ[C11_ENV] = shm.env_id
        return cls(shm, tables, lambda: function_local_counts(target))

    def measure(self, gen: int) -> int:
        """Max locals entered by the exec tagged *gen*; 0 when unmeasured."""
        if not self._filled and not self._fill():
            return 0
        return self._shm.read(gen)

    def _fill(self) -> bool:
        if self._dead:
            return False

        self._attempts += 1
        entries, vaddrs = self._entries()
        if entries.size == 0:
            if self._attempts >= READY_ATTEMPTS:
                self._give_up(_NO_PC_TABLE)
            return False

        locals_by_pc = self._locals_fn()
        if not locals_by_pc:
            self._give_up(_NO_DWARF)
            return False

        size = int(entries.max()) + 1
        self._shm.fill(guard_table(entries, vaddrs, locals_by_pc, size))
        self._filled = True
        return True

    def _entries(self) -> tuple[np.ndarray, np.ndarray]:
        # Function-entry guards and their ELF vaddrs, every paired module.
        guards, pcs = [], []
        for rec in self._tables.records():
            idx = rec.entry_indices()
            if not idx.size:
                continue
            is_entry = (rec.pcs_flags() & np.uint64(PC_FLAG_ENTRY)) != 0
            guards.append(idx)
            pcs.append(rec.pcs_vaddrs()[is_entry])

        if not guards:
            return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint64)
        return np.concatenate(guards), np.concatenate(pcs)

    def _give_up(self, why: str) -> None:
        self._dead = True
        print(why)

    def cleanup(self) -> None:
        self._shm.cleanup()
