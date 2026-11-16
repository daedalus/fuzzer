"""Copy-back of a loaded target's writable globals (``--snapshot-restore``).

wtf restores a whole VM snapshot between testcases. In-process modes have no
process boundary, so ``.data``/``.bss`` written by one input are read by the
next: a counter, a "seen header" flag, a cached pointer. This driver copies
those spans right after load and writes them back before every exec.

    load .so -> capture() ──┐
                            ▼
    exec: restore() -> call target -> restore() -> call target ...

Not restored: heap, TLS, the shim's own sections. A global that points into
heap freed by a later exec reads the freed pointer back after restore.
"""

import ctypes
import logging

from fuzzer_tool.adapters.libc_symbols import module_base
from fuzzer_tool.core.elf import data_ranges

log = logging.getLogger(__name__)

# Exported by every afl_shim.c build: its presence means shim globals exist.
_SHIM_MARK = "__afl_area"


class GlobalSnapshot:
    """Saved bytes of target-owned writable spans, restorable in place."""

    def __init__(self, base: int, spans: tuple[tuple[int, int], ...]):
        # (live byte view, saved bytes): bound once, restore just walks it.
        # Slice assignment is a memcpy without ctypes call overhead: 55 ns vs
        # 640 ns per span for ctypes.memmove, and 2x faster at 1-16 MiB.
        self._plan = tuple(
            (memoryview((ctypes.c_char * size).from_address(base + addr)).cast("B"),
             ctypes.string_at(base + addr, size))
            for addr, size in spans
        )  # fmt: skip
        self.size = sum(size for _, size in spans)

    @classmethod
    def capture(cls, lib: ctypes.CDLL, path: str, entry) -> "GlobalSnapshot | None":
        """Snapshot *lib*'s globals; None when unsafe or unknown.

        *entry* is a function of *lib* (its module base comes from dladdr).
        Refuses a linked shim without isolated sections: its state shares
        .data/.bss and restoring it would clobber the shim.
        """
        with open(path, "rb") as f:
            ranges = data_ranges(f.read())
        base = module_base(ctypes.cast(entry, ctypes.c_void_p).value or 0)
        if ranges is None or not base:
            log.warning("snapshot-restore: cannot map globals of %s, restore off", path)
            return None
        if not ranges.shim_isolated and getattr(lib, _SHIM_MARK, None) is not None:
            log.warning(
                "snapshot-restore: %s predates afl_shim_{data,bss}; rebuild with clang. Restore off.",
                path,
            )
            return None
        return cls(base, ranges.spans)

    def restore(self) -> None:
        """Write the captured bytes back over every span."""
        for live, saved in self._plan:
            live[:] = saved
