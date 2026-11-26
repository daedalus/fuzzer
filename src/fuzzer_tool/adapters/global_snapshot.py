"""Copy-back of a loaded target's writable globals (``--snapshot-restore``).

wtf restores a whole VM snapshot between testcases. In-process modes have no
process boundary, so ``.data``/``.bss`` written by one input are read by the
next: a counter, a "seen header" flag, a cached pointer. This driver copies
those spans right after load and writes them back before every exec.

    load .so -> capture() ──┐
                            ▼
    exec: restore() -> call target -> restore() -> call target ...

Not restored: heap, TLS, the shim's own sections, ASAN global redzones. A
global that points into heap freed by a later exec reads the freed pointer
back after restore.

Refused for targets linking a shared type registry (libgobject): a static
cache in .bss (g_once GType) is zeroed, the registry keeps the type, and the
re-registration returns 0. GStreamer 1.28.9: NULL GstMetaInfo -> SIGSEGV.
"""

import ctypes
import logging
from enum import Enum

from fuzzer_tool.adapters.libc_symbols import module_base
from fuzzer_tool.core.elf import data_ranges, needed_libs

log = logging.getLogger(__name__)

# Exported by every afl_shim.c build: its presence means shim globals exist.
_SHIM_MARK = "__afl_area"

# Soname prefixes of shared registries the target caches handles into.
_REGISTRY_LIBS = (b"libgobject-2.0.so",)

_ASAN_GRANULE = 8  # app bytes per ASAN shadow byte


class _AsanShadow:
    """ASAN poison queries, to keep restore off global redzones.

    ASAN pads each global with a poisoned redzone inside .data/.bss. A copy
    over it goes through ASAN's memcpy interceptor and reports an overflow
    (3 symbolized reports per exec on libbase64: 800 -> 130 eps).
    """

    def __init__(self, region, addr):
        self._region = region  # __asan_region_is_poisoned: first bad byte or 0
        self._addr = addr  # __asan_address_is_poisoned: bool

    @classmethod
    def load(cls) -> "_AsanShadow | None":
        """Bind the process's ASAN runtime; None when ASAN is not loaded."""
        proc = ctypes.CDLL(None)
        region = getattr(proc, "__asan_region_is_poisoned", None)
        addr = getattr(proc, "__asan_address_is_poisoned", None)
        if region is None or addr is None:
            return None

        region.restype = ctypes.c_size_t
        region.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        addr.restype = ctypes.c_int
        addr.argtypes = [ctypes.c_size_t]
        return cls(region, addr)

    def addressable(
        self, base: int, spans: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        """Split *spans* (relative to *base*) into their unpoisoned runs.

        e.g. shadow [00 04 f9 f9 00] over 40 bytes -> (0, 12), (32, 8).
        """
        out = []
        for rel, size in spans:
            cur, end = base + rel, base + rel + size
            while cur < end:
                bad = self._region(cur, end - cur)
                if not bad:
                    out.append((cur - base, end - cur))
                    break
                if bad > cur:
                    out.append((cur - base, bad - cur))

                # Rest of bad's granule is poisoned; skip whole redzone granules.
                cur = (bad | (_ASAN_GRANULE - 1)) + 1
                while cur < end and self._addr(cur):
                    cur += _ASAN_GRANULE
        return tuple(out)


class Registry(Enum):
    """Shared-registry policy at capture: refuse (GUARD) or restore anyway (IGNORE)."""

    GUARD = "guard"
    IGNORE = "ignore"


def _registry_lib(needed: tuple[bytes, ...]) -> bytes:
    """First soname in *needed* that is a shared registry; ``b""`` when none."""
    return next((n for n in needed if n.startswith(_REGISTRY_LIBS)), b"")


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
    def capture(
        cls,
        lib: ctypes.CDLL,
        path: str,
        entry,
        level: int = logging.WARNING,
        registry: Registry = Registry.GUARD,
    ) -> "GlobalSnapshot | None":
        """Snapshot *lib*'s globals; None when unsafe or unknown.

        *entry* is a function of *lib* (its module base comes from dladdr).
        Refuses a linked shim without isolated sections: its state shares
        .data/.bss and restoring it would clobber the shim. *level* logs the
        refusal (DEBUG when nobody asked for a restore).
        """
        with open(path, "rb") as f:
            elf = f.read()
        ranges = data_ranges(elf)

        # Restore would desync the target's caches from a shared registry.
        shared = _registry_lib(needed_libs(elf) or ()).decode(errors="replace")
        if shared and registry is Registry.GUARD:
            log.log(
                level,
                "snapshot-restore: %s links shared %s (type registry); restore off.",
                path,
                shared,
            )
            return None
        if shared:
            log.warning(
                "snapshot-restore: %s links shared %s; restoring anyway (globals-unsafe): "
                "crashes may be restore artifacts.",
                path,
                shared,
            )

        base = module_base(ctypes.cast(entry, ctypes.c_void_p).value or 0)
        if ranges is None or not base:
            log.log(level, "snapshot-restore: cannot map globals of %s, restore off", path)
            return None
        if not ranges.shim_isolated and getattr(lib, _SHIM_MARK, None) is not None:
            log.log(
                level,
                "snapshot-restore: %s predates afl_shim_{data,bss}; rebuild with clang. Restore off.",
                path,
            )
            return None

        shadow = _AsanShadow.load()
        spans = shadow.addressable(base, ranges.spans) if shadow else ranges.spans
        return cls(base, spans)

    def restore(self) -> None:
        """Write the captured bytes back over every span."""
        for live, saved in self._plan:
            live[:] = saved
