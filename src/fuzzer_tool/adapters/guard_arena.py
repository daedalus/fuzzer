"""Guard-page input placement: the input ends flush against a PROT_NONE page.

    base                      guard_addr
    |  stale  |  input (n)   | PROT_NONE |
                             ^ buf[n] faults, si_addr = guard_addr

A build without ASAN reads past an exact-size heap copy silently; here the
first byte past the input faults. Idea from sandsifter ``injector.c``.
Underreads are not caught: bytes before the input are stale, not guarded.
"""

import ctypes
import ctypes.util
import mmap

#: ``CrashMetadata.error_type`` for a fault inside the guard page.
GUARD_OVERREAD = "guard-overread"

_PAGE = mmap.PAGESIZE
_PROT_NONE = 0
_PROT_RW = mmap.PROT_READ | mmap.PROT_WRITE
_MAP_FLAGS = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
_MAP_FAILED = ctypes.c_void_p(-1).value
# Per-length array cache cap: building the array costs ~300 ns, a hit ~0.
_MAX_CACHED = 1024

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def _pages(n: int) -> int:
    """Data pages for an *n*-byte input (at least one)."""
    return max(1, -(-n // _PAGE))


class GuardArena:
    """One reused mapping: data pages, then one PROT_NONE guard page."""

    def __init__(self) -> None:
        self._base = 0
        self._size = 0
        self._cap = 0
        self._view: memoryview | None = None
        self._bufs: dict[int, ctypes.Array] = {}
        self.guard_addr = 0
        self._map(1)

    def _map(self, pages: int) -> None:
        """Replace the mapping with *pages* data pages + guard."""
        self.close()

        size = (pages + 1) * _PAGE
        base = _libc.mmap(None, size, _PROT_RW, _MAP_FLAGS, -1, 0)
        if base in (None, _MAP_FAILED):
            raise OSError(ctypes.get_errno(), "guard arena mmap failed")

        guard = base + pages * _PAGE
        if _libc.mprotect(guard, _PAGE, _PROT_NONE) != 0:
            _libc.munmap(base, size)
            raise OSError(ctypes.get_errno(), "guard arena mprotect failed")

        self._base, self._size, self._cap = base, size, pages * _PAGE
        self._view = memoryview((ctypes.c_uint8 * self._cap).from_address(base)).cast("B")
        self.guard_addr = guard

    def place(self, data) -> ctypes.Array:
        """Copy *data* so its last byte is the one before the guard.

        Grows to fit (bounded by the largest input seen); never shrinks.
        """
        n = len(data)
        if n > self._cap:
            self._map(_pages(n))

        off = self._cap - n
        self._view[off:] = data

        # One array per length, valid until the next remap.
        buf = self._bufs.get(n)
        if buf is not None:
            return buf
        if len(self._bufs) >= _MAX_CACHED:
            self._bufs.clear()
        buf = self._bufs[n] = (ctypes.c_uint8 * n).from_address(self._base + off)
        return buf

    def owns(self, addr: int | None) -> bool:
        """True when *addr* is inside the guard page (an overread)."""
        g = self.guard_addr
        return bool(g) and addr is not None and g <= addr < g + _PAGE

    def close(self) -> None:
        """Unmap; idempotent. Arrays from :meth:`place` dangle afterwards."""
        if not self._base:
            return
        self._bufs.clear()
        self._view.release()
        _libc.munmap(self._base, self._size)
        self._base = self._size = self._cap = self.guard_addr = 0
        self._view = None
