"""Exact-size heap placement: the input ends where ASAN's redzone starts.

    ctypes copy, n <= 16    | ctypes object | input |  <- inline, no redzone
    ctypes copy, n <= 512   | pymalloc pool: input | next object |
    HeapInput               | malloc(n): input | redzone |  <- buf[n] reported

``(c_uint8 * n).from_buffer_copy`` keeps small inputs in Python-managed
memory, so an overread of ``buf[n]`` hits a neighbour instead of a redzone.
Above ``_PYMALLOC_MAX`` ctypes already calls libc ``malloc(n)``. AFL++ tip:
never fuzz on a buffer larger than the input.
"""

import ctypes

# CPython's pymalloc serves requests up to this size from its own pools.
_PYMALLOC_MAX = 512

# Global lookup, not libc's handle: with libasan preloaded, malloc resolves
# to ASAN's interceptor (redzoned); find_library("c") would bypass it.
_proc = ctypes.CDLL(None)
_malloc = _proc.malloc
_malloc.restype = ctypes.c_void_p
_malloc.argtypes = [ctypes.c_size_t]
_free = _proc.free
_free.restype = None
_free.argtypes = [ctypes.c_void_p]


class HeapInput:
    """One cached ``malloc(n)`` block per small length; bounded by ``_PYMALLOC_MAX``."""

    def __init__(self) -> None:
        # n -> (array over the block, byte view for the copy).
        self._blocks: dict[int, tuple[ctypes.Array, memoryview]] = {}
        # Last large ctypes-owned copy, reused while its length repeats.
        self._large: ctypes.Array | None = None
        self._large_n = -1

    def place(self, data) -> ctypes.Array:
        """Copy *data* into an exact-size block; reused while the length repeats."""
        n = len(data)
        if n > _PYMALLOC_MAX:
            return self._place_large(data, n)

        block = self._blocks.get(n)
        if block is None:
            block = self._blocks[n] = self._alloc(n)

        # memoryview slice copy: ~4x cheaper than a ctypes.memmove call.
        block[1][:] = data
        return block[0]

    def _place_large(self, data, n: int) -> ctypes.Array:
        """ctypes' own ``malloc(n)`` copy; one slot, so memory stays bounded.

        The slot fills only on a repeated length: holding every large copy
        kept the previous block alive across the next malloc (+9% at 4 KiB).
        """
        if n != self._large_n:
            self._large_n = n
            return (ctypes.c_uint8 * n).from_buffer_copy(data)

        large = self._large
        if large is None or len(large) != n:
            large = self._large = (ctypes.c_uint8 * n).from_buffer_copy(data)
            return large

        # View built on reuse only: random lengths never pay for it.
        memoryview(large).cast("B")[:] = data
        return large

    @staticmethod
    def _alloc(n: int) -> tuple[ctypes.Array, memoryview]:
        ptr = _malloc(n)
        if ptr is None:
            raise MemoryError(f"malloc({n}) failed")
        buf = (ctypes.c_uint8 * n).from_address(ptr)
        return buf, memoryview(buf).cast("B")

    def close(self) -> None:
        """Free every block; idempotent. Arrays from :meth:`place` dangle afterwards."""
        for buf, view in self._blocks.values():
            view.release()
            _free(ctypes.addressof(buf))
        self._blocks.clear()
        self._large = None
        self._large_n = -1
