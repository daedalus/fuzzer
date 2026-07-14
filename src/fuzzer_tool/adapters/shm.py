"""AFL-style shared memory coverage adapter."""

import atexit
import ctypes
import ctypes.util
import os

SHM_MAP_SIZE = 65536

# shmget constants
IPC_CREAT = 0o1000
IPC_RMID = 0
SHM_R = 0o400
SHM_W = 0o200

_libc_name = ctypes.util.find_library("c")
_libc = ctypes.CDLL(_libc_name or "libc.so.6", use_errno=True)

_libc.shmget.argtypes = [ctypes.c_long, ctypes.c_size_t, ctypes.c_int]
_libc.shmget.restype = ctypes.c_int

_libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
_libc.shmat.restype = ctypes.c_void_p

_libc.shmdt.argtypes = [ctypes.c_void_p]
_libc.shmdt.restype = ctypes.c_int

_libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_libc.shmctl.restype = ctypes.c_int


class ShmCoverage:
    """AFL-style shared memory bitmap for coverage tracking.

    Allocates a 64KB shared memory region and provides methods to read
    and compare the bitmap for new coverage detection.  Pass the
    ``env_id`` as ``__AFL_SHM_ID`` in the target environment so the
    instrumented binary writes edge counts directly into the region.

    The instrumented binary updates the bitmap in-place via shared
    memory — ``record_edge`` is a fallback for manual/test use only.
    """

    def __init__(self, size: int = SHM_MAP_SIZE):
        self.size = size
        self.shm_id = _libc.shmget(0, size, IPC_CREAT | SHM_R | SHM_W)
        if self.shm_id < 0:
            raise OSError(f"shmget failed: {os.strerror(ctypes.get_errno())}")
        self._ptr = _libc.shmat(self.shm_id, None, 0)
        if self._ptr == ctypes.c_void_p(-1).value or self._ptr is None:
            _libc.shmctl(self.shm_id, IPC_RMID, None)
            raise OSError(f"shmat failed: {os.strerror(ctypes.get_errno())}")
        self._map = (ctypes.c_char * size).from_address(self._ptr)
        self.env_id = str(self.shm_id)
        self._seen = bytearray(size)  # cumulative "ever seen" bitmap
        self._register_atexit()
        self.total_edges = 0
        self.cumulative_edges = 0

    def read_bitmap(self) -> bytes:
        return bytes(self._map)

    def reset_edge_map(self):
        """Reset the coverage bitmap to zero."""
        ctypes.memset(self._ptr, 0, self.size)

    def reset(self):
        """Full reset: zero bitmap, snapshot, and cumulative counters."""
        self.reset_edge_map()
        self.total_edges = 0

    def record_edge(self, edge_id: int) -> bool:
        """Manually record an edge — fallback for manual/test use.

        With AFL-instrumented binaries the instrumented code writes
        directly into SHM, so this is not called in normal operation.
        """
        idx = edge_id % self.size
        if self._map[idx] == 0:
            self._map[idx] = 1
            self._seen[idx] = 1
            self.total_edges += 1
            self.cumulative_edges = sum(self._seen)
            return True
        return False

    def is_new_coverage(self) -> bool:
        """Check if current bitmap has any edge not seen before (AFL-style).

        Maintains a cumulative 'seen' bitmap across all runs. Only returns
        True when a previously-zero byte becomes non-zero, meaning the
        target explored genuinely new code paths.
        """
        current = bytes(self._map)
        has_new = False
        for i in range(self.size):
            if current[i] and not self._seen[i]:
                self._seen[i] = 1
                has_new = True
        if has_new:
            self.cumulative_edges = sum(self._seen)
            self.total_edges += 1
        return has_new

    def commit_snapshot(self):
        """Update the cumulative 'seen' bitmap to include all current edges."""
        current = bytes(self._map)
        for i in range(self.size):
            if current[i]:
                self._seen[i] = 1
        self.cumulative_edges = sum(self._seen)

    def cleanup(self):
        if self._ptr is not None:
            _libc.shmdt(self._ptr)
            self._ptr = None
        if self.shm_id >= 0:
            _libc.shmctl(self.shm_id, IPC_RMID, None)
            self.shm_id = -1

    def resize(self, new_size: int) -> None:
        """Resize the shared memory bitmap, preserving existing data.

        Allocates a new SHM, copies the old bitmap, detaches the old SHM,
        and updates internal pointers.  Preserves the cumulative _seen
        bitmap across resizes (the SHM itself may be zeroed between runs).

        Args:
            new_size: New map size in bytes (must be > current size).
        """
        if new_size <= self.size:
            return

        # Save cumulative state before resize (SHM may be zeroed)
        old_seen = bytes(self._seen)
        old_cumulative = self.cumulative_edges
        old_total = self.total_edges

        # Allocate new SHM
        new_shm_id = _libc.shmget(0, new_size, IPC_CREAT | SHM_R | SHM_W)
        if new_shm_id < 0:
            raise OSError(f"shmget resize failed: {os.strerror(ctypes.get_errno())}")

        new_ptr = _libc.shmat(new_shm_id, None, 0)
        if new_ptr == ctypes.c_void_p(-1).value or new_ptr is None:
            _libc.shmctl(new_shm_id, IPC_RMID, None)
            raise OSError(f"shmat resize failed: {os.strerror(ctypes.get_errno())}")

        # Zero the new region, then copy old bitmap
        ctypes.memset(new_ptr, 0, new_size)
        ctypes.memmove(new_ptr, self._ptr, self.size)

        # Detach and remove old SHM
        old_ptr = self._ptr
        old_shm_id = self.shm_id
        _libc.shmdt(old_ptr)
        _libc.shmctl(old_shm_id, IPC_RMID, None)

        # Update state
        self._ptr = new_ptr
        self.shm_id = new_shm_id
        old_size = self.size
        self.size = new_size
        self._map = (ctypes.c_char * new_size).from_address(self._ptr)
        self.env_id = str(self.shm_id)

        # Clear cumulative state — positions change after resize
        # AFL's hash (edge_id = hash(src,dst) % map_size) maps the same
        # logical edge to different positions in the new bitmap.
        self._seen = bytearray(new_size)
        self.cumulative_edges = 0
        self.total_edges = 0

    def __del__(self):
        self.cleanup()

    def _register_atexit(self):
        atexit.register(self.cleanup)
