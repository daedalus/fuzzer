"""AFL-style shared memory coverage adapter."""

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
        self._snapshot = bytes(self._map)
        self.total_edges = 0
        self.cumulative_edges = 0

    def read_bitmap(self) -> bytes:
        return bytes(self._map)

    def reset_edge_map(self):
        """Zero the bitmap and snapshot for the next execution."""
        ctypes.memset(self._ptr, 0, self.size)
        self._snapshot = bytes(self._map)

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
            self.total_edges += 1
            self.cumulative_edges += 1
            return True
        return False

    def is_new_coverage(self) -> bool:
        return bytes(self._map) != self._snapshot

    def commit_snapshot(self):
        self._snapshot = bytes(self._map)

    def cleanup(self):
        if self._ptr is not None:
            _libc.shmdt(self._ptr)
            self._ptr = None
        if self.shm_id >= 0:
            _libc.shmctl(self.shm_id, IPC_RMID, None)
            self.shm_id = -1

    def __del__(self):
        self.cleanup()
