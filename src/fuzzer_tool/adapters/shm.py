"""AFL-style shared memory coverage adapter."""

import atexit
import ctypes
import ctypes.util
import os

from fuzzer_tool.core.count_class import classify_counts, _HAS_NUMPY

if _HAS_NUMPY:
    import numpy as np
    from fuzzer_tool.core.count_class import _NP_CLASSIFY_TABLE

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

# memcmp function for fast comparison
_libc.memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_libc.memcmp.restype = ctypes.c_int


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
        self._seen_classified = bytearray(size)  # cumulative classified "ever seen"
        self._last_map_hash = 0  # cached hash for is_new_coverage fast path
        self._last_map_ptr = ctypes.create_string_buffer(size)  # snapshot for memcmp
        self._register_atexit()
        self.total_edges = 0
        self.cumulative_edges = 0
        # Peak edges ever seen across all bitmap sizes (preserved across resizes)
        self._peak_cumulative_edges: int = 0

    def read_bitmap(self) -> bytes:
        return bytes(self._map)

    def get_edge_bitmap_view(self):
        """Return a zero-copy numpy uint8 view of the raw SHM bitmap.

        Avoids the 1MB ``bytes()`` allocation that ``read_bitmap()`` does.
        Returns ``None`` when numpy is not available.
        """
        if not _HAS_NUMPY:
            return None
        return np.frombuffer(self._map, dtype=np.uint8)

    def reset_edge_map(self):
        """Reset the coverage bitmap to zero."""
        ctypes.memset(self._ptr, 0, self.size)

    def reset(self):
        """Full reset: zero bitmap, snapshot, and cumulative counters."""
        self.reset_edge_map()
        self._seen_classified = bytearray(self.size)
        self.total_edges = 0

    def record_edge(self, edge_id: int) -> bool:
        """Manually record an edge — fallback for manual/test use.

        With AFL-instrumented binaries the instrumented code writes
        directly into SHM, so this is not called in normal operation.
        """
        idx = edge_id % self.size
        if self._map[idx] == b"\x00":
            self._map[idx] = 1
            if not self._seen[idx]:
                self._seen[idx] = 1
                self.cumulative_edges += 1
                self._peak_cumulative_edges = max(
                    self._peak_cumulative_edges, self.cumulative_edges
                )
            self.total_edges += 1
            return True
        return False

    def is_new_coverage(self) -> bool:
        """Check if current bitmap has any edge not seen before (AFL-style).

        Uses a two-tier approach:
        1. Fast path: raw memcmp (zero allocation, single C call) — catches
           the common case where the bitmap is unchanged.
        2. Slow path: classify raw counts into logarithmic buckets, then
           update cumulative seen maps. This filters count-magnitude noise
           (e.g. 47 vs 52 both bucket to 32) while still detecting genuinely
           new edges.
        """
        # Fast path: raw memcmp — no allocation, single C call
        if _libc.memcmp(self._map, self._last_map_ptr, self.size) == 0:
            return False

        # Slow path: classify and scan for new edges
        if _HAS_NUMPY:
            return self._is_new_coverage_numpy()
        return self._is_new_coverage_python()

    def _is_new_coverage_numpy(self) -> bool:
        """Numpy-vectorized coverage scan (~400x faster on 131K buffers)."""
        # Classify: np.take does the entire lookup in C
        raw = np.frombuffer(self._map, dtype=np.uint8)
        classified = _NP_CLASSIFY_TABLE[raw]

        # Find new edges: classified != 0 AND not yet seen
        seen_cls = np.frombuffer(self._seen_classified, dtype=np.uint8)
        mask = (classified != 0) & (seen_cls == 0)
        new_edges = int(np.count_nonzero(mask))

        # Update snapshot for next comparison — zero-copy numpy slice
        # instead of classified.tobytes() + ctypes.memmove (avoid 1MB allocation).
        last_map_view = np.frombuffer(self._last_map_ptr, dtype=np.uint8)
        last_map_view[:] = classified

        if new_edges:
            seen_cls[mask] = classified[mask]
            seen_arr = np.frombuffer(self._seen, dtype=np.uint8)
            seen_arr[mask] = 1
            self.cumulative_edges += new_edges
            self._peak_cumulative_edges = max(self._peak_cumulative_edges, self.cumulative_edges)
            self.total_edges += 1
            return True

        return False

    def _is_new_coverage_python(self) -> bool:
        """Pure-Python fallback when numpy is not available."""
        classified = classify_counts(bytes(self._map))
        has_new = False
        for i in range(self.size):
            if classified[i] and not self._seen_classified[i]:
                self._seen_classified[i] = classified[i]
                self._seen[i] = 1
                self.cumulative_edges += 1
                self._peak_cumulative_edges = max(
                    self._peak_cumulative_edges, self.cumulative_edges
                )
                has_new = True

        ctypes.memmove(self._last_map_ptr, bytes(classified), self.size)
        if has_new:
            self.total_edges += 1
        return has_new

    def commit_snapshot(self):
        """Update the cumulative 'seen' bitmap to include all current edges."""
        if _HAS_NUMPY:
            raw = np.frombuffer(self._map, dtype=np.uint8)
            classified = _NP_CLASSIFY_TABLE[raw]
            seen_cls = np.frombuffer(self._seen_classified, dtype=np.uint8)
            mask = (classified != 0) & (seen_cls == 0)
            new_edges = int(np.count_nonzero(mask))
            if new_edges:
                seen_cls[mask] = classified[mask]
                np.frombuffer(self._seen, dtype=np.uint8)[mask] = 1
                self.cumulative_edges += new_edges
                self._peak_cumulative_edges = max(
                    self._peak_cumulative_edges, self.cumulative_edges
                )
        else:
            classified = classify_counts(bytes(self._map))
            for i in range(self.size):
                if classified[i] and not self._seen_classified[i]:
                    self._seen_classified[i] = classified[i]
                    self._seen[i] = 1
                    self.cumulative_edges += 1
                    self._peak_cumulative_edges = max(
                        self._peak_cumulative_edges, self.cumulative_edges
                    )

    def cleanup(self):
        if self._ptr is not None:
            _libc.shmdt(self._ptr)
            self._ptr = None
        if self.shm_id >= 0:
            _libc.shmctl(self.shm_id, IPC_RMID, None)
            self.shm_id = -1

    def resize(self, new_size: int) -> None:
        """Resize the shared memory bitmap.

        Allocates a new SHM, copies the old bitmap, detaches the old SHM,
        and updates internal pointers.  Clears cumulative state because
        AFL's hash (edge_id = hash(src,dst) % map_size) maps the same
        logical edge to different bitmap positions after resize —
        preserving the old bitmap would silently corrupt it with stale,
        incorrectly-repositioned bits.

        Args:
            new_size: New map size in bytes (must be > current size).
        """
        if new_size <= self.size:
            return

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
        self.size = new_size
        self._map = (ctypes.c_char * new_size).from_address(self._ptr)
        self.env_id = str(self.shm_id)

        # Save peak before clearing — the count of unique positions ever seen
        # is meaningful for the run summary even though position-indexed state
        # must be reset. Without this, a resize near the run end produces
        # "Edges discovered: 0" in the summary.
        self._peak_cumulative_edges = max(self._peak_cumulative_edges, self.cumulative_edges)
        # Clear position-indexed state — positions change after resize.
        # AFL's hash (edge_id = hash(src,dst) % map_size) maps the same
        # logical edge to different positions in the new bitmap.
        self._seen = bytearray(new_size)
        self._seen_classified = bytearray(new_size)
        self.cumulative_edges = 0
        self.total_edges = 0

        # Reallocate snapshot buffer to match new size — without this,
        # is_new_coverage() does memcmp/memmove of new_size bytes into
        # the old (smaller) buffer, causing heap overflow and
        # "free(): invalid pointer" crashes.
        self._last_map_ptr = ctypes.create_string_buffer(new_size)
        self._last_map_hash = 0

    def __del__(self):
        self.cleanup()

    def _register_atexit(self):
        atexit.register(self.cleanup)
