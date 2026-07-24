"""AFL-style sparse-entry shared memory for coverage tracking.

Allocates a shared memory region treated as an array of 8-byte entries:
    struct __afl_entry { uint32_t edge_id; uint32_t count; }
where each non-zero edge_id identifies exactly one edge (no hash collisions).

The MAP_SIZE parameter is the number of hash table entries (power of 2),
not the number of bytes.  SHM allocation is map_size * 8 bytes.
"""

import atexit
import ctypes
import ctypes.util
import logging
import os
from typing import NamedTuple

log = logging.getLogger(__name__)

# Default number of hash table entries.
# SHM default = 8192 entries * 8 bytes = 65536 bytes.
SHM_MAP_SIZE = 8192          # number of entries
SIZEOF_ENTRY = 8  # bytes per {edge_id: u32, count: u32}

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

# memcmp for fast comparison of entry arrays
_libc.memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_libc.memcmp.restype = ctypes.c_int


class CoverageEntry(NamedTuple):
    """A single coverage entry read from SHM."""

    edge_id: int
    count: int


def _entry_struct(size: int) -> type[ctypes.Structure]:
    """Create a ctypes Structure representing ``size`` entries."""
    class _AflEntry(ctypes.Structure):
        _fields_ = [
            ("edge_id", ctypes.c_uint32),
            ("count", ctypes.c_uint32),
        ]
    return _AflEntry * size


class ShmCoverage:
    """Sparse-entry SHM for edge coverage tracking.

    Allocates a shared memory segment treated as an array of
    ``{edge_id, count}`` 8-byte entries.  The target binary writes
    entries via open-addressing hashing (linear probing).  The fuzzer
    reads back entries with non-zero edge_id to discover which edges
    were hit.

    ``size`` is the SHM size in bytes (traditional AFL MAP_SIZE).
    The actual number of hash table entries is ``size // 8``.
    """

    def __init__(self, size: int = SHM_MAP_SIZE * SIZEOF_ENTRY):
        # size = SHM bytes (compat with AFL_MAP_SIZE convention)
        # num_entries = size / 8
        self.shm_bytes = size
        self.num_entries = size // SIZEOF_ENTRY

        self.shm_id = _libc.shmget(0, self.shm_bytes, IPC_CREAT | SHM_R | SHM_W)
        if self.shm_id < 0:
            raise OSError(f"shmget failed: {os.strerror(ctypes.get_errno())}")
        self._ptr = _libc.shmat(self.shm_id, None, 0)
        if self._ptr == ctypes.c_void_p(-1).value or self._ptr is None:
            _libc.shmctl(self.shm_id, IPC_RMID, None)
            raise OSError(f"shmat failed: {os.strerror(ctypes.get_errno())}")

        # Raw byte view for memset/memcmp
        self._map = (ctypes.c_char * self.shm_bytes).from_address(self._ptr)
        # Typed struct array view
        EntryArr = _entry_struct(self.num_entries)
        self._entries = EntryArr.from_address(self._ptr)

        self.env_id = str(self.shm_id)

        # Cumulative "ever seen" set of edge_ids (not positions)
        self._seen_edge_ids: set[int] = set()
        # Snapshot for is_new_coverage fast-path (raw byte comparison)
        self._last_map_snapshot = ctypes.create_string_buffer(self.shm_bytes)

        self.total_edges = 0
        self.cumulative_edges = 0
        self._peak_cumulative_edges: int = 0
        self._register_atexit()

    # ── Properties (compat shim) ────────────────────────────────────────
    @property
    def size(self) -> int:
        """Return the SHM size in bytes (compat with AFL_MAP_SIZE convention)."""
        return self.shm_bytes

    # ── Reading ──────────────────────────────────────────────────────────

    def read_bitmap(self) -> bytes:
        """Return the raw SHM byte buffer (all entries as bytes).

        Size = num_entries * 8 bytes.  Callers that need (edge_id, count)
        pairs should use :meth:`read_entries` instead.
        """
        return bytes(self._map)

    def read_entries(self) -> list[CoverageEntry]:
        """Parse SHM and return all non-empty (edge_id, count) pairs."""
        result: list[CoverageEntry] = []
        for i in range(self.num_entries):
            eid = self._entries[i].edge_id
            if eid != 0:
                result.append(CoverageEntry(eid, self._entries[i].count))
        return result

    def get_edge_ids(self) -> set[int]:
        """Return set of non-zero edge_ids currently in the hash table."""
        ids: set[int] = set()
        for i in range(self.num_entries):
            eid = self._entries[i].edge_id
            if eid != 0:
                ids.add(eid)
        return ids

    def get_edge_counts(self) -> dict[int, int]:
        """Return {edge_id: count} for all non-empty entries."""
        counts: dict[int, int] = {}
        for i in range(self.num_entries):
            eid = self._entries[i].edge_id
            if eid != 0:
                counts[eid] = self._entries[i].count
        return counts

    def get_edge_bitmap_view(self):
        """Return a numpy structured array view of entries.

        Returns None when numpy is not available.
        Callers can do:
            arr = shm.get_edge_bitmap_view()
            if arr is not None:
                active = arr[arr['edge_id'] != 0]
                for row in active:  ...
        """
        try:
            import numpy as np
        except ImportError:
            return None
        return np.frombuffer(
            self._map,
            dtype=np.dtype([("edge_id", "<u4"), ("count", "<u4")]),
            count=self.num_entries,
        )

    # ── Reset ────────────────────────────────────────────────────────────

    def reset_edge_map(self):
        """Zero all entries in the coverage hash table."""
        ctypes.memset(self._ptr, 0, self.shm_bytes)

    def reset(self):
        """Full reset: zero entries, clear cumulative state."""
        self.reset_edge_map()
        self._seen_edge_ids.clear()
        self.total_edges = 0

    # ── New-coverage detection ──────────────────────────────────────────

    def is_new_coverage(self) -> bool:
        """Check if the current hash table has any edge not seen before.

        Uses a two-tier approach:
        1. Fast path: raw memcmp (single C call) — catches unchanged state.
        2. Slow path: scan entries for unseen edge_ids.
        """
        if _libc.memcmp(self._map, self._last_map_snapshot, self.shm_bytes) == 0:
            return False

        # Slow path: extract edge_ids not yet in _seen_edge_ids
        new_found = False
        for i in range(self.num_entries):
            eid = self._entries[i].edge_id
            if eid != 0 and eid not in self._seen_edge_ids:
                self._seen_edge_ids.add(eid)
                self.cumulative_edges += 1
                self._peak_cumulative_edges = max(self._peak_cumulative_edges, self.cumulative_edges)
                new_found = True

        # Update snapshot for next comparison
        ctypes.memmove(self._last_map_snapshot, self._map, self.shm_bytes)

        if new_found:
            self.total_edges += 1
        return new_found

    def commit_snapshot(self):
        """Update the cumulative seen-edge set to include all current entries."""
        for i in range(self.num_entries):
            eid = self._entries[i].edge_id
            if eid != 0 and eid not in self._seen_edge_ids:
                self._seen_edge_ids.add(eid)
                self.cumulative_edges += 1
                self._peak_cumulative_edges = max(self._peak_cumulative_edges, self.cumulative_edges)

    # ── Manual recording (for tests) ─────────────────────────────────────

    def record_edge(self, edge_id: int) -> bool:
        """Manually record an edge — for tests only.

        Mirrors what the instrumented binary does: hash to slot, linear probe.
        """
        pos = edge_id % self.num_entries
        for i in range(self.num_entries):
            idx = (pos + i) % self.num_entries
            eid = self._entries[idx].edge_id
            if eid == 0:
                self._entries[idx].edge_id = edge_id
                self._entries[idx].count = 1
                if edge_id not in self._seen_edge_ids:
                    self._seen_edge_ids.add(edge_id)
                    self.cumulative_edges += 1
                    self._peak_cumulative_edges = max(self._peak_cumulative_edges, self.cumulative_edges)
                self.total_edges += 1
                return True
            if eid == edge_id:
                if self._entries[idx].count < 0xFFFFFFFF:
                    self._entries[idx].count += 1
                self.total_edges += 1
                return True
        return False  # table full

    # ── Resize ───────────────────────────────────────────────────────────

    def resize(self, new_num_entries: int) -> None:
        """Resize the hash table (allocates new SHM, copies entries).

        Args:
            new_num_entries: New table size (must be > current).
        """
        new_bytes = new_num_entries * SIZEOF_ENTRY
        if new_bytes <= self.shm_bytes:
            return

        new_shm_id = _libc.shmget(0, new_bytes, IPC_CREAT | SHM_R | SHM_W)
        if new_shm_id < 0:
            raise OSError(f"shmget resize failed: {os.strerror(ctypes.get_errno())}")

        new_ptr = _libc.shmat(new_shm_id, None, 0)
        if new_ptr == ctypes.c_void_p(-1).value or new_ptr is None:
            _libc.shmctl(new_shm_id, IPC_RMID, None)
            raise OSError(f"shmat resize failed: {os.strerror(ctypes.get_errno())}")

        ctypes.memset(new_ptr, 0, new_bytes)
        ctypes.memmove(new_ptr, self._ptr, self.shm_bytes)

        # Detach old SHM
        old_ptr = self._ptr
        old_shm_id = self.shm_id
        _libc.shmdt(old_ptr)
        _libc.shmctl(old_shm_id, IPC_RMID, None)

        self._ptr = new_ptr
        self.shm_id = new_shm_id
        self.num_entries = new_num_entries
        self.shm_bytes = new_bytes
        self._map = (ctypes.c_char * new_bytes).from_address(self._ptr)
        EntryArr = _entry_struct(new_num_entries)
        self._entries = EntryArr.from_address(self._ptr)
        self.env_id = str(self.shm_id)

        self._peak_cumulative_edges = max(self._peak_cumulative_edges, self.cumulative_edges)
        # Clear position-indexed seen set (positions change after resize)
        self._seen_edge_ids.clear()
        self.cumulative_edges = 0
        self.total_edges = 0
        self._last_map_snapshot = ctypes.create_string_buffer(new_bytes)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def cleanup(self):
        if self._ptr is not None:
            _libc.shmdt(self._ptr)
            self._ptr = None
        if self.shm_id >= 0:
            _libc.shmctl(self.shm_id, IPC_RMID, None)
            self.shm_id = -1

    def __del__(self):
        self.cleanup()

    def _register_atexit(self):
        atexit.register(self.cleanup)
