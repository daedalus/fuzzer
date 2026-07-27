"""AFL-style sparse-entry shared memory for coverage tracking.

Allocates a shared memory region treated as an array of 8-byte entries:
    struct __afl_entry { uint32_t edge_id; uint32_t count; }
where each non-zero edge_id identifies exactly one edge (no hash collisions).

The MAP_SIZE parameter is the number of hash table entries (power of 2),
not the number of bytes.  SHM layout: 24-byte front header + map_size * 8 bytes.
"""

import atexit
import ctypes
import ctypes.util
import logging
import os

log = logging.getLogger(__name__)

# Default number of hash table entries.
# SHM default = 8192 entries * 8 bytes = 65536 bytes.
SHM_MAP_SIZE = 8192  # number of entries
SIZEOF_ENTRY = 8  # bytes per {edge_id: u32, count: u32}

# Metadata region in the front header of SHM (before the edge table).
# Layout (24 bytes total):
#   offset 0: uint32 stack_depth   (max stack depth in bytes, from __sancov_lowest_stack)
#   offset 4: uint32 _pad0
#   offset 8: uint64 path_hash     (rolling hash: hash = hash * 31 ^ edge_id)
#   offset 16: uint64 edge_count   (monotonic new-slot insertion count)
SHM_METADATA_SIZE = 24  # bytes reserved at front of SHM for metadata

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

    ``size`` is the number of hash table entries (AFL_MAP_SIZE convention).
    SHM layout: front header (SHM_METADATA_SIZE bytes) + edge table (size * 8 bytes).
    """

    def __init__(self, size: int = SHM_MAP_SIZE):
        # size = number of entries (AFL_MAP_SIZE convention)
        self.num_entries = size
        self.table_bytes = size * SIZEOF_ENTRY
        self.shm_bytes = self.table_bytes + SHM_METADATA_SIZE

        self.shm_id = _libc.shmget(0, self.shm_bytes, IPC_CREAT | SHM_R | SHM_W)
        if self.shm_id < 0:
            raise OSError(f"shmget failed: {os.strerror(ctypes.get_errno())}")
        self._ptr = _libc.shmat(self.shm_id, None, 0)
        if self._ptr == ctypes.c_void_p(-1).value or self._ptr is None:
            _libc.shmctl(self.shm_id, IPC_RMID, None)
            raise OSError(f"shmat failed: {os.strerror(ctypes.get_errno())}")

        # Raw byte view for the edge table only (starts after front header)
        self._map = (ctypes.c_char * self.table_bytes).from_address(self._ptr + SHM_METADATA_SIZE)
        # Typed struct array view (starts after front header)
        EntryArr = _entry_struct(self.num_entries)
        self._entries = EntryArr.from_address(self._ptr + SHM_METADATA_SIZE)

        # Metadata is in the front header at self._ptr (offsets 0/8/16)

        self.env_id = str(self.shm_id)

        # Cumulative "ever seen" set of edge_ids (not positions)
        self._seen_edge_ids: set[int] = set()
        # Last seen edge_count for O(1) fast-path in is_new_coverage
        self._last_edge_count: int = 0
        # Last seen path_hash for fast-path — catches same-count but different-edge sets
        self._last_path_hash: int = 0

        self.total_edges = 0
        self.cumulative_edges = 0
        self._peak_cumulative_edges: int = 0
        self._register_atexit()

    # ── Properties (compat shim) ────────────────────────────────────────
    @property
    def size(self) -> int:
        """Return the number of hash table entries (AFL_MAP_SIZE convention)."""
        return self.num_entries

    # ── Reading ──────────────────────────────────────────────────────────

    def get_edge_ids(self) -> set[int]:
        """Return set of non-zero edge_ids currently in the hash table.

        Uses numpy for zero-copy vectorized scan — avoids per-entry
        Python loop overhead.
        """
        import numpy as np

        arr = np.frombuffer(
            self._map,
            dtype=np.dtype([("edge_id", "<u4"), ("count", "<u4")]),
            count=self.num_entries,
        )
        active = arr[arr["edge_id"] != 0]
        # .tolist() converts numpy uint32 to plain Python ints
        return set(active["edge_id"].tolist())

    def get_edge_counts(self) -> dict[int, int]:
        """Return {edge_id: count} for all non-empty entries."""
        import numpy as np

        arr = np.frombuffer(
            self._map,
            dtype=np.dtype([("edge_id", "<u4"), ("count", "<u4")]),
            count=self.num_entries,
        )
        active = arr[arr["edge_id"] != 0]
        # .tolist() converts numpy uint32 to plain Python ints
        return dict(zip(active["edge_id"].tolist(), active["count"].tolist()))

    # ── Reset ────────────────────────────────────────────────────────────

    def reset_edge_map(self):
        """Zero all entries in the coverage hash table (preserves front header)."""
        ctypes.memset(self._ptr + SHM_METADATA_SIZE, 0, self.table_bytes)

    # ── Metadata (stack depth + path hash + edge count) ────────────────

    def read_stack_depth(self) -> int:
        """Read the stack depth value from the SHM front header (offset 0).

        The C shim writes the max stack depth (in bytes) here when
        __sancov_lowest_stack is available. Returns 0 when unavailable.
        """
        return ctypes.c_uint32.from_address(self._ptr).value

    def read_path_hash(self) -> int:
        """Read the rolling path hash from the SHM front header (offset 8).

        The C shim maintains: path_hash = path_hash * 31 ^ edge_id
        Returns the 64-bit hash value.
        """
        return ctypes.c_uint64.from_address(self._ptr + 8).value

    def read_edge_count(self) -> int:
        """Read the monotonic edge count from the SHM front header (offset 16).

        The C shim increments this counter on each new-slot insertion.
        A changed value guarantees at least one new distinct edge was recorded
        this iteration.  Returns 0 when no edges were recorded.
        """
        return ctypes.c_uint64.from_address(self._ptr + 16).value

    def read_metadata(self) -> tuple[int, int, int]:
        """Read all metadata from the front header.

        Returns:
            (stack_depth, path_hash, edge_count) tuple.
        """
        return self.read_stack_depth(), self.read_path_hash(), self.read_edge_count()

    def compute_path_hash_from_edges(self, edge_ids: set[int]) -> int:
        """Compute a rolling path hash from edge IDs (Python fallback).

        Uses the same formula as honggfuzz: hash = hash * 31 ^ edge_id
        for deterministic ordering via sorted edge IDs.

        Args:
            edge_ids: Set of edge IDs from the current iteration.

        Returns:
            64-bit path hash.
        """
        h = 0
        for eid in sorted(edge_ids):
            h = (h * 31) ^ eid
        return h & 0xFFFFFFFFFFFFFFFF

    # ── New-coverage detection ──────────────────────────────────────────

    def _check_new_coverage(self) -> tuple[bool, set[int]]:
        """Check for new coverage and return (has_new, current_edge_ids).

        Two-tier approach:
        1. Fast path: compare edge_count AND path_hash against last seen
           (both O(1) header reads) — avoids touching the edge table when
           nothing changed.  The path_hash comparison catches the case where
           the same number of edges fire but the edge SET has changed (e.g.
           {1,2,3} vs {1,2,4} — both count=3 but hash differs).
        2. Slow path: numpy vectorized scan for unseen edge_ids.
        """
        edge_count = self.read_edge_count()
        path_hash = self.read_path_hash()
        # When path_hash is 0 (test mode / unset), fall back to edge_count-only
        if edge_count == self._last_edge_count and (
            path_hash == 0 or path_hash == self._last_path_hash
        ):
            return False, set()

        # Slow path: extract edge_ids not yet in _seen_edge_ids
        import numpy as np

        arr = np.frombuffer(
            self._map,
            dtype=np.dtype([("edge_id", "<u4"), ("count", "<u4")]),
            count=self.num_entries,
        )
        active = arr[arr["edge_id"] != 0]
        ids = set(active["edge_id"].tolist())
        new_found = False
        for eid in ids:
            if eid not in self._seen_edge_ids:
                self._seen_edge_ids.add(eid)
                self.cumulative_edges += 1
                self._peak_cumulative_edges = max(
                    self._peak_cumulative_edges, self.cumulative_edges
                )
                new_found = True

        self._last_edge_count = edge_count
        self._last_path_hash = path_hash

        if new_found:
            self.total_edges += 1
        return new_found, ids

    def is_new_coverage(self) -> bool:
        """Check if the current hash table has any edge not seen before.

        Uses the edge_count fast-path header field for O(1) common-case
        detection, falling through to a numpy vectorized scan when the
        count changed.
        """
        new_found, _ = self._check_new_coverage()
        return new_found

    def is_new_coverage_with_edges(self) -> tuple[bool, set[int]]:
        """Check for new coverage AND return current edge set in one scan.

        Avoids scanning the SHM buffer twice — callers that need both the
        boolean and the edge set (e.g. fuzz_one) should use this instead
        of calling is_new_coverage() + get_edge_ids() separately.
        """
        return self._check_new_coverage()

    # ── Manual recording (for tests) ─────────────────────────────────────

    def record_edge(self, edge_id: int) -> bool:
        """Manually record an edge — for tests only.

        Mirrors what the instrumented binary does: hash to slot, linear probe.
        Also maintains the edge_count and path_hash headers (offsets 16 and 8)
        for fast-path consistency with the C shim's behavior.
        """
        pos = edge_id % self.num_entries
        for i in range(self.num_entries):
            idx = (pos + i) % self.num_entries
            eid = self._entries[idx].edge_id
            if eid == 0:
                self._entries[idx].edge_id = edge_id
                self._entries[idx].count = 1
                # Increment edge_count header to reflect new-slot insertion
                ec = ctypes.c_uint64.from_address(self._ptr + 16)
                ec.value = ec.value + 1
                # Update path_hash: hash = hash * 31 ^ edge_id (matches C shim)
                ph = ctypes.c_uint64.from_address(self._ptr + 8)
                ph.value = ph.value * 31 ^ edge_id
                if edge_id not in self._seen_edge_ids:
                    self._seen_edge_ids.add(edge_id)
                    self.cumulative_edges += 1
                    self._peak_cumulative_edges = max(
                        self._peak_cumulative_edges, self.cumulative_edges
                    )
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
        new_table_bytes = new_num_entries * SIZEOF_ENTRY
        new_total_bytes = new_table_bytes + SHM_METADATA_SIZE
        if new_table_bytes <= self.table_bytes:
            return

        new_shm_id = _libc.shmget(0, new_total_bytes, IPC_CREAT | SHM_R | SHM_W)
        if new_shm_id < 0:
            raise OSError(f"shmget resize failed: {os.strerror(ctypes.get_errno())}")

        new_ptr = _libc.shmat(new_shm_id, None, 0)
        if new_ptr == ctypes.c_void_p(-1).value or new_ptr is None:
            _libc.shmctl(new_shm_id, IPC_RMID, None)
            raise OSError(f"shmat resize failed: {os.strerror(ctypes.get_errno())}")

        ctypes.memset(new_ptr, 0, new_total_bytes)
        ctypes.memmove(new_ptr, self._ptr, self.shm_bytes)

        # Detach old SHM
        old_ptr = self._ptr
        old_shm_id = self.shm_id
        _libc.shmdt(old_ptr)
        _libc.shmctl(old_shm_id, IPC_RMID, None)

        self._ptr = new_ptr
        self.shm_id = new_shm_id
        self.num_entries = new_num_entries
        self.table_bytes = new_table_bytes
        self.shm_bytes = new_total_bytes
        self._map = (ctypes.c_char * new_table_bytes).from_address(new_ptr + SHM_METADATA_SIZE)
        EntryArr = _entry_struct(new_num_entries)
        self._entries = EntryArr.from_address(new_ptr + SHM_METADATA_SIZE)
        self.env_id = str(self.shm_id)

        self._peak_cumulative_edges = max(self._peak_cumulative_edges, self.cumulative_edges)
        # Clear position-indexed seen set (positions change after resize)
        self._seen_edge_ids.clear()
        self._last_edge_count = 0
        self.cumulative_edges = 0
        self.total_edges = 0

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
