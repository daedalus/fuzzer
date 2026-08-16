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

import numpy as np

from fuzzer_tool.core.count_class import bucket_bits

log = logging.getLogger(__name__)

# Dtype of one edge-table entry, shared by every reader in this module.
_ENTRY_DTYPE = np.dtype([("edge_id", "<u4"), ("count", "<u4")])

# Default number of hash table entries.
# SHM default = 8192 entries * 8 bytes = 65536 bytes.
SHM_MAP_SIZE = 8192  # number of entries
SIZEOF_ENTRY = 8  # bytes per {edge_id: u32, count: u32}

# Metadata region in the front header of SHM (before the edge table).
# Layout (24 bytes total):
#   offset 0: uint32 stack_depth   (max stack depth in bytes, from __sancov_lowest_stack)
#   offset 4: uint32 diag          (bits 0-7: __AFL_CTX_BITS, bits 8-31: dropped edges)
#   offset 8: uint64 path_hash     (rolling hash: hash = hash * 31 ^ edge_id)
#   offset 16: uint64 edge_count   (monotonic new-slot insertion count)
SHM_METADATA_SIZE = 24  # bytes reserved at front of SHM for metadata

# AFLGo SHM-tail distance channel: after the edge table, 16 bytes hold
# the per-execution average distance accumulated by the target:
#   offset 0: uint64 dist_sum    (sum of block distances × 100)
#   offset 8: uint64 dist_count  (number of valued blocks hit)
# The tail is written by the shim in distance builds (__AFL_DISTANCE_MODE)
# and stays zeroed otherwise, so readers fall back to Python-side
# distance computation when count == 0.
SHM_TAIL_SIZE = 16  # bytes reserved at the end of SHM for the distance tail

# Upper bound on the edge_id the virgin bucket map indexes directly.
#
# The map is keyed by edge_id into a dense uint8 array rather than a dict,
# because it is consulted on every execution that changes coverage: a
# fancy-index over the whole active set is one vectorized call, where a
# dict lookup per active entry is a Python-level loop over (on a
# saturating target) tens of thousands of entries per exec.  Measured on
# 200k active edges: 1.8ms direct-indexed against 28.7ms via a sorted-array
# searchsorted and 108ms via the dict loop.  edges/second is the metric, so
# the novelty test must not itself be O(active) in the interpreter.
#
# Direct indexing is affordable because trace_pc_guard_init hands out small
# sequential guard values, so edge_id = prev_loc ^ cur_loc lands in a dense
# range of roughly 2 * guard_count -- and XOR with an __AFL_CTX_BITS-wide
# context term cannot widen a value past its wider operand, so the default
# 8-bit context does not change that.  One byte per reachable edge_id is
# ~512 KiB for a target with 200k live edges, against the 8 bytes per SLOT
# the edge table itself already costs.
#
# The exception is __AFL_CTX_BITS in the 24..32 range, which -- as the
# shim's own comment says -- scatters ids across the entire u32 space and
# would ask for up to 4 GiB here.  Ids at or above this bound go to a dict
# instead.  That path is the interpreter loop this design exists to avoid,
# but it is only reachable on a configuration whose live id count is
# already unbounded, and it stays correct there.
VIRGIN_DENSE_MAX = 1 << 24

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
        self.shm_bytes = self.table_bytes + SHM_METADATA_SIZE + SHM_TAIL_SIZE

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
        # Distance tail at self._ptr + SHM_METADATA_SIZE + table_bytes
        self._tail = self._ptr + SHM_METADATA_SIZE + self.table_bytes

        self.env_id = str(self.shm_id)

        # Cumulative "ever seen" set of edge_ids (not positions)
        self._seen_edge_ids: set[int] = set()
        # Virgin hit-count bucket map: for each edge_id, the OR of every
        # bucket bit that edge has ever been observed in.  Indexed by
        # edge_id directly (see VIRGIN_DENSE_MAX), with a dict for ids
        # past the dense range.
        self._virgin: np.ndarray = np.zeros(0, dtype=np.uint8)
        self._virgin_wide: dict[int, int] = {}
        # Count of (edge, bucket) pairs observed for the first time — the
        # coverage that set membership alone cannot see.  Diagnostic only.
        self.bucket_transitions: int = 0
        # Last seen edge_count for O(1) fast-path in is_new_coverage
        self._last_edge_count: int = 0
        # Last seen path_hash for fast-path — catches same-count but different-edge sets
        self._last_path_hash: int = 0
        # Current edge set as of the last slow-path scan. Returned by the
        # fast path instead of an empty set() when nothing changed, so
        # callers that snapshot the return value (e.g. Fuzzer._prev_edge_set)
        # get a real "same as before" set rather than a value that reads as
        # "the target fired zero edges this exec".
        self._last_ids: set[int] = set()

        self.total_edges = 0
        self.cumulative_edges = 0
        # Vestigial since resize() stopped zeroing cumulative_edges: the
        # counter is monotonic now, so peak == current. Kept because
        # stats.py reads it, and because it stays correct either way.
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
        arr = np.frombuffer(self._map, dtype=_ENTRY_DTYPE, count=self.num_entries)
        active = arr[arr["edge_id"] != 0]
        # .tolist() converts numpy uint32 to plain Python ints
        return set(active["edge_id"].tolist())

    def get_edge_counts(self) -> dict[int, int]:
        """Return {edge_id: count} for all non-empty entries."""
        arr = np.frombuffer(self._map, dtype=_ENTRY_DTYPE, count=self.num_entries)
        active = arr[arr["edge_id"] != 0]
        # .tolist() converts numpy uint32 to plain Python ints
        return dict(zip(active["edge_id"].tolist(), active["count"].tolist(), strict=False))

    # ── Reset ────────────────────────────────────────────────────────────

    def reset_edge_map(self):
        """Zero all entries in the coverage hash table (preserves front header).

        Also zeroes the distance tail so a stale sum/count from a previous
        execution can never be misread.
        """
        ctypes.memset(self._ptr + SHM_METADATA_SIZE, 0, self.table_bytes)
        ctypes.memset(self._tail, 0, SHM_TAIL_SIZE)

    # ── AFLGo distance tail (per-execution average distance) ────────────

    def read_distance_tail(self) -> tuple[int, int]:
        """Read (dist_sum, dist_count) written by the shim's distance channel.

        Average distance = dist_sum / dist_count / 100 (distances are
        stored scaled by 100 as integers).  Returns (0, 0) when the
        target was not built with the distance table.
        """
        dist_sum = ctypes.c_uint64.from_address(self._tail).value
        dist_count = ctypes.c_uint64.from_address(self._tail + 8).value
        return dist_sum, dist_count

    # ── Metadata (stack depth + path hash + edge count) ────────────────

    # ── Diagnostics word (offset 4) ─────────────────────────────────────
    #
    # The shim packs two things into what used to be a pad: the context
    # width the target was compiled with, and a saturating count of edges it
    # had to throw away because the open-addressing probe found no free slot.
    #
    # The drop count is the only honest occupancy signal available. Every
    # other measure -- len(_seen_edge_ids), EdgeTracker._global_edge_hits,
    # bitmap_density() -- is computed from edges that made it INTO the table,
    # so a table so full it is losing edges reads as under-occupied from
    # here. Reading occupancy alone, the fuzzer concludes the map is fine
    # exactly when it is at its worst.

    DIAG_CTX_MASK = 0xFF
    DIAG_DROP_SHIFT = 8
    DIAG_DROP_MAX = 0xFFFFFF

    def read_diag(self) -> int:
        """Read the raw diagnostics word from the SHM front header."""
        return ctypes.c_uint32.from_address(self._ptr + 4).value

    def read_ctx_bits(self) -> int:
        """Context width (__AFL_CTX_BITS) the running target was built with.

        0 means context-free coverage. Only meaningful after the target has
        run at least once -- before that the header has never been written.
        Use elf.detect_ctx_bits() for the pre-run, static answer.
        """
        return self.read_diag() & self.DIAG_CTX_MASK

    def read_dropped_edges(self) -> int:
        """Edges lost to a full table, cumulative over the run.

        Saturates at DIAG_DROP_MAX. Any non-zero value means the map is too
        small for this target and coverage is being silently discarded.
        """
        return self.read_diag() >> self.DIAG_DROP_SHIFT

    def drop_counter_saturated(self) -> bool:
        """True when the drop count has pinned and is no longer a magnitude."""
        return self.read_dropped_edges() >= self.DIAG_DROP_MAX

    def reset_diag(self) -> None:
        """Clear the drop count, preserving the context width.

        Called after a resize so the next saturation decision is made on
        evidence from the new table rather than the old one.
        """
        ctypes.c_uint32.from_address(self._ptr + 4).value = self.read_diag() & self.DIAG_CTX_MASK

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

    # ── Hit-count bucketing ─────────────────────────────────────────────
    #
    # Set membership answers "was this edge ever hit". It cannot answer
    # "was this edge ever hit THIS MANY times", and that second question is
    # the mechanism by which AFL crosses loop-count-guarded branches:
    # `if (n > 16)`, buffer-growth paths, parser backtrack limits. An input
    # driving a loop twice and one driving it 128 times produce the same
    # edge set and, without bucketing, the same verdict.
    #
    # The bucket ladder lives in core/count_class.py. Here we keep the
    # virgin map: edge_id -> OR of every bucket bit seen for that edge.

    def _update_virgin_buckets(self, edge_ids: np.ndarray, counts: np.ndarray) -> bool:
        """Fold this execution's hit counts into the virgin bucket map.

        Args:
            edge_ids: uint32 edge ids of the active (non-empty) entries.
            counts: matching uint32 raw hit counts.

        Returns:
            True when at least one (edge, bucket) pair was seen for the
            first time — i.e. this execution found coverage that the
            edge-id set alone would have called old.
        """
        bits = bucket_bits(counts)
        occupied = bits != 0
        if not occupied.any():
            # Every active entry has count 0. The C shim never writes such
            # an entry (a claimed slot starts at 1), so this is either a
            # hand-built table in a test or a torn read.
            return False
        eids = np.asarray(edge_ids, dtype=np.uint32)[occupied]
        bits = bits[occupied]

        found = False
        in_range = eids < VIRGIN_DENSE_MAX
        if not in_range.all():
            found = self._update_virgin_overflow(eids[~in_range], bits[~in_range])
            eids, bits = eids[in_range], bits[in_range]
            if eids.size == 0:
                return found

        hi = int(eids.max()) + 1
        if hi > self._virgin.size:
            grown = np.zeros(1 << max(10, (hi - 1).bit_length()), dtype=np.uint8)
            grown[: self._virgin.size] = self._virgin
            self._virgin = grown

        prev = self._virgin[eids]
        fresh = bits & ~prev
        if not fresh.any():
            return found
        self.bucket_transitions += int(np.count_nonzero(fresh))
        # edge_ids are unique within the table — the shim's probe gives each
        # edge exactly one slot — so this fancy-indexed store has no
        # duplicate targets and is not order-dependent.
        self._virgin[eids] = prev | bits
        return True

    def _update_virgin_overflow(self, eids: np.ndarray, bits: np.ndarray) -> bool:
        """Virgin update for edge ids past the dense array's range.

        Unreachable under any default build; see VIRGIN_DENSE_MAX.
        """
        found = False
        for eid, bit in zip(eids.tolist(), bits.tolist(), strict=False):
            prev = self._virgin_wide.get(eid, 0)
            if bit & ~prev:
                self._virgin_wide[eid] = prev | bit
                self.bucket_transitions += 1
                found = True
        return found

    def get_virgin_buckets(self) -> dict[int, int]:
        """Return {edge_id: OR of bucket bits ever seen} — diagnostics/tests."""
        hit = np.nonzero(self._virgin)[0]
        buckets = dict(zip(hit.tolist(), self._virgin[hit].tolist(), strict=False))
        buckets.update(self._virgin_wide)
        return buckets

    # ── New-coverage detection ──────────────────────────────────────────

    def _check_new_coverage(self) -> tuple[bool, set[int]]:
        """Check for new coverage and return (has_new, current_edge_ids).

        Two-tier approach:
        1. Fast path: compare edge_count AND path_hash against last seen
           (both O(1) header reads) — avoids touching the edge table when
           nothing changed.  The path_hash comparison catches the case where
           the same number of edges fire but the edge SET has changed (e.g.
           {1,2,3} vs {1,2,4} — both count=3 but hash differs).
        2. Slow path: numpy vectorized scan for unseen edge_ids and for
           unseen hit-count buckets.

        The fast path stays safe now that hit counts matter, because the
        shim advances path_hash on *every* edge fire — count bump and
        full-table miss included, not just new-slot insertion. An
        unchanged path_hash therefore means the same edges fired the same
        number of times in the same order, so no bucket can have moved.

        The `path_hash == 0` fallback is the exception: edge_count counts
        new-slot insertions only and is blind to multiplicity, so a pure
        count change is invisible there. That branch exists for tables
        built by hand in tests; a live target that ran at all has a
        non-zero hash.
        """
        edge_count = self.read_edge_count()
        path_hash = self.read_path_hash()
        # When path_hash is 0 (test mode / unset), fall back to edge_count-only
        if edge_count == self._last_edge_count and (
            path_hash == 0 or path_hash == self._last_path_hash
        ):
            # Nothing changed since the last scan: return the edge set we
            # already know is current, not an empty set. An empty set here
            # used to read as "this exec fired no edges", which corrupted
            # any caller that diffs consecutive returns (see
            # Fuzzer._prev_edge_set / format-learner new_edges tracking).
            return False, self._last_ids

        # Slow path: extract edge_ids not yet in _seen_edge_ids
        arr = np.frombuffer(self._map, dtype=_ENTRY_DTYPE, count=self.num_entries)
        active = arr[arr["edge_id"] != 0]
        ids = set(active["edge_id"].tolist())
        new = ids - self._seen_edge_ids
        self._seen_edge_ids.update(new)
        new_found = False
        if new:
            self.cumulative_edges += len(new)
            self._peak_cumulative_edges = max(self._peak_cumulative_edges, self.cumulative_edges)
            new_found = True

        # Hit-count bucketing. OR'd with the set-membership result rather
        # than replacing it: a new edge whose count is 0 occupies no bucket,
        # which the shim never produces but tests and torn reads do, and
        # cumulative_edges must stay an edge count either way.
        if self._update_virgin_buckets(active["edge_id"], active["count"]):
            new_found = True

        self._last_edge_count = edge_count
        self._last_path_hash = path_hash
        self._last_ids = ids

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

    def _note_bucket(self, edge_id: int, count: int) -> None:
        """Fold a single {edge_id, count} into the virgin bucket map."""
        self._update_virgin_buckets(
            np.array([edge_id], dtype=np.uint32),
            np.array([count], dtype=np.uint32),
        )

    def record_edge(self, edge_id: int) -> bool:
        """Manually record an edge — for tests only.

        Mirrors what the instrumented binary does: hash to slot, linear probe.
        Also maintains the edge_count and path_hash headers (offsets 16 and 8)
        for fast-path consistency with the C shim's behavior.  Like the C
        shim's __afl_map_edge (afl_shim.c), the path_hash advances on every
        edge fire — new-slot insert, count bump, or full-table miss — not
        just on new slots.

        It also folds the resulting count into the virgin bucket map, for
        the same reason it already updates _seen_edge_ids: this method
        stands in for both the writer and the reader, so leaving the reader
        half stale would make a recorded edge report as new coverage on the
        next scan. Folding per call marks each bucket the count passes
        through, which is what a sequence of executions would have done.
        """
        pos = edge_id % self.num_entries
        stored = False
        for i in range(self.num_entries):
            idx = (pos + i) % self.num_entries
            eid = self._entries[idx].edge_id
            if eid == 0:
                self._entries[idx].edge_id = edge_id
                self._entries[idx].count = 1
                # Increment edge_count header to reflect new-slot insertion
                ec = ctypes.c_uint64.from_address(self._ptr + 16)
                ec.value = ec.value + 1
                if edge_id not in self._seen_edge_ids:
                    self._seen_edge_ids.add(edge_id)
                    self.cumulative_edges += 1
                    self._peak_cumulative_edges = max(
                        self._peak_cumulative_edges, self.cumulative_edges
                    )
                self._note_bucket(edge_id, 1)
                stored = True
                break
            if eid == edge_id:
                if self._entries[idx].count < 0xFFFFFFFF:
                    self._entries[idx].count += 1
                self._note_bucket(edge_id, int(self._entries[idx].count))
                stored = True
                break
        # Update path_hash unconditionally: hash = hash * 31 ^ edge_id
        # (matches the C shim, which fires on every edge — full-table misses
        # included).
        ph = ctypes.c_uint64.from_address(self._ptr + 8)
        ph.value = ph.value * 31 ^ edge_id
        if stored:
            self.total_edges += 1
        return stored

    # ── Resize ───────────────────────────────────────────────────────────

    def resize(self, new_num_entries: int) -> None:
        """Resize the hash table (allocates new SHM, copies entries).

        Args:
            new_num_entries: New table size (must be > current).
        """
        new_table_bytes = new_num_entries * SIZEOF_ENTRY
        new_total_bytes = new_table_bytes + SHM_METADATA_SIZE + SHM_TAIL_SIZE
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
        # Copy the HEADER only, not the table.
        #
        # The old table's entries sit at positions computed under the old
        # modulus, so carrying them over puts every one of them in the wrong
        # place: the shim would start probing at edge_id % new_size, miss the
        # stale copy, and claim a SECOND slot for an edge already present.
        # That is invisible today purely because reset_edge_map() memsets the
        # table before every execution — but it becomes a live duplicate-entry
        # bug the moment that per-exec clear is replaced with generation
        # tagging. Copying only the header is both correct and cheaper: the
        # table is scratch, the header (path_hash, edge_count, diag) is not.
        ctypes.memmove(new_ptr, self._ptr, SHM_METADATA_SIZE)

        # Detach old SHM. Drop the views into it first: they are
        # from_address handles that would otherwise point at freed memory
        # between the shmdt and the rebind below, and any numpy array still
        # holding self._map from an earlier call keeps that stale target.
        old_ptr = self._ptr
        old_shm_id = self.shm_id
        self._map = None
        self._entries = None
        self._tail = None
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
        self._tail = new_ptr + SHM_METADATA_SIZE + new_table_bytes
        self.env_id = str(self.shm_id)

        # _seen_edge_ids is NOT position-indexed, despite what the comment
        # here used to say. It is populated from active["edge_id"], and
        # edge_id = ctx ^ prev_loc ^ cur_loc carries no map_size term — only
        # the starting probe position edge_id % map_size moves, and no
        # position is ever persisted. Clearing it made the next execution
        # report every already-known edge as new, which zeroed the reported
        # coverage, reset the stall detector that had just triggered this
        # resize, and saved a burst of inputs for coverage already held.
        # _virgin/_virgin_wide are keyed the same way and survive for
        # the same reason.
        #
        # _last_edge_count is reset because the header edge_count is a
        # per-process cumulative that the fast path only compares for
        # CHANGE; after the header copy above it is still meaningful, but
        # forcing one full scan on the next execution costs a single exec
        # and removes any dependence on that being true.
        self._last_edge_count = 0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def cleanup(self):
        """Detach and remove the segment, and drop every view into it.

        ``_map``, ``_entries`` and ``_tail`` are ctypes views created with
        ``from_address``: they hold a raw address and own nothing, so
        detaching without clearing them leaves three live handles on freed
        memory. That is not merely a crash risk -- it is mostly *silent*.
        The kernel hands back the same address on the next ``shmat`` (six
        successive attach/detach cycles all returned the same one here), so
        a stale view reads and writes whichever segment now occupies the
        address: another ShmCoverage's coverage table. It only segfaults in
        the window where nothing has re-attached.

        Clearing them turns that into a loud AttributeError/TypeError at the
        first use, which is what a use-after-free should look like.
        """
        self._map = None
        self._entries = None
        self._tail = None
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


class DistanceTableShm:
    """Shared memory holding the PC→distance table for distance builds.

    The shim (afl_shim.c, ``__AFL_DISTANCE_MODE``) attaches this segment
    via the ``__AFL_DIST_SHM_ID`` env var and probes it from
    ``__sanitizer_cov_trace_pc()``.  Layout::

        uint32 capacity
        struct { uint64 key; uint32 dist; } slots[capacity]

    The header is the *slot capacity*, not the entry count: a power of
    two >= 2x entries, so empty slots always exist and the shim's linear
    probe (``key % capacity``, break on key == 0) exits early on misses.
    Entries are inserted at their hash position with linear probing,
    mirroring the shim's reader exactly.

    *key* is the block start address relative to the object's lowest
    PT_LOAD vaddr (the shim looks up ``pc - dladdr_base``); *dist* is
    the AFLGo block distance scaled by 100.  Entries with key == 0 are
    empty slots.
    """

    def __init__(self, entries: dict[int, float]):
        """Create and populate the table SHM.

        Args:
            entries: {block-start key: distance} — keys already relative
                to the object base, distances unscaled floats.
        """
        scaled = {key: max(0, round(dist * 100)) for key, dist in entries.items() if key != 0}
        self.num_entries = len(scaled)
        entry_bytes = 12  # {u64 key, u32 dist}
        if self.num_entries == 0:
            self.shm_id = -1
            self._ptr = None
            self.env_id = "0"
            return

        self.capacity = 1
        while self.capacity < 2 * self.num_entries:
            self.capacity *= 2
        self.shm_bytes = 4 + self.capacity * entry_bytes

        self.shm_id = _libc.shmget(0, self.shm_bytes, IPC_CREAT | SHM_R | SHM_W)
        if self.shm_id < 0:
            raise OSError(f"shmget failed: {os.strerror(ctypes.get_errno())}")
        self._ptr = _libc.shmat(self.shm_id, None, 0)
        if self._ptr == ctypes.c_void_p(-1).value or self._ptr is None:
            _libc.shmctl(self.shm_id, IPC_RMID, None)
            raise OSError(f"shmat failed: {os.strerror(ctypes.get_errno())}")
        ctypes.memset(self._ptr, 0, self.shm_bytes)
        ctypes.c_uint32.from_address(self._ptr).value = self.capacity
        for key, dist in scaled.items():
            pos = key % self.capacity
            while True:
                off = self._ptr + 4 + pos * entry_bytes
                if ctypes.c_uint64.from_address(off).value == 0:
                    ctypes.c_uint64.from_address(off).value = key
                    ctypes.c_uint32.from_address(off + 8).value = dist
                    break
                pos = (pos + 1) % self.capacity
        self.env_id = str(self.shm_id)
        atexit.register(self.cleanup)

    def cleanup(self):
        """Detach and remove the segment.

        ``_ptr`` is cleared to ``None``, not ``0``: the populate loop above
        computes ``self._ptr + 4 + pos * entry_bytes``, and a zeroed pointer
        makes that arithmetic succeed and produce a plausible near-null
        address to write through. ``None`` raises TypeError instead.
        """
        if self._ptr:
            _libc.shmdt(self._ptr)
        self._ptr = None
        if self.shm_id >= 0:
            _libc.shmctl(self.shm_id, IPC_RMID, None)
            self.shm_id = -1
