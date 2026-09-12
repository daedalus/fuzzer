"""Sampled branch-record coverage: the AMD counterpart to ``core.intel_pt``.

There is no AMD equivalent of Intel PT.  No AMD x86 part emits a complete
control-flow trace; what exists is a 16-deep branch register file sampled at
a PMU overflow:

    Zen 3   BRS       (CONFIG_PERF_EVENTS_AMD_BRS, ``branch-brs`` event)
    Zen 4+  LbrExtV2  (hardware branch filtering, LBR-freeze-on-PMI)

Both surface through ``perf_event_open`` as ``PERF_SAMPLE_BRANCH_STACK`` on an
ordinary sampling event, so one reader covers them -- and, incidentally,
Intel LBR as well, since the sample format is vendor-neutral.  The vendor
differences live in whether the event opens at all, not in what comes back.

**This signal is not a trace, and the difference is not a detail.**  A PT
stream is every control transfer.  A branch stack is the last 16 taken
branches before each overflow, so with period P the fuzzer sees roughly
16/P of the branches that executed.

What saves it for coverage is an asymmetry:

    A recorded from->to pair is a branch that *really executed*.  Sampling
    produces false negatives, never false positives.

So "this input reached a new edge" is sound evidence, while "this input
reached nothing new" is not evidence of anything.  Discovery and admission
can consume this map.  Stability calibration and the trim's subset test
cannot: both read absence of coverage as a fact about the input, and here it
is a fact about the sampling period.  ``SAMPLED`` is exported so those call
sites can refuse the map rather than silently mis-read it.

One entry the hardware hands over must be dropped by us.  BRS has no
hardware privilege filter, so a stack requested with
``PERF_SAMPLE_BRANCH_USER`` can still carry kernel branch-from addresses --
SYSRET and interrupt returns (CVE-2026-72237, fixed in the kernel by
filtering in software; we cannot assume a fixed kernel underneath).  A
kernel address hashed into the map is coverage the target never had, which
is the one error class that breaks the asymmetry above.  ``_is_user()`` is
therefore load-bearing, not hygiene.

Unlike PT's TIP targets, a branch entry carries both ends of the transfer,
so these are genuine CFG edges: ``(from >> 1) ^ to`` is the AFL edge hash
over real predecessor/successor pairs rather than over path fragments.
"""

import logging
import struct
from typing import NamedTuple

from fuzzer_tool.core.count_class import classify_counts

log = logging.getLogger(__name__)

# This map is sampled, so it is incomplete by construction.  Consumers that
# read "no new coverage" as a property of the input must check this flag and
# opt out; see the module docstring.
SAMPLED = True

DEFAULT_MAP_SIZE = 65536

# The hardware records 16 consecutive taken branches per overflow on both
# BRS and LbrExtV2.  Used only to sanity-bound a decoded count: a stack
# longer than the file can hold means the record was misparsed.
MAX_BRANCH_ENTRIES = 64

# include/uapi/linux/perf_event.h
PERF_RECORD_LOST = 2
PERF_RECORD_SAMPLE = 9

RECORD_HEADER_BYTES = 8  # type u32, misc u16, size u16
BRANCH_ENTRY_BYTES = 24  # from u64, to u64, flags u64

# x86-64 splits the canonical address space in half; everything at or above
# this is kernel or non-canonical and cannot be target code.
KERNEL_ADDR_FLOOR = 0xFFFF800000000000


class BranchEntry(NamedTuple):
    """One taken branch: where it came from, where it went."""

    frm: int
    to: int
    flags: int


class BranchSample(NamedTuple):
    ip: int
    pid: int
    tid: int
    entries: list


def _is_user(addr: int) -> bool:
    return 0 < addr < KERNEL_ADDR_FLOOR


def _decode_sample(body: bytes) -> BranchSample | None:
    """Parse one PERF_RECORD_SAMPLE body.

    Field order is fixed by the uapi and depends on ``sample_type``; this
    matches ``SAMPLE_LAYOUT`` in ``adapters/lbr_trace.py`` exactly --
    IP, TID, then the branch stack.  PERF_SAMPLE_BRANCH_HW_INDEX is
    deliberately not requested: it inserts a u64 between the count and the
    entries, and a reader that expects it where it is absent decodes the
    first entry's ``from`` as the index and shears every pair after it.
    """
    if len(body) < 24:  # ip + pid/tid + nr
        return None

    ip, pid, tid, nr = struct.unpack_from("<QIIQ", body, 0)
    if nr > MAX_BRANCH_ENTRIES:
        log.debug("implausible branch stack depth %d, dropping record", nr)
        return None

    need = 24 + nr * BRANCH_ENTRY_BYTES
    if len(body) < need:
        return None

    entries = [
        BranchEntry(*struct.unpack_from("<QQQ", body, 24 + i * BRANCH_ENTRY_BYTES))
        for i in range(nr)
    ]
    return BranchSample(ip, pid, tid, entries)


def iter_samples(buf: bytes):
    """Yield the ``BranchSample``s in a copied span of the perf data ring.

    Non-sample records are skipped by their own header size rather than
    guessed at, so an unhandled record type costs nothing.  A record whose
    header runs past the end of *buf* is not yielded: the kernel publishes
    only whole records, so a truncated one means the span was cut, and its
    bytes belong to the next read.
    """
    pos = 0
    while pos + RECORD_HEADER_BYTES <= len(buf):
        rtype, _misc, size = struct.unpack_from("<IHH", buf, pos)
        if size < RECORD_HEADER_BYTES or pos + size > len(buf):
            return

        if rtype == PERF_RECORD_SAMPLE:
            sample = _decode_sample(buf[pos + RECORD_HEADER_BYTES : pos + size])
            if sample is not None:
                yield sample

        pos += size


def count_lost(buf: bytes) -> int:
    """Total of the PERF_RECORD_LOST counters in *buf*.

    The kernel reports overwritten records rather than letting the reader
    discover the gap, and the number matters here: lost records are edges
    the map will never learn, so a run that keeps losing them needs a
    bigger ring or a longer period, not more executions.
    """
    lost = 0
    pos = 0
    while pos + RECORD_HEADER_BYTES <= len(buf):
        rtype, _misc, size = struct.unpack_from("<IHH", buf, pos)
        if size < RECORD_HEADER_BYTES or pos + size > len(buf):
            break

        # body: u64 id, u64 lost
        if rtype == PERF_RECORD_LOST and size >= RECORD_HEADER_BYTES + 16:
            lost += struct.unpack_from("<Q", buf, pos + RECORD_HEADER_BYTES + 8)[0]

        pos += size

    return lost


class BranchCoverage:
    """Edge map fed by sampled branch records.

    Mirrors the surface ``PtCoverage`` and ``PtraceCoverage`` expose
    (``edge_map``, ``reset_edge_map``, ``is_new_coverage``) so the closed-source
    coverage backends stay interchangeable at the call site.  ``sampled`` is
    the one place they differ, and it is public for that reason.
    """

    sampled = SAMPLED

    def __init__(self, map_size: int = DEFAULT_MAP_SIZE):
        self.map_size = map_size
        self.edge_map = bytearray(map_size)
        self.cumulative_edges = 0
        self.total_edges = 0
        self.total_branches = 0
        self.total_samples = 0
        self.dropped_kernel = 0
        self.lost_records = 0
        self.total_bytes = 0
        self._base_address: int | None = None
        self._map_snapshot = classify_counts(bytes(self.edge_map))

    def set_base(self, base: int | None) -> None:
        """Hash addresses relative to *base*.

        A PIE target lands somewhere new on every exec, so absolute
        addresses would make the map unreproducible run to run -- the same
        reason ``PtraceCoverage.record_edge`` subtracts the load base.
        """
        self._base_address = base

    def ingest(self, raw: bytes) -> int:
        """Decode a data-ring span into the map; return how many edges are new."""
        self.total_bytes += len(raw)
        self.lost_records += count_lost(raw)
        new = 0
        for sample in iter_samples(raw):
            self.total_samples += 1
            for entry in sample.entries:
                if not _is_user(entry.frm) or not _is_user(entry.to):
                    self.dropped_kernel += 1
                    continue

                self.total_branches += 1
                new += self.record_edge(entry.frm, entry.to)

        return new

    def record_edge(self, frm: int, to: int) -> bool:
        base = self._base_address or 0
        # Both ends are real, so this is an AFL edge hash over an actual
        # predecessor/successor pair. The >>1 keeps a self-loop from
        # collapsing to bucket 0.
        bucket = (((frm - base) >> 1) ^ (to - base)) % self.map_size
        if self.edge_map[bucket]:
            return False

        self.edge_map[bucket] = 1
        self.total_edges += 1
        self.cumulative_edges += 1
        return True

    def reset_edge_map(self) -> None:
        """Clear per-execution state.  The map itself is owned by the caller,
        exactly as in the ptrace and PT backends."""
        self.total_edges = 0
        self._map_snapshot = classify_counts(bytes(self.edge_map))

    def is_new_coverage(self) -> bool:
        """Whether this execution added an edge.

        True is sound: the edge was recorded, so it executed.  False means
        only that nothing new was *sampled*.
        """
        classified = classify_counts(bytes(self.edge_map))
        if classified == self._map_snapshot:
            return False

        self._map_snapshot = classified
        return True

    @property
    def stats(self) -> dict:
        return {
            "br_samples": self.total_samples,
            "br_branches": self.total_branches,
            "br_map_entries": self.cumulative_edges,
            "br_dropped_kernel": self.dropped_kernel,
            "br_lost_records": self.lost_records,
            "br_bytes": self.total_bytes,
        }
