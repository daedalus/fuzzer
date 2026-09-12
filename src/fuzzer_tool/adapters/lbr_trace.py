"""Branch-record capture: the perf data ring behind ``core.branch_record``.

``core/branch_record.py`` decodes perf sample records; this is where they come
from.  The shape mirrors ``adapters/pt_trace.py`` -- ``attach()`` per
execution, ``drain()`` after it -- with two differences that come straight
from the hardware.

**The data ring, not an AUX ring.**  PT needs a second mapping because the
CPU writes the trace itself.  Branch records arrive as ordinary perf sample
records in the base mapping's data region, so there is one mmap and the
offsets come from ``data_offset``/``data_size`` in the control page.  The
counter arithmetic is identical, which is why ``ring_chunks()`` is imported
from ``pt_trace`` rather than restated here.

**There is no PMU directory to probe.**  PT publishes
``/sys/bus/event_source/devices/intel_pt``; branch sampling is a property of
the core PMU, so the only honest probe is opening the event.  Zen 3 BRS also
needs ``CONFIG_PERF_EVENTS_AMD_BRS``, which is opt-in at kernel build time,
and BRS and LBR are mutually exclusive in hardware -- so "this CPU is a Zen
3" does not imply the event opens, and vendor detection cannot replace the
probe.  ``available`` runs the probe once, without mapping a ring.

Sampling period: the hardware holds 16 taken branches, refreshed at each
overflow, so with period P roughly 16/P of branches are observed.  Lowering
P buys coverage and costs interrupts; it does not converge on a trace.
"""

import ctypes
import ctypes.util
import fcntl
import logging
import mmap
import os
import struct

from fuzzer_tool.adapters.perf_event import (
    _FLAG_DISABLED,
    _FLAG_EXCLUDE_HV,
    _FLAG_EXCLUDE_KERNEL,
    NR_PERF_EVENT_OPEN,
    PERF_FLAG_FD_CLOEXEC,
    PERF_IOC_DISABLE,
    PERF_IOC_ENABLE,
    PERF_TYPE_HARDWARE,
    perf_event_attr,
)
from fuzzer_tool.adapters.pt_trace import (
    OFF_DATA_HEAD,
    OFF_DATA_OFFSET,
    OFF_DATA_SIZE,
    OFF_DATA_TAIL,
    ring_chunks,
)

log = logging.getLogger(__name__)

PERF_COUNT_HW_BRANCH_INSTRUCTIONS = 4

# sample_type, in uapi declaration order -- core.branch_record._decode_sample
# parses exactly this and nothing else.
PERF_SAMPLE_IP = 1 << 0
PERF_SAMPLE_TID = 1 << 1
PERF_SAMPLE_BRANCH_STACK = 1 << 11
SAMPLE_LAYOUT = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_BRANCH_STACK

# branch_sample_type.  USER is requested even though BRS ignores it in
# hardware: on LbrExtV2 and Intel LBR it is a real filter, and asking for it
# costs nothing where it is not.  The software-side check in
# core.branch_record is what actually keeps kernel addresses out.
PERF_SAMPLE_BRANCH_USER = 1 << 0
PERF_SAMPLE_BRANCH_ANY = 1 << 3
BRANCH_FILTER = PERF_SAMPLE_BRANCH_USER | PERF_SAMPLE_BRANCH_ANY

# Retired taken branches between overflows.  4000 is what the kernel's own
# BRS reproducers use; low enough to fill the file often on a parser, high
# enough that the interrupt rate does not dominate a short execution.
DEFAULT_PERIOD = 4000

# Records are small (16 entries -> 408 bytes) but arrive in bursts, so the
# ring is sized for a few hundred of them rather than for one.
DEFAULT_DATA_PAGES = 64


class LbrSession:
    """One branch-sampling event plus its data ring.

    Args:
        period: retired taken branches between samples.
        data_pages: ring size in pages.  Must be a power of two.
        sink: object with ``ingest(bytes) -> int``, normally a
            ``core.branch_record.BranchCoverage``.
    """

    def __init__(
        self,
        period: int = DEFAULT_PERIOD,
        data_pages: int = DEFAULT_DATA_PAGES,
        sink=None,
    ):
        # Set before validation: __del__ runs on a partly built object too,
        # and close() raising out of a finalizer buries the real error.
        self._fd = -1
        self._base: mmap.mmap | None = None

        if data_pages <= 0 or data_pages & (data_pages - 1):
            raise ValueError(f"data_pages must be a power of two, got {data_pages}")

        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")

        self.page_size = mmap.PAGESIZE
        self.data_size = data_pages * self.page_size
        self.period = period
        self.sink = sink

        self.bytes_read = 0
        self.reads = 0
        self.attach_failures = 0
        self._available: bool | None = None

    def _build_attr(self) -> bytes:
        attr = perf_event_attr()
        attr.size = ctypes.sizeof(perf_event_attr)
        attr.type = PERF_TYPE_HARDWARE
        attr.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS
        attr.sample_period_or_freq = self.period
        attr.sample_type = SAMPLE_LAYOUT
        attr.branch_sample_type = BRANCH_FILTER
        attr.flags = _FLAG_DISABLED | _FLAG_EXCLUDE_KERNEL | _FLAG_EXCLUDE_HV
        return bytes(attr)

    def _open(self, pid: int) -> int:
        attr = self._build_attr()
        return self._syscall_open(attr, pid)

    def _syscall_open(self, attr: bytes, pid: int) -> int:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.syscall.restype = ctypes.c_long
        libc.syscall.argtypes = [
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        return libc.syscall(
            NR_PERF_EVENT_OPEN,
            (ctypes.c_char * len(attr))(*attr),
            pid,
            -1,  # cpu: follow the task
            -1,  # group_fd
            ctypes.c_ulong(PERF_FLAG_FD_CLOEXEC),
        )

    @property
    def available(self) -> bool:
        """Whether a branch-stack event opens on this host.

        Probed once against the calling thread, with no mmap: the ring is
        allocated at mmap time, so the probe costs a descriptor and nothing
        else.  There is no sysfs entry that answers this -- see the module
        docstring.
        """
        if self._available is None:
            fd = self._open(0)
            if fd < 0:
                errno = ctypes.get_errno()
                log.debug("branch-stack probe failed: %s", os.strerror(errno))
                self._available = False
            else:
                os.close(fd)
                self._available = True

        return self._available

    def attach(self, pid: int) -> bool:
        """Open a branch-sampling event on *pid* and start it."""
        if not self.available:
            self.attach_failures += 1
            return False

        self.close()
        fd = self._open(pid)
        if fd < 0:
            self.attach_failures += 1
            log.debug("perf_event_open(branch stack, pid=%d) failed", pid)
            return False

        self._fd = fd
        if not self._map_ring() or not self.enable():
            self.attach_failures += 1
            self.close()
            return False

        return True

    def _map_ring(self) -> bool:
        """Map the control page plus the data region behind it.

        One mapping, unlike PT: the header page and the data ring are
        contiguous and sized together, so ``data_offset`` is just where the
        records start inside it.
        """
        try:
            self._base = mmap.mmap(
                self._fd,
                self.page_size + self.data_size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except (OSError, ValueError) as exc:
            log.debug("mmap of perf data ring failed: %s", exc)
            return False

        return True

    def _read_u64(self, offset: int) -> int:
        assert self._base is not None
        return struct.unpack_from("<Q", self._base, offset)[0]

    def _write_u64(self, offset: int, value: int) -> None:
        assert self._base is not None
        struct.pack_into("<Q", self._base, offset, value)

    def enable(self) -> bool:
        return self._ioctl(PERF_IOC_ENABLE)

    def disable(self) -> bool:
        return self._ioctl(PERF_IOC_DISABLE)

    def _ioctl(self, request: int) -> bool:
        if self._fd < 0:
            return False

        try:
            fcntl.ioctl(self._fd, request)
            return True
        except OSError as exc:
            log.debug("perf ioctl 0x%x failed: %s", request, exc)
            return False

    def read_records(self) -> bytes:
        """Drain the data ring and publish the new tail.

        ``data_head``/``data_tail`` are byte counters taken modulo
        ``data_size``, exactly like the AUX pair, so the same
        ``ring_chunks()`` decides what to copy.  Unlike AUX there is no
        overflow accounting to do here: the kernel reports overwritten
        records as PERF_RECORD_LOST inside the stream itself, which
        ``core.branch_record.count_lost`` reads.
        """
        if self._base is None:
            return b""

        head = self._read_u64(OFF_DATA_HEAD)
        tail = self._read_u64(OFF_DATA_TAIL)
        if head - tail <= 0:
            return b""

        size = self._read_u64(OFF_DATA_SIZE) or self.data_size
        start = self._read_u64(OFF_DATA_OFFSET) or self.page_size
        span = min(head - tail, size)
        data = b"".join(
            bytes(self._base[start + off : start + off + length])
            for off, length in ring_chunks(head - span, head, size)
        )
        # Only after the copy: everything below the tail is the kernel's to
        # reuse the moment this store lands.
        self._write_u64(OFF_DATA_TAIL, head)
        self.reads += 1
        self.bytes_read += len(data)
        return data

    def drain(self) -> int:
        """Stop sampling, hand the records to the sink, release the event."""
        if self._fd < 0:
            return 0

        self.disable()
        raw = self.read_records()
        self.close()
        if not raw or self.sink is None:
            return 0

        return self.sink.ingest(raw)

    def close(self) -> None:
        if self._base is not None:
            self._base.close()
            self._base = None

        if self._fd >= 0:
            # Runs from __del__ too, where an exception is printed and
            # discarded -- a stale fd would surface as noise on an unrelated
            # traceback instead of anything diagnosable.
            fd, self._fd = self._fd, -1
            try:
                os.close(fd)
            except OSError as exc:
                log.debug("closing branch event fd %d failed: %s", fd, exc)

    @property
    def stats(self) -> dict:
        return {
            "br_trace_bytes": self.bytes_read,
            "br_trace_reads": self.reads,
            "br_trace_attach_failures": self.attach_failures,
            "br_period": self.period,
            "br_data_size": self.data_size,
        }

    def __del__(self):
        self.close()
