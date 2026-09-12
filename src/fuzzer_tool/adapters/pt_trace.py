"""Intel PT trace capture: the AUX ring buffer behind ``core.intel_pt``.

``core/intel_pt.py`` decodes a PT byte stream; this is where the bytes come
from.  The kernel writes the trace into a second mmap region (the "AUX" area)
attached to a ``perf_event_open`` descriptor, and the reader advances a tail
pointer as it consumes bytes.

Three things here are worth reading before changing anything.

**The ring arithmetic is a pure function.**  ``ring_chunks()`` maps
``(tail, head, size)`` to the byte ranges to copy, and knows nothing about
mmap.  That split is deliberate: this module cannot be exercised end to end
without PT hardware, so the part that is easy to get wrong is the part that
is testable on a synthetic buffer.

**``aux_head`` and ``aux_tail`` are byte counters, not indices.**  They grow
without bound and are taken modulo ``aux_size`` to find the data.  Comparing
them as indices looks right on the first wrap and silently truncates after
it; ``head - tail > size`` is how overflow is detected, and that check is the
only thing standing between a dropped trace and a coverage map that quietly
loses blocks.

**Config bits are read from sysfs, not hardcoded.**  The PMU publishes the bit
position of every format field under ``format/``; those positions differ
across kernels, and a stale constant would silently ask for a different trace
mode rather than fail.

Ordering: PT exists only on x86, whose TSO model orders load-load and
load-store, so the barriers the C readers need (``rmb()`` after reading
``aux_head``, and a release before publishing ``aux_tail``) are compiler
barriers there rather than fences.  CPython's bytecode dispatch supplies far
more separation than a compiler barrier, so the sequence below is correct as
written -- but it is correct *because of x86*, not because ordering does not
matter.

Not verified on hardware: this container has no ``intel_pt`` PMU, so every
path that requires the kernel to actually fill the buffer is unexercised.
What is verified is the arithmetic, the sysfs parsing, the mmap layout
arithmetic, and that the absent-PMU path degrades instead of raising.
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
    perf_event_attr,
)

log = logging.getLogger(__name__)

PT_SYSFS_ROOT = "/sys/bus/event_source/devices/intel_pt"

# struct perf_event_mmap_page, from a C oracle over the installed uapi header
# rather than from counting fields: the reserved gap before data_head is 928
# bytes and miscounting it puts every subsequent read on the wrong word.
OFF_DATA_HEAD = 1024
OFF_DATA_TAIL = 1032
OFF_DATA_OFFSET = 1040
OFF_DATA_SIZE = 1048
OFF_AUX_HEAD = 1056
OFF_AUX_TAIL = 1064
OFF_AUX_OFFSET = 1072
OFF_AUX_SIZE = 1080

# The kernel requires the data mmap to be 1 + 2^n pages: one control page
# followed by the sample ring.  PT puts nothing in the sample ring, so one
# page is the smallest legal ring rather than a tuning choice.
DEFAULT_DATA_PAGES = 1

# 2 MiB of trace.  A PT stream runs roughly 0.5 bit per instruction retired,
# so this holds tens of millions of instructions -- far past any single fuzz
# execution, which is the point: an overflow costs us the whole trace, not
# the tail of it.
DEFAULT_AUX_PAGES = 512

# What we ask the tracer for, by format-field name.  Absent names are left at
# zero.
#
#   pt         enable packet generation at all
#   branch     emit the branch packets; without it there are no TIP packets
#              and the decoder sees only timing and sync
#   noretcomp  do not compress returns.  With return compression on, a RET
#              target is implied by the call stack rather than carried in a
#              TIP packet, and core.intel_pt reads TIP targets only -- so
#              every return would be invisible to the map.
#
# tsc, mtc and cyc are deliberately absent: timing packets are pure volume
# for a block map, and CYC in particular makes the stream nondeterministic
# run to run, which would show up as phantom coverage churn.
PT_CONFIG_FIELDS = ("pt", "branch", "noretcomp")


def ring_chunks(tail: int, head: int, size: int) -> list[tuple[int, int]]:
    """Byte ranges holding the unread trace, as ``(offset, length)`` pairs.

    *tail* and *head* are the kernel's unbounded byte counters and *size* is
    the AUX buffer length.  Returns one range normally and two across a wrap.

    A full buffer (``head - tail == size``) gives ``tail % size == head %
    size``, which an index-style comparison reads as empty; the length is
    computed from the counter difference for that reason.
    """
    if size <= 0:
        raise ValueError(f"AUX size must be positive, got {size}")

    avail = head - tail
    if avail <= 0:
        return []

    if avail > size:
        raise ValueError(f"{avail} bytes available in a {size}-byte ring")

    start = tail % size
    first = min(avail, size - start)
    chunks = [(start, first)]
    if first < avail:
        chunks.append((0, avail - first))

    return chunks


def read_pmu_type(root: str = PT_SYSFS_ROOT) -> int | None:
    """The dynamic PMU number to put in ``attr.type``, or None if absent."""
    try:
        with open(os.path.join(root, "type")) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def read_format_bits(root: str = PT_SYSFS_ROOT) -> dict[str, tuple[int, int]]:
    """Map each format field to the ``(lo, hi)`` config bits it occupies.

    Entries read ``config:11`` for a single bit or ``config:14-16`` for a
    range.  Fields naming a register other than ``config`` are skipped: they
    exist (``config1``, ``config2``) and silently folding them into ``config``
    would set unrelated bits.
    """
    bits: dict[str, tuple[int, int]] = {}
    fmt_dir = os.path.join(root, "format")
    try:
        names = sorted(os.listdir(fmt_dir))
    except OSError:
        return bits

    for name in names:
        try:
            with open(os.path.join(fmt_dir, name)) as f:
                spec = f.read().strip()
        except OSError:
            continue

        reg, _, rng = spec.partition(":")
        if reg != "config" or not rng:
            continue

        lo, _, hi = rng.partition("-")
        try:
            low = int(lo)
            high = int(hi) if hi else low
        except ValueError:
            continue

        if low > high:
            continue

        bits[name] = (low, high)

    return bits


def build_config(bits: dict[str, tuple[int, int]], fields=PT_CONFIG_FIELDS) -> int:
    """Set each named field to 1 in a config word.

    A field the PMU does not publish is logged and skipped rather than
    guessed at a fixed position: asking for the wrong bit would select a
    different trace mode, which decodes without error into wrong coverage.
    """
    config = 0
    for name in fields:
        pos = bits.get(name)
        if pos is None:
            log.warning("intel_pt PMU has no format field %r; leaving it unset", name)
            continue

        config |= 1 << pos[0]

    return config


class PtTraceSession:
    """One Intel PT event plus its AUX ring.

    Lifecycle mirrors ``PerfCounters``: construct, check ``available``,
    ``open_for_pid``, run the target, ``read_trace``, ``close``.

    Args:
        aux_pages: AUX ring size in pages.  Must be a power of two; the
            kernel rejects anything else.
        data_pages: sample-ring pages.  PT writes no samples here.
        exclude_kernel: drop kernel-space trace.  Forced on for an
            unprivileged process, which cannot ask for kernel trace at all.
        root: sysfs PMU directory, overridable for tests.
        sink: object with ``ingest(bytes) -> int``, normally a
            ``core.intel_pt.PtCoverage``.  ``drain()`` feeds it so the
            execution path never has to know that the bytes are PT packets.
    """

    def __init__(
        self,
        aux_pages: int = DEFAULT_AUX_PAGES,
        data_pages: int = DEFAULT_DATA_PAGES,
        exclude_kernel: bool = True,
        root: str = PT_SYSFS_ROOT,
        sink=None,
    ):
        # Before the validation below, not after: __del__ runs on a partly
        # built object too, and close() would raise AttributeError out of the
        # finalizer -- which Python prints and discards, so the real
        # ValueError arrives alongside unrelated noise.
        self._fd = -1
        self._base: mmap.mmap | None = None
        self._aux: mmap.mmap | None = None

        if aux_pages <= 0 or aux_pages & (aux_pages - 1):
            raise ValueError(f"aux_pages must be a power of two, got {aux_pages}")

        self.page_size = mmap.PAGESIZE
        self.aux_size = aux_pages * self.page_size
        self.data_size = data_pages * self.page_size
        self.exclude_kernel = exclude_kernel
        self.root = root

        self.sink = sink
        self.bytes_read = 0
        self.lost_bytes = 0
        self.overflows = 0
        self.reads = 0
        self.attach_failures = 0

        self._pmu_type = read_pmu_type(root)
        self._format_bits = read_format_bits(root)
        self._libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        self._libc.syscall.restype = ctypes.c_long
        self._libc.syscall.argtypes = [
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
        ]

    @property
    def available(self) -> bool:
        """Whether this host exposes an Intel PT PMU at all.

        Presence of the sysfs directory is the whole test.  Unlike the
        hardware-counter probe in ``perf_event.py`` there is no cheap probe
        open: a PT event allocates its ring on open, so probing costs the
        real allocation.
        """
        return self._pmu_type is not None and bool(self._format_bits)

    @property
    def config(self) -> int:
        return build_config(self._format_bits)

    def _build_attr(self) -> bytes:
        attr = perf_event_attr()
        attr.size = ctypes.sizeof(perf_event_attr)
        attr.type = self._pmu_type
        attr.config = self.config
        flags = _FLAG_DISABLED | _FLAG_EXCLUDE_HV
        if self.exclude_kernel or os.geteuid() != 0:
            flags |= _FLAG_EXCLUDE_KERNEL

        attr.flags = flags
        return bytes(attr)

    def open_for_pid(self, pid: int) -> bool:
        """Open a PT event on *pid* and map its rings.

        The caller must have the target stopped, or already-executed blocks
        are missing from the trace -- unlike a counter, a coverage map cannot
        absorb a late attach.
        """
        if not self.available:
            return False

        self.close()
        attr = self._build_attr()
        fd = self._libc.syscall(
            NR_PERF_EVENT_OPEN,
            (ctypes.c_char * len(attr))(*attr),
            pid,
            -1,  # cpu: follow the task
            -1,  # group_fd
            ctypes.c_ulong(PERF_FLAG_FD_CLOEXEC),
        )
        if fd < 0:
            errno = ctypes.get_errno()
            log.debug("perf_event_open(intel_pt, pid=%d) failed: %s", pid, os.strerror(errno))
            return False

        self._fd = fd
        if not self._map_rings():
            self.close()
            return False

        return True

    def _map_rings(self) -> bool:
        """Map the control page, then the AUX area it points at.

        Two mappings in a fixed order: the AUX offset and size have to be
        written into the control page before the second mmap, because that is
        how the kernel is told what to allocate.
        """
        try:
            self._base = mmap.mmap(
                self._fd,
                self.page_size + self.data_size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except OSError as exc:
            log.debug("mmap of the PT control page failed: %s", exc)
            return False

        aux_offset = self._read_u64(OFF_DATA_OFFSET) + self._read_u64(OFF_DATA_SIZE)
        self._write_u64(OFF_AUX_OFFSET, aux_offset)
        self._write_u64(OFF_AUX_SIZE, self.aux_size)

        try:
            # PROT_WRITE marks the buffer non-overwrite: the kernel stops
            # tracing when it catches the tail instead of wrapping over
            # unread bytes, which turns a lost trace into a detectable stall.
            self._aux = mmap.mmap(
                self._fd,
                self.aux_size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=aux_offset,
            )
        except (OSError, ValueError) as exc:
            log.debug("mmap of the PT AUX area failed: %s", exc)
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
        """Stop tracing.  Required before reading: the head can otherwise
        advance mid-copy and hand back bytes the tail has already passed."""
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

    def read_trace(self) -> bytes:
        """Drain the ring and publish the new tail.

        Returns the raw PT bytes, ready for ``PtCoverage.ingest``.  On
        overflow the oldest bytes are gone, so the stream restarts at a PSB
        and the decoder resyncs there rather than being handed a fragment
        whose IP references it cannot resolve.
        """
        if self._aux is None or self._base is None:
            return b""

        head = self._read_u64(OFF_AUX_HEAD)
        tail = self._read_u64(OFF_AUX_TAIL)
        avail = head - tail
        if avail <= 0:
            return b""

        if avail > self.aux_size:
            self.overflows += 1
            self.lost_bytes += avail - self.aux_size
            tail = head - self.aux_size

        data = b"".join(
            bytes(self._aux[off : off + length])
            for off, length in ring_chunks(tail, head, self.aux_size)
        )
        # Only after the copy: the kernel may reuse everything below the tail
        # the moment this store lands.
        self._write_u64(OFF_AUX_TAIL, head)
        self.reads += 1
        self.bytes_read += len(data)
        return data

    def attach(self, pid: int) -> bool:
        """Open a PT event on *pid* and start tracing.

        One open plus two mmaps per execution, because the descriptor is
        bound to the pid and cannot be rebound.  That is the same cost
        ``PerfCounters`` pays on this path and it is what honggfuzz does; the
        alternative this backend replaces is a ptrace breakpoint per basic
        block, which is roughly two orders of magnitude worse.
        """
        if not self.open_for_pid(pid):
            self.attach_failures += 1
            return False

        if not self.enable():
            self.attach_failures += 1
            self.close()
            return False

        return True

    def drain(self) -> int:
        """End the trace, hand the bytes to the sink, release the event.

        Returns the number of new map entries, or 0 with no sink.  Stopping
        first and closing after is the whole point of doing this in one
        method: a caller that drains without disabling races the producer,
        and one that forgets to close leaks a descriptor and an AUX ring per
        execution.
        """
        if self._fd < 0:
            return 0

        self.disable()
        raw = self.read_trace()
        self.close()
        if not raw or self.sink is None:
            return 0

        return self.sink.ingest(raw)

    def close(self) -> None:
        for region in ("_aux", "_base"):
            buf = getattr(self, region)
            if buf is not None:
                buf.close()
                setattr(self, region, None)

        if self._fd >= 0:
            # close() runs from __del__ too, and an exception there is printed
            # and discarded by the interpreter -- which turns a stale fd into
            # noise on an unrelated traceback instead of a diagnosable error.
            fd, self._fd = self._fd, -1
            try:
                os.close(fd)
            except OSError as exc:
                log.debug("closing pt event fd %d failed: %s", fd, exc)

    @property
    def stats(self) -> dict:
        return {
            "pt_trace_bytes": self.bytes_read,
            "pt_trace_reads": self.reads,
            "pt_trace_lost_bytes": self.lost_bytes,
            "pt_trace_overflows": self.overflows,
            "pt_trace_attach_failures": self.attach_failures,
            "pt_aux_size": self.aux_size,
        }

    def __del__(self):
        self.close()
