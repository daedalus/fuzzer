"""Hardware performance counter support via perf_event_open(2).

Uses Linux perf_event_open syscall to read hardware counters (instruction
count, branch count, branch misses) for target processes. These counters
provide additional coverage signals beyond edge-based coverage — inputs
that execute more instructions or take different branch paths are interesting.

Requires CAP_PERFMON capability or root. Disabled by default via --hw-perf.

Usage:
    perf = PerfCounters()
    perf.open_for_pid(pid)
    # ... target runs ...
    counters = perf.read_and_reset()
    perf.close()

Ported from honggfuzz linux/perf.c.
"""

import ctypes
import ctypes.util
import logging
import os
import struct

log = logging.getLogger(__name__)

# perf_event_open syscall number (x86_64)
__NR_perf_event_open = 298

# perf_type values
PERF_TYPE_HARDWARE = 0
PERF_TYPE_SOFTWARE = 1

# perf_hw_config values (for PERF_TYPE_HARDWARE)
PERF_COUNT_HW_CPU_CYCLES = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_CACHE_REFERENCES = 2
PERF_COUNT_HW_CACHE_MISSES = 3
PERF_COUNT_HW_BRANCH_INSTRUCTIONS = 4
PERF_COUNT_HW_BRANCH_MISSES = 5
PERF_COUNT_HW_BUS_CYCLES = 6

# perf_sw_config values (for PERF_TYPE_SOFTWARE)
PERF_COUNT_SW_CPU_CLOCK = 0
PERF_COUNT_SW_PAGE_FAULTS_MIN = 1
PERF_COUNT_SW_PAGE_FAULTS_MAJ = 2

# perf_event_attr flags
PERF_FLAG_FD_CLOEXEC = 8


class perf_event_attr(ctypes.Structure):
    """Linux perf_event_attr structure (packed for syscall)."""

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period_or_freq", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),  # bitfield: disabled, inherit, pinned, etc.
        ("wakeup_events_or_watermark", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("__reserved_2", ctypes.c_uint16),
    ]


# Flags bitfield positions (within the 64-bit flags field)
_FLAG_DISABLED = 1 << 0
_FLAG_INHERIT = 1 << 1
_FLAG_PINNED = 1 << 2
_FLAG_EXCLUDE_USER = 1 << 3
_FLAG_EXCLUDE_KERNEL = 1 << 4
_FLAG_EXCLUDE_HV = 1 << 5
_FLAG_ENABLE_ON_EXEC = 1 << 11

# Counter definitions: (name, perf_type, perf_config, needs_inherit)
COUNTER_DEFS = {
    "instructions": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, True),
    "branches": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, True),
    "branch_misses": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, True),
    "cpu_cycles": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, True),
    "cache_refs": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, True),
    "cache_misses": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, True),
    "page_faults": (PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN, False),
}


class PerfCounters:
    """Hardware performance counter manager for target processes.

    Opens perf counters on a target PID, reads values after execution,
    and provides deltas between reads.

    Args:
        counter_names: Which counters to enable. Default: ["instructions", "branches", "branch_misses"].
        exclude_kernel: If True, exclude kernel-space events (default True).
        inherit: If True, inherit counters to child threads (default True).
    """

    def __init__(
        self,
        counter_names: list[str] | None = None,
        exclude_kernel: bool = True,
        inherit: bool = True,
    ):
        if counter_names is None:
            counter_names = ["instructions", "branches", "branch_misses"]
        self.counter_names = [n for n in counter_names if n in COUNTER_DEFS]
        self.exclude_kernel = exclude_kernel
        self.inherit = inherit
        self._fds: dict[str, int] = {}
        self._last_values: dict[str, int] = {}
        self._libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        self._libc.syscall.restype = ctypes.c_long
        self._libc.syscall.argtypes = [
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self._available = self._check_available()
        self._total_instructions = 0
        self._total_branches = 0
        self._total_branch_misses = 0
        self._read_count = 0

    def _check_available(self) -> bool:
        """Check if perf_event_open is available on this system."""
        if os.geteuid() != 0:
            # Check for CAP_PERFMON
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("CapEff:"):
                            caps = int(line.split()[1], 16)
                            # CAP_PERFMON is bit 38
                            if caps & (1 << 38):
                                return True
                            # CAP_SYS_ADMIN is bit 21 (allows perf_event_open)
                            if caps & (1 << 21):
                                return True
                            return False
            except (OSError, ValueError):
                return False
        return True

    @property
    def available(self) -> bool:
        """Whether perf counters can be opened on this system."""
        return self._available

    def open_for_pid(self, pid: int) -> bool:
        """Open perf counters attached to a specific PID.

        Args:
            pid: Target process PID. Use -1 for current process (inprocess mode).

        Returns:
            True if all counters were opened successfully.
        """
        if not self._available:
            return False

        self.close()
        success = True

        for name in self.counter_names:
            perf_type, config, needs_inherit = COUNTER_DEFS[name]

            pe = perf_event_attr()
            pe.size = ctypes.sizeof(perf_event_attr)
            pe.type = perf_type
            pe.config = config

            flags = 0
            flags |= _FLAG_DISABLED
            if self.exclude_kernel:
                flags |= _FLAG_EXCLUDE_KERNEL
            flags |= _FLAG_EXCLUDE_HV
            if needs_inherit and self.inherit:
                flags |= _FLAG_INHERIT
            # enable_on_exec for non-persistent targets
            flags |= _FLAG_ENABLE_ON_EXEC
            pe.flags = flags

            fd = self._libc.syscall(
                __NR_perf_event_open,
                ctypes.byref(pe),
                pid,  # target pid
                -1,  # cpu (any)
                -1,  # group_fd (none)
                ctypes.c_ulong(PERF_FLAG_FD_CLOEXEC),
            )
            if fd < 0:
                errno = ctypes.get_errno()
                log.debug(
                    "perf_event_open failed for %s: errno=%d (%s)",
                    name,
                    errno,
                    os.strerror(errno),
                )
                success = False
                continue

            self._fds[name] = fd
            self._last_values[name] = 0

        if self._fds:
            log.info("Opened %d perf counters: %s", len(self._fds), ", ".join(self._fds.keys()))
        return success

    def read_values(self) -> dict[str, int]:
        """Read current counter values (not deltas).

        Returns:
            Dict of counter_name -> raw value.
        """
        values = {}
        for name, fd in self._fds.items():
            try:
                # read() returns 8 bytes (uint64)
                buf = os.read(fd, 8)
                if len(buf) == 8:
                    val = struct.unpack("<Q", buf)[0]
                    values[name] = val
                else:
                    values[name] = 0
            except OSError:
                values[name] = 0
        return values

    def read_and_reset(self) -> dict[str, int]:
        """Read counter values and compute deltas since last read.

        Returns:
            Dict of counter_name -> delta (increase since last read).
        """
        current = self.read_values()
        deltas = {}
        for name, val in current.items():
            last = self._last_values.get(name, 0)
            delta = val - last if val >= last else val  # handle wrap
            deltas[name] = delta
            self._last_values[name] = val

        # Accumulate totals for stats
        self._total_instructions += deltas.get("instructions", 0)
        self._total_branches += deltas.get("branches", 0)
        self._total_branch_misses += deltas.get("branch_misses", 0)
        self._read_count += 1

        return deltas

    def reset_counters(self) -> None:
        """Reset all counter values to zero."""
        for name, fd in self._fds.items():
            try:
                # ioctl RESET = 0
                import fcntl

                fcntl.ioctl(fd, 0)
            except (OSError, ImportError):
                pass
            self._last_values[name] = 0

    def close(self) -> None:
        """Close all perf counter file descriptors."""
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        self._last_values.clear()

    @property
    def stats(self) -> dict:
        """Return cumulative perf stats for display."""
        return {
            "total_instructions": self._total_instructions,
            "total_branches": self._total_branches,
            "total_branch_misses": self._total_branch_misses,
            "read_count": self._read_count,
            "ipc": (
                self._total_instructions / max(1, self._total_branches)
                if self._total_branches > 0
                else 0
            ),
        }

    def __del__(self):
        self.close()
