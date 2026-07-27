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
NR_PERF_EVENT_OPEN = 298

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
        exclude_kernel: bool = False,
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
            ctypes.c_long,  # syscall number
            ctypes.c_void_p,  # perf_event_attr*
            ctypes.c_int,  # pid
            ctypes.c_int,  # cpu
            ctypes.c_int,  # group_fd
            ctypes.c_ulong,  # flags
        ]
        self._available = self._check_available()
        self._total_instructions = 0
        self._total_branches = 0
        self._total_branch_misses = 0
        self._read_count = 0

    def _check_available(self) -> bool:
        """Check if perf_event_open hardware counters are available.

        Checks three conditions:
        1. perf_event_paranoid allows access (-1 or 0), or CAP_PERFMON/CAP_SYS_ADMIN is present
        2. Hardware PMU exists (actual syscall probe with pid=0 — covers all architectures)
        3. perf_event_open syscall works (not blocked by seccomp, etc.)
        """
        # Check 1: perf_event_paranoid
        paranoid_ok = False
        try:
            with open("/proc/sys/kernel/perf_event_paranoid") as f:
                paranoid = int(f.read().strip())
                paranoid_ok = paranoid <= 0
        except (OSError, ValueError):
            pass

        if not paranoid_ok and os.geteuid() != 0:
            # Check capabilities
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("CapEff:"):
                            caps = int(line.split()[1], 16)
                            if caps & (1 << 38) or caps & (1 << 21):
                                paranoid_ok = True
                            break
            except (OSError, ValueError):
                pass

        if not paranoid_ok and os.geteuid() != 0:
            return False

        # Check 2: Actual syscall probe — open PERF_COUNT_HW_INSTRUCTIONS
        # on the current thread (pid=0, no extra perms needed). This is the
        # only reliable way to test PMU availability across architectures
        # (Intel, AMD, ARM, Hygon, etc.) without maintaining a fragile name
        # whitelist of /sys/bus/event_source/devices/ entries.
        pe = perf_event_attr()
        pe.size = ctypes.sizeof(perf_event_attr)
        pe.type = PERF_TYPE_HARDWARE
        pe.config = PERF_COUNT_HW_INSTRUCTIONS
        flags = _FLAG_DISABLED | _FLAG_EXCLUDE_KERNEL | _FLAG_EXCLUDE_HV
        pe.flags = flags
        pe_bytes = bytes(pe)

        fd = self._libc.syscall(
            NR_PERF_EVENT_OPEN,
            (ctypes.c_char * len(pe_bytes))(*pe_bytes),
            0,  # pid=0: current thread, no CAP_PERFMON required
            -1,  # cpu: any
            -1,  # group_fd: none
            ctypes.c_ulong(PERF_FLAG_FD_CLOEXEC),
        )
        if fd < 0:
            errno = ctypes.get_errno()
            log.debug("perf_event_open probe failed: errno=%d (%s)", errno, os.strerror(errno))
            return False

        os.close(fd)
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

            # Build perf_event_attr as raw bytes at known offsets
            # to avoid ctypes bitfield layout issues.

            flags = 0
            flags |= _FLAG_DISABLED
            if self.exclude_kernel:
                flags |= _FLAG_EXCLUDE_KERNEL
            flags |= _FLAG_EXCLUDE_HV
            if needs_inherit and self.inherit:
                flags |= _FLAG_INHERIT
            flags |= _FLAG_ENABLE_ON_EXEC

            # Use ctypes.Structure for correct layout (varies by kernel version)
            pe = perf_event_attr()
            pe.size = ctypes.sizeof(perf_event_attr)
            pe.type = perf_type
            pe.config = config
            pe.flags = flags
            pe = bytes(pe)

            fd = self._libc.syscall(
                NR_PERF_EVENT_OPEN,  # __NR_perf_event_open (x86_64)
                (ctypes.c_char * len(pe))(*pe),
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

            # Enable immediately — enable_on_exec only triggers on the
            # process that calls exec(), but we open on the parent PID.
            # The child inherits already-enabled counters via inherit=1.
            import fcntl

            PERF_IOC_ENABLE = 0x2400
            try:
                fcntl.ioctl(fd, PERF_IOC_ENABLE)
            except OSError:
                pass

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


# ── C shim interface (for subprocess mode) ──────────────────────────────


class PerfShim:
    """C shim interface for perf counters via ctypes.

    Loads the compiled perf_shim.so and provides perf_open/perf_read/perf_close.
    Useful for subprocess mode where Python can't easily attach perf counters
    to a child PID that's already running — the C shim can be called from
    the subprocess adapter.

    Usage:
        shim = PerfShim()
        if shim.available:
            fd = shim.perf_open(child_pid, PERF_COUNT_HW_INSTRUCTIONS)
            # ... child runs ...
            val = shim.perf_read(fd)
            shim.perf_close(fd)
    """

    def __init__(self, shim_path: str | None = None):
        self._lib = None
        self._shim_path = shim_path
        if shim_path:
            try:
                self._lib = ctypes.CDLL(shim_path)
                self._setup_functions()
            except (OSError, AttributeError):
                log.debug("Failed to load perf_shim from %s", shim_path)

    def _setup_functions(self):
        if not self._lib:
            return
        self._lib.perf_open.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._lib.perf_open.restype = ctypes.c_int
        self._lib.perf_read.argtypes = [ctypes.c_int]
        self._lib.perf_read.restype = ctypes.c_uint64
        self._lib.perf_reset.argtypes = [ctypes.c_int]
        self._lib.perf_reset.restype = ctypes.c_int
        self._lib.perf_close.argtypes = [ctypes.c_int]
        self._lib.perf_close.restype = None

    @property
    def available(self) -> bool:
        return self._lib is not None

    def perf_open(
        self, pid: int, hw_config: int, exclude_kernel: bool = True, inherit: bool = True
    ) -> int:
        """Open a perf counter on a PID via C shim."""
        if not self._lib:
            return -1
        return self._lib.perf_open(pid, hw_config, int(exclude_kernel), int(inherit))

    def perf_read(self, fd: int) -> int:
        """Read counter value via C shim."""
        if not self._lib or fd < 0:
            return 0
        return self._lib.perf_read(fd)

    def perf_reset(self, fd: int) -> int:
        """Reset counter via C shim."""
        if not self._lib or fd < 0:
            return -1
        return self._lib.perf_reset(fd)

    def perf_close(self, fd: int) -> None:
        """Close counter via C shim."""
        if self._lib and fd >= 0:
            self._lib.perf_close(fd)
