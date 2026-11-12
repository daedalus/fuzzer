"""Hardware performance counter support via perf_event_open(2).

Uses Linux perf_event_open syscall to read hardware counters (instruction
count, branch count, branch misses) for target processes. These counters
provide additional coverage signals beyond edge-based coverage — inputs
that execute more instructions or take different branch paths are interesting.

Counters open as one group (one read() per exec) with time enabled/running,
so values multiplexed off the PMU are scaled back up. Without a PMU (most
VMs) the software events -- CPU time, page faults, context switches -- still
count:

    leader fd ── read() ──> nr | enabled | running | v0 v1 ... vN-1
                                  value = v * enabled / running

Requires CAP_PERFMON capability or root. Disabled by default via --hw-perf.

Usage:
    perf = PerfCounters()
    perf.open_for_pid(pid)
    # ... target runs ...
    counters = perf.read_and_reset()
    perf.close()

Ported from honggfuzz linux/perf.c.
"""

import contextlib
import ctypes
import ctypes.util
import fcntl
import logging
import os
import struct

log = logging.getLogger(__name__)

# perf_event_open syscall number (x86_64)
NR_PERF_EVENT_OPEN = 298

# perf_type values
PERF_TYPE_HARDWARE = 0
PERF_TYPE_SOFTWARE = 1
PERF_TYPE_HW_CACHE = 3

# perf_hw_config values (for PERF_TYPE_HARDWARE)
PERF_COUNT_HW_CPU_CYCLES = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_CACHE_REFERENCES = 2
PERF_COUNT_HW_CACHE_MISSES = 3
PERF_COUNT_HW_BRANCH_INSTRUCTIONS = 4
PERF_COUNT_HW_BRANCH_MISSES = 5
PERF_COUNT_HW_BUS_CYCLES = 6
PERF_COUNT_HW_STALLED_CYCLES_FRONTEND = 7
PERF_COUNT_HW_STALLED_CYCLES_BACKEND = 8
PERF_COUNT_HW_REF_CPU_CYCLES = 9

# PERF_TYPE_HW_CACHE config = cache_id | (op << 8) | (result << 16)
PERF_COUNT_HW_CACHE_L1D = 0
PERF_COUNT_HW_CACHE_L1I = 1
PERF_COUNT_HW_CACHE_LL = 2
PERF_COUNT_HW_CACHE_DTLB = 3
PERF_COUNT_HW_CACHE_ITLB = 4
PERF_COUNT_HW_CACHE_BPU = 5
PERF_COUNT_HW_CACHE_OP_READ = 0
PERF_COUNT_HW_CACHE_OP_WRITE = 1
PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
PERF_COUNT_HW_CACHE_RESULT_MISS = 1


def _hw_cache_config(cache_id: int, op: int, result: int) -> int:
    return cache_id | (op << 8) | (result << 16)


# perf_sw_config values (for PERF_TYPE_SOFTWARE)
# include/uapi/linux/perf_event.h: enum perf_sw_ids. 1 is TASK_CLOCK and 2 is
# the total PAGE_FAULTS; the minor/major split lives at 5 and 6.
PERF_COUNT_SW_CPU_CLOCK = 0
PERF_COUNT_SW_TASK_CLOCK = 1
PERF_COUNT_SW_PAGE_FAULTS = 2
PERF_COUNT_SW_CONTEXT_SWITCHES = 3
PERF_COUNT_SW_CPU_MIGRATIONS = 4
PERF_COUNT_SW_PAGE_FAULTS_MIN = 5
PERF_COUNT_SW_PAGE_FAULTS_MAJ = 6
PERF_COUNT_SW_ALIGNMENT_FAULTS = 7

# enum perf_event_read_format
PERF_FORMAT_TOTAL_TIME_ENABLED = 1 << 0
PERF_FORMAT_TOTAL_TIME_RUNNING = 1 << 1
PERF_FORMAT_GROUP = 1 << 3
_READ_FORMAT = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING

# Group read layout: u64 nr, time_enabled, time_running, then nr values.
_GROUP_HDR = 3

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


# Flags bitfield positions, derived from the uapi declaration order so a
# missing field cannot shift the ones after it (see
# include/uapi/linux/perf_event.h; precise_ip is two bits, hence the cut).
_ATTR_FLAG_ORDER = (
    "disabled",
    "inherit",
    "pinned",
    "exclusive",
    "exclude_user",
    "exclude_kernel",
    "exclude_hv",
    "exclude_idle",
    "mmap",
    "comm",
    "freq",
    "inherit_stat",
    "enable_on_exec",
    "task",
    "watermark",
)


def _attr_flag(name: str) -> int:
    return 1 << _ATTR_FLAG_ORDER.index(name)


_FLAG_DISABLED = _attr_flag("disabled")
_FLAG_INHERIT = _attr_flag("inherit")
_FLAG_PINNED = _attr_flag("pinned")
_FLAG_EXCLUDE_USER = _attr_flag("exclude_user")
_FLAG_EXCLUDE_KERNEL = _attr_flag("exclude_kernel")
_FLAG_EXCLUDE_HV = _attr_flag("exclude_hv")
_FLAG_ENABLE_ON_EXEC = _attr_flag("enable_on_exec")


def _perf_ioc(nr: int) -> int:
    """_IO('$', nr) -- the perf ioctl request encoding."""
    return (ord("$") << 8) | nr


PERF_IOC_ENABLE = _perf_ioc(0)
PERF_IOC_DISABLE = _perf_ioc(1)
PERF_IOC_RESET = _perf_ioc(3)

# perf_event_paranoid values above this refuse kernel-space events to an
# unprivileged process.
PARANOID_KERNEL_OK = 0

# Counter definitions: (name, perf_type, perf_config, needs_inherit)
COUNTER_DEFS = {
    "instructions": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, True),
    "branches": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, True),
    "branch_misses": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, True),
    "cpu_cycles": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, True),
    "cache_refs": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, True),
    "cache_misses": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, True),
    "stalled_cycles_frontend": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, True),
    "stalled_cycles_backend": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND, True),
    "ref_cycles": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES, True),
    "bus_cycles": (PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES, True),
    # Inherited like the rest: a forkserver child's faults fold into the
    # counter on the forkserver when the child exits.
    "page_faults": (PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN, True),
    "page_faults_major": (PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ, True),
    "context_switches": (PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, True),
    "task_clock": (PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, True),
    "alignment_faults": (PERF_TYPE_SOFTWARE, PERF_COUNT_SW_ALIGNMENT_FAULTS, True),
    "l1d_read_access": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS
        ),
        True,
    ),
    "l1d_read_miss": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS
        ),
        True,
    ),
    "llc_read_access": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS
        ),
        True,
    ),
    "llc_read_miss": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS
        ),
        True,
    ),
    "dtlb_read_miss": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS
        ),
        True,
    ),
    "l1d_write_access": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_ACCESS
        ),
        True,
    ),
    "l1d_write_miss": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_WRITE, PERF_COUNT_HW_CACHE_RESULT_MISS
        ),
        True,
    ),
    "l1i_read_miss": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_L1I, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS
        ),
        True,
    ),
    # BPU read access/miss are the branch-predictor view of branches and
    # branch_misses on CPUs that expose them (often absent on VMs/AMD).
    "bpu_read_access": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_BPU, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS
        ),
        True,
    ),
    "bpu_read_miss": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_BPU, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS
        ),
        True,
    ),
    "itlb_read_miss": (
        PERF_TYPE_HW_CACHE,
        _hw_cache_config(
            PERF_COUNT_HW_CACHE_ITLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS
        ),
        True,
    ),
}

# Group read struct per member count, built once (the read is per exec).
_GROUP_LAYOUT = [struct.Struct(f"<{_GROUP_HDR + k}Q") for k in range(len(COUNTER_DEFS) + 1)]

DEFAULT_COUNTERS = [
    "instructions",
    "cpu_cycles",
    "branches",
    "branch_misses",
    "l1d_read_access",
    "l1d_read_miss",
    # Software events: no PMU slot, so they never push the group off it.
    "task_clock",
    "page_faults",
    "context_switches",
]


class PerfCounters:
    """Hardware performance counter manager for target processes.

    Opens perf counters on a target PID, reads values after execution,
    and provides deltas between reads.

    Args:
        counter_names: Which counters to enable. Default: DEFAULT_COUNTERS (instructions,
            cpu_cycles, branches, branch_misses, l1d_read_access, l1d_read_miss).
        exclude_kernel: If True, exclude kernel-space events (default False;
            forced on where perf_event_paranoid leaves no choice).
        inherit: If True, inherit counters to child threads (default True).
    """

    def __init__(
        self,
        counter_names: list[str] | None = None,
        exclude_kernel: bool = False,
        inherit: bool = True,
    ):
        if counter_names is None:
            counter_names = list(DEFAULT_COUNTERS)
        self.counter_names = [n for n in counter_names if n in COUNTER_DEFS]
        self.exclude_kernel = exclude_kernel
        self.inherit = inherit
        self._fds: dict[str, int] = {}
        # (read fd, counter names in value order); the leader's group first.
        self._groups: list[tuple[int, tuple[str, ...]]] = []
        self._pid: int | None = None
        # Cleared when a group never got onto the PMU; later opens go solo.
        self._grouped = True
        self._last_values: dict[str, int] = {}
        self._totals: dict[str, int] = {}
        self.has_pmu = False
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
        self._paranoid = self._read_paranoid()
        self._available = self._check_available()
        self._total_instructions = 0
        self._total_cpu_cycles = 0
        self._total_stalled_frontend = 0
        self._total_stalled_backend = 0
        self._total_branches = 0
        self._total_branch_misses = 0
        self._total_l1d_read_access = 0
        self._total_l1d_read_miss = 0
        self._read_count = 0

    @staticmethod
    def _read_paranoid() -> int | None:
        try:
            with open("/proc/sys/kernel/perf_event_paranoid") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def _exclude_kernel_required(self) -> bool:
        """Whether the kernel will refuse the attr without exclude_kernel.

        Unprivileged access at paranoid >= 1 is user-space only, so the flag
        is not optional there however the caller configured it.
        """
        if self._paranoid is None or self._paranoid <= PARANOID_KERNEL_OK:
            return False

        return os.geteuid() != 0

    def _check_available(self) -> bool:
        """Check if perf_event_open hardware counters are available.

        Checks three conditions:
        1. perf_event_paranoid allows access (-1 or 0), or CAP_PERFMON/CAP_SYS_ADMIN is present
        2. Hardware PMU exists (actual syscall probe with pid=0 — covers all
           architectures); else a software event opens (``has_pmu`` False)
        3. perf_event_open syscall works (not blocked by seccomp, etc.)
        """
        # Check 1: perf_event_paranoid
        paranoid_ok = self._paranoid is not None and self._paranoid <= PARANOID_KERNEL_OK

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

        # Check 2: Actual syscall probe on the current thread (pid=0, no
        # extra perms). The only reliable PMU test across architectures
        # (Intel, AMD, ARM, Hygon, ...) without a fragile name whitelist.
        self.has_pmu = self._probe(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS)
        if self.has_pmu:
            return True

        # No PMU (most VMs): software events still count CPU time, faults
        # and context switches.
        return self._probe(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK)

    def _probe(self, perf_type: int, config: int) -> bool:
        """Whether one event of *perf_type*/*config* opens on this thread."""
        pe = perf_event_attr()
        pe.size = ctypes.sizeof(perf_event_attr)
        pe.type = perf_type
        pe.config = config
        pe.flags = _FLAG_DISABLED | _FLAG_EXCLUDE_KERNEL | _FLAG_EXCLUDE_HV
        fd = self._sys_open(bytes(pe), 0, -1)
        if fd < 0:
            errno = ctypes.get_errno()
            log.debug("perf_event_open probe failed: errno=%d (%s)", errno, os.strerror(errno))
            return False

        os.close(fd)
        return True

    def _sys_open(self, attr: bytes, pid: int, group_fd: int) -> int:
        """Raw perf_event_open(2); -1 with errno set on failure."""
        return self._libc.syscall(
            NR_PERF_EVENT_OPEN,
            (ctypes.c_char * len(attr))(*attr),
            pid,
            -1,  # cpu: any
            group_fd,
            ctypes.c_ulong(PERF_FLAG_FD_CLOEXEC),
        )

    @property
    def available(self) -> bool:
        """Whether perf counters can be opened on this system."""
        return self._available

    def open_for_pid(self, pid: int) -> bool:
        """Open perf counters attached to a specific PID, always afresh.

        The spawn path calls this per exec: a recycled pid number is a
        different process, so open fds are never reused here.

        Args:
            pid: Target process PID. 0 for the calling thread (inprocess mode).

        Returns:
            True if all counters were opened successfully.
        """
        if not self._available:
            return False

        self.close()
        self._pid = pid
        return self._open_all(pid)

    def ensure_open(self, pid: int) -> bool:
        """Keep the fds already on *pid*; open them otherwise.

        For long-lived processes (inprocess, forkserver, persistent child):
        one open per process instead of one per exec.
        """
        if self._fds and self._pid == pid:
            return True

        return self.open_for_pid(pid)

    def _attr(self, name: str) -> bytes:
        """perf_event_attr for counter *name*, as the syscall takes it."""
        perf_type, config, needs_inherit = COUNTER_DEFS[name]
        flags = _FLAG_DISABLED | _FLAG_EXCLUDE_HV | _FLAG_ENABLE_ON_EXEC
        if self.exclude_kernel or self._exclude_kernel_required():
            flags |= _FLAG_EXCLUDE_KERNEL
        if needs_inherit and self.inherit:
            flags |= _FLAG_INHERIT

        # ctypes.Structure for the kernel layout; flags packed by hand to
        # avoid ctypes bitfield layout issues.
        pe = perf_event_attr()
        pe.size = ctypes.sizeof(perf_event_attr)
        pe.type = perf_type
        pe.config = config
        pe.flags = flags
        pe.read_format = _READ_FORMAT
        return bytes(pe)

    def _open_all(self, pid: int) -> bool:
        """Open every counter: into the leader's group, else on its own."""
        success = True
        leader = -1
        members: list[str] = []
        for name in self.counter_names:
            attr = self._attr(name)

            # Join the group: one read() serves all, values from one instant.
            fd = self._sys_open(attr, pid, leader) if leader >= 0 else -1
            if fd >= 0:
                members.append(name)
            else:
                fd = self._sys_open(attr, pid, -1)
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

                if self._grouped and leader < 0:
                    leader, members = fd, [name]
                else:
                    self._groups.append((fd, (name,)))

            # Enable now: enable_on_exec only fires on an exec after the
            # open, and the spawn path attaches after it.
            try:
                fcntl.ioctl(fd, PERF_IOC_ENABLE)
            except OSError as exc:
                log.debug("PERF_EVENT_IOC_ENABLE failed for %s: %s", name, exc)

            self._fds[name] = fd
            self._last_values[name] = 0

        if leader >= 0:
            self._groups.insert(0, (leader, tuple(members)))
        if self._fds:
            log.info("Opened %d perf counters: %s", len(self._fds), ", ".join(self._fds.keys()))
        return success

    def read_values(self) -> dict[str, int]:
        """Read current counter values (not deltas), scaled for multiplexing.

        Returns:
            Dict of counter_name -> raw value.
        """
        groups = self._groups
        # Common case, one group: no merge dict.
        if len(groups) == 1:
            return self._read_group(*groups[0])

        values: dict[str, int] = {}
        for fd, names in groups:
            values.update(self._read_group(fd, names))
        return values

    def _read_group(self, fd: int, names: tuple[str, ...]) -> dict[str, int]:
        """One read() for the group; v * enabled / running per value."""
        layout = _GROUP_LAYOUT[len(names)]
        try:
            buf = os.read(fd, layout.size)
        except OSError:
            return dict.fromkeys(names, 0)
        if len(buf) != layout.size:
            return dict.fromkeys(names, 0)

        row = layout.unpack(buf)
        enabled, running = row[1], row[2]
        if running == enabled:
            return dict(zip(names, row[_GROUP_HDR:]))  # noqa: B905 -- layout sized by names

        # Enabled but never on the PMU: the group is wider than the
        # hardware. Zero now; reopen solo so each one multiplexes.
        if running == 0:
            if enabled > 0 and len(names) > 1:
                self._split_groups(names)
            return dict.fromkeys(names, 0)

        return {nm: v * enabled // running for nm, v in zip(names, row[_GROUP_HDR:])}  # noqa: B905

    def _split_groups(self, names: tuple[str, ...]) -> None:
        if not self._grouped:
            return

        log.warning("perf group %s never scheduled; reopening counters ungrouped", names)
        self._grouped = False
        self._pid = None  # next ensure_open() reopens

    def read_and_reset(self) -> dict[str, int]:
        """Read counter values and compute deltas since last read.

        Returns:
            Dict of counter_name -> delta (increase since last read).
        """
        current = self.read_values()
        deltas = {}
        last = self._last_values
        totals = self._totals
        for name, val in current.items():
            # Never negative: a scaled estimate may shrink between reads.
            delta = val - last.get(name, 0)
            if delta < 0:
                delta = 0
            deltas[name] = delta
            last[name] = val
            totals[name] = totals.get(name, 0) + delta

        # Accumulate totals for stats
        self._total_instructions += deltas.get("instructions", 0)
        self._total_cpu_cycles += deltas.get("cpu_cycles", 0)
        self._total_stalled_frontend += deltas.get("stalled_cycles_frontend", 0)
        self._total_stalled_backend += deltas.get("stalled_cycles_backend", 0)
        self._total_branches += deltas.get("branches", 0)
        self._total_branch_misses += deltas.get("branch_misses", 0)
        self._total_l1d_read_access += deltas.get("l1d_read_access", 0)
        self._total_l1d_read_miss += deltas.get("l1d_read_miss", 0)
        self._read_count += 1

        return deltas

    def reset_counters(self) -> None:
        """Reset all counter values to zero."""
        for name, fd in self._fds.items():
            try:
                fcntl.ioctl(fd, PERF_IOC_RESET)
            except OSError as exc:
                # Zeroing _last_values below still makes the next read a
                # delta against now, so a failure here costs accuracy, not
                # correctness — but it must not be silent.
                log.debug("PERF_EVENT_IOC_RESET failed for %s: %s", name, exc)

            self._last_values[name] = 0

    def close(self) -> None:
        """Close all perf counter file descriptors."""
        for fd in self._fds.values():
            with contextlib.suppress(OSError):
                os.close(fd)
        self._fds.clear()
        self._groups.clear()
        self._pid = None
        self._last_values.clear()

    @property
    def stats(self) -> dict:
        """Return cumulative perf stats for display."""
        # total_<name> for every counter; the explicit keys below win.
        out = {f"total_{name}": self._totals.get(name, 0) for name in COUNTER_DEFS}
        return out | {
            "total_instructions": self._total_instructions,
            "total_branches": self._total_branches,
            "total_branch_misses": self._total_branch_misses,
            "branch_miss_rate": (
                self._total_branch_misses / self._total_branches if self._total_branches > 0 else 0
            ),
            "total_l1d_read_access": self._total_l1d_read_access,
            "total_l1d_read_miss": self._total_l1d_read_miss,
            "l1d_miss_rate": (
                self._total_l1d_read_miss / self._total_l1d_read_access
                if self._total_l1d_read_access > 0
                else 0
            ),
            "read_count": self._read_count,
            "total_cpu_cycles": self._total_cpu_cycles,
            "total_stalled_cycles_frontend": self._total_stalled_frontend,
            "total_stalled_cycles_backend": self._total_stalled_backend,
            # Instructions per cycle: 0 when cycles were not counted.
            "ipc": (
                self._total_instructions / self._total_cpu_cycles
                if self._total_cpu_cycles > 0
                else 0
            ),
            # Instructions per branch (what "ipc" used to report).
            "ipb": (
                self._total_instructions / self._total_branches if self._total_branches > 0 else 0
            ),
            "stall_frontend_rate": (
                self._total_stalled_frontend / self._total_cpu_cycles
                if self._total_cpu_cycles > 0
                else 0
            ),
            "stall_backend_rate": (
                self._total_stalled_backend / self._total_cpu_cycles
                if self._total_cpu_cycles > 0
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
