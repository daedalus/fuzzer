"""Regressions for the perf_event_attr flag bitfield and its ioctls.

Finding 50 of docs/bugreport_2026-08-21_merged.md, plus an off-by-one it did
not name.  Expected positions are derived from the uapi declaration order
rather than copied from the module, so a shifted field cannot pass by echoing
its own constant back.
"""

import ctypes
import os

from fuzzer_tool.adapters import perf_event

# include/uapi/linux/perf_event.h, struct perf_event_attr, in declaration
# order.  precise_ip is two bits wide, so the list stops before it.
UAPI_FLAG_ORDER = (
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


def bit(name: str) -> int:
    return 1 << UAPI_FLAG_ORDER.index(name)


def test_regression_exclude_flags_are_not_shifted():
    """The old constants omitted ``exclusive``, so every flag from bit 3 on
    was one position low: ``exclude_kernel`` wrote ``exclude_user`` and
    counted nothing in user space.  That is the real cause of the AMD symptom
    recorded in test_default_exclude_kernel_false, which read it as
    exclude_kernel being harmful."""
    actual = {
        "exclude_user": perf_event._FLAG_EXCLUDE_USER,
        "exclude_kernel": perf_event._FLAG_EXCLUDE_KERNEL,
        "exclude_hv": perf_event._FLAG_EXCLUDE_HV,
    }
    assert actual == {name: bit(name) for name in actual}


def test_regression_enable_on_exec_is_not_inherit_stat():
    assert bit("enable_on_exec") == perf_event._FLAG_ENABLE_ON_EXEC
    assert bit("inherit_stat") != perf_event._FLAG_ENABLE_ON_EXEC


def test_regression_disabled_and_inherit_unchanged():
    """The first three positions were right and must stay right."""
    actual = {
        "disabled": perf_event._FLAG_DISABLED,
        "inherit": perf_event._FLAG_INHERIT,
        "pinned": perf_event._FLAG_PINNED,
    }
    assert actual == {name: bit(name) for name in actual}


def test_regression_ioc_reset_is_not_enable():
    """``ioctl(fd, 0)`` is PERF_EVENT_IOC_ENABLE, not RESET, so the reset was
    a no-op in hardware and the next delta carried the whole accumulation."""
    ioc = perf_event._perf_ioc
    assert ioc(0) == perf_event.PERF_IOC_ENABLE
    assert ioc(3) == perf_event.PERF_IOC_RESET
    assert perf_event.PERF_IOC_RESET != perf_event.PERF_IOC_ENABLE
    # _IO('$', nr): type byte in bits 8-15, sequence number in bits 0-7.
    assert ioc(3) == (ord("$") << 8) | 3


def test_reset_uses_the_reset_ioctl(monkeypatch):
    """Falsification: with the old ``ioctl(fd, 0)`` this records request 0."""
    seen = []
    monkeypatch.setattr(perf_event.fcntl, "ioctl", lambda fd, req: seen.append((fd, req)))

    # A real descriptor of our own: close() runs in __del__, and handing it
    # a number pytest owns would close pytest's file.
    fd = os.open(os.devnull, os.O_RDONLY)
    pc = perf_event.PerfCounters()
    pc._fds = {"instructions": fd}
    pc._last_values = {"instructions": 123}
    pc.reset_counters()

    assert seen == [(fd, perf_event.PERF_IOC_RESET)]
    assert pc._last_values["instructions"] == 0
    pc.close()


def test_exclude_kernel_forced_when_paranoid_restricts(monkeypatch):
    """perf_event_open refuses kernel-space events at paranoid >= 1 for an
    unprivileged user, so the attr must carry exclude_kernel whatever the
    caller asked for.  Before the bit fix this happened by accident: the
    constant named exclude_hv landed on exclude_kernel."""
    pc = perf_event.PerfCounters(exclude_kernel=False)
    monkeypatch.setattr(pc, "_paranoid", 2)
    monkeypatch.setattr(perf_event.os, "geteuid", lambda: 1000)
    assert pc._exclude_kernel_required()

    monkeypatch.setattr(pc, "_paranoid", -1)
    assert not pc._exclude_kernel_required()


def test_attr_struct_size_matches_kernel_layout():
    """112 is PERF_ATTR_SIZE_VER5 exactly, which is why the kernel accepts
    this struct: perf_copy_attr() zero-extends a short attr to its own size
    but rejects one whose trailing bytes it cannot account for.  Growing the
    struct without moving to the next version boundary earns E2BIG."""
    assert ctypes.sizeof(perf_event.perf_event_attr) == 112
