"""Shim health: interpret afl_shim.c's markers and runtime counters.

Three sources, one vocabulary of human-readable issues:

    target ELF ── marker scan ──> stale_shim_issues()   (before any exec)
    first exec ── edge count ───> self_test_issues()    (startup self-test)
    every exec ── stderr ───────> ShimWatch.stderr()    (any mode)
    loaded .so ── __afl_shim_health() ──> ShimWatch.counters() (periodic,
                                          direct mode) / health_issues() (report)

ShimWatch reports each issue once per run, so a fault that repeats on every
execution produces one warning, not a stream.

Reading the counters needs a loaded library, which is an adapter concern
(``adapters.inprocess.read_shim_health``); this module only judges them.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from enum import Enum, IntEnum

from fuzzer_tool.core.elf import detect_scoped_crash_handler

#: Prefix of every diagnostic afl_shim.c writes to stderr (attach failures,
#: refused segments). Fixed by the C source; never printed by a healthy run.
SHIM_STDERR_TAG = "__afl_shim:"

#: Executions between reads of ``__afl_shim_health()`` in direct mode.
SHIM_HEALTH_PERIOD = 1000

#: Distinct stderr lines remembered for dedup; keeps memory bounded when a
#: target prints a fresh line on every execution.
STDERR_SEEN_MAX = 64

#: Input of the startup self-test execution (same shape as the speed probe).
SELF_TEST_INPUT = b"\x00" * 64

# Read by the shim at attach; selects the edge-table write guard.
WGUARD_ENV = "__AFL_WGUARD"


class ShimField(IntEnum):
    """Index of each counter in ``__afl_shim_health()``'s output (C ABI)."""

    ATTACHED = 0
    MAP_ENTRIES = 1
    SEG_REJECTED = 2
    CMPLOG_DROPPED = 3
    STRAY_SIGNALS = 4
    ABORTS_INTERCEPTED = 5
    HANDLERS_DISPLACED = 6
    WGUARD = 7


class WriteGuard(IntEnum):
    """``ShimField.WGUARD`` value: how the edge table is write-locked."""

    OFF = 0
    PKEY = 1
    MPROTECT = 2


def export_wguard(mode: WriteGuard) -> None:
    """Ask the shim for *mode* in every target started after this call.

    Off sets nothing: the shim arms no guard when the variable is unset.
    """
    if mode is WriteGuard.OFF:
        return
    os.environ[WGUARD_ENV] = mode.name.lower()


class Attach(Enum):
    """Whether the run asked the shim for an edge table."""

    EXPECTED = "expected"
    NOT_EXPECTED = "not_expected"


def stale_shim_issues(target: str) -> list[str]:
    """Issues readable from *target*'s symbol table alone.

    Only meaningful for a binary known to carry the shim: an uninstrumented
    binary lacks the marker too. Unreadable targets yield nothing.
    """
    if detect_scoped_crash_handler(target) is not False:
        return []
    return [
        f"{target} was built against a shim without __afl_scoped_crash_handler: "
        "crashes outside __afl_guarded_call are reported as SIGSEGV whatever "
        "the real signal, ASAN's SEGV report is lost, and an in-process host "
        "dies on SIGPIPE. Rebuild with tools/build_targets.sh."
    ]


def _field(counters: Sequence[int], field: ShimField) -> int:
    """Counter value, 0 when an older shim does not export it."""
    return counters[field] if field < len(counters) else 0


def health_issues(counters: Sequence[int], attach: Attach) -> list[str]:
    """Issues implied by one ``__afl_shim_health()`` snapshot.

    Example: ``[1, 8192, 2, 0, 0]`` -> ``["2 SHM segment(s) refused ..."]``.
    """
    issues = []

    attached = _field(counters, ShimField.ATTACHED) or len(counters) <= ShimField.ATTACHED
    if attach is Attach.EXPECTED and not attached:
        issues.append("edge table not attached: the coverage map stays empty")

    rejected = _field(counters, ShimField.SEG_REJECTED)
    if rejected:
        issues.append(
            f"{rejected} SHM segment(s) refused (bad id, size or header); "
            "the target's stderr names which"
        )

    dropped = _field(counters, ShimField.CMPLOG_DROPPED)
    if dropped:
        issues.append(f"{dropped} cmplog record(s) dropped (writer contention or failed write)")

    stray = _field(counters, ShimField.STRAY_SIGNALS)
    if stray:
        issues.append(f"{stray} crash signal(s) outside __afl_guarded_call, handed back")

    aborts = _field(counters, ShimField.ABORTS_INTERCEPTED)
    if aborts:
        issues.append(
            f"{aborts} abort() call(s) intercepted and returned; the target ran on "
            "past a failed assertion"
        )

    displaced = _field(counters, ShimField.HANDLERS_DISPLACED)
    if displaced:
        issues.append(
            "another crash handler was installed over the shim's; re-armed, but "
            "crashes before the re-arm could not be recovered"
        )

    return issues


def shim_stderr_lines(stderr: str) -> list[str]:
    """Lines of *stderr* that are shim diagnostics (tag at line start)."""
    lines = []
    for line in stderr.splitlines():
        text = line.strip()
        if text.startswith(SHIM_STDERR_TAG):
            lines.append(text)
    return lines


def self_test_issues(target: str, edges: int | None, attach: Attach) -> list[str]:
    """Verdict of the startup self-test execution.

    *edges* is None when the mode cannot measure them (no SHM segment:
    ptrace, --no-shm), which is not evidence of a fault.
    """
    if attach is not Attach.EXPECTED or edges is None or edges > 0:
        return []
    return [
        f"self-test: {target} recorded no edges on its first execution "
        "-- the coverage map is not reaching the fuzzer (see any __afl_shim: "
        "line above); coverage guidance is inactive"
    ]


class ShimWatch:
    """Run-long dedup of shim issues from stderr and health counters.

    Example: the same ``__afl_shim: ... too small`` line on every execution
    yields one warning; CMPLOG_DROPPED going 0 -> 3 -> 900 yields one.
    """

    def __init__(self) -> None:
        self._seen_lines: set[str] = set()
        self._reported: set[ShimField] = set()

    def seen_count(self) -> int:
        """Distinct stderr lines remembered (bounded by STDERR_SEEN_MAX)."""
        return len(self._seen_lines)

    def stderr(self, text: str) -> list[str]:
        """New shim lines in *text*; each distinct line is returned once."""
        fresh = []
        for line in shim_stderr_lines(text):
            if line in self._seen_lines or len(self._seen_lines) >= STDERR_SEEN_MAX:
                continue
            self._seen_lines.add(line)
            fresh.append(line)
        return fresh

    def counters(self, counters: Sequence[int], attach: Attach) -> list[str]:
        """Issues for fields that turned bad since the last call, once each."""
        fresh = []
        for field in (ShimField.ATTACHED, *_PROBLEM_FIELDS):
            if field in self._reported:
                continue
            issue = _field_issue(counters, field, attach)
            if issue is None:
                continue
            self._reported.add(field)
            fresh.append(issue)
        return fresh


_PROBLEM_FIELDS = (
    ShimField.SEG_REJECTED,
    ShimField.CMPLOG_DROPPED,
    ShimField.STRAY_SIGNALS,
    ShimField.ABORTS_INTERCEPTED,
    ShimField.HANDLERS_DISPLACED,
)


def _field_issue(counters: Sequence[int], field: ShimField, attach: Attach) -> str | None:
    """health_issues() restricted to one field; None when it is healthy."""
    only = [0] * len(ShimField)
    only[ShimField.ATTACHED] = 1
    if field < len(counters):
        only[field] = counters[field]
    issues = health_issues(only, attach)
    return issues[0] if issues else None
