"""Shim health: interpret afl_shim.c's markers and runtime counters.

Two sources, one vocabulary of human-readable issues:

    target ELF ── marker scan ──> stale_shim_issues()   (before any exec)
    loaded .so ── __afl_shim_health() ──> health_issues() (after a run)

Reading the counters needs a loaded library, which is an adapter concern
(``adapters.inprocess.read_shim_health``); this module only judges them.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, IntEnum

from fuzzer_tool.core.elf import detect_scoped_crash_handler


class ShimField(IntEnum):
    """Index of each counter in ``__afl_shim_health()``'s output (C ABI)."""

    ATTACHED = 0
    MAP_ENTRIES = 1
    SEG_REJECTED = 2
    CMPLOG_DROPPED = 3
    STRAY_SIGNALS = 4


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

    return issues
