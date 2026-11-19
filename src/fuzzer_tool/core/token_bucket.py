"""Token bucket, and a per-bug throttle for sanitizer reports on the real stderr.

A recover-mode ASAN report written outside the capture window (e.g. by the
snapshot restore's memcpy) goes straight to the terminal: 3 reports per exec
put 106 MB on stderr in 10k execs. Reports are keyed by bug (header + top
frame, numbers masked); each key gets its own bucket::

    ==7==ERROR: AddressSanitizer: global-buffer-overflow on address 0x...
        #0 0x... in parse /src/a.c:12:3
    key: "==#==ERROR: AddressSanitizer: global-buffer-overflow on address #|#0 # in parse ..."

    burst REPORT_BURST, then one per 1/REPORT_RATE s; the rest are counted
    and announced with the next report that passes.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

# Reports of one bug passed at once, then the steady rate (per second).
REPORT_BURST = 4
REPORT_RATE = 0.1
# All bugs together: a key explosion still cannot flood.
TOTAL_BURST = 64
TOTAL_RATE = 1.0
# Distinct bugs tracked; the oldest key is dropped past this (Hard Rule 54).
MAX_REPORT_KEYS = 256

# Report boundary: "==PID==ERROR:" (ASAN/MSAN/LSAN) or "runtime error:" (UBSAN).
_HEADER_RE = re.compile(r"^(?:==\d+==ERROR:|.*runtime error:)", re.M)
_NUM_RE = re.compile(r"0x[0-9a-fA-F]+|\d+")
_FRAME0 = "#0 "


class TokenBucket:
    """``capacity`` tokens, refilled at ``rate`` per second; ``take`` spends one."""

    def __init__(self, capacity: float, rate: float, clock: Callable[[], float] = time.monotonic):
        self._capacity = capacity
        self._rate = rate
        self._clock = clock
        self._tokens = capacity
        self._stamp = clock()

    def take(self, n: float = 1.0) -> bool:
        """Spend ``n`` tokens if available. A clock that steps back refills nothing."""
        now = self._clock()
        elapsed = max(0.0, now - self._stamp)
        self._stamp = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

        if self._tokens < n:
            return False
        self._tokens -= n
        return True


def _split(text: str) -> list[str]:
    """Cut ``text`` at report headers; text before the first is its own part."""
    cuts = [m.start() for m in _HEADER_RE.finditer(text)]
    if not cuts or cuts[0] != 0:
        cuts.insert(0, 0)
    cuts.append(len(text))
    return [text[a:b] for a, b in zip(cuts, cuts[1:], strict=False) if a < b]


def _key(report: str) -> str:
    """Header line + top frame, numbers masked: one key per bug site."""
    head, _, rest = report.partition("\n")
    frame = ""
    i = rest.find(_FRAME0)
    if i >= 0:
        end = rest.find("\n", i)
        frame = rest[i : end if end >= 0 else None]
    return _NUM_RE.sub("#", head + "|" + frame)


class ReportThrottle:
    """Forward sanitizer text to ``sink``, at most a burst per bug."""

    def __init__(self, sink: Callable[[str], None], clock: Callable[[], float] = time.monotonic):
        self._sink = sink
        self._clock = clock
        self._total = TokenBucket(TOTAL_BURST, TOTAL_RATE, clock)
        # key -> (bucket, reports dropped since the last one passed)
        self._buckets: dict[str, list] = {}
        self.suppressed = 0

    def keys(self) -> int:
        return len(self._buckets)

    def emit(self, text: str) -> None:
        """Pass or count each report in ``text``."""
        if not text:
            return

        out = []
        for report in _split(text):
            passed = self._admit(report)
            if passed is not None:
                out.append(passed)
        if out:
            self._sink("".join(out))

    def _admit(self, report: str) -> str | None:
        """``report`` (with a dropped-count note) if its buckets allow it, else None."""
        entry = self._entry(_key(report))
        if not entry[0].take() or not self._total.take():
            entry[1] += 1
            self.suppressed += 1
            return None

        dropped, entry[1] = entry[1], 0
        if not dropped:
            return report
        return f"[fuzzer] {dropped} repeated reports of this bug suppressed\n{report}"

    def _entry(self, key: str) -> list:
        entry = self._buckets.get(key)
        if entry is not None:
            return entry

        if len(self._buckets) >= MAX_REPORT_KEYS:
            del self._buckets[next(iter(self._buckets))]
        entry = [TokenBucket(REPORT_BURST, REPORT_RATE, self._clock), 0]
        self._buckets[key] = entry
        return entry
