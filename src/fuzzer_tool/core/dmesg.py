"""Kernel log parser for crash verification.

Parses dmesg output to detect kernel-reported crashes (segfaults, traps,
OOM kills, etc.) that correlate with fuzzer-discovered inputs. Provides
timestamp tracking to match crash events with fuzzer execution timeline.

Uses ``dmesg -l err,warn --json`` for structured output with fallback
to text parsing when JSON is unavailable.
"""

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_CRASH_PATTERNS = [
    re.compile(
        r"segfault at (?P<addr>[0-9a-f]+) ip (?P<ip>[0-9a-f]+) "
        r"sp (?P<sp>[0-9a-f]+) error (?P<error>\d+)"
    ),
    re.compile(r"trap (?:fault|divide error|overflow|bounds|opcode)"),
    re.compile(r"general protection fault"),
    re.compile(r"Kernel panic"),
    re.compile(r"Out of memory:"),
    re.compile(r"BUG:"),
    re.compile(r"Unable to handle kernel"),
    re.compile(r"internal error:"),
    re.compile(r"KASAN:"),
]

_TEXT_TIMESTAMP_RE = re.compile(r"^\[\s*(\d+\.\d+)\]")


@dataclass
class KernelCrash:
    """A kernel-reported crash event."""

    timestamp: float
    raw_message: str
    pid: int | None = None
    process_name: str | None = None
    crash_type: str = ""
    ip: str | None = None
    sp: str | None = None
    error_code: int | None = None


@dataclass
class DmesgSnapshot:
    """A snapshot of kernel crash events captured at a point in time."""

    timestamp: float = field(default_factory=time.time)
    crashes: list[KernelCrash] = field(default_factory=list)
    available: bool = True
    error: str | None = None


class DmesgParser:
    """Parse dmesg output for kernel-reported crashes.

    Tracks the last-read timestamp so each poll only returns new events.
    Falls back to text parsing when ``--json`` output is unavailable.
    """

    def __init__(self, boot_time: float | None = None):
        self._available: bool | None = None
        self._last_ts: float = boot_time or 0.0
        self._boot_time = boot_time
        self._warned = False

    def is_available(self) -> bool:
        """Check if dmesg is accessible."""
        if self._available is not None:
            return self._available
        try:
            result = subprocess.run(
                ["dmesg", "-l", "err,warn", "--json"],
                capture_output=True,
                timeout=2,
            )
            self._available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            self._available = False
        if not self._available and not self._warned:
            log.warning(
                "dmesg not available — kernel crash verification disabled "
                "(requires root or CAP_SYSLOG)"
            )
            self._warned = True
        return self._available

    def poll(self, since: float | None = None, pid: int | None = None) -> DmesgSnapshot:
        """Poll dmesg for new crash events since *since* (epoch seconds).

        Args:
            since: Only return events after this timestamp. If None, uses
                   the last-polled timestamp.
            pid: If provided, only return crashes attributed to this PID.

        Returns:
            DmesgSnapshot with any new kernel crashes found.
        """
        if not self.is_available():
            return DmesgSnapshot(available=False, error="dmesg not available")

        since = since if since is not None else self._last_ts
        snap = DmesgSnapshot()

        # Try JSON first, fall back to text
        crashes = self._poll_json(since)
        if crashes is None:
            crashes = self._poll_text(since)

        # Filter by PID if requested
        if pid is not None and crashes:
            crashes = [kc for kc in crashes if kc.pid == pid]

        snap.crashes = crashes
        if crashes:
            self._last_ts = max(c.timestamp for c in crashes)
            snap.timestamp = self._last_ts
        return snap

    def _poll_json(self, since: float) -> list[KernelCrash] | None:
        """Parse dmesg --json output. Returns None if format unavailable."""
        try:
            result = subprocess.run(
                ["dmesg", "-l", "err,warn", "--json"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode != 0:
                return None

            data = result.stdout.decode(errors="replace").strip()
            if not data:
                return []

            # dmesg --json may produce concatenated JSON objects or NDJSON
            crashes = []
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = self._parse_timestamp(entry)
                if ts is not None and ts <= since:
                    continue

                msg = entry.get("msg", "") or entry.get("MESSAGE", "")
                pid = entry.get("pid") or entry.get("SYSLOG_PID")
                proc = entry.get("comm") or entry.get("SYSLOG_IDENTIFIER")

                kc = self._match_crash(ts or 0.0, msg, pid, proc)
                if kc:
                    crashes.append(kc)

            return crashes
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return None

    def _poll_text(self, since: float) -> list[KernelCrash]:
        """Fallback: parse dmesg text output."""
        try:
            result = subprocess.run(
                ["dmesg", "-l", "err,warn"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode != 0:
                return []

            crashes = []
            for line in result.stdout.decode(errors="replace").splitlines():
                m = _TEXT_TIMESTAMP_RE.match(line)
                if not m:
                    continue
                ts = float(m.group(1))
                if self._boot_time and ts < self._boot_time:
                    continue
                if ts <= since:
                    continue
                msg = line[m.end():].strip()
                kc = self._match_crash(ts, msg)
                if kc:
                    crashes.append(kc)
            return crashes
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return []

    def _parse_timestamp(self, entry: dict) -> float | None:
        """Extract timestamp from a dmesg JSON entry."""
        for key in ("ts", " TIMESTAMP", "__REALTIME_TIMESTAMP"):
            val = entry.get(key)
            if val is not None:
                try:
                    v = float(val)
                    # __REALTIME_TIMESTAMP is microseconds
                    if v > 1e12:
                        return v / 1_000_000.0
                    return v
                except (ValueError, TypeError):
                    continue
        return None

    def _match_crash(
        self,
        ts: float,
        msg: str,
        pid: int | None = None,
        proc: str | None = None,
    ) -> KernelCrash | None:
        """Match a dmesg line against known crash patterns."""
        for pattern in _CRASH_PATTERNS:
            m = pattern.search(msg)
            if m:
                kc = KernelCrash(
                    timestamp=ts,
                    raw_message=msg,
                    pid=pid,
                    process_name=proc,
                )
                # Classify crash type
                if "segfault" in msg:
                    kc.crash_type = "segfault"
                    kc.ip = m.groupdict().get("ip")
                    kc.sp = m.groupdict().get("sp")
                    err = m.groupdict().get("error")
                    if err is not None:
                        kc.error_code = int(err, 16) if err.startswith("0x") else int(err)
                elif "trap" in msg:
                    kc.crash_type = "trap"
                elif "general protection" in msg:
                    kc.crash_type = "gp_fault"
                elif "Kernel panic" in msg:
                    kc.crash_type = "kernel_panic"
                elif "Out of memory" in msg:
                    kc.crash_type = "oom"
                elif "KASAN" in msg:
                    kc.crash_type = "kasan"
                elif "BUG:" in msg:
                    kc.crash_type = "bug"
                else:
                    kc.crash_type = "other"
                return kc
        return None
