"""Kernel log parser for crash verification.

Parses dmesg output to detect kernel-reported crashes (segfaults, traps,
OOM kills, etc.) that correlate with fuzzer-discovered inputs. Provides
timestamp tracking to match crash events with fuzzer execution timeline.

Uses ``dmesg -l err,warn,info --json`` for structured output with fallback
to text parsing when JSON is unavailable.

Real dmesg --json output shape (NOT NDJSON):
    {"dmesg": [{"pri": 4, "time": 12345.67, "msg": "...", ...}, ...]}

Each entry has fields: pri, time (boot-relative seconds), msg, fac, car,
pid, comm. The "time" field is seconds since boot, not epoch.
"""

import json
import logging
import re
import subprocess
import threading
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
_PID_IN_MSG_RE = re.compile(r"\w+\[(\d+)\]")
_DMESG_LEVELS = "err,warn,info"


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

    Note on timestamps: dmesg --json "time" field is boot-relative seconds,
    not epoch. For filtering, we use it as a monotonically increasing
    counter — absolute value doesn't matter, only ordering.
    """

    def __init__(self, boot_time: float | None = None):
        self._available: bool | None = None
        self._last_ts: float = boot_time or 0.0
        self._boot_time = boot_time
        self._warned = False
        # Async streaming state
        self._stream_proc: subprocess.Popen | None = None
        self._stream_thread: threading.Thread | None = None
        self._stream_buffer: list[KernelCrash] = []
        self._stream_lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if dmesg is accessible."""
        if self._available is not None:
            return self._available
        try:
            result = subprocess.run(
                ["dmesg", "-l", "err,warn,info", "--json"],
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

    def start_stream(self) -> bool:
        """Start async dmesg streaming in background thread.

        Uses text format (not --json) for reliable line-by-line parsing.
        Each line is ``[  123.456789] comm[pid]: message``.
        """
        if self._stream_proc is not None:
            return True
        if not self.is_available():
            return False
        try:
            self._stream_proc = subprocess.Popen(
                ["dmesg", "-l", _DMESG_LEVELS, "-w"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self._stream_thread = threading.Thread(
                target=self._stream_reader, daemon=True
            )
            self._stream_thread.start()
            log.info("dmesg async stream started")
            return True
        except (FileNotFoundError, PermissionError, OSError) as e:
            log.debug("Failed to start dmesg stream: %s", e)
            return False

    def stop_stream(self):
        """Stop the async dmesg stream."""
        if self._stream_proc is not None:
            self._stream_proc.terminate()
            try:
                self._stream_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._stream_proc.kill()
            self._stream_proc = None
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2)
            self._stream_thread = None
        log.info("dmesg async stream stopped")

    def _stream_reader(self):
        """Background thread: read dmesg text -w output and buffer crashes.

        Each line format: ``[  123.456789] comm[pid]: message``
        """
        proc = self._stream_proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            m = _TEXT_TIMESTAMP_RE.match(line)
            if not m:
                continue
            ts = float(m.group(1))
            msg = line[m.end() :].strip()
            # Extract PID from "comm[pid]: message" format
            pid = None
            proc_name = None
            pid_match = _PID_IN_MSG_RE.search(msg)
            if pid_match:
                pid = int(pid_match.group(1))
                proc_name = pid_match.group(0).split("[")[0]
            kc = self._match_crash(ts, msg, pid, proc_name)
            if kc:
                with self._stream_lock:
                    self._stream_buffer.append(kc)

    def _process_entry(self, entry: dict):
        """Process a single dmesg JSON entry."""
        msg = entry.get("msg", "") or entry.get("MESSAGE", "")
        pid = entry.get("pid") or entry.get("SYSLOG_PID")
        proc_name = entry.get("comm") or entry.get("SYSLOG_IDENTIFIER")
        # dmesg embeds PID in msg as "comm[pid]: message"
        if pid is None and not proc_name:
            m = _PID_IN_MSG_RE.search(msg)
            if m:
                pid = int(m.group(1))
                proc_name = m.group(0).split("[")[0]
        ts = self._parse_timestamp(entry)
        kc = self._match_crash(ts or time.time(), msg, pid, proc_name)
        if kc:
            with self._stream_lock:
                self._stream_buffer.append(kc)

    def drain_stream(self, pid: int | None = None) -> list[KernelCrash]:
        """Drain buffered crashes from the async stream.

        Args:
            pid: If provided, only return crashes attributed to this PID.

        Returns:
            List of kernel crashes since last drain.
        """
        with self._stream_lock:
            crashes = list(self._stream_buffer)
            self._stream_buffer.clear()
        if pid is not None:
            crashes = [kc for kc in crashes if kc.pid == pid]
        if crashes:
            self._last_ts = max(c.timestamp for c in crashes)
        return crashes

    def poll(self, since: float | None = None, pid: int | None = None) -> DmesgSnapshot:
        """Poll dmesg for new crash events since *since*.

        Args:
            since: Only return events after this timestamp. If None, uses
                   the last-polled timestamp. For JSON mode this is
                   boot-relative seconds; for text mode it's also
                   boot-relative (from ``[  123.45]`` brackets).
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
        """Parse dmesg --json output.

        Real output is a single JSON document: {"dmesg": [{...}, ...]}
        NOT NDJSON (one object per line).

        Returns None if JSON parsing fails (triggers text fallback).
        """
        try:
            result = subprocess.run(
                ["dmesg", "-l", "err,warn,info", "--json"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode != 0:
                return None

            raw = result.stdout.decode(errors="replace").strip()
            if not raw:
                return []

            doc = json.loads(raw)
            entries = doc.get("dmesg", [])
            if not isinstance(entries, list):
                return None  # unexpected format, fall back to text

            crashes = []
            for entry in entries:
                ts = self._parse_timestamp(entry)
                if ts is not None and ts <= since:
                    continue

                msg = entry.get("msg", "") or entry.get("MESSAGE", "")
                pid = entry.get("pid") or entry.get("SYSLOG_PID")
                proc = entry.get("comm") or entry.get("SYSLOG_IDENTIFIER")
                # dmesg embeds PID in msg as "comm[pid]: message"
                if pid is None and not proc:
                    m = _PID_IN_MSG_RE.search(msg)
                    if m:
                        pid = int(m.group(1))
                        proc = m.group(0).split("[")[0]  # "python3[123]" → "python3"
                kc = self._match_crash(ts or 0.0, msg, pid, proc)
                if kc:
                    crashes.append(kc)

            return crashes
        except (json.JSONDecodeError, ValueError):
            log.debug("dmesg --json produced unparseable output, falling back to text")
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return None

    def _poll_text(self, since: float) -> list[KernelCrash]:
        """Fallback: parse dmesg text output."""
        try:
            result = subprocess.run(
                ["dmesg", "-l", "err,warn,info"],
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
                msg = line[m.end() :].strip()
                kc = self._match_crash(ts, msg)
                if kc:
                    crashes.append(kc)
            return crashes
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return []

    def _parse_timestamp(self, entry: dict) -> float | None:
        """Extract boot-relative timestamp from a dmesg JSON entry.

        The "time" field is seconds since boot (monotonically increasing).
        Also checks legacy field names for compatibility.
        """
        for key in ("time", "ts", "__REALTIME_TIMESTAMP"):
            val = entry.get(key)
            if val is not None:
                try:
                    v = float(val)
                    # __REALTIME_TIMESTAMP is microseconds
                    if key == "__REALTIME_TIMESTAMP" and v > 1e12:
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
