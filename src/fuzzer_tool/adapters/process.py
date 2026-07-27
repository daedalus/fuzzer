"""Subprocess execution adapters.

Provides low-level process spawning and management for running target binaries
during fuzzing. Supports three execution modes:
- Fast path: posix_spawn with temp file input (no threads)
- Stdin mode: Popen with stdin pipe + watchdog thread
- File mode: Popen with temp file input + watchdog thread

Also supports optional hardware performance counter tracking on child processes.
"""

import contextlib
import logging
import os
import signal
import subprocess
import tempfile
import threading

log = logging.getLogger(__name__)

# Signal numbers that indicate a crash (not a clean exit)
SIGNAL_CRASH_CODES = {134, 135, 136, 139, -6, -7, -8, -11}  # SIGABRT/SIGBUS/SIGFPE/SIGSEGV

# ── Fast path (posix_spawn) ─────────────────────────────────────────────

# Reusable temp file for fast path (avoid per-iteration file creation)
_fast_path_fd: int | None = None
_fast_path_name: str | None = None
_fast_path_lock = threading.Lock()


def _get_fast_path_file() -> str:
    """Get or create the reusable fast-path temp file."""
    global _fast_path_fd, _fast_path_name
    if _fast_path_fd is None:
        with _fast_path_lock:
            if _fast_path_fd is None:
                _fast_path_fd, _fast_path_name = tempfile.mkstemp(prefix="fuzz_fast_")
    return _fast_path_name


def _clean_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Copy env and strip LD_PRELOAD entries that conflict with sanitizers.
    Caches the result for repeated calls with the same env."""
    global _clean_env_cache
    if env is None and _clean_env_cache is not None:
        return _clean_env_cache
    e = dict(env or os.environ)
    ld = e.get("LD_PRELOAD", "")
    if ld:
        import re

        cleaned = [p for p in re.split(r"[:\s]+", ld) if p and "ksm_preload" not in p]
        if cleaned:
            e["LD_PRELOAD"] = ":".join(cleaned)
        else:
            e.pop("LD_PRELOAD", None)
    if env is None:
        _clean_env_cache = e
    return e


_clean_env_cache: dict[str, str] | None = None


def run_target_fast(
    target: str,
    data: bytes,
    env: dict[str, str] | None = None,
    perf_counters=None,
) -> tuple[int, str, int]:
    """Fast execution path using os.posix_spawn + temp file.

    Avoids thread creation, watchdog overhead, and stdin pipe buffering.
    Uses posix_spawn which is 3-4x faster than fork+exec for simple targets.

    If *perf_counters* is provided, perf_event_open is called on the child
    PID after spawn (before waitpid), so the caller can read execution
    metrics after the target exits.

    Args:
        target: Path to target binary.
        data: Input data.
        env: Optional environment variables.
        perf_counters: Optional PerfCounters instance (opens on child PID).

    Returns:
        Tuple of (returncode, stderr, pid).
    """
    fname = _get_fast_path_file()
    try:
        # Write data to temp file (reuse fd to avoid open/close overhead)
        os.lseek(_fast_path_fd, 0, os.SEEK_SET)
        os.write(_fast_path_fd, data)
        os.ftruncate(_fast_path_fd, len(data))
        os.fsync(_fast_path_fd)

        e = _clean_env(env)
        # Redirect stdin from the temp file so targets that read(0, ...) get the data.
        stdin_fd = os.open(fname, os.O_RDONLY)
        # Capture stderr via pipe for ASAN/sanitizer output.
        stderr_r, stderr_w = os.pipe()
        file_actions = (
            (os.POSIX_SPAWN_DUP2, stdin_fd, 0),
            (os.POSIX_SPAWN_DUP2, stderr_w, 2),
        )
        pid = os.posix_spawn(target, [target, fname], e, file_actions=file_actions)
        os.close(stdin_fd)
        os.close(stderr_w)

        # Open perf counters on the child PID while still running
        # (inherit=1 doesn't survive exec, so we attach directly).
        if perf_counters is not None and pid > 0:
            perf_counters.open_for_pid(pid)

        _, status = os.waitpid(pid, 0)

        # Read stderr after child exits (data is in kernel pipe buffer)
        try:
            stderr_data = os.read(stderr_r, 65536)
        except OSError:
            stderr_data = b""
        os.close(stderr_r)

        if os.WIFEXITED(status):
            rc = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            rc = -os.WTERMSIG(status)
        else:
            rc = -2
        return rc, stderr_data.decode(errors="replace"), pid
    except Exception as e:
        return -2, str(e), 0


# ── Stdin mode ──────────────────────────────────────────────────────────

_TRACKED_PIDS: set[int] = set()
_TRACKED_PIDS_LOCK = threading.Lock()


def _track(pid: int) -> None:
    """Track a child PID for cleanup on fatal signals."""
    with _TRACKED_PIDS_LOCK:
        _TRACKED_PIDS.add(pid)


def _untrack(pid: int) -> None:
    """Stop tracking a (now-reaped) child PID."""
    with _TRACKED_PIDS_LOCK:
        _TRACKED_PIDS.discard(pid)


def _child_pids() -> list[int]:
    """Return a snapshot of tracked child PIDs."""
    with _TRACKED_PIDS_LOCK:
        return list(_TRACKED_PIDS)


def _write_and_close(stream, data: bytes):
    """Write data to a stream and close it, ignoring errors."""
    try:
        stream.write(data)
        stream.close()
    except (BrokenPipeError, OSError):
        pass


def run_target_stdin(
    target: str,
    data: bytes,
    timeout: float,
    env: dict[str, str] | None = None,
    perf_counters=None,
) -> tuple[int, str, int]:
    """Execute target with data on stdin.

    Uses blocking os.waitpid + watchdog thread instead of
    communicate(timeout=...) to avoid CPython's busy-poll backoff.

    Args:
        target: Path to target binary.
        data: Input data.
        timeout: Timeout in seconds.
        env: Optional environment variables.
        perf_counters: Optional PerfCounters instance (opens on child PID).

    Returns:
        Tuple of (returncode, stderr, subprocess_pid).
    """
    try:
        proc = None
        proc = subprocess.Popen(
            [target],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=_clean_env(env),
            preexec_fn=os.setsid,
        )
        _track(proc.pid)

        # Open perf counters on child PID immediately after spawn
        if perf_counters is not None and proc.pid > 0:
            perf_counters.open_for_pid(proc.pid)

        # Write data in a thread to avoid pipe deadlock
        writer = threading.Thread(target=_write_and_close, args=(proc.stdin, data), daemon=True)
        writer.start()

        # Watchdog: kill process group if still alive after timeout.
        done = threading.Event()
        timed_out = threading.Event()

        def _watchdog():
            if done.wait(timeout=timeout):
                return
            timed_out.set()
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        w = threading.Thread(target=_watchdog, daemon=True)
        w.start()

        # Blocking wait
        try:
            _, status = os.waitpid(proc.pid, 0)
            if os.WIFEXITED(status):
                proc.returncode = os.WEXITSTATUS(status)
            elif os.WIFSIGNALED(status):
                proc.returncode = -os.WTERMSIG(status)
            else:
                proc.returncode = -2
        except ChildProcessError:
            proc.returncode = -2

        done.set()  # wake the watchdog immediately
        _untrack(proc.pid)

        if timed_out.is_set():
            return -1, "timeout", proc.pid

        stderr = proc.stderr.read()
        return proc.returncode, stderr.decode(errors="replace"), proc.pid
    except Exception as e:
        real_pid = proc.pid if proc is not None else 0
        return -2, str(e), real_pid


# ── File mode ───────────────────────────────────────────────────────────


def run_target_file(
    target: str,
    data: bytes,
    timeout: float,
    tmp_dir: str,
    target_args: list[str],
    env: dict[str, str] | None = None,
    perf_counters=None,
) -> tuple[int, str, int]:
    """Execute target with data written to a temp file.

    Uses blocking os.waitpid + watchdog thread.

    Args:
        target: Path to target binary.
        data: Input data.
        timeout: Timeout in seconds.
        tmp_dir: Temporary directory for input files.
        target_args: Target arguments ({file} is replaced with temp file path).
        env: Optional environment variables.
        perf_counters: Optional PerfCounters instance (opens on child PID).

    Returns:
        Tuple of (returncode, stderr, subprocess_pid).
    """
    from pathlib import Path

    tmp_file = Path(tmp_dir) / f"fuzz_{os.getpid()}"
    try:
        proc = None
        tmp_file.write_bytes(data)
        if target_args:
            cmd = [target] + [a.replace("{file}", str(tmp_file)) for a in target_args]
        else:
            cmd = [target, str(tmp_file)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=_clean_env(env),
            preexec_fn=os.setsid,
        )
        _track(proc.pid)

        # Open perf counters on child PID immediately after spawn
        if perf_counters is not None and proc.pid > 0:
            perf_counters.open_for_pid(proc.pid)

        # Watchdog
        done = threading.Event()
        timed_out = threading.Event()

        def _watchdog():
            if done.wait(timeout=timeout):
                return
            timed_out.set()
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        w = threading.Thread(target=_watchdog, daemon=True)
        w.start()

        try:
            _, status = os.waitpid(proc.pid, 0)
            if os.WIFEXITED(status):
                proc.returncode = os.WEXITSTATUS(status)
            elif os.WIFSIGNALED(status):
                proc.returncode = -os.WTERMSIG(status)
            else:
                proc.returncode = -2
        except ChildProcessError:
            proc.returncode = -2

        done.set()
        _untrack(proc.pid)
        with contextlib.suppress(OSError):
            tmp_file.unlink()

        if timed_out.is_set():
            return -1, "timeout", proc.pid

        stderr = proc.stderr.read()
        return proc.returncode, stderr.decode(errors="replace"), proc.pid
    except Exception as e:
        real_pid = proc.pid if proc is not None else 0
        return -2, str(e), real_pid
