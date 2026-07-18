"""Target process execution adapter.

Uses blocking os.waitpid + watchdog thread instead of communicate(timeout=...)
to avoid CPython's busy-poll backoff (24% wall time in profiling).

Fast path: os.posix_spawn + temp file for maximum throughput (2000+ eps).
Standard path: subprocess.Popen + stdin pipe for general use.
"""

import contextlib
import os
import signal
import subprocess
import threading

SIGNAL_CRASH_CODES = {134, 135, 136, 139, -6, -7, -8, -11}  # SIGABRT/SIGBUS/SIGFPE/SIGSEGV

_child_pids: set[int] = set()


def _track(pid: int):
    _child_pids.add(pid)


def _untrack(pid: int):
    _child_pids.discard(pid)


_clean_env_cache: dict[str, str] | None = None


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


# Reusable temp file for fast-path execution (avoids per-iteration file creation)
_fast_path_fd = None
_fast_path_name = None


def _get_fast_path_file() -> str:
    """Get or create a reusable temp file for fast-path execution."""
    global _fast_path_fd, _fast_path_name
    if _fast_path_fd is None:
        import tempfile
        _fast_path_fd, _fast_path_name = tempfile.mkstemp(suffix=".bin", prefix="fuzz_")
    return _fast_path_name


def run_target_fast(target: str, data: bytes, env: dict[str, str] | None = None) -> tuple[int, str, int]:
    """Fast execution path using os.posix_spawn + temp file.

    Avoids thread creation, watchdog overhead, and stdin pipe buffering.
    Uses posix_spawn which is 3-4x faster than fork+exec for simple targets.

    Args:
        target: Path to target binary.
        data: Input data.
        env: Optional environment variables.

    Returns:
        Tuple of (returncode, stderr, pid).
    """
    fname = _get_fast_path_file()
    try:
        # Write data to temp file (reuse fd to avoid open/close overhead)
        os.lseek(_fast_path_fd, 0, os.SEEK_SET)
        os.write(_fast_path_fd, data)
        os.ftruncate(_fast_path_fd, len(data))

        e = _clean_env(env)
        pid = os.posix_spawn(target, [target, fname], e)
        _, status = os.waitpid(pid, 0)

        if os.WIFEXITED(status):
            rc = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            rc = -os.WTERMSIG(status)
        else:
            rc = -2
        return rc, "", pid
    except Exception as e:
        return -2, str(e), 0


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
) -> tuple[int, str, int]:
    """Execute target with data on stdin.

    Uses blocking os.waitpid + watchdog thread instead of
    communicate(timeout=...) to avoid CPython's busy-poll backoff.

    Args:
        target: Path to target binary.
        data: Input data.
        timeout: Timeout in seconds.
        env: Optional environment variables.

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

        # Write data in a thread to avoid pipe deadlock
        writer = threading.Thread(target=_write_and_close, args=(proc.stdin, data), daemon=True)
        writer.start()

        # Watchdog: kill process group if still alive after timeout.
        # `done` is set by the main thread the instant waitpid() returns, which
        # interrupts the watchdog's wait() immediately instead of it sleeping
        # for the full `timeout` regardless of how fast the child actually exited.
        # `timed_out` is set only by the watchdog itself, and only on a genuine
        # timeout — it's what the return value below actually checks.
        done = threading.Event()
        timed_out = threading.Event()

        def _watchdog():
            if done.wait(timeout=timeout):
                return  # main thread finished first — nothing to do
            timed_out.set()
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        w = threading.Thread(target=_watchdog, daemon=True)
        w.start()

        # Blocking wait — no busy-poll like communicate(timeout=)
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

        done.set()  # wake the watchdog immediately; no join needed — daemon thread
        _untrack(proc.pid)

        if timed_out.is_set():
            return -1, "timeout", proc.pid

        stderr = proc.stderr.read()
        return proc.returncode, stderr.decode(errors="replace"), proc.pid
    except Exception as e:
        # Return actual pid if process was created, else 0.
        # Callers use pid for dmesg filtering — wrong pid would
        # match kernel messages from the wrong process.
        real_pid = proc.pid if proc is not None else 0
        return -2, str(e), real_pid


def run_target_file(
    target: str,
    data: bytes,
    timeout: float,
    tmp_dir: str,
    target_args: list[str],
    env: dict[str, str] | None = None,
) -> tuple[int, str, int]:
    """Execute target with data written to a temp file.

    Uses blocking os.waitpid + watchdog thread. The watchdog uses
    Event.wait(timeout) so it returns instantly when the main thread
    signals completion, instead of sleeping for the full timeout.

    Args:
        target: Path to target binary.
        data: Input data.
        timeout: Timeout in seconds.
        tmp_dir: Temporary directory for input files.
        target_args: Target arguments ({file} is replaced with temp file path).
        env: Optional environment variables.

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

        # Blocking wait — no busy-poll like communicate(timeout=)
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
        # Return actual pid if process was created, else 0.
        # Callers use pid for dmesg filtering — wrong pid would
        # match kernel messages from the wrong process.
        real_pid = proc.pid if proc is not None else 0
        return -2, str(e), real_pid
