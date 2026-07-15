"""Target process execution adapter.

Uses blocking os.waitpid + watchdog thread instead of communicate(timeout=...)
to avoid CPython's busy-poll backoff (24% wall time in profiling).
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


def _clean_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Copy env and strip LD_PRELOAD entries that conflict with sanitizers."""
    e = dict(env or os.environ)
    ld = e.get("LD_PRELOAD", "")
    if ld:
        import re

        cleaned = [p for p in re.split(r"[:\s]+", ld) if p and "ksm_preload" not in p]
        if cleaned:
            e["LD_PRELOAD"] = ":".join(cleaned)
        else:
            e.pop("LD_PRELOAD", None)
    return e


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
