"""Target process execution adapter."""

import contextlib
import os
import signal
import subprocess

SIGNAL_CRASH_CODES = {1, 134, 135, 136, 139, -6, -7, -8, -11}  # exit(1)/SIGABRT/SIGBUS/SIGFPE/SIGSEGV

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
        cleaned = [p for p in ld.split(":") if "ksm_preload" not in p]
        if cleaned:
            e["LD_PRELOAD"] = ":".join(cleaned)
        else:
            e.pop("LD_PRELOAD", None)
    return e


def run_target_stdin(
    target: str,
    data: bytes,
    timeout: float,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Execute target with data on stdin.

    Args:
        target: Path to target binary.
        data: Input data.
        timeout: Timeout in seconds.
        env: Optional environment variables.

    Returns:
        Tuple of (returncode, stderr).
    """
    try:
        proc = subprocess.Popen(
            [target],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=_clean_env(env),
            preexec_fn=os.setsid,
        )
        _track(proc.pid)
        try:
            _, stderr = proc.communicate(input=data, timeout=timeout)
            return proc.returncode, stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            for _ in range(10):
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            return -1, "timeout"
        finally:
            _untrack(proc.pid)
    except Exception as e:
        return -2, str(e)


def run_target_file(
    target: str,
    data: bytes,
    timeout: float,
    tmp_dir: str,
    target_args: list[str],
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Execute target with data written to a temp file.

    Args:
        target: Path to target binary.
        data: Input data.
        timeout: Timeout in seconds.
        tmp_dir: Temporary directory for input files.
        target_args: Target arguments ({file} is replaced with temp file path).
        env: Optional environment variables.

    Returns:
        Tuple of (returncode, stderr).
    """
    from pathlib import Path

    tmp_file = Path(tmp_dir) / f"fuzz_{os.getpid()}"
    try:
        tmp_file.write_bytes(data)
        cmd = [target] + [a.replace("{file}", str(tmp_file)) for a in target_args]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=_clean_env(env),
            preexec_fn=os.setsid,
        )
        _track(proc.pid)
        try:
            _, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stderr.decode(errors="replace")
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            for _ in range(10):
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            return -1, "timeout"
        finally:
            _untrack(proc.pid)
            with contextlib.suppress(OSError):
                tmp_file.unlink()
    except Exception as e:
        return -2, str(e)
