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
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time

log = logging.getLogger(__name__)

# Signal numbers that indicate a crash (not a clean exit)
SIGNAL_CRASH_CODES = {134, 135, 136, 139, -6, -7, -8, -11}  # SIGABRT/SIGBUS/SIGFPE/SIGSEGV

# ── Shared child-process machinery ──────────────────────────────────────
#
# Used by all three execution modes. This lived under "Stdin mode" while only
# the Popen paths tracked their children; the fast path now does too, and a
# leaked pid from any mode is the same bug.

_TRACKED_PIDS: set[int] = set()
_TRACKED_PIDS_LOCK = threading.Lock()

# Cap on retained stderr, matching what this path used to read in one call.
# Bytes past the cap are still read and discarded -- the read is what keeps
# the child from blocking, so stopping early would reintroduce the deadlock.
_STDERR_CAP = 65536


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


def _kill_process_group(pid: int) -> None:
    """SIGKILL the child's process group, falling back to the child alone.

    killpg reaches anything the target spawned. It fails if the child already
    exited or never got its own group, which is not an error worth raising
    from a cleanup path.
    """
    with contextlib.suppress(OSError, ProcessLookupError):
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        return
    with contextlib.suppress(OSError, ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _drain_until_eof(fd: int, timeout: float | None) -> tuple[bytes, bool]:
    """Read *fd* until EOF or *timeout*, returning (data, timed_out).

    EOF means every writer closed the pipe, which for a spawned child means
    it exited. Polling the pipe is therefore both the stderr drain and the
    liveness wait, in one syscall per wakeup and no threads.

    Returns the bytes captured so far when the deadline expires; a timed-out
    target's partial stderr is still worth having for triage.
    """
    chunks: list[bytes] = []
    total = 0
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    deadline = None if timeout is None else time.monotonic() + timeout

    while True:
        if deadline is None:
            ready = poller.poll()
        else:
            remaining_ms = (deadline - time.monotonic()) * 1000.0
            if remaining_ms <= 0:
                return b"".join(chunks), True
            ready = poller.poll(remaining_ms)
        if not ready:
            return b"".join(chunks), True
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break  # EOF: the child closed its stderr, i.e. exited
        if total < _STDERR_CAP:
            chunks.append(chunk[: _STDERR_CAP - total])
            total += len(chunk)
    return b"".join(chunks), False


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
    Caches the result for repeated calls with the same env.

    ``None`` means "inherit the parent environment"; any dict -- *including an
    empty one* -- means "use exactly this". The two were conflated by
    ``env or os.environ``, which is falsy for ``{}``, so a caller asking for a
    scrubbed environment silently got the full parent env, LD_PRELOAD and all.
    That is the opposite of what this function exists to do.
    """
    global _clean_env_cache
    if env is None and _clean_env_cache is not None:
        return _clean_env_cache
    e = dict(os.environ if env is None else env)
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


# ── ASLR control ────────────────────────────────────────────────────────
#
# Every coverage identity in this fuzzer is compared ACROSS target
# processes: ShmCoverage._seen_edge_ids lives in the fuzzer parent and
# persists for the whole session, while the default execution path spawns
# a fresh process per input.  Anything derived from a runtime address must
# therefore be stable from one exec to the next.
#
# It is not, by default.  afl_shim.c's caller-context edge hashing
# (-D__AFL_CTX_SENSITIVE=1, enabled by tools/build_targets.sh for the
# _nosan .so targets and the whole --vendor-tracecmp path) computes
# edge_id = hash(__builtin_return_address(1)) ^ prev_loc ^ cur_loc.
# Under PIE + ASLR that hash changes on every exec, so every edge_id in
# every run looks new: is_new_coverage() never returns False, the edge
# table saturates immediately, and the corpus fills with inputs saved for
# coverage that does not exist.
#
# personality(ADDR_NO_RANDOMIZE) is inherited across fork AND preserved
# across execve, so setting it once in the fuzzer parent covers every
# child regardless of how it is launched -- posix_spawn (which has no
# preexec_fn hook), Popen, the in-process subprocess loader, and the
# forkserver alike.
#
# This also removes a source of run-to-run variance from every
# measurement in the tree, including tools/bench_paired.py.

ADDR_NO_RANDOMIZE = 0x0040000
_PERSONALITY_QUERY = 0xFFFFFFFF  # personality(0xffffffff) reads without setting

_aslr_disabled: bool | None = None


def disable_aslr() -> bool:
    """Disable address-space randomization for this process and its children.

    Idempotent and safe to call on any platform: returns False and logs at
    debug level when the syscall is unavailable or refused.

    Set FUZZER_KEEP_ASLR=1 to opt out.  The one situation that needs it is
    an ASAN target on a kernel whose fixed mmap layout collides with ASAN's
    shadow range ("Shadow memory range interleaves with an existing memory
    mapping").  If ASAN targets start failing to start immediately after
    this lands, that is the cause.

    Returns:
        True if ADDR_NO_RANDOMIZE is set on this process when we return.
    """
    global _aslr_disabled
    if _aslr_disabled is not None:
        return _aslr_disabled

    if os.environ.get("FUZZER_KEEP_ASLR") == "1":
        log.info("ASLR left enabled (FUZZER_KEEP_ASLR=1)")
        _aslr_disabled = False
        return False

    if not sys.platform.startswith("linux"):
        log.debug("ASLR not disabled: personality() is Linux-only (%s)", sys.platform)
        _aslr_disabled = False
        return False

    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        libc.personality.argtypes = [ctypes.c_ulong]
        libc.personality.restype = ctypes.c_int

        current = libc.personality(_PERSONALITY_QUERY)
        if current < 0:
            log.debug("ASLR not disabled: personality() query failed")
            _aslr_disabled = False
            return False
        if current & ADDR_NO_RANDOMIZE:
            _aslr_disabled = True
            return True

        # Mask off the query bits before OR-ing in our flag: personality()
        # returns the full persona word, and writing it back verbatim is
        # what we want plus ADDR_NO_RANDOMIZE.
        if libc.personality(ctypes.c_ulong(current | ADDR_NO_RANDOMIZE)) < 0:
            err = ctypes.get_errno()
            log.debug("ASLR not disabled: personality() set failed (errno %d)", err)
            _aslr_disabled = False
            return False

        # Verify rather than trust: seccomp filters and some container
        # runtimes return success and ignore the write.
        readback = libc.personality(_PERSONALITY_QUERY)
        if readback < 0 or not (readback & ADDR_NO_RANDOMIZE):
            log.debug("ASLR not disabled: personality() write did not take effect")
            _aslr_disabled = False
            return False
    except Exception as e:  # noqa: BLE001 - never let this abort a fuzzing run
        log.debug("ASLR not disabled: %s", e)
        _aslr_disabled = False
        return False

    log.info("ASLR disabled for this process and its children (ADDR_NO_RANDOMIZE)")
    _aslr_disabled = True
    return True


def aslr_disabled() -> bool | None:
    """Return the cached result of disable_aslr(), or None if never called."""
    return _aslr_disabled


def run_target_fast(
    target: str,
    data: bytes,
    env: dict[str, str] | None = None,
    perf_counters=None,
    timeout: float | None = None,
) -> tuple[int, str, int]:
    """Fast execution path using os.posix_spawn + temp file.

    Avoids thread creation, watchdog overhead, and stdin pipe buffering.
    Uses posix_spawn which is 3-4x faster than fork+exec for simple targets.

    If *perf_counters* is provided, perf_event_open is called on the child
    PID after spawn (before waitpid), so the caller can read execution
    metrics after the target exits.

    *timeout* bounds the run. ``None`` means unbounded, which is only
    appropriate for a caller that already knows the target terminates --
    every fuzzing caller must pass one. The bound is enforced by polling the
    stderr pipe rather than by a watchdog thread, because the whole point of
    this path is that it creates no threads, and because this process forks
    elsewhere (see tests/conftest.py on the multi-threaded-fork hazard).

    Draining stderr concurrently is not an optimisation either. The pipe holds
    64 KiB; a target that writes more blocks in ``write()`` while the parent
    blocks in ``waitpid()``, and neither ever wakes. Reading only after the
    reap -- as this did -- deadlocks on any sufficiently chatty target.

    Residual case, stated rather than papered over: a target that closes fd 2
    and *then* loops forever produces EOF without exiting, and the reap below
    blocks. Polling the reap instead would put a sleep on the hot path for
    every execution to cover a target that deliberately closes its own stderr.
    The common hang -- a target that loops without exiting -- never reaches
    EOF and is caught by the poll deadline.

    Args:
        target: Path to target binary.
        data: Input data.
        env: Optional environment variables.
        perf_counters: Optional PerfCounters instance (opens on child PID).
        timeout: Seconds before the child is killed, or None for unbounded.

    Returns:
        Tuple of (returncode, stderr, pid). rc is -1 on timeout and -2 on
        infrastructure failure, matching run_target_stdin/run_target_file.
    """
    fname = _get_fast_path_file()
    pid = 0
    stderr_r = -1
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
        # Own process group, so a timeout kill reaches anything the target
        # spawned. The sibling paths get this from preexec_fn=os.setsid.
        pid = os.posix_spawn(target, [target, fname], e, file_actions=file_actions, setpgroup=0)
        os.close(stdin_fd)
        os.close(stderr_w)
        _track(pid)

        # Open perf counters on the child PID while still running
        # (inherit=1 doesn't survive exec, so we attach directly).
        if perf_counters is not None and pid > 0:
            perf_counters.open_for_pid(pid)

        stderr_data, timed_out = _drain_until_eof(stderr_r, timeout)
        if timed_out:
            _kill_process_group(pid)

        _, status = os.waitpid(pid, 0)
        _untrack(pid)

        if timed_out:
            return -1, "timeout", pid

        if os.WIFEXITED(status):
            rc = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            rc = -os.WTERMSIG(status)
        else:
            rc = -2
        return rc, stderr_data.decode(errors="replace"), pid
    except Exception as e:
        # A spawned child must never outlive the call that failed, and the
        # caller needs the real pid: returning 0 lost crash attribution and
        # leaked the process.
        if pid > 0:
            _kill_process_group(pid)
            with contextlib.suppress(ChildProcessError, OSError):
                os.waitpid(pid, 0)
            _untrack(pid)
        return -2, str(e), pid
    finally:
        if stderr_r >= 0:
            with contextlib.suppress(OSError):
                os.close(stderr_r)


# ── Stdin mode ──────────────────────────────────────────────────────────


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
