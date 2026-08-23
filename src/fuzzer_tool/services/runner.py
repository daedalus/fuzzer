"""Target execution and crash detection.

Extracted from Fuzzer class (~lines 784-1115). Contains:
- _run_target() — dispatches to appropriate execution backend
- _run_target_ptrace() — ptrace-based execution with breakpoint instrumentation
- _ptrace_handle_breakpoint() — handles SIGTRAP during ptrace execution
- _check_python_crashes() — detects Python process crashes
- _is_interesting() — checks if execution result is interesting
- _is_crash() — checks if execution result is a crash
"""

import contextlib
import ctypes
import logging
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time

from fuzzer_tool.adapters.process import (
    SIGNAL_CRASH_CODES,
    run_target_fast,
    run_target_file,
    run_target_stdin,
)
from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.services.ptrace_coverage import (
    PTRACE_CONT,
    PTRACE_GETREGS,
    PTRACE_GETSIGINFO,
    PTRACE_SETREGS,
    PTRACE_TRACEME,
    PtraceCoverage,
)

log = logging.getLogger(__name__)

# Ceiling on the wait for a freshly forked child's first ptrace stop, in
# seconds. Independent of the per-exec timeout, which the run loop spends in
# full; see _run_ptrace_coverage.
_INITIAL_STOP_TIMEOUT = 1.0


def _get_fault_addr(pid: int, libc) -> int | None:
    """Return the faulting address (si_addr) for the current ptrace stop.

    Only trusts kernel-reported hardware faults: SIGSEGV/SIGBUS/SIGILL/SIGFPE
    with si_code > 0 (user-raised signals carry si_code == SI_USER(0) or
    negative SI_* values and their si_addr is meaningless). For SIGSEGV and
    SIGBUS si_addr is the faulting memory address (NULL-deref vs wild-pointer
    classification); for SIGILL/SIGFPE it is the faulting instruction address.
    """
    buf = ctypes.create_string_buffer(128)  # sizeof(siginfo_t) == 128 on x86-64
    if libc.ptrace(PTRACE_GETSIGINFO, pid, None, ctypes.cast(buf, ctypes.c_void_p)) != 0:
        return None
    si_signo = struct.unpack_from("<i", buf.raw, 0)[0]
    si_code = struct.unpack_from("<i", buf.raw, 8)[0]
    if si_signo not in (signal.SIGSEGV, signal.SIGBUS, signal.SIGILL, signal.SIGFPE):
        return None
    if si_code <= 0:
        return None
    return struct.unpack_from("<Q", buf.raw, 16)[0]


def _capture_crash_state(pid: int, libc, fuzzer) -> None:
    """Stash fault address + registers from a fatal-signal stop on *fuzzer*."""
    fault = _get_fault_addr(pid, libc)
    if fault is not None:
        fuzzer._last_fault_addr = fault
    regs_buf = (ctypes.c_char * (27 * 8))()
    if libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf) == 0:
        regs = bytes(regs_buf)
        fuzzer._last_regs = {
            "rip": struct.unpack_from("<Q", regs, 128)[0],
            "rbp": struct.unpack_from("<Q", regs, 32)[0],
            "rsp": struct.unpack_from("<Q", regs, 152)[0],
        }


def _ptrace_report_timeout(pid: int) -> tuple[int, str]:
    """Kill and reap a tracee that outlived its deadline; report it as a timeout.

    Returns the ``(-1, "timeout")`` pair the rest of the tool treats as "hung".
    That sentinel is the contract: ``Fuzzer`` tests ``rc == -1`` to set
    ``is_timeout``, which in turn keeps the input out of ``crashes/``, out of
    signature dedup, and (under ``--metropolis``) out of the corpus.  The
    stderr text is supplied for the backends that match on it, but no caller
    should need it -- rc alone decides.

    The tracee is SIGKILLed rather than detached: it is stopped under ptrace
    with an unknown amount of work left, and PTRACE_DETACH would resume it as
    an orphan competing for the same CPU as the next execution.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, 0)
    return -1, "timeout"


def _write_and_close(fd: int, data: bytes) -> None:
    """Write *data* to *fd* then close it — designed to run in a thread."""
    try:
        os.write(fd, data)
    finally:
        try:
            os.close(fd)
        except OSError:
            log.debug("Failed to close fd %d (already closed?)", fd)


def ptrace_available() -> bool:
    """Return True if PTRACE_TRACEME works in this environment.

    Probes once by forking a child that TRACEMEs itself and exits 42 on
    failure (yama ptrace_scope can block attach/trace even for children).
    """
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.ptrace.argtypes = [
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    libc.ptrace.restype = ctypes.c_long
    pid = os.fork()
    if pid == 0:
        try:
            if libc.ptrace(PTRACE_TRACEME, 0, None, None) != 0:
                os._exit(42)
            os._exit(0)
        except BaseException:
            os._exit(127)
    try:
        _, status = os.waitpid(pid, os.WUNTRACED)
        return not (os.WIFEXITED(status) and os.WEXITSTATUS(status) in (42, 127))
    except ChildProcessError:
        return False
    finally:
        with contextlib.suppress(ProcessLookupError, ChildProcessError):
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)


class TargetRunner:
    """Manages target execution across multiple backends.

    Holds a reference to the Fuzzer instance for accessing shared state.
    """

    def __init__(self, fuzzer):
        self.f = fuzzer

    def run_target(self, data: bytes) -> tuple[int, str]:
        f = self.f
        # Resolve per-target SHM for multi-target mode
        shm = f._target_shm_covs.get(f.target, f.shm_cov) if f.multi_targets else f.shm_cov

        # Set up cmplog env BEFORE any execution path — covers inprocess,
        # persistent, ptrace, and subprocess runners. The cmplog shim
        # (whether LD_PRELOAD'd or compiled into the target .so) needs
        # _CMPLOG_OUT set before the target runs.
        if f._cmplog:
            f._cmplog.setup_env_for_run()

        if f._inprocess_runner:
            # Reset per-run crash state so stale values never leak into later
            # iterations' metadata (save_crash runs after run_target in fuzz_one).
            f._last_fault_addr = None
            f._last_regs = {}
            if shm and not f._inprocess_runner.direct_lite:
                shm.reset_edge_map()
            # Open perf counters on current process (pid=0, no extra perms needed)
            # so in-process target calls (direct/direct_lite) are counted.
            if f._perf_counters:
                f._perf_counters.open_for_pid(0)
            rc, err = f._inprocess_runner.run_one(data)
            f._last_fault_addr = f._inprocess_runner._last_fault_addr
            f._last_regs = f._inprocess_runner._last_regs or {}
            # In direct_lite mode the target writes directly to shm_cov's
            # SHM via __afl_area — no read_bitmap/memmove needed.
            # For other inprocess modes, copy from the runner's bitmap.
            if shm and not f._inprocess_runner.direct_lite:
                bitmap = f._inprocess_runner.read_bitmap()
                if bitmap and len(bitmap) <= shm.size:
                    ctypes.memmove(shm._ptr, bitmap, len(bitmap))
            # Note: cmplog log cleanup (truncation / __cmplog_reset) is
            # handled by fuzz_one() after collect_tokens() reads the data.
            # Read perf counters for inprocess mode
            if f._perf_counters:
                f._last_perf_deltas = f._perf_counters.read_and_reset()
            return rc, err

        if f._persistent_runner:
            rc, err = f._persistent_runner.run_one(data)
            if f._perf_counters:
                f._last_perf_deltas = f._perf_counters.read_and_reset()
            return rc, err

        if f._network_runner:
            # No reply is read on this socket — coverage comes from shm,
            # which we reset here so fuzz_one()'s post-run
            # is_new_coverage_with_edges() reflects only this iteration.
            if shm:
                shm.reset_edge_map()
            t0 = time.monotonic()
            rc, err = f._network_runner.run_one(data)
            elapsed = time.monotonic() - t0
            # Feed the elapsed wall-clock time as a proxy for target
            # processing latency.  The KF's filtered estimate drives
            # the adaptive settle in NetworkRunner._settle().
            if f._network_runner.settle_kf is not None:
                f._network_runner.settle_kf.predict(dt=1.0)
                f._network_runner.settle_kf.update(elapsed)
            if f._perf_counters:
                f._last_perf_deltas = f._perf_counters.read_and_reset()
            return rc, err

        if f.ptrace_cov:
            return self._run_target_ptrace(data)

        if f._forkserver and f._forkserver._ready:
            # No bitmap is copied back: the loader's exec'd child inherited
            # __AFL_SHM_ID and the shim's constructor attached to *this*
            # segment, so it wrote its edges here directly. Reset for the
            # same reason the spawn path below does — is_new_coverage_with_edges()
            # must see this execution only.
            if shm:
                shm.reset_edge_map()
            return f._forkserver.run_one(data)

        if shm:
            shm.reset_edge_map()

        env = os.environ.copy()
        if f.use_coverage:
            env["AFL_MAP_SIZE"] = str(f.map_size)
        if shm:
            env["__AFL_SHM_ID"] = shm.env_id
        if f._cmplog:
            env = f._cmplog.setup_env(env)

        # Fast path: posix_spawn + temp file (no threads, no watchdog)
        if not f.file_mode and not f._cmplog:
            rc, stderr, pid = run_target_fast(
                f.target, data, env=env, perf_counters=f._perf_counters, timeout=f.timeout
            )
            f._last_child_pid = pid
            if f._perf_counters:
                f._last_perf_deltas = f._perf_counters.read_and_reset()
            return rc, stderr

        if f.file_mode:
            rc, stderr, pid = run_target_file(
                f.target,
                data,
                f.timeout,
                str(f._tmp_dir),
                f.target_args,
                env=env,
                perf_counters=f._perf_counters,
            )
            f._last_child_pid = pid
            if f._perf_counters:
                f._last_perf_deltas = f._perf_counters.read_and_reset()
            return rc, stderr
        rc, stderr, pid = run_target_stdin(
            f.target, data, f.timeout, env=env, perf_counters=f._perf_counters
        )
        f._last_child_pid = pid
        if f._perf_counters:
            f._last_perf_deltas = f._perf_counters.read_and_reset()
        return rc, stderr

    def _ptrace_handle_breakpoint(self, pid: int, libc, cov: PtraceCoverage, regs_buf) -> bool:
        if not cov._is_x86_64:
            log.warning("ptrace coverage requires x86_64")
            return False
        libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf)
        rip = struct.unpack_from("<Q", bytes(regs_buf), 128)[0]
        bp_addr = rip - 1

        if bp_addr not in cov.original_bytes:
            libc.ptrace(PTRACE_CONT, pid, None, None)
            return True

        orig = cov.original_bytes[bp_addr]
        val = cov._read_memory(pid, bp_addr)
        cov._write_memory(pid, bp_addr, (val & ~0xFF) | orig)
        del cov.original_bytes[bp_addr]

        # x86-64 user_regs_struct offsets: rbp@32, rip@128, rsp@152
        # (128+48 would be gs_base, which is 0 for the main thread and
        # previously made this check always-false, skipping every
        # breakpoint's first instruction).
        rsp = struct.unpack_from("<Q", bytes(regs_buf), 152)[0]
        if rsp > 0x1000:
            cov._stack_initialized = True
            cov.record_edge(bp_addr)
            cov.discover_new_bbs(pid, bp_addr)
            regs_buf2 = (ctypes.c_char * (27 * 8))()
            libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf2)
            regs = bytearray(regs_buf2)
            struct.pack_into("<Q", regs, 128, bp_addr)
            libc.ptrace(PTRACE_SETREGS, pid, None, bytes(regs))
        libc.ptrace(PTRACE_CONT, pid, None, None)
        return True

    def _run_target_ptrace(self, data: bytes) -> tuple[int, str]:
        f = self.f
        cov = f.ptrace_cov
        cov.reset_edge_map()
        # Reset per-run crash state so stale values never leak into later
        # iterations' metadata (save_crash runs after run_target in fuzz_one).
        f._last_fault_addr = None
        f._last_regs = {}
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.ptrace.argtypes = [
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        libc.ptrace.restype = ctypes.c_long

        stdin_r, stdin_w = os.pipe()
        writer = None
        pid = os.fork()
        f._last_child_pid = pid
        if pid == 0:
            # Everything here runs in the forked child, which shares the
            # parent's entire Python stack (pytest included). If execv
            # raises without being caught, the exception unwinds back into
            # that inherited stack instead of exiting — the child then keeps
            # running as a second, orphaned copy of the parent process.
            # os._exit() must be unconditionally reached on any failure here.
            try:
                os.setsid()
                os.dup2(stdin_r, 0)
                os.close(stdin_r)
                os.close(stdin_w)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                os.close(devnull)
                ld_preload = os.environ.get("LD_PRELOAD", "")
                if ld_preload:
                    cleaned = [p for p in ld_preload.split(":") if "ksm_preload" not in p]
                    if cleaned:
                        os.environ["LD_PRELOAD"] = ":".join(cleaned)
                    else:
                        os.environ.pop("LD_PRELOAD", None)
                libc.ptrace(PTRACE_TRACEME, 0, None, None)
                signal.signal(signal.SIGTRAP, signal.SIG_IGN)
                os.execv(f.target, [f.target])
            except BaseException:
                os._exit(127)
            os._exit(127)

        os.close(stdin_r)
        writer = threading.Thread(target=_write_and_close, args=(stdin_w, data))
        writer.start()

        try:
            # Bounded wait for the initial ptrace stop. A blocking waitpid()
            # here can hang forever: os.fork() clones only the calling thread,
            # so if another thread (e.g. a prior run's _write_and_close writer)
            # holds the malloc or dynamic-loader lock at fork time, the child
            # can deadlock before reaching execv and never deliver SIGTRAP.
            # Poll against a deadline instead and treat expiry as a killed run.
            # (Deeper fix: posix_spawn / a pre-forked server so we never fork
            # from a multi-threaded process at all.)
            #
            # Bound this wait *separately* from the run loop below, which
            # starts its own full f.timeout. Charging f.timeout twice makes
            # the worst case per exec 2x the configured budget. Reaching the
            # first ptrace stop is fork+execv only — milliseconds — so a
            # one-second ceiling closes the hang without costing throughput
            # on a slow or overloaded box.
            initial_deadline = time.time() + min(f.timeout, _INITIAL_STOP_TIMEOUT)
            status = None
            while True:
                waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if waited != 0:
                    break
                if time.time() >= initial_deadline:
                    with contextlib.suppress(ProcessLookupError, ChildProcessError):
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
                    return -2, "exec timeout"
                time.sleep(0.0005)  # same poll interval as the loop below
            if os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGTRAP:
                pass
            elif os.WIFSTOPPED(status):
                sig = os.WSTOPSIG(status)
                _capture_crash_state(pid, libc, f)
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return -sig, ""
            elif os.WIFSIGNALED(status):
                return -os.WTERMSIG(status), ""
            elif os.WIFEXITED(status):
                return os.WEXITSTATUS(status), ""
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                return -2, "exec failed"

            cov.install_breakpoints(pid)
            libc.ptrace(PTRACE_CONT, pid, None, None)

            deadline = time.time() + f.timeout

            last_action = None
            last_sig = 0
            returncode = 0
            child_reaped = False
            # Deadline expiry is tracked explicitly rather than inferred
            # after the fact. `status` holds the LAST CONSUMED event, which
            # on expiry is whatever stop the tracee was last seen in -- so
            # the post-loop reconstruction below read a timeout as a crash:
            # with >=1 breakpoint handled, the stale SIGTRAP stop yielded
            # rc -5 ("crash signal 5"); with none, the SIGKILL path yielded
            # rc -9. `is_timeout` tests rc == -1, so it could never fire in
            # ptrace mode, and every slow input landed in crashes/ and took
            # a slot in signature dedup.
            timed_out = False
            while True:
                if time.time() >= deadline:
                    timed_out = True
                    break
                waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                # Check the PID, not just status: waitpid returns (0, 0) for
                # "no event" but (pid, 0) for a clean exit with rc=0 — both
                # have status == 0. Discarding the pid made every clean exit
                # look like "no event", then the next poll hit ECHILD and the
                # run was misreported as -2 ("exec failed").
                if waited == 0:
                    time.sleep(0.0005)
                    continue

                if os.WIFEXITED(status):
                    returncode = os.WEXITSTATUS(status)
                    child_reaped = True
                    break
                if os.WIFSIGNALED(status):
                    returncode = -os.WTERMSIG(status)
                    child_reaped = True
                    break

                if os.WIFSTOPPED(status):
                    sig = os.WSTOPSIG(status)
                    last_sig = sig
                    if sig == signal.SIGTRAP:
                        regs_buf = (ctypes.c_char * (27 * 8))()
                        if self._ptrace_handle_breakpoint(pid, libc, cov, regs_buf):
                            last_action = "cont"
                        else:
                            break
                    else:
                        # Fatal-signal stop (SIGSEGV/SIGBUS/...): capture the
                        # faulting address + registers before the tracee is
                        # reaped, since si_addr is unrecoverable after death.
                        _capture_crash_state(pid, libc, f)
                        break

            if timed_out:
                return _ptrace_report_timeout(pid)

            if child_reaped:
                pass
            elif last_action == "cont" and last_sig == signal.SIGTRAP:
                waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if waited != 0 and os.WIFSTOPPED(status):
                    libc.ptrace(PTRACE_CONT, pid, None, None)
                    # Bounded, not blocking. The tracee was just resumed with
                    # a blind PTRACE_CONT after its breakpoint handler failed;
                    # it may never stop again. `os.waitpid(pid, 0)` here hung
                    # the fuzzer indefinitely on such a target, past a
                    # deadline that had already been checked -- and swallowed
                    # any fatal signal delivered while it blocked.
                    resumed = False
                    while time.time() < deadline:
                        waited, st = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                        if waited != 0:
                            status = st
                            resumed = True
                            break
                        time.sleep(0.0005)
                    if not resumed:
                        return _ptrace_report_timeout(pid)
                elif waited != 0:
                    if os.WIFSIGNALED(status):
                        returncode = -os.WTERMSIG(status)
                    elif os.WIFEXITED(status):
                        returncode = os.WEXITSTATUS(status)
            else:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)

            if returncode == 0 and not child_reaped:
                if os.WIFSIGNALED(status):
                    returncode = -os.WTERMSIG(status)
                elif os.WIFEXITED(status):
                    returncode = os.WEXITSTATUS(status)
                elif os.WIFSTOPPED(status):
                    returncode = -os.WSTOPSIG(status)
                    _capture_crash_state(pid, libc, f)
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
            return returncode, ""

        except ChildProcessError:
            return -2, ""
        except Exception as e:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except Exception:
                log.debug("Failed to kill orphan pid %d", pid, exc_info=True)
            return -2, str(e)
        finally:
            if writer is not None:
                writer.join(timeout=f.timeout)

    def _run_triage_ptrace(self, data: bytes) -> tuple[int, str]:
        """Re-run *data* through a ptrace-attached loader script for crash triage.

        The ptrace runner execs the target binary, which is useless for .so
        targets — direct_lite crashes happen inside the fuzzer process with no
        fault info. This spawns the subprocess loader script self-traced
        (PTRACE_TRACEME), so a fatal signal stops the tracee before delivery
        and _capture_crash_state() can read si_addr + registers. Best-effort:
        if ptrace is blocked the script runs untraced and capture is skipped.
        """
        f = self.f
        runner = f._inprocess_runner
        f._last_fault_addr = None
        f._last_regs = {}
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.ptrace.argtypes = [
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        libc.ptrace.restype = ctypes.c_long

        # direct_lite never creates a loader script — write one on demand.
        if runner._loader_path is None:
            from fuzzer_tool.adapters.inprocess import _LOADER_SCRIPT

            fd, path = tempfile.mkstemp(suffix=".py", prefix="fuzz_loader_")
            with os.fdopen(fd, "w") as fh:
                fh.write(_LOADER_SCRIPT)
            runner._loader_path = path

        env = os.environ.copy()
        env["_PTRACE_TRACEME"] = "1"
        env["_TIMEOUT"] = str(int(f.timeout))
        proc = subprocess.Popen(
            [sys.executable, runner._loader_path, f.target, runner.function_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            proc.stdin.write(data)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        pid = proc.pid
        deadline = time.time() + f.timeout
        returncode = 0
        reaped = False
        try:
            while time.time() < deadline:
                waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if waited == 0:
                    time.sleep(0.0005)
                    continue
                if os.WIFEXITED(status):
                    returncode = os.WEXITSTATUS(status)
                    reaped = True
                    break
                if os.WIFSIGNALED(status):
                    returncode = -os.WTERMSIG(status)
                    reaped = True
                    break
                if os.WIFSTOPPED(status):
                    sig = os.WSTOPSIG(status)
                    if sig == signal.SIGTRAP:
                        # No exec happens after TRACEME — defensive only.
                        libc.ptrace(PTRACE_CONT, pid, None, None)
                        continue
                    _capture_crash_state(pid, libc, f)
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
                    return -sig, ""
            if not reaped:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                return -1, "timeout"
        except ChildProcessError:
            return -2, ""
        finally:
            stderr = b""
            try:
                if proc.stderr:
                    stderr = proc.stderr.read()
            except OSError:
                pass
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
        return returncode, stderr.decode(errors="replace")

    def verify_kernel_crash(self, child_pid: int | None) -> bool:
        return False

    def _check_python_crashes(self):
        pass

    def is_interesting(self, returncode: int, stderr: str) -> bool:
        f = self.f
        if returncode in SIGNAL_CRASH_CODES or returncode in f.extra_crash_codes:
            return True
        # -1 (timeout) and -2 (infrastructure failure) are adapter sentinels,
        # never target behavior; every other negative code is a fatal signal.
        if returncode < 0 and returncode not in (-1, -2):
            return True
        if returncode in (-1, 0) and ("ASAN" in stderr or "AddressSanitizer" in stderr):
            return True
        if "Segmentation fault" in stderr:
            return True
        return "Aborted" in stderr

    def is_crash(self, returncode: int, stderr: str) -> bool:
        f = self.f
        f.last_report = None
        if returncode in (-2, -1):
            return False

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            f.last_report = report
            return True

        if returncode in SIGNAL_CRASH_CODES or returncode in f.extra_crash_codes:
            return True
        if returncode < 0:
            return True
        return any(
            sig in stderr
            for sig in [
                "SIGSEGV",
                "SIGABRT",
                "SIGFPE",
                "SIGBUS",
                "Segmentation fault",
                "Aborted",
            ]
        )
