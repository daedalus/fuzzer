"""Persistent subprocess loader for coverage-guided fuzzing.

Keeps one Python subprocess alive across iterations, communicating
via stdin/stdout pipes. Eliminates Python startup + ctypes.CDLL load
overhead on every iteration.

Protocol:
  Init:   "INIT <target> <func>\n"  ->  "READY\n"
  Run:    "RUN <len>\n<data>"       ->  "RC <rc> <bmp_len> <fault> <rip> <rsp> <rbp>"
  Quit:   "QUIT\n"

Coverage never travels this pipe: instrumented targets write SHM directly
(via __AFL_SHM_ID) and the caller reads the segment itself, so bmp_len is
always 0. The field is kept so older readers keep parsing.

Not to be confused with ``persistent_signal.py``, the other persistent mode
in this package. That one fork+execve's an executable and drives it with
SIGUSR1/SIGSTOP over a SysV segment; this one keeps a Python subprocess
alive and calls a ``.so`` through ctypes. They share three method names and
nothing else -- ``run_one`` does not even return the same type (``(rc,
bitmap)`` here, ``(rc, message)`` there) -- so they are not interchangeable
and have no common base.
"""

import collections
import contextlib
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

log = logging.getLogger(__name__)


def _close_streams(proc: subprocess.Popen) -> None:
    """Close the raw fds of *proc*, unblocking any thread stuck reading them.

    The fd is closed first because a drain thread blocked inside
    ``readline()`` holds the BufferedReader's own lock: calling
    ``stream.close()`` directly would spin on that lock forever
    (``_enter_buffered_busy`` at interpreter shutdown), while closing the fd
    underneath makes the blocked read fail with EBADF and the thread exit on
    its own.  The wrapper is then closed (so its destructor does not raise
    over the fd we stole) — except during interpreter finalization, where a
    frozen daemon thread may never release the lock.
    """
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(proc, name, None)
        fileno = getattr(stream, "fileno", None)
        if fileno is None:
            continue
        with contextlib.suppress(OSError, ValueError):
            os.close(fileno())
        if not sys.is_finalizing():
            with contextlib.suppress(Exception):
                stream.close()


_PERSISTENT_LOADER = r"""#!/usr/bin/env python3
import ctypes, ctypes.util, os, signal, sys

target = None
func = None
lib = None

def load_target(target_path, func_name):
    global target, func, lib
    lib = ctypes.CDLL(target_path)
    func = getattr(lib, func_name)
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

# --- ptrace fault-address capture (mirrors services/runner.py helpers) ---
PTRACE_TRACEME = 0
PTRACE_CONT = 7
PTRACE_GETREGS = 12
PTRACE_GETSIGINFO = 0x4202

_ptrace_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
_ptrace_libc.ptrace.restype = ctypes.c_long
_ptrace_libc.ptrace.argtypes = [
    ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p,
]


def _capture_ptrace_state(pid):
    # Return (fault_addr, regs) for a fatal-signal ptrace stop.
    # Mirrors services/runner.py:_get_fault_addr/_capture_crash_state - keep the
    # x86-64 siginfo/regs offsets in sync with that module (si_signo@0, si_code@8,
    # si_addr@16; rbp@32, rip@128, rsp@152).
    fault = None
    si_buf = ctypes.create_string_buffer(128)  # sizeof(siginfo_t) == 128
    if _ptrace_libc.ptrace(PTRACE_GETSIGINFO, pid, None, ctypes.cast(si_buf, ctypes.c_void_p)) == 0:
        si_signo = ctypes.c_int.from_buffer(si_buf, 0).value
        si_code = ctypes.c_int.from_buffer(si_buf, 8).value
        if (
            si_signo in (signal.SIGSEGV, signal.SIGBUS, signal.SIGILL, signal.SIGFPE)
            and si_code > 0
        ):
            fault = ctypes.c_uint64.from_buffer(si_buf, 16).value
    regs = None
    regs_buf = (ctypes.c_char * (27 * 8))()
    if _ptrace_libc.ptrace(PTRACE_GETREGS, pid, None, regs_buf) == 0:
        regs = {
            "rip": ctypes.c_uint64.from_buffer(regs_buf, 128).value,
            "rbp": ctypes.c_uint64.from_buffer(regs_buf, 32).value,
            "rsp": ctypes.c_uint64.from_buffer(regs_buf, 152).value,
        }
    return fault, regs


# Read init line
header = sys.stdin.buffer.readline().decode()
parts = header.strip().split()
if len(parts) >= 3 and parts[0] == "INIT":
    load_target(parts[1], parts[2])
    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()
else:
    sys.stdout.buffer.write(b"ERROR\n")
    sys.stdout.buffer.flush()
    sys.exit(1)

timeout_seconds = int(os.environ.get("_TIMEOUT", "5"))
# Track child PID file so outer layer can kill orphaned grandchild on timeout
child_pid_file = os.environ.get("_CHILD_PID_FILE", "")

# Main loop
while True:
    line = sys.stdin.buffer.readline()
    if not line:
        break
    cmd = line.decode().strip()
    if cmd == "QUIT":
        break
    if cmd.startswith("RUN "):
        data_len = int(cmd.split()[1])
        data = sys.stdin.buffer.read(data_len)
        buf = (ctypes.c_uint8 * len(data))(*data)

        # Fork child to enforce timeout on .so target calls.
        # Child calls os.setsid() to create its own process group, so the
        # outer layer can kill it (and any of its children) on timeout.
        # PEP 475 causes Python's os.waitpid to silently retry on EINTR,
        # making signal.alarm ineffective here — the actual timeout is
        # enforced by PersistentLoader.run_one's threaded readline.
        read_pipe, write_pipe = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            # Child: own process group, run target function, write rc
            os.close(read_pipe)
            os.setsid()

            # Best-effort ptrace self-trace so P1 (our direct parent) can read
            # the fault address + registers at the fatal-signal stop. Opt-in
            # via --ptrace (_PTRACE_ENABLE=1): adds per-exec overhead and can
            # be blocked by yama ptrace_scope. If ptrace is blocked or
            # disabled, the guarded call still reports the crash via rc —
            # capture just degrades gracefully.
            if os.environ.get("_PTRACE_ENABLE") == "1":
                try:
                    _ptrace_libc.ptrace(PTRACE_TRACEME, 0, None, None)
                except Exception:
                    pass

            # Use __afl_guarded_call when available — it uses
            # sigsetjmp/siglongjmp to survive signals (SIGSEGV,
            # SIGABRT, etc.) and returns a negative signal number.
            # This avoids the problem where Python catches the signal
            # as an exception and the crash code is silently lost.
            _guarded = getattr(lib, '__afl_guarded_call', None)
            if _guarded is not None:
                _guarded.restype = ctypes.c_int
                _guarded.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_size_t,
                ]
                _func_ptr_raw = ctypes.cast(func, ctypes.c_void_p)
                rc = _guarded(_func_ptr_raw, buf, len(data))
            else:
                try:
                    rc = func(buf, len(data))
                except Exception:
                    rc = -11

            # Flush cmplog buffer before exiting — os._exit() skips
            # destructors, so the buffered CMP lines would be lost.
            if hasattr(lib, '__tracecmp_flush'):
                try:
                    lib.__tracecmp_flush()
                except Exception:
                    pass
            # Negative rc from __afl_guarded_call indicates a crash
            # (the signal number negated). Encode as 128+signum so
            # the parent recognizes it as a crash (139 for SIGSEGV
            # is in SIGNAL_CRASH_CODES).
            if rc < 0:
                os.write(write_pipe, bytes([128 + (-rc)]))
            else:
                rc = max(0, min(rc, 125))
                os.write(write_pipe, bytes([rc]))
            os.close(write_pipe)
            os._exit(0)

        # Parent: track child PID for outer timeout cleanup
        os.close(write_pipe)
        if child_pid_file:
            try:
                with open(child_pid_file, "w") as f:
                    f.write(str(child_pid))
            except OSError:
                pass
        fault_addr = None
        regs = None
        try:
            # The child self-traced (PTRACE_TRACEME): a crash surfaces as a
            # WIFSTOPPED stop BEFORE the signal is delivered. Loop on stops —
            # capture the fault state, resume with the signal, and let the
            # final exit (or signal death) fall through below. Without the
            # loop, a traced crash landed in the else branch and misreported
            # rc=-2 after the outer timeout.
            while True:
                _, status = os.waitpid(child_pid, os.WUNTRACED)
                if not os.WIFSTOPPED(status):
                    break
                sig = os.WSTOPSIG(status)
                if sig in (
                    signal.SIGSEGV,
                    signal.SIGABRT,
                    signal.SIGBUS,
                    signal.SIGILL,
                    signal.SIGFPE,
                ):
                    fault_addr, regs = _capture_ptrace_state(child_pid)
                # Deliver the stop signal: a guarded call (siglongjmp) survives
                # it, otherwise the default disposition kills the child. SIGKILL
                # from the outer timeout cannot be intercepted and never stops here.
                _ptrace_libc.ptrace(PTRACE_CONT, child_pid, None, ctypes.c_void_p(sig))
            # Check if child was killed by signal (SIGSEGV, SIGABRT, etc.)
            if os.WIFSIGNALED(status):
                rc = -(os.WTERMSIG(status))
            elif os.WIFEXITED(status) and os.WEXITSTATUS(status) >= 128:
                # Exit via _exit(128 + sig) from crash handler (afl_shim.c).
                # Convert to negative signal code for consistency with the
                # WIFSIGNALED path above.
                rc = -(os.WEXITSTATUS(status) - 128)
            else:
                rc_byte = os.read(read_pipe, 1)
                rc = rc_byte[0] if rc_byte else -2
        except ChildProcessError:
            rc = -2
        os.close(read_pipe)
        # Clean up PID file
        if child_pid_file:
            try:
                os.unlink(child_pid_file)
            except OSError:
                pass

        # Relay captured fault state (if any) as trailing RC-line tokens:
        # "RC <rc> <bmp_len> <fault_addr> <rip> <rsp> <rbp>" ('-' when absent).
        fault_s = "-" if fault_addr is None else f"{fault_addr:#x}"
        rip_s = "-" if regs is None else f"{regs['rip']:#x}"
        rsp_s = "-" if regs is None else f"{regs['rsp']:#x}"
        rbp_s = "-" if regs is None else f"{regs['rbp']:#x}"
        resp = f"RC {rc} 0 {fault_s} {rip_s} {rsp_s} {rbp_s}\n".encode()
        sys.stdout.buffer.write(resp)
        sys.stdout.buffer.flush()
"""


class PersistentLoader:
    """Persistent subprocess — one process, many calls.

    Keeps a single Python subprocess alive. Each call loads the library
    once and calls the target function many times via stdin/stdout protocol.

    Timeout enforcement:
    - Loader script forks a child per call with os.setsid() (own process group)
    - Loader writes child PID to a temp file for outer-layer cleanup
    - run_one uses threaded readline with timeout
    - On timeout: kills loader + orphaned grandchild (via PID file or process tree)
    """

    def __init__(
        self,
        target: str,
        function_name: str = "LLVMFuzzerTestOneInput",
        timeout: float = 5.0,
        use_ptrace: bool = False,
    ):
        self.target = target
        self.function_name = function_name
        self.timeout = timeout
        # Off by default: PTRACE_TRACEME on every forked call adds per-exec
        # overhead and can be blocked by yama ptrace_scope. Opt in with
        # --ptrace when fault-address/register capture is worth the cost.
        self.use_ptrace = use_ptrace
        self._proc = None
        self._ready = False
        self._last_bitmap = None
        self._restarting = False
        self._child_pid_file: str | None = None

        # Fault-address/register relay from the last run (None/{} when absent)
        self._last_fault_addr: int | None = None
        self._last_regs: dict[str, int] = {}

        # Stderr buffer: drained stderr lines accumulated since last consume
        self._stderr_lock = threading.Lock()
        self._stderr_buffer: collections.deque[str] = collections.deque(maxlen=2000)

        # Throughput monitoring
        self._exec_times: list[float] = []  # rolling window of exec durations
        self._exec_window_size = 100  # track last N execs
        self._baseline_eps: float = 0.0  # calibrated execs/sec
        self._slow_restart_threshold = 0.10  # restart if below 10% of baseline

    def start(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True

        fd, self._loader_path = tempfile.mkstemp(suffix=".py", prefix="fuzz_persist_")
        os.write(fd, _PERSISTENT_LOADER.encode())
        os.close(fd)

        # Create PID file for grandchild tracking across timeouts
        pid_fd, self._child_pid_file = tempfile.mkstemp(suffix=".pid", prefix="fuzz_child_")
        os.close(pid_fd)

        env = os.environ.copy()
        if "AFL_MAP_SIZE" not in env:
            env["AFL_MAP_SIZE"] = "8192"
        env["_CHILD_PID_FILE"] = self._child_pid_file
        env["_PTRACE_ENABLE"] = "1" if self.use_ptrace else "0"

        self._proc = subprocess.Popen(
            [sys.executable, self._loader_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Drain stderr in background — prevents pipe-buffer deadlock when
        # ASAN/instrumented targets write diagnostic output to stderr.
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        init = f"INIT {self.target} {self.function_name}\n"
        self._proc.stdin.write(init.encode())
        self._proc.stdin.flush()

        resp = self._proc.stdout.readline()
        if resp.strip() == b"READY":
            self._ready = True
            log.info("Persistent loader started: %s", self.target)
            return True

        log.warning("Persistent loader failed to start")
        return False

    def _drain_stderr(self):
        """Buffer stderr output from the loader subprocess.

        Lines are stored in a ring buffer for per-call retrieval via
        consume_stderr(). Also logs first 200 chars of each line at debug
        level for diagnostics.
        """
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                text = line.decode(errors="replace").rstrip()
                if text:
                    log.debug("loader stderr: %s", text[:200])
                    with self._stderr_lock:
                        self._stderr_buffer.append(text + "\n")
        except (ValueError, OSError):
            pass

    def consume_stderr(self) -> str:
        """Return and clear stderr accumulated since the last call."""
        with self._stderr_lock:
            lines = list(self._stderr_buffer)
            self._stderr_buffer.clear()
        return "".join(lines)

    def run_one(self, data: bytes) -> tuple[int, bytes | None]:
        # Reset per-run crash state so a stale value never leaks from an
        # earlier run into this iteration's metadata.
        self._last_fault_addr = None
        self._last_regs = {}
        if not self._ready or not self._proc:
            return -2, None

        t_start = time.monotonic()
        cmd = f"RUN {len(data)}\n"
        try:
            self._proc.stdin.write(cmd.encode())
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            # Subprocess died — try restart
            self._ready = False
            if not self._restarting:
                self._restarting = True
                try:
                    if self.start():
                        return self.run_one(data)
                finally:
                    self._restarting = False
            return -2, None

        # Threaded readline with timeout — prevents hang if loader gets stuck
        result = [None]

        def _readline():
            result[0] = self._proc.stdout.readline()

        t = threading.Thread(target=_readline, daemon=True)
        t.start()
        t.join(timeout=self.timeout)
        if t.is_alive():
            log.warning("Persistent loader timed out after %.1fs", self.timeout)
            # Kill orphaned grandchild first (it's in its own process group)
            self._kill_orphaned_child()
            # Then kill the loader itself
            proc = self._proc
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait()
            _close_streams(proc)
            self._ready = False
            # Don't retry — hangs are input-deterministic, retrying just
            # costs another full timeout wait for the same hung input.
            # Restart the loader so future inputs work, but return immediately.
            if not self._restarting:
                self._restarting = True
                try:
                    self.start()
                finally:
                    self._restarting = False
            return -1, None

        header = result[0]
        if not header:
            return -2, None

        parts = header.decode().strip().split()
        if len(parts) < 3 or parts[0] != "RC":
            return -2, None

        rc = int(parts[1])
        bmp_len = int(parts[2])

        # Optional trailing tokens relayed by newer loaders:
        # <fault_addr> <rip> <rsp> <rbp> ('-' when absent).
        if len(parts) >= 7:

            def _hex_or_none(v: str) -> int | None:
                return None if v == "-" else int(v, 16)

            fault, rip, rsp, rbp = (_hex_or_none(p) for p in parts[3:7])
            self._last_fault_addr = fault
            if rip is not None or rsp is not None or rbp is not None:
                self._last_regs = {"rip": rip or 0, "rsp": rsp or 0, "rbp": rbp or 0}

        bitmap = None
        if bmp_len > 0:
            bitmap = self._proc.stdout.read(bmp_len)

        self._last_bitmap = bitmap

        # Track throughput
        elapsed = time.monotonic() - t_start
        self._exec_times.append(elapsed)
        if len(self._exec_times) > self._exec_window_size:
            self._exec_times.pop(0)

        # Calibrate baseline on first batch
        if self._baseline_eps == 0.0 and len(self._exec_times) >= self._exec_window_size:
            avg = sum(self._exec_times) / len(self._exec_times)
            if avg > 0:
                self._baseline_eps = 1.0 / avg

        # Check for sustained slowdown (only after calibration)
        if self._baseline_eps > 0 and len(self._exec_times) >= self._exec_window_size:
            recent_avg = sum(self._exec_times[-20:]) / min(20, len(self._exec_times))
            if recent_avg > 0:
                current_eps = 1.0 / recent_avg
                if current_eps < self._baseline_eps * self._slow_restart_threshold:
                    log.warning(
                        "Persistent loader throughput dropped: %.1f eps -> %.1f eps (%.0f%% of baseline)",
                        self._baseline_eps,
                        current_eps,
                        100 * current_eps / self._baseline_eps,
                    )
                    # Restart the loader. stop() first: start() early-returns
                    # True while the old process is still alive, so calling it
                    # without stopping would leave _ready False and wedge
                    # every later run_one at -2 forever.
                    self.stop()
                    self._exec_times.clear()
                    self._baseline_eps = 0.0
                    self.start()

        return rc, bitmap

    def stop(self):
        proc = self._proc
        self._proc = None
        self._ready = False
        self._kill_orphaned_child()
        if proc is None:
            return
        try:
            proc.stdin.write(b"QUIT\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            with contextlib.suppress(OSError, ValueError):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
        # Close the pipes. A grandchild that inherited a copy of the write
        # end would otherwise keep the pipe open forever, so the stderr-drain
        # thread never sees EOF and the process stays permanently
        # multi-threaded (poisoning every later os.fork()).
        _close_streams(proc)
        # Clean up PID file
        if self._child_pid_file:
            with contextlib.suppress(OSError):
                os.unlink(self._child_pid_file)
            self._child_pid_file = None

    def _kill_orphaned_child(self):
        """Kill the grandchild process (target function) if it was orphaned by timeout.

        The loader script writes the grandchild PID to a temp file before
        waiting. On timeout, we read that file and SIGKILL the process group.
        """
        if not self._child_pid_file:
            return
        try:
            with open(self._child_pid_file) as f:
                child_pid = int(f.read().strip())
            # Kill the process group (grandchild called os.setsid())
            os.killpg(child_pid, signal.SIGKILL)
        except (OSError, ValueError, ProcessLookupError):
            pass  # child already dead or PID file missing

    def __del__(self):
        self.stop()
