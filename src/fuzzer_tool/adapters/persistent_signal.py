"""Persistent mode adapter for custom SIGUSR1-handling targets.

IMPORTANT: This does NOT implement the standard AFL persistent mode protocol
(fd 198/199 status pipe). Standard AFL-instrumented targets using __AFL_LOOP()
communicate via a status pipe, not signals. This runner is for CUSTOM targets
that explicitly handle SIGUSR1 and read from __AFL_SHM_ID shared memory.

For standard AFL persistent targets, use afl-showmap -P or AFL++'s own
persistent mode support instead.

Not to be confused with ``persistent_subprocess.py``, the other persistent
mode in this package. That one keeps a Python subprocess alive and calls a
``.so`` through ctypes over a pipe protocol; this one fork+execve's an
executable and drives it with signals over a SysV segment. They share three
method names and nothing else -- ``run_one`` does not even return the same
type (``(rc, message)`` here, ``(rc, bitmap)`` there) -- so they are not
interchangeable and have no common base.

Protocol:
  - Runner writes [4-byte length][input data] to SHM
  - Runner sends SIGUSR1 to the target
  - Target processes input, writes result, sends SIGSTOP (via __AFL_LOOP exit)
  - Runner reads result from SHM, resumes target with SIGCONT
"""

import contextlib
import ctypes
import logging
import os
import signal
import struct
import tempfile
import time

from fuzzer_tool.adapters import libc_shm
from fuzzer_tool.adapters.process import signal_rc

log = logging.getLogger(__name__)

IPC_RMID = libc_shm.IPC_RMID
IPC_PRIVATE = libc_shm.IPC_PRIVATE

# Most stderr kept per iteration; a sanitizer report fits well inside this.
_STDERR_TAIL = 64 * 1024


class PersistentRunner:
    """Run a custom SIGUSR1-handling target in persistent mode.

    Uses IPC_PRIVATE shared memory (no key collisions across workers).
    Target must: read input from __AFL_SHM_ID, handle SIGUSR1, SIGSTOP on
    each iteration boundary.
    """

    HEADER_SIZE = 8  # 4 bytes len + 4 bytes padding

    def __init__(
        self, target: str, timeout: float = 5.0, map_size: int = 65536, perf_counters=None
    ):
        self.target = target
        # Attached once per started target; each iteration is a delta.
        self._perf_counters = perf_counters
        self.timeout = timeout
        self.map_size = map_size
        self.pid: int | None = None
        self.shm_id: int | None = None
        self.shm_ptr: int = 0
        self._started = False
        # Target stderr lands here, not in /dev/null: an ASAN report followed
        # by exit(1) (abort_on_error=0) is otherwise indistinguishable from
        # a clean exit, since the exit code alone carries no crash signal.
        self._err = None
        self._err_off = 0

    def _read_stderr(self) -> str:
        """Stderr the target wrote since the current iteration began."""
        if self._err is None:
            return ""
        try:
            size = os.fstat(self._err.fileno()).st_size
            start = max(self._err_off, size - _STDERR_TAIL)
            if size <= start:
                return ""
            return os.pread(self._err.fileno(), size - start, start).decode(errors="replace")
        except OSError:
            return ""

    def start(self) -> bool:
        if self._started:
            return True

        # Use IPC_PRIVATE to avoid key collisions between concurrent runs
        self.shm_id = libc_shm.shmget(self.map_size, key=IPC_PRIVATE)
        if self.shm_id is None:
            log.warning("shmget failed with errno %d", ctypes.get_errno())
            return False

        # libc_shm.shmat returns None on failure -- do not compare against -1
        # here: under the required restype=c_void_p the (void *) -1 sentinel
        # arrives as 0xffffffffffffffff, which never equals -1.
        ptr = libc_shm.shmat(self.shm_id)
        if ptr is None:
            log.warning("shmat failed with errno %d", ctypes.get_errno())
            self._cleanup_shm()
            return False
        self.shm_ptr = ptr

        env = os.environ.copy()
        env["__AFL_SHM_ID"] = str(self.shm_id)

        stdin_r, stdin_w = os.pipe()
        self._err = tempfile.TemporaryFile()  # noqa: SIM115 - closed in _cleanup_shm
        self._err_off = 0
        err_fd = self._err.fileno()
        pid = os.fork()
        if pid == 0:
            # Everything here runs in the forked child, which shares the
            # parent's entire Python stack (pytest included). If execve
            # raises (e.g. target missing/not executable) without being
            # caught, the exception unwinds back into that inherited stack
            # instead of exiting — the child then keeps running as a second,
            # orphaned copy of the parent process. os._exit() must be
            # unconditionally reached on any failure here.
            try:
                os.setsid()
                os.dup2(stdin_r, 0)
                os.close(stdin_r)
                os.close(stdin_w)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 1)
                os.dup2(err_fd, 2)
                os.close(devnull)
                os.execve(self.target, [self.target], env)
            except BaseException:
                os._exit(127)
            os._exit(127)

        os.close(stdin_r)
        os.close(stdin_w)
        self.pid = pid
        if self._perf_counters is not None:
            self._perf_counters.open_for_pid(pid)

        try:
            # Poll for child status with retries.  In a multi-threaded
            # process, WNOHANG can miss a fast-exiting child on the first
            # try — the kernel may not have reaped it yet.  Retry for up
            # to 200ms to handle this race reliably.
            deadline = time.monotonic() + 0.2
            while time.monotonic() < deadline:
                waited, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
                if waited == 0:
                    time.sleep(0.01)
                    continue
                if os.WIFSTOPPED(status):
                    # Leave the init stop: a stopped target never sees SIGUSR1.
                    os.kill(pid, signal.SIGCONT)
                    self._started = True
                    log.info("Persistent target started (pid=%d)", pid)
                    return True
                # Child exited/signaled — not a persistent target
                break
            log.warning("Persistent target exited immediately")
            self._cleanup()
            return False
        except Exception as e:
            log.warning("Failed to start persistent target: %s", e)
            self._cleanup()
            return False

    def run_one(self, data: bytes) -> tuple[int, str]:
        if not self._started or self.pid is None:
            return -2, "persistent runner not started"

        data_len = min(len(data), self.map_size - self.HEADER_SIZE - 4)
        buf = struct.pack("<I", data_len) + b"\x00" * 4 + data[:data_len]
        ctypes.memmove(self.shm_ptr, buf, len(buf))
        if self._err is not None:
            with contextlib.suppress(OSError):
                self._err_off = os.fstat(self._err.fileno()).st_size

        try:
            os.kill(self.pid, signal.SIGUSR1)
        except ProcessLookupError:
            return -2, "target process not found"

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            # Branch on the pid: a clean exit(0) also has status 0.
            waited, status = os.waitpid(self.pid, os.WNOHANG | os.WUNTRACED)
            if waited == 0:
                time.sleep(0.0005)
                continue

            if os.WIFEXITED(status):
                self._started = False
                return os.WEXITSTATUS(status), self._read_stderr() or "target exited"
            if os.WIFSIGNALED(status):
                self._started = False
                return signal_rc(os.WTERMSIG(status)), self._read_stderr()
            if os.WIFSTOPPED(status):
                sig = os.WSTOPSIG(status)
                if sig in (signal.SIGSTOP, signal.SIGTRAP):
                    # Read return code via ctypes.cast (safe pointer arithmetic)
                    rc_ptr = ctypes.cast(
                        self.shm_ptr + self.HEADER_SIZE + data_len,
                        ctypes.POINTER(ctypes.c_int),
                    )
                    returncode = rc_ptr.contents.value
                    err = self._read_stderr()
                    os.kill(self.pid, signal.SIGCONT)
                    return returncode, err
                else:
                    self._started = False
                    return -sig, self._read_stderr()

        # Timeout: SIGKILL and reap
        try:
            os.kill(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass
        self._started = False
        return -1, self._read_stderr() or "timeout"

    def restart(self) -> bool:
        """Replace a dead target (crash, exit, timeout) with a fresh one."""
        self.stop()
        return self.start()

    @property
    def alive(self) -> bool:
        """Whether a target is parked and ready for the next input."""
        return self._started

    def stop(self):
        """Stop the target. SIGUSR2 is dead code for __AFL_LOOP targets — just SIGKILL."""
        if self.pid is not None and self._started:
            try:
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
        self._cleanup()

    def _cleanup(self):
        # Kill process FIRST, then detach SHM — detaching while the
        # target is still writing to shared memory can cause SEGV.
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
            self.pid = None
        self._cleanup_shm()
        self._started = False

    def _cleanup_shm(self):
        if self.shm_ptr:
            libc_shm.shmdt(self.shm_ptr)
            self.shm_ptr = 0
        if self.shm_id is not None:
            libc_shm.shmctl_rmid(self.shm_id)
            self.shm_id = None
        if self._err is not None:
            self._err.close()
            self._err = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
