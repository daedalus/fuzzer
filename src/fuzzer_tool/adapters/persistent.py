"""Persistent mode adapter for custom SIGUSR1-handling targets.

IMPORTANT: This does NOT implement the standard AFL persistent mode protocol
(fd 198/199 status pipe). Standard AFL-instrumented targets using __AFL_LOOP()
communicate via a status pipe, not signals. This runner is for CUSTOM targets
that explicitly handle SIGUSR1 and read from __AFL_SHM_ID shared memory.

For standard AFL persistent targets, use afl-showmap -P or AFL++'s own
persistent mode support instead.

Protocol:
  - Runner writes [4-byte length][input data] to SHM
  - Runner sends SIGUSR1 to the target
  - Target processes input, writes result, sends SIGSTOP (via __AFL_LOOP exit)
  - Runner reads result from SHM, resumes target with SIGCONT
"""

import ctypes
import logging
import os
import signal
import struct
import time

log = logging.getLogger(__name__)

IPC_RMID = 0
IPC_PRIVATE = 0


class PersistentRunner:
    """Run a custom SIGUSR1-handling target in persistent mode.

    Uses IPC_PRIVATE shared memory (no key collisions across workers).
    Target must: read input from __AFL_SHM_ID, handle SIGUSR1, SIGSTOP on
    each iteration boundary.
    """

    HEADER_SIZE = 8  # 4 bytes len + 4 bytes padding

    def __init__(self, target: str, timeout: float = 5.0, map_size: int = 65536):
        self.target = target
        self.timeout = timeout
        self.map_size = map_size
        self.pid: int | None = None
        self.shm_id: int | None = None
        self.shm_ptr: int = 0
        self._started = False
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)

    def start(self) -> bool:
        if self._started:
            return True

        # Use IPC_PRIVATE to avoid key collisions between parallel workers
        self.shm_id = self._libc.shmget(IPC_PRIVATE, self.map_size, 0o600)
        if self.shm_id < 0:
            log.warning("shmget failed with errno %d", ctypes.get_errno())
            return False

        ptr = self._libc.shmat(self.shm_id, None, 0)
        if ptr == -1:
            log.warning("shmat failed")
            self._cleanup_shm()
            return False
        self.shm_ptr = ptr

        env = os.environ.copy()
        env["__AFL_SHM_ID"] = str(self.shm_id)

        stdin_r, stdin_w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.setsid()
            os.dup2(stdin_r, 0)
            os.close(stdin_r)
            os.close(stdin_w)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
            os.execve(self.target, [self.target], env)
            os._exit(127)

        os.close(stdin_r)
        os.close(stdin_w)
        self.pid = pid

        try:
            _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
            if os.WIFSTOPPED(status):
                self._started = True
                log.info("Persistent target started (pid=%d)", pid)
                return True
            time.sleep(0.05)
            _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
            if os.WIFSTOPPED(status):
                self._started = True
                log.info("Persistent target started (pid=%d)", pid)
                return True
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

        try:
            os.kill(self.pid, signal.SIGUSR1)
        except ProcessLookupError:
            return -2, "target process not found"

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            _, status = os.waitpid(self.pid, os.WNOHANG | os.WUNTRACED)
            if status == 0:
                time.sleep(0.0005)
                continue

            if os.WIFEXITED(status):
                self._started = False
                return os.WEXITSTATUS(status), "target exited"
            if os.WIFSIGNALED(status):
                self._started = False
                return -os.WTERMSIG(status), ""
            if os.WIFSTOPPED(status):
                sig = os.WSTOPSIG(status)
                if sig in (signal.SIGSTOP, signal.SIGTRAP):
                    # Read return code via ctypes.cast (safe pointer arithmetic)
                    rc_ptr = ctypes.cast(
                        self.shm_ptr + self.HEADER_SIZE + data_len,
                        ctypes.POINTER(ctypes.c_int),
                    )
                    returncode = rc_ptr.contents.value
                    return returncode, ""
                else:
                    self._started = False
                    return -sig, ""

        # Timeout: SIGKILL and reap
        try:
            os.kill(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass
        self._started = False
        return -1, "timeout"

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
        self._cleanup_shm()
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
            self.pid = None
        self._started = False

    def _cleanup_shm(self):
        if self.shm_ptr:
            self._libc.shmdt(self.shm_ptr)
            self.shm_ptr = 0
        if self.shm_id is not None:
            self._libc.shmctl(self.shm_id, IPC_RMID, None)
            self.shm_id = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
