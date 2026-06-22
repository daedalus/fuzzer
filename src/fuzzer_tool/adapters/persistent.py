"""Persistent mode adapter: run AFL-loop targets without fork overhead.

For targets compiled with __AFL_LOOP(N), keeps a single process alive
and signals it between iterations using shared memory + SIGSTOP/SIGCONT.
"""

import ctypes
import logging
import os
import signal
import struct
import time

log = logging.getLogger(__name__)

# Signal to tell the target to process the next input
PERSISTENT_SIGNAL = signal.SIGUSR1
# Signal to tell the target to exit
PERSISTENT_EXIT = signal.SIGUSR2


class PersistentRunner:
    """Run a target compiled with __AFL_LOOP() in persistent mode.

    Instead of forking a new process for each input, keeps the target alive
    and signals it between iterations. Requires the target to be compiled
    with AFL-style persistent mode support:

        __AFL_HAVE_MANUAL_CONTROL

        int main() {
            __AFL_INIT();
            unsigned char *buf = __AFL_FUZZ_TEST_CASE_BUF;
            while (__AFL_LOOP(1000)) {
                int len = __AFL_FUZZ_TEST_CASE_LEN;
                // process buf[0..len]
            }
        }

    The runner communicates via a shared memory segment that holds:
    - 4 bytes: input length
    - N bytes: input data
    - 4 bytes: return code (written by target, read by runner)
    """

    SHM_KEY = 0x414C4552  # "ALER" as int
    HEADER_SIZE = 8  # 4 bytes len + 4 bytes padding

    def __init__(self, target: str, timeout: float = 5.0, map_size: int = 65536):
        self.target = target
        self.timeout = timeout
        self.map_size = map_size
        self.pid: int | None = None
        self.shm_id: int | None = None
        self.shm_ptr: ctypes.c_void_p | None = None
        self._started = False

    def start(self) -> bool:
        """Start the target in persistent mode.

        Returns True if the target started successfully.
        """
        if self._started:
            return True

        # Create shared memory for input/output
        libc = ctypes.CDLL("libc.so.6", use_errno=True)

        # shmget(IPC_CREAT, size, 0o600)
        self.shm_id = libc.shmget(self.SHM_KEY, self.map_size, 0o600 | 0o2000)
        if self.shm_id < 0:
            errno = ctypes.get_errno()
            log.warning("shmget failed with errno %d", errno)
            return False

        # shmat(shmid, NULL, 0)
        self.shm_ptr = libc.shmat(self.shm_id, None, 0)
        if self.shm_ptr == ctypes.c_void_p(-1):
            log.warning("shmat failed")
            self._cleanup_shm()
            return False

        # Set environment for the target
        env = os.environ.copy()
        env["__AFL_SHM_ID"] = str(self.shm_id)
        env["__AFL_HAVE_MANUAL_CONTROL"] = "1"

        # Start the target
        stdin_r, stdin_w = os.pipe()
        pid = os.fork()
        if pid == 0:
            # Child
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

        # Wait for the target to start (it should stop at __AFL_LOOP)
        try:
            _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
            if os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGSTOP:
                self._started = True
                log.info("Persistent target started (pid=%d)", pid)
                return True
            # If not stopped yet, give it a moment
            time.sleep(0.05)
            _, status = os.waitpid(pid, os.WNOHANG | os.WUNTRACED)
            if os.WIFSTOPPED(status):
                self._started = True
                log.info("Persistent target started (pid=%d)", pid)
                return True
            # Target might have exited already
            log.warning("Persistent target exited immediately")
            self._cleanup()
            return False
        except Exception as e:
            log.warning("Failed to start persistent target: %s", e)
            self._cleanup()
            return False

    def run_one(self, data: bytes) -> tuple[int, str]:
        """Send one input to the persistent target and get the result.

        Args:
            data: Input bytes to send.

        Returns:
            Tuple of (returncode, stderr_output).
        """
        if not self._started or self.pid is None or self.shm_ptr is None:
            return -2, "persistent runner not started"

        # Write input to shared memory: [4 bytes len][input data]
        data_len = min(len(data), self.map_size - self.HEADER_SIZE)
        buf = struct.pack("<I", data_len) + b"\x00" * 4 + data[:data_len]
        ctypes.memmove(self.shm_ptr, buf, len(buf))

        # Signal the target to process
        try:
            os.kill(self.pid, PERSISTENT_SIGNAL)
        except ProcessLookupError:
            return -2, "target process not found"

        # Wait for it to stop again (indicating it processed the input)
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
                if sig == signal.SIGSTOP or sig == signal.SIGTRAP:
                    # Normal: target stopped, ready for next input
                    # Read return code from shared memory if available
                    try:
                        rc_bytes = ctypes.string_at(self.shm_ptr + self.HEADER_SIZE + data_len, 4)
                        returncode = struct.unpack("<i", rc_bytes)[0]
                    except Exception:
                        returncode = 0
                    return returncode, ""
                elif sig == PERSISTENT_EXIT:
                    self._started = False
                    return 0, ""
                else:
                    self._started = False
                    return -sig, ""

        # Timeout: kill the target
        try:
            os.kill(self.pid, signal.SIGKILL)
            os.waitpid(self.pid, 0)
        except Exception:
            pass
        self._started = False
        return -1, "timeout"

    def stop(self):
        """Stop the persistent target gracefully."""
        if self.pid is not None and self._started:
            try:
                os.kill(self.pid, PERSISTENT_EXIT)
                time.sleep(0.1)
                _, status = os.waitpid(self.pid, os.WNOHANG)
                if not os.WIFEXITED(status):
                    os.kill(self.pid, signal.SIGKILL)
                    os.waitpid(self.pid, 0)
            except ProcessLookupError:
                pass
        self._cleanup()

    def _cleanup(self):
        """Clean up shared memory and process."""
        self._cleanup_shm()
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGKILL)
                os.waitpid(self.pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
            self.pid = None

    def _cleanup_shm(self):
        """Remove the shared memory segment."""
        if self.shm_ptr is not None:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.shmdt(self.shm_ptr)
            self.shm_ptr = None
        if self.shm_id is not None:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.shmctl(self.shm_id, 0, None)  # IPC_RMID = 0
            self.shm_id = None
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
