"""Persistent subprocess loader for coverage-guided fuzzing.

Keeps one Python subprocess alive across iterations, communicating
via stdin/stdout pipes. Eliminates Python startup + ctypes.CDLL load
overhead on every iteration.

Protocol:
  Init:   "INIT <target> <func>\n"  ->  "READY\n"
  Run:    "RUN <len>\n<data>"       ->  "RC <rc> <bmp_len>\n<bmp>"
  Quit:   "QUIT\n"
"""

import contextlib
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading

log = logging.getLogger(__name__)

_PERSISTENT_LOADER = r"""#!/usr/bin/env python3
import ctypes, ctypes.util, os, signal, sys

target = None
func = None

def load_target(target_path, func_name):
    global target, func
    lib = ctypes.CDLL(target_path)
    func = getattr(lib, func_name)
    func.restype = ctypes.c_int
    func.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

def read_shm():
    shm_id_str = os.environ.get("__AFL_SHM_ID")
    if not shm_id_str:
        return b""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.shmat.restype = ctypes.c_void_p
        ptr = libc.shmat(int(shm_id_str), None, 0)
        if ptr and ptr != -1:
            map_size = int(os.environ.get("AFL_MAP_SIZE", "65536"))
            return bytes((ctypes.c_uint8 * map_size).from_address(ptr))
    except Exception:
        pass
    return b""

NO_BMP = os.environ.get("_LOADER_NO_BMP", "0") == "1"

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
            try:
                rc = func(buf, len(data))
            except Exception:
                rc = -11
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
        try:
            os.waitpid(child_pid, 0)
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

        if NO_BMP:
            resp = f"RC {rc} 0\n".encode()
            sys.stdout.buffer.write(resp)
            sys.stdout.buffer.flush()
        else:
            bmp = read_shm()
            resp = f"RC {rc} {len(bmp)}\n".encode()
            sys.stdout.buffer.write(resp)
            if bmp:
                sys.stdout.buffer.write(bmp)
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
        self, target: str, function_name: str = "LLVMFuzzerTestOneInput", timeout: float = 5.0
    ):
        self.target = target
        self.function_name = function_name
        self.timeout = timeout
        self._proc = None
        self._ready = False
        self._last_bitmap = None
        self._restarting = False
        self._child_pid_file: str | None = None

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
            env["AFL_MAP_SIZE"] = "65536"
        env["_LOADER_NO_BMP"] = "1"
        env["_CHILD_PID_FILE"] = self._child_pid_file

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
        """Consume stderr to prevent pipe-buffer deadlock."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                # Log first 200 chars to avoid flooding; drop the rest
                text = line.decode(errors="replace").rstrip()
                if text:
                    log.debug("loader stderr: %s", text[:200])
        except (ValueError, OSError):
            pass

    def run_one(self, data: bytes) -> tuple[int, bytes | None]:
        if not self._ready or not self._proc:
            return -2, None

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
            log.warning("Persistent loader timed out after %.1fs, restarting", self.timeout)
            # Kill orphaned grandchild first (it's in its own process group)
            self._kill_orphaned_child()
            # Then kill the loader itself
            with contextlib.suppress(Exception):
                self._proc.kill()
                self._proc.wait()
            self._ready = False
            if not self._restarting:
                self._restarting = True
                try:
                    if self.start():
                        return self.run_one(data)
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

        bitmap = None
        if bmp_len > 0:
            bitmap = self._proc.stdout.read(bmp_len)

        self._last_bitmap = bitmap
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
