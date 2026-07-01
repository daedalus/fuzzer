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
import subprocess
import sys
import tempfile

log = logging.getLogger(__name__)

_PERSISTENT_LOADER = r"""#!/usr/bin/env python3
import ctypes, ctypes.util, os, sys

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
        try:
            rc = func(buf, len(data))
        except Exception:
            rc = -11
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

    def start(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True

        fd, self._loader_path = tempfile.mkstemp(suffix=".py", prefix="fuzz_persist_")
        os.write(fd, _PERSISTENT_LOADER.encode())
        os.close(fd)

        env = os.environ.copy()
        if "AFL_MAP_SIZE" not in env:
            env["AFL_MAP_SIZE"] = "65536"

        self._proc = subprocess.Popen(
            [sys.executable, self._loader_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )

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

    def run_one(self, data: bytes) -> tuple[int, bytes | None]:
        if not self._ready or not self._proc:
            return -2, None

        cmd = f"RUN {len(data)}\n"
        self._proc.stdin.write(cmd.encode())
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

        header = self._proc.stdout.readline()
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
        if self._proc:
            with contextlib.suppress(Exception):
                self._proc.stdin.write(b"QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=2)
            with contextlib.suppress(Exception):
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        self._ready = False

    def __del__(self):
        self.stop()
