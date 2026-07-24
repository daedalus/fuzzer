"""Forkserver adapter — manages the compiled fuzz_loader binary.

The fuzz_loader (fuzz_loader.c) provides a persistent C process that
handles target execution with minimal overhead:

  - .so targets: dlopen + direct call with sigsetjmp timeout (zero fork)
  - executables: fork+exec per iteration with SIGALRM timeout

This adapter manages the fuzz_loader process lifecycle and communicates
via its stdin/stdout binary protocol.

Protocol:
  Init:   "INIT <target> <func> <bitmap_out> <timeout>\n"  ->  "READY\n"
  Run:    "RUN <len>\n<data>"                              ->  "RC <rc> <bmp_len>\n<bmp>"
  Quit:   "QUIT\n"
"""

import contextlib
import logging
import os
import subprocess
import tempfile
import threading

log = logging.getLogger(__name__)

_FUZZ_LOADER_BIN = os.path.join(os.path.dirname(__file__), "fuzz_loader")

# ── Memory bounds ────────────────────────────────────────────────────
STDERR_LINES_MAX = 100  # max stderr lines retained from child processes


def _ensure_compiled() -> str | None:
    """Return path to compiled fuzz_loader, or None if compilation fails."""
    if os.path.isfile(_FUZZ_LOADER_BIN) and os.access(_FUZZ_LOADER_BIN, os.X_OK):
        return _FUZZ_LOADER_BIN
    # Try to compile
    c_source = _FUZZ_LOADER_BIN + ".c"
    if not os.path.isfile(c_source):
        return None
    try:
        subprocess.run(
            ["gcc", "-O2", "-o", _FUZZ_LOADER_BIN, c_source, "-ldl"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return _FUZZ_LOADER_BIN
    except Exception as e:
        log.warning("Failed to compile fuzz_loader: %s", e)
        return None


class ForkserverRunner:
    """Run targets via the compiled C fuzz_loader binary.

    Launches fuzz_loader once and keeps it alive across iterations.
    For .so targets, this gives zero-fork persistent execution.
    For executables, each iteration still forks+execs but avoids
    the Python subprocess wrapper overhead.
    """

    def __init__(
        self,
        target: str,
        function_name: str = "LLVMFuzzerTestOneInput",
        timeout: float = 5.0,
    ):
        self.target = target
        self.function_name = function_name
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._ready = False
        self._last_bitmap: bytes | None = None
        self._bitmap_out: str | None = None
        self._restarting = False
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._proc and self._proc.poll() is None:
            return True

        loader_bin = _ensure_compiled()
        if loader_bin is None:
            log.warning("fuzz_loader binary not available")
            return False

        fd, self._bitmap_out = tempfile.mkstemp(suffix=".bmp", prefix="fuzz_fork_")
        os.close(fd)

        env = os.environ.copy()
        if "AFL_MAP_SIZE" not in env:
            env["AFL_MAP_SIZE"] = "8192"

        self._proc = subprocess.Popen(
            [loader_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Drain stderr in background
        self._stderr_lines = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        init = f"INIT {self.target} {self.function_name} {self._bitmap_out} {int(self.timeout)}\n"
        try:
            self._proc.stdin.write(init.encode())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            log.warning("Forkserver: failed to send INIT")
            return False

        resp = self._proc.stdout.readline()
        if resp.strip() == b"READY":
            self._ready = True
            log.info("Forkserver started: %s", self.target)
            return True

        log.warning("Forkserver failed to start: %r", resp)
        return False

    def _drain_stderr(self):
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                text = line.decode(errors="replace").rstrip()
                if text:
                    self._stderr_lines.append(text)
                    if len(self._stderr_lines) > STDERR_LINES_MAX:
                        del self._stderr_lines[:50]
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
            self._ready = False
            if not self._restarting:
                self._restarting = True
                try:
                    if self.start():
                        return self.run_one(data)
                finally:
                    self._restarting = False
            return -2, None

        # Threaded readline with timeout
        result = [None]

        def _readline():
            result[0] = self._proc.stdout.readline()

        t = threading.Thread(target=_readline, daemon=True)
        t.start()
        t.join(timeout=self.timeout)
        if t.is_alive():
            log.warning("Forkserver timed out after %.1fs, restarting", self.timeout)
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
        if self._bitmap_out and os.path.exists(self._bitmap_out):
            with contextlib.suppress(OSError):
                os.unlink(self._bitmap_out)
            self._bitmap_out = None

    def stderr_output(self) -> str:
        return "\n".join(self._stderr_lines[-20:])

    def __del__(self):
        self.stop()
