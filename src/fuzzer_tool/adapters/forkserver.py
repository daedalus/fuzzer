"""Forkserver adapter — manages the compiled fuzz_loader binary.

The fuzz_loader (fuzz_loader.c) provides a persistent C process that
handles target execution with minimal overhead:

  - .so targets: dlopen + direct call with sigsetjmp timeout (zero fork)
  - executables: fork+exec per iteration with SIGALRM timeout

This adapter manages the fuzz_loader process lifecycle and communicates
via its stdin/stdout binary protocol.

Coverage does not travel over that protocol. The loader is spawned with
``__AFL_SHM_ID`` / ``AFL_MAP_SIZE`` in its environment, every exec'd child
inherits them, and ``afl_shim.c``'s constructor attaches to the segment on
its own — so the target writes edges directly into the fuzzer's SHM. The
caller resets the map before ``run_one`` and reads it after.

Protocol:
  Init:   "INIT <target> <func> <input_file> <timeout>\n" ->  "READY <mode>\n"
  Run:    "RUN <len>\n<data>"                             ->  "RC <rc> <err_len>\n<err>"
  Quit:   "QUIT\n"
"""

import contextlib
import logging
import os
import subprocess
import sys
import tempfile
import threading

from fuzzer_tool.adapters.process import _clean_env

log = logging.getLogger(__name__)

_FUZZ_LOADER_BIN = os.path.join(os.path.dirname(__file__), "fuzz_loader")


def _drain_stream(stream, sink: list) -> None:
    """Copy decoded lines from *stream* into *sink* until it closes.

    Module-level, and taking only the stream and the list, so that the
    stderr-drain thread holds no reference to the ForkserverRunner that
    started it. See the comment in ForkserverRunner.start(): a bound method
    here kept every un-stopped runner permanently reachable.
    """
    try:
        for line in stream:
            text = line.decode(errors="replace").rstrip()
            if text:
                sink.append(text)
                if len(sink) > STDERR_LINES_MAX:
                    del sink[:50]
    except (ValueError, OSError):
        pass


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


# ── Memory bounds ────────────────────────────────────────────────────
STDERR_LINES_MAX = 100  # max stderr lines retained from child processes

# Seconds allowed on top of the per-exec timeout before the loader process
# itself is presumed hung. The loader arms an interval timer at the exact
# (possibly fractional) timeout; this covers reaping the child and draining
# its stderr afterwards. It stays a flat second rather than scaling with the
# timeout, because the work it covers is process teardown, whose cost does
# not shrink just because the target got a tighter deadline.
_LOADER_GRACE = 1.0


def _ensure_compiled() -> str | None:
    """Return path to compiled fuzz_loader, rebuilding it if stale, else None.

    The binary is gitignored, so it survives every checkout, pull and bisect
    while ``fuzz_loader.c`` around it changes. Returning it on existence alone
    meant an edit to the source was silently ignored for the life of the
    working tree: the loader kept speaking whatever protocol it was built with
    and the only symptom was tests failing against assertions the current
    source satisfies.

    Rebuild whenever the source is newer than the binary. mtime rather than a
    content hash because the source is the only input and a stale-by-seconds
    rebuild costs a fraction of a second.
    """
    c_source = _FUZZ_LOADER_BIN + ".c"
    if os.path.isfile(_FUZZ_LOADER_BIN) and os.access(_FUZZ_LOADER_BIN, os.X_OK):
        try:
            if os.path.getmtime(_FUZZ_LOADER_BIN) >= os.path.getmtime(c_source):
                return _FUZZ_LOADER_BIN
            log.info("fuzz_loader is older than its source, rebuilding")
        except OSError:
            # Source gone (installed wheel, stripped tree): the binary we have
            # is the only one there will be.
            return _FUZZ_LOADER_BIN
    if not os.path.isfile(c_source):
        return None
    # clang first, gcc only as a fallback — same preference order as
    # tools/build_targets.sh, so the loader and the targets it runs are
    # built by the same toolchain wherever clang exists.
    for cc in ("clang", "gcc"):
        try:
            subprocess.run(
                [cc, "-O2", "-o", _FUZZ_LOADER_BIN, c_source, "-ldl"],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return _FUZZ_LOADER_BIN
        except Exception as e:
            log.debug("Failed to compile fuzz_loader with %s: %s", cc, e)
    log.warning("Failed to compile fuzz_loader (tried clang, gcc)")
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
        env: dict[str, str] | None = None,
    ):
        self.target = target
        self.function_name = function_name
        self.timeout = timeout
        # Environment overlaid on os.environ when the loader is spawned.
        # Carries __AFL_SHM_ID / AFL_MAP_SIZE: every exec'd child inherits
        # them, which is how the target finds the fuzzer's coverage map.
        self.env_overrides: dict[str, str] = dict(env or {})
        self._proc: subprocess.Popen | None = None
        self._ready = False
        # Set from the loader's READY line by start(); see start().
        self.exec_mode: str = ""
        # Capability tokens trailing the mode on the READY line. Empty for
        # a loader built before they existed, which is exactly what
        # set_timeout() needs to know.
        self.capabilities: frozenset[str] = frozenset()
        self._last_stderr: str = ""
        self._input_file: str | None = None
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

        if self._input_file is None:
            fd, self._input_file = tempfile.mkstemp(suffix=".cur", prefix="fuzz_fork_")
            os.close(fd)

        # Same env treatment run_target_fast() gives its children: merge the
        # caller's overrides over os.environ, then strip the LD_PRELOAD
        # entries that break sanitizer targets.
        env = _clean_env({**os.environ, **self.env_overrides})
        if "AFL_MAP_SIZE" not in env:
            env["AFL_MAP_SIZE"] = "8192"

        self._proc = subprocess.Popen(
            [loader_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Drain stderr in background.
        #
        # The thread must NOT hold a strong reference to `self`. It used to
        # (target=self._drain_stderr, a bound method), and that reference is
        # self-sustaining: the thread blocks reading the child's stderr until
        # the child exits, the child only exits when stop() sends QUIT, and
        # stop() only runs from __del__, which cannot run while the thread
        # holds the runner alive. Nothing ever broke the loop, so every
        # runner that was not explicitly stopped leaked a blocked thread, an
        # orphaned fuzz_loader, its target child, and a SHM segment pinned in
        # `dest` state until it detached. Measured before this change: two
        # test files left 98 live runners, and gc.collect() freed none of
        # them, because they were not garbage -- they were reachable from the
        # thread. A full suite run peaked at ~185 orphaned processes.
        #
        # A plain list plus the stream is all the thread actually needs. The
        # list stays shared, so _stderr_lines still fills as before, and the
        # thread now dies of natural causes when stop() closes the pipe.
        self._stderr_lines = []
        self._stderr_thread = threading.Thread(
            target=_drain_stream,
            args=(self._proc.stderr, self._stderr_lines),
            daemon=True,
        )
        self._stderr_thread.start()

        # %.6f, not int(): int(0.04) is 0, and the loader's `<= 0` fallback
        # then substituted 5s -- so asking for a tighter timeout silently
        # produced a 125x looser one. The loader parses this with strtod.
        init = f"INIT {self.target} {self.function_name} {self._input_file} {self.timeout:.6f}\n"
        try:
            self._proc.stdin.write(init.encode())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            log.warning("Forkserver: failed to send INIT")
            return False

        resp = self._proc.stdout.readline()
        parts = resp.decode(errors="replace").strip().split()
        if parts and parts[0] == "READY":
            self._ready = True
            # "forkserver" | "exec" | "dlopen"; "" from a loader built before
            # the suffix existed. Only ever informational -- the protocol is
            # identical in every mode, so nothing branches on it.
            self.exec_mode = parts[1] if len(parts) > 1 else ""
            self.capabilities = frozenset(parts[2:])
            log.info(
                "Forkserver started: %s (mode=%s, caps=%s)",
                self.target,
                self.exec_mode or "?",
                ",".join(sorted(self.capabilities)) or "-",
            )
            return True

        log.warning("Forkserver failed to start: %r", resp)
        return False

    def _drain_stderr(self):
        """Kept for callers/tests that invoke it directly.

        Not used as a thread target any more -- see start(); a bound method
        there pinned the runner and defeated __del__.
        """
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        _drain_stream(proc.stderr, self._stderr_lines)

    def run_one(self, data: bytes) -> tuple[int, str]:
        """Run *data* once, returning ``(returncode, stderr)``.

        Coverage is not returned: the target wrote it straight into the SHM
        segment named by ``__AFL_SHM_ID``, which the caller reads directly.
        The child's stderr *is* returned, because sanitizer reports are the
        only crash signal for an ASAN build that exits 1 (see
        ``ExecutionRunner.is_crash``).
        """
        if not self._ready or not self._proc:
            return -2, ""

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
            return -2, ""

        # Threaded readline with timeout
        result = [None]

        def _readline():
            result[0] = self._proc.stdout.readline()

        t = threading.Thread(target=_readline, daemon=True)
        t.start()
        # Wait past the loader's own timeout before declaring the
        # loader itself hung: it needs to reap the child and drain its stderr
        # after firing, and joining at exactly the alarm deadline would tear
        # down a healthy loader on every timing-out input.
        t.join(timeout=self.timeout + _LOADER_GRACE)
        if t.is_alive():
            log.warning("Forkserver timed out after %.1fs, restarting", self.timeout)
            proc = self._proc
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait()
            _close_streams(proc)
            self._ready = False
            if not self._restarting:
                self._restarting = True
                try:
                    if self.start():
                        return self.run_one(data)
                finally:
                    self._restarting = False
            return -1, ""

        header = result[0]
        if not header:
            return -2, ""

        parts = header.decode().strip().split()
        if len(parts) < 3 or parts[0] != "RC":
            return -2, ""

        rc = int(parts[1])
        err_len = int(parts[2])

        stderr = ""
        if err_len > 0:
            raw = self._proc.stdout.read(err_len)
            if raw:
                stderr = raw.decode(errors="replace")

        self._last_stderr = stderr
        return rc, stderr

    def set_timeout(self, seconds: float) -> bool:
        """Retune the loader's per-exec deadline in place.

        Returns True when the loader confirmed the new value.

        The timeout is otherwise baked into the INIT handshake and fixed for
        the life of the process, which is why ``suggested_timeout()`` was
        only ever printed and never applied. Retuning by restarting would
        work but throws away the forkserver and its warmed target on every
        adjustment.

        Declines rather than guesses when the loader did not advertise
        ``retune``: an older binary ignores an unrecognised command
        silently, so sending it anyway would leave the Python side reporting
        a deadline the loader is not using — the same silent substitution as
        the int()/atoi() truncation this replaces. The caller can restart to
        apply the value instead.
        """
        if not (seconds > 0):
            return False
        if not self._ready or not self._proc:
            self.timeout = seconds
            return False
        if "retune" not in self.capabilities:
            log.debug("Forkserver: loader has no 'retune' capability; timeout unchanged")
            return False

        try:
            self._proc.stdin.write(f"TIMEOUT {seconds:.6f}\n".encode())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._ready = False
            return False

        # Bounded wait on the ack. The loader answers immediately -- it is
        # between execs when it reads a command -- so a stall here means the
        # loader is wedged, and blocking forever on stdout would take the
        # fuzz loop with it.
        result: list = [None]

        def _readline():
            result[0] = self._proc.stdout.readline()

        t = threading.Thread(target=_readline, daemon=True)
        t.start()
        t.join(timeout=self.timeout + _LOADER_GRACE)
        if t.is_alive():
            log.warning("Forkserver: no reply to TIMEOUT; leaving deadline at %.3fs", self.timeout)
            return False

        parts = (result[0] or b"").decode(errors="replace").split()
        if len(parts) >= 2 and parts[0] == "TIMEOUT_OK":
            # Trust the loader's echo, not the request: it clamps, and the
            # two sides disagreeing about the deadline is the whole failure
            # mode being fixed here.
            self.timeout = float(parts[1])
            return True

        log.warning("Forkserver: unexpected reply to TIMEOUT: %r", result[0])
        return False

    def update_shm_after_resize(self, new_env_id: str, new_size: int) -> None:
        """Restart the loader so its children inherit the resized SHM.

        ``ShmCoverage.resize()`` allocates a *new* segment and removes the
        old one, so the id baked into the running loader's environment is
        stale — its children would attach to a removed segment and their
        coverage would be dropped on the floor. The environment is only read
        at exec time, so the only way to update it is to respawn.
        """
        self.env_overrides["__AFL_SHM_ID"] = new_env_id
        self.env_overrides["AFL_MAP_SIZE"] = str(new_size)
        if self._proc is None:
            return
        self.stop()
        if not self.start():
            log.warning("Forkserver failed to restart after SHM resize")

    def stop(self):
        proc = self._proc
        self._proc = None
        self._ready = False
        if proc is None:
            return
        try:
            proc.stdin.write(b"QUIT\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            # ValueError: stop() ran already and closed the stream. __del__
            # calls stop() a second time at GC, which raised "write to closed
            # file" through the ignored-exception path on every clean exit.
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            with contextlib.suppress(OSError, ValueError):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=1)
        # Close the pipes: a child that inherited a copy of the write end
        # would otherwise keep the stderr-drain thread blocked forever,
        # leaving the process permanently multi-threaded.
        _close_streams(proc)
        if self._input_file and os.path.exists(self._input_file):
            with contextlib.suppress(OSError):
                os.unlink(self._input_file)
            self._input_file = None

    def __del__(self):
        self.stop()
