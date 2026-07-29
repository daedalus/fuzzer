"""Network adapter: fire-and-forget TCP/UDP fuzzing of a persistent target.

Unlike PersistentRunner (persistent.py), there is no SIGUSR1/SIGSTOP
iteration boundary here — we're driving an ordinary socket-facing target
as a black box, not a custom AFL-loop harness. Each iteration is:
connect (or reuse a kept-alive connection), send exactly once, sleep a
fixed settle window, return. We never recv() a reply.

Coverage does NOT come from anything on this socket. The target process
must be started (by the caller, or already running externally) with
__AFL_SHM_ID / AFL_MAP_SIZE in its environment so its LD_PRELOAD'd
afl_shim.so attaches to the same SHM segment this fuzzer process reads
via ShmCoverage. This module makes no changes to afl_shim.c — it only
drives the target over the wire and leaves coverage reading to the
existing caller-side shm.is_new_coverage_with_edges() path.

Crash signal is necessarily weak for a network target: we don't get a
signal number the way fork+exec or PersistentRunner do. If the caller
owns the target's pid (spawned it themselves), a dead connection can be
correlated with a reaped exit status for a real -sig code. If not (an
externally-run service), a failed reconnect is the only available
signal, reported as a generic -1/"connect failed" — the caller should
treat that as "target unreachable," not as a confirmed crash, unless
corroborated some other way (e.g. a health-check endpoint).
"""

import contextlib
import logging
import os
import socket
import time

log = logging.getLogger(__name__)


class NetworkRunner:
    """Send one fuzz iteration's data as a single TCP or UDP packet.

    Args:
        host: Destination IPv4 address or hostname.
        port: Destination port.
        proto: "tcp" (default) or "udp".
        keepalive: If True, reuse one connection/socket across run_one()
            calls instead of reconnecting every iteration.
        settle: Seconds to sleep after send() before returning, since
            there's no reply to wait on. Tune up if coverage looks racy
            (edges from iteration N showing up while reading iteration
            N-1's snapshot).
        connect_timeout: Seconds to wait for TCP connect() (irrelevant
            for UDP, whose connect() only fixes the peer locally).
        server_pid: If the caller spawned the target itself, its pid —
            enables real exit-status reaping on a dead connection.
            Leave None for an externally-running target.
    """

    def __init__(
        self,
        host: str,
        port: int,
        proto: str = "tcp",
        keepalive: bool = False,
        settle: float = 0.01,
        connect_timeout: float = 2.0,
        server_pid: int | None = None,
    ):
        self.host = host
        self.port = port
        self.proto = proto.lower()
        self.keepalive = keepalive
        self.settle = settle
        self.connect_timeout = connect_timeout
        self.server_pid = server_pid
        self._sock: socket.socket | None = None

    # ── connection management ────────────────────────────────────────

    def _open_socket(self) -> socket.socket | None:
        try:
            if self.proto == "udp":
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            else:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(self.connect_timeout)
            s.connect((self.host, self.port))
            s.settimeout(None)  # fire-and-forget send needs no per-call timeout
            return s
        except OSError as e:
            log.debug("connect(%s:%d) failed: %s", self.host, self.port, e)
            return None

    def _close(self):
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None

    # ── liveness (only meaningful if we own server_pid) ──────────────

    def _reap_server_exit(self) -> int:
        """Reap the owned server pid if it has exited, translated to the
        -sig convention the rest of the fuzzer's crash detection expects.
        Returns 0 if there's nothing to reap or we don't own the pid."""
        if self.server_pid is None:
            return 0
        try:
            pid, status = os.waitpid(self.server_pid, os.WNOHANG)
        except ChildProcessError:
            return 0
        if pid == 0:
            return 0
        if os.WIFSIGNALED(status):
            return -os.WTERMSIG(status)
        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        return 0

    # ── main entry point ──────────────────────────────────────────────

    def run_one(self, data: bytes) -> tuple[int, str]:
        """Send one iteration's data. Returns (rc, err):

        rc == 0            normal send.
        rc == -sig          target's owned pid was reaped with that signal.
        rc == -1            connect/send failed and we couldn't confirm a
                             crash (unowned target, or clean exit).
        """
        sock = self._sock if self.keepalive else None
        if sock is None:
            sock = self._open_socket()
            if sock is None:
                sig_rc = self._reap_server_exit()
                return (sig_rc, "target exited") if sig_rc != 0 else (-1, "connect failed")
            if self.keepalive:
                self._sock = sock

        try:
            if self.proto == "udp":
                sock.send(data)
            else:
                sock.sendall(data)
        except OSError as e:
            self._close()
            sig_rc = self._reap_server_exit()
            return (sig_rc, "target exited") if sig_rc != 0 else (-1, f"send failed: {e}")

        if self.settle > 0:
            time.sleep(self.settle)

        if not self.keepalive:
            self._close()

        return 0, ""

    def stop(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()
