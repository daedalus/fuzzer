"""TCP seed inbox for ``--tcp-listen-port``.

One connection = one seed: the listener reads until the peer closes, then
queues the bytes. ``OperatorEngine.mutate()`` pops one per round, so each seed
runs the normal exec / coverage / crash / admit path.

    client A --+                  +--> [reader A] --+
    client B --+--> [acceptor] --+--> [reader B] --+--append--> deque
    client C --+   (max_connections)               |              |
                                     OperatorEngine.mutate() <--pop--+

Connections are served concurrently: the acceptor hands each one to its own
reader thread, at most ``max_connections`` at a time; past that, clients wait
in the listen backlog. A stalled client holds only its own reader, for at most
``conn_timeout``, and its partial bytes are kept as the message. Memory is
bounded by ``(capacity + max_connections) * max_bytes``.

    printf 'GIF89a...' | nc -N 127.0.0.1 4444
"""

from __future__ import annotations

import collections
import contextlib
import logging
import socket
import sys
import threading
import time
from typing import NamedTuple

from fuzzer_tool.adapters.filesystem import hash_data

log = logging.getLogger(__name__)

# Loopback only unless --tcp-listen-host says otherwise: seeds run the target.
DEFAULT_HOST = "127.0.0.1"
# Queued, unpopped seeds; a message arriving to a full inbox is dropped.
INBOX_CAPACITY = 1024
# Per-message cap when the caller gives none (the fuzzer passes max_len).
DEFAULT_MAX_BYTES = 1 << 20
# Wall budget for one connection, from accept to last byte.
CONN_TIMEOUT_S = 5.0
# accept() poll: bounds close() latency.
ACCEPT_POLL_S = 0.2
RECV_CHUNK = 1 << 16
# Concurrent reader threads; further clients wait in the backlog.
MAX_CONNECTIONS = 64
LISTEN_BACKLOG = 128


class TcpSeed(NamedTuple):
    """A received message and its id (``hash_data``, as logged on receipt)."""

    data: bytes
    seed_id: str


class TcpSeedInbox:
    """Bounded queue of seeds received over TCP, filled by per-connection threads.

    Binds in the constructor, so a busy port fails at startup (``OSError``).
    """

    def __init__(
        self,
        port: int,
        host: str = DEFAULT_HOST,
        max_bytes: int = DEFAULT_MAX_BYTES,
        capacity: int = INBOX_CAPACITY,
        conn_timeout: float = CONN_TIMEOUT_S,
        max_connections: int = MAX_CONNECTIONS,
    ) -> None:
        if max_connections < 1:
            raise ValueError(f"max_connections must be >= 1, got {max_connections}")
        self._max_bytes = max_bytes
        self._capacity = capacity
        self._conn_timeout = conn_timeout

        # Many producers (readers), one consumer (fuzz loop). The lock makes
        # the capacity check + append + counters atomic across readers; pop()
        # skips it (popleft is atomic and only frees room).
        self._lock = threading.Lock()
        self._queue: collections.deque[TcpSeed] = collections.deque()
        self._received = 0
        self._dropped = 0

        # Reader slots and the live connections, for close().
        self._slots = threading.BoundedSemaphore(max_connections)
        self._conns: dict[threading.Thread, socket.socket] = {}

        self._stop = threading.Event()
        self._sock = socket.create_server((host, port), backlog=LISTEN_BACKLOG)
        self._sock.settimeout(ACCEPT_POLL_S)
        self._thread = threading.Thread(target=self._loop, name="tcp-seed-inbox", daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        """Bound port (resolves a requested port 0)."""
        return self._sock.getsockname()[1]

    @property
    def received(self) -> int:
        """Messages queued so far."""
        return self._received

    @property
    def dropped(self) -> int:
        """Messages lost to a full inbox."""
        return self._dropped

    @property
    def pending(self) -> int:
        """Seeds queued and not popped yet."""
        return len(self._queue)

    def pop(self) -> bytes | None:
        """Oldest queued seed's bytes, or None when empty."""
        seed = self.pop_seed()
        return None if seed is None else seed.data

    def pop_seed(self) -> TcpSeed | None:
        """Oldest queued seed with its id, or None when empty."""
        if not self._queue:
            return None
        return self._queue.popleft()

    def close(self) -> None:
        """Stop accepting, end open connections, release the port. Idempotent.

        Open connections are shut down rather than waited for, so a stalled
        client cannot hold close() for conn_timeout; their bytes are dropped.
        """
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join()
        self._sock.close()

        with self._lock:
            active = dict(self._conns)
        for conn in active.values():
            # The reader may have closed it already: nothing left to end.
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
        for reader in active:
            reader.join()

    def _loop(self) -> None:
        """Accept until close(), one reader thread per connection, at most max_connections."""
        while not self._stop.is_set():
            if not self._slots.acquire(timeout=ACCEPT_POLL_S):
                continue  # all readers busy: clients wait in the backlog
            try:
                conn, addr = self._sock.accept()
            except TimeoutError:
                self._slots.release()
                continue
            except OSError as e:
                # EMFILE and friends: back off instead of spinning.
                self._slots.release()
                log.warning("tcp seed accept failed: %s", e)
                self._stop.wait(ACCEPT_POLL_S)
                continue
            self._spawn(conn, addr)

    def _spawn(self, conn: socket.socket, addr: tuple) -> None:
        """Start a reader for *conn*; registered before start so it can deregister itself."""
        reader = threading.Thread(
            target=self._serve, args=(conn, addr), name="tcp-seed-conn", daemon=True
        )
        with self._lock:
            self._conns[reader] = conn
        try:
            reader.start()
        except RuntimeError as e:
            # "can't start new thread": undo what _serve would have, keep accepting.
            with self._lock:
                self._conns.pop(reader, None)
            conn.close()
            self._slots.release()
            log.warning("tcp seed from %s dropped: reader did not start: %s", _peer(addr), e)

    def _serve(self, conn: socket.socket, addr: tuple) -> None:
        """Read one message and queue it; always frees the slot and the socket."""
        try:
            data = self._read(conn)
            # Shut down by close(): the run is ending, the bytes go nowhere.
            if not self._stop.is_set():
                self._enqueue(data, _peer(addr))
        except Exception:
            # One bad client must not kill its reader silently (bug-classes).
            log.warning("tcp seed from %s failed", _peer(addr), exc_info=True)
        finally:
            conn.close()
            with self._lock:
                self._conns.pop(threading.current_thread(), None)
            self._slots.release()

    def _read(self, conn: socket.socket) -> bytes:
        """Bytes until EOF, max_bytes or the connection deadline."""
        buf = bytearray()
        deadline = time.monotonic() + self._conn_timeout
        cap = self._max_bytes

        while len(buf) < cap:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            conn.settimeout(left)
            try:
                chunk = conn.recv(min(RECV_CHUNK, cap - len(buf)))
            except TimeoutError:
                break
            except OSError as e:
                # Peer reset mid-message: keep what arrived.
                log.debug("tcp seed recv: %s", e)
                break
            if not chunk:
                break
            buf += chunk

        return bytes(buf)

    def _enqueue(self, data: bytes, peer: str) -> None:
        """Queue *data* and announce it; empty messages (port probes) are not seeds.

        Announced before the counters move, so a reader polling them sees the line.
        The id is the corpus file name the seed gets if admitted (``id_<h>``).
        """
        if not data:
            return
        # Hashed outside the lock: ~34 ms/MiB, other readers keep going; the
        # id rides along so the fuzz loop never hashes it again.
        seed_id = hash_data(data)
        seed = f"seed id_{seed_id} with {len(data)} bytes from {peer}"

        with self._lock:
            if len(self._queue) >= self._capacity:
                _say(f"[!] Dropped {seed}: inbox full")
                self._dropped += 1
                return
            _say(f"[*] Received {seed}")
            self._queue.append(TcpSeed(data, seed_id))
            self._received += 1


def _say(line: str) -> None:
    """Print *line* from a reader thread in one write.

    print() writes the text and the newline separately, so another thread's
    line could land between them ("...:54218[*] Ran seed ...").
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _peer(addr: tuple) -> str:
    """``ip:port``, bracketing IPv6 (``[::1]:5000``)."""
    host, port = addr[0], addr[1]
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
