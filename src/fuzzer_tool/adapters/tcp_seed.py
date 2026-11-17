"""TCP seed inbox for ``--tcp-listen-port``.

One connection = one seed: the listener reads until the peer closes, then
queues the bytes. ``OperatorEngine.mutate()`` pops one per round, so each seed
runs the normal exec / coverage / crash / admit path.

    client --connect, send, close--> [listener thread] --append--> deque
                                                                     |
                                     OperatorEngine.mutate() <--pop--+

One listener thread serves connections in turn. Memory is bounded by
``capacity * max_bytes``; a stalled client holds the listener for at most
``conn_timeout`` and its partial bytes are kept as the message.

    printf 'GIF89a...' | nc -N 127.0.0.1 4444
"""

from __future__ import annotations

import collections
import logging
import socket
import threading
import time

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
LISTEN_BACKLOG = 16


class TcpSeedInbox:
    """Bounded queue of seeds received over TCP, filled by a daemon thread.

    Binds in the constructor, so a busy port fails at startup (``OSError``).
    """

    def __init__(
        self,
        port: int,
        host: str = DEFAULT_HOST,
        max_bytes: int = DEFAULT_MAX_BYTES,
        capacity: int = INBOX_CAPACITY,
        conn_timeout: float = CONN_TIMEOUT_S,
    ) -> None:
        self._max_bytes = max_bytes
        self._capacity = capacity
        self._conn_timeout = conn_timeout

        # Single producer (listener), single consumer (fuzz loop): deque
        # append/popleft are atomic, so no lock.
        self._queue: collections.deque[bytes] = collections.deque()
        self._received = 0
        self._dropped = 0

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

    def pop(self) -> bytes | None:
        """Oldest queued seed, or None when empty."""
        if not self._queue:
            return None
        return self._queue.popleft()

    def close(self) -> None:
        """Stop the listener and release the port. Idempotent."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join()
        self._sock.close()

    def _loop(self) -> None:
        """Accept connections until close(); one bad client never ends the loop."""
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError as e:
                # EMFILE and friends: back off instead of spinning.
                log.warning("tcp seed accept failed: %s", e)
                self._stop.wait(ACCEPT_POLL_S)
                continue

            with conn:
                data = self._read(conn)
            self._enqueue(data)

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

    def _enqueue(self, data: bytes) -> None:
        """Queue *data*; empty messages (port probes) are not seeds."""
        if not data:
            return
        if len(self._queue) >= self._capacity:
            self._dropped += 1
            return
        self._queue.append(data)
        self._received += 1
