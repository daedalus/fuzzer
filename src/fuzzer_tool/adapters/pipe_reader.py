"""Deadline-bounded reads from a loader's stdout pipe.

Replaces the per-exec ``threading.Thread`` + ``readline()`` + ``join(timeout)``
pattern (~190 us spawn + join per exec) with ``poll`` on the raw fd.

    loader stdout ──fd──> PipeReader._buf ──> readline() / read_exact()

The fd is read directly, never through the Popen ``BufferedReader``; mixing
the two would strand bytes in the wrapper's buffer where ``poll`` cannot see
them. Bytes past a line stay in ``_buf`` for the next call.
"""

from __future__ import annotations

import math
import os
import select
import time

_CHUNK = 65536
_NEWLINE = b"\n"
_MS_PER_S = 1000

# Deadline meaning "block until data or EOF" (startup handshakes).
NO_DEADLINE = math.inf


class PipeReader:
    """Buffered reader over a raw pipe fd with per-call deadlines."""

    def __init__(self, fd: int):
        self._fd = fd
        self._buf = bytearray()
        self._poll = select.poll()
        self._poll.register(fd, select.POLLIN)

    def _fill(self, deadline: float) -> bool | None:
        """Read one chunk. True on data, False on EOF, None on timeout."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None

        # poll(None) blocks; ceil() of inf would raise.
        timeout_ms = None if deadline == NO_DEADLINE else math.ceil(remaining * _MS_PER_S)
        if not self._poll.poll(timeout_ms):
            return None

        chunk = os.read(self._fd, _CHUNK)
        if not chunk:
            return False

        self._buf += chunk
        return True

    def readline(self, deadline: float) -> bytes | None:
        """Next line incl. newline; ``None`` on timeout, ``b""`` at EOF.

        At EOF a trailing partial line is returned once, as ``readline()``
        does. A timeout keeps any partial line buffered.
        """
        while True:
            idx = self._buf.find(_NEWLINE)
            if idx >= 0:
                line = bytes(self._buf[: idx + 1])
                del self._buf[: idx + 1]
                return line

            got = self._fill(deadline)
            if got is None:
                return None
            if got:
                continue

            line = bytes(self._buf)
            self._buf.clear()
            return line

    def read_exact(self, n: int, deadline: float) -> bytes | None:
        """Exactly *n* bytes; ``None`` on timeout or EOF before *n*."""
        while len(self._buf) < n:
            if not self._fill(deadline):
                return None

        data = bytes(self._buf[:n])
        del self._buf[:n]
        return data
