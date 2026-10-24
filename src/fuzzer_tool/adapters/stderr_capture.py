"""Capture fd 2 around an in-process target call, reusing one pipe.

    begin():  fd 2 ──dup2──> pipe write end   (target's ASAN report lands here)
    finish(): fd 2 <──dup2── saved original;  drain pipe read end -> str

The pipe, the saved fd and the non-blocking flag are set up once; per exec
the cost is two ``dup2`` and the drain reads, against ~10 syscalls for a
fresh pipe per exec.
"""

from __future__ import annotations

import fcntl
import logging
import os

log = logging.getLogger(__name__)

_CHUNK = 65536
# Pipe capacity asked for at open. An in-process target blocks forever once
# its unread stderr fills the pipe; the default 64 KiB is one large report.
_PIPE_BYTES = 1 << 20
_STDERR_FD = 2


class StderrCapture:
    """Redirect fd 2 into a persistent pipe between ``begin`` and ``finish``."""

    def __init__(self):
        self._r: int | None = None
        self._w: int | None = None
        self._saved: int | None = None

    def _open(self) -> None:
        self._r, self._w = os.pipe()
        os.set_blocking(self._r, False)
        try:
            fcntl.fcntl(self._w, fcntl.F_SETPIPE_SZ, _PIPE_BYTES)
        except OSError:
            log.debug("F_SETPIPE_SZ refused; keeping default pipe size", exc_info=True)
        self._saved = os.dup(_STDERR_FD)

    def begin(self) -> None:
        if self._r is None:
            self._open()
        os.dup2(self._w, _STDERR_FD)

    def finish(self) -> str:
        """Restore fd 2 and return everything written since ``begin``."""
        os.dup2(self._saved, _STDERR_FD)

        # A short read means the pipe is empty: no trailing EAGAIN syscall.
        parts = []
        while True:
            try:
                chunk = os.read(self._r, _CHUNK)
            except BlockingIOError:
                break
            parts.append(chunk)
            if len(chunk) < _CHUNK:
                break
        return b"".join(parts).decode(errors="replace")

    def _fds(self) -> tuple[int, ...]:
        return tuple(fd for fd in (self._r, self._w, self._saved) if fd is not None)

    def close(self) -> None:
        for fd in self._fds():
            os.close(fd)
        self._r = self._w = self._saved = None
