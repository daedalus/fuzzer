"""Memory-backed input file (wtf fshooks port).

wtf serves file reads from the testcase buffer instead of disk. Here the
input lives in a memfd: a real file the kernel keeps in RAM, reachable by a
path that never changes for the life of the fuzzer.

    fuzzer: MemfdInput.write(data)      target: open("/proc/<pid>/fd/<n>")
              └── pwrite(fd, data, 0)            └── fresh file description,
                  ftruncate on shrink                own offset, fstat/mmap OK

One path for every exec is what lets file-mode targets run under the
forkserver (argv fixed at exec) and gives in-process harnesses a path
(``FUZZ_INPUT_PATH``). ``/proc/<pid>`` rather than ``/proc/self`` so a
child resolves the fuzzer's fd without inheriting it.
"""

import os

# Exported to targets: the path holding the current input.
INPUT_PATH_ENV = "FUZZ_INPUT_PATH"
# A hardcoded path the shim's open() hooks swap for INPUT_PATH_ENV (afl_shim.c).
ALIAS_ENV = "FUZZ_INPUT_ALIAS"
_MEMFD_NAME = "fuzz_input"


class MemfdInput:
    """One reusable in-memory input file."""

    def __init__(self) -> None:
        # CLOEXEC: children reach it through /proc/<pid>/fd, not inheritance.
        self._fd = os.memfd_create(_MEMFD_NAME, os.MFD_CLOEXEC)
        self._size = 0
        self.path = f"/proc/{os.getpid()}/fd/{self._fd}"

    def write(self, data: bytes) -> None:
        """Make *data* the file's whole content: one pwrite, ftruncate only on shrink."""
        n = len(data)
        if n < self._size:
            os.ftruncate(self._fd, n)
        if n:
            os.pwrite(self._fd, data, 0)
        self._size = n

    def close(self) -> None:
        """Release the memfd; later calls are no-ops."""
        if self._fd < 0:
            return
        os.close(self._fd)
        self._fd = -1

    def __del__(self) -> None:
        self.close()
