"""glibc heap release: hand freed malloc pages back to the kernel.

CPython frees into glibc, which keeps the pages mapped. Measured on
fuzzgoat: 5-7 MB of RSS reclaimable by ``malloc_trim(0)`` after 26k execs.
"""

import ctypes
import ctypes.util
import logging
import os
import threading

log = logging.getLogger(__name__)

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")

# glibc-only; musl and macOS libc lack it.
_malloc_trim = getattr(_libc, "malloc_trim", None)
if _malloc_trim is not None:
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int

# Slack left at the heap top: none, release everything releasable.
_NO_PAD = 0

# Set once an in-process target aborted on the shared glibc heap: the arena
# lock may be held forever, and malloc_trim (which locks every arena) would
# block the process for good.
_heap_tainted = False


def mark_heap_tainted() -> None:
    """Record that a target crash may have left a glibc arena lock held."""
    global _heap_tainted
    _heap_tainted = True


def heap_tainted() -> bool:
    return _heap_tainted


_STATM = "/proc/self/statm"
_PAGE = os.sysconf("SC_PAGE_SIZE")


def _rss_bytes() -> int:
    """Current (not peak) resident set size; statm field 1 is resident pages."""
    with open(_STATM) as fh:
        return int(fh.read().split()[1]) * _PAGE


def current_rss() -> int:
    """Current resident set size in bytes (``rss:`` on the stats line is the peak)."""
    return _rss_bytes()


def trim_heap() -> int:
    """Release free heap memory to the OS.

    Returns:
        Bytes of RSS released; 0 if none, no ``malloc_trim``, or the heap is tainted
        (see ``mark_heap_tainted``). Clamped at 0:
        another thread can grow RSS during the call.
    """
    if _malloc_trim is None or _heap_tainted:
        return 0

    before = _rss_bytes()
    if not _trim_bounded():
        return 0
    return max(0, before - _rss_bytes())


# malloc_trim locks every arena. One left held by a call the guard abandoned
# (a crash or timeout that landed inside glibc) blocks it for good, so the
# trim runs on a helper thread and the caller gives up after this long.
_TRIM_TIMEOUT_S = 10.0


def _trim_bounded() -> bool:
    """malloc_trim(0) with a deadline; False (and tainted) if it did not return."""
    worker = threading.Thread(target=_malloc_trim, args=(_NO_PAD,), daemon=True, name="malloc_trim")
    worker.start()
    worker.join(_TRIM_TIMEOUT_S)
    if not worker.is_alive():
        return True
    log.error(
        "malloc_trim did not return in %.0fs: a glibc arena lock is held (an in-process "
        "target died inside malloc/free). Heap trimming is disabled; the host may "
        "deadlock on its next large allocation. Rebuild the target with "
        "-D__AFL_PRIVATE_HEAP=1 or use spawn/loader mode.",
        _TRIM_TIMEOUT_S,
    )
    mark_heap_tainted()
    return False
