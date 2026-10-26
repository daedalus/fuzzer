"""glibc heap release: hand freed malloc pages back to the kernel.

CPython frees into glibc, which keeps the pages mapped. Measured on
fuzzgoat: 5-7 MB of RSS reclaimable by ``malloc_trim(0)`` after 26k execs.
"""

import ctypes
import ctypes.util
import os

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
    _malloc_trim(_NO_PAD)
    return max(0, before - _rss_bytes())
