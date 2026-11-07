"""Driver for the per-exec C scan kernels (``native_scan.c``).

Compiles the source once into ``~/.cache/fuzzer_native/`` (keyed by its
content hash, so an edit rebuilds and other processes reuse the build),
loads it with ctypes and exposes numpy-level calls. Callers work in terms
of "live entries", "fold" and "drain"; pointers and scratch stay here.

    ShmCoverage._scan      -> LiveScan.live / live_touched
    ShmCoverage._scan_new  -> LiveScan.fold, or LiveScan.live_fold (gather + fold)
    CmpCounterShm.drain_*  -> CmpDrain.counts / sites
    CrashMITracker.record  -> MIBump.bump

``load()`` returns None when no compiler is available; callers keep their
numpy paths for that case.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import subprocess
import tempfile
from typing import NamedTuple

import numpy as np

from fuzzer_tool.core.count_class import BUCKET_BIT_TABLE
from fuzzer_tool.core.hashing import hash64

log = logging.getLogger(__name__)

_SRC = os.path.join(os.path.dirname(__file__), "native_scan.c")
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fuzzer_native")
_COMPILERS = ("clang", "gcc")
_COMPILE_TIMEOUT = 60

# Index-matched to the ST_* enum in native_scan.c.
_ST_TRANSITIONS, _ST_OLD_MOVED, _ST_GAIN, _ST_LIVE, _ST_FIELDS = range(5)

# fs_fold's return when an id is past the virgin or max array.
_FOLD_GROW = -1

_P = ctypes.c_void_p
_L = ctypes.c_long

_SIGNATURES = {
    "fs_live": [_P, _L, ctypes.c_uint32, _P, _P],
    "fs_live_touched": [_P, _L, ctypes.c_uint32, _P, _L, _P, _P],
    "fs_fold": [_P, _P, _L, _P, _L, _P, _L, _P, ctypes.c_double, _P, _P, _P],
    "fs_live_fold": [_P, _P, _L, ctypes.c_uint32, _P, _L],
    "fs_drain_counts": [_P, _P, _L, _P, _P, _P],
    "fs_drain_sites": [_P, _L, _P, _P, _P],
    "fs_mi_bump": [ctypes.c_char_p, _L, _L, _P],
}

# Loaded library, or False once a build failed (don't retry per instance).
_cached: ctypes.CDLL | bool | None = None


def _clean_env() -> dict[str, str]:
    """Compiler env without the target's sanitizer runtime."""
    env = os.environ.copy()
    for key in ("ASAN_OPTIONS", "LSAN_OPTIONS", "LD_PRELOAD"):
        env.pop(key, None)
    return env


def _compile(out: str) -> bool:
    """Build *out* atomically: compile to a temp name, then rename."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=_CACHE_DIR)
    os.close(fd)

    # clang first (Hard Rule 4), gcc only as a fallback.
    for cc in _COMPILERS:
        cmd = [cc, "-O2", "-shared", "-fPIC", "-o", tmp, _SRC, "-lm"]
        try:
            res = subprocess.run(
                cmd, capture_output=True, timeout=_COMPILE_TIMEOUT, env=_clean_env()
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.debug("native_scan: %s failed: %s", cc, exc)
            continue
        if res.returncode == 0:
            os.replace(tmp, out)
            return True
        log.debug("native_scan: %s failed: %s", cc, res.stderr.decode()[:400])

    with contextlib.suppress(OSError):
        os.unlink(tmp)
    return False


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    for name, args in _SIGNATURES.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = _L
    return lib


def _build() -> ctypes.CDLL | None:
    with open(_SRC, "rb") as fh:
        tag = f"{hash64(fh.read()):016x}"
    so = os.path.join(_CACHE_DIR, f"native_scan_{tag}.so")

    if not os.path.exists(so) and not _compile(so):
        log.warning("native_scan: no compiler, using numpy scan paths")
        return None
    try:
        return _bind(ctypes.CDLL(so))
    except OSError as exc:
        log.warning("native_scan: load failed (%s), using numpy scan paths", exc)
        return None


def load() -> ctypes.CDLL | None:
    """The kernel library, built on first call; None when it cannot build."""
    global _cached
    if _cached is None:
        _cached = _build() or False
    return _cached or None


def _addr(a: np.ndarray) -> int:
    return a.__array_interface__["data"][0]


class FoldOut(NamedTuple):
    """One fold: ids with virgin 0 before it (and their bits), plus counters."""

    cand_ids: np.ndarray
    cand_bits: np.ndarray
    transitions: int
    old_moved: bool
    gain: int


class _FoldState(ctypes.Structure):
    """Mirrors ``struct fold_state`` in native_scan.c."""

    _fields_ = [
        ("ids", _P),
        ("counts", _P),
        ("cand_ids", _P),
        ("cand_bits", _P),
        ("stats", _P),
        ("bucket", _P),
        ("virgin", _P),
        ("vsize", _L),
        ("maxc", _P),
        ("msize", _L),
        ("factor", ctypes.c_double),
    ]


class LiveScan:
    """Edge-table gather and virgin/max fold, with reusable scratch."""

    def __init__(self, lib: ctypes.CDLL):
        self._live = lib.fs_live
        self._live_touched = lib.fs_live_touched
        self._fold = lib.fs_fold
        self._live_fold = lib.fs_live_fold
        self._bucket = BUCKET_BIT_TABLE
        self._bucket_addr = _addr(self._bucket)
        self._stats = np.zeros(_ST_FIELDS, dtype=np.uint64)
        self._stats_addr = _addr(self._stats)

        # live_fold's bound arrays: rebound when scratch, virgin or maxc move.
        self._state = _FoldState(stats=self._stats_addr, bucket=self._bucket_addr)
        self._state_ref = ctypes.byref(self._state)
        self._bound: tuple = ()
        self._cap = -1
        self._grow(0)

    def _grow(self, n: int) -> None:
        """Scratch for *n* entries; reallocated only when it must grow."""
        if n <= self._cap:
            return
        self._cap = max(n, 2 * self._cap)
        self._ids = np.empty(self._cap, dtype=np.uint32)
        self._counts = np.empty(self._cap, dtype=np.uint32)
        self._cand_ids = np.empty(self._cap, dtype=np.uint32)
        self._cand_bits = np.empty(self._cap, dtype=np.uint8)
        self._ids_addr, self._counts_addr = _addr(self._ids), _addr(self._counts)
        self._bits_addr = _addr(self._cand_bits)
        self._state.ids, self._state.counts = self._ids_addr, self._counts_addr
        self._state.cand_ids, self._state.cand_bits = _addr(self._cand_ids), self._bits_addr

    def _copy_out(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        # Copies: callers cache these (scan memo, known paths).
        return self._ids[:k].copy(), self._counts[:k].copy()

    def live(self, table: int, n: int, gen: int) -> tuple[np.ndarray, np.ndarray]:
        """``(edge_id, count)`` of entries live in *gen*, ascending slot order."""
        self._grow(n)
        k = self._live(table, n, gen, self._ids_addr, self._counts_addr)
        return self._copy_out(k)

    def live_touched(
        self, table: int, n: int, gen: int, words: int, nwords: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`live` over the slots set in the touched bitmap only."""
        self._grow(n)
        k = self._live_touched(table, n, gen, words, nwords, self._ids_addr, self._counts_addr)
        return self._copy_out(k)

    def fold(
        self,
        ids: np.ndarray,
        counts: np.ndarray,
        virgin: np.ndarray,
        maxc: np.ndarray,
        factor: float,
    ) -> FoldOut | None:
        """Fold *ids*/*counts* into *virgin*/*maxc* in place.

        Returns None, with nothing written, when an id is past either array.
        Candidate arrays are views of scratch: valid until the next call.
        """
        ids = np.ascontiguousarray(ids, dtype=np.uint32)
        counts = np.ascontiguousarray(counts, dtype=np.uint32)
        n = ids.size
        self._grow(n)

        # Candidates go to the ids scratch: fold reads ids from the caller's array.
        nc = self._fold(
            _addr(ids), _addr(counts), n,
            _addr(virgin), virgin.size, _addr(maxc), maxc.size,
            self._bucket_addr, factor,
            self._ids_addr, self._bits_addr, self._stats_addr,
        )  # fmt: skip
        if nc == _FOLD_GROW:
            return None

        st = self._stats.tolist()
        return FoldOut(
            self._ids[:nc],
            self._cand_bits[:nc],
            st[_ST_TRANSITIONS],
            bool(st[_ST_OLD_MOVED]),
            st[_ST_GAIN],
        )

    def live_fold(
        self,
        table: int,
        n: int,
        gen: int,
        words: int,
        nwords: int,
        virgin: np.ndarray,
        maxc: np.ndarray,
        factor: float,
    ) -> tuple[np.ndarray, np.ndarray, FoldOut | None]:
        """:meth:`live` (or :meth:`live_touched` when *words*) then :meth:`fold`, one call.

        Returns copies of the live ids/counts and the fold; the fold is None,
        with nothing written, when an id is past *virgin* or *maxc* (grow
        them and call :meth:`fold` on the returned columns).
        """
        self._grow(n)
        self._bind(virgin, maxc, factor)
        nc = self._live_fold(self._state_ref, table, n, gen, words, nwords)
        k = int(self._stats[_ST_LIVE])
        ids, counts = self._copy_out(k)
        if nc == _FOLD_GROW:
            return ids, counts, None

        st = self._stats.tolist()
        out = FoldOut(
            self._cand_ids[:nc],
            self._cand_bits[:nc],
            st[_ST_TRANSITIONS],
            bool(st[_ST_OLD_MOVED]),
            st[_ST_GAIN],
        )
        return ids, counts, out

    def _bind(self, virgin: np.ndarray, maxc: np.ndarray, factor: float) -> None:
        """Point the fold state at *virgin*/*maxc*; skipped while they are the same objects."""
        b = self._bound
        if b and b[0] is virgin and b[1] is maxc and b[2] == factor:
            return

        st = self._state
        st.virgin, st.vsize = _addr(virgin), virgin.size
        st.maxc, st.msize = _addr(maxc), maxc.size
        st.factor = factor
        self._bound = (virgin, maxc, factor)


class CmpDrain:
    """Read-and-zero of the cmplog counter segment."""

    def __init__(self, lib: ctypes.CDLL, n_counts: int, n_sites: int):
        self._counts = lib.fs_drain_counts
        self._sites = lib.fs_drain_sites
        self._n_counts, self._n_sites = n_counts, n_sites
        cap = max(n_counts, n_sites)
        self._out = np.empty((3, cap), dtype=np.int64)
        self._a, self._b, self._c = (_addr(row) for row in self._out)

    def counts(self, fired: int, hit: int) -> tuple[list[int], list[int], list[int]]:
        """``(index, fired, hit)`` lists of callbacks that fired; zeroes them."""
        k = self._counts(fired, hit, self._n_counts, self._a, self._b, self._c)
        idx, f, h = self._out[:, :k].tolist()
        return idx, f, h

    def sites(self, sites: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(slots, fired, hit)`` of sites that fired; zeroes counters, keeps keys."""
        k = self._sites(sites, self._n_sites, self._a, self._b, self._c)
        out = self._out[:, :k].copy()  # kept by the caller's fold
        return out[0], out[1], out[2]


class _MIState(ctypes.Structure):
    """Mirrors ``struct mi_state`` in native_scan.c."""

    _fields_ = [
        ("total", _P),
        ("crash", _P),
        ("pos_counts", _P),
        ("crash_rows", _P),
        ("s_total", _P),
        ("s_crash", _P),
        ("s_clean", _P),
        ("step", _P),
        ("step_size", _L),
    ]


# (attribute, dtype) of each array MIBump.bind takes, in _MIState order.
_MI_ARRAYS = (
    ("total", np.uint32),
    ("crash", np.uint32),
    ("pos_counts", np.uint64),
    ("crash_rows", np.uint64),
    ("s_total", np.float64),
    ("s_crash", np.float64),
    ("s_clean", np.float64),
)


class MIBump:
    """CrashMITracker histogram + running-sum update, one C pass per exec.

    Array addresses are bound once (``bind``) and held, so a call passes
    four arguments. Rebind whenever the caller replaces an array: the held
    references keep the old ones alive, so a missed rebind writes stale
    arrays but never freed memory.
    """

    def __init__(self, lib: ctypes.CDLL, step: np.ndarray):
        self._fn = lib.fs_mi_bump
        self._step = np.ascontiguousarray(step, dtype=np.float64)
        self._state = _MIState()
        self._ref = ctypes.byref(self._state)
        self._held: tuple[np.ndarray, ...] = ()

    def bind(self, *arrays: np.ndarray) -> None:
        """Point the kernel at *arrays*, in ``_MI_ARRAYS`` order."""
        for arr, (name, dtype) in zip(arrays, _MI_ARRAYS, strict=True):
            if arr.dtype != dtype or not arr.flags.c_contiguous:
                raise ValueError(f"MIBump.bind: {name} must be C-contiguous {dtype.__name__}")

        for arr, (name, _) in zip(arrays, _MI_ARRAYS, strict=True):
            setattr(self._state, name, _addr(arr))
        self._state.step = _addr(self._step)
        self._state.step_size = self._step.size
        self._held = arrays

    def bump(self, data: bytes, n: int, is_crash: bool) -> None:
        """Count *data*[:n] (one byte per position) under *is_crash*."""
        self._fn(data, n, is_crash, self._ref)


def mi_bump(step: np.ndarray) -> MIBump | None:
    """An unbound MIBump over *step*; None when the kernel cannot build."""
    lib = load()
    return MIBump(lib, step) if lib is not None else None
