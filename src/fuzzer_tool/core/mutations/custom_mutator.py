"""libFuzzer-style custom mutator / crossover plugins (``--custom-mutator``).

A plugin is a shared object exporting either or both of::

    size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                   size_t MaxSize, unsigned int Seed);

    size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
                                     const uint8_t *Data2, size_t Size2,
                                     uint8_t *Out, size_t MaxOutSize,
                                     unsigned int Seed);

so a harness written for libFuzzer / OSS-Fuzz (or for atheris, whose
``Setup(custom_mutator=..., custom_crossover=...)`` mirrors these) can be
reused without change.  Each exported symbol becomes one operator
(``custom_mutator``, ``custom_crossover``) in ``REGISTRY``, so the operator
schedulers weigh it against the built-in mutators like any other.

Two callbacks, two directions
-----------------------------

* fuzzer -> plugin: the two functions above.  The wrapper copies the seed
  into a ``MaxSize`` buffer, calls the plugin, and returns the first
  ``ret`` bytes as a new ``bytes``.  A return of 0 or above ``MaxSize`` is
  a decline (``None``).
* plugin -> fuzzer: ``size_t LLVMFuzzerMutate(uint8_t *Data, size_t Size,
  size_t MaxSize)``.  A typical plugin parses the input, mutates the
  parsed structure, and calls this to apply ordinary byte-level havoc to
  one field.  The plugin leaves the symbol undefined and expects the host
  to provide it, but this host is Python, so a few lines of C (built with
  the system ``cc`` on first use, loaded ``RTLD_GLOBAL`` before the
  plugin) export the symbol and forward to a ``ctypes`` callback that runs
  :func:`havoc`.

The havoc behind ``LLVMFuzzerMutate`` is self-contained (bit/byte flips,
interesting values, arithmetic, block delete/duplicate/insert/copy/swap),
not a reach into ``OperatorEngine``: ``core/mutator_interface.py`` exists to
stop mutators depending on the whole fuzzer, and a callback invoked from
foreign C has no engine to hand anyway.  It draws from the fuzzer's own rng,
so runs stay reproducible under ``--seed``.

Limits
------

* Crash isolation is a guard, not a process boundary.  The plugin still
  runs inside the fuzzer; its calls are wrapped in C by ``run_guarded``:
  fault handlers for SIGSEGV/SIGBUS/SIGFPE/SIGILL/SIGABRT/SIGTRAP on an
  alternate stack (so stack overflow is caught too) plus a POSIX timer
  (``--custom-mutator-timeout``, default 1000 ms) for hangs, with
  ``siglongjmp`` back to the call.  Handlers and the timer exist only for
  the duration of a call, and the previous dispositions are restored, so
  the fuzzer's own handlers are untouched.  A faulting call returns
  ``None`` and is counted in ``CustomMutatorPlugin.failures``, with the
  offending input in ``last_failure_input``.  What a guard cannot undo:
  memory the plugin corrupted before faulting, and a fault inside glibc
  malloc/free, which leaves the arena lock held and can deadlock the next
  allocation (the hazard noted for persistent mode).  Both are bugs in the
  plugin; ``--custom-mutator-isolation none`` removes the guard.
  ``LLVMFuzzerMutate`` copies the plugin's buffer into host memory before
  Python sees it, so a wild pointer passed to it faults in C, where the
  guard recovers, never inside a Python frame.  A fault inside the Python
  callback itself is not recoverable and ends the process.
* One plugin per process: the ``LLVMFuzzerMutate`` forwarder is a single
  global symbol.  A second ``register_custom_mutator`` call rebinds the
  existing operators to the new plugin.
* Needs a C compiler (``$CC`` or ``cc``) the first time.
* AFL++'s ``afl_custom_*`` API is not implemented.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import logging
import os
import random
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase

log = logging.getLogger(__name__)

#: Buffer capacity handed to the plugin when the fuzzer has no length cap
#: (``max_len == 0``).  libFuzzer's own default ``-max_len`` is 4096; the
#: capacity grows with the input so a long seed is never truncated first.
DEFAULT_CAP = 4096

#: Per-call wall-clock budget for a plugin call under the guard.
DEFAULT_TIMEOUT_MS = 1000

#: Status the C guard returns for a timeout (any other non-zero is a signal number).
_FZ_TIMEOUT = 1000

MUTATOR_SYMBOL = "LLVMFuzzerCustomMutator"
CROSSOVER_SYMBOL = "LLVMFuzzerCustomCrossOver"

_HOST_SRC = r"""
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef sigev_notify_thread_id
#define sigev_notify_thread_id _sigev_un._tid
#endif

#define FZ_TIMEOUT 1000
#define FZ_SCRATCH_MAX (16u << 20)
#define FZ_ALTSTK (64 * 1024)

typedef size_t (*fz_mutate_cb)(uint8_t *, size_t, size_t);
typedef size_t (*fz_mut_fn)(uint8_t *, size_t, size_t, unsigned);
typedef size_t (*fz_cross_fn)(const uint8_t *, size_t, const uint8_t *, size_t,
                              uint8_t *, size_t, unsigned);

static fz_mutate_cb g_cb;
static __thread sigjmp_buf tl_env;
static __thread volatile sig_atomic_t tl_guarded, tl_in_cb, tl_pending_timeout;
static __thread uint8_t *tl_scratch;
static __thread size_t tl_scratch_cap;
static __thread char *tl_altstk;

void fz_set_mutate_cb(fz_mutate_cb cb) { g_cb = cb; }

/* LLVMFuzzerMutate. The plugin's buffer is never handed to Python: it is
 * copied into host memory first, so a wild pointer from the plugin faults
 * here, in C, where the guard can recover, not inside a Python frame. */
size_t LLVMFuzzerMutate(uint8_t *d, size_t n, size_t max_size) {
    if (!g_cb || max_size > FZ_SCRATCH_MAX || n > max_size) return n;
    size_t cap = max_size ? max_size : 1;
    if (tl_scratch_cap < cap) {
        uint8_t *p = realloc(tl_scratch, cap);
        if (!p) return n;
        tl_scratch = p;
        tl_scratch_cap = cap;
    }
    memcpy(tl_scratch, d, n);
    tl_in_cb = 1;
    size_t m = g_cb(tl_scratch, n, max_size);
    tl_in_cb = 0;
    if (tl_pending_timeout) {
        tl_pending_timeout = 0;
        if (tl_guarded) siglongjmp(tl_env, FZ_TIMEOUT);
    }
    if (m > max_size) m = max_size;
    memcpy(d, tl_scratch, m);
    return m;
}

static void on_fault(int sig, siginfo_t *si, void *uc) {
    (void)si; (void)uc;
    if (tl_guarded && !tl_in_cb) siglongjmp(tl_env, sig);
    /* Fault inside the Python callback (or unguarded): not recoverable.
     * Restore the default action; returning re-raises it and ends the process. */
    signal(sig, SIG_DFL);
}

static void on_timeout(int sig, siginfo_t *si, void *uc) {
    (void)sig; (void)si; (void)uc;
    if (!tl_guarded) return;
    if (tl_in_cb) { tl_pending_timeout = 1; return; }
    siglongjmp(tl_env, FZ_TIMEOUT);
}

static const int k_sigs[] = {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, SIGTRAP};
#define N_SIGS 6

/* Runs thunk(arg) with fault handlers (and an optional timer) installed for
 * its duration only, and the previous dispositions restored afterwards.
 * Returns 0, a signal number, or FZ_TIMEOUT. */
static int run_guarded(void (*thunk)(void *), void *arg, int timeout_ms) {
    struct sigaction sa, old[N_SIGS], old_rt;
    stack_t ss, old_ss;
    timer_t tm;
    int have_timer = 0, have_rt = 0, rc, i;

    if (!tl_altstk) tl_altstk = malloc(FZ_ALTSTK);
    if (tl_altstk) {
        ss.ss_sp = tl_altstk; ss.ss_size = FZ_ALTSTK; ss.ss_flags = 0;
        if (sigaltstack(&ss, &old_ss) != 0) old_ss.ss_flags = SS_DISABLE;
    }
    memset(&sa, 0, sizeof sa);
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_NODEFER | (tl_altstk ? SA_ONSTACK : 0);
    sa.sa_sigaction = on_fault;
    for (i = 0; i < N_SIGS; i++) sigaction(k_sigs[i], &sa, &old[i]);

    if (timeout_ms > 0) {
        struct sigaction st = sa;
        st.sa_sigaction = on_timeout;
        if (sigaction(SIGRTMIN, &st, &old_rt) == 0) {
            struct sigevent ev;
            memset(&ev, 0, sizeof ev);
            ev.sigev_notify = SIGEV_THREAD_ID;
            ev.sigev_signo = SIGRTMIN;
            ev.sigev_notify_thread_id = (pid_t)syscall(SYS_gettid);
            have_rt = 1;
            if (timer_create(CLOCK_MONOTONIC, &ev, &tm) == 0) {
                struct itimerspec its;
                memset(&its, 0, sizeof its);
                its.it_value.tv_sec = timeout_ms / 1000;
                its.it_value.tv_nsec = (long)(timeout_ms % 1000) * 1000000L;
                if (timer_settime(tm, 0, &its, NULL) == 0) have_timer = 1;
                else timer_delete(tm);
            }
        }
    }

    tl_in_cb = 0;
    tl_pending_timeout = 0;
    rc = sigsetjmp(tl_env, 1);
    if (rc == 0) {
        tl_guarded = 1;
        thunk(arg);
    }
    tl_guarded = 0;
    tl_in_cb = 0;

    if (have_timer) timer_delete(tm);
    if (have_rt) sigaction(SIGRTMIN, &old_rt, NULL);
    for (i = 0; i < N_SIGS; i++) sigaction(k_sigs[i], &old[i], NULL);
    if (tl_altstk) sigaltstack(&old_ss, NULL);
    return rc;
}

struct mut_args { fz_mut_fn f; uint8_t *d; size_t n, max; unsigned seed; size_t ret; };
static void mut_thunk(void *p) {
    struct mut_args *a = p;
    a->ret = a->f(a->d, a->n, a->max, a->seed);
}
int fz_guard_mutate(fz_mut_fn f, uint8_t *d, size_t n, size_t max, unsigned seed,
                    int timeout_ms, size_t *ret) {
    struct mut_args a = {f, d, n, max, seed, 0};
    int rc = run_guarded(mut_thunk, &a, timeout_ms);
    *ret = rc ? 0 : a.ret;
    return rc;
}

struct cross_args { fz_cross_fn f; const uint8_t *a; size_t na; const uint8_t *b;
                    size_t nb; uint8_t *out; size_t max; unsigned seed; size_t ret; };
static void cross_thunk(void *p) {
    struct cross_args *a = p;
    a->ret = a->f(a->a, a->na, a->b, a->nb, a->out, a->max, a->seed);
}
int fz_guard_cross(fz_cross_fn f, const uint8_t *a, size_t na, const uint8_t *b, size_t nb,
                   uint8_t *out, size_t max, unsigned seed, int timeout_ms, size_t *ret) {
    struct cross_args x = {f, a, na, b, nb, out, max, seed, 0};
    int rc = run_guarded(cross_thunk, &x, timeout_ms);
    *ret = rc ? 0 : x.ret;
    return rc;
}
"""

# ---------------------------------------------------------------------------
# Byte-level havoc behind LLVMFuzzerMutate
# ---------------------------------------------------------------------------

_INTERESTING_8 = (0, 1, 16, 32, 64, 100, 127, 128, 255)
_INTERESTING_16 = (0, 128, 255, 256, 512, 1000, 1024, 4096, 32767, 32768, 65535)
_INTERESTING_32 = (0, 32768, 65535, 65536, 100663045, 2147483647, 2147483648, 4294967295)
_MAX_BLOCK = 64
_N_OPS = 10


def _write_int(buf: bytearray, pos: int, value: int, width: int, rng) -> None:
    order = "little" if rng.randint(0, 1) else "big"
    buf[pos : pos + width] = (value & ((1 << (8 * width)) - 1)).to_bytes(width, order)


def _havoc_once(buf: bytearray, max_size: int, rng) -> None:
    n = len(buf)
    op = rng.randrange(_N_OPS)
    if n == 0:
        if max_size > 0:
            buf.extend(rng.randbytes(rng.randint(1, min(max_size, 8))))
        return
    if op == 0:  # bit flip
        buf[rng.randrange(n)] ^= 1 << rng.randrange(8)
    elif op == 1:  # random byte
        buf[rng.randrange(n)] = rng.randrange(256)
    elif op == 2:  # interesting byte
        buf[rng.randrange(n)] = rng.choice(_INTERESTING_8)
    elif op == 3:  # byte arithmetic
        pos = rng.randrange(n)
        delta = rng.randint(1, 35)
        buf[pos] = (buf[pos] + (delta if rng.randint(0, 1) else -delta)) & 0xFF
    elif op == 4:  # interesting 16/32-bit value
        width = 2 if n < 4 or rng.randint(0, 1) else 4
        if n >= width:
            vals = _INTERESTING_16 if width == 2 else _INTERESTING_32
            _write_int(buf, rng.randrange(n - width + 1), rng.choice(vals), width, rng)
    elif op == 5:  # delete a block
        if n >= 2:
            ln = rng.randint(1, min(_MAX_BLOCK, n - 1))
            start = rng.randrange(n - ln + 1)
            del buf[start : start + ln]
    elif op == 6:  # duplicate a block
        ln = rng.randint(1, min(_MAX_BLOCK, n, max_size - n)) if max_size > n else 0
        if ln:
            start = rng.randrange(n - ln + 1)
            at = rng.randrange(n + 1)
            buf[at:at] = buf[start : start + ln]
    elif op == 7:  # insert random bytes
        if max_size > n:
            at = rng.randrange(n + 1)
            buf[at:at] = rng.randbytes(rng.randint(1, min(_MAX_BLOCK, max_size - n)))
    elif op == 8:  # overwrite a block with another block of the buffer
        if n >= 2:
            ln = rng.randint(1, min(_MAX_BLOCK, n - 1))
            src = rng.randrange(n - ln + 1)
            dst = rng.randrange(n - ln + 1)
            buf[dst : dst + ln] = bytes(buf[src : src + ln])
    else:  # swap two bytes
        i, j = rng.randrange(n), rng.randrange(n)
        buf[i], buf[j] = buf[j], buf[i]


def havoc(data: bytes, max_size: int, rng=None) -> bytes:
    """One libFuzzer-``Mutate``-style byte mutation of *data*, at most *max_size* long.

    Retries a few times so a no-op draw (e.g. a bit flip undone by nothing,
    a swap of a byte with itself) does not come back unchanged; when the
    input is already longer than *max_size* it is truncated first, as
    ``LLVMFuzzerMutate`` requires ``Size <= MaxSize``.
    """
    rng = rng if rng is not None else random.Random()
    buf = bytearray(data[:max_size] if max_size >= 0 else data)
    original = bytes(buf)
    for _ in range(8):
        _havoc_once(buf, max_size, rng)
        if len(buf) > max_size:
            del buf[max_size:]
        if bytes(buf) != original:
            break
    return bytes(buf)


# ---------------------------------------------------------------------------
# Host shim: exports LLVMFuzzerMutate and forwards it into Python
# ---------------------------------------------------------------------------

def _signal_name(num: int) -> str:
    try:
        return signal.Signals(num).name
    except ValueError:
        return f"signal {num}"


_MUTATE_CB = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)

#: rng for the plugin call in flight; read by the LLVMFuzzerMutate callback.
_current_rng: Any = None
_fallback_rng = random.Random()


def _on_mutate(ptr, size, max_size) -> int:
    """Body of ``LLVMFuzzerMutate``: mutate ``Data[:Size]`` in place, return the new size."""
    if not ptr:
        return 0
    try:
        out = havoc(ctypes.string_at(ptr, size), max_size, _current_rng or _fallback_rng)
        ctypes.memmove(ptr, out, len(out))
        return len(out)
    except Exception:  # noqa: BLE001 - must not unwind into foreign C
        log.warning("LLVMFuzzerMutate callback failed", exc_info=True)
        return size


_host: ctypes.CDLL | None = None
_host_cb: Any = None  # keeps the CFUNCTYPE object alive for the process
_host_dir: str | None = None


def _cleanup_host_dir() -> None:
    if _host_dir:
        shutil.rmtree(_host_dir, ignore_errors=True)


def _load_host() -> ctypes.CDLL:
    """Build (once) and load the ``LLVMFuzzerMutate`` forwarder, ``RTLD_GLOBAL``."""
    global _host, _host_cb, _host_dir
    if _host is not None:
        return _host
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("--custom-mutator needs a C compiler (set $CC or install cc/gcc/clang)")
    tmp = tempfile.mkdtemp(prefix="fz_cm_host_")
    src = Path(tmp) / "host.c"
    so = Path(tmp) / "libfzhost.so"
    src.write_text(_HOST_SRC)
    proc = subprocess.run(
        [cc, "-shared", "-fPIC", "-O2", "-o", str(so), str(src), "-lrt"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"building the LLVMFuzzerMutate host shim failed: {proc.stderr.strip()}")
    _host_dir = tmp
    atexit.register(_cleanup_host_dir)
    host = ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
    host.fz_set_mutate_cb.argtypes = [ctypes.c_void_p]
    host.fz_set_mutate_cb.restype = None
    host.fz_guard_mutate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    host.fz_guard_mutate.restype = ctypes.c_int
    host.fz_guard_cross.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    host.fz_guard_cross.restype = ctypes.c_int
    _host_cb = _MUTATE_CB(_on_mutate)
    host.fz_set_mutate_cb(ctypes.cast(_host_cb, ctypes.c_void_p))
    _host = host
    return host


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class CustomMutatorPlugin:
    """A loaded plugin ``.so`` with whichever of the two entry points it exports."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        isolate: bool = True,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        self.path = str(path)
        #: Run plugin calls under the C-side fault/timeout guard.
        self.isolate = isolate
        #: Wall-clock budget per plugin call; 0 disables the timer.
        self.timeout_ms = max(0, int(timeout_ms))
        #: Plugin calls that crashed or hung, keyed by signal name / "timeout".
        self.failures: dict[str, int] = {}
        #: The input that provoked the most recent failure (for reproduction).
        self.last_failure_input: bytes | None = None
        self.calls = 0
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"--custom-mutator: no such file: {self.path}")
        self._host = _load_host()  # first: the plugin's undefined LLVMFuzzerMutate must resolve at dlopen
        try:
            self._lib = ctypes.CDLL(os.path.abspath(self.path))
        except OSError as exc:
            raise RuntimeError(f"--custom-mutator: cannot load {self.path}: {exc}") from exc

        self._mutate = self._bind(
            MUTATOR_SYMBOL,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint],
        )
        self._cross = self._bind(
            CROSSOVER_SYMBOL,
            [
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint,
            ],
        )
        if self._mutate is None and self._cross is None:
            raise RuntimeError(
                f"--custom-mutator: {self.path} exports neither {MUTATOR_SYMBOL} nor {CROSSOVER_SYMBOL}"
            )

    def _bind(self, symbol: str, argtypes: list) -> Any:
        try:
            fn = getattr(self._lib, symbol)
        except AttributeError:
            return None
        fn.argtypes = argtypes
        fn.restype = ctypes.c_size_t
        return fn

    @property
    def has_mutator(self) -> bool:
        return self._mutate is not None

    @property
    def has_crossover(self) -> bool:
        return self._cross is not None

    @staticmethod
    def _capacity(max_len: int, *lengths: int) -> int:
        return max_len or max(DEFAULT_CAP, 2 * max(lengths))

    def _record_failure(self, status: int, data: bytes) -> None:
        name = "timeout" if status == _FZ_TIMEOUT else _signal_name(status)
        self.failures[name] = self.failures.get(name, 0) + 1
        self.last_failure_input = bytes(data)
        total = sum(self.failures.values())
        # First few, then one in a hundred: a plugin that dies on every call
        # would otherwise drown the log.
        if total <= 5 or total % 100 == 0:
            log.warning(
                "custom mutator plugin %s: %s on a %d-byte input (failure #%d, head %s); call skipped",
                self.path,
                name,
                len(data),
                total,
                data[:16].hex(),
            )

    def mutate(self, data: bytes, max_len: int, rng) -> bytes | None:
        """Call ``LLVMFuzzerCustomMutator``.

        ``None`` for a size of 0 or above ``MaxSize`` and for a call the
        guard had to abandon (fault or timeout; see :attr:`failures`).
        """
        global _current_rng
        if self._mutate is None:
            return None
        cap = self._capacity(max_len, len(data))
        data = data[:cap]
        buf = (ctypes.c_uint8 * cap)()
        ctypes.memmove(buf, data, len(data))
        seed = rng.randint(0, 0xFFFFFFFF)
        self.calls += 1
        _current_rng = rng
        try:
            if self.isolate:
                ret = ctypes.c_size_t(0)
                status = self._host.fz_guard_mutate(
                    ctypes.cast(self._mutate, ctypes.c_void_p),
                    buf,
                    len(data),
                    cap,
                    seed,
                    self.timeout_ms,
                    ctypes.byref(ret),
                )
                if status:
                    self._record_failure(status, data)
                    return None
                n = ret.value
            else:
                n = self._mutate(buf, len(data), cap, seed)
        finally:
            _current_rng = None
        if n == 0 or n > cap:
            return None
        return ctypes.string_at(buf, n)

    def crossover(self, a: bytes, b: bytes, max_len: int, rng) -> bytes | None:
        """Call ``LLVMFuzzerCustomCrossOver``; same decline rules as :meth:`mutate`."""
        global _current_rng
        if self._cross is None:
            return None
        cap = self._capacity(max_len, len(a), len(b))
        out = (ctypes.c_uint8 * cap)()
        seed = rng.randint(0, 0xFFFFFFFF)
        self.calls += 1
        _current_rng = rng
        try:
            if self.isolate:
                ret = ctypes.c_size_t(0)
                status = self._host.fz_guard_cross(
                    ctypes.cast(self._cross, ctypes.c_void_p),
                    a,
                    len(a),
                    b,
                    len(b),
                    out,
                    cap,
                    seed,
                    self.timeout_ms,
                    ctypes.byref(ret),
                )
                if status:
                    self._record_failure(status, a)
                    return None
                n = ret.value
            else:
                n = self._cross(a, len(a), b, len(b), out, cap, seed)
        finally:
            _current_rng = None
        if n == 0 or n > cap:
            return None
        return ctypes.string_at(out, n)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class _PluginOp(MutatorBase):
    category = "adaptive"

    def __init__(self, plugin: CustomMutatorPlugin) -> None:
        self.plugin = plugin

    def rebind(self, plugin: CustomMutatorPlugin) -> None:
        self.plugin = plugin


class CustomMutatorOp(_PluginOp):
    """``custom_mutator``: the plugin's ``LLVMFuzzerCustomMutator``."""

    name = "custom_mutator"

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        return self.plugin.mutate(data, max_len, rng)


class CustomCrossoverOp(_PluginOp):
    """``custom_crossover``: the plugin's ``LLVMFuzzerCustomCrossOver`` against a corpus entry."""

    name = "custom_crossover"

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        return len(context.corpus) > 1

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        corpus = context.corpus if context is not None else ()
        if len(corpus) < 2:
            return None
        other = corpus[rng.randrange(len(corpus))]
        if not isinstance(other, bytes | bytearray) or other == data:
            return None
        return self.plugin.crossover(data, bytes(other), max_len, rng)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_OP_NAMES = (CustomMutatorOp.name, CustomCrossoverOp.name)


def register_custom_mutator(
    path: str | os.PathLike[str],
    *,
    isolate: bool = True,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> list[MutatorBase]:
    """Load *path* and register one operator per exported entry point.

    Safe to call again: an operator that already exists is rebound to the
    new plugin instead of raising a duplicate-registration error.  Raises
    ``FileNotFoundError`` / ``RuntimeError`` on a missing, unloadable or
    symbol-less plugin; the caller asked for it by name, so it fails loudly.
    """
    from fuzzer_tool.core import operator_categories
    from fuzzer_tool.core.operator_registry import REGISTRY

    plugin = CustomMutatorPlugin(path, isolate=isolate, timeout_ms=timeout_ms)
    existing = {getattr(m, "name", None): m for m in REGISTRY.mutators()}
    wanted: list[tuple[type[_PluginOp], bool]] = [
        (CustomMutatorOp, plugin.has_mutator),
        (CustomCrossoverOp, plugin.has_crossover),
    ]
    out: list[MutatorBase] = []
    for cls, present in wanted:
        if not present:
            continue
        mutator = existing.get(cls.name)
        if mutator is not None:
            mutator.rebind(plugin)
        else:
            mutator = cls(plugin)
            REGISTRY.register_mutator(mutator)
        out.append(mutator)
    operator_categories.refresh()
    log.info(
        "Custom mutator plugin %s: %s",
        plugin.path,
        ", ".join(m.name for m in out),
    )
    return out


def unregister_custom_mutators() -> None:
    """Remove the plugin operators from ``REGISTRY`` (test teardown)."""
    from fuzzer_tool.core import operator_categories
    from fuzzer_tool.core.operator_registry import REGISTRY

    removed = False
    for name in _OP_NAMES:
        if REGISTRY._ops.pop(name, None) is not None:  # noqa: SLF001
            removed = True
    if removed:
        REGISTRY._categories_cache = None  # noqa: SLF001
        REGISTRY._version += 1  # noqa: SLF001
        with contextlib.suppress(Exception):
            operator_categories.refresh()


__all__ = [
    "CROSSOVER_SYMBOL",
    "MUTATOR_SYMBOL",
    "CustomCrossoverOp",
    "CustomMutatorOp",
    "CustomMutatorPlugin",
    "havoc",
    "register_custom_mutator",
    "unregister_custom_mutators",
]
