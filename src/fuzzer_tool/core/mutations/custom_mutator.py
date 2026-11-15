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

AFL++ ``afl_custom_*`` (``--afl-custom-mutator``)
-------------------------------------------------

:class:`AflCustomMutator` loads an AFL++ custom mutator: a ``.so`` exporting
``afl_custom_*`` or a ``.py`` module with AFL++'s Python hook names. Hooks
served: ``init`` (``afl`` is NULL; seed from the fuzzer rng), ``fuzz`` and
``havoc_mutation`` (operators ``afl_custom_fuzz`` / ``afl_custom_havoc``),
``post_process`` (bytes handed to the target; the corpus keeps the
original), the ``init_trim``/``trim``/``post_trim`` trio (replaces the
default trim walk), ``queue_get`` (skip a seed), ``queue_new_entry`` (on
admission), ``deinit`` (at exit). C calls share the guard above. Not
served: ``fuzz_count``, ``describe``, ``havoc_mutation_probability``,
``splice_optout``, ``fuzz_send``, ``post_run`` and plugins that read
``afl_state_t``.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import importlib.util
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase
from fuzzer_tool.core.rand_pool import get_default_rand_pool

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

/* afl_custom_* entry points: every argument is a pointer, size_t, u8 or
 * u32, all INTEGER class, so one 7-word call shape serves them all on the
 * SysV x86-64 and AAPCS64 ABIs; callers mask the return to its real width. */
typedef size_t (*fz_word7_fn)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                              uintptr_t, uintptr_t);
struct word7_args { fz_word7_fn f; const uintptr_t *a; size_t ret; };
static void word7_thunk(void *p) {
    struct word7_args *w = p;
    w->ret = w->f(w->a[0], w->a[1], w->a[2], w->a[3], w->a[4], w->a[5], w->a[6]);
}
int fz_guard_call7(fz_word7_fn f, const uintptr_t *args, int timeout_ms, size_t *ret) {
    struct word7_args w = {f, args, 0};
    int rc = run_guarded(word7_thunk, &w, timeout_ms);
    *ret = rc ? 0 : w.ret;
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
    rng = rng if rng is not None else get_default_rand_pool()
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


def _status_name(status: int) -> str:
    return "timeout" if status == _FZ_TIMEOUT else _signal_name(status)


def _count_failure(owner: Any, name: str, data: bytes) -> None:
    """Count a skipped plugin call on *owner* (``failures``, ``last_failure_input``)."""
    owner.failures[name] = owner.failures.get(name, 0) + 1
    owner.last_failure_input = bytes(data)
    total = sum(owner.failures.values())
    # First few, then one in a hundred: a plugin that dies on every call
    # would otherwise drown the log.
    if total <= 5 or total % 100 == 0:
        log.warning(
            "custom mutator plugin %s: %s on a %d-byte input (failure #%d, head %s); call skipped",
            owner.path,
            name,
            len(data),
            total,
            data[:16].hex(),
        )


_MUTATE_CB = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)

#: rng for the plugin call in flight; read by the LLVMFuzzerMutate callback.
_current_rng: Any = None


def _on_mutate(ptr, size, max_size) -> int:
    """Body of ``LLVMFuzzerMutate``: mutate ``Data[:Size]`` in place, return the new size."""
    if not ptr:
        return 0
    try:
        out = havoc(ctypes.string_at(ptr, size), max_size, _current_rng or get_default_rand_pool())
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
    host.fz_guard_call7.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    host.fz_guard_call7.restype = ctypes.c_int
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
        self._host = (
            _load_host()
        )  # first: the plugin's undefined LLVMFuzzerMutate must resolve at dlopen
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
        _count_failure(self, _status_name(status), data)

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
# AFL++ afl_custom_* plugins
# ---------------------------------------------------------------------------
#
#   AflCustomMutator (facade: bytes in/out, failure counting, queue files)
#        |-- _AflCBackend   .so, afl_custom_<hook> symbols, fz_guard_call7
#        `-- _AflPyBackend  .py, <hook> functions (AFL_PYTHON_MODULE style)

AFL_PREFIX = "afl_custom_"


class AflHook(Enum):
    """``afl_custom_*`` entry points this host calls; ``init`` is mandatory."""

    INIT = "init"
    FUZZ = "fuzz"
    POST_PROCESS = "post_process"
    HAVOC = "havoc_mutation"
    INIT_TRIM = "init_trim"
    TRIM = "trim"
    POST_TRIM = "post_trim"
    QUEUE_GET = "queue_get"
    QUEUE_NEW_ENTRY = "queue_new_entry"
    DEINIT = "deinit"


_TRIM_HOOKS = (AflHook.INIT_TRIM, AflHook.TRIM, AflHook.POST_TRIM)
_LIFECYCLE_HOOKS = (AflHook.INIT, AflHook.DEINIT)
_WORDS = 7  # widest afl_custom_* signature (afl_custom_fuzz)
_U8 = 0xFF
_U32 = 0xFFFFFFFF


def _s32(word: int) -> int:
    """Low 32 bits of a returned word as a signed int (init_trim / post_trim)."""
    word &= _U32
    return word - (1 << 32) if word >> 31 else word


class _PluginFault(Exception):
    """A guarded C call that faulted or timed out; ``args[0]`` names it."""


class _AflCBackend:
    """``afl_custom_<hook>`` symbols of a shared object, called through the guard."""

    def __init__(self, path: str, isolate: bool, timeout_ms: int) -> None:
        self._host = _load_host()
        try:
            self._lib = ctypes.CDLL(os.path.abspath(path))
        except OSError as exc:
            raise RuntimeError(f"--afl-custom-mutator: cannot load {path}: {exc}") from exc
        self._isolate = isolate
        self._timeout_ms = timeout_ms
        self._fns: dict[AflHook, Any] = {}
        for hook in AflHook:
            fn = getattr(self._lib, AFL_PREFIX + hook.value, None)
            if fn is None:
                continue
            fn.restype = ctypes.c_size_t
            self._fns[hook] = fn
        self._data = 0

    def has(self, hook: AflHook) -> bool:
        return hook in self._fns

    def _call(self, hook: AflHook, *words: int) -> int:
        fn = self._fns[hook]
        if not self._isolate:
            return fn(*(ctypes.c_size_t(w) for w in words))

        args = (ctypes.c_size_t * _WORDS)(*words)
        ret = ctypes.c_size_t(0)
        status = self._host.fz_guard_call7(
            ctypes.cast(fn, ctypes.c_void_p), args, self._timeout_ms, ctypes.byref(ret)
        )
        if status:
            raise _PluginFault(_status_name(status))
        return ret.value

    @staticmethod
    def _out(out: ctypes.c_void_p, n: int, cap: int) -> bytes | None:
        # 0 = decline; a NULL out or an oversized length is a plugin bug, also declined.
        if n == 0 or n > cap or not out.value:
            return None
        return ctypes.string_at(out.value, n)

    @staticmethod
    def _buf(data: bytes) -> Any:
        buf = (ctypes.c_uint8 * max(len(data), 1))()
        ctypes.memmove(buf, data, len(data))
        return buf

    def init(self, seed: int) -> None:
        self._data = self._call(AflHook.INIT, 0, seed & _U32)
        if not self._data:
            raise RuntimeError("--afl-custom-mutator: afl_custom_init returned NULL")

    def fuzz(self, buf: bytes, add: bytes, max_size: int) -> bytes | None:
        b, a, out = self._buf(buf), self._buf(add), ctypes.c_void_p()
        n = self._call(
            AflHook.FUZZ,
            self._data,
            ctypes.addressof(b),
            len(buf),
            ctypes.addressof(out),
            ctypes.addressof(a),
            len(add),
            max_size,
        )
        return self._out(out, n, max_size)

    def post_process(self, buf: bytes) -> bytes | None:
        b, out = self._buf(buf), ctypes.c_void_p()
        n = self._call(
            AflHook.POST_PROCESS, self._data, ctypes.addressof(b), len(buf), ctypes.addressof(out)
        )
        return self._out(out, n, n)

    def havoc_mutation(self, buf: bytes, max_size: int) -> bytes | None:
        b, out = self._buf(buf), ctypes.c_void_p()
        n = self._call(
            AflHook.HAVOC,
            self._data,
            ctypes.addressof(b),
            len(buf),
            ctypes.addressof(out),
            max_size,
        )
        return self._out(out, n, max_size)

    def init_trim(self, buf: bytes) -> int:
        b = self._buf(buf)
        return _s32(self._call(AflHook.INIT_TRIM, self._data, ctypes.addressof(b), len(buf)))

    def trim(self) -> bytes | None:
        out = ctypes.c_void_p()
        n = self._call(AflHook.TRIM, self._data, ctypes.addressof(out))
        return ctypes.string_at(out.value, n) if out.value else None

    def post_trim(self, success: bool) -> int:
        return _s32(self._call(AflHook.POST_TRIM, self._data, int(success)))

    def queue_get(self, path: str) -> bool:
        name = ctypes.create_string_buffer(os.fsencode(path))
        return bool(self._call(AflHook.QUEUE_GET, self._data, ctypes.addressof(name)) & _U8)

    def queue_new_entry(self, path: str, orig: str | None) -> None:
        new = ctypes.create_string_buffer(os.fsencode(path))
        old = ctypes.create_string_buffer(os.fsencode(orig)) if orig else None
        self._call(
            AflHook.QUEUE_NEW_ENTRY,
            self._data,
            ctypes.addressof(new),
            ctypes.addressof(old) if old is not None else 0,
        )

    def deinit(self) -> None:
        self._call(AflHook.DEINIT, self._data)


class _AflPyBackend:
    """A Python module with AFL++'s Python hook names (``init``, ``fuzz``, ...)."""

    def __init__(self, path: str) -> None:
        name = f"fz_afl_custom_{Path(path).stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"--afl-custom-mutator: cannot import {path}")
        self._mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._mod)
        self._fns = {
            hook: getattr(self._mod, hook.value)
            for hook in AflHook
            if callable(getattr(self._mod, hook.value, None))
        }

    def has(self, hook: AflHook) -> bool:
        return hook in self._fns

    @staticmethod
    def _out(result: Any, cap: int) -> bytes | None:
        if not result or len(result) > cap:
            return None
        return bytes(result)

    def init(self, seed: int) -> None:
        self._fns[AflHook.INIT](seed & _U32)

    def fuzz(self, buf: bytes, add: bytes, max_size: int) -> bytes | None:
        res = self._fns[AflHook.FUZZ](bytearray(buf), bytearray(add), max_size)
        return self._out(res, max_size)

    def post_process(self, buf: bytes) -> bytes | None:
        res = self._fns[AflHook.POST_PROCESS](bytearray(buf))
        return bytes(res) if res else None

    def havoc_mutation(self, buf: bytes, max_size: int) -> bytes | None:
        return self._out(self._fns[AflHook.HAVOC](bytearray(buf), max_size), max_size)

    def init_trim(self, buf: bytes) -> int:
        return int(self._fns[AflHook.INIT_TRIM](bytearray(buf)))

    def trim(self) -> bytes | None:
        res = self._fns[AflHook.TRIM]()
        return bytes(res) if res is not None else None

    def post_trim(self, success: bool) -> int:
        return int(self._fns[AflHook.POST_TRIM](success))

    def queue_get(self, path: str) -> bool:
        return bool(self._fns[AflHook.QUEUE_GET](path))

    def queue_new_entry(self, path: str, orig: str | None) -> None:
        self._fns[AflHook.QUEUE_NEW_ENTRY](path, orig)

    def deinit(self) -> None:
        self._fns[AflHook.DEINIT]()


class AflCustomMutator:
    """An AFL++ custom mutator: ``.so`` exporting ``afl_custom_*`` or a ``.py`` module.

    Bytes in, bytes out. A faulting / timed-out C call or a raising Python
    hook is skipped and counted in :attr:`failures`, then answered with the
    hook's neutral value (``None``, ``-1`` for trim progress, ``True`` for
    ``queue_get``). Queue hooks get transient files: the corpus stores
    deltas, which a plugin cannot parse, so each call writes the raw bytes
    to a per-plugin temp dir and the path is only valid during the call.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        seed: int,
        isolate: bool = True,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        self.path = str(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"--afl-custom-mutator: no such file: {self.path}")

        if self.path.endswith(".py"):
            self._backend: Any = _AflPyBackend(self.path)
            self._faults: tuple[type[BaseException], ...] = (Exception,)
        else:
            self._backend = _AflCBackend(self.path, isolate, max(0, int(timeout_ms)))
            self._faults = (_PluginFault,)

        if not self._backend.has(AflHook.INIT):
            raise RuntimeError(f"--afl-custom-mutator: {self.path} has no {AFL_PREFIX}init")
        if not any(self._backend.has(h) for h in AflHook if h not in _LIFECYCLE_HOOKS):
            raise RuntimeError(f"--afl-custom-mutator: {self.path} has no {AFL_PREFIX}* hooks")

        self.failures: dict[str, int] = {}
        self.last_failure_input: bytes | None = None
        self.calls = 0
        self._backend.init(seed)
        self._qdir = tempfile.mkdtemp(prefix="fz_afl_q_")
        self._live = True
        atexit.register(self.deinit)

    def has(self, hook: AflHook) -> bool:
        """True when the plugin serves *hook*; trimming needs all three trim hooks."""
        if hook in _TRIM_HOOKS:
            return all(self._backend.has(h) for h in _TRIM_HOOKS)
        return self._backend.has(hook)

    def _guard(self, fallback: Any, data: bytes, fn: Any, *args: Any) -> Any:
        # One place that turns a plugin failure into a counted, skipped call.
        self.calls += 1
        try:
            return fn(*args)
        except self._faults as exc:
            name = exc.args[0] if isinstance(exc, _PluginFault) else type(exc).__name__
            _count_failure(self, name, data)
            return fallback

    def _qfile(self, name: str, data: bytes) -> str:
        path = os.path.join(self._qdir, name)
        Path(path).write_bytes(data)
        return path

    def fuzz(self, buf: bytes, add_buf: bytes, max_size: int) -> bytes | None:
        return self._guard(None, buf, self._backend.fuzz, buf, add_buf, max_size)

    def post_process(self, buf: bytes) -> bytes | None:
        """The bytes the target should see; ``None`` keeps *buf* as is."""
        return self._guard(None, buf, self._backend.post_process, buf)

    def havoc_mutation(self, buf: bytes, max_size: int) -> bytes | None:
        return self._guard(None, buf, self._backend.havoc_mutation, buf, max_size)

    def init_trim(self, buf: bytes) -> int:
        """Trim steps for *buf*; 0 = nothing to trim, < 0 = error."""
        return self._guard(-1, buf, self._backend.init_trim, buf)

    def trim(self) -> bytes | None:
        return self._guard(None, b"", self._backend.trim)

    def post_trim(self, success: bool) -> int:
        """Next trim step after a kept (*success*) or rejected cut; < 0 = error."""
        return self._guard(-1, b"", self._backend.post_trim, success)

    def queue_get(self, data: bytes) -> bool:
        """False when the plugin asks to skip fuzzing *data*."""
        return self._guard(True, data, self._backend.queue_get, self._qfile("cur", data))

    def queue_new_entry(self, data: bytes, parent: bytes | None) -> None:
        orig = self._qfile("orig", parent) if parent is not None else None
        self._guard(None, data, self._backend.queue_new_entry, self._qfile("new", data), orig)

    def deinit(self) -> None:
        """Call ``afl_custom_deinit`` once and drop the queue temp dir; idempotent."""
        if not self._live:
            return
        self._live = False
        if self._backend.has(AflHook.DEINIT):
            self._guard(None, b"", self._backend.deinit)
        shutil.rmtree(self._qdir, ignore_errors=True)


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


class AflFuzzOp(_PluginOp):
    """``afl_custom_fuzz``: the plugin's ``afl_custom_fuzz``, spliced with a corpus entry."""

    name = "afl_custom_fuzz"

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        # add_buf: another corpus entry, as AFL++ passes a random queue entry.
        corpus = context.corpus if context is not None else ()
        add = b""
        if len(corpus) > 1:
            other = corpus[rng.randrange(len(corpus))]
            if isinstance(other, bytes | bytearray) and other != data:
                add = bytes(other)
        cap = CustomMutatorPlugin._capacity(max_len, len(data), len(add))
        return self.plugin.fuzz(data, add, cap)


class AflHavocOp(_PluginOp):
    """``afl_custom_havoc``: the plugin's ``afl_custom_havoc_mutation``."""

    name = "afl_custom_havoc"

    def mutate(
        self,
        data: bytes,
        rng: Any,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx: Any,
    ) -> bytes | None:
        return self.plugin.havoc_mutation(data, CustomMutatorPlugin._capacity(max_len, len(data)))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_OP_NAMES = (CustomMutatorOp.name, CustomCrossoverOp.name, AflFuzzOp.name, AflHavocOp.name)


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
    plugin = CustomMutatorPlugin(path, isolate=isolate, timeout_ms=timeout_ms)
    wanted: list[tuple[type[_PluginOp], bool]] = [
        (CustomMutatorOp, plugin.has_mutator),
        (CustomCrossoverOp, plugin.has_crossover),
    ]
    out = _register_ops(wanted, plugin)
    log.info(
        "Custom mutator plugin %s: %s",
        plugin.path,
        ", ".join(m.name for m in out),
    )
    return out


def _register_ops(wanted: list[tuple[type[_PluginOp], bool]], plugin: Any) -> list[MutatorBase]:
    # Register (or rebind) one operator per present entry point.
    from fuzzer_tool.core import operator_categories
    from fuzzer_tool.core.operator_registry import REGISTRY

    existing = {getattr(m, "name", None): m for m in REGISTRY.mutators()}
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
    return out


def register_afl_custom_mutator(
    path: str | os.PathLike[str],
    *,
    seed: int,
    isolate: bool = True,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> tuple[AflCustomMutator, list[MutatorBase]]:
    """Load an AFL++ custom mutator; register ``afl_custom_fuzz`` / ``afl_custom_havoc``.

    The non-operator hooks (post_process, trim, queue_*) are the caller's to
    wire: the plugin is returned with the operators.
    """
    plugin = AflCustomMutator(path, seed=seed, isolate=isolate, timeout_ms=timeout_ms)
    wanted: list[tuple[type[_PluginOp], bool]] = [
        (AflFuzzOp, plugin.has(AflHook.FUZZ)),
        (AflHavocOp, plugin.has(AflHook.HAVOC)),
    ]
    ops = _register_ops(wanted, plugin)
    hooks = [h.value for h in AflHook if h not in _LIFECYCLE_HOOKS and plugin.has(h)]
    log.info("AFL++ custom mutator %s: %s", plugin.path, ", ".join(hooks))
    return plugin, ops


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
    "AflCustomMutator",
    "AflFuzzOp",
    "AflHavocOp",
    "AflHook",
    "CROSSOVER_SYMBOL",
    "MUTATOR_SYMBOL",
    "CustomCrossoverOp",
    "CustomMutatorOp",
    "CustomMutatorPlugin",
    "havoc",
    "register_afl_custom_mutator",
    "register_custom_mutator",
    "unregister_custom_mutators",
]
