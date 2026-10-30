"""Regressions: shim crash-guard blind spots (TODO afl_shim audit, item g).

- exit() on a worker thread the entry spawned ended the host: the worker
  has no guard frame of its own
- a target that blocked the crash signal before faulting killed the host:
  the kernel forces a blocked hardware-fault signal to SIG_DFL
- a target that installed its own crash handler mid-call returned from it
  into the same fault forever (SIG_IGN: the kernel killed the host);
  repaired only at the next health read
"""

from __future__ import annotations

import signal

import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build
from tests.test_regression_shim_second_pass import _run_driver

_SRC = """
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/syscall.h>

static volatile int sink;
static volatile int strays;

static void segv(void) { volatile int *p = (int *)(uintptr_t)(sink - sink); sink = *p; }

static void *exit_worker(void *a) { exit((int)(intptr_t)a); return 0; }
static void *uexit_worker(void *a) { _exit((int)(intptr_t)a); return 0; }

static int spawn(void *(*fn)(void *), size_t n) {
    pthread_t t;
    pthread_create(&t, 0, fn, (void *)(intptr_t)n);
    pthread_join(t, 0);
    return 99;   /* never reached when the worker's exit is honoured */
}
int worker_exit_entry(const uint8_t *d, size_t n) { (void)d; return spawn(exit_worker, n); }
int worker_uexit_entry(const uint8_t *d, size_t n) { (void)d; return spawn(uexit_worker, n); }

int blocked_segv_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    sigset_t s;
    sigemptyset(&s);
    sigaddset(&s, SIGSEGV);
    sigprocmask(SIG_BLOCK, &s, 0);
    segv();
    return 0;
}

int blocked_all_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    sigset_t s;
    sigfillset(&s);
    pthread_sigmask(SIG_SETMASK, &s, 0);
    segv();
    return 0;
}

/* Blocks USR1 and SEGV; reports what the mask kept: 10*usr1 + segv. */
int mask_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    sigset_t s, cur;
    sigemptyset(&s);
    sigaddset(&s, SIGUSR1);
    sigaddset(&s, SIGSEGV);
    sigprocmask(SIG_BLOCK, &s, 0);
    sigprocmask(SIG_BLOCK, 0, &cur);
    int r = 10 * sigismember(&cur, SIGUSR1) + sigismember(&cur, SIGSEGV);
    sigprocmask(SIG_UNBLOCK, &s, 0);
    return r;
}

static void returning_handler(int sig) { (void)sig; }
static void counting_handler(int sig) { (void)sig; strays++; }

int handler_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    struct sigaction sa = {0};
    sa.sa_handler = returning_handler;
    sigaction(SIGSEGV, &sa, 0);
    segv();
    return 0;
}

int ignore_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    signal(SIGSEGV, SIG_IGN);
    segv();
    return 0;
}

/* Kernel's view of a disposition, past any interposer. */
struct ksa { void *handler; unsigned long flags; void *restorer; unsigned long mask; };
static int kernel_handler_is(int sig, void (*h)(int)) {
    struct ksa k;
    syscall(SYS_rt_sigaction, sig, 0, &k, sizeof(k.mask));
    return k.handler == (void *)h;
}

int install_counting(void) {
    struct sigaction sa = {0};
    sa.sa_handler = counting_handler;
    return sigaction(SIGBUS, &sa, 0);
}

/* In a guard: 10 * (target reads back its handler) + (kernel runs it). */
int install_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    install_counting();
    struct sigaction cur;
    sigaction(SIGBUS, 0, &cur);
    return 10 * (cur.sa_handler == counting_handler) + kernel_handler_is(SIGBUS, counting_handler);
}

int kernel_runs_counting(void) { return kernel_handler_is(SIGBUS, counting_handler); }
int raise_bus(void) { raise(SIGBUS); return strays; }

int block_segv(void) {
    sigset_t s;
    sigemptyset(&s);
    sigaddset(&s, SIGSEGV);
    return sigprocmask(SIG_BLOCK, &s, 0);
}
int kernel_blocks_segv(void) {
    unsigned long m = 0;
    syscall(SYS_rt_sigprocmask, SIG_BLOCK, 0, &m, sizeof(m));
    return (int)((m >> (SIGSEGV - 1)) & 1);
}

int ok_entry(const uint8_t *d, size_t n) { (void)d; return (int)n; }
"""

SIGSEGV_RC = -int(signal.SIGSEGV)
EXIT_STATUS = 5


@pytest.fixture
def so(tmp_path):
    return _build(tmp_path, _SRC, "-fsanitize-coverage=trace-pc-guard", "-pthread", shared=True)


def _calls(so, body: str) -> list[str]:
    r = _run_driver(so, body)
    assert r is not None, "host hung"
    assert r.returncode == 0, (r.returncode, r.stderr[-500:])
    return r.stdout.split()


# ── exit() on a worker thread ───────────────────────────────────────────


@requires_clang
@pytest.mark.parametrize("entry", ["worker_exit_entry", "worker_uexit_entry"])
def test_regression_worker_exit_returns_through_guard(so, entry):
    out = _calls(so, f'print(call("{entry}", b"x" * {EXIT_STATUS}), call("ok_entry", b"ab"))')

    assert out == [str(EXIT_STATUS), "2"]


@requires_clang
def test_worker_exit_status_keeps_low_byte(so):
    """Adversarial: exit(261) is status 5, as waitpid would report it."""
    out = _calls(so, f'print(call("worker_exit_entry", b"x" * {256 + EXIT_STATUS}))')

    assert out == [str(EXIT_STATUS)]


@requires_clang
def test_worker_exit_outside_guard_exits(so):
    """Falsification: no guard running, exit() is the real one."""
    r = _run_driver(so, f'lib.worker_exit_entry(None, {EXIT_STATUS}); print("survived")')

    assert r is not None
    assert (r.returncode, r.stdout) == (EXIT_STATUS, "")


# ── Blocked crash signal ────────────────────────────────────────────────


@requires_clang
@pytest.mark.parametrize("entry", ["blocked_segv_entry", "blocked_all_entry"])
def test_regression_blocked_crash_signal_caught(so, entry):
    out = _calls(so, f'print(call("{entry}"), call("ok_entry", b"ab"))')

    assert out == [str(SIGSEGV_RC), "2"]


@requires_clang
def test_guard_keeps_other_blocked_signals(so):
    """Falsification: only guard signals stay deliverable; USR1 is blocked."""
    out = _calls(so, 'print(call("mask_entry"))')

    assert out == ["10"]


@requires_clang
def test_block_outside_guard_is_real(so):
    """Falsification: unguarded, the target's mask reaches the kernel."""
    out = _calls(so, "lib.block_segv(); print(lib.kernel_blocks_segv())")

    assert out == ["1"]


# ── Crash handler installed mid-call ────────────────────────────────────


@requires_clang
@pytest.mark.parametrize("entry", ["handler_entry", "ignore_entry"])
def test_regression_midcall_handler_does_not_hide_crash(so, entry):
    out = _calls(so, f'print(call("{entry}"), call("ok_entry", b"ab"))')

    assert out == [str(SIGSEGV_RC), "2"]


@requires_clang
def test_midcall_handler_is_transparent_to_target(so):
    """Target reads back its own handler; the kernel keeps the guard's."""
    out = _calls(so, 'print(call("install_entry"))')

    assert out == ["10"]


@requires_clang
def test_midcall_handler_still_gets_strays(so):
    """Adversarial: a signal outside any guard reaches the target's handler."""
    out = _calls(so, 'call("install_entry"); print(lib.raise_bus())')

    assert out == ["1"]


@requires_clang
def test_install_outside_guard_is_real(so):
    """Falsification: unguarded, sigaction() reaches the kernel."""
    out = _calls(so, "lib.install_counting(); print(lib.kernel_runs_counting())")

    assert out == ["1"]
