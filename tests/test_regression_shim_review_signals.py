"""Regressions from the afl_shim.c signal / private-heap review.

- re-arming after a partial displacement saved the shim's own handler as
  the "previous owner": a stray signal re-raised into itself forever
- exit()/_exit()/errx() inside a guard leaked the call's private-heap slots
- a timeout tick jumped out of the private heap's critical section
- crash handlers did not block the guard tick: a tick in the handler
  turned a crash into a timeout
- a nested timed call left the outer frame on the inner budget's period
- big-block tombstones were never compacted
- the private-heap stamp and lock were process-global: one thread's crash
  freed another thread's live slots and dropped the lock it held

Fake ticks: a SI_TIMER rt_tgsigqueueinfo to self is what the guard timer
delivers, so tests drive the tick count exactly instead of racing a timer.
"""

from __future__ import annotations

import signal
import subprocess

import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

RUN_TIMEOUT_S = 30
NESTED_MAX_MS = 5000
EXIT_ROUNDS = 30
BIG_SLOTS = 1 << 12  # __AFL_PH_BIG_SLOTS
TOMB_FRACTION = 4  # compaction once tombs exceed BIG_SLOTS / TOMB_FRACTION
PH_FLAGS = ("-D__AFL_PRIVATE_HEAP=1", "-fno-builtin", "-pthread")

_SRC = r"""
#include <err.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

typedef int (*entry_t)(const uint8_t *, size_t);

/* What the guard timer delivers: SI_TIMER, to this thread. */
static void fake_ticks(int n) {
    siginfo_t si;
    memset(&si, 0, sizeof(si));
    si.si_signo = __AFL_TIMEOUT_SIG;
    si.si_code = SI_TIMER;
    for (int i = 0; i < n; i++)
        syscall(SYS_rt_tgsigqueueinfo, getpid(), (pid_t)syscall(SYS_gettid),
                __AFL_TIMEOUT_SIG, &si);
}

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int ok_entry(const uint8_t *d, size_t n) { (void)d; (void)n; return 0; }
static int segv_entry(const uint8_t *d, size_t n) { (void)d; (void)n; raise(SIGSEGV); return 0; }

/* ── handlers ── */
static volatile int host_hits;
static void host_handler(int s) { (void)s; host_hits++; }

static int displaced(int stray) {
    static const int host_sigs[] = {SIGSEGV, SIGABRT, SIGFPE, SIGBUS, SIGILL};  /* not SIGSYS */
    __afl_guarded_call(ok_entry, 0, 0);
    for (unsigned i = 0; i < sizeof(host_sigs) / sizeof(host_sigs[0]); i++)
        signal(host_sigs[i], host_handler);
    uint64_t h[32];
    __afl_shim_health(h, 32);              /* flags the displacement */
    __afl_guarded_call(ok_entry, 0, 0);    /* takes the signals back */
    raise(stray);                          /* outside any guard */
    printf("survived hits=%d\n", host_hits);
    return 0;
}

static int mask_blocks_tick(void) {
    __afl_guarded_call(ok_entry, 0, 0);
    int bad = 0;
    for (int i = 0; i < __afl_NUM_GUARD_SIGNALS; i++) {
        struct sigaction cur;
        sigaction(__afl_guard_signals[i], NULL, &cur);
        bad += !sigismember(&cur.sa_mask, __AFL_TIMEOUT_SIG);
    }
    printf("unmasked=%d\n", bad);
    return 0;
}

/* SIGSEGV and TICKS+1 ticks pending at once: the kernel delivers the
 * synchronous signal first, then the ticks on top of its handler. */
static int crash_with_ticks(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    sigset_t s, old;
    sigemptyset(&s);
    sigaddset(&s, SIGSEGV);
    sigaddset(&s, __AFL_TIMEOUT_SIG);
    pthread_sigmask(SIG_BLOCK, &s, &old);
    raise(SIGSEGV);
    fake_ticks(__AFL_TIMEOUT_TICKS + 1);
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    return 0;
}

/* ── nested timed call ── */
#define OUTER_US 50000ull
#define INNER_US 20000000ull
static volatile int spin_sink;
static int inner_mode;
static int outer_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    __afl_guarded_call_timeout(inner_mode ? segv_entry : ok_entry, 0, 0, INNER_US);
    for (;;) spin_sink++;
}

static int nested(int mode) {
    inner_mode = mode;
    double t0 = now_ms();
    int rc = __afl_guarded_call_timeout(outer_entry, 0, 0, OUTER_US);
    printf("rc=%d ms=%.0f\n", rc, now_ms() - t0);
    return 0;
}

#if __AFL_PRIVATE_HEAP
/* ── tick inside the heap lock ── */
static int spin_after;
static int locked_ticks(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    __afl_ph_lock();
    fake_ticks(__AFL_TIMEOUT_TICKS + 1);
    __afl_ph_unlock();
    if (spin_after) for (;;) spin_sink++;
    return 7;
}

static int lock_tick(int after, uint64_t budget) {
    spin_after = after;
    int rc = __afl_guarded_call_timeout(locked_ticks, 0, 0, budget);
    int rc2 = __afl_guarded_call(ok_entry, 0, 0);
    free(malloc(64));                      /* the lock is free again */
    printf("rc=%d rc2=%d\n", rc, rc2);
    return 0;
}

/* ── exit inside a guard ── */
#define EXIT_ALLOCS 100
static int exit_kind;
static char *kept;
static int keep_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    kept = malloc(64);
    memcpy(kept, "kept", 5);
    return 0;
}
static int exit_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    for (int i = 0; i < EXIT_ALLOCS; i++) malloc(64 + i);
    if (exit_kind == 0) exit(3);
    if (exit_kind == 1) _exit(3);
    errx(3, "bad input");
}

static int exits(int kind, int rounds) {
    exit_kind = kind;
    __afl_guarded_call(keep_entry, 0, 0);
    long before = __afl_ph_live();
    int rc = 0;
    for (int i = 0; i < rounds; i++) rc = __afl_guarded_call(exit_entry, 0, 0);
    long after = __afl_ph_live();
    int reused = 0;                        /* the earlier call's slot stays live */
    for (int i = 0; i < 2 * EXIT_ALLOCS; i++) reused += malloc(64) == kept;
    printf("rc=%d before=%ld after=%ld reused=%d kept=%s\n", rc, before, after, reused, kept);
    return 0;
}

/* ── big-block tombstones ── */
#define BIG_N 1800
#define BIG_MIN (__AFL_PH_MAX_SLOT + 1)
static void *bigs[BIG_N];
static int big_churn(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    for (int i = 0; i < BIG_N; i++) bigs[i] = malloc(BIG_MIN);
    for (int i = 0; i < BIG_N; i++)
        if (i % 3) { free(bigs[i]); bigs[i] = NULL; }
    return 0;
}

static int tombs(void) {
    int rc = __afl_guarded_call(big_churn, 0, 0);
    unsigned ntomb = 0, lost = 0;
    for (unsigned i = 0; i < __AFL_PH_BIG_SLOTS; i++) ntomb += __afl_ph.big[i].p == __AFL_PH_TOMB;
    for (int i = 0; i < BIG_N; i++)
        if (bigs[i]) lost += malloc_usable_size(bigs[i]) < BIG_MIN;
    for (int i = 0; i < BIG_N; i++) free(bigs[i]);
    printf("rc=%d tombs=%u lost=%u live=%ld\n", rc, ntomb, lost, __afl_ph_live());
    return 0;
}

/* ── two guarded threads ── */
static volatile int phase;
static void wait_phase(int p) { while (phase < p) sched_yield(); }

static int crasher(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    phase = 1;
    wait_phase(2);
    raise(SIGSEGV);
    return 0;
}
static void *crash_thread(void *a) {
    (void)a;
    int rc = __afl_guarded_call(crasher, 0, 0);
    phase = 3;
    return (void *)(intptr_t)rc;
}

static int stamp_mode;
static int owner(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    wait_phase(1);
    if (stamp_mode == 0) {
        char *x = malloc(64);
        phase = 2;
        wait_phase(3);
        char *y = malloc(64);
        printf("aliased=%d\n", x == y);
        free(y);
        free(x);
        return 0;
    }
    __afl_ph_lock();
    phase = 2;
    double t0 = now_ms();
    int cleared = 0;
    while (!cleared && now_ms() - t0 < 1000) cleared = __afl_ph.lock == 0;
    __afl_ph_unlock();
    printf("cleared=%d\n", cleared);
    return 0;
}

static int threads(int mode) {
    stamp_mode = mode;
    free(malloc(64));                      /* init the heap on this thread */
    pthread_t t;
    pthread_create(&t, NULL, crash_thread, NULL);
    int rc = __afl_guarded_call(owner, 0, 0);
    void *crc;
    pthread_join(t, &crc);
    printf("rc=%d crash=%d\n", rc, (int)(intptr_t)crc);
    return 0;
}
#endif

int main(int argc, char **argv) {
    struct rlimit no_core = {0, 0};
    setrlimit(RLIMIT_CORE, &no_core);
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *m = argc > 1 ? argv[1] : "";
    if (!strcmp(m, "stray_sys")) return displaced(SIGSYS);
    if (!strcmp(m, "stray_abrt")) return displaced(SIGABRT);
    if (!strcmp(m, "mask")) return mask_blocks_tick();
    if (!strcmp(m, "crash_ticks")) {
        printf("rc=%d\n", __afl_guarded_call_timeout(crash_with_ticks, 0, 0, 10000000));
        return 0;
    }
    if (!strcmp(m, "nested_ok")) return nested(0);
    if (!strcmp(m, "nested_crash")) return nested(1);
#if __AFL_PRIVATE_HEAP
    if (!strcmp(m, "lock_tick")) return lock_tick(0, 10000000);
    if (!strcmp(m, "lock_tick_spin")) return lock_tick(1, 200000);
    if (!strcmp(m, "exit")) return exits(atoi(argv[2]), atoi(argv[3]));
    if (!strcmp(m, "tombs")) return tombs();
    if (!strcmp(m, "stamp")) return threads(0);
    if (!strcmp(m, "lock_owner")) return threads(1);
#endif
    return 99;
}
"""


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("review_plain"), _SRC, "-pthread")


@pytest.fixture(scope="module")
def heap(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("review_heap"), _SRC, *PH_FLAGS)


def _run(exe, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [str(exe), *args],
            capture_output=True, text=True, timeout=RUN_TIMEOUT_S, env=_clean_env(),
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail(f"{args} hung")


def _fields(out: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in out.split() if "=" in kv)


# ── 1. partial displacement ────────────────────────────────────────────


@requires_clang
def test_regression_rearm_keeps_owner_of_undisplaced_signal(plain):
    """SIGSYS was never displaced: a stray one must reach its real owner (default: die)."""
    r = _run(plain, "stray_sys")
    assert r.returncode == -signal.SIGSYS, (r.returncode, r.stdout)


@requires_clang
def test_regression_rearm_saves_the_displacing_host_handler(plain):
    """Adversarial: a stray SIGABRT goes to the host handler that displaced the shim, once."""
    r = _run(plain, "stray_abrt")
    assert r.returncode == 0, r.stderr[-500:]
    assert _fields(r.stdout)["hits"] == "1"


# ── 4. crash handler masks the guard tick ──────────────────────────────


@requires_clang
def test_regression_crash_handlers_block_guard_tick(plain):
    r = _run(plain, "mask")
    assert r.returncode == 0 and _fields(r.stdout)["unmasked"] == "0", r.stdout


@requires_clang
def test_regression_tick_in_crash_handler_stays_a_crash(plain):
    """Adversarial: enough ticks to time out land on top of the SIGSEGV handler."""
    r = _run(plain, "crash_ticks")
    assert r.returncode == 0, r.stderr[-500:]
    assert int(_fields(r.stdout)["rc"]) == -signal.SIGSEGV


# ── 5. nested timed call restores the outer period ─────────────────────


@requires_clang
@pytest.mark.parametrize("mode", ["nested_ok", "nested_crash"])
def test_regression_nested_timed_call_restores_outer_budget(plain, mode):
    """Inner 20 s budget returns (or crashes); the outer 50 ms budget must still hold."""
    r = _run(plain, mode)
    f = _fields(r.stdout)
    assert r.returncode == 0, r.stderr[-500:]
    assert int(f["rc"]) == -1
    assert float(f["ms"]) < NESTED_MAX_MS


# ── 3. no timeout jump out of the heap lock ────────────────────────────


@requires_clang
def test_regression_timeout_deferred_inside_heap_lock(heap):
    r = _run(heap, "lock_tick")
    assert r.returncode == 0, r.stderr[-500:]
    assert _fields(r.stdout) == {"rc": "7", "rc2": "0"}


@requires_clang
def test_regression_deferred_timeout_fires_after_unlock(heap):
    """Adversarial: the deferred tick must still end a hang once the lock is dropped."""
    r = _run(heap, "lock_tick_spin")
    assert r.returncode == 0, r.stderr[-500:]
    assert _fields(r.stdout) == {"rc": "-1", "rc2": "0"}


# ── 2. exit inside a guard reclaims ────────────────────────────────────


@requires_clang
@pytest.mark.parametrize("kind", ["0", "1", "2"], ids=["exit", "_exit", "errx"])
def test_regression_exit_reclaims_call_allocations(heap, kind):
    r = _run(heap, "exit", kind, str(EXIT_ROUNDS))
    f = _fields(r.stdout)
    assert r.returncode == 0, r.stderr[-500:]
    assert f["rc"] == "3"
    assert f["after"] == f["before"]


@requires_clang
def test_regression_exit_reclaim_spares_earlier_calls(heap):
    """Adversarial: a slot from an earlier, normal call survives an exiting call."""
    r = _run(heap, "exit", "0", "1")
    f = _fields(r.stdout)
    assert (f["reused"], f["kept"]) == ("0", "kept")


# ── 6. big-block tombstones ────────────────────────────────────────────


@requires_clang
def test_regression_big_tombstones_are_compacted(heap):
    r = _run(heap, "tombs")
    f = _fields(r.stdout)
    assert r.returncode == 0, r.stderr[-500:]
    assert int(f["tombs"]) <= BIG_SLOTS // TOMB_FRACTION


@requires_clang
def test_regression_compaction_keeps_live_big_blocks(heap):
    """Adversarial: blocks kept between the freed ones stay findable after a rebuild."""
    r = _run(heap, "tombs")
    f = _fields(r.stdout)
    assert (f["rc"], f["lost"], f["live"]) == ("0", "0", "0")


# ── 7. per-thread stamp and lock owner ─────────────────────────────────


@requires_clang
def test_regression_crash_spares_other_threads_slots(heap):
    r = _run(heap, "stamp")
    f = _fields(r.stdout)
    assert r.returncode == 0, r.stderr[-500:]
    assert (f["aliased"], f["rc"], f["crash"]) == ("0", "0", str(-signal.SIGSEGV))


@requires_clang
def test_regression_crash_keeps_other_threads_lock(heap):
    """Adversarial: the crashing thread's reclaim must not drop a lock it does not hold."""
    r = _run(heap, "lock_owner")
    f = _fields(r.stdout)
    assert r.returncode == 0, r.stderr[-500:]
    assert (f["cleared"], f["rc"], f["crash"]) == ("0", "0", str(-signal.SIGSEGV))
