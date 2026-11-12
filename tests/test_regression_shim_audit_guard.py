"""Regressions from the afl_shim.c audit (2026-11-12): guard, tick, private heap.

  - a crash/exit landing kept the 50 us libc-deferral tick: the next call with
    the same budget timed out after ~250 us
  - double free / realloc of a freed big (> 1 MiB) private block went to glibc
  - realloc's same-class keep path returned freed or interior class pointers
  - the 32-bit call stamp wrapped: a crash reclaimed a slot owned by a call
    2^32 calls older
  - a fork while another thread held the private heap lock left the child
    spinning in malloc forever

Each case runs a C host that dlopens the target and drives the guard, so the
shim's signal handlers never reach pytest. The target is compiled with the
shim -included, so it reaches the shim's statics (``__afl_ph``, the tick
state) as test seams.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

FLAGS = ("-fsanitize-coverage=trace-pc-guard", "-D__AFL_PRIVATE_HEAP=1")
HOST_TIMEOUT_S = 30
BUDGET_US = 20000  # tick 5 ms; a 2 ms call never reaches tick 5
SIGABRT_RC = -6
TIMEOUT_RC = -1
EXIT_STATUS = 3

_TARGET = r"""
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define BIG (4u << 20)          /* above the largest class */
#define SLEEP_NS 60000000L      /* 3x the 20 ms budget, all inside read() */
#define BUSY_NS 2000000L

void *volatile gp;              /* host-provided libc chunk */
void *volatile gstale;          /* last private big block freed */
void *volatile gkeep;           /* slot meant to outlive its call */
volatile uint64_t gkeep_stamp;

static int wr_fd;
static void *writer(void *a) {
    (void)a;
    struct timespec ts = {0, SLEEP_NS};
    nanosleep(&ts, 0);
    (void)!write(wr_fd, "x", 1);
    return 0;
}

/* Blocks in libc read() past the budget's tick count: libc deferral. */
static void park_in_libc(void) {
    int fd[2];
    char c;
    pthread_t th;
    (void)!pipe(fd);
    wr_fd = fd[1];
    pthread_create(&th, 0, writer, 0);
    (void)!read(fd[0], &c, 1);
    pthread_join(th, 0);
    close(fd[0]);
    close(fd[1]);
}

static long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int entry(const uint8_t *d, size_t n) {
    (void)n;
    switch (d[0]) {
    case 's': park_in_libc(); *(volatile int *)(uintptr_t)(n - n) = 1; break;
    case 'e': park_in_libc(); exit(3);
    case 'n': park_in_libc(); break;
    case 'b': { long t = now_ns(); while (now_ns() - t < BUSY_NS) {} break; }
    case 'B': { void *p = malloc(BIG); free(p); free(p); break; }
    case 'P': { void *p = malloc(BIG); free(p); p = realloc(p, 2 * BIG); break; }
    case 'F': { void *p = malloc(BIG); gstale = p; free(p); break; }
    case 'S': free(gstale); break;
    case 'L': free(gp); break;
    case 'k': { char *p = malloc(100); free(p); p = realloc(p, 120); break; }
    case 'i': { char *p = malloc(100); p = realloc(p + 16, 120); break; }
    case 'K': { char *p = malloc(100); char *q = realloc(p, 120); free(q);
                return q == p ? 0 : 1; }
    case 'A': gkeep = malloc(100); gkeep_stamp = __afl_ph_stamp; break;
    case 'C': *(volatile int *)(uintptr_t)(n - n) = 1; break;
    case 'D': gkeep = malloc(100); *(volatile int *)(uintptr_t)(n - n) = 1; break;
    case 'h': for (volatile int spin = 1; spin;) {} break;
    case 'X': free(gkeep); break;
    }
    return 0;
}

/* Seams for the host. */
int tick_fast(void) { return __afl_tick_fast; }
void calls_wrap(void) {
    /* Next stamp equals gkeep's modulo 2^32. */
    __afl_ph.calls = (__typeof__(__afl_ph.calls))(gkeep_stamp - 1 + (1ULL << 32));
}
void ph_lock(void) { __afl_ph_lock(); }
void ph_unlock(void) { __afl_ph_unlock(); }
int ph_locked(void) { return __afl_ph.lock != 0; }
void *ph_malloc(size_t n) { return malloc(n); }
"""

# Steps: g<c>[:us] guarded call, m<n> gp = libc malloc(n), a<n> gp = libc
# page-aligned memalign(n), f<n> fake libc mmapped chunk of n bytes at gstale,
# t tick probe, w stamp wrap.
_HOST = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#define PAGE 4096
typedef int (*entry_t)(const uint8_t *, size_t);
typedef int (*guard_t)(entry_t, const uint8_t *, size_t, uint64_t);
static void *idle(void *a) { (void)a; for (;;) sleep(100); return 0; }

/* glibc's layout of a page-aligned mmapped chunk: one page in front,
 * prev_size = offset to the mapping, size | IS_MMAPPED. */
static void *fake_chunk(void *at, size_t n) {
    char *b = mmap((char *)at - PAGE, PAGE + n, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (b == MAP_FAILED || b != (char *)at - PAGE) return 0;
    size_t *hdr = (size_t *)((char *)at - 16);
    hdr[0] = PAGE - 16;
    hdr[1] = (n + 16) | 2;
    return at;
}

int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    guard_t g = (guard_t)dlsym(h, "__afl_guarded_call_timeout");
    entry_t e = (entry_t)dlsym(h, "entry");
    void **gp = (void **)dlsym(h, "gp");
    void **gstale = (void **)dlsym(h, "gstale");
    int (*tick)(void) = (int (*)(void))dlsym(h, "tick_fast");
    void (*wrap)(void) = (void (*)(void))dlsym(h, "calls_wrap");
    pthread_t th;
    pthread_create(&th, 0, idle, 0);
    for (int i = 2; i < argc; i++) {
        const char *s = argv[i];
        switch (s[0]) {
        case 'g': {
            uint64_t us = s[2] == ':' ? strtoull(s + 3, 0, 10) : 0;
            printf("rc %d\n", g(e, (const uint8_t *)s + 1, 1, us));
            break;
        }
        case 'm': *gp = malloc(strtoul(s + 1, 0, 10)); break;
        case 'a': posix_memalign(gp, PAGE, strtoul(s + 1, 0, 10)); break;
        case 'f': *gp = fake_chunk(*gstale, strtoul(s + 1, 0, 10));
                  printf("fake %d\n", *gp != 0); break;
        case 't': printf("tick %d\n", tick()); break;
        case 'w': wrap(); break;
        }
        fflush(stdout);
    }
    void *big = malloc(300000);
    printf("alive %d\n", big != NULL);
    return 0;
}
"""

# Fork child mallocs while another thread holds the heap lock. SIGALRM ends a
# spinning child; the parent reports how the child ended.
_FORK_HOST = r"""
#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#define CHILD_ALARM_S 5
static void (*lock)(void), (*unlock)(void);
static int (*locked)(void);
static void *(*pmalloc)(size_t);
static volatile int held, release;
static void *holder(void *a) {
    (void)a;
    lock();
    held = 1;
    while (!release) usleep(1000);
    unlock();
    return 0;
}
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s\n", dlerror()); return 2; }
    lock = (void (*)(void))dlsym(h, "ph_lock");
    unlock = (void (*)(void))dlsym(h, "ph_unlock");
    locked = (int (*)(void))dlsym(h, "ph_locked");
    pmalloc = (void *(*)(size_t))dlsym(h, "ph_malloc");
    (void)pmalloc(32);                      /* heap ready before the fork */
    int hold = !strcmp(argv[2], "hold");
    pthread_t th;
    if (hold) {
        pthread_create(&th, 0, holder, 0);
        while (!held) usleep(1000);
    }
    pid_t pid = fork();
    if (pid == 0) {
        alarm(CHILD_ALARM_S);
        _exit(pmalloc(32) ? 0 : 1);
    }
    int st;
    waitpid(pid, &st, 0);
    printf("child %s %d\n", WIFEXITED(st) ? "exit" : "signal",
           WIFEXITED(st) ? WEXITSTATUS(st) : WTERMSIG(st));
    printf("parent_locked %d\n", locked());
    if (hold) { release = 1; pthread_join(th, 0); }
    return 0;
}
"""


def _cc(tmp: Path, name: str, src: str) -> Path:
    c = tmp / f"{name}.c"
    c.write_text(src)
    out = tmp / name
    r = subprocess.run(
        ["clang", "-O1", str(c), "-o", str(out), "-ldl", "-lpthread"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return out


@pytest.fixture(scope="module")
def rig(tmp_path_factory) -> tuple[Path, Path, Path]:
    tmp = tmp_path_factory.mktemp("shim_audit_guard")
    so = _build(tmp, _TARGET, *FLAGS, shared=True)
    return _cc(tmp, "host", _HOST), _cc(tmp, "fork_host", _FORK_HOST), so


def _go(rig, *steps: str) -> subprocess.CompletedProcess:
    host, _, so = rig
    # A hang is the bug: TimeoutExpired fails the test.
    return subprocess.run(
        [str(host), str(so), *steps],
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=HOST_TIMEOUT_S,
    )


def _rcs(r: subprocess.CompletedProcess) -> list[int]:
    return [int(line.split()[1]) for line in r.stdout.splitlines() if line.startswith("rc ")]


def _alive(r: subprocess.CompletedProcess) -> bool:
    return "alive 1" in r.stdout


@requires_clang
class TestStickyFastTick:
    @pytest.mark.parametrize(("mode", "first_rc"), [("s", -11), ("e", EXIT_STATUS)])
    def test_regression_landing_restores_tick(self, rig, mode, first_rc):
        r = _go(rig, f"g{mode}:{BUDGET_US}", "t", f"gb:{BUDGET_US}")

        assert _rcs(r) == [first_rc, 0], (r.stdout, r.stderr)
        assert "tick 0" in r.stdout

    def test_control_return_restores_tick(self, rig):
        r = _go(rig, f"gn:{BUDGET_US}", "t", f"gb:{BUDGET_US}")

        assert _rcs(r) == [0, 0], (r.stdout, r.stderr)
        assert "tick 0" in r.stdout

    def test_adversarial_real_hang_still_times_out(self, rig):
        """The landing reset must not disarm the timer: a call parked in libc
        spinning in target code still ends as a timeout."""
        r = _go(rig, f"gs:{BUDGET_US}", f"gh:{BUDGET_US}")

        assert _rcs(r) == [-11, TIMEOUT_RC], (r.stdout, r.stderr)


@requires_clang
class TestBigDoubleFree:
    def test_regression_big_double_free_recovers(self, rig):
        r = _go(rig, "gB")

        assert _rcs(r) == [SIGABRT_RC] and _alive(r), (r.stdout, r.stderr)
        assert "double free" in r.stderr

    def test_regression_big_realloc_after_free_recovers(self, rig):
        r = _go(rig, "gP")

        assert _rcs(r) == [SIGABRT_RC] and _alive(r), (r.stdout, r.stderr)
        assert "double free" in r.stderr

    def test_regression_stale_big_free_next_call(self, rig):
        r = _go(rig, "gF", "gS")

        assert _rcs(r) == [0, SIGABRT_RC] and _alive(r), (r.stdout, r.stderr)

    @pytest.mark.parametrize("alloc", ["m4194304", "a4194304"])
    def test_control_libc_big_chunk_free_passes(self, rig, alloc):
        r = _go(rig, "gF", alloc, "gL")

        assert _rcs(r) == [0, 0] and _alive(r), (r.stdout, r.stderr)

    def test_adversarial_libc_chunk_at_freed_address(self, rig):
        """A libc page-aligned mmapped chunk mapped where a private big block
        was freed is genuine: forwarded, not reported as a double free."""
        r = _go(rig, "gF", "f4194304", "gL")

        assert "fake 1" in r.stdout, r.stdout
        assert _rcs(r) == [0, 0] and _alive(r), (r.stdout, r.stderr)


@requires_clang
class TestReallocKeep:
    @pytest.mark.parametrize("mode", ["k", "i"])
    def test_regression_keep_path_checks_slot(self, rig, mode):
        r = _go(rig, f"g{mode}")

        assert _rcs(r) == [SIGABRT_RC] and _alive(r), (r.stdout, r.stderr)

    def test_control_same_class_realloc_keeps_pointer(self, rig):
        r = _go(rig, "gK")

        assert _rcs(r) == [0], (r.stdout, r.stderr)


@requires_clang
class TestStampWrap:
    def test_regression_wrapped_stamp_spares_old_slot(self, rig):
        r = _go(rig, "gA", "w", "gC", "gX")

        assert _rcs(r) == [0, -11, 0], (r.stdout, r.stderr)

    def test_control_crash_spares_old_slot(self, rig):
        r = _go(rig, "gA", "gC", "gX")

        assert _rcs(r) == [0, -11, 0], (r.stdout, r.stderr)

    def test_adversarial_crash_reclaims_own_slot(self, rig):
        """Reclaim still frees the crashing call's own slot: freeing it after
        is a double free."""
        r = _go(rig, "gD", "gX")

        assert _rcs(r) == [-11, SIGABRT_RC] and _alive(r), (r.stdout, r.stderr)


def _fork(rig, mode: str) -> subprocess.CompletedProcess:
    _, fork_host, so = rig
    return subprocess.run(
        [str(fork_host), str(so), mode],
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=HOST_TIMEOUT_S,
    )


@requires_clang
class TestForkLock:
    def test_regression_child_mallocs_while_lock_held(self, rig):
        r = _fork(rig, "hold")

        assert "child exit 0" in r.stdout, (r.stdout, r.stderr)

    def test_control_child_mallocs_without_holder(self, rig):
        r = _fork(rig, "free")

        assert "child exit 0" in r.stdout, (r.stdout, r.stderr)

    def test_adversarial_parent_keeps_holder_lock(self, rig):
        """The reset is the child's only: the parent's holder still owns it."""
        r = _fork(rig, "hold")

        assert "parent_locked 1" in r.stdout, (r.stdout, r.stderr)
