"""Regressions from the afl_shim.c audit (2026-11-13).

Handover: docs/handover/handover_afl_shim_audit_2026-11-13.md.

Each case compiles a small C driver with the shim -included, so the driver
reaches the shim's statics as test seams, and runs it as a subprocess: the
shim's signal handlers never reach pytest.
"""

from __future__ import annotations

import os
import signal
import struct
import subprocess
import time
from pathlib import Path

from fuzzer_tool.adapters.shm import ShmCoverage
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import SHIM, STATUS_TIMEOUT_S, _build, _clean_env, _Forkserver

RUN_TIMEOUT_S = 30
BUDGET_US = 20000
TIMEOUT_RC = -1


def _run(exe: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(exe), *args],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=_clean_env(**env),
    )


# ── Crash handler: backtrace over a smashed frame chain ───────────────

# leaf() overwrites its saved rbp; parent() then faults on an rbp-relative
# load with a frame chain the unwinder cannot follow.
_SMASHED_FRAME = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define BUF 16
#define SMASH (BUF + 8)
#define FILL 0x41
#define BENIGN 7

__attribute__((noinline)) static void leaf(const uint8_t *d, size_t n) {
    char buf[BUF];
    memcpy(buf, d, n > BUF ? BUF : n);
    if (n > BUF) memcpy(__builtin_frame_address(0), d, sizeof(void *));
}

__attribute__((noinline)) static int parent(const uint8_t *d, size_t n) {
    volatile int local = BENIGN;
    leaf(d, n);
    return local;
}

int entry(const uint8_t *d, size_t n) { return parent(d, n); }

int main(void) {
    uint8_t in[SMASH];
    memset(in, FILL, sizeof in);
    printf("crash %d\n", __afl_guarded_call(entry, in, sizeof in));
    printf("benign %d\n", __afl_guarded_call(entry, in, BUF / 2));
    return 0;
}
"""
_SMASH_FLAGS = ("-O0", "-fno-stack-protector", "-U_FORTIFY_SOURCE")


@requires_clang
def test_regression_smashed_frame_keeps_host_alive(tmp_path):
    exe = _build(tmp_path, _SMASHED_FRAME, *_SMASH_FLAGS)
    sym = tmp_path / "sym.out"

    # Control: without the symbol sink the capture never unwinds.
    ctl = _run(exe)
    assert ctl.returncode == 0, ctl.stderr
    crash_rc = int(ctl.stdout.splitlines()[0].split()[1])
    assert -crash_rc in (signal.SIGSEGV, signal.SIGBUS)

    r = _run(exe, __AFL_CRASH_SYM_OUT=str(sym))
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert r.stdout == ctl.stdout

    # The record survives with the faulting pc only.
    line = sym.read_text().splitlines()[0].split()
    assert line[0] == "SYM"
    assert int(line[1]) == -crash_rc


# ── Guard timer signals vs a target resetting every disposition ───────

_RT_RESET = r"""
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#define SETTLE_US 50000

static void (*disp)(int);

static int reset_all(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    for (int s = 1; s < NSIG; s++) signal(s, disp);
    return 0;
}

static int hang(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    for (volatile int spin = 1; spin;) {}
    return 0;
}

int main(int argc, char **argv) {
    (void)argv;
    disp = argc > 1 ? SIG_IGN : SIG_DFL;
    printf("reset %d\n", __afl_guarded_call_timeout(reset_all, 0, 0, BUDGET_US));
    usleep(SETTLE_US);                 /* a stray tick must not kill the host */
    printf("hang %d\n", __afl_guarded_call_timeout(hang, 0, 0, BUDGET_US));
    return 0;
}
"""


def _rt_reset(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    exe = _build(tmp_path, _RT_RESET, f"-DBUDGET_US={BUDGET_US}")
    return _run(exe, *args)


@requires_clang
def test_regression_target_sig_dfl_keeps_timer_signals(tmp_path):
    r = _rt_reset(tmp_path)
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert r.stdout.splitlines() == ["reset 0", f"hang {TIMEOUT_RC}"]


@requires_clang
def test_regression_target_sig_ign_keeps_timeouts(tmp_path):
    r = _rt_reset(tmp_path, "ign")
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert r.stdout.splitlines() == ["reset 0", f"hang {TIMEOUT_RC}"]


# ── Worker exit() while the owner is inside glibc malloc ──────────────

# The owner allocates in a loop; a worker calls exit(). A jump out of
# malloc leaves the arena lock held and the host's next malloc never
# returns: SIGALRM reports it.
_WORKER_EXIT = r"""
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define BIG 70000
#define SMALL 5000
#define HOST_ALLOCS 1000
#define WORKER_DELAY_US 3000
#define DEADLOCK_ALARM_S 3
#define DEADLOCK_RC 9

static void churn(void) {
    void *volatile p = malloc(BIG);
    void *volatile q = malloc(SMALL);
    free(p);
    free(q);
}

static void *worker(void *a) {
    (void)a;
    usleep(WORKER_DELAY_US);
    exit(EXIT_STATUS);
    return 0;
}

static int entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    pthread_t t;
    pthread_create(&t, 0, worker, 0);
    for (;;) churn();
    return 0;
}

static void hung(int s) {
    (void)s;
    static const char msg[] = "deadlock\n";
    (void)!write(1, msg, sizeof(msg) - 1);
    syscall(SYS_exit_group, DEADLOCK_RC);
}

int main(void) {
    signal(SIGALRM, hung);
    for (int i = 0; i < REPS; i++) {
        int rc = __afl_guarded_call(entry, 0, 0);
        alarm(DEADLOCK_ALARM_S);
        for (int k = 0; k < HOST_ALLOCS; k++) churn();
        alarm(0);
        printf("rc %d\n", rc);
        fflush(stdout);
    }
    return 0;
}
"""
EXIT_STATUS = 3
WORKER_EXIT_REPS = 20  # unfixed: the jump lands in malloc in > 1/3 of calls


@requires_clang
def test_regression_worker_exit_defers_in_malloc(tmp_path):
    exe = _build(
        tmp_path,
        _WORKER_EXIT,
        f"-DEXIT_STATUS={EXIT_STATUS}",
        f"-DREPS={WORKER_EXIT_REPS}",
        "-pthread",
    )
    r = _run(exe)
    assert r.returncode == 0, (r.returncode, r.stdout[-200:], r.stderr)
    assert r.stdout.splitlines() == [f"rc {EXIT_STATUS}"] * WORKER_EXIT_REPS


# ── vfork() child leaving through _exit() ─────────────────────────────

_VFORK = r"""
#include <stdint.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

static int entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    int st = 0;
    pid_t p = vfork();
    if (p == 0) _exit(CHILD_STATUS);    /* as after a failed execve() */
    waitpid(p, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) + PARENT_BIAS : -1;
}

int main(void) {
    pid_t host = getpid();
    int rc = __afl_guarded_call(entry, 0, 0);
    printf("rc %d host %d\n", rc, getpid() == host);
    return 0;
}
"""
VFORK_CHILD_STATUS = 7
VFORK_PARENT_BIAS = 100  # tells the entry's return from the child's exit code


@requires_clang
def test_regression_vfork_child_exit_stays_in_child(tmp_path):
    exe = _build(
        tmp_path,
        _VFORK,
        f"-DCHILD_STATUS={VFORK_CHILD_STATUS}",
        f"-DPARENT_BIAS={VFORK_PARENT_BIAS}",
    )
    r = _run(exe)
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert r.stdout.splitlines() == [f"rc {VFORK_CHILD_STATUS + VFORK_PARENT_BIAS} host 1"]


# ── Persistent loop: per-input work a fork child gets at exit ─────────

# main is uninstrumented so only work() ticks counters. Seams: the sancov
# regions and the cmplog dedup generation.
_PERSIST_SYNC = r"""
#include <stdint.h>
#include <stdio.h>

__attribute__((noinline)) static int work(const unsigned char *b, long n) {
    int s = 0;
    if (n > 0 && b[0] == 'a') s += 1;
    if (n > 1 && b[1] == 'b') s += 2;
    return s;
}

__attribute__((no_sanitize("coverage"))) static unsigned live_counters(void) {
    unsigned live = 0;
    for (uint32_t r = 0; r < __afl_sancov_nregions; r++)
        for (uint8_t *p = __afl_sancov_regions[r].start; p < __afl_sancov_regions[r].stop; p++)
            live += *p != 0;
    return live;
}

volatile int sink;

__attribute__((no_sanitize("coverage"))) int main(int argc, char **argv) {
    (void)argv;
    sink = work((const unsigned char *)"ab", argc + 1);   /* opaque length */
    unsigned before = live_counters();
    uint32_t scope = __afl_cmplog_gen;

    __afl_persist_sync();

    printf("attached %d ticked %d live %u scopes %u\n", __afl_area != 0, before > 0,
           live_counters(), (unsigned)(__afl_cmplog_gen - scope));
    return 0;
}
"""
MAP_ENTRIES = 8192


@requires_clang
def test_regression_persist_sync_folds_and_rescopes(tmp_path):
    exe = _build(
        tmp_path,
        _PERSIST_SYNC,
        "-fsanitize-coverage=inline-8bit-counters",
        "-D__AFL_CMPLOG=1",
    )
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        r = _run(exe, __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(shm.size))
        edges = len(shm.get_edge_ids())
    finally:
        shm.cleanup()

    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["attached", "1", "ticked", "1", "live", "0", "scopes", "1"]
    assert edges >= 1


# ── mprotect write guard across a re-attach ───────────────────────────

_WGUARD_COMMON = r"""
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <unistd.h>
#define POLL_US 1000
#define ID_CHARS 32
#define SEG_MODE 0600
#define EDGE_LOC 0x1234

/* Header + table + tail + touched bitmap, as shm.py sizes it. */
static int seg(unsigned n) {
    size_t sz = SHM_TABLE_OFFSET + (size_t)n * sizeof(struct __afl_entry) + SHM_TAIL_BYTES
              + SHM_TOUCHED_HEADER + (n / 64) * sizeof(uint64_t);
    return shmget(IPC_PRIVATE, sz, IPC_CREAT | SEG_MODE);
}

static void attach(int id, unsigned n) {
    char b[ID_CHARS];
    snprintf(b, sizeof b, "%d", id);
    setenv("__AFL_SHM_ID", b, 1);
    snprintf(b, sizeof b, "%u", n);
    setenv("AFL_MAP_SIZE", b, 1);
    __afl_map_shm();
}
"""

# A thread holds a bracket across the resize re-attach, then closes it.
_WGUARD_RESIZE = (
    _WGUARD_COMMON
    + r"""
enum { IDLE, HELD, RELEASE, CLOSED };
static volatile int stage = IDLE;

static void *holder(void *a) {
    (void)a;
    __afl_wtok_t t = __afl_wguard_open();
    stage = HELD;
    while (stage != RELEASE) usleep(POLL_US);
    __afl_wguard_close(t);
    stage = CLOSED;
    return 0;
}

int main(void) {
    int a = seg(MAP), b = seg(2 * MAP);
    attach(a, MAP);

    pthread_t th;
    pthread_create(&th, 0, holder, 0);
    while (stage != HELD) usleep(POLL_US);
    attach(b, 2 * MAP);
    stage = RELEASE;
    pthread_join(th, 0);

    shmctl(a, IPC_RMID, 0);
    shmctl(b, IPC_RMID, 0);
    printf("mode %d depth %u\n", __afl_wguard_mode == __AFL_WGUARD_MPROTECT, __afl_wguard_depth);
    fflush(stdout);
    __afl_map_edge(EDGE_LOC);
    puts("edge ok");
    return 0;
}
"""
)

# The same id attached twice: the mapping is reused, already write-locked.
_WGUARD_REUSE = (
    _WGUARD_COMMON
    + r"""
int main(void) {
    int a = seg(MAP);
    attach(a, MAP);
    shmctl(a, IPC_RMID, 0);
    attach(a, MAP);
    printf("mode %d touched %d\n", __afl_wguard_mode == __AFL_WGUARD_MPROTECT, __afl_touched != 0);
    return 0;
}
"""
)


def _wguard(tmp_path: Path, src: str) -> subprocess.CompletedProcess:
    exe = _build(tmp_path, src, f"-DMAP={MAP_ENTRIES}", "-pthread")
    return _run(exe, __AFL_WGUARD="mprotect")


@requires_clang
def test_regression_wguard_depth_survives_reattach(tmp_path):
    r = _wguard(tmp_path, _WGUARD_RESIZE)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert r.stdout.splitlines() == ["mode 1 depth 0", "edge ok"]


@requires_clang
def test_regression_wguard_same_id_reattach(tmp_path):
    r = _wguard(tmp_path, _WGUARD_REUSE)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert r.stdout.splitlines() == ["mode 1 touched 1"]


def _cc(out: Path, *args: str | Path) -> Path:
    """clang with a custom command line (link flags, preload-only builds)."""
    r = subprocess.run(["clang", *map(str, args), "-o", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


# ── Pending re-resolve vs a nested first-time miss ────────────────────

# A dlsym wrapper that calls strncmp on every lookup and memchr on "strn*"
# names: the memchr miss lands while the strncmp slot is being re-resolved.
_PENDING = r"""
#include <stdio.h>
#include <string.h>
#define PROBE_LEN 4
#define SCAN_LEN 7
#define EARLY_CTOR 101

void *__real_dlsym(void *h, const char *name);

void *__wrap_dlsym(void *h, const char *name) {
    static void *volatile sink;
    if (strncmp(name, "zzz_", PROBE_LEN) == 0) return NULL;
    if (strncmp(name, "strn", PROBE_LEN) == 0) sink = memchr(name, 'q', SCAN_LEN);
    return __real_dlsym(h, name);
}

static const char *who(void *p) {
    if (p == (void *)__afl_fb_strncmp || p == (void *)__afl_fb_memchr) return "fallback";
    if (p == __real_dlsym(RTLD_NEXT, "strncmp")) return "strncmp";
    if (p == __real_dlsym(RTLD_NEXT, "memchr")) return "memchr";
    return "other";
}

/* Ahead of the shim's constructors: the first interposed call, a lookup. */
__attribute__((constructor(EARLY_CTOR))) static void early(void) {
    static volatile int r;
    r = memcmp("ab", "ac", 2);
}

int main(void) {
    printf("strncmp=%s memchr=%s\n", who((void *)real_strncmp), who((void *)real_memchr));
    return 0;
}
"""


@requires_clang
def test_regression_pending_resolve_keeps_its_slot(tmp_path):
    src = tmp_path / "pend.c"
    src.write_text(_PENDING)
    exe = _cc(
        tmp_path / "pend",
        "-O1", "-D__AFL_CMPLOG=1", "-fno-builtin", "-include", SHIM, src,
        "-Wl,--wrap=dlsym", "-ldl", "-pthread",
    )  # fmt: skip
    r = _run(exe)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["strncmp=strncmp", "memchr=memchr"]


# ── Preload-only shim: software-sent fault signals ────────────────────

_RAISE = r"""
#include <signal.h>
#include <stdio.h>
int main(int argc, char **argv) {
    (void)argv;
    if (argc > 1) *(volatile int *)0 = 1;   /* control: a hardware fault */
    raise(SIGSEGV);
    puts("survived");
    return 0;
}
"""


@requires_clang
def test_regression_preload_shim_keeps_raised_fault(tmp_path):
    src = tmp_path / "p.c"
    src.write_text(_RAISE)
    exe = _cc(tmp_path / "p", "-O1", src)
    pre = _cc(
        tmp_path / "pre.so",
        "-O1", "-fPIC", "-shared", "-D__AFL_PRELOAD_ONLY", "-x", "c", SHIM, "-ldl",
    )  # fmt: skip

    # Control: the hardware-fault path was never broken.
    hw = _run(exe, "hw", LD_PRELOAD=str(pre))
    assert hw.returncode == -signal.SIGSEGV

    sw = _run(exe, LD_PRELOAD=str(pre))
    assert sw.returncode == -signal.SIGSEGV, sw.stdout


# ── Caller-context caches ─────────────────────────────────────────────

# A handler on the main stack walks its own context between the miss
# path's two stores; main then paired its return address with that ctx.
# Statistical by nature: the control (no signals) must read 0 too.
_CTX_SIGNAL = r"""
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#define NI __attribute__((noinline))

static volatile uint32_t sink;
NI static uint32_t probe(void) { uint32_t c = __afl_get_caller_ctx(); __asm__(""); return c; }
NI static uint32_t g1(void) { uint32_t c = probe(); __asm__(""); return c; }
NI static uint32_t g2(void) { uint32_t c = probe(); __asm__(""); return c; }
/* The ctx is keyed on the return address into c1/c2: one call site each. */
NI static uint32_t c1(void) { uint32_t c = g1(); __asm__(""); return c; }
NI static uint32_t c2(void) { uint32_t c = g2(); __asm__(""); return c; }
NI static uint32_t in_handler(void) { uint32_t c = probe(); __asm__(""); return c; }

static void on_sig(int s) { (void)s; sink = in_handler(); }

static atomic_int stop;
static pthread_t main_th;
static void *killer(void *a) {
    (void)a;
    while (!stop) pthread_kill(main_th, SIGUSR1);
    return 0;
}

int main(int argc, char **argv) {
    (void)argv;
    uint32_t e1 = c1(), e2 = c2();
    struct sigaction sa = {0};
    sa.sa_handler = on_sig;
    sigaction(SIGUSR1, &sa, 0);

    main_th = pthread_self();
    pthread_t t;
    int signals = argc == 1;
    if (signals) pthread_create(&t, 0, killer, 0);

    long wrong = 0;
    for (long i = 0; i < CALLS; i++) {    /* miss, then a hit on what it cached */
        wrong += c1() != e1;
        wrong += c1() != e1;
        wrong += c2() != e2;
        wrong += c2() != e2;
    }

    stop = 1;
    if (signals) pthread_join(t, 0);
    printf("distinct %d wrong %ld\n", e1 != e2, wrong);
    return 0;
}
"""
CTX_CALLS = 3_000_000  # unfixed: hundreds of wrong contexts at this count


@requires_clang
def test_regression_ctx_cache_survives_signal_handler(tmp_path):
    exe = _build(tmp_path, _CTX_SIGNAL, f"-DCALLS={CTX_CALLS}", "-D__AFL_CTX_BITS=32", "-pthread")

    ctl = _run(exe, "nosignals")
    assert ctl.stdout.split() == ["distinct", "1", "wrong", "0"], ctl.stderr

    r = _run(exe)
    assert r.stdout.split() == ["distinct", "1", "wrong", "0"], r.stderr


# Another writer holds the slot: this one must resolve without caching.
_CTX_BASE_CLAIM = r"""
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#ifndef __AFL_CTX_BASE_BUSY
#define __AFL_CTX_BASE_BUSY 1
#endif
#define FOREIGN_BASE 0x5a5a000u

int main(void) {
    uintptr_t key = (uintptr_t)&main + 8;
    Dl_info info;
    dladdr((void *)key, &info);
    uint32_t slot = (uint32_t)((key >> 2) & (__AFL_CTX_CACHE_SIZE - 1));

    __afl_ctx_base_cache[slot].ra = __AFL_CTX_BASE_BUSY;
    __afl_ctx_base_cache[slot].base = FOREIGN_BASE;
    uintptr_t got = __afl_ctx_resolve_base((void *)key);

    printf("resolved %d untouched %d\n", got == (uintptr_t)info.dli_fbase,
           __afl_ctx_base_cache[slot].ra == __AFL_CTX_BASE_BUSY &&
           __afl_ctx_base_cache[slot].base == FOREIGN_BASE);

    /* A free slot is still claimed and cached. */
    __afl_ctx_base_cache[slot].ra = 0;
    __afl_ctx_resolve_base((void *)key);
    printf("cached %d\n", __afl_ctx_base_cache[slot].ra == key &&
           __afl_ctx_base_cache[slot].base == (uintptr_t)info.dli_fbase);
    return 0;
}
"""


@requires_clang
def test_regression_ctx_base_cache_writer_claims_slot(tmp_path):
    r = _run(_build(tmp_path, _CTX_BASE_CLAIM))
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["resolved 1 untouched 1", "cached 1"]


# ── Node bitmap: two threads, two bits of one byte ────────────────────

# Each round clears the byte and releases both threads onto one probe
# each. A plain |= loses one bit when the two load-or-store pairs overlap.
_NODE_BITS = r"""
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#define THREADS 2
#define TABLE_CAP 4u
#define BOTH_BITS 3

static struct __afl_dist_entry table[TABLE_CAP];
static uint8_t bitmap[1];
static struct __afl_entry area_stub[1];
static atomic_long go[THREADS], done;
static atomic_int stop;

static void *prober(void *a) {
    long me = (long)a, seen = 0;
    for (;;) {
        while (atomic_load(&go[me]) == seen)
            if (atomic_load(&stop)) return 0;
        seen++;
        __afl_probe_distance((uint64_t)me + 1);
        atomic_fetch_add(&done, 1);
    }
}

int main(void) {
    for (uint32_t k = 1; k <= THREADS; k++) {    /* key k sits in slot k, sets bit k-1 */
        table[k].key = k;
        table[k].node_idx = k - 1;
    }
    __afl_dist_table = table;
    __afl_dist_cap = TABLE_CAP;
    __afl_dist_window = TABLE_CAP;
    __afl_node_bitmap = bitmap;
    __afl_node_bitmap_bytes = sizeof bitmap;
    __afl_area = area_stub;

    pthread_t th[THREADS];
    for (long i = 0; i < THREADS; i++) pthread_create(&th[i], 0, prober, (void *)i);

    long lost = 0;
    for (long r = 1; r <= ROUNDS; r++) {
        bitmap[0] = 0;
        for (int i = 0; i < THREADS; i++) atomic_store(&go[i], r);
        while (atomic_load(&done) != r * THREADS) {}
        lost += bitmap[0] != BOTH_BITS;
    }

    atomic_store(&stop, 1);
    for (int i = 0; i < THREADS; i++) pthread_join(th[i], 0);
    printf("lost %ld\n", lost);
    return 0;
}
"""
NODE_ROUNDS = 400_000


@requires_clang
def test_regression_node_bitmap_or_is_atomic(tmp_path):
    exe = _build(tmp_path, _NODE_BITS, f"-DROUNDS={NODE_ROUNDS}", "-pthread")
    r = _run(exe)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["lost", "0"]


# ── Refused re-attach: header pointers under a thread mid-edge ────────

_REFUSED = (
    _WGUARD_COMMON
    + r"""
int main(void) {
    int a = seg(MAP);
    attach(a, MAP);
    shmctl(a, IPC_RMID, 0);

    setenv("AFL_MAP_SIZE", "-1", 1);      /* refused: the segment is detached */
    __afl_map_shm();

    /* Coverage is off, yet a thread already past its checks still loads
     * through these: they stay on the retired, still mapped segment. */
    printf("off %d header %d\n", __afl_area == 0,
           __afl_gen_word && __afl_path_hash && __afl_edge_count && __afl_stack_depth &&
           __afl_dropped && __afl_touched);
    return 0;
}
"""
)


@requires_clang
def test_regression_refused_reattach_keeps_header_mapped(tmp_path):
    r = _run(_build(tmp_path, _REFUSED, f"-DMAP={MAP_ENTRIES}", "-pthread"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["off", "1", "header", "1"]


# ── Sancov dump fd: stolen number ─────────────────────────────────────

_SANCOV_FD = r"""
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#define FILE_MODE 0600

static const uintptr_t pcs[] = {0x1000, 0, 0x1010, 0};

int main(int argc, char **argv) {
    (void)argc;
    int fd = __afl_sancov_open();
    close(fd);                              /* closefrom()-style target */
    int own = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, FILE_MODE);

    __afl_sancov_dump(__AFL_SANCOV_PCS, 0, pcs, pcs + sizeof pcs / sizeof pcs[0]);

    struct stat st;
    fstat(own, &st);
    printf("opened %d reused %d own_bytes %ld\n", fd >= 0, own == fd, (long)st.st_size);
    return 0;
}
"""


@requires_clang
def test_regression_sancov_dump_skips_stolen_fd(tmp_path):
    exe = _build(tmp_path, _SANCOV_FD)
    out = tmp_path / "sancov.out"
    r = _run(exe, str(tmp_path / "own.txt"), __AFL_SANCOV_OUT=str(out))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["opened", "1", "reused", "1", "own_bytes", "0"]
    assert out.stat().st_size > 0


# ── Private heap: big blocks mapped and unmapped under the heap lock ──

# A timeout jump between mmap and the table insert (or between the removal
# and munmap) leaked the mapping. Under the lock the tick defers.
_BIG_LOCKED = r"""
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#define BIG (4u << 20)          /* above the largest class */

void *__real_mmap(void *a, size_t n, int prot, int flags, int fd, off_t off);
int __real_munmap(void *a, size_t n);
static int map_locked = -1, unmap_locked = -1;

void *__wrap_mmap(void *a, size_t n, int prot, int flags, int fd, off_t off) {
    if (n >= BIG) map_locked = __afl_ph.lock != 0;
    return __real_mmap(a, n, prot, flags, fd, off);
}

int __wrap_munmap(void *a, size_t n) {
    if (n >= BIG) unmap_locked = __afl_ph.lock != 0;
    return __real_munmap(a, n);
}

int main(void) {
    void *volatile p = malloc(BIG);
    free(p);
    printf("map_locked %d unmap_locked %d\n", map_locked, unmap_locked);
    return 0;
}
"""


@requires_clang
def test_regression_big_block_mapped_under_heap_lock(tmp_path):
    src = tmp_path / "big.c"
    src.write_text(_BIG_LOCKED)
    exe = _cc(
        tmp_path / "big",
        "-O1", "-D__AFL_PRIVATE_HEAP=1", "-include", SHIM, src,
        "-Wl,--wrap=mmap", "-Wl,--wrap=munmap", "-ldl", "-pthread",
    )  # fmt: skip
    r = _run(exe)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["map_locked", "1", "unmap_locked", "1"]


# ── Fallback testcase read vs the previous call's poisoned tail ───────

# Stand-ins for the ASAN region calls record the poisoned tail; read() is
# wrapped to check the range it is about to fill.
_POISONED_READ = r"""
#include <stdio.h>
#include <unistd.h>

static const char *poison_lo, *poison_hi;
static int read_into_poison;

void __asan_poison_memory_region(void const volatile *a, size_t n) {
    poison_lo = (const char *)a;
    poison_hi = poison_lo + n;
}

void __asan_unpoison_memory_region(void const volatile *a, size_t n) {
    const char *lo = (const char *)a, *hi = lo + n;
    if (lo <= poison_lo && hi > poison_lo) poison_lo = hi < poison_hi ? hi : poison_hi;
}

ssize_t __real_read(int fd, void *buf, size_t n);
ssize_t __wrap_read(int fd, void *buf, size_t n) {
    const char *lo = (const char *)buf, *hi = lo + n;
    if (lo < poison_hi && hi > poison_lo) read_into_poison = 1;
    return __real_read(fd, buf, n);
}

int main(void) {
    int first = __afl_fuzz_len();       /* stdin, then poisons the tail */
    int second = __afl_fuzz_len();      /* reads into the same buffer */
    printf("first %d second %d read_into_poison %d\n", first, second, read_into_poison);
    return 0;
}
"""
POISON_INPUT = b"abc"


@requires_clang
def test_regression_fallback_read_unpoisons_buffer(tmp_path):
    src = tmp_path / "rd.c"
    src.write_text(_POISONED_READ)
    exe = _cc(tmp_path / "rd", "-O1", "-include", SHIM, src, "-Wl,--wrap=read", "-ldl", "-pthread")
    r = subprocess.run(
        [str(exe)], input=POISON_INPUT, capture_output=True, timeout=RUN_TIMEOUT_S, env=_clean_env()
    )
    assert r.returncode == 0, r.stderr
    assert (
        r.stdout.split()
        == f"first {len(POISON_INPUT)} second 0 read_into_poison 0".encode().split()
    )


# ── LTL sink: an opener that never finishes ───────────────────────────

# Seam: the fd word is left at "opening", as a fork during another
# thread's open (child) or a signal handler on the opening thread sees it.
_LTL_OPENING = r"""
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>
#define HANG_ALARM_S 5

int main(int argc, char **argv) {
    (void)argv;
    __ltl_fd = LTL_FD_OPENING;
    alarm(HANG_ALARM_S);

    if (argc > 1) {                       /* fork child: the opener is gone */
        pid_t pid = fork();
        if (pid == 0) _exit(__ltl_open() >= 0 ? 0 : 1);
        int st = 0;
        waitpid(pid, &st, 0);
        printf("child_opened %d\n", WIFEXITED(st) && WEXITSTATUS(st) == 0);
        return 0;
    }

    printf("gave_up %d\n", __ltl_open() < 0);    /* same thread: bounded wait */
    return 0;
}
"""


@requires_clang
def test_regression_ltl_open_wait_is_bounded(tmp_path):
    exe = _build(tmp_path, _LTL_OPENING)
    r = _run(exe, __LTL_EVENTS_OUT=str(tmp_path / "ltl.out"))
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert r.stdout.split() == ["gave_up", "1"]


@requires_clang
def test_regression_ltl_open_recovers_in_fork_child(tmp_path):
    exe = _build(tmp_path, _LTL_OPENING)
    r = _run(exe, "fork", __LTL_EVENTS_OUT=str(tmp_path / "ltl.out"))
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert r.stdout.split() == ["child_opened", "1"]


# ── Crash path outside a guard: the process's own alarm ───────────────

# The target recovers from a fault with its own handler. The shim's
# handler symbolizes first, under its own alarm.
_ALARM_KEPT = r"""
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#define WATCHDOG_S 100

static sigjmp_buf recover;
static void on_segv(int s) { (void)s; siglongjmp(recover, 1); }
static int ok(const uint8_t *d, size_t n) { (void)d; (void)n; return 0; }

int main(void) {
    __afl_guarded_call(ok, 0, 0);         /* shim handlers installed */
    signal(SIGSEGV, on_segv);             /* kept as the previous owner */
    alarm(WATCHDOG_S);

    if (sigsetjmp(recover, 1) == 0) *(volatile int *)0 = 1;

    printf("watchdog_left %d\n", alarm(0) > 0);
    return 0;
}
"""


@requires_clang
def test_regression_crash_path_keeps_pending_alarm(tmp_path):
    exe = _build(tmp_path, _ALARM_KEPT)
    sym = tmp_path / "sym.out"
    r = _run(exe, __AFL_CRASH_SYM_OUT=str(sym))
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert r.stdout.split() == ["watchdog_left", "1"]
    assert sym.read_text().startswith("SYM ")  # the symbolizing path ran


# ── Forkserver: the parked persistent child ───────────────────────────

# One byte appended per input run; SIGHUP ignored as under nohup, so an
# orphaned parked child survives the kernel's SIGHUP + SIGCONT.
_PARKED = r"""
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#define EARLY_CTOR 101
#define LOOP_MAX 1000
#define LOG_MODE 0600

__attribute__((constructor(EARLY_CTOR))) static void early(void) { signal(SIGHUP, SIG_IGN); }

int main(void) {
    while (__AFL_LOOP(LOOP_MAX)) {
        int fd = open(getenv("RUN_LOG"), O_WRONLY | O_APPEND | O_CREAT, LOG_MODE);
        (void)!write(fd, "x", 1);
        close(fd);
    }
    return 0;
}
"""
PROC_STATE_FIELD = 2  # /proc/<pid>/stat: pid (comm) state ...
DEAD_STATES = ("Z", "X")
POLL_S = 0.01


def _gone(pid: int) -> bool:
    """True once *pid* is dead: reaped, or a zombie awaiting its parent."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return True
    return stat.rsplit(")", 1)[1].split()[PROC_STATE_FIELD - 2] in DEAD_STATES


def _wait_gone(pid: int) -> bool:
    deadline = time.monotonic() + STATUS_TIMEOUT_S
    while time.monotonic() < deadline:
        if _gone(pid):
            return True
        time.sleep(POLL_S)
    return _gone(pid)


class _PersistentServer(_Forkserver):
    """As _Forkserver, keeping the pid the server reports for each input."""

    def run_one(self) -> int | None:
        os.write(self._ctl_w, b"\0\0\0\0")
        (self.child,) = struct.unpack("<i", self._read4())
        status = self._read4()
        return struct.unpack("<i", status)[0] if len(status) == 4 else None


def _parked_server(tmp_path: Path) -> tuple[_PersistentServer, Path]:
    exe = _build(tmp_path, _PARKED, "-fsanitize-coverage=trace-pc-guard")
    log = tmp_path / "runs.log"
    return _PersistentServer(exe, [], _clean_env(__AFL_PERSISTENT="1", RUN_LOG=str(log))), log


def _reap(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


@requires_clang
def test_regression_forkserver_replaces_dead_parked_child(tmp_path):
    fs, log = _parked_server(tmp_path)
    try:
        assert fs.run_one() == 0
        first = fs.child
        os.kill(first, signal.SIGKILL)  # a late timeout kill, the OOM killer
        assert _wait_gone(first)

        status = fs.run_one()
        second = fs.child
    finally:
        fs.close()
        _reap(fs.child)

    assert status == 0, "the corpse's status was reported for an input that never ran"
    assert second != first
    assert len(log.read_text()) == 2


@requires_clang
def test_regression_forkserver_eof_takes_parked_child(tmp_path):
    fs, _log = _parked_server(tmp_path)
    try:
        assert fs.run_one() == 0
        os.close(fs._ctl_w)  # loader gone: the server ends on EOF
        fs.proc.wait(timeout=STATUS_TIMEOUT_S)
        gone = _wait_gone(fs.child)
    finally:
        os.close(fs._st_r)
        _reap(fs.child)

    assert gone, "parked child outlived its forkserver"


@requires_clang
def test_regression_forkserver_kill_takes_parked_child(tmp_path):
    fs, _log = _parked_server(tmp_path)
    try:
        assert fs.run_one() == 0
        fs.proc.kill()  # as the loader's stop_forkserver
        fs.proc.wait(timeout=STATUS_TIMEOUT_S)
        gone = _wait_gone(fs.child)
    finally:
        os.close(fs._ctl_w)
        os.close(fs._st_r)
        _reap(fs.child)

    assert gone, "parked child outlived its forkserver"
