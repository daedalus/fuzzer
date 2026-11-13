"""Regressions from the afl_shim.c audit (2026-11-13).

Handover: docs/handover/handover_afl_shim_audit_2026-11-13.md.

Each case compiles a small C driver with the shim -included, so the driver
reaches the shim's statics as test seams, and runs it as a subprocess: the
shim's signal handlers never reach pytest.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path

from fuzzer_tool.adapters.shm import ShmCoverage
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

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
    assert r.stdout.split() == "attached 1 ticked 1 live 0 scopes 1".split()
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
