"""Regressions from the second afl_shim.c audit (2026-10-02).

- a crash on a worker thread siglongjmp'd onto the guarded thread's stack:
  the host kept running on the wrong thread and hung at exit
- a stack overflow inside __afl_guarded_call killed the host: no
  alternate signal stack for the handler
- a stray SIGSYS (seccomp trap) was swallowed: only hardware faults may
  be re-executed
- a fork() while another thread held the cmplog lock left the child
  unable to log, and the child re-flushed the parent's buffered records
- a refused segment was reported (and counted) again on every retried
  __afl_map_shm() call
- memrchr logged the front of the buffer; it searches from the end
- health gaps: intercepted abort() calls and displaced crash handlers
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.shim_health import ShimField
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env, _cmplog_records

HANG_TIMEOUT_S = 30
MAP_ENTRIES = 8192

_SO_SRC = """
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
static volatile int sink;
__attribute__((noinline)) static int rec(int n) {
    volatile char pad[256];
    pad[0] = (char)n;
    return rec(n + 1) + pad[0];
}
int overflow_entry(const uint8_t *d, size_t n) { (void)d; return rec((int)n); }
static void *worker(void *a) { (void)a; volatile int *p = 0; sink = *p; return 0; }
int thread_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    pthread_t t;
    pthread_create(&t, 0, worker, 0);
    pthread_join(t, 0);
    return 7;
}
int crash_entry(const uint8_t *d, size_t n) { (void)d; volatile int *p = (int *)(n - n); return *p; }
int ok_entry(const uint8_t *d, size_t n) { (void)d; return (int)n; }
"""

_DRIVER = """
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
g = lib.__afl_guarded_call
g.restype = ctypes.c_int
g.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
def call(name, data=b"x"):
    return g(ctypes.cast(getattr(lib, name), ctypes.c_void_p), data, len(data))
"""


def _run_driver(so, body: str) -> subprocess.CompletedProcess | None:
    """Run *body* after _DRIVER in a child interpreter; None on a hang."""
    script = textwrap.dedent(_DRIVER) + textwrap.dedent(body)
    try:
        return subprocess.run(
            [sys.executable, "-c", script, str(so)],
            capture_output=True, text=True, timeout=HANG_TIMEOUT_S, env=_clean_env(),
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return None


def _so(tmp_path):
    return _build(tmp_path, _SO_SRC, "-fsanitize-coverage=trace-pc-guard", shared=True)


# ── Crash handler ──────────────────────────────────────────────────────


@requires_clang
def test_regression_worker_thread_crash_does_not_hijack_host(tmp_path):
    r = _run_driver(
        _so(tmp_path),
        """
        print("rc", call("thread_entry"), flush=True)
        """,
    )

    assert r is not None, "host hung after a worker-thread crash"
    assert r.returncode == -signal.SIGSEGV
    assert "rc" not in r.stdout  # never returned into the host on the wrong thread


@requires_clang
def test_regression_stack_overflow_recovers_in_guard(tmp_path):
    r = _run_driver(
        _so(tmp_path),
        """
        print(call("overflow_entry"), call("ok_entry", b"ab"), call("overflow_entry"))
        """,
    )

    assert r is not None and r.returncode == 0, r and r.stderr[-500:]
    sigsegv = int(signal.SIGSEGV)
    assert r.stdout.split() == [str(-sigsegv), "2", str(-sigsegv)]


@requires_clang
def test_regression_stray_sigsys_not_swallowed(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <stddef.h>
        #include <linux/filter.h>
        #include <linux/seccomp.h>
        #include <sys/prctl.h>
        #include <sys/syscall.h>
        #include <stdio.h>
        int main(void) {
            struct sock_filter f[] = {
                BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
                BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_getppid, 0, 1),
                BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
                BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
            };
            struct sock_fprog p = { sizeof f / sizeof f[0], f };
            prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
            prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &p);
            long r = syscall(SYS_getppid);
            printf("survived %ld\\n", r);
            return 0;
        }
        """,
        "-fsanitize-coverage=trace-pc-guard",
    )
    r = subprocess.run([str(exe)], capture_output=True, text=True, env=_clean_env())

    assert r.returncode == -signal.SIGSYS
    assert "survived" not in r.stdout


# ── cmplog across fork ─────────────────────────────────────────────────


_FORK_SRC = """
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
int main(void) {
    char a[8] = "AAAAAAA", b[8] = "BBBBBBB", z[8] = "ZZZZZZZ";
    volatile int r = memcmp(a, b, 8);           /* buffered in the parent */
    HOLD_LOCK
    pid_t pid = fork();
    if (pid == 0) {
        r += memcmp(a, z, 8);
        exit(0);                                /* destructors flush */
    }
    waitpid(pid, 0, 0);
    __atomic_store_n(&__afl_cmplog_lock, 0, __ATOMIC_RELEASE);
    return r == 0;
}
"""


@requires_clang
def test_regression_fork_child_logs_despite_inherited_lock(tmp_path):
    """Another parent thread held the lock at fork: the child must not inherit it."""
    src = _FORK_SRC.replace(
        "HOLD_LOCK", "__atomic_store_n(&__afl_cmplog_lock, 1, __ATOMIC_RELEASE);"
    )
    exe = _build(tmp_path, src, "-D__AFL_CMPLOG=1", "-fno-builtin-memcmp")
    recs = _cmplog_records(exe, tmp_path)

    assert sum("5a5a5a5a5a5a5a00" in x for x in recs) == 1, recs


@requires_clang
def test_regression_fork_child_does_not_reflush_parent_records(tmp_path):
    exe = _build(
        tmp_path, _FORK_SRC.replace("HOLD_LOCK", ""), "-D__AFL_CMPLOG=1", "-fno-builtin-memcmp"
    )
    recs = _cmplog_records(exe, tmp_path)

    assert sum("4242424242424200" in x for x in recs) == 1, recs
    assert sum("5a5a5a5a5a5a5a00" in x for x in recs) == 1, recs


@requires_clang
def test_regression_memrchr_logs_searched_tail(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <string.h>
        int main(void) {
            char buf[100];
            memset(buf, 'H', 50);
            memset(buf + 50, 'T', 50);
            return memrchr(buf, 'Z', sizeof buf) != 0;
        }
        """,
        "-D__AFL_CMPLOG=1",
        "-fno-builtin",
    )
    tail = ("48" * 14) + ("54" * 50)  # last 64 bytes: 14 x 'H', 50 x 'T'
    recs = [x.split()[1] for x in _cmplog_records(exe, tmp_path)]

    assert tail in recs, recs


# ── Segment refusal ────────────────────────────────────────────────────


@requires_clang
def test_regression_refused_segment_reported_once(tmp_path):
    so = _build(
        tmp_path, "int f(void) { return 0; }", "-fsanitize-coverage=trace-pc-guard", shared=True
    )
    shm = ShmCoverage(size=1024)
    script = """
        import ctypes, sys
        from fuzzer_tool.adapters.inprocess import read_shim_health
        lib = ctypes.CDLL(sys.argv[1])
        lib.__afl_map_shm()          # adapters/inprocess.py retries like this
        lib.__afl_map_shm()
        print(*read_shim_health(lib))
    """
    try:
        r = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script), str(so)],
            capture_output=True, text=True, timeout=60,
            env=_clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES)),
        )  # fmt: skip
    finally:
        shm.cleanup()

    assert r.returncode == 0, r.stderr
    assert r.stderr.count("too small") == 1
    assert int(r.stdout.split()[ShimField.SEG_REJECTED]) == 1


# ── Health counters ────────────────────────────────────────────────────


@requires_clang
def test_health_counts_intercepted_aborts(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <stdio.h>
        int main(void) {
            abort();
            abort();
            uint64_t h[16] = {0};
            uint32_t n = __afl_shim_health(h, 16);
            printf("%u %llu\\n", n, (unsigned long long)h[%d]);
            return 0;
        }
        """.replace("%d", str(int(ShimField.ABORTS_INTERCEPTED))),
        "-fsanitize-coverage=trace-pc-guard",
    )
    out = subprocess.run([str(exe)], capture_output=True, text=True, env=_clean_env(), check=True)
    n, aborts = (int(x) for x in out.stdout.split())

    assert n >= len(ShimField)
    assert aborts == 2


@requires_clang
def test_health_detects_and_repairs_displaced_handlers(tmp_path):
    """Adversarial: the host takes SIGSEGV after load; the guard must take it back."""
    r = _run_driver(
        _so(tmp_path),
        """
        import signal
        from fuzzer_tool.adapters.inprocess import read_shim_health
        signal.signal(signal.SIGSEGV, lambda *a: None)   # displaces the shim
        health = read_shim_health(lib)
        print(health[%d], call("crash_entry"), call("ok_entry", b"ab"), flush=True)
        """.replace("%d", str(int(ShimField.HANDLERS_DISPLACED))),
    )

    assert r is not None, "guarded crash looped under the host's handler"
    assert r.returncode == 0, r.stderr[-500:]
    displaced, rc, ok = r.stdout.split()
    assert int(displaced) >= 1
    assert (int(rc), int(ok)) == (-int(signal.SIGSEGV), 2)
