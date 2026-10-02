"""Adversarial hardening of afl_shim.c (2026-10-02 attack pass).

Each case is a target behaving badly inside, or around, __afl_guarded_call:

  - an infinite loop: the host's SIGALRM handler only sets a flag, so the
    call never returned and direct mode hung forever
  - exit()/_exit() inside the entry: the fuzzer process itself exited
  - a nested guarded call: after it returned, a crash in the outer entry
    killed the host (the inner call disarmed the shared jump buffer)
  - fork() inside the entry: the child inherited the armed guard, so its
    crash jumped into its copy of the fuzzer, which kept running
  - the target closed every fd and reopened one: cmplog records went into
    the target's own file
  - a garbage distance-SHM id (`<id>junk`) attached segment <id> via atoi
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap

from fuzzer_tool.adapters.shm import DistanceTableShm, ShmCoverage
from fuzzer_tool.core.shim_health import ShimField
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

HANG_TIMEOUT_S = 30
GUARD_TIMEOUT_RC = -1
TIMEOUT_US = 200_000
MAP_ENTRIES = 8192

_SO_SRC = """
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <err.h>
#include <errno.h>
#include <error.h>
typedef int (*entry_t)(const uint8_t *, size_t);
int __afl_guarded_call(entry_t, const uint8_t *, size_t);

int ok_entry(const uint8_t *d, size_t n) { (void)d; return (int)n; }
int crash_entry(const uint8_t *d, size_t n) { (void)d; volatile int *p = (int *)(n - n); return *p; }
int spin_entry(const uint8_t *d, size_t n) { (void)d; volatile size_t x = n; for (;;) x++; return 0; }
int exit_entry(const uint8_t *d, size_t n) { (void)d; (void)n; exit(3); }
int errx_entry(const uint8_t *d, size_t n) { (void)d; (void)n; errx(4, "bad input"); }
int error_entry(const uint8_t *d, size_t n) { (void)d; (void)n; error(6, ENOENT, "missing %s", "x"); return 0; }
int warn_entry(const uint8_t *d, size_t n) { (void)d; error(0, 0, "just a warning"); return (int)n; }
int nested_spin_entry(const uint8_t *d, size_t n) { return __afl_guarded_call(spin_entry, d, n); }
int uexit_entry(const uint8_t *d, size_t n) { (void)d; (void)n; _exit(5); }
int nested_entry(const uint8_t *d, size_t n) {
    __afl_guarded_call(ok_entry, d, n);
    return crash_entry(d, n);
}
int fork_crash_entry(const uint8_t *d, size_t n) {
    (void)d;
    pid_t p = fork();
    if (p == 0) { volatile int *q = (int *)(n - n); return *q; }
    int st = 0;
    waitpid(p, &st, 0);
    return WIFSIGNALED(st) ? 100 + WTERMSIG(st) : 200 + WEXITSTATUS(st);
}
"""

_DRIVER = """
import ctypes, os, sys
lib = ctypes.CDLL(sys.argv[1])
g = lib.__afl_guarded_call
g.restype = ctypes.c_int
g.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
def fn(name):
    return ctypes.cast(getattr(lib, name), ctypes.c_void_p)
def call(name, data=b"xyz"):
    return g(fn(name), data, len(data))
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


# ── Hangs ──────────────────────────────────────────────────────────────


@requires_clang
def test_regression_guarded_timeout_interrupts_hang(tmp_path):
    r = _run_driver(
        _so(tmp_path),
        f"""
        gt = lib.__afl_guarded_call_timeout
        gt.restype = ctypes.c_int
        gt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
        print(gt(fn("spin_entry"), b"x", 1, {TIMEOUT_US}), gt(fn("ok_entry"), b"ab", 2, {TIMEOUT_US}),
              gt(fn("crash_entry"), b"x", 1, {TIMEOUT_US}), call("ok_entry"))
        """,
    )

    assert r is not None, "hang was not interrupted"
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.split() == [str(GUARD_TIMEOUT_RC), "2", str(-int(signal.SIGSEGV)), "3"]


@requires_clang
def test_late_timer_never_reaches_host(tmp_path):
    """Adversarial: a 1 us budget races every return; the host must survive."""
    r = _run_driver(
        _so(tmp_path),
        """
        gt = lib.__afl_guarded_call_timeout
        gt.restype = ctypes.c_int
        gt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
        rcs = {gt(fn("ok_entry"), b"abc", 3, 1) for _ in range(2000)}
        print(sorted(rcs))
        """,
    )

    assert r is not None and r.returncode == 0, r and r.stderr[-500:]
    assert set(eval(r.stdout)) <= {3, GUARD_TIMEOUT_RC}


@requires_clang
def test_inprocess_runner_reports_c_hang_as_timeout(tmp_path):
    so = _so(tmp_path)
    script = """
        import sys
        from fuzzer_tool.adapters.inprocess import InProcessRunner
        r = InProcessRunner(target=sys.argv[1], function_name="spin_entry", timeout=0.3,
                            shm_size=4096, direct_lite=True)
        print(r.run_one(b"x"))
    """
    try:
        res = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script), str(so)],
            capture_output=True, text=True, timeout=HANG_TIMEOUT_S, env=_clean_env(),
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        res = None

    assert res is not None, "direct_lite hung on a C infinite loop"
    assert res.returncode == 0, res.stderr[-500:]
    assert res.stdout.strip() == "(-1, 'timeout')"


# ── exit() ─────────────────────────────────────────────────────────────


@requires_clang
def test_regression_exit_inside_guard_returns_status(tmp_path):
    r = _run_driver(
        _so(tmp_path),
        """
        print(call("exit_entry"), call("uexit_entry"), call("ok_entry"), flush=True)
        """,
    )

    assert r is not None and r.returncode == 0, r and (r.returncode, r.stderr[-500:])
    assert r.stdout.split() == ["3", "5", "3"]


@requires_clang
def test_exit_outside_guard_still_exits_and_flushes(tmp_path):
    """Adversarial: a one-shot target's own exit() must behave exactly as before."""
    exe = _build(
        tmp_path,
        """
        #include <stdlib.h>
        int main(void) {
            char a[8] = "AAAAAAA", b[8] = "BBBBBBB";
            volatile int r = memcmp(a, b, 8);
            exit(7 + (r == 0));
        }
        """,
        "-D__AFL_CMPLOG=1",
        "-fno-builtin-memcmp",
    )
    log = tmp_path / "cmplog.txt"
    r = subprocess.run([str(exe)], capture_output=True, env=_clean_env(_CMPLOG_OUT=str(log)))

    assert r.returncode == 7
    assert "4242424242424200" in log.read_text()  # destructors ran


@requires_clang
def test_regression_libc_error_helpers_return_through_guard(tmp_path):
    """errx()/error() reach exit() through libc's internal alias."""
    r = _run_driver(
        _so(tmp_path),
        'print(call("errx_entry"), call("error_entry"), call("warn_entry"), call("ok_entry"))',
    )

    assert r is not None and r.returncode == 0, r and (r.returncode, r.stderr[-500:])
    assert r.stdout.split() == ["4", "6", "3", "3"]
    assert "bad input" in r.stderr
    assert "missing x: No such file or directory" in r.stderr
    assert "just a warning" in r.stderr


@requires_clang
def test_nested_plain_guard_keeps_outer_budget(tmp_path):
    """Adversarial: a plain guarded call inside a timed one must not disable it."""
    r = _run_driver(
        _so(tmp_path),
        f"""
        gt = lib.__afl_guarded_call_timeout
        gt.restype = ctypes.c_int
        gt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
        print(gt(fn("nested_spin_entry"), b"x", 1, {TIMEOUT_US}), call("ok_entry"))
        """,
    )

    assert r is not None, "nested hang was not interrupted"
    assert r.stdout.split() == [str(GUARD_TIMEOUT_RC), "3"]


# ── Nesting and fork ───────────────────────────────────────────────────


@requires_clang
def test_regression_nested_guard_outer_crash_recovered(tmp_path):
    r = _run_driver(_so(tmp_path), 'print(call("nested_entry"), call("ok_entry"))')

    assert r is not None and r.returncode == 0, r and (r.returncode, r.stderr[-500:])
    assert r.stdout.split() == [str(-int(signal.SIGSEGV)), "3"]


@requires_clang
def test_regression_forked_child_keeps_its_crash(tmp_path):
    r = _run_driver(_so(tmp_path), 'print("host", call("fork_crash_entry"), flush=True)')

    assert r is not None and r.returncode == 0, r and r.stderr[-500:]
    lines = r.stdout.split("\n")
    assert [ln for ln in lines if ln] == [f"host {100 + int(signal.SIGSEGV)}"]


# ── File descriptors and ids ───────────────────────────────────────────


@requires_clang
def test_regression_cmplog_fd_reuse_not_written(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <fcntl.h>
        #include <stdio.h>
        #include <unistd.h>
        int main(int argc, char **argv) {
            (void)argc;
            char a[8] = "AAAAAAA", b[8] = "BBBBBBB";
            volatile int r = memcmp(a, b, 8);
            for (int fd = 3; fd < 1024; fd++) close(fd);
            int mine = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, 0644);
            dprintf(mine, "TARGET-DATA\\n");
            r += memcmp(a, "ZZZZZZZ", 8);
            return r == 0;
        }
        """,
        "-D__AFL_CMPLOG=1",
        "-fno-builtin-memcmp",
    )
    owned = tmp_path / "owned.txt"
    log = tmp_path / "cmplog.txt"
    subprocess.run([str(exe), str(owned)], check=True, env=_clean_env(_CMPLOG_OUT=str(log)))

    assert owned.read_text() == "TARGET-DATA\n"
    text = log.read_text()
    assert "4242424242424200" in text and "5a5a5a5a5a5a5a00" in text


@requires_clang
def test_cmplog_fd_is_close_on_exec(tmp_path):
    exe = _build(
        tmp_path,
        """
        #include <fcntl.h>
        #include <stdio.h>
        int main(void) {
            printf("%d\\n", __afl_cmplog_fd >= 0 && (fcntl(__afl_cmplog_fd, F_GETFD) & FD_CLOEXEC));
            return 0;
        }
        """,
        "-D__AFL_CMPLOG=1",
    )
    r = subprocess.run(
        [str(exe)], capture_output=True, text=True,
        env=_clean_env(_CMPLOG_OUT=str(tmp_path / "log.txt")),
    )  # fmt: skip

    assert r.stdout.strip() == "1"


_HEALTH_MAIN = """
#include <stdio.h>
int main(void) {
    uint64_t h[16] = {0};
    uint32_t n = __afl_shim_health(h, 16);
    for (uint32_t i = 0; i < n && i < 16; i++) printf("%llu ", (unsigned long long)h[i]);
    return 0;
}
"""


@requires_clang
def test_regression_garbage_aux_shm_id_refused(tmp_path):
    exe = _build(tmp_path, _HEALTH_MAIN, "-D__AFL_DISTANCE_MODE", "-fsanitize-coverage=trace-pc")
    shm = ShmCoverage(size=MAP_ENTRIES)
    table = DistanceTableShm({0x1000: 1.0})
    try:
        r = subprocess.run(
            [str(exe)], capture_output=True, text=True,
            env=_clean_env(
                __AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES),
                __AFL_DIST_SHM_ID=f"{table.env_id}junk",
            ),
        )  # fmt: skip
    finally:
        shm.cleanup()
        table.cleanup()

    fields = [int(x) for x in r.stdout.split()]
    assert fields[ShimField.ATTACHED] == 1
    assert fields[ShimField.SEG_REJECTED] == 1
