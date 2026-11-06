"""Regression: a guard timeout inside the cmplog record writer froze cmplog.

The timer tick siglongjmps out of a hung guarded call. Landing mid-record
(ffmpeg: ~400k records per run, so the odds are real) left
``__afl_cmplog_lock`` held: every later record in the process was dropped
and counted ``CMPLOG_REENTRANT``, and the pair pool stopped growing. The
crash handler released the lock; the timeout path did not.

Fix: the guard's landing path releases a lock this thread still holds.

On ffmpeg_read_9.0.2_asan.so (clang -O2) the tick landed between the lock's
atomic exchange and the thread-local ownership store (a __tls_get_addr call
apart), so ownership read "not mine" and the lock stuck anyway, counted
CMPLOG_CONTENDED. The lock word now names its owner: the CAS is the only step.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

from tests.conftest import requires_clang

SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
RUN_TIMEOUT_S = 60
TIMEOUT_US = 50_000
MARK = "5a5a5a5a5a5a5a00"  # hex of "ZZZZZZZ\0", the post-timeout record

# Entry functions hang under the guard; main() then logs one record.
#   held    entry owns the writer lock when the tick lands (the bug)
#   taken   tick lands right after the lock's atomic step, before anything
#           else acquire does (the window measured on ffmpeg)
#   free    entry hangs without it (control)
#   other   another thread owns the lock throughout; it must stay owned
_SRC = """
#include <pthread.h>
#include <string.h>
static int hang_held(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    __afl_cmplog_acquire();
    for (volatile int v = 0;; v++) {}
    return 0;
}
static int hang_taken(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    __afl_cmplog_take();
    for (volatile int v = 0;; v++) {}
    return 0;
}
static int hang_free(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    for (volatile int v = 0;; v++) {}
    return 0;
}
static void *grab(void *arg) { (void)arg; __afl_cmplog_acquire(); return NULL; }

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "held";
    if (!strcmp(mode, "other")) {
        pthread_t t;
        pthread_create(&t, NULL, grab, NULL);
        pthread_join(t, NULL);
    }
    int (*entry)(const uint8_t *, size_t) = strcmp(mode, "free") ? hang_held : hang_free;
    if (!strcmp(mode, "other")) entry = hang_free;
    if (!strcmp(mode, "taken")) entry = hang_taken;
    int rc = __afl_guarded_call_timeout(entry, (const uint8_t *)"x", 1, TIMEOUT_US);

    char a[8] = "AAAAAAA", z[8] = "ZZZZZZZ";
    volatile int r = memcmp(a, z, 8);
    __afl_cmplog_flush();
    return (rc == -1 && r != 0) ? (__afl_cmplog_lock ? 3 : 0) : 1;
}
"""


def _build(tmp: Path) -> Path:
    c = tmp / "t.c"
    c.write_text(textwrap.dedent(_SRC))
    exe = tmp / "t"
    cmd = [
        "clang", "-O1", "-g", "-D__AFL_CMPLOG=1", "-fno-builtin-memcmp",
        f"-DTIMEOUT_US={TIMEOUT_US}", "-include", str(SHIM), str(c),
        "-o", str(exe), "-ldl", "-lpthread", "-lrt",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def _run(exe: Path, tmp: Path, mode: str) -> tuple[int, str]:
    log = tmp / f"cmplog_{mode}.txt"
    drop = ("__AFL_", "AFL_", "_CMPLOG", "ASAN_OPTIONS", "LD_PRELOAD", "FUZZER_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env["_CMPLOG_OUT"] = str(log)
    r = subprocess.run([str(exe), mode], env=env, capture_output=True, timeout=RUN_TIMEOUT_S)
    return r.returncode, log.read_text() if log.exists() else ""


@requires_clang
def test_regression_cmplog_lock_timeout(tmp_path):
    exe = _build(tmp_path)

    rc, log = _run(exe, tmp_path, "held")

    assert rc == 0, f"lock still held after timeout (rc={rc})"
    assert MARK in log


@requires_clang
def test_regression_cmplog_lock_timeout_mid_acquire(tmp_path):
    exe = _build(tmp_path)

    rc, log = _run(exe, tmp_path, "taken")

    assert rc == 0, f"lock stuck when the tick split acquire (rc={rc})"
    assert MARK in log


@requires_clang
def test_timeout_without_lock_keeps_records(tmp_path):
    # Falsification: the control must pass on old and new code alike.
    exe = _build(tmp_path)

    rc, log = _run(exe, tmp_path, "free")

    assert rc == 0
    assert MARK in log


@requires_clang
def test_timeout_keeps_other_threads_lock(tmp_path):
    # Adversarial: a lock owned by another thread is not this landing's to free.
    exe = _build(tmp_path)

    rc, log = _run(exe, tmp_path, "other")

    assert rc == 3, "landing path released a lock it did not own"
    assert MARK not in log
