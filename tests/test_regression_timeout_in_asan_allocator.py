"""Regression: a guard timeout inside ASAN's allocator leaked its mutex.

ASAN's size-class allocator holds a region mutex while it maps, refills and
releases memory (``ReleaseFreeMemoryToOS`` -> ``internal_madvise``). The
timeout tick deferred only while pc sat in glibc, so a tick there jumped out
with the mutex held, and the next allocation of that size class waited
forever. Measured: ffmpeg_read_9.0.2_asan.so in-process at ``-t 0.002``,
3 of the first 5 timeout jumps had pc in libasan.so; the host then hung in
``GetFromAllocator`` (writer bit set, no owner).

Fix: the tick defers while pc is in the sanitizer runtime, as it does in libc.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from fuzzer_tool.cli.ldpreload_wrapper import _resolve_asan
from tests.conftest import requires_clang

SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
SPIN_CALLS = 200  # 50 hung 6 of 6 runs before the fix
TIMEOUT_US = 2000
HANG_S = 60  # a leaked mutex never returns; anything past this is one
GUARD_TIMEOUT_RC = -1

# churn() fills one size class and frees it: refills and releases to the OS
# both run under that class's region mutex.
_SRC = """
#include <stdint.h>
#include <stdlib.h>
enum { N = 4096 };
static void *p[N];
static void churn(void) {
    for (int i = 0; i < N; i++) p[i] = malloc(1312);
    for (int i = 0; i < N; i++) free(p[i]);
}
int t_churn_forever(const uint8_t *d, size_t n) { (void)d; (void)n; for (;;) churn(); }
int t_churn_once(const uint8_t *d, size_t n) { (void)d; (void)n; churn(); return 0; }
int t_spin(const uint8_t *d, size_t n) { (void)d; volatile size_t x = n; for (;;) x++; }
"""

# Timed calls of argv[2], then one untimed t_churn_once: it hangs on a leak.
_HOST = """
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef int (*entry_t)(const uint8_t *, size_t);
int main(int argc, char **argv) {
    (void)argc;
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { puts(dlerror()); return 90; }
    entry_t hang = (entry_t)dlsym(h, argv[2]), once = (entry_t)dlsym(h, "t_churn_once");
    int (*gt)(entry_t, const uint8_t *, size_t, uint64_t) = dlsym(h, "__afl_guarded_call_timeout");
    int n = atoi(argv[3]), timeouts = 0;
    for (int i = 0; i < n; i++)
        timeouts += gt(hang, (const uint8_t *)"x", 1, (uint64_t)atoll(argv[4])) == -1;
    printf("timeouts=%d once=%d\\n", timeouts, gt(once, (const uint8_t *)"x", 1, 0));
    return 0;
}
"""


def _cc(args: list[str]) -> None:
    r = subprocess.run(["clang", *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1200:]


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    d = tmp_path_factory.mktemp("asan_alloc")
    (d / "t.c").write_text(textwrap.dedent(_SRC))
    (d / "host.c").write_text(textwrap.dedent(_HOST))
    _cc(["-O1", str(d / "host.c"), "-o", str(d / "host"), "-ldl"])
    _cc([
        "-O0", "-fno-builtin", "-fsanitize=address", "-shared", "-fPIC", "-Wl,-Bsymbolic",
        "-include", str(SHIM), str(d / "t.c"), "-o", str(d / "t.so"), "-lasan", "-ldl",
    ])  # fmt: skip
    return d


def _run(build: Path, entry: str) -> subprocess.CompletedProcess | None:
    """*SPIN_CALLS* timed calls of *entry*, then one untimed churn; None on a hang."""
    rt = _resolve_asan()
    if rt is None:
        pytest.skip("libasan not installed")
    drop = ("__AFL_", "AFL_", "ASAN_OPTIONS", "LD_PRELOAD", "FUZZER_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env["LD_PRELOAD"] = str(rt)
    env["ASAN_OPTIONS"] = "detect_leaks=0:verify_asan_link_order=0"
    cmd = [str(build / "host"), str(build / "t.so"), entry, str(SPIN_CALLS), str(TIMEOUT_US)]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=HANG_S, env=env)
    except subprocess.TimeoutExpired:
        return None


@requires_clang
def test_regression_timeout_in_asan_allocator(build):
    r = _run(build, "t_churn_forever")

    assert r is not None, "a timeout inside ASAN's allocator leaked its region mutex"
    assert f"timeouts={SPIN_CALLS} once=0" in r.stdout, r.stderr[-500:]


@requires_clang
def test_timeout_outside_asan_still_fires(build):
    # Falsification: the deferral must not stop a hang in target code.
    r = _run(build, "t_spin")

    assert r is not None
    assert f"timeouts={SPIN_CALLS} once=0" in r.stdout, r.stderr[-500:]
