"""Regression: forkserver children re-resolved the ctx stack bound per exec.

__afl_ctx_stack_hi is resolved lazily on the first ctx edge. The forkserver
parent detaches __afl_area before its fork loop, so it never reaches the ctx
walk, the TLS slot stays 0, and every forked child paid pthread_getattr_np
(a /proc/self/maps parse, ~114k instructions under callgrind on fuzzgoat).

The target TU is built without coverage instrumentation, so main() can read
the slot before any edge fires.
"""

import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.adapters.shm import ShmCoverage
from tests.conftest import requires_clang
from tests.test_regression_forkserver_shm import SHIM

# Prints the slot as seen at main() entry, then the bound computed
# independently with pthread_getattr_np.
_TARGET = r"""
#include <pthread.h>
#include <stdio.h>

int main(void) {
    unsigned long seen = (unsigned long)__afl_ctx_stack_hi;
    unsigned long hi = 0;
    pthread_attr_t a;
    if (pthread_getattr_np(pthread_self(), &a) == 0) {
        void *lo = NULL;
        size_t sz = 0;
        if (pthread_attr_getstack(&a, &lo, &sz) == 0 && sz)
            hi = (unsigned long)lo + sz;
        pthread_attr_destroy(&a);
    }
    fprintf(stderr, "HI %lx %lx\n", seen, hi);
    __afl_map_edge(1);
    return 0;
}
"""


class Ctx:
    """__AFL_CTX_SENSITIVE values."""

    ON = "1"
    OFF = "0"


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
    d = tmp_path_factory.mktemp("fsrv_hi")
    src = d / "t.c"
    src.write_text(_TARGET)
    exe = d / "t"
    r = subprocess.run(
        [
            "clang",
            "-O0",
            "-fno-omit-frame-pointer",
            "-include",
            SHIM,
            "-o",
            str(exe),
            str(src),
            "-lpthread",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return str(exe)


def _run_child(target: str, ctx: str) -> tuple[int, int]:
    """One forkserver exec; returns (slot at main entry, independent bound)."""
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    shm = ShmCoverage(size=8192)
    r = ForkserverRunner(
        target,
        timeout=2.0,
        env={"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.size), "__AFL_CTX_SENSITIVE": ctx},
    )
    try:
        if not r.start():
            pytest.skip("forkserver failed to start")
        rc, stderr = r.run_one(b"x")
    finally:
        r.stop()
        shm.cleanup()

    assert rc == 0, stderr
    line = next(ln for ln in stderr.splitlines() if ln.startswith("HI "))
    _, seen, hi = line.split()
    return int(seen, 16), int(hi, 16)


@requires_clang
def test_regression_forkserver_child_inherits_stack_hi(target):
    """Falsification: with ctx on, the child starts with the parent's bound."""
    seen, hi = _run_child(target, Ctx.ON)
    assert hi > 1
    assert seen == hi


@requires_clang
def test_regression_forkserver_no_prime_when_ctx_off(target):
    """Adversarial: ctx off never walks, so nothing is resolved for it."""
    seen, _ = _run_child(target, Ctx.OFF)
    assert seen == 0
