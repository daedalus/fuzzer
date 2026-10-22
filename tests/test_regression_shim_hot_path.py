"""Per-edge path cuts in afl_shim.c (ffmpeg_read: 182 -> 138 Ir per guard call).

Two cuts change state a test can see:

* distance: the guard callback reached ``__afl_pc_key`` (dladdr base, return
  address) on every edge before ``__afl_probe_distance`` found no table.
  ``__afl_base`` stays 0 now unless a table is attached.
* TLS: the caller-ctx walk read its stack bounds through ``__tls_get_addr``
  on every edge (a ``.so`` cannot use initial-exec TLS: the shim's TLS block
  exceeds glibc's dlopen surplus). The main thread's bounds now sit in plain
  globals; other threads keep the TLS path.

The home-slot-first, COMPCOV-gate and out-of-line claim cuts are pure
refactors, covered by test_shim_map_slot.py, test_compcov.py and
test_regression_shm_stale_reclaim.py.
"""

from __future__ import annotations

import subprocess

import pytest

from fuzzer_tool.adapters.shm import DistanceTableShm, ShmCoverage
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

MAP_ENTRIES = 1024
#: Full-width ctx: a walk that reached the return address is nonzero
#: except with probability 2^-32, so 0 means the walk bailed.
CTX_FULL = "-D__AFL_CTX_BITS=32"
#: Any key: the cut only asks whether a table is attached.
DIST_TABLE = {0x1234: 1.0}

_DIST_DRIVER = """
#include <stdio.h>
int main(void) {
    uint32_t g = 1;
    __sanitizer_cov_trace_pc_guard(&g);
    printf("%llu\\n", (unsigned long long)__afl_base);
    return 0;
}
"""

# One call site; the walk from probe() reads run()'s frame either way.
#   c[0] TLS path (main bounds unset, the resolve publishes them)
#   c[1] main-bounds path
#   c[2] TLS path again (main bounds cleared just before, restored after)
_CTX_DRIVER = """
#include <pthread.h>
#include <stdio.h>

__attribute__((noinline)) static uint32_t probe(void) { return __afl_get_caller_ctx(); }

static void bounds(uintptr_t *lo, uintptr_t *hi) {
    pthread_attr_t a; void *p; size_t sz;
    pthread_getattr_np(pthread_self(), &a);
    pthread_attr_getstack(&a, &p, &sz);
    pthread_attr_destroy(&a);
    *lo = (uintptr_t)p; *hi = (uintptr_t)p + sz;
}

static void *thread_fn(void *arg) {
    uint32_t c = probe();
    printf("thread %u %lu\\n", c, (unsigned long)__afl_ctx_stack_hi);
    return arg;
}

__attribute__((noinline)) static void run(void) {
    uintptr_t lo, hi, main_hi = 0;
    bounds(&lo, &hi);
    uint32_t c[3];
    int published = 0;
    for (int i = 0; i < 3; i++) {
        if (i == 2) __afl_ctx_main_lo = __afl_ctx_main_hi = 0;
        c[i] = probe();
        if (i == 0) {
            published = __afl_ctx_main_lo == lo && __afl_ctx_main_hi == hi;
            main_hi = __afl_ctx_main_hi;
        }
    }
    printf("main %u %u %u %d\\n", c[0], c[1], c[2], published);

    __afl_ctx_main_hi = main_hi;
    __afl_ctx_main_lo = lo;
    pthread_t t;
    pthread_create(&t, NULL, thread_fn, NULL);
    pthread_join(t, NULL);
    printf("kept %d %lu\\n", __afl_ctx_main_hi == main_hi, (unsigned long)main_hi);
}

int main(void) { run(); return 0; }
"""


def _run(exe, **env: str) -> list[str]:
    """Run *exe* with an attached edge map; return stdout lines."""
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        e = _clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES), **env)
        r = subprocess.run([str(exe)], env=e, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        return r.stdout.split("\n")
    finally:
        shm.cleanup()


@pytest.fixture(scope="module")
def dist_exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("dist"), _DIST_DRIVER)


@pytest.fixture(scope="module")
def ctx_exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("ctx"), _CTX_DRIVER, CTX_FULL)


@requires_clang
def test_regression_dist_skips_pc_key_without_table(dist_exe):
    """Falsification: no distance table, so the guard never resolves the object base."""
    assert _run(dist_exe)[0] == "0"


@requires_clang
def test_dist_table_still_probed(dist_exe):
    """Adversarial: with a table attached the guard still computes the PC key."""
    table = DistanceTableShm(DIST_TABLE)
    try:
        out = _run(dist_exe, __AFL_DIST_SHM_ID=table.env_id)
    finally:
        table.cleanup()

    assert int(out[0]) != 0


@requires_clang
def test_regression_main_bounds_match_tls_path(ctx_exe):
    """Falsification: main-thread bounds are published and give the TLS path's ctx."""
    fields = next(line for line in _run(ctx_exe) if line.startswith("main")).split()
    c0, c1, c2, published = map(int, fields[1:])

    assert published == 1
    assert c0 == c1 == c2
    assert c0 != 0


@requires_clang
def test_other_thread_keeps_tls_path(ctx_exe):
    """Adversarial: a second thread resolves its own bounds and leaves main's alone."""
    out = _run(ctx_exe)
    _, ctx, thread_hi = next(line for line in out if line.startswith("thread")).split()
    _, kept, main_hi = next(line for line in out if line.startswith("kept")).split()

    assert int(ctx) != 0
    assert int(thread_hi) > 1
    assert int(thread_hi) != int(main_hi)
    assert kept == "1"
