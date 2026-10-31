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


# ── ctx cache: last return address -> ctx, main thread only ────────────
# 91% of consecutive ffmpeg_read edges share the return address the walk
# reads, and ctx is a pure function of it. A sentinel planted in the cache
# shows who reads it: the same call site must, another site or thread not.
SENTINEL = 0xDEADBEEF

# common() is reached from main and from a second thread, so the return
# address the walk reads (into common) is the same in both.
_CACHE_DRIVER = """
#include <pthread.h>
#include <stdio.h>

/* NT: no tail call, so each frame really sits under the one above it. */
#define NT(x) ({ uint32_t r_ = (x); __asm__ volatile("" ::: "memory"); r_; })
__attribute__((noinline)) static uint32_t probe(void) { return __afl_get_caller_ctx(); }
__attribute__((noinline)) static uint32_t site(void) { return NT(probe()); }
__attribute__((noinline)) static uint32_t other(void) { return NT(probe()); }
__attribute__((noinline)) static uint32_t common(int which) { return NT(which ? other() : site()); }

static void *thread_fn(void *arg) {
    printf("thread %u\\n", common(0));
    return arg;
}

int main(void) {
    common(0);  /* first walk resolves the main bounds via TLS: no cache yet */
    uint32_t real = common(0);
    __afl_ctx_last_ctx = %SENTINEL%u;
    printf("same %u %u\\n", real, common(0));
    printf("other %u\\n", common(1));
    common(0);  /* cache the address the thread will read */
    __afl_ctx_last_ctx = %SENTINEL%u;
    pthread_t t;
    pthread_create(&t, NULL, thread_fn, NULL);
    pthread_join(t, NULL);
    return 0;
}
""".replace("%SENTINEL%", str(SENTINEL))


@pytest.fixture(scope="module")
def cache_exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("cache"), _CACHE_DRIVER, CTX_FULL)


def _cache_out(exe) -> dict[str, list[int]]:
    return {f[0]: list(map(int, f[1:])) for f in (line.split() for line in _run(exe) if line)}


@requires_clang
def test_regression_ctx_cache_hit_returns_cached(cache_exe):
    """Falsification: the same return address returns the cached ctx, not a fresh hash."""
    real, again = _cache_out(cache_exe)["same"]

    assert real not in (0, SENTINEL)
    assert again == SENTINEL


@requires_clang
def test_ctx_cache_miss_and_thread_recompute(cache_exe):
    """Adversarial: another site misses; another thread never reads main's cache."""
    out = _cache_out(cache_exe)
    real = out["same"][0]

    assert out["other"][0] not in (0, SENTINEL)
    assert out["thread"][0] == real


# cfp-keyed hit: a frame validated on the main stack is not re-checked
# against the bounds. Zeroing main_hi after caching sends the old walk down
# the TLS path (fresh hash); the shortcut never reads it (sentinel).
_CFP_DRIVER = (
    _CACHE_DRIVER.split("int main(void)")[0]
    + """
int main(void) {
    common(0);
    common(0);
    __afl_ctx_last_ctx = %SENTINEL%u;
    uintptr_t hi = __afl_ctx_main_hi;
    __afl_ctx_main_hi = 0;
    uint32_t v = common(0);
    __afl_ctx_main_hi = hi;
    printf("hit %u\\n", v);
    return 0;
}
""".replace("%SENTINEL%", str(SENTINEL))
)


@pytest.fixture(scope="module")
def cfp_exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("cfp"), _CFP_DRIVER, CTX_FULL)


@requires_clang
def test_ctx_cache_cfp_hit_skips_bounds(cfp_exe):
    """Same validated frame and return address: cached ctx without the bound loads."""
    assert _cache_out(cfp_exe)["hit"] == [SENTINEL]


# ── fused guard flag: one check selects the specialized fast path ──────
# argv[1] == "slow" forces the generic path after attach, so one binary
# compares both on the same edge sequence (ASLR off: ctx hashes addresses).
_FAST_DRIVER = """
#include <stdio.h>
#include <string.h>
#define NT(x) ({ uint32_t r_ = (x); __asm__ volatile("" ::: "memory"); r_; })
__attribute__((noinline)) static uint32_t edge(uint32_t *g) {
    __sanitizer_cov_trace_pc_guard(g);
    return NT(*g);
}
__attribute__((noinline)) static uint32_t a(uint32_t *g) { return NT(edge(g)); }
__attribute__((noinline)) static uint32_t b(uint32_t *g) { return NT(edge(g)); }

int main(int argc, char **argv) {
    printf("fast %d\\n", __afl_fast);
    if (argc > 1 && strcmp(argv[1], "slow") == 0) __afl_fast = 0;
    uint32_t g[4] = {11, 22, 33, 44};
    for (int i = 0; i < 4; i++) { a(&g[i]); b(&g[i]); a(&g[(i + 1) % 4]); }
    printf("path %llu\\n", (unsigned long long)__afl_path_hash_acc);
    if (!__afl_area) return 0;
    for (uint32_t i = 0; i < __afl_map_size; i++)
        if (__afl_area[i].edge_id) printf("slot %u %u %u\\n", i, __afl_area[i].edge_id,
                                          __afl_area[i].count & 0xFFFFFF);
    return 0;
}
"""


@pytest.fixture(scope="module")
def fast_exe(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("fast"), _FAST_DRIVER, CTX_FULL)


def _fast_run(exe, *args: str, attach: bool = True, **env: str) -> list[str]:
    cmd = ["setarch", "-R", str(exe), *args]
    if not attach:
        r = subprocess.run(cmd, env=_clean_env(**env), capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()
    shm = ShmCoverage(size=MAP_ENTRIES)
    try:
        shm.reset_edge_map()
        e = _clean_env(__AFL_SHM_ID=shm.env_id, AFL_MAP_SIZE=str(MAP_ENTRIES), **env)
        r = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()
    finally:
        shm.cleanup()


@requires_clang
def test_regression_fast_path_matches_generic(fast_exe):
    """Falsification: default attach takes the fast path, and it records what the generic one does."""
    fast = _fast_run(fast_exe)
    slow = _fast_run(fast_exe, "slow")

    assert fast[0] == "fast 1"
    assert fast[1:] == slow[1:]
    assert any(line.startswith("slot") for line in fast)


@requires_clang
@pytest.mark.parametrize(
    ("env", "attach"),
    [
        ({}, False),
        ({"__AFL_WGUARD": "mprotect"}, True),
        ({"FUZZER_KEEP_ASLR": "1"}, True),
        ({"__AFL_CTX_SENSITIVE": "0"}, True),
    ],
)
def test_fast_path_off_when_a_setting_needs_checks(fast_exe, env, attach):
    """Adversarial: no map, write guard, ASLR-relative ctx or ctx off keep the generic path."""
    assert _fast_run(fast_exe, attach=attach, **env)[0] == "fast 0"


@requires_clang
def test_fast_path_off_with_distance_table(fast_exe):
    """Adversarial: an attached distance table needs the generic path's probe."""
    table = DistanceTableShm(DIST_TABLE)
    try:
        out = _fast_run(fast_exe, __AFL_DIST_SHM_ID=table.env_id)
    finally:
        table.cleanup()

    assert out[0] == "fast 0"


# ── guard NULL test: the compiler never passes one ─────────────────────
_ZERO_DRIVER = """
#include <stdio.h>
int main(void) {
    uint32_t off = 0, on = 7;
    __sanitizer_cov_trace_pc_guard(&off);
    unsigned before = 0;
    for (uint32_t i = 0; i < __afl_map_size; i++) before += __afl_area[i].edge_id != 0;
    __sanitizer_cov_trace_pc_guard(&on);
    unsigned after = 0;
    for (uint32_t i = 0; i < __afl_map_size; i++) after += __afl_area[i].edge_id != 0;
    printf("%u %u\\n", before, after);
    return 0;
}
"""


def test_regression_guard_null_test_dropped():
    """-fsanitize-coverage=trace-pc-guard always passes a guard pointer."""
    from tests.test_regression_shim_audit import SHIM

    assert "if (!guard || *guard == 0) return;" not in SHIM.read_text()


@requires_clang
def test_disabled_guard_records_nothing(tmp_path):
    """Adversarial: *guard == 0 (a guard the runtime switched off) still returns first."""
    exe = _build(tmp_path, _ZERO_DRIVER)
    assert _run(exe)[0].split() == ["0", "1"]
