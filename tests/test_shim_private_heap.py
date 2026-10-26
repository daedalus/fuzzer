"""``-D__AFL_PRIVATE_HEAP=1``: the target's malloc family runs on a private heap.

A host (libc heap) dlopens a target .so built with the shim and calls it through
``__afl_guarded_call``. Heap corruption in the target must not reach the host.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang

AFL_SHIM = Path(__file__).resolve().parents[1] / "src/fuzzer_tool/adapters/afl_shim.c"
PH = "-D__AFL_PRIVATE_HEAP=1"
HOST_STRESS = 3000
BIG = 2 * 1024 * 1024

_HOST = r"""
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int (*entry_t)(const uint8_t *, size_t);
int main(int argc, char **argv) {
    (void)argc;
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { puts(dlerror()); return 90; }
    entry_t e = (entry_t)dlsym(h, argv[2]);
    int (*g)(entry_t, const uint8_t *, size_t) = dlsym(h, "__afl_guarded_call");
    int (*gt)(entry_t, const uint8_t *, size_t, uint64_t) = dlsym(h, "__afl_guarded_call_timeout");
    long (*live)(void) = dlsym(h, "__afl_ph_live");
    int n = atoi(argv[3]);
    uint64_t to = strtoull(argv[4], 0, 10);
    uint8_t in[4] = {1, 2, 3, 4};
    for (int i = 0; i < n; i++) {
        int rc = to ? gt(e, in, 4, to) : g(e, in, 4);
        printf("rc=%d\n", rc);
    }
    char *p[@STRESS@];
    for (int i = 0; i < @STRESS@; i++) { p[i] = malloc(1 + (i * 7) % 3000); memset(p[i], i, 1); }
    for (int i = 0; i < @STRESS@; i++) { if (p[i][0] != (char)i) return 91; free(p[i]); }
    char *b = malloc(1 << 20); memset(b, 1, 1 << 20); free(b);
    printf("host=OK live=%ld\n", live ? live() : -1L);
    return 0;
}
""".replace("@STRESS@", str(HOST_STRESS))

_TARGET = r"""
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <signal.h>

int t_overflow(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    char *p = malloc(2000);
    memset(p, 0x41, 2000 + 512);
    return 0;
}
int t_uaf_reuse(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    void **p = malloc(48), **q = malloc(48);
    free(q);
    free(p);
    *p = (void *)0x4141414141414141ULL;
    char *a = malloc(48), *b = malloc(48);
    memset(b, 1, 48);
    return (a && b && a != b) ? 0 : 1;
}
int t_dfree(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    void *p = malloc(48);
    free(p);
    free(p);
    return 0;
}
int t_dfree_big(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    void *p = malloc(@BIG@);
    free(p);
    free(p);
    return 0;
}
int t_bad_free(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    char *p = malloc(64);
    free(p + 8);
    return 0;
}
int t_leak_crash(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    for (int i = 0; i < 1000; i++) malloc(64 + i);
    raise(SIGABRT);
    return 0;
}
int t_spin(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    for (;;) free(malloc(100));
}
int t_calloc(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    char *a = malloc(256);
    memset(a, 0xAA, 256);
    free(a);
    char *c = calloc(16, 16);
    for (int i = 0; i < 256; i++) if (c[i]) return 1;
    if (calloc(SIZE_MAX / 2, 4)) return 2;
    return 0;
}
int t_realloc(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    char *p = realloc(NULL, 10);
    memcpy(p, "0123456789", 10);
    p = realloc(p, 5000);
    if (memcmp(p, "0123456789", 10)) return 1;
    p = realloc(p, 3);
    if (memcmp(p, "012", 3)) return 2;
    p = realloc(p, @BIG@);
    if (memcmp(p, "012", 3)) return 3;
    if (realloc(p, 0)) return 4;
    return 0;
}
int t_align(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    void *p = NULL;
    if (posix_memalign(&p, 64, 100) || ((uintptr_t)p & 63)) return 1;
    free(p);
    p = aligned_alloc(4096, 100);
    if (!p || ((uintptr_t)p & 4095)) return 2;
    free(p);
    p = memalign(256, 10);
    if (!p || ((uintptr_t)p & 255)) return 3;
    free(p);
    if (posix_memalign(&p, 3, 8) != EINVAL) return 4;
    return 0;
}
int t_foreign(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    char *s = strdup("libc-owned");
    char *t = realloc(s, 100);
    if (strcmp(t, "libc-owned")) return 1;
    free(t);
    free(NULL);
    return malloc_usable_size(malloc(100)) >= 100 ? 0 : 2;
}
/* A cached allocation kept in a global across calls; the first call aborts
 * after making it, so the guard frees it behind the target's back. */
static void *g_keep;
static int g_calls;
int t_stale_naive(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    free(g_keep);
    g_keep = malloc(64);
    if (++g_calls == 1) raise(SIGABRT);
    return 0;
}
#if __AFL_PRIVATE_HEAP
static uint32_t g_seen;
int t_stale_aware(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    uint32_t now = __afl_ph_abandon_count();
    if (now != g_seen) { g_seen = now; g_keep = NULL; }
    free(g_keep);
    g_keep = malloc(64);
    if (++g_calls == 1) raise(SIGABRT);
    return 0;
}
#endif
int t_zero(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    void *a = malloc(0), *b = malloc(0);
    return (a && b && a != b) ? 0 : 1;
}
""".replace("@BIG@", str(BIG))


def _cc(out: Path, args: list[str]) -> None:
    proc = subprocess.run(["clang", *args, "-o", str(out)], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"build failed: {proc.stderr[-1200:]}")


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    d = tmp_path_factory.mktemp("private_heap")
    (d / "host.c").write_text(_HOST)
    (d / "t.c").write_text(_TARGET)
    _cc(d / "host", ["-O1", str(d / "host.c"), "-ldl"])
    base = [
        "-O0",
        "-fno-builtin",
        "-shared",
        "-fPIC",
        "-Wl,-Bsymbolic",
        "-include",
        str(AFL_SHIM),
        str(d / "t.c"),
    ]
    _cc(d / "priv.so", [*base, PH])
    _cc(d / "plain.so", base)
    return d


def _run(build, so, entry, n=1, timeout_us=0):
    proc = subprocess.run(
        [str(build / "host"), str(build / so), entry, str(n), str(timeout_us)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    rcs = [int(x[3:]) for x in proc.stdout.split() if x.startswith("rc=")]
    return proc.returncode, rcs, proc.stdout


@requires_clang
def test_overflow_stays_off_the_host_heap(build):
    rc, rcs, out = _run(build, "priv.so", "t_overflow")
    assert rc == 0 and rcs == [0] and "host=OK" in out


@requires_clang
def test_control_overflow_kills_host_without_private_heap(build):
    rc, _, out = _run(build, "plain.so", "t_overflow")
    assert rc != 0 and "host=OK" not in out


@requires_clang
def test_uaf_write_cannot_poison_allocator(build):
    rc, rcs, out = _run(build, "priv.so", "t_uaf_reuse")
    assert rc == 0 and rcs == [0] and "host=OK" in out


@requires_clang
def test_control_uaf_write_poisons_libc_tcache(build):
    _, rcs, out = _run(build, "plain.so", "t_uaf_reuse")
    assert rcs != [0] or "host=OK" not in out


@requires_clang
@pytest.mark.parametrize("entry", ["t_dfree", "t_dfree_big", "t_bad_free"])
def test_invalid_frees_are_contained_crashes(build, entry):
    rc, rcs, out = _run(build, "priv.so", entry)
    assert rc == 0 and len(rcs) == 1 and rcs[0] < 0 and "host=OK" in out


@requires_clang
def test_double_free_reports_sigabrt(build):
    _, rcs, _ = _run(build, "priv.so", "t_dfree")
    assert rcs == [-6]


@requires_clang
@pytest.mark.parametrize("entry", ["t_calloc", "t_realloc", "t_align", "t_foreign", "t_zero"])
def test_allocator_semantics(build, entry):
    rc, rcs, out = _run(build, "priv.so", entry)
    assert rc == 0 and rcs == [0], out


@requires_clang
def test_crash_reclaims_the_call_allocations(build):
    _, _, one = _run(build, "priv.so", "t_leak_crash", n=1)
    _, rcs, many = _run(build, "priv.so", "t_leak_crash", n=30)
    live = lambda s: int(s.split("live=")[1])  # noqa: E731
    assert live(one) >= 0
    assert rcs == [-6] * 30
    assert live(many) == live(one)


@requires_clang
def test_timeout_never_leaves_the_heap_locked(build):
    rc, rcs, out = _run(build, "priv.so", "t_spin", n=20, timeout_us=20000)
    assert rc == 0 and rcs == [-1] * 20 and "host=OK" in out


@requires_clang
def test_stale_global_after_abort_poisons_every_later_call(build):
    """Control: a pointer cached in a global dangles once its call is abandoned."""
    _, rcs, _ = _run(build, "priv.so", "t_stale_naive", n=5)
    assert rcs == [-6] * 5


@requires_clang
def test_abandon_count_lets_the_target_drop_stale_state(build):
    rc, rcs, out = _run(build, "priv.so", "t_stale_aware", n=5)
    assert rc == 0 and rcs == [-6, 0, 0, 0, 0] and "host=OK" in out, out
