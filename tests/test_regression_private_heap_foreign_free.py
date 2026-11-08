"""Regression: a bad free of a host chunk must not leave glibc's arena lock held.

The private heap forwards pointers it does not own to glibc. For a chunk in
the main brk heap glibc takes the arena lock and only then checks the
neighbours, so ``double free or corruption (!prev)`` aborted with the lock
held; the guard's siglongjmp left it that way and, with a second thread in
the process, the host's next large malloc waited forever (direct_lite,
fuzzgoat). The heap now makes the same checks without the lock and raises the
SIGABRT itself. Valid foreign frees and reallocs must still pass.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

FLAGS = ("-fsanitize-coverage=trace-pc-guard", "-D__AFL_PRIVATE_HEAP=1")

_TARGET = """
#include <stdint.h>
#include <stdlib.h>
void *volatile gp;
volatile size_t gn;
int entry(const uint8_t *d, size_t n) {
    (void)n;
    switch (d[0]) {
    case 'd': free(gp); free(gp); break;                 /* double free */
    case 'v': free(gp); break;                           /* valid foreign free */
    case 'r': { void *q = realloc(gp, gn * 2); free(q); break; }   /* valid realloc */
    case 'w': free((char *)gp + 16); break;              /* aligned, mid-chunk */
    case 'R': free(gp); gp = realloc(gp, 64); break;     /* realloc after free */
    }
    return 0;
}
"""

# A second thread makes glibc lock its arenas (single-threaded it skips them,
# which hides the bug). The host mallocs after the guarded call: it hangs when
# the lock is left held.
_HOST = """
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
typedef int (*entry_t)(const uint8_t *, size_t);
typedef int (*guard_t)(entry_t, const uint8_t *, size_t);
static void *idle(void *a) { (void)a; for (;;) sleep(100); return 0; }
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s\\n", dlerror()); return 2; }
    guard_t g = (guard_t)dlsym(h, "__afl_guarded_call");
    entry_t e = (entry_t)dlsym(h, "entry");
    void **gp = (void **)dlsym(h, "gp");
    size_t *gn = (size_t *)dlsym(h, "gn");
    size_t size = strtoul(argv[3], 0, 10);
    pthread_t th;
    pthread_create(&th, 0, idle, 0);
    char *pad = malloc(4096); (void)pad;
    char *blk = calloc(1, size);
    char *pad2 = strcmp(argv[4], "top") ? malloc(64) : NULL; (void)pad2;
    *gp = blk; *gn = size;
    int rc = g(e, (const uint8_t *)argv[2], 1);
    void *big = malloc(300000);
    printf("rc %d alive %d\\n", rc, big != NULL);
    return 0;
}
"""


@pytest.fixture(scope="module")
def rig(tmp_path_factory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("foreign_free")
    so = _build(tmp, _TARGET, *FLAGS, shared=True)
    host_c = tmp / "host.c"
    host_c.write_text(_HOST)
    host = tmp / "host"
    r = subprocess.run(
        ["clang", "-O1", str(host_c), "-o", str(host), "-ldl", "-lpthread"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return host, so


def _go(rig, mode: str, size: int = 4096, where: str = "mid"):
    host, so = rig
    # A hang is the bug: TimeoutExpired fails the test.
    return subprocess.run(
        [str(host), str(so), mode, str(size), where],
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )


@requires_clang
class TestForeignFree:
    @pytest.mark.parametrize("size", [24, 100, 1000, 4096, 70000, 200000])
    def test_control_valid_foreign_free_and_realloc_pass(self, rig, size):
        for mode in ("v", "r"):
            r = _go(rig, mode, size)
            assert "rc 0 alive 1" in r.stdout, (mode, size, r.stdout, r.stderr)

    def test_control_valid_free_next_to_top(self, rig):
        r = _go(rig, "v", 4096, "top")
        assert "rc 0 alive 1" in r.stdout, r.stderr

    def test_regression_double_free_leaves_host_alive(self, rig):
        r = _go(rig, "d", 4096)
        assert "rc -6 alive 1" in r.stdout, (r.stdout, r.stderr)
        assert "private heap guard" in r.stderr

    def test_regression_double_free_next_to_top(self, rig):
        r = _go(rig, "d", 4096, "top")
        assert "rc -6 alive 1" in r.stdout, (r.stdout, r.stderr)

    def test_adversarial_mid_chunk_pointer(self, rig):
        r = _go(rig, "w", 4096)
        assert "rc -6 alive 1" in r.stdout, (r.stdout, r.stderr)

    def test_adversarial_realloc_after_free(self, rig):
        r = _go(rig, "R", 4096)
        assert "rc -6 alive 1" in r.stdout, (r.stdout, r.stderr)
