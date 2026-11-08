"""Regression: hook mute, single evaluation and the malloc-hook table (ff01c12f review).

* A siglongjmp out of a forwarded libc compare skipped ``__AFL_MUTED``'s
  restore, leaving that thread's sanitizer weak hooks muted for good.
* ``__AFL_MUTED`` cleared the flag instead of restoring it, so a nested
  forwarded call unmuted its caller.
* ``__AFL_CMP_COUNT_AT`` evaluated ``id``/``satisfied`` twice, or not at all.
* ``__sanitizer_install_malloc_and_free_hooks`` returned 0 (table full) unnoticed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env
from tests.test_sanitizer_death_callback import requires_asan_rt

CMPLOG = ("-fsanitize-coverage=trace-pc-guard", "-D__AFL_CMPLOG=1")
TRACE = ("-fsanitize-coverage=trace-pc-guard",)

# call 1 faults inside the real memcmp (ASAN's range check passes for
# address 16); call 2 compares through the interceptor, i.e. the weak hook.
_ESCAPE = """
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int __interceptor_memcmp(const void *, const void *, size_t);
int __afl_guarded_call(int (*)(const uint8_t *, size_t), const uint8_t *, size_t);
volatile int sink;
static int crash_entry(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    sink = memcmp((const void *)16, "MAGICTOKEN", 10);
    return 0;
}
static int cmp_entry(const uint8_t *d, size_t n) {
    char *in = malloc(16);
    memcpy(in, d, n < 15 ? n : 15);
    in[15] = 0;
    sink = __interceptor_memcmp(in, "MAGICTOKEN", 10);
    free(in);
    return 0;
}
int main(int argc, char **argv) {
    int r1 = argc > 1 ? 0 : __afl_guarded_call(crash_entry, (const uint8_t *)"XAGICTOKEN", 10);
    int r2 = __afl_guarded_call(cmp_entry, (const uint8_t *)"XAGICTOKEN", 10);
    printf("rc %d %d\\n", r1, r2);
    return 0;
}
"""

_NESTED = """
int main(void) {
    int inner = -1;
    int outer = __AFL_MUTED(({ int a = __AFL_MUTED(0); inner = __afl_hook_mute; a; }));
    (void)outer;
    return (inner == 1 && __afl_hook_mute == 0) ? 0 : 1;
}
"""

_ONCE = """
static int n;
static int bump(void) { return ++n; }
int main(void) {
    __AFL_CMP_COUNT_AT(bump() * 0, bump() > 0, (bump(), (void *)0));
    return n == 3 ? 0 : 10 + n;
}
"""

_FULL = """
#include <stddef.h>
extern int __sanitizer_install_malloc_and_free_hooks(void (*)(const volatile void *, size_t),
                                                     void (*)(const volatile void *));
static void mh(const volatile void *p, size_t n) { (void)p; (void)n; }
static void fh(const volatile void *p) { (void)p; }
__attribute__((constructor(101))) static void fill(void) {
    for (int i = 0; i < 8; i++) __sanitizer_install_malloc_and_free_hooks(mh, fh);
}
int main(void) { return 0; }
"""


def _run(exe: Path, *argv: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(exe), *argv], env=_clean_env(**env), capture_output=True, text=True, timeout=60
    )


def _records(log: Path) -> int:
    return sum(line.startswith("CMP ") for line in log.read_text().splitlines())


@requires_clang
@requires_asan_rt
class TestMuteSurvivesEscape:
    def _go(self, tmp_path, *argv):
        exe = _build(tmp_path, _ESCAPE, *CMPLOG, asan=True)
        log = tmp_path / "cmplog.txt"
        log.write_text("")
        r = _run(exe, *argv, _CMPLOG_OUT=str(log), ASAN_OPTIONS="handle_segv=0")
        return r, _records(log)

    def test_control_hook_records_without_a_crash(self, tmp_path):
        r, n = self._go(tmp_path, "skip")
        assert "rc 0 0" in r.stdout
        assert n == 1

    def test_regression_hook_records_after_escape_from_real_compare(self, tmp_path):
        r, n = self._go(tmp_path)
        assert r.stdout.split()[:2] == ["rc", "-11"], r.stderr
        assert n == 1


@requires_clang
class TestMacros:
    def test_nested_mute_restores_previous_value(self, tmp_path):
        exe = _build(tmp_path, _NESTED, *CMPLOG)
        assert _run(exe).returncode == 0

    def test_cmp_count_evaluates_each_argument_once(self, tmp_path):
        exe = _build(tmp_path, _ONCE, *CMPLOG)
        r = _run(exe)
        assert r.returncode == 0, r.returncode


@requires_clang
@requires_asan_rt
class TestMallocHookTable:
    def test_full_table_is_reported(self, tmp_path):
        exe = _build(tmp_path, _FULL, *TRACE, asan=True)
        r = _run(exe, __AFL_ALLOC_SIZES="1")
        assert r.returncode == 0
        assert "hook table full" in r.stderr

    def test_control_quiet_when_off(self, tmp_path):
        exe = _build(tmp_path, _FULL, *TRACE, asan=True)
        assert "hook table full" not in _run(exe).stderr
