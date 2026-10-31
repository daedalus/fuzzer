"""Atomic touched-bit OR only once the process has a second thread.

Atomic mode (the default) asks glibc's ``__libc_single_threaded`` per first
touch: one thread means no other writer, so a plain OR is exact. glibc
clears the flag before a new thread starts and never sets it back.
``__AFL_TOUCHED_ATOMIC=0`` stays plain whatever the thread count.
"""

from __future__ import annotations

import subprocess

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.shim_health import TOUCHED_ATOMIC_ENV
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env
from tests.test_shm_touched_bitmap import _bits, _live_slots

SIZE = 4096
THREADS = 4
IDS_PER_THREAD = 200

# Decision before and after a thread exists, then THREADS threads map
# disjoint id ranges at once (their slots share bitmap words).
_DRIVER = f"""
#include <pthread.h>
#include <stdio.h>

static void *idle(void *a) {{ return a; }}

static void *claim(void *a) {{
    uint32_t base = (uint32_t)(uintptr_t)a * {IDS_PER_THREAD} + 1;
    for (uint32_t i = 0; i < {IDS_PER_THREAD}; i++) __afl_map_id_raw(base + i);
    return a;
}}

int main(void) {{
    printf("%d\\n", __afl_touch_atomic_now());
    pthread_t t[{THREADS}];
    pthread_create(&t[0], 0, idle, 0);
    pthread_join(t[0], 0);
    printf("%d\\n", __afl_touch_atomic_now());
    for (uintptr_t i = 0; i < {THREADS}; i++) pthread_create(&t[i], 0, claim, (void *)i);
    for (int i = 0; i < {THREADS}; i++) pthread_join(t[i], 0);
    return 0;
}}
"""


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("touch_st"), _DRIVER, "-D__AFL_CTX_SENSITIVE=0", "-pthread"
    )


def _run(exe, env_value):
    cov = ShmCoverage(size=SIZE, touched_bitmap=True)
    try:
        extra = {} if env_value is None else {TOUCHED_ATOMIC_ENV: env_value}
        env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(cov.num_entries), **extra)
        r = subprocess.run([str(exe)], env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        before, after = map(int, r.stdout.split())
        return before, after, _bits(cov.touched_snapshot()), _live_slots(cov)
    finally:
        cov.cleanup()


@requires_clang
class TestTouchAtomicOnlyWhenThreaded:
    @pytest.mark.parametrize("env_value", [None, "1"])
    def test_plain_until_a_thread_exists(self, target, env_value):
        before, after, _, _ = _run(target, env_value)
        assert (before, after) == (0, 1)

    def test_falsify_plain_build_never_goes_atomic(self, target):
        """Override 0 must hold after threads appear, or auto ignores the user."""
        before, after, _, _ = _run(target, "0")
        assert (before, after) == (0, 0)

    def test_adversarial_concurrent_claims_lose_no_bit(self, target):
        _, _, bits, live = _run(target, None)
        assert bits == live
        assert len(bits) == THREADS * IDS_PER_THREAD
