"""Shim bit tricks: SWAR nonzero-byte mask in the sancov fold, masked dist probe.

A harness is compiled with ``-include afl_shim.c`` so it reaches the shim's
static helpers in the same TU (the pattern ``test_sancov_modes.py`` uses).
Its main is uninstrumented, so nothing it does ticks coverage itself.
"""

import ctypes
import os
import subprocess

import pytest

from fuzzer_tool.adapters.shm import DistanceTableShm
from tests.conftest import requires_clang

SHIM = "src/fuzzer_tool/adapters/afl_shim.c"

_HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NOCOV __attribute__((no_sanitize("coverage")))

/* Reference: one bit per nonzero byte, byte j -> bit j. */
NOCOV static uint64_t ref_mask(const uint8_t *p) {
    uint64_t m = 0;
    for (int j = 0; j < 64; j++)
        if (p[j]) m |= 1ULL << j;
    return m;
}

NOCOV static int check(const uint8_t *p) {
    return __afl_sancov_nz_mask(p) == ref_mask(p);
}

/* Every byte value at every position, uniform fills, then xorshift chunks
 * with ~1/8 of bytes nonzero. Prints the count checked, or the bad case. */
NOCOV static int run_nz(void) {
    uint8_t buf[64];
    long n = 0;

    for (int j = 0; j < 64; j++)
        for (int v = 0; v < 256; v++, n++) {
            memset(buf, 0, 64);
            buf[j] = (uint8_t)v;
            if (!check(buf)) { printf("bad pos %d val %d\n", j, v); return 1; }
        }

    for (int v = 0; v < 256; v++, n++) {
        memset(buf, v, 64);
        if (!check(buf)) { printf("bad fill %d\n", v); return 1; }
    }

    uint64_t s = 0x9e3779b97f4a7c15ULL;
    for (int r = 0; r < 20000; r++, n++) {
        for (int j = 0; j < 64; j++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            buf[j] = (s & 7) ? 0 : (uint8_t)(s >> 8);
        }
        if (!check(buf)) { printf("bad random %d\n", r); return 1; }
    }
    printf("ok %ld\n", n);
    return 0;
}

/* Attach the table, probe each key in argv, report attach + hits + sum. */
NOCOV static int run_dist(int argc, char **argv) {
    __afl_map_dist_shm();
    for (int i = 2; i < argc; i++)
        __afl_probe_distance(strtoull(argv[i], NULL, 0));
    printf("%d %llu %llu %u\n", __afl_dist_table != NULL,
           (unsigned long long)__afl_dist_hits, (unsigned long long)__afl_dist_sum,
           (unsigned)__afl_health[__AFL_HEALTH_SEG_REJECTED]);
    return 0;
}

NOCOV int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "nz")) return run_nz();
    return run_dist(argc, argv);
}
"""


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    d = tmp_path_factory.mktemp("shim_bitops")
    src = d / "h.c"
    src.write_text(_HARNESS)
    obj = d / "h.o"
    exe = d / "h"
    # Compile with the coverage flag, link without it: linking with it pulls
    # a sanitizer runtime minimal toolchains lack (see test_sancov_modes.py).
    cc = [
        "clang",
        "-O2",
        "-D__AFL_CTX_SENSITIVE=0",
        "-D__AFL_DISTANCE_MODE",
        "-fsanitize-coverage=trace-pc",
        "-include",
        SHIM,
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    r = subprocess.run(cc, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]

    r = subprocess.run(["clang", str(obj), "-o", str(exe), "-ldl"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    return str(exe)


def _run(exe, *args, env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    for k in ("__AFL_SHM_ID", "__AFL_DIST_SHM_ID", "LD_PRELOAD"):
        env.pop(k, None)
    env.update(env_extra or {})
    r = subprocess.run([exe, *args], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.split()


@requires_clang
class TestNonzeroMask:
    def test_regression_nz_mask_matches_bytes(self, harness):
        """SWAR + multiply mask equals the byte loop on every single-byte case."""
        out = _run(harness, "nz")
        assert out[0] == "ok", out
        assert int(out[1]) >= 64 * 256


# Two keys sharing home slot 3 of a 4-slot table: the second wraps to slot 0.
# The 2^40 key checks a u64 key masks like `% capacity` did.
_WRAP_KEY = (1 << 40) | 3
_ENTRIES = {3: 1.0, _WRAP_KEY: 2.0}
_MISS_KEY = 11  # also homes at 3: walks 3, 0, then stops at empty slot 1


def _table():
    t = DistanceTableShm(_ENTRIES)
    assert t.capacity == 4
    return t


@requires_clang
class TestDistProbe:
    def test_probe_wraps_and_misses(self, harness):
        """Control: power-of-two table, wrap-around hit and an early miss."""
        t = _table()
        try:
            env = {"__AFL_DIST_SHM_ID": t.env_id}
            out = _run(harness, "dist", "3", str(_WRAP_KEY), str(_MISS_KEY), env_extra=env)
        finally:
            t.cleanup()
        attached, hits, total, _rej = map(int, out)
        assert attached == 1
        assert hits == 2
        assert total == sum(round(v * 100) for v in _ENTRIES.values())

    def test_regression_dist_rejects_non_pow2(self, harness):
        """Adversarial: a non-power-of-two capacity would break the masked probe."""
        t = _table()
        try:
            ctypes.c_uint32.from_address(t._ptr).value = 3
            env = {"__AFL_DIST_SHM_ID": t.env_id}
            out = _run(harness, "dist", "3", str(_WRAP_KEY), env_extra=env)
        finally:
            t.cleanup()
        attached, hits, _total, rejected = map(int, out)
        assert attached == 0
        assert hits == 0
        assert rejected >= 1
