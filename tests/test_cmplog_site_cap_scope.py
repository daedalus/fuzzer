"""Cmplog shim record knobs: per-site cap and dedup scope.

``$__AFL_CMPLOG_SITE_CAP=N``: a trace-cmp site writes at most N distinct
integer records per drain. ``$__AFL_CMPLOG_DEDUP_SCOPE=campaign``: the
record dedup table survives ``__cmplog_reset``; ``__cmplog_rescope()``
clears it on demand (the I2S probe needs one exec's full record set).

Two loop sites run 100 iterations in phase A, ``__cmplog_reset`` drains,
then phase B repeats them. Each site record carries the seed and a
per-iteration operand, so the iteration it came from is recoverable.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_clang

AFL_SHIM = Path(__file__).parent.parent / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

_ITERS = 100
_SEED = 0x1234
_SITES = {"a": (7919, 0x10000), "b": (104729, 0x20000)}

_TARGET_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void __tracecmp_flush(void);
void __cmplog_reset(void);
void __cmplog_rescope(void) __attribute__((weak));
volatile uint32_t seed = 0x1234;
volatile int sink;

__attribute__((noinline)) static void site_a(int n) {
#pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < n; i++) sink += (seed == (uint32_t)i * 7919u + 0x10000u);
}
__attribute__((noinline)) static void site_b(int n) {
#pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < n; i++) sink += (seed == (uint32_t)i * 104729u + 0x20000u);
}

/* Copy this phase's records to stdout under a header. */
static void dump(const char *phase) {
    char buf[4096];
    size_t k;
    __tracecmp_flush();
    FILE *f = fopen(getenv("_CMPLOG_OUT"), "r");
    printf("== %s\n", phase);
    while (f && (k = fread(buf, 1, sizeof buf, f)) > 0) fwrite(buf, 1, k, stdout);
    if (f) fclose(f);
    fflush(stdout);
}

int main(int argc, char **argv) {
    int rescope = argc > 1 && strcmp(argv[1], "rescope") == 0;
    site_a(100); site_b(100);
    dump("A");
    __cmplog_reset();
    if (rescope && __cmplog_rescope) __cmplog_rescope();
    site_a(100); site_b(100);
    dump("B");
    return 0;
}
"""


@pytest.fixture(scope="module")
def cap_target(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("cmplog_site_cap")
    src = d / "target.c"
    src.write_text(_TARGET_C)
    exe = d / "target"
    proc = subprocess.run(
        [
            "clang",
            "-O2",
            "-fsanitize-coverage=trace-pc-guard,trace-cmp",
            "-D__AFL_CMPLOG=1",
            "-include",
            str(AFL_SHIM),
            "-o",
            str(exe),
            str(src),
            "-ldl",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"site cap target did not build: {proc.stderr[-400:]}")
    return exe


def _le_hex(v: int) -> str:
    return (v & 0xFFFFFFFF).to_bytes(4, "little").hex()


# Operand b hex -> (site, iteration).
_ITER_OF = {
    _le_hex(i * mul + base): (site, i)
    for site, (mul, base) in _SITES.items()
    for i in range(_ITERS)
}


def _run(exe: Path, tmp_path: Path, mode: str = "plain", **env_kv: str) -> dict:
    """Phase -> site -> sorted iterations seen in that phase's records."""
    env = dict(os.environ, _CMPLOG_OUT=str(tmp_path / "rec.cmplog"))
    for var in ("__AFL_CMPLOG_SITE_CAP", "__AFL_CMPLOG_DEDUP_SCOPE"):
        env.pop(var, None)
    env.update(env_kv)
    run = subprocess.run([str(exe), mode], capture_output=True, text=True, env=env, timeout=60)
    assert run.returncode == 0, run.stderr

    seen: dict = {"A": {"a": [], "b": []}, "B": {"a": [], "b": []}}
    phase = None
    seed_hex = _le_hex(_SEED)
    for ln in run.stdout.splitlines():
        if ln.startswith("== "):
            phase = ln[3:]
            continue
        p = ln.split()
        if phase is None or len(p) < 3 or p[0] != "CMP" or p[1] != seed_hex:
            continue
        hit = _ITER_OF.get(p[2])
        if hit:
            seen[phase][hit[0]].append(hit[1])
    for phase in seen.values():
        for site in phase:
            phase[site].sort()
    return seen


_ALL = list(range(_ITERS))


@requires_clang
class TestSiteCap:
    def test_default_logs_every_iteration_each_drain(self, cap_target, tmp_path):
        """Falsification: no knob, both phases log all 100 per site."""
        seen = _run(cap_target, tmp_path)
        assert all(seen[ph][s] == _ALL for ph in "AB" for s in "ab")

    def test_cap_keeps_first_n_per_site_per_drain(self, cap_target, tmp_path):
        seen = _run(cap_target, tmp_path, __AFL_CMPLOG_SITE_CAP="8")
        assert all(seen[ph][s] == list(range(8)) for ph in "AB" for s in "ab")

    @pytest.mark.parametrize("val", ["0", "-3", "abc", ""])
    def test_adversarial_bad_cap_means_off(self, cap_target, tmp_path, val):
        seen = _run(cap_target, tmp_path, __AFL_CMPLOG_SITE_CAP=val)
        assert all(seen[ph][s] == _ALL for ph in "AB" for s in "ab")


@requires_clang
class TestDedupScope:
    def test_campaign_scope_drops_repeats_across_drains(self, cap_target, tmp_path):
        seen = _run(cap_target, tmp_path, __AFL_CMPLOG_DEDUP_SCOPE="campaign")
        assert seen["A"] == {"a": _ALL, "b": _ALL}
        assert seen["B"] == {"a": [], "b": []}

    def test_rescope_restores_full_set(self, cap_target, tmp_path):
        seen = _run(cap_target, tmp_path, "rescope", __AFL_CMPLOG_DEDUP_SCOPE="campaign")
        assert seen["B"] == {"a": _ALL, "b": _ALL}

    def test_adversarial_unknown_scope_is_drain(self, cap_target, tmp_path):
        seen = _run(cap_target, tmp_path, __AFL_CMPLOG_DEDUP_SCOPE="bogus")
        assert seen["B"] == {"a": _ALL, "b": _ALL}

    def test_cap_counts_emitted_records_so_window_advances(self, cap_target, tmp_path):
        """Repeats dropped by dedup do not use the cap: phase B logs the next 8."""
        seen = _run(
            cap_target, tmp_path, __AFL_CMPLOG_SITE_CAP="8", __AFL_CMPLOG_DEDUP_SCOPE="campaign"
        )
        assert seen["A"] == {"a": list(range(8)), "b": list(range(8))}
        assert seen["B"] == {"a": list(range(8, 16)), "b": list(range(8, 16))}
