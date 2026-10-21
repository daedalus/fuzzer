"""Regressions: per-execution shim state across in-process runs (review group B).

- prev_loc / path_hash / stack depth carried from one guarded run into the next
- SGFuzz transition state (__sfuzz_prev / __sfuzz_digest) never reset
- forkserver children inherited the parent's distance accumulators
- __afl_map_reset left the touched bitmap set across a generation bump
- InProcessRunner.reset_exec_state bumped the generation a second time
"""

from __future__ import annotations

import os
import subprocess
import types
from pathlib import Path

import numpy as np

from fuzzer_tool.adapters.inprocess import InProcessRunner
from fuzzer_tool.adapters.shm import DistanceTableShm, ShmCoverage
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env, _Forkserver

MAP_ENTRIES = 8192
DIST_KEY = 0x7F00_0000_0001  # far above any pc - base, never a real block
DEEP_FRAMES = 400

# Same-TU driver: the shim is -included first, so its statics are readable.
_DRIVER = """
#include <stdint.h>
#include <stdio.h>
#include <string.h>


/* Snapshot taken inside the entry, so driver edges after it do not count. */
static uint64_t seen_hash;
static uint32_t seen_depth, seen_loc;
#define SNAP() (seen_hash = __afl_path_hash_acc, seen_depth = __afl_max_stack_depth, \\
                seen_loc = __afl_prev_loc)

__attribute__((noinline)) static int rec(int n) {
    char pad[256];
    pad[0] = (char)n;
    __asm__ volatile("" : : "r"(pad) : "memory");
    if (n <= 0) return pad[0];
    return rec(n - 1) + pad[0];
}

static int shallow(const uint8_t *d, size_t n) { (void)d; (void)n; int r = rec(1); SNAP(); return r; }
static int deep(const uint8_t *d, size_t n) { (void)d; (void)n; return rec(@DEEP@); }
static int deep_crash(const uint8_t *d, size_t n) {
    (void)d;
    rec(@DEEP@);
    volatile int *p = (int *)(n - n);
    return *p;
}

static int sfuzz_set(const uint8_t *d, size_t n) { (void)d; (void)n; __sfuzz_state(1, 5); SNAP(); return 0; }
static uint64_t seen_digest, seen_prev;
static int sfuzz_peek(const uint8_t *d, size_t n) {
    (void)d; (void)n;
    seen_digest = __sfuzz_digest;
    seen_prev = __sfuzz_prev[1];
    return 0;
}

static uintptr_t base_in, base_out;
static int inner(const uint8_t *d, size_t n) { (void)d; (void)n; return rec(3); }
static int outer(const uint8_t *d, size_t n) {
    rec(1);
    base_in = __afl_stack_base;
    __afl_guarded_call(inner, d, n);
    base_out = __afl_stack_base;
    return 0;
}

static uint8_t buf[1];
static void run(int (*f)(const uint8_t *, size_t)) { __afl_guarded_call(f, buf, 1); }
static void state(void) {
    printf("%llx %u %x\\n", (unsigned long long)seen_hash, seen_depth, seen_loc);
}

/* One call site for every run: the caller context is part of the edge id.
 * Prints the state after the first and the last run. */
typedef int (*entry_fn)(const uint8_t *, size_t);
__attribute__((noinline)) static void seq3(entry_fn a, entry_fn b, entry_fn c) {
    entry_fn fs[3] = {a, b, c};
#pragma clang loop unroll(disable)
    for (int i = 0; i < 3; i++) {
        __afl_guarded_call(fs[i], buf, 1);
        if (i != 1) state();
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    const char *m = argv[1];
    if (!strcmp(m, "chain")) seq3(shallow, deep, shallow);
    if (!strcmp(m, "crash")) seq3(shallow, deep_crash, shallow);
    if (!strcmp(m, "sfuzz")) {
        run(sfuzz_set);
        printf("%d\\n", __sfuzz_digest != 0);
        run(sfuzz_peek);
        printf("%llu %llu\\n", (unsigned long long)seen_digest, (unsigned long long)seen_prev);
    }
    if (!strcmp(m, "sfuzz_twice")) seq3(sfuzz_set, sfuzz_set, sfuzz_set);
    if (!strcmp(m, "sfuzz_map_reset")) {
        sfuzz_set(buf, 1);
        __afl_map_reset();
        printf("%llu %llu\\n", (unsigned long long)__sfuzz_digest,
               (unsigned long long)__sfuzz_prev[1]);
    }
    if (!strcmp(m, "nested")) {
        run(outer);
        printf("%d %d\\n", base_in != 0, base_in == base_out);
    }
    if (!strcmp(m, "exec_reset")) {
        run(shallow);
        uint32_t gen = *__afl_gen_word;
        __afl_exec_reset();
        printf("%d %llu\\n", gen == *__afl_gen_word, (unsigned long long)__afl_path_hash_acc);
    }
    if (!strcmp(m, "touched")) { run(shallow); __afl_map_reset(); }
    if (!strcmp(m, "touched_only_run")) run(shallow);
    return 0;
}
""".replace("@DEEP@", str(DEEP_FRAMES))

# Simulates edges probed in the forkserver parent before fork: priority 101
# runs before the shim's own (unprioritised) constructor, with __afl_area NULL.
_DIST_DRIVER = """
#include <stdlib.h>
__attribute__((constructor(101))) static void early(void) {
    if (!getenv("PRE_PROBE")) return;
    __afl_map_dist_shm();
    __afl_probe_distance(@KEY@ULL);
}
int main(void) { __afl_probe_distance(@KEY@ULL); return (int)__afl_dist_hits; }
""".replace("@KEY@", hex(DIST_KEY))


def _driver(tmp_path: Path) -> Path:
    return _build(tmp_path, _DRIVER, "-fsanitize-coverage=trace-pc-guard")


def _run(exe: Path, mode: str, cov: ShmCoverage) -> list[str]:
    env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(cov.num_entries))
    r = subprocess.run([str(exe), mode], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def _run_fresh(exe: Path, mode: str, **shm_kw) -> list[str]:
    cov = ShmCoverage(size=MAP_ENTRIES, **shm_kw)
    try:
        return _run(exe, mode, cov)
    finally:
        cov.cleanup()


# ── Finding 1: per-run edge state ──────────────────────────────────────


@requires_clang
def test_regression_guarded_run_does_not_inherit_prev_run(tmp_path):
    """Same input after a deep one: same path hash and same stack depth."""
    first, again = _run_fresh(_driver(tmp_path), "chain")
    assert first == again


@requires_clang
def test_regression_guarded_run_clean_after_crash(tmp_path):
    """Adversarial: a run abandoned mid-way by a crash leaks nothing forward."""
    first, again = _run_fresh(_driver(tmp_path), "crash")
    assert first == again


@requires_clang
def test_regression_nested_guard_keeps_outer_run_state(tmp_path):
    """Adversarial: only the outermost guarded call starts a new run."""
    (line,) = _run_fresh(_driver(tmp_path), "nested")
    assert line == "1 1"


@requires_clang
def test_regression_exec_reset_keeps_generation(tmp_path):
    """The exported per-exec reset clears run state without a generation bump."""
    (line,) = _run_fresh(_driver(tmp_path), "exec_reset")
    assert line == "1 0"


def test_regression_reset_exec_state_prefers_exec_reset():
    """__afl_map_reset bumps the generation; reset_edge_map already did."""
    calls: list[str] = []
    runner = InProcessRunner.__new__(InProcessRunner)
    runner._lib = types.SimpleNamespace(
        __afl_exec_reset=lambda: calls.append("exec"),
        __afl_map_reset=lambda: calls.append("map"),
    )
    runner.reset_exec_state()
    assert calls == ["exec"]


def test_regression_reset_exec_state_old_shim_falls_back():
    """Falsification: a shim without the new symbol still gets reset."""
    calls: list[str] = []
    runner = InProcessRunner.__new__(InProcessRunner)
    runner._lib = types.SimpleNamespace(__afl_map_reset=lambda: calls.append("map"))
    runner.reset_exec_state()
    assert calls == ["map"]


# ── Finding 2: SGFuzz state ────────────────────────────────────────────


@requires_clang
def test_regression_sfuzz_state_reset_per_run(tmp_path):
    set_flag, peek = _run_fresh(_driver(tmp_path), "sfuzz")
    assert set_flag == "1", "precondition: the first run moved a state variable"
    assert peek == "0 0"


@requires_clang
def test_regression_sfuzz_same_transition_twice_same_hash(tmp_path):
    """Adversarial: 0->5 twice is the same edge, not 0->5 then 5->5.

    Compares the transition's own location (prev_loc after it). The path
    hash cannot be compared: the shim's inlined probe is instrumented, so a
    first-claim run and a slot-hit run take different shim edges.
    """
    first, again = _run_fresh(_driver(tmp_path), "sfuzz_twice")
    assert first.split()[-1] == again.split()[-1]


@requires_clang
def test_regression_map_reset_clears_sfuzz(tmp_path):
    (line,) = _run_fresh(_driver(tmp_path), "sfuzz_map_reset")
    assert line == "0 0"


# ── Finding 3: distance accumulators across fork ───────────────────────


def _forkserver_hits(exe: Path, pre_probe: bool) -> int:
    # The distance segment attaches only after the edge segment does.
    cov = ShmCoverage(size=MAP_ENTRIES)
    table = DistanceTableShm({DIST_KEY: 1.0})
    extra = {"PRE_PROBE": "1"} if pre_probe else {}
    env = _clean_env(
        __AFL_SHM_ID=cov.env_id,
        AFL_MAP_SIZE=str(MAP_ENTRIES),
        __AFL_DIST_SHM_ID=table.env_id,
        **extra,
    )
    fs = _Forkserver(exe, [], env)
    try:
        statuses = [fs.run_one(), fs.run_one()]
    finally:
        fs.close()
        table.cleanup()
        cov.cleanup()
    assert all(s is not None and os.WIFEXITED(s) for s in statuses)
    hits = {os.WEXITSTATUS(s) for s in statuses}
    assert len(hits) == 1, f"children disagree: {hits}"
    return hits.pop()


@requires_clang
def test_regression_forkserver_child_starts_without_parent_distance(tmp_path):
    exe = _build(tmp_path, _DIST_DRIVER)
    assert _forkserver_hits(exe, pre_probe=True) == 1


@requires_clang
def test_regression_forkserver_child_distance_control(tmp_path):
    """Control: no parent probe, the child's own probe counts once."""
    exe = _build(tmp_path, _DIST_DRIVER)
    assert _forkserver_hits(exe, pre_probe=False) == 1


# ── Finding 5: touched bitmap on C-side reset ──────────────────────────


def _stale_touched(cov: ShmCoverage) -> int:
    """Set bits whose slot is not live this generation (must be none)."""
    words = np.frombuffer(cov._map, dtype="<u8", count=cov.num_entries)
    live = (words.astype(np.uint32) != 0) & ((words >> np.uint64(56)) == cov.read_generation())
    return int((~live[cov._touched_slots()]).sum())


@requires_clang
def test_regression_map_reset_clears_touched_bitmap(tmp_path):
    """Bits of the bumped generation go; the shim's own later edges may set new ones."""
    exe = _driver(tmp_path)
    cov = ShmCoverage(size=MAP_ENTRIES, touched_bitmap=True)
    try:
        _run(exe, "touched", cov)
        assert cov.touched_supported, "the bitmap header must survive the clear"
        assert _stale_touched(cov) == 0
    finally:
        cov.cleanup()


@requires_clang
def test_regression_touched_bitmap_set_by_run(tmp_path):
    """Falsification: without a reset the run's bits are there to clear."""
    exe = _driver(tmp_path)
    cov = ShmCoverage(size=MAP_ENTRIES, touched_bitmap=True)
    try:
        _run(exe, "touched_only_run", cov)
        assert cov.touched_snapshot().any()
        assert _stale_touched(cov) == 0
    finally:
        cov.cleanup()
