"""Regression: ``--inprocess-direct`` memset the whole edge table per exec.

``_run_c_direct`` called ``reset_bitmap()`` after ``run_target`` had already
bumped the generation (``reset_edge_map``). The memset cost a full table write
per exec (4 MB at 2^19 entries) and zeroed every slot, so each exec re-claimed
its edges cold and bumped the header ``edge_count``. That defeats
``_scan_new``'s fast path, which keys on an unchanged ``edge_count``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
MAP_SIZE = 65536
EXECS = 6

PROBE_SRC = r"""
#include <stddef.h>
extern void __sfuzz_state(unsigned var_id, unsigned long long value);
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    __sfuzz_state(1, len ? (unsigned long long)buf[0] : 999ULL);
    return 0;
}
"""


@pytest.fixture(scope="module")
def probe_so(tmp_path_factory):
    """A target marking one distinct edge per first input byte."""
    if shutil.which("clang") is None:
        pytest.skip("clang not available")

    build = tmp_path_factory.mktemp("direct_memset")
    src = build / "probe.c"
    src.write_text(PROBE_SRC)
    so = build / "probe_direct.so"
    proc = subprocess.run(
        ["clang", "-O1", "-fPIC", "-shared", "-o", str(so), str(src), str(SHIM)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"probe target failed to build: {proc.stderr[-400:]}")
    return str(so)


@pytest.fixture(scope="module")
def series(probe_so):
    """(edge_count, live edges) after each exec; same input, then a new one.

    Collected once: dlopen of one path returns one handle, so the shim stays
    bound to the first segment it attached to.
    """
    from fuzzer_tool.adapters.inprocess import InProcessRunner
    from fuzzer_tool.adapters.shm import ShmCoverage

    shm = ShmCoverage(size=MAP_SIZE)
    prev = {k: os.environ.get(k) for k in ("__AFL_SHM_ID", "AFL_MAP_SIZE")}
    os.environ["__AFL_SHM_ID"] = str(shm.env_id)
    os.environ["AFL_MAP_SIZE"] = str(MAP_SIZE)
    try:
        runner = InProcessRunner(
            probe_so,
            "fuzz_test",
            direct=True,
            shm_size=MAP_SIZE,
            coverage_env_id=shm.env_id,
            timeout=5.0,
        )
        runner._start()

        # Same input EXECS times, then one new input.
        inputs = [b"\x07"] * EXECS + [b"\x08"]
        out = []
        for data in inputs:
            shm.reset_edge_map()
            runner.run_one(data)
            out.append((shm.read_edge_count(), frozenset(shm.get_edge_ids())))
        yield out
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shm.cleanup()


def test_regression_direct_reset_memset(series):
    """Repeating an input claims no new slot: edge_count stays put."""
    counts = [c for c, _ in series[:EXECS]]
    assert len(set(counts)) == 1, f"slots re-claimed every exec: {counts}"


def test_new_edge_still_bumps_count(series):
    """Falsification: a new edge must still move edge_count."""
    assert series[EXECS][0] > series[EXECS - 1][0]


def test_no_ghost_edges(series):
    """Adversarial: without the memset, the previous exec's edge must not leak."""
    for _, edges in series:
        assert len(edges) == 1, f"edges leaked across execs: {edges}"
    assert series[EXECS][1].isdisjoint(series[EXECS - 1][1])
