"""Regression tests for finding #20 — direct_lite never reset the edge map.

``TargetRunner.run_target`` reset the SHM edge map before every execution
except in ``direct_lite`` mode, which was excluded by 66b026e ("perf: skip SHM
bitmap reset in direct_lite mode"). That exclusion was a genuine saving when
it landed — ``reset_edge_map()`` was a full ``table_bytes`` memset — but
generation tagging (1eb7979) made the reset O(1) the *next day* and the
exclusion was never revisited.

Without a generation bump nothing ages entries out, so every entry ever
written reads as live and ``get_edge_ids()`` returns the cumulative union of
the run instead of the edges of the execution just performed.

The first test compiles a shim-linked ``.so`` that marks exactly one distinct
edge per input, which is what makes the difference between "this execution"
and "every execution so far" directly countable. It is skipped when no C
toolchain is available; the source-level test below always runs.
"""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"

PROBE_SRC = r"""
#include <stddef.h>
extern void __sfuzz_state(unsigned var_id, unsigned long long value);
__attribute__((visibility("default")))
int fuzz_test(const unsigned char *buf, size_t len) {
    __sfuzz_state(1, len ? (unsigned long long)buf[0] : 999ULL);
    return 0;
}
"""


def _compiler() -> str | None:
    for cc in ("clang", "gcc", "cc"):
        if shutil.which(cc):
            return cc
    return None


@pytest.fixture(scope="module")
def probe_so(tmp_path_factory):
    """A target marking one distinct edge per first input byte."""
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler available")
    if not SHIM.is_file():
        pytest.skip("afl_shim.c not found")

    build = tmp_path_factory.mktemp("probe20")
    src = build / "probe.c"
    src.write_text(PROBE_SRC)
    so = build / "probe.so"
    proc = subprocess.run(
        [cc, "-O1", "-fPIC", "-shared", "-o", str(so), str(src), str(SHIM)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not so.is_file():
        pytest.skip(f"probe target failed to build: {proc.stderr[-400:]}")
    return str(so)


@pytest.fixture(scope="module")
def observations(probe_so):
    """Live edge set after each of several executions, collected once.

    Collected once on purpose: the shim binds __afl_area from __AFL_SHM_ID at
    dlopen time, and dlopen of the same path in one process returns the same
    handle, so a second ShmCoverage segment in the same interpreter would not
    be the one the target writes to. Both tests below read this one series.
    """
    from fuzzer_tool.adapters.inprocess import InProcessRunner
    from fuzzer_tool.adapters.shm import ShmCoverage

    map_size = 65536
    shm = ShmCoverage(size=map_size)
    prev = {k: os.environ.get(k) for k in ("__AFL_SHM_ID", "AFL_MAP_SIZE")}
    os.environ["__AFL_SHM_ID"] = str(shm.env_id)
    os.environ["AFL_MAP_SIZE"] = str(map_size)
    try:
        runner = InProcessRunner(
            probe_so,
            "fuzz_test",
            direct_lite=True,
            shm_size=map_size,
            coverage_env_id=shm.env_id,
            timeout=5.0,
        )
        runner._start()
        seen = []
        for i in range(8):
            # This is the line under test: run_target performs this for every
            # backend, direct_lite included.
            shm.reset_edge_map()
            runner.run_one(bytes([i]))
            seen.append(frozenset(shm.get_edge_ids()))
        return seen
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestDirectLiteCoverageIsPerExecution:
    def test_edge_count_does_not_accumulate(self, observations):
        """Each input marks one edge, so each execution must report one edge.

        Measured without the reset: [1, 2, 3, 4, 5, 6, 7, 8].
        """
        counts = [len(s) for s in observations]
        assert counts == [1] * len(counts), (
            f"direct_lite is reporting the cumulative union, not this "
            f"execution: {counts}"
        )

    def test_edge_identity_tracks_the_input(self, observations):
        """Not just the count: no edge may survive into the next execution."""
        assert len(set(observations)) == len(observations), (
            f"inputs collided onto one edge: {observations}"
        )
        for earlier, later in zip(observations[:-1], observations[1:], strict=True):
            assert not (earlier & later), "an edge survived into the next execution"


class TestNoBackendIsExcluded:
    """The wiring itself, so the exclusion cannot come back as a perf tweak."""

    def test_run_target_resets_for_every_backend(self):
        from fuzzer_tool.services.runner import TargetRunner

        src = inspect.getsource(TargetRunner.run_target)
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert "shm.reset_edge_map()" in code
        assert "not f._inprocess_runner.direct_lite:\n                shm.reset_edge_map" not in code
        # The one remaining direct_lite branch is the read side (skip the
        # read_bitmap copy because the target writes into shm_cov directly),
        # which is correct and must survive.
        assert "read_bitmap()" in code
