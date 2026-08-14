"""Regression tests for the forkserver on the default execution path.

Covers the four defects that kept it commented out (or would have bitten the
moment it was switched on):

  * coverage never reached the parent — the loader round-tripped a bitmap
    through a file while the target wrote to SHM;
  * the child's stderr was dropped, which on an ASAN build that exits 1 is
    the *only* crash signal ``ExecutionRunner.is_crash`` has;
  * the forkserver parent recorded its own control-flow into the map, so the
    same input produced different edge ids on its first executions;
  * an oversized RUN payload was skipped without being consumed, desyncing
    every later command on the pipe.

The shim is compiled with clang here because the edge callbacks are built on
-fsanitize-coverage=trace-pc-guard, which gcc does not implement.
"""

import os
import subprocess

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.adapters.shm import ShmCoverage
from tests.conftest import requires_clang

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIM = os.path.join(_ROOT, "src", "fuzzer_tool", "adapters", "afl_shim.c")

# argc==2 reads argv[1], otherwise stdin: the same shape as the real targets
# (png_read.c, grep_read.c), so both input channels are exercised.
_TARGET = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    unsigned char buf[512];
    size_t n = 0;
    if (argc == 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 1;
        n = fread(buf, 1, sizeof(buf), f);
        fclose(f);
    } else {
        n = fread(buf, 1, sizeof(buf), stdin);
    }
    if (n >= 6 && memcmp(buf, "CRASH", 5) == 0) {
        if (buf[5] == 'E') { fprintf(stderr, "ERROR: AddressSanitizer: fake-report\\n"); return 1; }
        if (buf[5] == 'H') { for (;;) pause(); }
    }
    if (n > 3 && buf[0] == 'a') return 0;
    if (n > 3 && buf[0] == 'b') return 0;
    return 0;
}
"""


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    d = tmp_path_factory.mktemp("fsrv")
    src = d / "t.c"
    src.write_text(_TARGET)
    exe = d / "t"
    r = subprocess.run(
        [
            "clang",
            "-O0",
            "-fsanitize-coverage=trace-pc-guard",
            "-include",
            SHIM,
            "-o",
            str(exe),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"target failed to build: {r.stderr[:300]}")
    return str(exe)


@pytest.fixture
def runner(target):
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")
    shm = ShmCoverage(size=8192)
    r = ForkserverRunner(
        target,
        timeout=2.0,
        env={"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.size)},
    )
    if not r.start():
        shm.cleanup()
        pytest.skip("forkserver failed to start")
    try:
        yield r, shm
    finally:
        r.stop()
        shm.cleanup()


@requires_clang
def test_regression_forkserver_coverage_reaches_parent_shm(runner):
    """The exec'd child must write edges into the parent's SHM directly.

    Derived independently of the loader: the assertion is that the parent's
    own segment is non-empty after a run, which is only true if the child
    attached to __AFL_SHM_ID on its own.

    On failure this reports the SHM header alongside the child's exit
    status, which is what the analysis doc's "Loose thread" section asks for
    — the first ShmCoverage built in a process has twice been seen reading
    back an empty table after a child that exited 0, and the header was
    never captured at the time. Observed once here too, on the first run in
    a fresh clone, then not again in 8 consecutive runs.
    """
    r, shm = runner
    shm.reset_edge_map()
    rc, stderr = r.run_one(b"aaaa")
    assert rc == 0
    edges = shm.get_edge_ids()
    assert edges, (
        f"empty edge table after rc={rc}: edge_count={shm.read_edge_count()} "
        f"diag=0x{shm.read_diag():08x} dropped={shm.read_dropped_edges()} "
        f"path_hash=0x{shm.read_path_hash():016x} stderr={stderr!r}"
    )


@requires_clang
def test_regression_forkserver_distinguishes_inputs(runner):
    """Coverage must be per-input, not a constant echo of process startup."""
    r, shm = runner
    sets = []
    for payload in (b"aaaa", b"CRASHE"):
        shm.reset_edge_map()
        r.run_one(payload)
        sets.append(frozenset(shm.get_edge_ids()))
    assert sets[0] != sets[1]


@requires_clang
def test_regression_forkserver_parent_does_not_record_own_edges(runner):
    """Same input, same edge ids — every execution over.

    The forkserver loop is instrumented (it is in the target's translation
    unit). Left recording, it wrote its own edges into the map after the
    per-exec reset and advanced __afl_prev_loc between forks, so an input's
    first two executions each produced a different edge set.
    """
    r, shm = runner
    observed = set()
    for _ in range(6):
        shm.reset_edge_map()
        r.run_one(b"aaaa")
        observed.add(frozenset(shm.get_edge_ids()))
    assert len(observed) == 1


@requires_clang
def test_regression_forkserver_forwards_child_stderr(runner):
    """A sanitizer report on the child's stderr must reach the caller.

    ASAN exits 1, so without stderr `is_crash()` sees a clean run and the
    finding is lost.
    """
    r, shm = runner
    shm.reset_edge_map()
    rc, stderr = r.run_one(b"CRASHE")
    assert rc == 1
    assert "AddressSanitizer" in stderr


@requires_clang
def test_regression_forkserver_stderr_is_not_carried_between_runs(runner):
    """The stderr pipe outlives every child — output must not leak forward."""
    r, shm = runner
    r.run_one(b"CRASHE")
    _, stderr = r.run_one(b"aaaa")
    assert stderr == ""


@requires_clang
def test_regression_forkserver_timeout_reports_minus_one(runner):
    """A hung child is -1, not -SIGKILL.

    -9 is in SIGNAL_CRASH_CODES, so reporting the raw wait status would file
    every slow input as a fatal-signal crash.
    """
    r, shm = runner
    rc, _ = r.run_one(b"CRASHH")
    assert rc == -1


@requires_clang
def test_regression_forkserver_survives_a_timeout(runner):
    """The loader must still answer after killing a hung child."""
    r, shm = runner
    r.run_one(b"CRASHH")
    shm.reset_edge_map()
    rc, _ = r.run_one(b"aaaa")
    assert rc == 0
    assert len(shm.get_edge_ids()) > 0


@requires_clang
def test_regression_oversized_run_does_not_desync_protocol(runner):
    """An over-cap payload must be drained and answered, not skipped.

    The old loader `continue`d without consuming the body, leaving it in the
    pipe to be parsed as the next command — every subsequent run then read a
    reply belonging to a different request.
    """
    r, shm = runner
    huge = b"x" * (17 * 1024 * 1024)  # over the loader's 16 MiB ceiling
    rc, _ = r.run_one(huge)
    assert rc == -2
    # The next command must still line up with its own reply.
    shm.reset_edge_map()
    rc, _ = r.run_one(b"aaaa")
    assert rc == 0


class TestUpdateShmAfterResize:
    def test_env_overrides_track_the_new_segment(self):
        """resize() allocates a new segment; the loader's env must follow it.

        Without a restart the children keep attaching to the removed segment
        and their coverage is dropped on the floor.
        """
        r = ForkserverRunner("/fake/target", env={"__AFL_SHM_ID": "1", "AFL_MAP_SIZE": "8192"})
        r.update_shm_after_resize("4242", 262144)
        assert r.env_overrides["__AFL_SHM_ID"] == "4242"
        assert r.env_overrides["AFL_MAP_SIZE"] == "262144"

    def test_no_restart_when_never_started(self):
        r = ForkserverRunner("/fake/target")
        r.update_shm_after_resize("7", 16384)  # must not raise
        assert r._proc is None
