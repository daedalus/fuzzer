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

import gc
import os
import shutil
import subprocess
import time
import weakref

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled
from fuzzer_tool.adapters.shm import ShmCoverage
from tests.conftest import requires_clang


def _pid_alive(pid: int) -> bool:
    """True while *pid* exists. Zombies count as gone: the loader is our
    grandchild's parent, so a reaped-but-unwaited process is still an exit."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


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
    # Guard here as well as on the tests: subprocess.run(["clang", ...])
    # raises FileNotFoundError when clang is absent, and a fixture that
    # raises is reported as an *error*, not a skip -- which is how two of
    # these showed up as "collection errors" on a machine without clang.
    if not shutil.which("clang"):
        pytest.skip("clang not installed")
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


@requires_clang
def test_regression_forkserver_is_actually_used(runner):
    """The shim-built executable must be driven by the forkserver, not fork+exec.

    This is the assertion the suite was missing. Gating
    ``__afl_start_forkserver()`` on ``__AFL_FORKSRV`` without having any
    loader set the variable made the handshake fail on every INIT, so
    ``use_forksrv`` was always 0 and each RUN silently fell back to
    ``run_executable()`` — reverting the forkserver work at no visible cost,
    because every other assertion in this file holds identically under both
    paths. The mode on the READY line is the only thing that separates them.
    """
    r, _ = runner
    assert r.exec_mode == "forkserver"


@requires_clang
def test_regression_target_stdout_does_not_desync_protocol(tmp_path):
    """A target that prints must not corrupt the loader's RUN/RC stream.

    stdout is the loader's half of the protocol. Children were spawned with
    stdin and stderr redirected but not stdout, so the target's own output was
    parsed as an ``RC <rc> <err_len>`` header: run_one returned -2 for a run
    that had in fact exited 0, and the stream stayed desynced from then on.
    """
    if _ensure_compiled() is None:
        pytest.skip("fuzz_loader failed to compile")

    src = tmp_path / "chatty.c"
    src.write_text(
        """
#include <stdio.h>
int main(void) {
    char b[64];
    size_t n = fread(b, 1, sizeof b, stdin);
    printf("RC 99 0\\n");   /* a well-formed but bogus reply header */
    fflush(stdout);
    return (int)(n > 1024);
}
"""
    )
    exe = tmp_path / "chatty"
    build = subprocess.run(
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
    if build.returncode != 0:
        pytest.skip(f"target failed to build: {build.stderr[:300]}")

    shm = ShmCoverage(size=8192)
    r = ForkserverRunner(
        str(exe),
        timeout=2.0,
        env={"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.size)},
    )
    try:
        assert r.start(), "loader failed to start"
        # The handshake probe child also prints; READY itself must survive it.
        assert r.exec_mode == "forkserver"
        for _ in range(4):
            rc, _err = r.run_one(b"aaaa")
            assert rc == 0, f"target stdout leaked into the reply stream (rc={rc})"
    finally:
        r.stop()
        shm.cleanup()


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


class TestRunnerIsCollectable:
    """An un-stopped runner must not pin itself alive forever.

    The stderr-drain thread used to be started with `target=self._drain_stderr`,
    a bound method, which made the runner reachable from a thread that could
    only exit once the runner was stopped — and stop() only runs from
    __del__, which cannot run while the object is reachable. The loop had no
    exit: every runner that was not explicitly stopped leaked a blocked
    thread, an orphaned fuzz_loader, its target child, and a SHM segment
    pinned in `dest` state.

    Measured before the fix: two test files left 98 live runners with 98 live
    children, and gc.collect() freed none of them — they were reachable, not
    garbage. A full-suite run peaked at ~185 orphaned processes.
    """

    @requires_clang
    def test_dropped_runner_is_garbage_collected(self, target):
        if _ensure_compiled() is None:
            pytest.skip("fuzz_loader failed to compile")
        shm = ShmCoverage(size=8192)
        try:
            r = ForkserverRunner(
                target,
                timeout=2.0,
                env={"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.size)},
            )
            if not r.start():
                pytest.skip("forkserver failed to start")
            child = r._proc.pid
            ref = weakref.ref(r)

            del r  # no stop(), which is the whole point
            gc.collect()

            assert ref() is None, "runner still reachable — the drain thread pins it"

            # __del__ -> stop() should have taken the child with it.
            for _ in range(50):
                if not _pid_alive(child):
                    break
                time.sleep(0.1)
            assert not _pid_alive(child), f"loader {child} outlived its runner"
        finally:
            shm.cleanup()

    @requires_clang
    def test_drain_thread_holds_no_reference_to_the_runner(self, target):
        """The specific defect, asserted directly rather than by its effect."""
        if _ensure_compiled() is None:
            pytest.skip("fuzz_loader failed to compile")
        shm = ShmCoverage(size=8192)
        r = ForkserverRunner(
            target,
            timeout=2.0,
            env={"__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.size)},
        )
        try:
            if not r.start():
                pytest.skip("forkserver failed to start")
            thread = r._stderr_thread
            # A bound method target keeps __self__; a plain function does not.
            assert getattr(thread._target, "__self__", None) is not r
            assert r not in gc.get_referents(thread)
        finally:
            r.stop()
            shm.cleanup()
