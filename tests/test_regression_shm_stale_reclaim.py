"""Regression: generation-tagged reset let the edge table fill with
duplicates of the *same* edges until coverage went dark mid-campaign.

`__afl_map_reset()` (generation tagging, commit 1eb7979) stopped zeroing the
table and instead bumped an 8-bit generation counter, tagging live entries
with it. `__afl_map_edge()`'s probe loop was supposed to reclaim an entry
whose edge_id matched but whose generation was stale -- there is even a
comment saying so -- but the branch fell through and kept probing instead.

So every execution inserted a *fresh duplicate* of every edge that fired,
in a new slot, and the stale copy was never freed. Table occupancy grew by
(edges per execution) per execution, independent of how few distinct edges
the target actually had. Consequences, in the order they bite:

1. Occupancy reaches map_size after roughly map_size/edges_per_exec
   executions. Measured pre-fix on the 8192-entry floor with 43 distinct
   edges (png_read's shape): saturated at exec ~190.
2. Once saturated no slot can be claimed, so the current execution writes
   nothing and `get_edge_ids()` -- which filters by generation -- returns
   the empty set. Coverage guidance silently stops. Measured pre-fix: 43
   edges visible at exec 190, 0 at exec 200.
3. It does not even register as a drop. `__afl_note_drop()` was gated on
   every slot being *live*, and a saturated table is full of *stale*
   entries, so `read_dropped_edges()` stayed at 0 throughout.
4. Past 256 executions the 8-bit generation wraps, so entries written
   exactly 256 executions earlier alias as current. A saturated table then
   reports that old execution's edges as if they belonged to this one.

This affects every execution mode that resets between runs -- in-process
(non-direct_lite), forkserver, and fork+exec -- i.e. everything except
`direct_lite`, which never calls `reset_edge_map()`. The forkserver is the
default path.

The fix reclaims a stale same-edge entry in place. Table occupancy then
converges to the union of distinct edges ever seen, bounded by the target's
guard count, which is what the generation design intended.

These tests drive the real C shim, not a Python model of it: a Python-side
mirror could not have caught this, because `ShmCoverage.record_edge()` never
had the bug.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import textwrap

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

_HARNESS = textwrap.dedent(
    """
    #include <stdlib.h>
    extern void __afl_map_edge(unsigned int cur_loc);
    extern void __afl_map_reset(void);
    /* argv[1] = number of executions, argv[2] = distinct edges per execution.
       The reset between executions is what a real runner does; the final
       execution deliberately leaves its edges live so the parent can read
       the current generation's set. */
    int main(int argc, char **argv) {
        int n_exec = argc > 1 ? atoi(argv[1]) : 1;
        int n_edge = argc > 2 ? atoi(argv[2]) : 43;
        int oneshot = argc > 3 ? atoi(argv[3]) : 0;
        /* oneshot mode: fire a distinctive edge ONCE up front, then never
           again. Exercises generation-tag aliasing, which the steady-state
           mode below cannot: there every edge fires every execution and so
           is always refreshed with the current tag. */
        if (oneshot) __afl_map_edge(0xAAAAu);
        for (int e = 0; e < n_exec; e++) {
            if (oneshot) {
                __afl_map_reset();
                __afl_map_edge(0x1111u);
                continue;
            }
            for (int k = 0; k < n_edge; k++)
                __afl_map_edge((unsigned)(k * 7919));
            if (e < n_exec - 1) __afl_map_reset();
        }
        return 0;
    }
    """
)


def _have_cc() -> bool:
    from shutil import which

    return which("gcc") is not None or which("clang") is not None


pytestmark = pytest.mark.skipif(
    not _have_cc() or not os.path.exists(SHIM),
    reason="needs a C compiler and afl_shim.c",
)


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    from shutil import which

    cc = which("gcc") or which("clang")
    d = tmp_path_factory.mktemp("shim_probe")
    src = d / "harness.c"
    src.write_text(_HARNESS)
    exe = d / "harness"
    proc = subprocess.run(
        [
            cc,
            "-O2",
            "-g",
            # The harness asserts literal edge IDs (e.g. {0x1111}); ctx
            # hashing is default-on and would XOR a caller term into them.
            "-D__AFL_CTX_SENSITIVE=0",
            f"-include{SHIM}",
            "-o",
            str(exe),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"shim harness failed to build: {proc.stderr[-400:]}")
    return str(exe)


def _run(harness: str, cov: ShmCoverage, n_exec: int, n_edge: int = 43, oneshot: int = 0) -> None:
    env = dict(
        os.environ,
        __AFL_SHM_ID=str(cov.shm_id),
        AFL_MAP_SIZE=str(cov.num_entries),
    )
    subprocess.run([harness, str(n_exec), str(n_edge), str(oneshot)], env=env, capture_output=True)


def _occupied(cov: ShmCoverage) -> int:
    """Count non-empty slots regardless of generation."""
    n = 0
    for i in range(cov.num_entries):
        e = ctypes.c_uint32.from_address(cov._ptr + 24 + i * 8)
        if e.value != 0:
            n += 1
    return n


class TestStaleEntriesAreReclaimed:
    def test_occupancy_does_not_grow_with_execution_count(self, harness):
        """The whole bug in one assertion.

        43 distinct edges must occupy 43 slots whether they have fired once
        or five thousand times. Pre-fix this grew by 43 per execution.
        """
        counts = {}
        for n_exec in (1, 10, 100, 500):
            cov = ShmCoverage()
            try:
                _run(harness, cov, n_exec)
                counts[n_exec] = _occupied(cov)
            finally:
                cov.cleanup()
        assert set(counts.values()) == {43}, f"occupancy grew with exec count: {counts}"

    def test_coverage_survives_past_the_old_saturation_point(self, harness):
        """Pre-fix, edge visibility went 43 -> 0 between exec 190 and 200 on
        an 8192-entry map. Bracket that cliff and go well past it."""
        for n_exec in (100, 190, 200, 300, 1000):
            cov = ShmCoverage()
            try:
                _run(harness, cov, n_exec)
                visible = len(cov.get_edge_ids())
            finally:
                cov.cleanup()
            assert visible == 43, f"{n_exec} execs: {visible} edges visible, expected 43"

    def test_generation_wrap_with_edges_that_keep_firing(self, harness):
        """Steady state across a generation wrap: edges that fire every
        execution are refreshed with the current tag, so they stay correct.

        NOTE this is the *easy* half and on its own it is misleading -- an
        earlier version of this file had only this case and reported the
        wrap as safe. See the one-shot test below for why."""
        cov = ShmCoverage()
        try:
            _run(harness, cov, 260)
            assert len(cov.get_edge_ids()) == 43
            assert _occupied(cov) == 43
        finally:
            cov.cleanup()

    @pytest.mark.parametrize("n_exec", [1, 100, 255, 256, 257, 511, 512, 513, 1024])
    def test_a_quiet_edge_does_not_come_back_as_a_ghost(self, harness, n_exec):
        """Generation tags are 8 bits, so they repeat every 256 resets. An
        entry keeps the tag of the last execution in which its edge fired,
        so an edge that fires once and then goes quiet reads as live again
        exactly 256 executions later -- credited to an execution that never
        reached that code.

        Reclaiming in place does not help: the tag space is too small, not
        the reclaim logic. Fixed by wiping the table when the counter wraps
        to 0, which bounds staleness to one cycle and cannot alias.

        Measured pre-fix: ghost at n_exec = 256 and 512, absent at 255, 257
        and 511 -- the signature of an aliasing bug rather than a leak."""
        cov = ShmCoverage()
        try:
            _run(harness, cov, n_exec, oneshot=1)
            visible = cov.get_edge_ids()
        finally:
            cov.cleanup()
        assert visible == {0x1111}, (
            f"after {n_exec} execs the live set is {sorted(visible)}; "
            f"0xAAAA fired once at the start and must not reappear"
        )

    def test_repeat_fires_bump_the_count_not_the_slot_count(self, harness):
        """A second execution of the same edge must bump its counter and
        reuse its slot, leaving the cumulative edge_count header alone."""
        cov = ShmCoverage()
        try:
            _run(harness, cov, 1)
            first = cov.read_edge_count()
        finally:
            cov.cleanup()

        cov = ShmCoverage()
        try:
            _run(harness, cov, 50)
            later = cov.read_edge_count()
        finally:
            cov.cleanup()

        assert first == later == 43, f"cumulative edge_count inflated: {first} -> {later}"


class TestProbeWindowIsBounded:
    def test_no_drops_at_realistic_load(self, harness):
        """Every instrumented target in the tree sits far below the load
        where a bounded window starts discarding edges. 43 edges in 8192
        slots must cost nothing."""
        cov = ShmCoverage()
        try:
            _run(harness, cov, 200)
            assert cov.read_dropped_edges() == 0
        finally:
            cov.cleanup()

    def test_python_mirror_uses_the_same_window_as_the_shim(self):
        """`record_edge()` mirrors the shim's probe loop for tests. If the
        two windows disagree, the mirror can place an edge at a distance the
        shim would never look at, and the test-only path stops predicting
        the real one."""
        with open(SHIM) as fh:
            shim_src = fh.read()
        assert "#define __AFL_PROBE_MAX" in shim_src
        line = next(x for x in shim_src.splitlines() if x.startswith("#define __AFL_PROBE_MAX"))
        c_value = int(line.split()[-1].rstrip("u"))
        assert c_value == ShmCoverage.PROBE_MAX, (
            f"shim __AFL_PROBE_MAX={c_value} but ShmCoverage.PROBE_MAX={ShmCoverage.PROBE_MAX}"
        )
