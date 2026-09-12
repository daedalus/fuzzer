"""The dropped-edge counter was 16 bits packed into the diag word.

Three separate problems, all from the packing:

1. It was documented as bits 8..31 but bits 24..31 are the generation tag and
   the code masked 0xFFFF, so it was 16 bits with a 24-bit claim.

2. It pinned at 65,535. Measured on a 1024-entry table fed 4000 guards
   (1,953 drops per execution): pinned after 34 EXECUTIONS. Its only
   magnitude consumer -- the stall-triggered resize -- does not run until
   `--stall` executions have passed with no new edge (default 1,000), so on
   any target that saturates, every value that consumer ever read was 65,535.

3. A pinned counter has no derivative, which rules out the per-execution
   question that actually matters: was THIS execution's edge set truncated?
   That one has a consumer with teeth -- `_calibrate_seed_stability` masks
   non-reproducing edges permanently, and drops make set divergence say
   nothing about determinism.

Now it is a dedicated u32 at SHM_DROP_OFFSET, between the header and the
edge table, and the cumulative figure is accumulated on the Python side as
an unbounded int differenced out of that word.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import SHM_DROP_OFFSET, SHM_METADATA_SIZE, ShmCoverage

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

_DRIVER = """
#include <stdlib.h>
int main(int argc, char **argv) {
    uint32_t n = (uint32_t)atoi(argv[1]);
    for (uint32_t g = 1; g <= n; g++) { uint32_t guard = g;
        __sanitizer_cov_trace_pc_guard(&guard); }
    return 0;
}
"""

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")


@pytest.fixture(scope="module")
def saturating(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("drops")
    src = d / "d.c"
    src.write_text(_DRIVER)
    exe = d / "d"
    r = subprocess.run(
        ["gcc", "-O1", "-g", "-D__AFL_CTX_SENSITIVE=0", "-include", SHIM, "-o", str(exe), str(src)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"shim failed to build: {r.stderr[:300]}")
    return str(exe)


def _run(exe, cov, guards=4000):
    env = {**os.environ, "__AFL_SHM_ID": cov.env_id, "AFL_MAP_SIZE": str(cov.num_entries)}
    return subprocess.run([exe, str(guards)], env=env, capture_output=True)


class TestCounterHasItsOwnWord:
    @needs_cc
    def test_shim_writes_the_dedicated_offset(self, saturating):
        import ctypes

        cov = ShmCoverage(size=1024)
        try:
            _run(saturating, cov)
            word = ctypes.c_uint32.from_address(cov._ptr + SHM_DROP_OFFSET).value
            assert word > 0, "shim did not write the dedicated drop word"
            assert word == cov.read_dropped_edges()
        finally:
            cov.cleanup()

    @needs_cc
    def test_drops_no_longer_touch_the_diag_word(self, saturating):
        """The point of the move: this word has no hot-path writer left.

        Two writers of one word is what let inprocess.reset_bitmap() destroy
        the generation, the ctx width and the drop count together, every
        execution, for as long as it memset from the segment base.
        """
        cov = ShmCoverage(size=1024)
        try:
            _run(saturating, cov)
            assert cov.read_dropped_edges() > 0, "test target did not saturate"
            diag = cov.read_diag()
            # ctx bits 0..7 (0 for this build) and generation 24..31 only.
            assert (diag >> 8) & 0xFFFF == 0, f"diag bits 8..23 were written: 0x{diag:08x}"
        finally:
            cov.cleanup()

    def test_counter_survives_the_per_execution_reset(self):
        """reset_edge_map() must not clear it: the counter is cumulative for
        the segment, and per-execution figures come from differencing."""
        cov = ShmCoverage(size=1024)
        try:
            for _ in range(10):
                cov._note_drop()
            cov.reset_edge_map()
            assert cov.read_dropped_edges() == 10
        finally:
            cov.cleanup()

    def test_reset_dropped_edges_clears_word_and_total(self):
        cov = ShmCoverage(size=1024)
        try:
            for _ in range(10):
                cov._note_drop()
            assert cov.read_dropped_edges() == 10
            cov.reset_dropped_edges()
            assert cov.read_dropped_edges() == 0
            assert cov.read_dropped_edges() == 0
        finally:
            cov.cleanup()


class TestPinIsGone:
    @needs_cc
    def test_counter_passes_the_old_sixteen_bit_ceiling(self, saturating):
        """Pre-fix this pinned at 65,535 after 34 executions."""
        cov = ShmCoverage(size=1024)
        try:
            for _ in range(40):
                _run(saturating, cov)
            total = cov.read_dropped_edges()
            assert total > 0xFFFF, (
                f"drop count did not exceed the old 16-bit ceiling after 40 "
                f"saturating executions: {total}"
            )
            assert not cov.drop_counter_saturated()
        finally:
            cov.cleanup()

    def test_pin_is_still_reportable(self):
        import ctypes

        cov = ShmCoverage(size=1024)
        try:
            ctypes.c_uint32.from_address(cov._ptr + SHM_DROP_OFFSET).value = 0xFFFFFFFF
            assert cov.read_dropped_edges() == 0xFFFFFFFF
            assert cov.drop_counter_saturated()
        finally:
            cov.cleanup()

    def test_cumulative_read_does_not_consume_the_delta_cursor(self):
        """Reporting reads this on a timer; one decision path differences it.

        An earlier version of read_dropped_edges() folded deltas into an
        accumulator, which meant a stats print landing between two of the
        calibration loop's delta reads consumed the drops it was about to
        look at — the veto then silently failed to fire on a saturated map.
        """
        cov = ShmCoverage(size=1024)
        try:
            for _ in range(5):
                cov._note_drop()
            for _ in range(3):
                assert cov.read_dropped_edges() == 5  # as a reporting path would
            assert cov.dropped_edges_delta() == 5, "cumulative read stole the delta"
            assert cov.dropped_edges_delta() == 0
        finally:
            cov.cleanup()

    def test_word_saturates_rather_than_wrapping(self):
        """A wrap would read as zero drops — the exact self-masking the
        counter exists to prevent."""
        import ctypes

        cov = ShmCoverage(size=1024)
        try:
            ctypes.c_uint32.from_address(cov._ptr + SHM_DROP_OFFSET).value = 0xFFFFFFFF
            cov._note_drop()
            assert cov.read_dropped_edges() == 0xFFFFFFFF
        finally:
            cov.cleanup()


class TestPerExecutionDelta:
    @needs_cc
    def test_delta_reports_this_execution_only(self, saturating):
        cov = ShmCoverage(size=1024)
        try:
            _run(saturating, cov)
            first = cov.dropped_edges_delta()
            assert first > 0
            second_none = cov.dropped_edges_delta()
            assert second_none == 0, "delta did not consume the previous reading"
            _run(saturating, cov)
            assert cov.dropped_edges_delta() == pytest.approx(first, rel=0.25)
        finally:
            cov.cleanup()

    def test_delta_is_clamped_after_a_resize_rebinds_the_segment(self):
        """resize() carries the header across, but a caller may clear the
        word; a delta must never go negative and must never be a huge
        spurious positive."""
        cov = ShmCoverage(size=1024)
        try:
            for _ in range(5):
                cov._note_drop()
            assert cov.dropped_edges_delta() == 5
            cov.resize(4096)
            cov.reset_dropped_edges()
            assert cov.dropped_edges_delta() == 0
            cov._note_drop()
            assert cov.dropped_edges_delta() == 1
        finally:
            cov.cleanup()


class TestPythonMirrorCountsDrops:
    def test_record_edge_notes_a_full_window_miss(self):
        """record_edge() is the mirror of the shim's insertion path, and it
        was silently losing edges without counting them — reproducing, in the
        mirror, the exact self-masking the counter exists to prevent."""
        cov = ShmCoverage(size=64)
        try:
            # PROBE_MAX is 64 and the table is 64 entries, so once the table
            # is full every further edge exhausts its window.
            placed = sum(1 for e in range(1, 200) if cov.record_edge(e))
            assert placed < 199, "table did not fill; test is not exercising drops"
            assert cov.read_dropped_edges() == 199 - placed
        finally:
            cov.cleanup()


class TestFrontRegionSizing:
    def test_table_offset_moved_past_the_drop_word(self):
        assert SHM_DROP_OFFSET + 4 <= SHM_METADATA_SIZE

    def test_inprocess_table_reset_cannot_reach_the_counter(self):
        """inprocess.reset_bitmap() memsets from SHM_METADATA_SIZE. With the
        counter below that offset, the clobber that once destroyed the whole
        diag word every execution is unreachable by construction."""
        import ctypes

        cov = ShmCoverage(size=256)
        try:
            for _ in range(7):
                cov._note_drop()
            ctypes.memset(cov._ptr + SHM_METADATA_SIZE, 0, cov.table_bytes)
            assert cov.read_dropped_edges() == 7
        finally:
            cov.cleanup()
