"""Tests for context width (__AFL_CTX_BITS) and coverage map sizing.

The shim tests compile afl_shim.c against a tiny driver that calls
__sanitizer_cov_trace_pc_guard directly. That avoids needing clang: gcc has
no -fsanitize-coverage=trace-pc-guard, but the shim's edge table, context
masking, and drop counter are all exercised by calling the callback by hand,
which is the code path under test anyway.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.elf import (
    MAP_SIZE_DEFAULT,
    MAP_SIZE_MAX,
    _map_size_max,
    _size_from_blocks,
    ctx_inflation_factor,
    detect_ctx_bits,
)

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
def built(tmp_path_factory):
    """Compile the driver at several context widths. Returns {bits: path}."""
    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("ctx")
    src = d / "drv.c"
    src.write_text(_DRIVER)
    out = {}
    configs = {
        0: [],
        8: ["-D__AFL_CTX_SENSITIVE=1", "-fno-omit-frame-pointer"],
        4: ["-D__AFL_CTX_SENSITIVE=1", "-D__AFL_CTX_BITS=4", "-fno-omit-frame-pointer"],
    }
    for bits, flags in configs.items():
        exe = d / f"drv_{bits}"
        r = subprocess.run(
            ["gcc", "-O1", "-g", *flags, "-include", SHIM, "-o", str(exe), str(src)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            pytest.skip(f"shim failed to build at ctx_bits={bits}: {r.stderr[:300]}")
        out[bits] = str(exe)
    return out


class TestShimBuildMatrix:
    @needs_cc
    def test_marker_symbol_encodes_width(self, built):
        for bits, path in built.items():
            syms = subprocess.run(["nm", path], capture_output=True, text=True).stdout
            assert f"__afl_ctx_bits_{bits}" in syms

    @needs_cc
    def test_ctx_bits_forced_to_zero_when_ctx_disabled(self, tmp_path):
        """-D__AFL_CTX_BITS=12 with CTX off must still report 0, not 12."""
        src = tmp_path / "d.c"
        src.write_text(_DRIVER)
        exe = tmp_path / "d"
        r = subprocess.run(
            [
                "gcc",
                "-O1",
                "-D__AFL_CTX_SENSITIVE=0",
                "-D__AFL_CTX_BITS=12",
                "-include",
                SHIM,
                "-o",
                str(exe),
                str(src),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        assert detect_ctx_bits(str(exe)) == 0

    @needs_cc
    def test_out_of_range_width_is_a_build_error(self, tmp_path):
        src = tmp_path / "d.c"
        src.write_text(_DRIVER)
        r = subprocess.run(
            [
                "gcc",
                "-fsyntax-only",
                "-D__AFL_CTX_SENSITIVE=1",
                "-D__AFL_CTX_BITS=40",
                "-include",
                SHIM,
                str(src),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
        assert "__AFL_CTX_BITS must be in [0, 32]" in r.stderr


class TestStaticDetection:
    @needs_cc
    def test_reads_width_without_running_the_target(self, built):
        for bits, path in built.items():
            assert detect_ctx_bits(path) == bits

    def test_absent_marker_is_none_not_zero(self):
        """None means 'unknown shim', 0 means 'known context-free'."""
        assert detect_ctx_bits("/bin/true") is None

    def test_unreadable_path_does_not_raise(self):
        assert detect_ctx_bits("/nonexistent/binary") is None


class TestDropCounter:
    @needs_cc
    def test_counts_edges_lost_to_a_full_table(self, built):
        """The signal that occupancy alone cannot provide."""
        shm = ShmCoverage(size=1024)
        try:
            env = {**os.environ, "__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": "1024"}
            proc = subprocess.run([built[0], "4000"], env=env, capture_output=True)
            # Diagnostics, not decoration: this assertion has gone red once,
            # non-reproducibly, reading back a table the child should have
            # filled (see "Loose thread" in docs/edge-coverage-analysis.md).
            # Without the header and the child's status in the message there
            # is no way to tell a child that failed to attach from a parent
            # that raced the read, which is exactly what the last sighting
            # could not distinguish.
            state = (
                f"rc={proc.returncode} "
                f"edge_count={shm.read_edge_count()} "
                f"diag=0x{shm.read_diag():08x} "
                f"dropped={shm.read_dropped_edges()} "
                f"occupied={len(shm.get_edge_ids())} "
                f"stderr={proc.stderr[-200:]!r}"
            )
            assert len(shm.get_edge_ids()) == 1024, f"table should be full — {state}"
            assert shm.read_dropped_edges() > 0, f"full table must report drops — {state}"
        finally:
            shm.cleanup()

    @needs_cc
    def test_silent_when_the_table_has_room(self, built):
        shm = ShmCoverage(size=1024)
        try:
            env = {**os.environ, "__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": "1024"}
            subprocess.run([built[0], "400"], env=env, capture_output=True)
            assert 0 < len(shm.get_edge_ids()) < 1024
            assert shm.read_dropped_edges() == 0
        finally:
            shm.cleanup()

    @needs_cc
    def test_ctx_bits_published_to_header(self, built):
        for bits, path in built.items():
            shm = ShmCoverage(size=1024)
            try:
                env = {**os.environ, "__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": "1024"}
                subprocess.run([path, "100"], env=env, capture_output=True)
                assert shm.read_ctx_bits() == bits
            finally:
                shm.cleanup()

    @needs_cc
    def test_reset_diag_clears_drops_but_keeps_width(self, built):
        shm = ShmCoverage(size=1024)
        try:
            env = {**os.environ, "__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": "1024"}
            subprocess.run([built[8], "4000"], env=env, capture_output=True)
            assert shm.read_dropped_edges() > 0
            assert shm.read_ctx_bits() == 8
            shm.reset_diag()
            assert shm.read_dropped_edges() == 0
            assert shm.read_ctx_bits() == 8, "width must survive the reset"
        finally:
            shm.cleanup()


class TestMapSizing:
    def test_accounts_for_edges_per_block_and_load_factor(self):
        """Old sizing was next_pow2(blocks); it should now ask for ~4x that."""
        assert _size_from_blocks(5_000, 0) == 32_768  # was 8_192
        assert _size_from_blocks(20_000, 0) == 131_072  # was 32_768

    def test_context_builds_get_more_room(self):
        assert _size_from_blocks(1_000, 8) > _size_from_blocks(1_000, 0)

    def test_unknown_ctx_is_treated_as_context_free(self):
        assert _size_from_blocks(5_000, None) == _size_from_blocks(5_000, 0)

    def test_never_below_default_or_above_cap(self):
        assert _size_from_blocks(1, 0) == MAP_SIZE_DEFAULT
        assert _size_from_blocks(10_000_000, 32) == MAP_SIZE_MAX

    def test_result_is_a_power_of_two(self):
        for blocks in (1, 999, 5_000, 33_333, 250_000):
            n = _size_from_blocks(blocks, 0)
            assert n & (n - 1) == 0, n

    def test_inflation_is_bounded(self):
        """Sizing for the 2**bits ceiling would demand gigabytes."""
        assert ctx_inflation_factor(0) == 1.0
        assert ctx_inflation_factor(None) == 1.0
        assert ctx_inflation_factor(32) <= 16.0
        assert ctx_inflation_factor(8) >= ctx_inflation_factor(4)


class TestMapSizeMaxOverride:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AFL_MAP_SIZE_MAX", "1048576")
        assert _map_size_max() == 1_048_576

    def test_override_rounds_to_power_of_two(self, monkeypatch):
        monkeypatch.setenv("AFL_MAP_SIZE_MAX", "300000")
        assert _map_size_max() == 524_288

    def test_garbage_override_falls_back(self, monkeypatch):
        monkeypatch.setenv("AFL_MAP_SIZE_MAX", "not-a-number")
        assert _map_size_max() == MAP_SIZE_MAX

    def test_absurdly_small_override_is_floored(self, monkeypatch):
        monkeypatch.setenv("AFL_MAP_SIZE_MAX", "16")
        assert _map_size_max() == MAP_SIZE_DEFAULT


class TestRecommendedMapSize:
    def _tracker(self, n_edges, map_size):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        t = EdgeTracker(map_size=map_size)
        for i in range(n_edges):
            t._global_edge_hits[i] = 1
        return t

    def test_drops_trigger_resize_even_at_low_apparent_load(self):
        """The self-masking case: occupancy looks fine because the edges that
        would have raised it were the ones thrown away."""
        t = self._tracker(500, 8192)  # load 0.06 — far below the 0.7 threshold
        assert t.recommended_map_size(dropped_edges=0) == 0
        assert t.recommended_map_size(dropped_edges=50_000) > 8192

    def test_high_load_still_triggers_without_drops(self):
        t = self._tracker(7000, 8192)  # load 0.85
        assert t.recommended_map_size(dropped_edges=0) > 8192

    def test_no_resize_when_healthy(self):
        t = self._tracker(1000, 8192)  # load 0.12
        assert t.recommended_map_size(dropped_edges=0) == 0

    def test_sizes_from_observed_plus_dropped(self):
        """Drops are part of the true edge count, so they must widen the ask."""
        t = self._tracker(5000, 8192)
        with_drops = t.recommended_map_size(dropped_edges=20_000)
        without = self._tracker(5000, 8192).recommended_map_size(dropped_edges=1)
        assert with_drops > without

    def test_never_recommends_a_smaller_map(self):
        t = self._tracker(100, MAP_SIZE_MAX)
        assert t.recommended_map_size(dropped_edges=10**6) == 0


class TestEdgeIdZeroRegression:
    """edge_id == 0 must still be recorded, not treated as an empty slot."""

    _ZERO_DRIVER = """
    #include <stdlib.h>
    int main(int argc, char **argv) {
        /* Guard sequence [2, 1] produces edge_id = (2>>1)^1 = 1^1 = 0
         * on the second call when CTX is off. */
        uint32_t g0 = 2;
        __sanitizer_cov_trace_pc_guard(&g0);
        uint32_t g1 = 1;
        __sanitizer_cov_trace_pc_guard(&g1);
        return 0;
    }
    """

    @needs_cc
    def test_zero_edge_id_is_recorded_not_dropped(self, tmp_path):
        """A valid edge that hashes to 0 must survive the probe loop."""
        src = tmp_path / "zero_edge.c"
        src.write_text(self._ZERO_DRIVER)
        exe = tmp_path / "zero_edge"
        r = subprocess.run(
            ["gcc", "-O1", "-g", "-include", SHIM, "-o", str(exe), str(src)],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr

        shm = ShmCoverage(size=1024)
        try:
            env = {**os.environ, "__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": "1024"}
            subprocess.run([str(exe)], env=env, capture_output=True)
            edges = shm.get_edge_ids()
            # Without the fix the second edge hashes to 0 and is silently
            # dropped, so only 1 edge survives. With the fix both survive.
            assert len(edges) == 2, f"expected 2 edges, got {len(edges)}: {edges}"
        finally:
            shm.cleanup()


class TestAttachFailureIsLoud:
    """A failed attach must say so, not run the target and record nothing.

    All three early returns in `__afl_map_shm()` used to be silent, so a
    child that could not attach exited 0 with an empty stderr and left an
    all-zero header — indistinguishable from a parent that raced the read.
    That ambiguity is the "Loose thread" in docs/edge-coverage-analysis.md,
    unresolved across three sightings for exactly this reason.
    """

    @needs_cc
    def test_silent_when_not_under_the_fuzzer(self, built):
        """No __AFL_SHM_ID means standalone, which is not an error."""
        env = {k: v for k, v in os.environ.items() if k != "__AFL_SHM_ID"}
        r = subprocess.run([built[0], "10"], env=env, capture_output=True)
        assert r.returncode == 0
        assert b"__afl_shim" not in r.stderr

    @needs_cc
    def test_unparseable_id_is_reported(self, built):
        env = {**os.environ, "__AFL_SHM_ID": "0", "AFL_MAP_SIZE": "1024"}
        r = subprocess.run([built[0], "10"], env=env, capture_output=True)
        assert b"__afl_shim" in r.stderr
        assert b"not a valid segment id" in r.stderr

    @needs_cc
    def test_failed_shmat_is_reported_with_errno(self, built):
        """A stale segment id is the realistic form of this failure."""
        env = {**os.environ, "__AFL_SHM_ID": "999999", "AFL_MAP_SIZE": "1024"}
        r = subprocess.run([built[0], "10"], env=env, capture_output=True)
        assert b"shmat(999999) failed" in r.stderr

    @needs_cc
    def test_diagnostic_cannot_be_read_as_a_crash(self, built):
        """The message must not trip ExecutionRunner.is_crash()'s stderr scan."""
        env = {**os.environ, "__AFL_SHM_ID": "999999", "AFL_MAP_SIZE": "1024"}
        r = subprocess.run([built[0], "10"], env=env, capture_output=True)
        text = r.stderr.decode(errors="replace")
        for token in ("SIGSEGV", "SIGABRT", "SIGFPE", "SIGBUS", "Segmentation fault", "Aborted"):
            assert token not in text

    @needs_cc
    def test_successful_attach_stays_silent(self, built):
        shm = ShmCoverage(size=1024)
        try:
            env = {**os.environ, "__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": "1024"}
            r = subprocess.run([built[0], "400"], env=env, capture_output=True)
            assert b"__afl_shim" not in r.stderr
            assert len(shm.get_edge_ids()) > 0
        finally:
            shm.cleanup()
