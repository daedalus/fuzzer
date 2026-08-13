"""Tests for SHM resize preserving coverage state.

The invariant under test: ``edge_id = ctx ^ prev_loc ^ cur_loc`` carries no
``map_size`` term, so growing the table cannot change any edge's identity.
Only the starting probe position ``edge_id % map_size`` moves, and no position
is ever persisted -- ``reset_edge_map()`` memsets the table before every
execution regardless.

The regression these guard against: resize used to wipe every accumulated
statistic, so the next execution re-reported all known edges as new. That
zeroed the displayed coverage, reset ``execs_since_edge`` (defeating the very
stall detector that had triggered the resize), and poisoned the rarity
schedule and operator attribution.
"""

import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.process import disable_aslr
from fuzzer_tool.adapters.shm import ShmCoverage
from fuzzer_tool.core.edge_tracker import EdgeTracker

_DRIVER = """
#include <stdlib.h>
int main(int argc, char **argv) {
    uint32_t n = (uint32_t)atoi(argv[1]);
    for (uint32_t g = 1; g <= n; g++) { uint32_t guard = g;
        __sanitizer_cov_trace_pc_guard(&guard); }
    return 0;
}
"""

SHIM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "fuzzer_tool",
    "adapters",
    "afl_shim.c",
)

needs_cc = pytest.mark.skipif(shutil.which("gcc") is None, reason="no C compiler")


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    """Context-free and context-sensitive drivers."""
    if shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    # Context IDs derive from runtime return addresses, so they are only
    # stable across processes with ASLR off. Without this the ctx=8 case
    # would fail for an unrelated reason.
    disable_aslr()
    d = tmp_path_factory.mktemp("resize")
    src = d / "drv.c"
    src.write_text(_DRIVER)
    out = {}
    for bits, flags in {
        0: [],
        8: ["-D__AFL_CTX_SENSITIVE=1", "-fno-omit-frame-pointer"],
    }.items():
        exe = d / f"drv_{bits}"
        r = subprocess.run(
            ["gcc", "-O1", "-g", *flags, "-include", SHIM, "-o", str(exe), str(src)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            pytest.skip(f"shim build failed at ctx={bits}: {r.stderr[:300]}")
        out[bits] = str(exe)
    return out


def _exec(shm, exe, n):
    env = {**os.environ, "__AFL_SHM_ID": shm.env_id, "AFL_MAP_SIZE": str(shm.num_entries)}
    shm.reset_edge_map()
    subprocess.run([exe, str(n)], env=env, capture_output=True)
    return shm.is_new_coverage_with_edges()


class TestShmResizePreservesEdgeIdentity:
    @needs_cc
    @pytest.mark.parametrize("ctx_bits", [0, 8])
    def test_replaying_the_same_input_is_not_new_coverage(self, drivers, ctx_bits):
        """The headline regression: after a resize, known edges stayed known."""
        shm = ShmCoverage(size=4096)
        try:
            had_new, _ = _exec(shm, drivers[ctx_bits], 1500)
            assert had_new, "first execution should be new coverage"
            cumulative = shm.cumulative_edges
            seen = len(shm._seen_edge_ids)
            assert seen > 500, "need a meaningful edge set for this to prove anything"

            shm.resize(16384)

            had_new_after, _ = _exec(shm, drivers[ctx_bits], 1500)
            assert not had_new_after, "identical input re-reported as new coverage"
            assert shm.cumulative_edges == cumulative
            assert len(shm._seen_edge_ids) == seen
        finally:
            shm.cleanup()

    @needs_cc
    def test_edge_ids_are_bit_identical_across_resize(self, drivers):
        shm_a, shm_b = ShmCoverage(size=4096), ShmCoverage(size=65536)
        try:
            _exec(shm_a, drivers[0], 1500)
            _exec(shm_b, drivers[0], 1500)
            assert set(shm_a.get_edge_ids()) == set(shm_b.get_edge_ids())
        finally:
            shm_a.cleanup()
            shm_b.cleanup()

    @needs_cc
    def test_resize_preserves_the_header(self, drivers):
        """Context width must survive; the table itself is scratch."""
        shm = ShmCoverage(size=4096)
        try:
            _exec(shm, drivers[8], 100)
            assert shm.read_ctx_bits() == 8
            shm.resize(16384)
            assert shm.read_ctx_bits() == 8
        finally:
            shm.cleanup()

    @needs_cc
    def test_no_duplicate_slots_for_one_edge_after_resize(self, drivers):
        """resize() copies the header only.

        Copying the old table would place entries at positions computed under
        the old modulus, so the shim would miss them when probing and claim a
        second slot for an edge already present.
        """
        shm = ShmCoverage(size=4096)
        try:
            _exec(shm, drivers[0], 1500)
            shm.resize(16384)
            _exec(shm, drivers[0], 1500)
            import numpy as np

            arr = np.frombuffer(
                shm._map,
                dtype=np.dtype([("edge_id", "<u4"), ("count", "<u4")]),
                count=shm.num_entries,
            )
            raw = [int(x) for x in arr["edge_id"] if x != 0]
            assert raw, "table should not be empty"
            assert len(raw) == len(set(raw)), "an edge occupies more than one slot"
        finally:
            shm.cleanup()


class TestEdgeTrackerOnResize:
    def _tracker_with(self, keys, as_bytes=False):
        t = EdgeTracker(map_size=8192)
        if as_bytes:
            bitmap = bytearray(8192)
            for k in keys:
                bitmap[k] = 1
            t.record_edges("seed", bytes(bitmap))
        else:
            t.record_edges("seed", set(keys))
        return t

    def test_edge_id_keys_survive(self):
        t = self._tracker_with([10, 20, 30, 40])
        t.on_resize(65536)
        assert t.map_size == 65536
        assert t.cumulative_edges == {10, 20, 30, 40}
        assert len(t._global_edge_hits) == 4
        assert t.seed_edges["seed"] == {10, 20, 30, 40}

    def test_position_keys_are_cleared(self):
        """Byte-bitmap input really is slot-indexed, so it must still wipe."""
        t = self._tracker_with([10, 20, 30, 40], as_bytes=True)
        assert len(t._global_edge_hits) == 4
        t.on_resize(65536)
        assert t.map_size == 65536
        assert t._global_edge_hits == {}
        assert t.cumulative_edges == set()

    def test_key_space_is_recorded(self):
        assert self._tracker_with([1, 2])._key_space == "edge_id"
        assert self._tracker_with([1, 2], as_bytes=True)._key_space == "position"

    def test_mixing_key_spaces_warns(self, caplog):
        """Both spaces share the same dicts; mixing corrupts per-edge stats."""
        t = self._tracker_with([1, 2])
        bitmap = bytearray(8192)
        bitmap[5] = 1
        with caplog.at_level("WARNING"):
            t.record_edges("s2", bytes(bitmap))
        assert t._key_space == "mixed"
        assert any("mixed" in r.message or "slot indices" in r.message for r in caplog.records)

    def test_mixed_does_not_wipe(self):
        """'mixed' is already corrupt; wiping would not make it less so, and
        would additionally discard the valid edge-ID half."""
        t = self._tracker_with([1, 2])
        bitmap = bytearray(8192)
        bitmap[5] = 1
        t.record_edges("s2", bytes(bitmap))
        t.on_resize(65536)
        assert t._global_edge_hits != {}

    def test_deprecated_alias_keeps_map_size(self):
        t = self._tracker_with([1, 2])
        t.reset_after_resize()
        assert t.map_size == 8192
        assert t.cumulative_edges == {1, 2}
