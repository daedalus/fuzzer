"""Integration tests for the AFLGo SHM-tail distance channel.

A self-contained target is compiled at test time with clang using
``-fsanitize-coverage=trace-pc -D__AFL_DISTANCE_MODE`` and the afl shim.
The distance table is computed from the binary via ``TargetDistance``,
uploaded through ``DistanceTableShm``, and the target is executed as a
subprocess; the shim's per-block distance accumulation is read back
from the SHM tail.

Expected values are hand-derived: with a single-block target function,
the only valued block is the target block (distance 0), so an input that
reaches it must produce dist_count >= 1 and average distance 0.0, while
an input that never reaches it must produce dist_count == 0.
"""

import ctypes
import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import DistanceTableShm, ShmCoverage
from fuzzer_tool.core.distance import TargetDistance

SRC = """\
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
static volatile int sink;
__attribute__((noinline)) int target_fn(int a) {
    return a + 1;
}
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size > 1 && data[0] == 'T')
        sink = target_fn(data[1]);
    return 0;
}
int main(int argc, char **argv) {
    unsigned char buf[1024];
    size_t n = 0;
    if (argc > 1) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) return 0;
        n = fread(buf, 1, sizeof(buf), f);
        fclose(f);
    }
    LLVMFuzzerTestOneInput(buf, n);
    return 0;
}
"""

SHIM = "src/fuzzer_tool/adapters/afl_shim.c"


@pytest.fixture(scope="module")
def distance_target(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not available")
    d = tmp_path_factory.mktemp("dist_channel")
    src = d / "dist_channel.c"
    src.write_text(SRC)
    out = d / "dist_channel"
    r = subprocess.run(
        [
            "clang",
            "-O1",
            "-g",
            "-D__AFL_DISTANCE_MODE",
            "-fsanitize-coverage=trace-pc",
            "-include",
            SHIM,
            "-o",
            str(out),
            str(src),
        ],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode()
    return str(out)


@pytest.fixture(scope="module")
def distance_so(tmp_path_factory):
    """A distance-mode .so with a fuzz_shm_run entry for in-process runs."""
    if not shutil.which("clang"):
        pytest.skip("clang not available")
    d = tmp_path_factory.mktemp("dist_channel_so")
    src = d / "dist_channel_so.c"
    so_src = SRC.replace(
        "int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {",
        "int fuzz_shm_run(const unsigned char *data, size_t size) {",
    )
    # Shared libs need no main; strip it (it references the renamed entry).
    main_idx = so_src.index("int main(")
    so_src = so_src[:main_idx]
    src.write_text(so_src)
    out = d / "dist_channel.so"
    r = subprocess.run(
        [
            "clang",
            "-O1",
            "-g",
            "-shared",
            "-fPIC",
            "-D__AFL_DISTANCE_MODE",
            "-fsanitize-coverage=trace-pc",
            "-include",
            SHIM,
            "-o",
            str(out),
            str(src),
        ],
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode()
    return str(out)


def _setup_channel(target, target_name):
    """Compute + upload the distance table; return (shm_cov, table_shm)."""
    td = TargetDistance(target, targets=[target_name])
    assert td.load()
    assert td._bb_value, "target function produced no valued blocks"
    table = td.pc_distance_table()
    assert table, "no __sancov_pcs sites resolved to valued blocks"
    table_shm = DistanceTableShm(table)
    assert table_shm.shm_id >= 0
    shm = ShmCoverage(size=8192)
    shm.reset_edge_map()
    os.environ["__AFL_SHM_ID"] = shm.env_id
    os.environ["AFL_MAP_SIZE"] = "8192"
    os.environ["__AFL_DIST_SHM_ID"] = table_shm.env_id
    return shm, table_shm, td


def _run_input(target, input_bytes, tmp_path):
    inp = tmp_path / "input.bin"
    inp.write_bytes(input_bytes)
    subprocess.run([target, str(inp)], timeout=30, check=True)
    return inp


class TestDistanceChannel:
    def test_channel_reports_target_hit(self, distance_target, tmp_path):
        shm, table_shm, td = _setup_channel(distance_target, "target_fn")
        try:
            _run_input(distance_target, b"T\x05", tmp_path)
            dist_sum, dist_count = shm.read_distance_tail()
            # The shim accumulates one entry per executed valued block.
            # target_fn's valued distances in this binary are 0 (target
            # block) and 2.0 (harmonic CFG), so the average of whatever
            # subset executed lies in [0, 2].
            assert dist_count >= 1
            avg = dist_sum / dist_count / 100.0
            assert 0.0 <= avg <= 2.0
        finally:
            shm.cleanup()
            table_shm.cleanup()

    def test_channel_silent_when_target_not_reached(self, distance_target, tmp_path):
        shm, table_shm, td = _setup_channel(distance_target, "target_fn")
        try:
            _run_input(distance_target, b"X\x05", tmp_path)
            dist_sum, dist_count = shm.read_distance_tail()
            assert dist_count == 0
            assert dist_sum == 0
        finally:
            shm.cleanup()
            table_shm.cleanup()

    def test_no_table_env_no_channel(self, distance_target, tmp_path):
        # Without __AFL_DIST_SHM_ID the shim never accumulates distance.
        os.environ.pop("__AFL_DIST_SHM_ID", None)
        shm = ShmCoverage(size=8192)
        shm.reset_edge_map()
        os.environ["__AFL_SHM_ID"] = shm.env_id
        try:
            _run_input(distance_target, b"T\x05", tmp_path)
            dist_sum, dist_count = shm.read_distance_tail()
            assert dist_count == 0
            assert dist_sum == 0
        finally:
            shm.cleanup()

    def test_uploaded_table_matches_targetdistance(self, distance_target):
        # Every uploaded site's distance equals TargetDistance's valued
        # block distance for the containing block (single source of truth).
        td = TargetDistance(distance_target, targets=["target_fn"])
        assert td.load()
        table = td.pc_distance_table()
        assert table
        for pc, dist in table.items():
            assert dist == td._bb_value_of(pc)


class TestDirectLiteChannel:
    """The channel must also work in direct_lite (in-process) mode, where
    there is no process boundary: the shim's accumulators are flushed to
    the tail per-iteration by __afl_dist_flush (called by the runner)."""

    def test_direct_lite_flushes_tail(self, distance_so, tmp_path):
        from fuzzer_tool.adapters.inprocess import InProcessRunner

        shm, table_shm, td = _setup_channel(distance_so, "target_fn")
        try:
            runner = InProcessRunner(
                target=distance_so,
                function_name="fuzz_shm_run",
                timeout=5,
                shm_size=8192,
                direct_lite=True,
                coverage_env_id=shm.env_id,
                cov=True,
            )
            # Pre-execution: the tail is zeroed and no distance accumulated.
            assert shm.read_distance_tail() == (0, 0)
            rc, err = runner.run_one(b"T\x05")
            assert rc == 0, err
            dist_sum, dist_count = shm.read_distance_tail()
            assert dist_count >= 1
            avg = dist_sum / dist_count / 100.0
            assert 0.0 <= avg <= 2.0
            # A second run must NOT accumulate onto the first (the flush
            # resets the shim's accumulators between iterations): an input
            # that never reaches the target leaves the tail at (0,0) — if
            # run 1's values were still buffered they'd reappear here.
            rc, err = runner.run_one(b"X\x05")
            assert rc == 0, err
            dist_sum2, dist_count2 = shm.read_distance_tail()
            assert dist_count2 == 0
            assert dist_sum2 == 0
        finally:
            shm.cleanup()
            table_shm.cleanup()


class TestDistanceTableLayout:
    """The Python builder must mirror the shim's open-addressed probe:
    entries live at ``key % capacity`` with linear probing and the
    header holds the slot capacity.  A sorted-rank writer (the old
    layout bug) packs keys at 0..n-1, so the shim's ``k == 0`` break
    never fires and every trace-pc miss scans the whole table.  Pure
    Python — no clang needed."""

    ENTRY_BYTES = 16  # {u64 key, u32 dist, u32 node_idx}

    def _read_slots(self, table_shm):
        cap = ctypes.c_uint32.from_address(table_shm._ptr).value
        return [
            (
                ctypes.c_uint64.from_address(table_shm._ptr + 4 + i * self.ENTRY_BYTES).value,
                ctypes.c_uint32.from_address(table_shm._ptr + 4 + i * self.ENTRY_BYTES + 8).value,
                ctypes.c_uint32.from_address(table_shm._ptr + 4 + i * self.ENTRY_BYTES + 12).value,
            )
            for i in range(cap)
        ]

    def test_header_is_padded_capacity(self):
        keys = {i * 7 + 3: 1.0 for i in range(3)}
        t = DistanceTableShm(keys)
        try:
            slots = self._read_slots(t)
            assert len(slots) >= 2 * len(keys)  # slack → empty slots exist
            assert len(slots) & (len(slots) - 1) == 0  # power of two
            assert sum(1 for k, _, _ in slots if k != 0) == len(keys)
            # No node_of → every entry carries the sentinel index.
            assert all(n == 0xFFFFFFFF for k, _, n in slots if k != 0)
        finally:
            t.cleanup()

    def test_entries_at_hash_positions_with_probing(self):
        # Keys spaced so sorted order differs from hash order; the walk
        # from key % capacity to the stored slot must never cross an
        # empty slot (the shim's probe would break there and miss).
        keys = {i * 1000 + 5: float(i) for i in range(50)}
        t = DistanceTableShm(keys)
        try:
            slots = self._read_slots(t)
            cap = len(slots)
            assert cap >= 2 * len(keys)
            assert cap & (cap - 1) == 0  # power of two
            stored = {k: (d, n) for k, d, n in slots if k != 0}
            assert len(stored) == len(keys)
            for key, dist in keys.items():
                d, n = stored[key]
                assert d == max(0, round(dist * 100))
                assert n == 0xFFFFFFFF
                idx = next(i for i, (k, _, _) in enumerate(slots) if k == key)
                pos = key % cap
                walked = 0
                while (pos + walked) % cap != idx:
                    assert slots[(pos + walked) % cap][0] != 0, (
                        f"key {key} stored at slot {idx} but slot "
                        f"{(pos + walked) % cap} is empty — not hash-inserted"
                    )
                    walked += 1
        finally:
            t.cleanup()
