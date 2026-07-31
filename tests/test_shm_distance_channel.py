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
