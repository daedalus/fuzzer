"""Tests for the K-Scheduler node-visit bitmap channel (W2).

Two pieces under test:

``DistanceTableShm`` grows a per-entry ``node_idx`` column (stride
12 → 16 bytes): the shim's ``__sanitizer_cov_trace_pc`` probe matches a
key and sets ``bitmap[node_idx >> 3] |= 1 << (node_idx & 7)`` in a second
segment uploaded via ``__AFL_NODE_BITMAP_ID``. Entries without a mapping
carry the 0xFFFFFFFF sentinel, which must never address the bitmap.

``NodeBitmapShm`` is that segment: ``u32 size_bytes`` header + payload.
The shim writes eagerly during execution, so Python reads-and-clears after
each run — no destructor writer, unlike the lazy distance tail.
"""

import ctypes
import os
import shutil
import subprocess

import pytest

from fuzzer_tool.adapters.shm import DistanceTableShm, NodeBitmapShm

NODE_IDX_NONE = 0xFFFFFFFF


class TestDistanceTableNodeColumn:
    """Pure-Python layout mirrors of the widened table."""

    ENTRY_BYTES = 16  # {u64 key, u32 dist, u32 node_idx}

    def _slots(self, t):
        cap = ctypes.c_uint32.from_address(t._ptr).value
        out = []
        for i in range(cap):
            base = t._ptr + 4 + i * self.ENTRY_BYTES
            out.append(
                (
                    ctypes.c_uint64.from_address(base).value,
                    ctypes.c_uint32.from_address(base + 8).value,
                    ctypes.c_uint32.from_address(base + 12).value,
                )
            )
        return cap, out

    def test_stride_is_16_and_sentinel_written_without_node_of(self):
        keys = {i * 7 + 3: float(i) for i in range(4)}
        t = DistanceTableShm(keys)
        try:
            cap, slots = self._slots(t)
            assert t.shm_bytes == 4 + cap * 16
            stored = {k: (d, n) for k, d, n in slots if k != 0}
            assert len(stored) == len(keys)
            for key, dist in keys.items():
                d, n = stored[key]
                assert d == max(0, round(dist * 100))
                assert n == NODE_IDX_NONE
        finally:
            t.cleanup()

    def test_node_of_maps_indices(self):
        keys = {i * 11 + 1: 1.0 for i in range(5)}
        node_of = {k: i for i, k in enumerate(sorted(keys))}
        t = DistanceTableShm(keys, node_of=node_of)
        try:
            _, slots = self._slots(t)
            stored = {k: n for k, _, n in slots if k != 0}
            assert stored == node_of
        finally:
            t.cleanup()

    def test_partial_node_of_sends_sentinel_for_the_rest(self):
        keys = {i * 13 + 2: 1.0 for i in range(4)}
        covered = {k: i for i, k in enumerate(list(keys)[:2])}
        t = DistanceTableShm(keys, node_of=covered)
        try:
            _, slots = self._slots(t)
            stored = {k: n for k, _, n in slots if k != 0}
            assert stored == {**covered, **{k: NODE_IDX_NONE for k in keys if k not in covered}}
        finally:
            t.cleanup()


class TestNodeBitmapSegment:
    def test_layout_header_plus_payload(self):
        b = NodeBitmapShm(num_nodes=100)
        try:
            assert b.size_bytes == 13  # ceil(100/8)
            assert ctypes.c_uint32.from_address(b._ptr).value == 13
        finally:
            b.cleanup()

    def test_read_and_clear(self):
        b = NodeBitmapShm(num_nodes=64)
        try:
            ctypes.c_uint8.from_address(b._ptr + 4 + 3).value = 0xFF
            buf = b.read_and_clear()
            assert buf[3] == 0xFF
            assert b.read_and_clear() == bytes(8), "payload must be zeroed after read"
        finally:
            b.cleanup()

    def test_cleanup_detaches_and_removes(self):
        b = NodeBitmapShm(num_nodes=8)
        b.cleanup()
        assert b._ptr is None
        assert b.shm_id == -1


# ── live channel (clang) ─────────────────────────────────────────────

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
def node_target(tmp_path_factory):
    if not shutil.which("clang"):
        pytest.skip("clang not available")
    d = tmp_path_factory.mktemp("node_channel")
    src = d / "node_channel.c"
    src.write_text(SRC)
    out = d / "node_channel"
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


def _popcount(buf):
    return bin(int.from_bytes(buf, "little")).count("1")


def _setup(target, input_bytes, tmp_path, cover_all=True):
    """Upload table + node bitmap; run one input; return bitmap bytes."""
    from fuzzer_tool.adapters.shm import ShmCoverage
    from fuzzer_tool.core.distance import TargetDistance

    td = TargetDistance(target, targets=["target_fn"])
    assert td.load()
    table = td.pc_distance_table()
    assert table
    node_of = {k: i for i, k in enumerate(sorted(table))} if cover_all else {}
    table_shm = DistanceTableShm(table, node_of=node_of or None)
    bmp = NodeBitmapShm(num_nodes=max(1, len(table)))
    shm = ShmCoverage(size=8192)
    shm.reset_edge_map()
    os.environ["__AFL_SHM_ID"] = shm.env_id
    os.environ["AFL_MAP_SIZE"] = "8192"
    os.environ["__AFL_DIST_SHM_ID"] = table_shm.env_id
    if bmp.shm_id >= 0:
        os.environ["__AFL_NODE_BITMAP_ID"] = bmp.env_id
    inp = tmp_path / "input.bin"
    inp.write_bytes(input_bytes)
    try:
        subprocess.run([target, str(inp)], timeout=30, check=True)
    finally:
        data = bmp.read_and_clear()
        shm.cleanup()
        table_shm.cleanup()
        os.environ.pop("__AFL_NODE_BITMAP_ID", None)
        bmp.cleanup()
    return data, len(table)


class TestNodeBitmapChannel:
    def test_hit_sets_bits_within_uploaded_range(self, node_target, tmp_path):
        data, n = _setup(node_target, b"T\x05", tmp_path)
        assert _popcount(data) >= 1
        assert _popcount(data) <= n

    def test_miss_sets_no_bits(self, node_target, tmp_path):
        data, _ = _setup(node_target, b"X\x05", tmp_path)
        assert _popcount(data) == 0

    def test_unmapped_entries_never_set_bits(self, node_target, tmp_path):
        """Sentinel node_idx must be rejected by the bounds check even on
        a full hit."""
        from fuzzer_tool.adapters.shm import ShmCoverage
        from fuzzer_tool.core.distance import TargetDistance

        td = TargetDistance(node_target, targets=["target_fn"])
        assert td.load()
        table = td.pc_distance_table()
        assert table
        table_shm = DistanceTableShm(table)  # no node_of → all sentinels
        bmp = NodeBitmapShm(num_nodes=len(table))
        shm = ShmCoverage(size=8192)
        shm.reset_edge_map()
        os.environ["__AFL_SHM_ID"] = shm.env_id
        os.environ["AFL_MAP_SIZE"] = "8192"
        os.environ["__AFL_DIST_SHM_ID"] = table_shm.env_id
        os.environ["__AFL_NODE_BITMAP_ID"] = bmp.env_id
        inp = tmp_path / "in.bin"
        inp.write_bytes(b"T\x05")
        try:
            subprocess.run([node_target, str(inp)], timeout=30, check=True)
            assert _popcount(bmp.read_and_clear()) == 0
        finally:
            shm.cleanup()
            table_shm.cleanup()
            os.environ.pop("__AFL_NODE_BITMAP_ID", None)
            bmp.cleanup()

    def test_no_bitmap_env_is_inert(self, node_target, tmp_path):
        """Distance channel alone must keep working with no bitmap attached."""
        from fuzzer_tool.adapters.shm import ShmCoverage
        from fuzzer_tool.core.distance import TargetDistance

        td = TargetDistance(node_target, targets=["target_fn"])
        assert td.load()
        table = td.pc_distance_table()
        table_shm = DistanceTableShm(table)
        shm = ShmCoverage(size=8192)
        shm.reset_edge_map()
        os.environ["__AFL_SHM_ID"] = shm.env_id
        os.environ["AFL_MAP_SIZE"] = "8192"
        os.environ["__AFL_DIST_SHM_ID"] = table_shm.env_id
        os.environ.pop("__AFL_NODE_BITMAP_ID", None)
        inp = tmp_path / "in.bin"
        inp.write_bytes(b"T\x05")
        try:
            subprocess.run([node_target, str(inp)], timeout=30, check=True)
            dist_sum, dist_count = shm.read_distance_tail()
            assert dist_count >= 1, "distance channel must be unaffected"
        finally:
            shm.cleanup()
            table_shm.cleanup()
