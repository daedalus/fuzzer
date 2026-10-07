"""Read side of the touched-slot bitmap: _scan decodes bits instead of walking the table.

Contract being pinned: with touched_scan on and a shim that announced the
bitmap, _scan returns exactly what the full-table scan returns, in the same
order, for any table the shim could have produced. Stale bits are harmless
(the entry filter still applies); a live entry with no bit is invisible, which
is why the bitmap is cleared at the one place entries go stale
(reset_edge_map) and filled by the one writer that creates live entries.
"""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import (
    _ENTRY_DTYPE,
    SHM_GENERATION_OFFSET,
    SHM_TOUCHED_HEADER,
    SHM_TOUCHED_MAGIC,
    ShmCoverage,
)
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env

GEN = 7


def _cov(size: int, **kw) -> ShmCoverage:
    cov = ShmCoverage(size=size, touched_scan=True, **kw)
    ctypes.memmove(
        cov._touched_ptr, SHM_TOUCHED_MAGIC, SHM_TOUCHED_HEADER
    )  # the shim's announcement
    ctypes.c_uint32.from_address(cov._ptr + SHM_GENERATION_OFFSET).value = GEN
    return cov


def _table(cov: ShmCoverage) -> np.ndarray:
    return np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)


def _set_bit(cov: ShmCoverage, slot: int) -> None:
    words = cov._touched_view()
    words[slot >> 6] |= np.uint64(1) << np.uint64(slot & 63)


def _plant(
    cov: ShmCoverage, slot: int, edge: int, count: int = 1, gen: int = GEN, bit: bool = True
) -> None:
    t = _table(cov)
    t["edge_id"][slot] = edge
    t["count"][slot] = (gen << 24) | count
    if bit:
        _set_bit(cov, slot)


def _both(cov: ShmCoverage, need_counts: bool = True):
    cov.touched_scan = True
    cov._scan_memo = None
    fast = cov._scan(need_counts)
    cov.touched_scan = False
    cov._scan_memo = None
    full = cov._scan(need_counts)
    cov.touched_scan = True
    cov._scan_memo = None
    return fast, full


class TestSameAnswerAsTheFullScan:
    @pytest.mark.parametrize("size", [64, 1000, 8192, 65536])
    @pytest.mark.parametrize("seed", range(5))
    def test_random_tables(self, size, seed):
        rng = np.random.default_rng(seed)
        cov = _cov(size)
        try:
            n = int(rng.integers(0, min(size, 2500)))
            slots = rng.choice(size, size=n, replace=False)
            for i, s in enumerate(slots):
                stale = i % 4 == 0  # registry entries from older generations
                _plant(
                    cov,
                    int(s),
                    int(rng.integers(1, 2**32 - 1)),
                    int(rng.integers(1, 255)),
                    gen=3 if stale else GEN,
                    bit=not stale,
                )
            (fi, fc), (ui, uc) = _both(cov)
            assert np.array_equal(fi, ui)
            assert np.array_equal(fc, uc)
        finally:
            cov.cleanup()

    def test_ids_only_request_returns_no_counts(self):
        cov = _cov(256)
        try:
            _plant(cov, 5, 99)
            (fi, fc), (ui, uc) = _both(cov, need_counts=False)
            assert fc is None and uc is None
            assert fi.tolist() == ui.tolist() == [99]
        finally:
            cov.cleanup()

    def test_slots_straddling_word_and_byte_boundaries(self):
        cov = _cov(1000)
        try:
            slots = [0, 7, 8, 63, 64, 65, 127, 128, 999]
            for s in slots:
                _plant(cov, s, 1000 + s)
            (fi, _), (ui, _) = _both(cov)
            assert fi.tolist() == ui.tolist() == [1000 + s for s in slots]
        finally:
            cov.cleanup()


class TestHarmlessAndPinnedEdges:
    def test_a_stale_bit_over_an_empty_slot_is_ignored(self):
        cov = _cov(256)
        try:
            _set_bit(cov, 40)  # e.g. a bit left by an execution whose table was since zeroed
            ids, counts = cov._scan(True)
            assert ids.size == 0 and counts.size == 0
        finally:
            cov.cleanup()

    def test_a_stale_generation_entry_with_a_bit_is_ignored(self):
        cov = _cov(256)
        try:
            _plant(cov, 40, 123, gen=GEN - 1, bit=True)
            assert cov._scan(True)[0].size == 0
        finally:
            cov.cleanup()

    def test_padding_bits_beyond_the_table_are_ignored(self):
        cov = _cov(1000)  # last word covers slots 960..1023; 1000..1023 do not exist
        try:
            _plant(cov, 3, 55)
            _set_bit(cov, 1005)
            assert cov._scan(False)[0].tolist() == [55]
        finally:
            cov.cleanup()

    def test_a_live_entry_without_a_bit_is_invisible(self):
        """The contract's other half: only the shim may create live entries."""
        cov = _cov(256)
        try:
            _plant(cov, 9, 77, bit=False)
            assert cov._scan(False)[0].size == 0
            cov.touched_scan = False
            cov._scan_memo = None
            assert cov._scan(False)[0].tolist() == [77]
        finally:
            cov.cleanup()


class TestWhenItStepsAside:
    def test_unannounced_bitmap_falls_back_to_the_full_scan(self):
        cov = ShmCoverage(size=256, touched_scan=True)  # no magic: an older shim
        try:
            ctypes.c_uint32.from_address(cov._ptr + SHM_GENERATION_OFFSET).value = GEN
            _plant(cov, 9, 77, bit=False)
            assert cov._scan(False)[0].tolist() == [77]
        finally:
            cov.cleanup()

    def test_scan_mode_without_the_flag_never_reads_the_bitmap(self):
        cov = ShmCoverage(size=256, touched_bitmap=True)  # calibration-only bitmap
        try:
            ctypes.memmove(cov._touched_ptr, SHM_TOUCHED_MAGIC, SHM_TOUCHED_HEADER)
            ctypes.c_uint32.from_address(cov._ptr + SHM_GENERATION_OFFSET).value = GEN
            _plant(cov, 9, 77, bit=False)
            assert not cov.touched_scan
            assert cov._scan(False)[0].tolist() == [77]
        finally:
            cov.cleanup()

    def test_resize_drops_the_announcement_and_the_scan_falls_back(self):
        cov = _cov(256)
        try:
            cov.resize(1024)
            ctypes.c_uint32.from_address(cov._ptr + SHM_GENERATION_OFFSET).value = GEN
            _plant(cov, 9, 77, bit=False)
            assert cov._scan(False)[0].tolist() == [77]
        finally:
            cov.cleanup()

    def test_scan_flag_allocates_the_region(self):
        cov = ShmCoverage(size=128, touched_scan=True)
        try:
            assert cov.touched_enabled and cov.touched_scan
        finally:
            cov.cleanup()


class TestReset:
    def test_reset_edge_map_clears_the_bits_in_scan_mode(self):
        cov = _cov(256)
        try:
            _plant(cov, 9, 77)
            cov.reset_edge_map()
            assert not cov.touched_snapshot().any()
            assert cov.touched_supported  # the announcement survives
        finally:
            cov.cleanup()

    def test_reset_edge_map_leaves_the_bits_alone_otherwise(self):
        """Calibration owns the bitmap there; the per-exec reset must not touch it."""
        cov = ShmCoverage(size=256, touched_bitmap=True)
        try:
            _set_bit(cov, 9)
            cov.reset_edge_map()
            assert cov.touched_snapshot().any()
        finally:
            cov.cleanup()

    def test_memo_still_serves_a_second_call(self):
        cov = _cov(256)
        try:
            _plant(cov, 9, 77)
            ctypes.c_uint64.from_address(cov._ptr + 8).value = 1  # nonzero path hash: memoizable
            first = cov._scan(True)
            assert cov._scan(True)[0] is first[0]
        finally:
            cov.cleanup()


_LOC_DRIVER = """
#include <stdlib.h>
int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
        __afl_map_loc((uint32_t)strtoul(argv[i], NULL, 10));
    return 0;
}
"""
SIZE = 4096


@pytest.fixture(scope="module")
def loc_target(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("touched_scan"), _LOC_DRIVER, "-D__AFL_CTX_SENSITIVE=0")


def _fire(exe, cov, blocks):
    cov.reset_edge_map()
    env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(cov.num_entries))
    r = subprocess.run([str(exe), *map(str, blocks)], env=env, capture_output=True, timeout=30)
    assert r.returncode == 0, r.stderr


@requires_clang
class TestAgainstTheRealShim:
    def test_matches_the_full_scan_across_generations(self, loc_target):
        cov = ShmCoverage(size=SIZE, touched_scan=True)
        try:
            rng = np.random.default_rng(1)
            for _ in range(6):  # each round: reset (gen bump + bit clear), run, compare
                blocks = rng.choice(0x7FFF, size=int(rng.integers(1, 600)), replace=False) + 1
                _fire(loc_target, cov, blocks.tolist())
                assert cov.touched_supported
                (fi, fc), (ui, uc) = _both(cov)
                assert fi.size > 0
                assert np.array_equal(fi, ui) and np.array_equal(fc, uc)
        finally:
            cov.cleanup()

    def test_hit_counts_match_even_for_repeated_blocks(self, loc_target):
        cov = ShmCoverage(size=SIZE, touched_scan=True)
        try:
            _fire(loc_target, cov, [11, 22, 11, 22, 11, 33])
            (fi, fc), (ui, uc) = _both(cov)
            assert np.array_equal(fi, ui) and np.array_equal(fc, uc)
            assert int((fc & 0xFFFFFF).max()) >= 2
        finally:
            cov.cleanup()

    def test_edges_of_an_earlier_generation_do_not_leak_into_the_next_scan(self, loc_target):
        cov = ShmCoverage(size=SIZE, touched_scan=True)
        try:
            _fire(loc_target, cov, [1, 2, 3, 4, 5])
            first = set(cov._scan(False)[0].tolist())
            _fire(loc_target, cov, [100, 200])
            second = set(cov._scan(False)[0].tolist())
            assert first and second and not (first & second)
        finally:
            cov.cleanup()
