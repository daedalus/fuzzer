"""Touched-slot bitmap: one bit per edge-table slot first touched this generation.

The region sits after the distance tail, so no existing offset moves and a
segment allocated without it behaves exactly as before. The shim announces
support by writing a magic word at attach; ShmCoverage.touched_supported is
that word, so an older shim (which never writes it) degrades to "unsupported"
rather than to a bitmap full of zeros that reads as "nothing fired".
"""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import (
    _ENTRY_DTYPE,
    SHM_METADATA_SIZE,
    SHM_TAIL_SIZE,
    SHM_TOUCHED_HEADER,
    SHM_TOUCHED_MAGIC,
    SIZEOF_ENTRY,
    ShmCoverage,
    home_slot,
    touched_region_bytes,
    unstable_slots,
)
from tests.conftest import requires_clang
from tests.test_regression_shim_audit import _build, _clean_env


def _bits(words: np.ndarray) -> set[int]:
    out: set[int] = set()
    for w in np.flatnonzero(words):
        v = int(words[w])
        out.update(int(w) * 64 + b for b in range(64) if v >> b & 1)
    return out


def _plant_magic(cov: ShmCoverage) -> None:
    """Stand in for the shim's attach-time announcement."""
    ctypes.memmove(cov._touched_ptr, SHM_TOUCHED_MAGIC, SHM_TOUCHED_HEADER)


class TestRegion:
    def test_absent_by_default_and_segment_size_unchanged(self):
        cov = ShmCoverage(size=1024)
        try:
            assert cov.shm_bytes == 1024 * SIZEOF_ENTRY + SHM_METADATA_SIZE + SHM_TAIL_SIZE
            assert not cov.touched_enabled
            assert not cov.touched_supported
        finally:
            cov.cleanup()

    def test_enabled_region_follows_the_tail(self):
        cov = ShmCoverage(size=1024, touched_bitmap=True)
        try:
            base = 1024 * SIZEOF_ENTRY + SHM_METADATA_SIZE + SHM_TAIL_SIZE
            assert cov.shm_bytes == base + touched_region_bytes(1024)
            assert cov._touched_ptr == cov._ptr + base
            assert cov.touched_enabled
        finally:
            cov.cleanup()

    @pytest.mark.parametrize(("size", "words"), [(64, 1), (65, 2), (1000, 16), (8192, 128)])
    def test_word_count_rounds_up(self, size, words):
        assert touched_region_bytes(size) == SHM_TOUCHED_HEADER + words * 8

    def test_unsupported_until_the_shim_announces_itself(self):
        cov = ShmCoverage(size=256, touched_bitmap=True)
        try:
            assert not cov.touched_supported  # fresh segment: zeros, no magic
            _plant_magic(cov)
            assert cov.touched_supported
        finally:
            cov.cleanup()

    def test_snapshot_is_a_copy_and_clear_zeroes(self):
        cov = ShmCoverage(size=256, touched_bitmap=True)
        try:
            _plant_magic(cov)
            words = (ctypes.c_uint64 * cov.touched_words).from_address(
                cov._touched_ptr + SHM_TOUCHED_HEADER
            )
            words[1] = 0b101
            snap = cov.touched_snapshot()
            assert _bits(snap) == {64, 66}
            snap[:] = 0
            assert words[1] == 0b101  # mutating the copy never reaches shared memory
            cov.touched_clear()
            assert not cov.touched_snapshot().any()
            assert cov.touched_supported  # clearing must not erase the announcement
        finally:
            cov.cleanup()

    def test_snapshot_refuses_when_not_enabled(self):
        cov = ShmCoverage(size=256)
        try:
            with pytest.raises(RuntimeError):
                cov.touched_snapshot()
        finally:
            cov.cleanup()

    def test_resize_regrows_the_region_and_drops_the_announcement(self):
        cov = ShmCoverage(size=256, touched_bitmap=True)
        try:
            _plant_magic(cov)
            cov.resize(1024)
            assert cov.touched_words == 16
            assert (
                cov.shm_bytes
                == 1024 * SIZEOF_ENTRY
                + SHM_METADATA_SIZE
                + SHM_TAIL_SIZE
                + touched_region_bytes(1024)
            )
            # New segment, new attach: until the target re-announces, the bitmap
            # cannot be trusted, so the caller must fall back to a full scan.
            assert not cov.touched_supported
            assert not cov.touched_snapshot().any()
        finally:
            cov.cleanup()


class TestSlotLookup:
    def test_slot_edge_ids_gathers_the_ids_at_set_slots(self):
        cov = ShmCoverage(size=256, touched_bitmap=True)
        try:
            table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=256)
            table["edge_id"][[3, 70, 200]] = [111, 222, 333]
            mask = np.zeros(cov.touched_words, dtype=np.uint64)
            mask[0] |= np.uint64(1) << np.uint64(3)
            mask[1] |= np.uint64(1) << np.uint64(6)  # slot 70
            assert sorted(cov.slot_edge_ids(mask).tolist()) == [111, 222]
        finally:
            cov.cleanup()

    def test_empty_slots_never_yield_id_zero(self):
        cov = ShmCoverage(size=128, touched_bitmap=True)
        try:
            mask = np.full(cov.touched_words, ~np.uint64(0), dtype=np.uint64)
            assert cov.slot_edge_ids(mask).size == 0
        finally:
            cov.cleanup()


class TestUnstableSlots:
    def test_slot_in_some_runs_only_is_unstable(self):
        a = np.array([0b0111], dtype=np.uint64)
        b = np.array([0b1111], dtype=np.uint64)
        c = np.array([0b0111], dtype=np.uint64)
        assert _bits(unstable_slots([a, b, c])) == {3}

    def test_identical_runs_have_none(self):
        a = np.array([0b1011, 5], dtype=np.uint64)
        assert not unstable_slots([a, a.copy(), a.copy()]).any()

    def test_two_runs_reduce_to_xor(self):
        rng = np.random.default_rng(0)
        a, b = (rng.integers(0, 2**63, size=32, dtype=np.uint64) for _ in range(2))
        assert np.array_equal(unstable_slots([a, b]), a ^ b)

    def test_present_in_every_run_is_stable(self):
        a = np.array([1 << 63], dtype=np.uint64)
        assert not unstable_slots([a, a, a]).any()


# argv[1:]: edge ids (decimal), each mapped raw into the table.
_DRIVER = """
#include <stdlib.h>
int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
        __afl_map_id_raw((uint32_t)strtoul(argv[i], NULL, 10));
    return 0;
}
"""
SIZE = 1000  # not a multiple of 64: the last bitmap word is partial


@pytest.fixture(scope="module")
def target(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("touched"), _DRIVER, "-D__AFL_CTX_SENSITIVE=0")


def _run(exe, cov: ShmCoverage, ids: list[int]) -> None:
    env = _clean_env(__AFL_SHM_ID=cov.env_id, AFL_MAP_SIZE=str(cov.num_entries))
    r = subprocess.run(
        [str(exe), *map(str, ids)], env=env, capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0, r.stderr


def _live_slots(cov: ShmCoverage) -> set[int]:
    table = np.frombuffer(cov._map, dtype=_ENTRY_DTYPE, count=cov.num_entries)
    return {int(i) for i in np.flatnonzero(table["edge_id"])}


@pytest.fixture(scope="module")
def plain_target(tmp_path_factory):
    return _build(
        tmp_path_factory.mktemp("touched_plain"),
        _DRIVER,
        "-D__AFL_CTX_SENSITIVE=0",
        "-D__AFL_TOUCHED_ATOMIC=0",
    )


@requires_clang
class TestPlainBuild:
    """Atomic is the default; the single-threaded opt-out must agree with it."""

    def test_plain_build_sets_the_same_bits(self, plain_target, target):
        results = []
        for exe in (target, plain_target):
            cov = ShmCoverage(size=SIZE, touched_bitmap=True)
            try:
                _run(exe, cov, [1, 2, 999, 1001, 5000, 0xFFFFFFFE])
                results.append(_bits(cov.touched_snapshot()))
            finally:
                cov.cleanup()
        assert results[0] == results[1]
        assert len(results[0]) == 6


@requires_clang
class TestShimSetsBits:
    def test_shim_announces_support_at_attach(self, target):
        cov = ShmCoverage(size=SIZE, touched_bitmap=True)
        try:
            assert not cov.touched_supported
            _run(target, cov, [5])
            assert cov.touched_supported
        finally:
            cov.cleanup()

    def test_first_claim_sets_exactly_the_claimed_slots(self, target):
        cov = ShmCoverage(size=SIZE, touched_bitmap=True)
        try:
            ids = [1, 2, 999, 1001, 5000, 0xFFFFFFFE]
            _run(target, cov, ids)
            assert _bits(cov.touched_snapshot()) == _live_slots(cov)
            assert len(_live_slots(cov)) == len(ids)
        finally:
            cov.cleanup()

    def test_colliding_edges_set_every_probed_slot(self, target):
        cov = ShmCoverage(size=SIZE, touched_bitmap=True)
        try:
            home = SIZE - 2  # probes wrap past the table end into slot 0
            ids = [i for i in range(1, 1 << 22) if home_slot(i, SIZE) == home][:4]
            _run(target, cov, ids)
            assert _bits(cov.touched_snapshot()) == {(home + k) % SIZE for k in range(4)}
        finally:
            cov.cleanup()

    def test_reclaim_in_a_new_generation_sets_the_bit_again(self, target):
        cov = ShmCoverage(size=SIZE, touched_bitmap=True)
        try:
            _run(target, cov, [10, 20, 30])
            cov.reset_edge_map()  # generation bump: entries go stale, stay in place
            cov.touched_clear()
            _run(target, cov, [10, 30])
            assert _bits(cov.touched_snapshot()) == {home_slot(10, SIZE), home_slot(30, SIZE)}
        finally:
            cov.cleanup()

    def test_a_plain_hit_does_not_touch_the_bitmap(self, target):
        """Semantics: the bit means 'became live this generation', not 'fired'."""
        cov = ShmCoverage(size=SIZE, touched_bitmap=True)
        try:
            _run(target, cov, [10, 20])
            cov.touched_clear()  # no generation bump: the next run only hits
            _run(target, cov, [10, 20])
            assert not cov.touched_snapshot().any()
        finally:
            cov.cleanup()

    def test_segment_without_the_region_is_left_alone(self, target):
        cov = ShmCoverage(size=SIZE)  # exact old size: any stray write would fault or corrupt
        try:
            _run(target, cov, [1, 2, 3])
            assert len(_live_slots(cov)) == 3
            assert not cov.touched_supported
        finally:
            cov.cleanup()
