"""Hit-count bucketing on the SHM coverage path.

`_check_new_coverage` used to decide novelty by set membership alone, so an
input driving a loop twice and one driving it 128 times were the same
coverage.  Bucketed hitcounts are how AFL crosses loop-count-guarded
branches (`if (n > 16)`, buffer-growth paths, parser backtrack limits).
"""

import ctypes

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import VIRGIN_DENSE_MAX, ShmCoverage
from fuzzer_tool.core.count_class import (
    BUCKET_BIT_TABLE,
    _classify_byte,
    bucket_bit,
    bucket_bits,
    classify_single,
)

# AFL's count_class_lookup8, written out rather than derived, so this test
# fails if the ladder drifts instead of drifting with it.
AFL_LOOKUP8 = (
    {0: 0, 1: 1, 2: 2, 3: 4}
    | dict.fromkeys(range(4, 8), 8)
    | dict.fromkeys(range(8, 16), 16)
    | dict.fromkeys(range(16, 32), 32)
    | dict.fromkeys(range(32, 128), 64)
    | dict.fromkeys(range(128, 256), 128)
)


def write_entry(cov: ShmCoverage, edge_id: int, count: int) -> None:
    """Write one {edge_id, count} the way the C shim's probe would.

    Deliberately does NOT touch any reader-side state, unlike
    ``record_edge`` — these tests need the shim's half only.
    """
    idx = edge_id % cov.num_entries
    while cov._entries[idx].edge_id not in (0, edge_id):
        idx = (idx + 1) % cov.num_entries
    if cov._entries[idx].edge_id == 0:
        cov._entries[idx].edge_id = edge_id
        ctypes.c_uint64.from_address(cov._ptr + 16).value += 1
    cov._entries[idx].count = count


def run_exec(cov: ShmCoverage, hits: dict[int, int]) -> bool:
    """Simulate one execution hitting each edge the given number of times."""
    cov.reset_edge_map()
    path_hash = 0
    for edge_id, count in hits.items():
        write_entry(cov, edge_id, count)
        for _ in range(count):
            path_hash = (path_hash * 31) ^ edge_id
    ctypes.c_uint64.from_address(cov._ptr + 8).value = path_hash & 0xFFFFFFFFFFFFFFFF
    return cov.is_new_coverage_with_edges()[0]


@pytest.fixture
def cov():
    c = ShmCoverage()
    try:
        yield c
    finally:
        c.cleanup()


class TestBucketLadder:
    def test_matches_afl_count_class_lookup8(self):
        assert {v: bucket_bit(v) for v in range(256)} == AFL_LOOKUP8

    def test_every_bucket_is_a_single_distinct_bit(self):
        bits = {bucket_bit(v) for v in range(1, 256)}
        assert len(bits) == 8
        assert all(b & (b - 1) == 0 for b in bits)

    def test_empty_slot_occupies_no_bucket(self):
        assert bucket_bit(0) == 0
        assert bucket_bit(-1) == 0

    def test_counts_above_255_clamp_into_the_top_bucket(self):
        # The SHM count field is uint32, not AFL's uint8, so these are real
        # values rather than a wrapped counter.
        for v in (256, 5000, 2**31, 2**32 - 1):
            assert bucket_bit(v) == 0x80

    def test_classify_single_bits_alias_and_bucket_bits_do_not(self):
        """Why this ladder exists rather than reusing classify_single().

        classify_single returns representative values, not disjoint bits:
        class 3 is 0b11, which is class 1 OR'd with class 2.  A virgin map
        accumulates by OR, so under those values an edge seen once and then
        twice masks a later hit count of exactly 3.
        """
        aliased = _classify_byte(1) | _classify_byte(2)
        assert classify_single(3) & ~aliased == 0  # the bug

        accumulated = bucket_bit(1) | bucket_bit(2)
        assert bucket_bit(3) & ~accumulated != 0  # the fix

    def test_table_agrees_with_scalar(self):
        assert list(BUCKET_BIT_TABLE) == [bucket_bit(v) for v in range(256)]


class TestBucketBitsVectorized:
    def test_matches_scalar(self):
        counts = np.array([0, 1, 2, 3, 4, 7, 8, 31, 32, 127, 128, 255], dtype=np.uint32)
        assert list(bucket_bits(counts)) == [bucket_bit(int(c)) for c in counts]

    def test_clamps_without_overflow(self):
        # uint32 -> uint8 must clamp, not truncate: 256 & 0xFF == 0 would
        # silently turn a hot edge into an empty slot.
        assert list(bucket_bits(np.array([256, 512, 2**32 - 1], dtype=np.uint32))) == [0x80] * 3

    def test_empty_input(self):
        out = bucket_bits(np.zeros(0, dtype=np.uint32))
        assert out.size == 0 and out.dtype == np.uint8

    def test_dtype_is_uint8(self):
        assert bucket_bits(np.array([5], dtype=np.uint32)).dtype == np.uint8


class TestLoopCountDiscrimination:
    def test_crossing_bucket_boundaries_is_new_coverage(self, cov):
        """The regression this whole change exists for.

        Same edge set every time; only the loop body's hit count moves.
        Before bucketing, every execution after the first reported False.
        """
        assert run_exec(cov, {10: 1, 20: 1, 99: 2}) is True  # first sighting
        for hits, expected in ((3, True), (5, True), (40, True), (128, True)):
            assert run_exec(cov, {10: 1, 20: 1, 99: hits}) is expected, hits

    def test_same_bucket_twice_is_not_new(self, cov):
        run_exec(cov, {10: 1, 20: 1, 99: 2})
        assert run_exec(cov, {10: 1, 20: 1, 99: 2}) is False

    def test_different_counts_in_one_bucket_are_not_new(self, cov):
        run_exec(cov, {10: 1, 99: 40})  # 32-127 bucket
        assert run_exec(cov, {10: 1, 99: 100}) is False  # same bucket
        assert run_exec(cov, {10: 1, 99: 5000}) is True  # 128+ bucket

    def test_descending_into_a_seen_bucket_is_not_new(self, cov):
        run_exec(cov, {10: 1, 99: 200})
        run_exec(cov, {10: 1, 99: 2})
        assert run_exec(cov, {10: 1, 99: 200}) is False

    def test_virgin_accumulates_the_or_of_every_bucket_seen(self, cov):
        for hits in (2, 3, 5, 40, 128):
            run_exec(cov, {99: hits})
        expected = bucket_bit(2) | bucket_bit(3) | bucket_bit(5) | bucket_bit(40) | bucket_bit(128)
        assert cov.get_virgin_buckets()[99] == expected

    def test_a_new_edge_is_still_new_coverage(self, cov):
        run_exec(cov, {10: 1, 99: 2})
        assert run_exec(cov, {10: 1, 99: 2, 55: 1}) is True

    def test_bucket_transitions_counts_first_sightings(self, cov):
        run_exec(cov, {10: 1, 20: 1, 99: 2})
        assert cov.bucket_transitions == 3
        run_exec(cov, {10: 1, 20: 1, 99: 2})
        assert cov.bucket_transitions == 3  # nothing new
        run_exec(cov, {10: 1, 20: 1, 99: 9})
        assert cov.bucket_transitions == 4  # only edge 99 moved bucket


class TestCountZeroEntries:
    def test_zero_count_entry_occupies_no_bucket(self, cov):
        """The shim never writes count 0 into a claimed slot, but tests and
        torn reads do — such an entry must not consume a bucket."""
        cov.reset_edge_map()
        cov._entries[0].edge_id = 7  # count stays 0
        ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
        assert cov.is_new_coverage() is True  # new edge_id
        assert cov.get_virgin_buckets() == {}

    def test_new_edge_with_zero_count_still_counts_as_an_edge(self, cov):
        cov.reset_edge_map()
        cov._entries[0].edge_id = 7
        ctypes.c_uint64.from_address(cov._ptr + 16).value = 1
        cov.is_new_coverage()
        assert 7 in cov._seen_edge_ids
        assert cov.cumulative_edges == 1


class TestFastPath:
    def test_unchanged_path_hash_short_circuits(self, cov):
        """path_hash advances on every edge fire, so an unchanged hash means
        no count can have moved — the fast path stays sound."""
        run_exec(cov, {10: 1, 99: 4})
        before = cov.bucket_transitions
        assert cov.is_new_coverage() is False
        assert cov.bucket_transitions == before

    def test_count_change_moves_the_path_hash(self, cov):
        run_exec(cov, {10: 1, 99: 4})
        first = cov.read_path_hash()
        run_exec(cov, {10: 1, 99: 9})
        assert cov.read_path_hash() != first


class TestRecordEdgeCoherence:
    def test_record_edge_keeps_the_virgin_map_current(self, cov):
        """record_edge stands in for both writer and reader, so leaving the
        reader half stale would make a recorded edge report as new."""
        cov.reset_edge_map()
        cov.record_edge(55)
        new, edges = cov.is_new_coverage_with_edges()
        assert new is False
        assert 55 in edges

    def test_record_edge_marks_each_bucket_it_passes_through(self, cov):
        cov.reset_edge_map()
        for _ in range(3):
            cov.record_edge(55)
        assert cov.get_virgin_buckets()[55] == bucket_bit(1) | bucket_bit(2) | bucket_bit(3)

    def test_full_table_miss_records_no_bucket(self, cov):
        small = ShmCoverage(size=4)
        try:
            small.reset_edge_map()
            for eid in (4, 5, 10, 15):
                small.record_edge(eid)
            before = dict(small.get_virgin_buckets())
            assert small.record_edge(1) is False  # probes all slots, misses
            assert small.get_virgin_buckets() == before
        finally:
            small.cleanup()


class TestVirginStorage:
    def test_dense_array_grows_to_fit_the_edge_id(self, cov):
        cov._update_virgin_buckets(
            np.array([5000], dtype=np.uint32), np.array([1], dtype=np.uint32)
        )
        assert cov._virgin.size >= 5001
        assert cov.get_virgin_buckets() == {5000: 0x01}

    def test_growth_preserves_existing_buckets(self, cov):
        cov._update_virgin_buckets(np.array([3], dtype=np.uint32), np.array([2], dtype=np.uint32))
        cov._update_virgin_buckets(
            np.array([1 << 20], dtype=np.uint32), np.array([1], dtype=np.uint32)
        )
        assert cov.get_virgin_buckets() == {3: 0x02, 1 << 20: 0x01}

    def test_ids_past_the_dense_bound_use_the_overflow_dict(self, cov):
        """Reachable only with __AFL_CTX_BITS in the 24..32 range, which
        scatters edge ids across the whole u32 space."""
        wide = np.array([VIRGIN_DENSE_MAX + 7], dtype=np.uint32)
        assert cov._update_virgin_buckets(wide, np.array([1], dtype=np.uint32)) is True
        assert cov._update_virgin_buckets(wide, np.array([1], dtype=np.uint32)) is False
        assert cov._update_virgin_buckets(wide, np.array([9], dtype=np.uint32)) is True
        assert cov._virgin_wide == {VIRGIN_DENSE_MAX + 7: bucket_bit(1) | bucket_bit(9)}
        assert cov._virgin.size == 0  # nothing spilled into the dense array

    def test_mixed_dense_and_overflow_in_one_execution(self, cov):
        ids = np.array([12, VIRGIN_DENSE_MAX + 12], dtype=np.uint32)
        counts = np.array([1, 1], dtype=np.uint32)
        assert cov._update_virgin_buckets(ids, counts) is True
        assert cov.get_virgin_buckets() == {12: 0x01, VIRGIN_DENSE_MAX + 12: 0x01}
        assert cov._update_virgin_buckets(ids, counts) is False

    def test_duplicate_free_store_is_not_order_dependent(self, cov):
        """edge_ids are unique within the table, so the fancy-indexed store
        has no duplicate targets."""
        ids = np.array([4, 8, 15, 16, 23, 42], dtype=np.uint32)
        cov._update_virgin_buckets(ids, np.full(6, 7, dtype=np.uint32))
        assert set(cov.get_virgin_buckets()) == set(ids.tolist())
        assert all(v == bucket_bit(7) for v in cov.get_virgin_buckets().values())

    def test_virgin_survives_resize(self, cov):
        """Buckets are keyed by edge_id, which carries no map_size term."""
        run_exec(cov, {10: 1, 99: 40})
        before = cov.get_virgin_buckets()
        cov.resize(cov.num_entries * 4)
        assert cov.get_virgin_buckets() == before
        assert run_exec(cov, {10: 1, 99: 40}) is False
