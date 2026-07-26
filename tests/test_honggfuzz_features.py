"""Tests for honggfuzz-ported features.

Covers: TLV mutation, token shuffle, gradient CMP, crash stack hash,
rare edge tracking, honggfuzz power factors, hw perf counters,
persistent mode health monitoring, parser token extraction.
"""

import time

import pytest


# ── TLV Mutation ────────────────────────────────────────────────────


class TestTlvMutate:
    def test_basic_mutation(self):
        from fuzzer_tool.core.tlv_mutate import tlv_mutate

        data = b"GET /index.html HTTP/1.1\r\nHost: example.com"
        result = tlv_mutate(data)
        assert isinstance(result, bytes)
        assert len(result) >= len(data)  # may insert TLV fallback

    def test_small_input_fallback(self):
        from fuzzer_tool.core.tlv_mutate import tlv_mutate

        result = tlv_mutate(b"ab")
        assert isinstance(result, bytes)
        assert len(result) >= 2  # original + possible TLV insert

    def test_empty_input(self):
        from fuzzer_tool.core.tlv_mutate import tlv_mutate

        result = tlv_mutate(b"")
        assert isinstance(result, bytes)
        assert len(result) == 4  # just the TLV fallback

    def test_length_field_detection(self):
        from fuzzer_tool.core.tlv_mutate import tlv_mutate

        # Input where byte at offset 0 = 5, and there are 5+ bytes remaining
        data = bytes([5]) + b"AAAAABBBBB"
        # Multiple runs should eventually mutate the length field
        mutated = False
        for _ in range(100):
            result = tlv_mutate(data)
            if result != data:
                mutated = True
                break
        assert mutated

    def test_deterministic_with_seed(self):
        from fuzzer_tool.core.tlv_mutate import tlv_mutate

        import random

        data = b"test data for mutation"
        r1 = tlv_mutate(data, rng=random.Random(42))
        r2 = tlv_mutate(data, rng=random.Random(42))
        assert r1 == r2


# ── Token Shuffle ───────────────────────────────────────────────────


class TestTokenShuffle:
    def test_basic_shuffle(self):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        data = b"key1=val1 key2=val2 key3=val3"
        result = token_shuffle(data)
        assert isinstance(result, bytes)
        assert len(result) == len(data)  # same length after swap

    def test_different_length_tokens(self):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        data = b"a bb ccc dddd"
        result = token_shuffle(data)
        assert isinstance(result, bytes)
        # Length may change when swapping different-length tokens
        assert len(result) > 0

    def test_single_token_no_swap(self):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        data = b"nodelimiter"
        result = token_shuffle(data)
        assert result == data  # no delimiters, no swap

    def test_short_input(self):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        result = token_shuffle(b"ab")
        assert result == b"ab"  # too short

    def test_deterministic_with_seed(self):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        import random

        data = b"a b c d e f"
        r1 = token_shuffle(data, rng=random.Random(42))
        r2 = token_shuffle(data, rng=random.Random(42))
        assert r1 == r2


# ── Gradient CMP ────────────────────────────────────────────────────


class TestGradientCmp:
    def test_partial_match(self):
        from fuzzer_tool.core.gradient_cmp import gradient_cmp

        data = b"GET /index HTTP/1.1"
        cmp_values = [(b"HTTP/1.1", b"HTTP/2.0")]
        result = gradient_cmp(data, cmp_values)
        assert isinstance(result, bytes)
        assert len(result) == len(data)

    def test_no_match_inserts(self):
        from fuzzer_tool.core.gradient_cmp import gradient_cmp

        data = b"AAAA"
        cmp_values = [(b"BBBB", b"CCCC")]
        result = gradient_cmp(data, cmp_values)
        # Should insert the CMP value
        assert len(result) >= len(data)

    def test_empty_data(self):
        from fuzzer_tool.core.gradient_cmp import gradient_cmp

        result = gradient_cmp(b"", [(b"test", b"best")])
        assert isinstance(result, bytes)

    def test_no_cmp_values(self):
        from fuzzer_tool.core.gradient_cmp import gradient_cmp

        result = gradient_cmp(b"hello", [])
        assert result == b"hello"

    def test_multiple_cmp_pairs(self):
        from fuzzer_tool.core.gradient_cmp import gradient_cmp

        data = b"AAAA BBBB CCCC"
        cmp_values = [(b"AAAA", b"XXXX"), (b"BBBB", b"YYYY")]
        result = gradient_cmp(data, cmp_values)
        assert isinstance(result, bytes)


# ── Crash Stack Hash ────────────────────────────────────────────────


class TestCrashStackHash:
    def test_stack_hash_with_asan_output(self):
        from fuzzer_tool.core.sanitizer import SanitizerReport

        stderr = (
            "==12345==ERROR: AddressSanitizer: heap-buffer-overflow\n"
            "    #0 0x55a1b2c3d4e0 in func1 target.c:42\n"
            "    #1 0x55a1b2c3d600 in func2 target.c:100\n"
            "    #2 0x7f1234567890 in __libc_start_main\n"
        )
        report = SanitizerReport.parse(stderr)
        h = report.stack_hash()
        assert isinstance(h, str)
        assert len(h) == 16  # 64-bit hex
        assert h != "0" * 16  # not all zeros

    def test_stack_hash_deterministic(self):
        from fuzzer_tool.core.sanitizer import SanitizerReport

        stderr = (
            "==12345==ERROR: AddressSanitizer: heap-buffer-overflow\n"
            "    #0 0x55a1b2c3d4e0 in func1 target.c:42\n"
            "    #1 0x55a1b2c3d600 in func2 target.c:100\n"
        )
        r1 = SanitizerReport.parse(stderr)
        r2 = SanitizerReport.parse(stderr)
        assert r1.stack_hash() == r2.stack_hash()

    def test_different_stacks_different_hashes(self):
        from fuzzer_tool.core.sanitizer import SanitizerReport

        s1 = "==1==ERROR: AddressSanitizer: x\n    #0 0x1111 in a\n    #1 0x2222 in b\n"
        s2 = "==1==ERROR: AddressSanitizer: x\n    #0 0x3333 in c\n    #1 0x4444 in d\n"
        r1 = SanitizerReport.parse(s1)
        r2 = SanitizerReport.parse(s2)
        assert r1 is not None and r2 is not None
        assert r1.stack_hash() != r2.stack_hash()

    def test_single_frame_masked(self):
        from fuzzer_tool.core.sanitizer import SanitizerReport

        stderr = "==1==ERROR: AddressSanitizer: x\n    #0 0x12345678 in func\n"
        report = SanitizerReport.parse(stderr)
        assert report is not None
        h = report.stack_hash()
        # Should have mask applied
        assert isinstance(h, str)
        assert len(h) == 16

    def test_no_sanitizer_returns_empty(self):
        from fuzzer_tool.core.sanitizer import SanitizerReport

        report = SanitizerReport.parse("Segmentation fault")
        assert report is None


# ── Rare Edge Tracking ──────────────────────────────────────────────


class TestRareEdgeTracking:
    def test_owner_count_tracking(self):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=1024)
        tracker.record_edges("s1", {100, 200})
        tracker.record_edges("s2", {100, 300})
        tracker.record_edges("s3", {200})

        assert tracker._edge_owner_count[100] == 2
        assert tracker._edge_owner_count[200] == 2
        assert tracker._edge_owner_count[300] == 1

    def test_rare_edge_count(self):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=1024)
        tracker.record_edges("s1", {100, 200, 300})
        tracker.record_edges("s2", {100, 200, 400})
        tracker.record_edges("s3", {100, 500})

        # Edge 100: owner_count=3, rare (< threshold=4)
        # Edge 200: owner_count=2, rare
        # Edge 300: owner_count=1, rare
        # s1 hits {100, 200, 300} -> 3 rare edges
        rare = tracker.rare_edge_count("s1")
        assert rare == 3  # all three are rare (< 4 owners)

    def test_rare_edge_threshold(self):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=1024)
        for i in range(5):
            tracker.record_edges(f"s{i}", {100})

        # Edge 100: owner_count=5, not rare with threshold=4
        assert tracker.rare_edge_count("s0", threshold=4) == 0
        # But rare with threshold=6
        assert tracker.rare_edge_count("s0", threshold=6) == 1

    def test_persistence(self, tmp_path):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=1024)
        tracker.record_edges("s1", {100, 200})
        tracker.record_edges("s2", {100, 300})

        path = str(tmp_path / "tracker.json")
        tracker.save(path)

        tracker2 = EdgeTracker(map_size=1024)
        tracker2.load(path)

        assert tracker2._edge_owner_count == tracker._edge_owner_count


# ── Honggfuzz Power Factors ─────────────────────────────────────────


class TestHonggfuzzFactors:
    def test_freshness_boost(self):
        from fuzzer_tool.core.schedules import SeedScorer

        scorer = SeedScorer("base")
        now = time.time()
        # Fresh seed (<60s old) gets 4x freshness boost
        s_fresh = scorer.score(
            exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
            handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
            time_added=now - 30, now=now,
        )
        s_old = scorer.score(
            exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
            handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
            time_added=now - 600, now=now,
        )
        assert s_fresh > s_old

    def test_timeout_penalty(self):
        from fuzzer_tool.core.schedules import SeedScorer

        scorer = SeedScorer("base")
        s_normal = scorer.score(
            exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
            handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
        )
        s_timeout = scorer.score(
            exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
            handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
            timed_out=True,
        )
        assert s_timeout < s_normal * 0.1  # 1/32 penalty

    def test_density_boost(self):
        from fuzzer_tool.core.schedules import SeedScorer

        scorer = SeedScorer("base")
        # High density: 50 edges in 20 bytes = 250%
        s_dense = scorer.score(
            exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
            handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
            input_size=20,
        )
        # Low density: 50 edges in 1000 bytes = 5%
        s_sparse = scorer.score(
            exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
            handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
            input_size=1000,
        )
        assert s_dense > s_sparse

    def test_fertility_boost(self):
        from fuzzer_tool.core.schedules import SeedScorer

        scorer = SeedScorer("base")
        s_child = scorer.score(
            exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
            handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
            child_count=3,
        )
        s_no_child = scorer.score(
            exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
            handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
            child_count=0,
        )
        assert s_child > s_no_child

    def test_all_schedules_get_hw_perf(self):
        from fuzzer_tool.core.schedules import SeedScorer

        for schedule in ("base", "fast", "rare", "mopt"):
            scorer = SeedScorer(schedule)
            s_low = scorer.score(
                exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
                handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
                hw_instructions=100,
            )
            # Run a few to build EMA
            for _ in range(20):
                scorer.score(
                    exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
                    handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
                    hw_instructions=1000,
                )
            s_high = scorer.score(
                exec_us=100, avg_exec_us=100, bitmap_size=50, avg_bitmap_size=50,
                handicap=0, depth=1, fuzz_level=1, n_fuzz=1, total_execs=100,
                hw_instructions=1000,
            )
            assert s_high >= s_low, f"hw_perf boost failed for schedule={schedule}"


# ── HW Perf Counters ────────────────────────────────────────────────


class TestPerfCounters:
    def test_init(self):
        from fuzzer_tool.adapters.perf_event import PerfCounters

        pc = PerfCounters()
        assert isinstance(pc.counter_names, list)
        assert "instructions" in pc.counter_names

    def test_unavailable_detection(self):
        from fuzzer_tool.adapters.perf_event import PerfCounters

        pc = PerfCounters()
        # Should not crash even if perf is unavailable
        assert isinstance(pc.available, bool)

    def test_stats_initial(self):
        from fuzzer_tool.adapters.perf_event import PerfCounters

        pc = PerfCounters()
        stats = pc.stats
        assert stats["total_instructions"] == 0
        assert stats["total_branches"] == 0
        assert stats["read_count"] == 0

    def test_close_noop(self):
        from fuzzer_tool.adapters.perf_event import PerfCounters

        pc = PerfCounters()
        pc.close()  # should not crash

    def test_shim_init(self):
        from fuzzer_tool.adapters.perf_event import PerfShim

        shim = PerfShim()
        assert shim.available is False  # no .so compiled


# ── Edge Tracker HW Perf Metrics ────────────────────────────────────


class TestEdgeTrackerHwPerf:
    def test_record_hw_metrics(self):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=1024)
        tracker.record_edges(
            "s1", {100, 200},
            hw_instructions=50000,
            hw_branches=10000,
            hw_branch_misses=500,
        )
        assert tracker.get_seed_hw_instructions("s1") == 50000
        assert tracker.get_seed_hw_branches("s1") == 10000
        assert tracker.get_seed_hw_branch_misses("s1") == 500

    def test_no_hw_metrics(self):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=1024)
        tracker.record_edges("s1", {100})
        assert tracker.get_seed_hw_instructions("s1") == 0
        assert tracker.get_seed_hw_branches("s1") == 0

    def test_persistence(self, tmp_path):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=1024)
        tracker.record_edges("s1", {100}, hw_instructions=12345, hw_branches=6789)
        path = str(tmp_path / "tracker.json")
        tracker.save(path)

        tracker2 = EdgeTracker(map_size=1024)
        tracker2.load(path)
        assert tracker2.get_seed_hw_instructions("s1") == 12345
        assert tracker2.get_seed_hw_branches("s1") == 6789


# ── Parser Token Extraction ─────────────────────────────────────────


class TestParserTokens:
    def test_profile_has_parser_tokens_field(self):
        from fuzzer_tool.core.target_profiler import TargetProfile

        p = TargetProfile()
        assert hasattr(p, "parser_tokens")
        assert p.parser_tokens == []

    def test_profile_serialization(self):
        from fuzzer_tool.core.target_profiler import TargetProfile

        p = TargetProfile(parser_tokens=[b"token1", b"token2"])
        d = p.to_dict()
        assert "parser_tokens" in d
        assert len(d["parser_tokens"]) == 2

        p2 = TargetProfile.from_dict(d)
        assert p2.parser_tokens == [b"token1", b"token2"]


# ── Save Crash with Stack Hash ──────────────────────────────────────


class TestSaveCrashStackHash:
    def test_blocklist_filtering(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import save_crash

        stderr = (
            "==1==ERROR: ASAN: heap-buffer-overflow\n"
            "    #0 0x1111 in func1\n"
            "    #1 0x2222 in func2\n"
        )
        crash_hashes = set()
        crash_sigs = {}

        # First save should succeed
        result = save_crash(
            b"crash_input_1", 1, stderr, tmp_path,
            crash_hashes, crash_sigs,
        )
        assert result is not False

    def test_smaller_crash_replacement(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import save_crash

        stderr = (
            "==1==ERROR: ASAN: x\n"
            "    #0 0x1111 in func\n"
            "    #1 0x2222 in func2\n"
        )
        crash_hashes = set()
        crash_sigs = {}
        crash_min_sizes = {}

        # Save large trigger first
        save_crash(
            b"A" * 1000, 1, stderr, tmp_path,
            crash_hashes, crash_sigs,
            crash_min_sizes=crash_min_sizes,
        )

        # Save smaller trigger for same stack hash
        # (needs different data hash to pass dedup)
        save_crash(
            b"B" * 10, 1, stderr, tmp_path,
            crash_hashes, crash_sigs,
            crash_min_sizes=crash_min_sizes,
        )
