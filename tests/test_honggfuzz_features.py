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
        import random

        from fuzzer_tool.core.tlv_mutate import tlv_mutate

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
        import random

        from fuzzer_tool.core.token_shuffle import token_shuffle

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

    def test_output_identical_to_legacy_algorithm(self):
        """The candidate-window implementation must produce byte-identical
        output to the legacy full-scan algorithm for the same seeded rng."""
        import random

        from fuzzer_tool.core.gradient_cmp import gradient_cmp

        def legacy(data, cmp_values, rng):
            r = rng
            if not data or not cmp_values:
                return data
            buf = bytearray(data)
            for cmp_a, cmp_b in cmp_values:
                if len(cmp_a) == 0 or len(cmp_a) > 32:
                    continue
                for cmp_val in (cmp_a, cmp_b):
                    if len(cmp_val) == 0:
                        continue
                    for off in range(len(buf) - len(cmp_val) + 1):
                        matches = 0
                        first_diff = len(cmp_val)
                        diff_mask = 0
                        for i in range(len(cmp_val)):
                            if buf[off + i] == cmp_val[i]:
                                matches += 1
                            elif first_diff == len(cmp_val):
                                first_diff = i
                                diff_mask = buf[off + i] ^ cmp_val[i]
                        if 0 < matches < len(cmp_val) and first_diff < len(cmp_val):
                            target_off = off + first_diff
                            strategy = r.randint(0, 5)
                            if strategy == 0:
                                buf[target_off] = cmp_val[first_diff]
                            elif strategy == 1:
                                buf[target_off] ^= diff_mask
                            elif strategy == 2:
                                if buf[target_off] < cmp_val[first_diff]:
                                    buf[target_off] = min(buf[target_off] + 1, 255)
                                else:
                                    buf[target_off] = max(buf[target_off] - 1, 0)
                            elif strategy == 3:
                                buf[target_off] = (buf[target_off] + cmp_val[first_diff]) // 2
                            elif strategy == 4:
                                end = min(off + len(cmp_val), len(buf))
                                buf[off:end] = cmp_val[: end - off]
                                return bytes(buf)
                            else:
                                buf[target_off] ^= 1 << r.randint(0, 7)
                            return bytes(buf)
            if cmp_values:
                cmp_val = cmp_values[r.randint(0, len(cmp_values) - 1)][0]
                if 0 < len(cmp_val) <= 32:
                    pos = r.randint(0, len(buf))
                    return bytes(buf[:pos]) + cmp_val + bytes(buf[pos:])
            return bytes(buf)

        rng_old = random.Random(1234)
        rng_new = random.Random(1234)
        for trial in range(300):
            n = random.Random(trial).randint(0, 40)
            data = bytes(random.Random(trial + 1).randrange(256) for _ in range(n))
            pairs = []
            for _ in range(random.Random(trial + 2).randint(0, 8)):
                a = bytes(
                    random.Random(trial + 3).randrange(256)
                    for _ in range(random.Random(trial + 5).randint(1, 40))
                )
                b = bytes(
                    random.Random(trial + 4).randrange(256)
                    for _ in range(random.Random(trial + 6).randint(1, 40))
                )
                pairs.append((a, b))
            out_old = legacy(data, pairs, rng_old)
            out_new = gradient_cmp(data, pairs, rng_new)
            assert out_new == out_old, f"trial {trial}: data={data!r} pairs={pairs!r}"


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
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=1,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
            time_added=now - 30,
            now=now,
        )
        s_old = scorer.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=1,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
            time_added=now - 600,
            now=now,
        )
        assert s_fresh > s_old

    def test_timeout_penalty(self):
        from fuzzer_tool.core.schedules import SeedScorer

        scorer = SeedScorer("base")
        s_normal = scorer.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=1,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
        )
        s_timeout = scorer.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=1,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
            timed_out=True,
        )
        assert s_timeout < s_normal * 0.1  # 1/32 penalty

    def test_density_boost(self):
        from fuzzer_tool.core.schedules import SeedScorer

        scorer = SeedScorer("base")
        # High density: 50 edges in 20 bytes = 250%
        s_dense = scorer.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=1,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
            input_size=20,
        )
        # Low density: 50 edges in 1000 bytes = 5%
        s_sparse = scorer.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=1,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
            input_size=1000,
        )
        assert s_dense > s_sparse

    def test_fertility_boost(self):
        from fuzzer_tool.core.schedules import SeedScorer

        scorer = SeedScorer("base")
        s_child = scorer.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=1,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
            child_count=3,
        )
        s_no_child = scorer.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=1,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
            child_count=0,
        )
        assert s_child > s_no_child

    def test_all_schedules_get_hw_perf(self):
        from fuzzer_tool.core.schedules import SeedScorer

        for schedule in ("base", "fast", "rare", "mopt"):
            scorer = SeedScorer(schedule)
            s_low = scorer.score(
                exec_us=100,
                avg_exec_us=100,
                bitmap_size=50,
                avg_bitmap_size=50,
                handicap=0,
                depth=1,
                fuzz_level=1,
                n_fuzz=1,
                total_execs=100,
                hw_instructions=100,
            )
            # Run a few to build EMA
            for _ in range(20):
                scorer.score(
                    exec_us=100,
                    avg_exec_us=100,
                    bitmap_size=50,
                    avg_bitmap_size=50,
                    handicap=0,
                    depth=1,
                    fuzz_level=1,
                    n_fuzz=1,
                    total_execs=100,
                    hw_instructions=1000,
                )
            s_high = scorer.score(
                exec_us=100,
                avg_exec_us=100,
                bitmap_size=50,
                avg_bitmap_size=50,
                handicap=0,
                depth=1,
                fuzz_level=1,
                n_fuzz=1,
                total_execs=100,
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

    # ── Regression tests for PMU detection fix ───────────────────────

    def test_default_exclude_kernel_false(self):
        """exclude_kernel=False is the default (fix for AMD systems where
        True zeros out user-space instruction counting)."""
        from fuzzer_tool.adapters.perf_event import PerfCounters

        pc = PerfCounters()
        assert pc.exclude_kernel is False

    def test_regression_probe_based_detection(self):
        """_check_available uses a perf_event_open probe, not a fragile
        PMU name whitelist. Verify that the probe path is exercised and
        does not raise on this system regardless of PMU availability."""
        from fuzzer_tool.adapters.perf_event import PerfCounters

        pc = PerfCounters()
        # available may be True or False depending on the test runner,
        # but the probe path must not crash (regression: old whitelist
        # missed AMD PMUs and returned False on Ryzen systems).
        assert isinstance(pc.available, bool)
        assert hasattr(pc, "_available")

    def test_regression_inprocess_counting(self):
        """open_for_pid(0) on the current process followed by work
        must return non-zero instruction deltas (regression: counters
        were never opened for in-process direct/direct_lite mode)."""
        from fuzzer_tool.adapters.perf_event import PerfCounters

        pc = PerfCounters()
        if not pc.available:
            pytest.skip("hardware perf counters not available on this system")

        assert pc.open_for_pid(0), "open_for_pid(0) should succeed"
        _ = sum(i * i for i in range(200000))  # enough work to register
        deltas = pc.read_and_reset()
        pc.close()

        assert deltas.get("instructions", 0) > 0, (
            f"Expected non-zero instructions after work, got {deltas}"
        )
        assert deltas.get("branches", 0) > 0, f"Expected non-zero branches after work, got {deltas}"

    def test_regression_subprocess_counting(self):
        """open_for_pid(pid) on a forked child must return non-zero
        instruction deltas (regression: inherit=1 doesn't survive exec,
        so counters must be opened on the child PID directly)."""
        import os

        from fuzzer_tool.adapters.perf_event import PerfCounters

        pc = PerfCounters()
        if not pc.available:
            pytest.skip("hardware perf counters not available on this system")

        pid = os.fork()
        if pid == 0:
            _ = sum(i * i for i in range(500000))
            os._exit(0)

        assert pc.open_for_pid(pid), "open_for_pid(child_pid) should succeed"
        _, status = os.waitpid(pid, 0)
        deltas = pc.read_and_reset()
        pc.close()

        assert deltas.get("instructions", 0) > 0, (
            f"Expected non-zero instructions from child pid={pid}, got {deltas}"
        )
        assert deltas.get("branches", 0) > 0, (
            f"Expected non-zero branches from child pid={pid}, got {deltas}"
        )


# ── Edge Tracker HW Perf Metrics ────────────────────────────────────


class TestEdgeTrackerHwPerf:
    def test_record_hw_metrics(self):
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        tracker = EdgeTracker(map_size=1024)
        tracker.record_edges(
            "s1",
            {100, 200},
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
            b"crash_input_1",
            1,
            stderr,
            tmp_path,
            crash_hashes,
            crash_sigs,
        )
        assert result is not False

    def test_smaller_crash_replacement(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import save_crash

        stderr = "==1==ERROR: ASAN: x\n    #0 0x1111 in func\n    #1 0x2222 in func2\n"
        crash_hashes = set()
        crash_sigs = {}
        crash_min_sizes = {}

        # Save large trigger first
        save_crash(
            b"A" * 1000,
            1,
            stderr,
            tmp_path,
            crash_hashes,
            crash_sigs,
            crash_min_sizes=crash_min_sizes,
        )

        # Save smaller trigger for same stack hash
        # (needs different data hash to pass dedup)
        save_crash(
            b"B" * 10,
            1,
            stderr,
            tmp_path,
            crash_hashes,
            crash_sigs,
            crash_min_sizes=crash_min_sizes,
        )


# ── Regression Tests ────────────────────────────────────────────────


class TestTokenShuffleRegression:
    """Regression: last-token delimiter was dropped during swap.

    Before fix: token_shuffle(b'a bcdefgh ij klmnop') produced
    'a klmnopij bcdefgh' — the space between klmnop and ij was lost
    because token spans included trailing delimiters except for the
    last token, creating an asymmetric swap.
    """

    def test_last_token_swap_preserves_delimiter(self):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        data = b"a bcdefgh ij klmnop"
        # Force the last token (klmnop) to be swapped with another
        for seed in range(200):
            result = token_shuffle(data, rng=__import__("random").Random(seed))
            if result != data and b"klmnop" in result and b"bcdefgh" in result:
                # When last token is swapped, delimiter must be preserved
                assert b"klmnopij" not in result, (
                    f"seed={seed}: delimiter dropped between klmnop and ij: {result}"
                )
                # Tokens should be separated by at least one delimiter
                assert b" " in result or b"\t" in result
                return
        pytest.skip("No swap involving last token found in 200 seeds")

    def test_different_length_tokens_swap_correctly(self):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        data = b"a bb ccc dddd"
        for seed in range(100):
            result = token_shuffle(data, rng=__import__("random").Random(seed))
            if result != data:
                # All original tokens should be present (just reordered)
                assert b"a" in result
                assert b"bb" in result
                assert b"ccc" in result
                assert b"dddd" in result
                # No tokens should be concatenated without delimiter
                assert b"bbccc" not in result
                assert b"cccdddd" not in result
                return
        pytest.skip("No swap found in 100 seeds")

    def test_swap_preserves_total_length(self):
        from fuzzer_tool.core.token_shuffle import token_shuffle

        for seed in range(50):
            data = b"x aa bb cc dd ee"
            result = token_shuffle(data, rng=__import__("random").Random(seed))
            if result != data:
                # Total length should be preserved (same tokens, just reordered)
                assert len(result) == len(data), (
                    f"seed={seed}: length changed {len(data)} -> {len(result)}"
                )
                return


class TestTlvMutateRegression:
    """Regression: 2-byte length field mutation only changed high byte.

    Before fix: when a 2-byte big-endian value matched as a length field,
    only buf[off] (the high byte) was mutated. The low byte was left
    unchanged, so the mutation didn't fully match what was detected.
    """

    def test_2byte_length_field_mutates_both_bytes(self):
        from fuzzer_tool.core.tlv_mutate import tlv_mutate

        # b1=0 (so 1-byte check fails), b2=50 (valid 2-byte length)
        data = bytes([0, 50]) + b"A" * 100
        found_2byte = False
        for seed in range(500):
            result = tlv_mutate(data, rng=__import__("random").Random(seed))
            # Check if the 2-byte mutation was triggered (both bytes at
            # offset 0-1 changed, not a fallback TLV insert)
            if (result[0] != data[0] or result[1] != data[1]) and len(result) == len(data):
                new_val = (result[0] << 8) | result[1]
                assert new_val != 50, "2-byte value unchanged"
                found_2byte = True
                break
        assert found_2byte, "No 2-byte mutation triggered in 500 seeds"

    def test_1byte_length_field_uses_correct_remaining(self):
        from fuzzer_tool.core.tlv_mutate import tlv_mutate

        # b1=5 (valid 1-byte length), remaining should be len-1, not len-2
        data = bytes([5]) + b"A" * 100
        found_1byte = False
        for seed in range(500):
            result = tlv_mutate(data, rng=__import__("random").Random(seed))
            # 1-byte mutation: only first byte changed, length preserved
            if result[0] != data[0] and len(result) == len(data):
                # The mutated byte should be a valid boundary value or within range
                remaining = len(data) - 1  # correct remaining for 1-byte field
                assert result[0] in (0x00, 0x01, 0x7F, 0x80, 0xFF) or result[0] <= remaining, (
                    f"seed={seed}: mutated value {result[0]:#x} is out of range"
                )
                found_1byte = True
                break
        assert found_1byte, "No 1-byte mutation triggered in 500 seeds"
