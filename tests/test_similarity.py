"""Tests for core/similarity.py — Levenshtein and Hamming distance."""

import pytest

from fuzzer_tool.core.similarity import (
    crash_signature_similarity,
    edit_script_summary,
    find_nearest_bytes,
    frame_sequence_similarity,
    hamming_distance,
    hamming_distance_padded,
    hamming_similarity,
    levenshtein_align,
    levenshtein_diff_offsets,
    levenshtein_distance,
    levenshtein_similarity,
    normalize_frame,
    stack_trace_similarity,
)


class TestHammingDistance:
    def test_identical(self):
        assert hamming_distance(b"AAAA", b"AAAA") == 0

    def test_all_differ(self):
        assert hamming_distance(b"AAAA", b"BBBB") == 4

    def test_single_diff(self):
        assert hamming_distance(b"ABCD", b"ABCE") == 1

    def test_empty(self):
        assert hamming_distance(b"", b"") == 0

    def test_unequal_lengths_raises(self):
        with pytest.raises(ValueError):
            hamming_distance(b"AA", b"AAA")

    def test_binary_data(self):
        assert hamming_distance(b"\x00\xff", b"\x00\x00") == 1

    def test_single_byte(self):
        assert hamming_distance(b"\x00", b"\xff") == 1
        assert hamming_distance(b"\x00", b"\x00") == 0


class TestHammingSimilarity:
    def test_identical(self):
        assert hamming_similarity(b"AAAA", b"AAAA") == 1.0

    def test_all_differ(self):
        assert hamming_similarity(b"AAAA", b"BBBB") == 0.0

    def test_half_differ(self):
        assert hamming_similarity(b"AABB", b"ABAB") == 0.5

    def test_empty(self):
        assert hamming_similarity(b"", b"") == 0.0

    def test_unequal_lengths(self):
        assert hamming_similarity(b"AA", b"AAA") == 0.0

    def test_single_bit(self):
        assert hamming_similarity(b"\x00", b"\x01") == pytest.approx(0.0)


class TestHammingDistancePadded:
    def test_equal_length(self):
        assert hamming_distance_padded(b"AA", b"BB") == 2

    def test_padded_shorter(self):
        # A vs AB: B is zero-padded -> A vs A0 -> A vs \x41\x00
        dist = hamming_distance_padded(b"A", b"AB")
        assert dist == 1  # 'B' vs \x00

    def test_padded_both(self):
        # This shouldn't happen in practice, but test the function
        dist = hamming_distance_padded(b"A", b"")
        assert dist == 1

    def test_empty_both(self):
        assert hamming_distance_padded(b"", b"") == 0

    def test_symmetry(self):
        a, b = b"hello", b"world"
        assert hamming_distance_padded(a, b) == hamming_distance_padded(b, a)


class TestLevenshteinDistance:
    def test_identical(self):
        assert levenshtein_distance(b"AAAA", b"AAAA") == 0

    def test_single_substitution(self):
        assert levenshtein_distance(b"ABC", b"AXC") == 1

    def test_single_insertion(self):
        assert levenshtein_distance(b"AC", b"ABC") == 1

    def test_single_deletion(self):
        assert levenshtein_distance(b"ABC", b"AC") == 1

    def test_empty_a(self):
        assert levenshtein_distance(b"", b"ABC") == 3

    def test_empty_b(self):
        assert levenshtein_distance(b"ABC", b"") == 3

    def test_both_empty(self):
        assert levenshtein_distance(b"", b"") == 0

    def test_completely_different(self):
        assert levenshtein_distance(b"AAAA", b"BBBB") == 4

    def test_longer_string(self):
        assert levenshtein_distance(b"kitten", b"sitting") == 3

    def test_reversed(self):
        assert levenshtein_distance(b"ABC", b"CBA") == 2

    def test_symmetry(self):
        assert levenshtein_distance(b"hello", b"world") == levenshtein_distance(b"world", b"hello")

    def test_single_char(self):
        assert levenshtein_distance(b"a", b"b") == 1
        assert levenshtein_distance(b"a", b"a") == 0

    def test_binary_data(self):
        assert levenshtein_distance(b"\x00\x01\x02", b"\x00\x03\x02") == 1


class TestLevenshteinSimilarity:
    def test_identical(self):
        assert levenshtein_similarity(b"AAAA", b"AAAA") == 1.0

    def test_completely_different(self):
        assert levenshtein_similarity(b"AAAA", b"BBBB") == 0.0

    def test_one_edit(self):
        # "ABC" vs "AXC": distance=1, max_len=3, sim = 1 - 1/3
        assert levenshtein_similarity(b"ABC", b"AXC") == pytest.approx(2 / 3)

    def test_both_empty(self):
        assert levenshtein_similarity(b"", b"") == 1.0

    def test_empty_one(self):
        assert levenshtein_similarity(b"", b"ABC") == 0.0

    def test_unequal_length(self):
        # "AB" vs "ABCD": distance=2, max_len=4, sim = 1 - 2/4 = 0.5
        assert levenshtein_similarity(b"AB", b"ABCD") == pytest.approx(0.5)


class TestNormalizeFrame:
    def test_strips_address(self):
        assert normalize_frame("parse+0x1234") == "parse+"

    def test_strips_numbers(self):
        assert normalize_frame("func.c:42") == "func.c:"

    def test_strips_both(self):
        result = normalize_frame("#0 0x401234 in parse+0x56 (/lib/libc.so.6+0x12345)")
        assert "0x401234" not in result
        assert "0x12345" not in result
        assert "parse" in result

    def test_no_change(self):
        assert normalize_frame("main") == "main"


class TestCrashSignatureSimilarity:
    def test_identical(self):
        sig = "ASAN:heap-buffer-overflow@parse@main"
        assert crash_signature_similarity(sig, sig) == 1.0

    def test_same_func_different_offset(self):
        sig_a = "ASAN:heap-buffer-overflow@parse+0x1234@main"
        sig_b = "ASAN:heap-buffer-overflow@parse+0x5678@main"
        sim = crash_signature_similarity(sig_a, sig_b)
        assert sim > 0.8  # same function, different offset

    def test_different_error_type(self):
        sig_a = "ASAN:heap-buffer-overflow@parse@main"
        sig_b = "ASAN:heap-use-after-free@parse@main"
        sim = crash_signature_similarity(sig_a, sig_b)
        assert sim > 0.5  # same structure, different error

    def test_completely_different(self):
        sig_a = "ASAN:heap-buffer-overflow@parse@main"
        sig_b = "UBSAN:shift-exponent@compute@worker"
        sim = crash_signature_similarity(sig_a, sig_b)
        assert sim < 0.5


class TestStackTraceSimilarity:
    def test_identical(self):
        frames = ["parse", "main", "__libc_start_main"]
        assert stack_trace_similarity(frames, frames) == 1.0

    def test_similar_traces(self):
        a = ["parse", "main", "__libc_start_main"]
        b = ["parse", "handle_input", "main", "__libc_start_main"]
        sim = stack_trace_similarity(a, b)
        assert sim > 0.5

    def test_different_traces(self):
        a = ["parse", "main"]
        b = ["compute", "worker", "dispatch"]
        sim = stack_trace_similarity(a, b)
        assert sim < 0.5

    def test_empty_traces(self):
        assert stack_trace_similarity([], []) == 1.0

    def test_one_empty(self):
        assert stack_trace_similarity(["main"], []) < 0.5


class TestFindNearestBytes:
    def test_identical(self):
        target = b"AAAA"
        candidates = [b"BBBB", b"AAAA", b"CCCC"]
        idx, sim = find_nearest_bytes(target, candidates)
        assert idx == 1
        assert sim == 1.0

    def test_similar(self):
        target = b"AABBCCDD"
        candidates = [b"XXXXXXXX", b"AABBCCDE", b"YYYYYYYY"]
        idx, sim = find_nearest_bytes(target, candidates)
        assert idx == 1
        assert sim > 0.5

    def test_empty_candidates(self):
        idx, sim = find_nearest_bytes(b"AAAA", [])
        assert idx == -1
        assert sim == 0.0

    def test_unequal_lengths(self):
        target = b"AAAA"
        candidates = [b"AAA", b"AAAAA"]
        idx, sim = find_nearest_bytes(target, candidates)
        assert idx >= 0

    def test_max_check(self):
        candidates = [bytes([i % 256]) * 4 for i in range(200)]
        idx, sim = find_nearest_bytes(b"\x00\x00\x00\x00", candidates, max_check=10)
        assert 0 <= idx < 10

    def test_binary_data(self):
        target = b"\x00\x01\x02\x03"
        candidates = [b"\x00\x01\x02\x04", b"\xff\xfe\xfd\xfc"]
        idx, sim = find_nearest_bytes(target, candidates)
        assert idx == 0
        assert sim > 0.5


class TestLevenshteinAlign:
    def test_identical(self):
        script = levenshtein_align(b"ABC", b"ABC")
        assert all(op == "match" for op, _, _ in script)
        assert len(script) == 3

    def test_single_substitution(self):
        script = levenshtein_align(b"ABC", b"AXC")
        ops = [(op, pos, data) for op, pos, data in script if op != "match"]
        assert len(ops) == 1
        assert ops[0][0] == "replace"
        assert ops[0][2] == b"X"

    def test_single_insertion(self):
        script = levenshtein_align(b"AC", b"ABC")
        ops = [(op, pos, data) for op, pos, data in script if op != "match"]
        assert len(ops) == 1
        assert ops[0][0] == "insert"
        assert ops[0][2] == b"B"

    def test_single_deletion(self):
        script = levenshtein_align(b"ABC", b"AC")
        ops = [(op, pos, data) for op, pos, data in script if op != "match"]
        assert len(ops) == 1
        assert ops[0][0] == "delete"

    def test_reconstruction(self):
        """Applying the edit script should produce the target."""
        a = b"hello world"
        b = b"hello brave new world"
        script = levenshtein_align(a, b)
        # Reconstruct by applying ops
        result = bytearray(a)
        offset_delta = 0
        for op, pos, data in script:
            adjusted = pos + offset_delta
            if op == "replace":
                result[adjusted] = data[0]
            elif op == "insert":
                result[adjusted:adjusted] = data
                offset_delta += 1
            elif op == "delete":
                del result[adjusted]
                offset_delta -= 1
        assert bytes(result) == b

    def test_empty_both(self):
        script = levenshtein_align(b"", b"")
        assert len(script) == 0

    def test_empty_a(self):
        script = levenshtein_align(b"", b"ABC")
        ops = [(opname, pos, data) for opname, pos, data in script if opname != "match"]
        assert all(opname == "insert" for opname, _, _ in ops)
        assert len(ops) == 3

    def test_empty_b(self):
        script = levenshtein_align(b"ABC", b"")
        ops = [(opname, pos, data) for opname, pos, data in script if opname != "match"]
        assert all(opname == "delete" for opname, _, _ in ops)
        assert len(ops) == 3


class TestEditScriptSummary:
    def test_identical(self):
        assert edit_script_summary(b"ABC", b"ABC") == "identical"

    def test_single_substitution(self):
        summary = edit_script_summary(b"ABC", b"AXC")
        assert "substitution" in summary

    def test_insertion(self):
        summary = edit_script_summary(b"AC", b"ABC")
        assert "insertion" in summary

    def test_deletion(self):
        summary = edit_script_summary(b"ABC", b"AC")
        assert "deletion" in summary

    def test_multiple_ops(self):
        summary = edit_script_summary(b"AAAA", b"BBBB")
        assert "4 substitution" in summary


class TestLevenshteinDiffOffsets:
    def test_identical(self):
        assert levenshtein_diff_offsets(b"ABC", b"ABC") == []

    def test_substitution(self):
        offsets = levenshtein_diff_offsets(b"ABC", b"AXC")
        assert offsets == [1]

    def test_insertion_at_start(self):
        """The textbook case the old positional diff got wrong."""
        a = b"\x00" * 25
        b = b"\x00" * 13 + b"\xff" + b"\x00" * 12
        # Insert \xff at offset 13 — old code would report 24 diffs
        offsets = levenshtein_diff_offsets(a, b)
        assert len(offsets) == 1
        assert offsets[0] == 13

    def test_max_ops(self):
        offsets = levenshtein_diff_offsets(b"AAAA", b"BBBB", max_ops=2)
        assert len(offsets) == 2


class TestFrameSequenceSimilarity:
    def test_identical(self):
        frames = ["parse", "main", "__libc_start_main"]
        assert frame_sequence_similarity(frames, frames) == 1.0

    def test_order_matters(self):
        """A->B->C should differ from C->B->A."""
        a = ["A", "B", "C"]
        b = ["C", "B", "A"]
        sim = frame_sequence_similarity(a, b)
        assert sim < 0.8  # should be low, not 1.0 like Jaccard-on-sets

    def test_extra_frame(self):
        a = ["parse", "main"]
        b = ["parse", "handle_input", "main"]
        sim = frame_sequence_similarity(a, b)
        assert sim > 0.3  # one insertion — still similar

    def test_completely_different(self):
        a = ["parse", "main"]
        b = ["compute", "worker"]
        sim = frame_sequence_similarity(a, b)
        assert sim < 0.5

    def test_empty_frames(self):
        assert frame_sequence_similarity([], []) == 1.0


class TestDeltaV2:
    def test_same_length_substitution(self):
        from fuzzer_tool.adapters.filesystem import apply_delta_v2, compute_delta_v2

        parent = b"AAAA"
        child = b"ABBA"
        diff = compute_delta_v2(parent, child)
        assert diff is not None
        reconstructed = apply_delta_v2(parent, diff)
        assert reconstructed == child

    def test_insertion(self):
        from fuzzer_tool.adapters.filesystem import apply_delta_v2, compute_delta_v2

        parent = b"AC"
        child = b"ABC"
        diff = compute_delta_v2(parent, child)
        assert diff is not None
        reconstructed = apply_delta_v2(parent, diff)
        assert reconstructed == child

    def test_deletion(self):
        from fuzzer_tool.adapters.filesystem import apply_delta_v2, compute_delta_v2

        parent = b"ABC"
        child = b"AC"
        diff = compute_delta_v2(parent, child)
        assert diff is not None
        reconstructed = apply_delta_v2(parent, diff)
        assert reconstructed == child

    def test_splice_like(self):
        from fuzzer_tool.adapters.filesystem import apply_delta_v2, compute_delta_v2

        parent = b"hello world"
        child = b"hello brave new world"
        diff = compute_delta_v2(parent, child)
        assert diff is not None
        reconstructed = apply_delta_v2(parent, diff)
        assert reconstructed == child

    def test_large_diff_returns_none(self):
        from fuzzer_tool.adapters.filesystem import compute_delta_v2

        parent = b"\x00" * 100
        child = b"\xff" * 100
        diff = compute_delta_v2(parent, child)
        assert diff is None  # > 25% of child size

    def test_empty_parent(self):
        from fuzzer_tool.adapters.filesystem import apply_delta_v2, compute_delta_v2

        diff = compute_delta_v2(b"", b"ABC")
        assert diff is not None
        reconstructed = apply_delta_v2(b"", diff)
        assert reconstructed == b"ABC"
