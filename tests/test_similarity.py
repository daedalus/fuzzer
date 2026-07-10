"""Tests for core/similarity.py — Levenshtein and Hamming distance."""

import pytest

from fuzzer_tool.core.similarity import (
    crash_signature_similarity,
    find_nearest_bytes,
    hamming_distance,
    hamming_distance_padded,
    hamming_similarity,
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
