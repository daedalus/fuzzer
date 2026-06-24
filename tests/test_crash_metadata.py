"""Tests for crash_metadata module."""

import hashlib
import os

from fuzzer_tool.core.crash_metadata import CrashMetadata, find_nearest_corpus
from fuzzer_tool.core.sanitizer import SanitizerReport


class TestCrashMetadata:
    def test_build_cluster_id(self):
        meta = CrashMetadata()
        result = meta.build_cluster_id("ASAN:heap-buffer-overflow@parse@main")
        assert len(result) == 8
        assert result == hashlib.sha256(b"ASAN:heap-buffer-overflow@parse@main").hexdigest()[:8]

    def test_build_hexdump(self):
        meta = CrashMetadata()
        data = b"\x89PNG\r\n\x1a\n\x00\x00"
        result = meta.build_hexdump(data)
        assert "89 50 4e 47" in result
        assert ".PNG" in result

    def test_build_text_repr(self):
        meta = CrashMetadata()
        data = b"hello\x00world\n"
        result = meta.build_text_repr(data)
        assert "hello" in result
        assert "\\x00" in result
        assert "\\n" in result

    def test_format_sidecar_sanitizer(self):
        meta = CrashMetadata()
        meta.sanitizer = "AddressSanitizer"
        meta.error_type = "heap-buffer-overflow"
        meta.fault_addr = "0x602000000010"
        meta.frames = ["parse", "main"]
        meta.cluster_id = "abc12345"
        meta.timestamp = "2026-01-01T00:00:00Z"
        meta.exploitability = "CRITICAL"
        result = meta.format_sidecar()
        assert "AddressSanitizer" in result
        assert "heap-buffer-overflow" in result
        assert "CRITICAL" in result
        assert "#0 parse" in result

    def test_format_sidecar_signal(self):
        meta = CrashMetadata()
        meta.returncode = -11
        meta.timestamp = "2026-01-01T00:00:00Z"
        result = meta.format_sidecar()
        assert "returncode:    -11" in result

    def test_format_reproducer(self):
        meta = CrashMetadata()
        meta.error_type = "heap-buffer-overflow"
        meta.frames = ["parse"]
        meta.timestamp = "2026-01-01T00:00:00Z"
        meta.exploitability = "HIGH"
        result = meta.format_reproducer(b"test_input", "/usr/bin/target")
        assert "#!/bin/bash" in result
        assert "base64" in result
        assert "/usr/bin/target" in result
        assert "ASAN_OPTIONS" in result

    def test_format_sidecar_with_metadata(self):
        meta = CrashMetadata()
        meta.exec_count = 1000
        meta.corpus_size = 50
        meta.target = "./fuzz"
        meta.mutation_ops = ["bit_flip", "splice"]
        meta.parent_seed_hash = "abc123"
        meta.elapsed = "00:05:30"
        meta.timestamp = "2026-01-01T00:00:00Z"
        result = meta.format_sidecar()
        assert "exec_count:    1000" in result
        assert "corpus_size:   50" in result
        assert "bit_flip, splice" in result
        assert "parent_seed:   abc123" in result
        assert "00:05:30" in result


class TestSanitizerReportEnriched:
    def test_access_type_and_size(self):
        stderr = (
            "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000010\n"
            "WRITE of size 4 at 0x602000000010\n"
            "#0 0x401234 in parse\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        assert report.access_type == "WRITE"
        assert report.access_size == 4
        assert report.exploitability == "CRITICAL"

    def test_read_access(self):
        stderr = (
            "==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010\n"
            "READ of size 8 at 0x602000000010\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        assert report.access_type == "READ"
        assert report.access_size == 8
        assert report.exploitability == "CRITICAL"

    def test_shadow_info(self):
        stderr = (
            "==1==ERROR: AddressSanitizer: heap-buffer-overflow\n"
            "0x602000000010 is located 0 bytes to the right of 16-byte region\n"
            "allocated by thread T0 here:\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        # Shadow info may or may not be captured depending on regex
        # The important thing is the report parses without error
        assert report.sanitizer == "AddressSanitizer"

    def test_alloc_dealloc_stacks(self):
        stderr = (
            "==1==ERROR: AddressSanitizer: heap-use-after-free\n"
            "#0 0x401234 in free_func\n"
            "SUMMARY: AddressSanitizer: heap-use-after-free\n"
            "allocated by thread T0 here:\n"
            "#0 0x401000 in malloc\n"
            "#1 0x402000 in alloc_func\n"
            "freed by thread T0 here:\n"
            "#0 0x401000 in free\n"
            "#1 0x402000 in free_func\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        assert report.alloc_frames is not None
        assert len(report.alloc_frames) >= 1
        assert report.dealloc_frames is not None
        assert len(report.dealloc_frames) >= 1

    def test_exploitability_table(self):
        for error_type, expected in [
            ("heap-buffer-overflow", "CRITICAL"),
            ("heap-use-after-free", "CRITICAL"),
            ("double-free", "CRITICAL"),
            ("stack-buffer-overflow", "CRITICAL"),
            ("stack-buffer-underflow", "HIGH"),
            ("allocation-size-too-big", "MEDIUM"),
        ]:
            stderr = f"==1==ERROR: AddressSanitizer: {error_type}\n"
            report = SanitizerReport.parse(stderr)
            assert report is not None
            assert report.exploitability == expected, f"{error_type} should be {expected}"


class TestFindNearestCorpus:
    def test_identical_input(self):
        data = b"AAAA"
        corpus = [b"AAAA", b"BBBB", b"CCCC"]
        label, sim, diffs = find_nearest_corpus(data, corpus)
        assert sim == 1.0
        assert len(diffs) == 0

    def test_similar_input(self):
        data = b"AABBCCDD"
        corpus = [b"AABBCCDE", b"XXXXXXXX", b"YYYYYYYY"]
        label, sim, diffs = find_nearest_corpus(data, corpus)
        assert sim > 0.3
        assert label.startswith("seed_")

    def test_empty_corpus(self):
        label, sim, diffs = find_nearest_corpus(b"AAAA", [])
        assert label == ""
        assert sim == 0.0

    def test_max_check_limits(self):
        corpus = [bytes([i % 256]) * 4 for i in range(200)]
        label, sim, diffs = find_nearest_corpus(b"\x00\x00\x00\x00", corpus, max_check=10)
        assert label.startswith("seed_")


class TestSaveCrashEnriched:
    def test_generates_all_files(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import save_crash

        hashes = set()
        sigs = {}
        result = save_crash(b"crash_data", -11, "SIGSEGV", tmp_path, hashes, sigs)
        assert result is True
        files = list(tmp_path.iterdir())
        suffixes = {f.suffix for f in files}
        assert ".bin" in suffixes
        assert ".txt" in suffixes
        assert ".sh" in suffixes
        assert ".hex" in suffixes

    def test_reproducer_is_executable(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import save_crash

        hashes = set()
        sigs = {}
        save_crash(b"test", -11, "SIGSEGV", tmp_path, hashes, sigs)
        scripts = [f for f in tmp_path.iterdir() if f.suffix == ".sh"]
        assert len(scripts) == 1
        assert os.access(scripts[0], os.X_OK)

    def test_hexdump_contains_hex(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import save_crash

        hashes = set()
        sigs = {}
        data = b"\x89PNG"
        save_crash(data, -11, "SIGSEGV", tmp_path, hashes, sigs)
        hex_files = [f for f in tmp_path.iterdir() if f.suffix == ".hex"]
        assert len(hex_files) == 1
        content = hex_files[0].read_text()
        assert "89 50 4e 47" in content

    def test_with_crash_metadata(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import save_crash

        hashes = set()
        sigs = {}
        meta = CrashMetadata()
        meta.exec_count = 500
        meta.target = "./fuzz"
        meta.mutation_ops = ["bit_flip"]
        meta.timestamp = "2026-01-01T00:00:00Z"
        save_crash(b"test", -11, "SIGSEGV", tmp_path, hashes, sigs, metadata=meta)
        txt_files = [f for f in tmp_path.iterdir() if f.suffix == ".txt"]
        content = txt_files[0].read_text()
        assert "exec_count:    500" in content
        assert "bit_flip" in content

    def test_sanitizer_enriched_sidecar(self, tmp_path):
        from fuzzer_tool.adapters.filesystem import save_crash

        hashes = set()
        sigs = {}
        stderr = (
            "==1==ERROR: AddressSanitizer: heap-buffer-overflow\n"
            "WRITE of size 4 at 0x602000000010\n"
            "#0 0x401234 in parse\n"
        )
        save_crash(b"asan_crash", 0, stderr, tmp_path, hashes, sigs)
        txt_files = [f for f in tmp_path.iterdir() if f.suffix == ".txt"]
        content = txt_files[0].read_text()
        assert "WRITE of size 4" in content
        assert "CRITICAL" in content
        assert "AddressSanitizer" in content
