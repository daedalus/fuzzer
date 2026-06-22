"""Tests for SanitizerReport."""

from fuzzer_tool.core.sanitizer import SanitizerReport


class TestSanitizerReport:
    def test_parse_asan(self):
        stderr = (
            "==1234==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x7f0000000000\n"
            "    #0 0x401234 in foo\n"
            "    #1 0x401260 in bar\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        assert report.sanitizer == "AddressSanitizer"
        assert report.error_type == "heap-buffer-overflow"
        assert report.fault_addr == "0x7f0000000000"
        assert len(report.frames) >= 1
        assert report.is_valid()

    def test_parse_msan(self):
        stderr = (
            "==1234==ERROR: MemorySanitizer: use-of-uninitialized-value\n    #0 0x401000 in main\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        assert report.sanitizer == "MemorySanitizer"
        assert report.error_type == "use-of-uninitialized-value"

    def test_parse_no_match(self):
        report = SanitizerReport.parse("normal output\n")
        assert report is None

    def test_signature_builds(self):
        stderr = (
            "==1234==ERROR: AddressSanitizer: heap-use-after-free\n"
            "    #0 0x401000 in func_a\n"
            "    #1 0x402000 in func_b\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        assert "AddressSanitizer:heap-use-after-free" in report.signature
        assert "func_a" in report.signature
        assert report.frames[0] == "func_a"

    def test_is_valid_empty(self):
        report = SanitizerReport("", "", "", [], "")
        assert not report.is_valid()

    def test_parse_ubsan(self):
        stderr = "==1234==ERROR: UndefinedBehaviorSanitizer: undefined\n"
        report = SanitizerReport.parse(stderr)
        assert report is not None
        assert report.sanitizer == "UndefinedBehaviorSanitizer"
        assert report.error_type == "undefined"
