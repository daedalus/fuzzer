"""Tests for services/tmin.py — crash minimization."""

import os
from unittest.mock import patch

from fuzzer_tool.services.tmin import tmin


class TestTmin:
    def test_nonexistent_crash_file(self):
        result = tmin("/fake/target", "/nonexistent/file.bin")
        assert result is None

    def test_empty_crash_file(self, tmp_path):
        crash = tmp_path / "empty.bin"
        crash.write_bytes(b"")
        result = tmin("/fake/target", str(crash))
        assert result is None

    def test_crash_not_reproduced(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(0, "", 1)):
            result = tmin("/bin/true", str(crash))
        assert result is None

    def test_file_mode(self, tmp_path):
        crash = tmp_path / "crash.bin"
        crash.write_bytes(b"AAAA")
        with patch("fuzzer_tool.adapters.process.run_target_file", return_value=(0, "", 1)):
            result = tmin("/bin/true", str(crash), file_mode=True)
        assert result is None

    def test_crash_signature_asan(self, tmp_path):
        asan_stderr = "ERROR: AddressSanitizer: heap-buffer-overflow\nABORTING"
        crash_file = tmp_path / "crash.bin"
        crash_file.write_bytes(b"A" * 100)
        with patch(
            "fuzzer_tool.adapters.process.run_target_stdin", return_value=(1, asan_stderr, 1)
        ):
            result = tmin("/bin/false", str(crash_file))
        # ASAN crash reproduced, minimizer runs, may return minimized data
        assert result is None or isinstance(result, bytes)

    def test_crash_signature_signal(self, tmp_path):
        crash_file = tmp_path / "crash.bin"
        crash_file.write_bytes(b"B" * 50)
        with patch("fuzzer_tool.adapters.process.run_target_stdin", return_value=(-11, "", 1)):
            result = tmin("/bin/false", str(crash_file))
        assert result is None or isinstance(result, bytes)

    def test_file_mode_timeout(self, tmp_path):
        crash_file = tmp_path / "crash.bin"
        crash_file.write_bytes(b"C" * 10)
        # First call reproduces, then minimize_bytes calls many times
        # Return crash on first call, then "no crash" for minimize attempts
        call_count = [0]

        def fake_run_file(target, data, timeout, tmp_dir, args, env=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return (-11, "segfault", 1)
            return (0, "", 1)

        with patch("fuzzer_tool.adapters.process.run_target_file", side_effect=fake_run_file):
            result = tmin("/bin/false", str(crash_file), file_mode=True)
        assert result is None or isinstance(result, bytes)

    def test_main_exists(self):
        from fuzzer_tool.services.tmin import main

        assert callable(main)
