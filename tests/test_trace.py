"""Tests for core/trace.py — CrashTracer, TraceReport, _get_exported_functions."""

import os
from unittest.mock import MagicMock, patch

from fuzzer_tool.core.trace import TraceReport, _get_exported_functions


class TestGetExportedFunctions:
    def test_real_binary(self):
        """Test with the actual compiled test target."""
        import os
        target = "targets/png_read_afl.so"
        if not os.path.isfile(target):
            return
        funcs = _get_exported_functions(target)
        assert isinstance(funcs, list)
        # Should find fuzz_png and possibly main
        assert "fuzz_png" in funcs

    def test_nonexistent_binary(self):
        funcs = _get_exported_functions("/nonexistent/binary")
        assert funcs == []

    def test_max_funcs_limit(self):
        import os
        target = "targets/png_read_afl.so"
        if not os.path.isfile(target):
            return
        funcs = _get_exported_functions(target, max_funcs=1)
        assert len(funcs) <= 1

    def test_nm_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            funcs = _get_exported_functions("some_target")
            assert funcs == []

    def test_nm_timeout(self):
        import subprocess
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="nm", timeout=5)
            funcs = _get_exported_functions("some_target")
            assert funcs == []

    def test_nm_empty_output(self):
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = ""
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            funcs = _get_exported_functions("some_target")
            assert funcs == []

    def test_nm_skips_short_names(self):
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "0000000000400000 T _init\n0000000000401000 T my_func\n"
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            funcs = _get_exported_functions("some_target")
            assert "my_func" in funcs
            assert "_init" not in funcs  # starts with _ or too short

    def test_nm_skips_non_t_symbols(self):
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "0000000000400000 W weak_func\n0000000000401000 T strong_func\n"
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            funcs = _get_exported_functions("some_target")
            assert funcs == ["strong_func"]


class TestTraceReport:
    def test_defaults(self):
        r = TraceReport()
        assert r.backtrace == ""
        assert r.frames == []
        assert r.registers == ""
        assert r.disassembly == ""
        assert r.fault_addr == ""
        assert r.target == ""

    def test_fields(self):
        r = TraceReport(
            backtrace="#0 0x401000 in main",
            frames=[{"frame": 0, "func": "main"}],
            registers="rax=0x0",
            fault_addr="0x401000",
        )
        assert "main" in r.backtrace
        assert len(r.frames) == 1
        assert r.fault_addr == "0x401000"

    def test_format(self):
        r = TraceReport(
            backtrace="#0 main at foo.c:10",
            signal="SIGSEGV",
            signal_num=11,
        )
        text = r.format()
        assert "main" in text
        assert "SIGSEGV" in text

    def test_format_empty(self):
        r = TraceReport()
        text = r.format()
        assert isinstance(text, str)

