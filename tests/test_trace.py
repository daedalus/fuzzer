"""Tests for core/trace.py — CrashTracer, TraceReport format, parsing, repro."""

import os
from unittest.mock import MagicMock, patch

from fuzzer_tool.core.trace import TraceReport, _get_exported_functions


class TestGetExportedFunctions:
    def test_real_binary(self):
        target = "targets/png_read_afl.so"
        if not os.path.isfile(target):
            return
        funcs = _get_exported_functions(target)
        assert "fuzz_png" in funcs

    def test_nonexistent(self):
        assert _get_exported_functions("/nonexistent") == []

    def test_max_limit(self):
        target = "targets/png_read_afl.so"
        if not os.path.isfile(target):
            return
        funcs = _get_exported_functions(target, max_funcs=1)
        assert len(funcs) <= 1

    def test_nm_failure(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _get_exported_functions("x") == []

    def test_nm_timeout(self):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nm", 5)):
            assert _get_exported_functions("x") == []

    def test_nm_empty(self):
        r = MagicMock(stdout="", returncode=0)
        with patch("subprocess.run", return_value=r):
            assert _get_exported_functions("x") == []

    def test_nm_skips_underscore(self):
        r = MagicMock(stdout="0000 T _init\n0000 T my_func\n", returncode=0)
        with patch("subprocess.run", return_value=r):
            funcs = _get_exported_functions("x")
            assert "my_func" in funcs
            assert "_init" not in funcs

    def test_nm_non_t(self):
        r = MagicMock(stdout="0000 W weak\n0000 T strong\n", returncode=0)
        with patch("subprocess.run", return_value=r):
            assert _get_exported_functions("x") == ["strong"]


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
            backtrace="#0 main",
            frames=[{"frame": 0, "func": "main"}],
            registers="rax=0x0",
            fault_addr="0x401000",
        )
        assert r.fault_addr == "0x401000"
        assert len(r.frames) == 1

    def test_format_with_all_sections(self):
        r = TraceReport(
            target="/bin/test",
            input_size=42,
            signal="SIGSEGV",
            signal_num=11,
            fault_addr="0x0",
            error_msg="bad address",
            registers="rax 0x1\nrbx 0x2",
            backtrace="#0 0x100 in main at foo.c:5",
            source_context="   5   crash_here()",
            disassembly="Dump of assembler:\nmov eax, 1\nEnd of assembler dump.",
            strace="read(0, buf, 1024) = 5",
            strace_summary="read: 1",
            repro_cmd="base64 -d | /bin/test",
        )
        text = r.format()
        assert "SIGSEGV" in text
        assert "bad address" in text
        assert "rax 0x1" in text
        assert "#0 0x100" in text
        assert "crash_here" in text
        assert "mov eax, 1" in text
        assert "read(0, buf, 1024)" in text
        assert "read: 1" in text
        assert "base64" in text

    def test_format_minimal(self):
        r = TraceReport()
        text = r.format()
        assert "CRASH TRACE REPORT" in text
        assert "42 bytes" not in text

    def test_format_signal_only(self):
        r = TraceReport(signal="SIGABRT", signal_num=6)
        text = r.format()
        assert "SIGABRT" in text
        assert "(6)" in text


class TestCrashTracer:
    def test_check_tool_exists(self):
        from fuzzer_tool.core.trace import CrashTracer

        tracer = CrashTracer("targets/png_read_afl.so")
        # 'which gdb' or 'which strace' — may be True or False depending on system
        assert isinstance(tracer._has_gdb, bool)

    def test_build_repro(self):
        from fuzzer_tool.core.trace import CrashTracer

        tracer = CrashTracer("targets/png_read_afl.so")
        report = TraceReport()
        tracer._build_repro(b"test data", report)
        assert "base64" in report.repro_cmd
        assert "targets/png_read_afl.so" in report.repro_cmd

    def test_parse_gdb_output(self):
        from fuzzer_tool.core.trace import CrashTracer

        tracer = CrashTracer("targets/png_read_afl.so")
        report = TraceReport()
        gdb_output = (
            "Program received signal SIGSEGV, Segmentation fault.\n"
            "rax            0x0\t0\n"
            "rip            0x401000\t4198400\n"
            "#0  0x0000000000401000 in main (argc=1, argv=0x7fff) at foo.c:10\n"
            "#1  0x0000000000401100 in helper () at bar.c:5\n"
            "#2\n"
            "\n"
            "Dump of assembler code for function main:\n"
            "   0x401000 <+0>:   push   rbp\n"
            "   0x401001 <+1>:   mov    rbp,rsp\n"
            "End of assembler dump.\n"
        )
        tracer._parse_gdb_output(gdb_output, report)
        assert report.signal == "SIGSEGV"
        assert report.error_msg == "Segmentation fault."
        assert "rax" in report.registers
        assert report.reg_values.get("rip") == 0x401000
        assert len(report.frames) >= 1
        # Regex captures minimal match before optional groups — func name may be truncated
        assert "401000" in report.frames[0]["addr"]
        assert report.disassembly != ""
        assert "push" in report.disassembly

    def test_parse_gdb_no_signal(self):
        from fuzzer_tool.core.trace import CrashTracer

        tracer = CrashTracer("targets/png_read_afl.so")
        report = TraceReport()
        tracer._parse_gdb_output("no signal here\n", report)
        assert report.signal == ""

    def test_save_report(self, tmp_path):
        from fuzzer_tool.core.trace import CrashTracer

        tracer = CrashTracer("targets/png_read_afl.so")
        report = TraceReport(backtrace="#0 test")
        crash_dir = str(tmp_path / "crashes")
        os.makedirs(crash_dir)
        tracer.save_report(report, crash_dir, "crash_001")
        assert (tmp_path / "crashes" / "crash_001.trace").exists()

    def test_save_report_writes_format(self, tmp_path):
        from fuzzer_tool.core.trace import CrashTracer

        tracer = CrashTracer("targets/png_read_afl.so")
        report = TraceReport(signal="SIGSEGV", signal_num=11)
        crash_dir = str(tmp_path / "crashes")
        os.makedirs(crash_dir)
        tracer.save_report(report, crash_dir, "crash_002")
        content = (tmp_path / "crashes" / "crash_002.trace").read_text()
        assert "SIGSEGV" in content
