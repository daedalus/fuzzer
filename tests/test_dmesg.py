"""Tests for core/dmesg.py — kernel crash log parsing."""

from unittest.mock import MagicMock, patch

from fuzzer_tool.core.dmesg import DmesgParser, KernelCrash


class TestKernelCrash:
    def test_defaults(self):
        kc = KernelCrash(timestamp=0.0, raw_message="")
        assert kc.crash_type == ""
        assert kc.pid is None
        assert kc.ip is None

    def test_fields(self):
        kc = KernelCrash(
            timestamp=1.5,
            crash_type="segfault",
            raw_message="segfault at 0x0",
            pid=42,
            process_name="fuzz",
            ip="0xdeadbeef",
            sp="0x7fff0000",
            error_code=14,
        )
        assert kc.timestamp == 1.5
        assert kc.crash_type == "segfault"
        assert kc.pid == 42
        assert kc.ip == "0xdeadbeef"


class TestDmesgParser:
    def test_init(self):
        dp = DmesgParser()
        assert dp._last_ts == 0.0

    def test_parse_timestamp_time(self):
        dp = DmesgParser()
        ts = dp._parse_timestamp({"time": 3.14})
        assert ts == 3.14

    def test_parse_timestamp_ts(self):
        dp = DmesgParser()
        ts = dp._parse_timestamp({"ts": 2.5})
        assert ts == 2.5

    def test_parse_timestamp_microseconds(self):
        dp = DmesgParser()
        ts = dp._parse_timestamp({"__REALTIME_TIMESTAMP": "1700000000000000"})
        assert ts is not None
        assert ts == 1700000000.0

    def test_parse_timestamp_invalid(self):
        dp = DmesgParser()
        assert dp._parse_timestamp({"time": "not_a_number"}) is None
        assert dp._parse_timestamp({}) is None

    def test_match_crash_segfault(self):
        dp = DmesgParser()
        msg = "[  123.456] segfault at 0 ip 0000000000401000 sp 00007fff12345678 error 14"
        kc = dp._match_crash(123.456, msg)
        assert kc is not None
        assert kc.crash_type == "segfault"

    def test_match_crash_trap(self):
        dp = DmesgParser()
        kc = dp._match_crash(1.0, "[1.0] trap divide error")
        assert kc is not None
        assert kc.crash_type == "trap"

    def test_match_crash_gp_fault(self):
        dp = DmesgParser()
        kc = dp._match_crash(1.0, "[1.0] general protection fault")
        assert kc is not None
        assert kc.crash_type == "gp_fault"

    def test_match_crash_kernel_panic(self):
        dp = DmesgParser()
        kc = dp._match_crash(1.0, "[1.0] Kernel panic - not syncing")
        assert kc is not None
        assert kc.crash_type == "kernel_panic"

    def test_match_crash_oom(self):
        dp = DmesgParser()
        kc = dp._match_crash(1.0, "[1.0] Out of memory: Killed process 1234")
        assert kc is not None
        assert kc.crash_type == "oom"

    def test_match_crash_bug(self):
        dp = DmesgParser()
        kc = dp._match_crash(1.0, "[1.0] BUG: unable to handle page fault")
        assert kc is not None
        assert kc.crash_type == "bug"

    def test_match_crash_kasan(self):
        dp = DmesgParser()
        kc = dp._match_crash(1.0, "[1.0] KASAN: use-after-free")
        assert kc is not None
        assert kc.crash_type == "kasan"

    def test_match_crash_unhandled_type(self):
        dp = DmesgParser()
        # Match via BUG: pattern but "BUG" not in the message classification
        # Actually we need to test the "other" branch — use a pattern that
        # matches but doesn't hit any specific type keyword
        kc = dp._match_crash(1.0, "[1.0] internal error: Oops")
        assert kc is not None
        assert kc.crash_type == "other"

    def test_match_crash_no_match(self):
        dp = DmesgParser()
        assert dp._match_crash(1.0, "[1.0] normal kernel log message") is None

    def test_match_crash_with_pid(self):
        dp = DmesgParser()
        kc = dp._match_crash(
            1.0, "[1.0] segfault at 0 ip deadbeef sp deadbeef error 14", pid=99, proc="test"
        )
        assert kc is not None
        assert kc.pid == 99
        assert kc.process_name == "test"

    def test_match_crash_error_code_hex(self):
        dp = DmesgParser()
        kc = dp._match_crash(
            1.0,
            "[1.0] segfault at 0 ip 0000000000401000 sp 00007fff error 14"
        )
        assert kc is not None
        assert kc.error_code == 14

    def test_drain_stream_empty(self):
        dp = DmesgParser()
        result = dp.drain_stream(pid=None)
        assert result == []

    def test_stop_stream_noop(self):
        dp = DmesgParser()
        dp.stop_stream()  # should not raise

    def test_drain_stream_with_crashes(self):
        dp = DmesgParser()
        kc = KernelCrash(timestamp=1.0, raw_message="segfault", crash_type="segfault", pid=42)
        dp._stream_buffer.append(kc)
        result = dp.drain_stream()
        assert len(result) == 1
        assert result[0].pid == 42
        assert len(dp._stream_buffer) == 0  # buffer cleared

    def test_drain_stream_pid_filter(self):
        dp = DmesgParser()
        kc1 = KernelCrash(timestamp=1.0, raw_message="a", pid=1)
        kc2 = KernelCrash(timestamp=2.0, raw_message="b", pid=2)
        kc3 = KernelCrash(timestamp=3.0, raw_message="c", pid=1)
        dp._stream_buffer.extend([kc1, kc2, kc3])
        result = dp.drain_stream(pid=1)
        assert len(result) == 2
        assert all(kc.pid == 1 for kc in result)

    def test_drain_stream_updates_last_ts(self):
        dp = DmesgParser()
        kc = KernelCrash(timestamp=5.0, raw_message="x")
        dp._stream_buffer.append(kc)
        dp.drain_stream()
        assert dp._last_ts == 5.0

    def test_is_available_cached(self):
        dp = DmesgParser()
        dp._available = True
        assert dp.is_available() is True
        dp._available = False
        assert dp.is_available() is False

    def test_is_available_exception(self):
        dp = DmesgParser()
        with patch("fuzzer_tool.core.dmesg.subprocess.run", side_effect=FileNotFoundError):
            result = dp.is_available()
        assert result is False
        assert dp._warned is True

    def test_start_stream_not_available(self):
        dp = DmesgParser()
        dp._available = False
        assert dp.start_stream() is False

    def test_start_stream_already_running(self):
        dp = DmesgParser()
        dp._stream_proc = MagicMock()
        assert dp.start_stream() is True

    def test_stop_stream_active(self):
        dp = DmesgParser()
        mock_proc = MagicMock()
        mock_proc.wait.return_value = None
        dp._stream_proc = mock_proc
        mock_thread = MagicMock()
        dp._stream_thread = mock_thread
        dp.stop_stream()
        mock_proc.terminate.assert_called_once()
        mock_thread.join.assert_called_once()
        assert dp._stream_proc is None
        assert dp._stream_thread is None

    def test_stop_stream_kill_on_timeout(self):
        dp = DmesgParser()
        import subprocess
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="dmesg", timeout=2)
        dp._stream_proc = mock_proc
        dp.stop_stream()
        mock_proc.kill.assert_called_once()

    def test_process_entry_crash(self):
        dp = DmesgParser()
        entry = {"msg": "segfault at 0 ip 0 sp 0 error 14", "time": 1.0, "pid": 42}
        dp._process_entry(entry)
        assert len(dp._stream_buffer) == 1
        assert dp._stream_buffer[0].crash_type == "segfault"
        assert dp._stream_buffer[0].pid == 42

    def test_process_entry_no_crash(self):
        dp = DmesgParser()
        entry = {"msg": "normal log line", "time": 1.0}
        dp._process_entry(entry)
        assert len(dp._stream_buffer) == 0

    def test_process_entry_pid_from_msg(self):
        dp = DmesgParser()
        entry = {"msg": "python3[999]: segfault at 0 ip 0 sp 0 error 14", "time": 1.0}
        dp._process_entry(entry)
        assert len(dp._stream_buffer) == 1
        assert dp._stream_buffer[0].pid == 999
        assert dp._stream_buffer[0].process_name == "python3"

    def test_process_entry_message_fields(self):
        dp = DmesgParser()
        entry = {"MESSAGE": "KASAN: use-after-free", "__REALTIME_TIMESTAMP": "2000000"}
        dp._process_entry(entry)
        assert len(dp._stream_buffer) == 1
        assert dp._stream_buffer[0].crash_type == "kasan"

    def test_stream_reader(self):
        dp = DmesgParser()
        mock_proc = MagicMock()
        mock_proc.stdout = [
            "[  1.0] segfault at 0 ip 0 sp 0 error 14\n",
            "[  2.0] normal message\n",
            "[  3.0] python3[42]: KASAN: use-after-free\n",
            "\n",
        ]
        dp._stream_proc = mock_proc
        dp._stream_reader()
        assert len(dp._stream_buffer) == 2
        types = {kc.crash_type for kc in dp._stream_buffer}
        assert "segfault" in types
        assert "kasan" in types

    def test_stream_reader_no_proc(self):
        dp = DmesgParser()
        dp._stream_proc = None
        dp._stream_reader()  # should not raise

    def test_match_crash_oom_type(self):
        dp = DmesgParser()
        kc = dp._match_crash(1.0, "[1.0] Out of memory: Killed process 1234")
        assert kc is not None
        assert kc.crash_type == "oom"
