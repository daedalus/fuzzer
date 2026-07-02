"""Tests for core/dmesg.py — kernel crash log parsing."""

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

    def test_match_crash_bug(self):
        dp = DmesgParser()
        kc = dp._match_crash(1.0, "[1.0] BUG: unable to handle page fault")
        assert kc is not None

    def test_match_crash_kasan(self):
        dp = DmesgParser()
        kc = dp._match_crash(1.0, "[1.0] KASAN: use-after-free")
        assert kc is not None

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
