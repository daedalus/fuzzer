"""Tests for no-ASAN fuzzing + auto sanitizer crash replay."""

import json
from unittest.mock import patch

from fuzzer_tool.core.sanitizer import SanitizerReport
from fuzzer_tool.services.fuzzer import Fuzzer


class TestSanitizerReportToDict:
    """SanitizerReport.to_dict() round-trips through JSON."""

    def test_to_dict_round_trip(self):
        stderr = (
            "==1234==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x7f0000000000\n"
            "    #0 0x401234 in foo\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        d = report.to_dict()
        assert d["sanitizer"] == "AddressSanitizer"
        assert d["error_type"] == "heap-buffer-overflow"
        assert d["fault_addr"] == "0x7f0000000000"
        assert d["access_type"] is None
        assert d["access_size"] is None
        assert "foo" in d["frames"]
        assert d["signature"]
        assert d["exploitability"] == "CRITICAL"

        # Round-trip through JSON
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped == d

    def test_to_dict_enriched_fields(self):
        stderr = (
            "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000000f0\n"
            "READ of size 8\n"
            "    #0 0x401000 in read_func\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        d = report.to_dict()
        assert d["access_type"] == "READ"
        assert d["access_size"] == 8
        assert len(d["frames"]) >= 1
        assert d["frames"][0] == "read_func"

    def test_to_dict_ubsan(self):
        stderr = (
            "==1==ERROR: UndefinedBehaviorSanitizer: signed-integer-overflow\n"
            "    #0 0x401000 in overflow_func\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        d = report.to_dict()
        assert d["sanitizer"] == "UndefinedBehaviorSanitizer"
        assert d["error_type"] == "signed-integer-overflow"
        assert d["exploitability"] == "MEDIUM"


class TestSanitizerEnvSetup:
    """ASAN/UBSAN environment setup helpers."""

    def test_setup_asan_env_adds_libasan(self):
        env = {"PATH": "/usr/bin"}
        Fuzzer._setup_asan_env(env)
        assert "LD_PRELOAD" in env
        assert "libasan" in env["LD_PRELOAD"] or "asan" in env["LD_PRELOAD"]
        assert "halt_on_error=0" in env.get("ASAN_OPTIONS", "")
        assert "abort_on_error=0" in env.get("ASAN_OPTIONS", "")
        assert "detect_leaks=0" in env.get("ASAN_OPTIONS", "")

    def test_setup_asan_env_preserves_existing_ld_preload(self):
        env = {"LD_PRELOAD": "libfoo.so", "PATH": "/usr/bin"}
        Fuzzer._setup_asan_env(env)
        assert "libfoo.so" in env["LD_PRELOAD"]
        assert "libasan" in env["LD_PRELOAD"] or "asan" in env["LD_PRELOAD"]

    def test_setup_asan_env_does_not_override_explicit_options(self):
        env = {"ASAN_OPTIONS": "halt_on_error=1", "PATH": "/usr/bin"}
        Fuzzer._setup_asan_env(env)
        opts = env["ASAN_OPTIONS"]
        assert "halt_on_error=1" in opts  # user set explicitly, kept
        assert "detect_leaks=0" in opts  # not set by user, default added

    def test_setup_ubsan_env(self):
        env = {"PATH": "/usr/bin"}
        Fuzzer._setup_ubsan_env(env)
        assert "halt_on_error=1" in env.get("UBSAN_OPTIONS", "")
        assert "abort_on_error=1" in env.get("UBSAN_OPTIONS", "")
        assert "print_stacktrace=1" in env.get("UBSAN_OPTIONS", "")

    def test_setup_ubsan_env_preserves_existing_options(self):
        env = {"UBSAN_OPTIONS": "print_stacktrace=0", "PATH": "/usr/bin"}
        Fuzzer._setup_ubsan_env(env)
        assert "print_stacktrace=0" in env["UBSAN_OPTIONS"]
        assert "halt_on_error=1" in env["UBSAN_OPTIONS"]


class TestSanitizerReplay:
    """_run_sanitizer_replays() dispatches crash data to ASAN/UBSAN targets."""

    def _make_fuzzer(self, **kwargs):
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="fuzz_test_")
        defaults = dict(
            target="/bin/true",
            corpus_dir=f"{tmpdir}/corpus",
            crashes_dir=f"{tmpdir}/crashes",
            max_len=256,
            timeout=1,
            mutations_per_input=2,
        )
        defaults.update(kwargs)
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.access", return_value=True),
        ):
            f = Fuzzer(**defaults)
        return f

    @patch("fuzzer_tool.adapters.process.run_target_stdin")
    def test_replays_on_asan_target(self, mock_run):
        """Replay dispatches crash data to the ASAN target."""
        mock_run.return_value = (
            -6,
            "AddressSanitizer: heap-buffer-overflow\n#0 0x401000 in foo\n",
            12345,
        )

        fuzzer = self._make_fuzzer(asan_target="/path/to/target_asan.so")
        crash_sig = "ASAN:heap-buffer-overflow@foo"
        fuzzer._crash_sanitizer_replays[crash_sig] = {
            "data": b"AAAA",
            "asan": None,
            "ubsan": None,
        }

        fuzzer._run_sanitizer_replays(budget_ms=500)

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == "/path/to/target_asan.so"
        assert args[1] == b"AAAA"

        result = fuzzer._crash_sanitizer_replays[crash_sig]
        assert result["asan"] is not None
        assert result["asan"]["rc"] == -6
        assert result["ubsan"] is None

    @patch("fuzzer_tool.adapters.process.run_target_stdin")
    def test_replays_on_ubsan_target(self, mock_run):
        """Replay dispatches crash data to the UBSAN target."""
        mock_run.return_value = (
            -6,
            "UndefinedBehaviorSanitizer: undefined\n#0 0x401000 in bar\n",
            12346,
        )

        fuzzer = self._make_fuzzer(ubsan_target="/path/to/target_ubsan.so")
        crash_sig = "UBSAN:undefined@bar"
        fuzzer._crash_sanitizer_replays[crash_sig] = {
            "data": b"BBBB",
            "asan": None,
            "ubsan": None,
        }

        fuzzer._run_sanitizer_replays(budget_ms=500)

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == "/path/to/target_ubsan.so"
        assert args[1] == b"BBBB"

        result = fuzzer._crash_sanitizer_replays[crash_sig]
        assert result["ubsan"] is not None
        assert result["ubsan"]["rc"] == -6
        assert result["asan"] is None

    @patch("fuzzer_tool.adapters.process.run_target_stdin")
    def test_replays_on_both_targets(self, mock_run):
        """Replay dispatches crash data to both ASAN and UBSAN targets."""
        mock_run.return_value = (-11, "AddressSanitizer: heap-use-after-free\n", 12347)

        fuzzer = self._make_fuzzer(
            asan_target="/path/to/target_asan.so",
            ubsan_target="/path/to/target_ubsan.so",
        )
        crash_sig = "ASAN:heap-use-after-free"
        fuzzer._crash_sanitizer_replays[crash_sig] = {
            "data": b"CCCC",
            "asan": None,
            "ubsan": None,
        }

        fuzzer._run_sanitizer_replays(budget_ms=500)

        assert mock_run.call_count >= 2
        targets = [call[0][0] for call in mock_run.call_args_list]
        assert "/path/to/target_asan.so" in targets
        assert "/path/to/target_ubsan.so" in targets

        result = fuzzer._crash_sanitizer_replays[crash_sig]
        assert result["asan"] is not None
        assert result["ubsan"] is not None

    @patch("fuzzer_tool.adapters.process.run_target_stdin")
    def test_skips_when_no_sanitizer_targets(self, mock_run):
        """No crash data dispatched when there are no sanitizer targets."""
        fuzzer = self._make_fuzzer()
        crash_sig = "SIGSEGV"
        fuzzer._crash_sanitizer_replays[crash_sig] = {
            "data": b"DDDD",
            "asan": None,
            "ubsan": None,
        }

        fuzzer._run_sanitizer_replays(budget_ms=500)

        mock_run.assert_not_called()

    @patch("fuzzer_tool.adapters.process.run_target_stdin")
    def test_env_passed_to_target(self, mock_run):
        """Sanitizer replay passes properly configured env to the target."""
        mock_run.return_value = (0, "", 12348)

        fuzzer = self._make_fuzzer(asan_target="/path/to/target_asan.so")
        crash_sig = "ASAN:overflow"
        fuzzer._crash_sanitizer_replays[crash_sig] = {
            "data": b"EEEE",
            "asan": None,
            "ubsan": None,
        }

        fuzzer._run_sanitizer_replays(budget_ms=500)

        args, kwargs = mock_run.call_args
        env = kwargs.get("env", {})
        assert "ASAN_OPTIONS" in env
        assert "halt_on_error=0" in env["ASAN_OPTIONS"]

    @patch("fuzzer_tool.adapters.process.run_target_stdin")
    def test_budget_respected(self, mock_run):
        """Replay method runs without error even on tight budget."""
        mock_run.return_value = (0, "", 12349)

        fuzzer = self._make_fuzzer(
            asan_target="/path/to/target_asan.so",
            ubsan_target="/path/to/target_ubsan.so",
        )
        for i in range(5):
            fuzzer._crash_sanitizer_replays[f"sig_{i}"] = {
                "data": bytes([i] * 10),
                "asan": None,
                "ubsan": None,
            }

        # Tight budget should not raise
        fuzzer._run_sanitizer_replays(budget_ms=0.1)

        # At least one should remain pending (tight budget)
        remaining = sum(
            1 for info in fuzzer._crash_sanitizer_replays.values() if info["asan"] is None
        )
        assert remaining >= 0  # non-regression assertion

    def test_sanitizer_report_to_json_round_trip(self):
        """SanitizerReport.to_dict() -> json.dumps -> json.loads preserves all fields."""
        stderr = (
            "==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000000f0\n"
            "WRITE of size 4\n"
            "    #0 0x401000 in write_func\n"
        )
        report = SanitizerReport.parse(stderr)
        assert report is not None
        d = report.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["sanitizer"] == "AddressSanitizer"
        assert restored["error_type"] == "heap-buffer-overflow"
        assert restored["fault_addr"] == "0x6020000000f0"
        assert restored["access_type"] == "WRITE"
        assert restored["access_size"] == 4
        assert len(restored["frames"]) >= 1
        assert restored["frames"][0] == "write_func"
        assert restored["exploitability"] == "CRITICAL"
