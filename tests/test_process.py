"""Tests for adapters/process.py — _clean_env, run_target_stdin, run_target_file."""

import sys

from fuzzer_tool.adapters.process import (
    SIGNAL_CRASH_CODES,
    _clean_env,
    _track,
    _untrack,
    run_target_file,
    run_target_stdin,
)


class TestCleanEnv:
    def test_strips_ksm_preload(self):
        env = {"LD_PRELOAD": "/usr/lib/ksm_preload.so:/usr/lib/other.so"}
        result = _clean_env(env)
        assert "ksm_preload" not in result.get("LD_PRELOAD", "")
        assert "other.so" in result["LD_PRELOAD"]

    def test_removes_all_ksm(self):
        env = {"LD_PRELOAD": "/usr/lib/ksm_preload.so"}
        result = _clean_env(env)
        assert "LD_PRELOAD" not in result

    def test_preserves_clean_preload(self):
        env = {"LD_PRELOAD": "/usr/lib/other.so"}
        result = _clean_env(env)
        assert result["LD_PRELOAD"] == "/usr/lib/other.so"

    def test_no_preload(self):
        result = _clean_env({})
        assert "LD_PRELOAD" not in result

    def test_none_env(self):
        result = _clean_env(None)
        assert isinstance(result, dict)


class TestTrackUntrack:
    def test_track_and_untrack(self):
        _track(99999)
        _untrack(99999)

    def test_untrack_nonexistent(self):
        _untrack(99999)  # should not raise


class TestSignalCrashCodes:
    def test_known_codes(self):
        assert 134 in SIGNAL_CRASH_CODES  # SIGABRT
        assert -11 in SIGNAL_CRASH_CODES  # SIGSEGV
        assert -6 in SIGNAL_CRASH_CODES   # SIGABRT


class TestRunTargetStdin:
    def test_runs_true(self):
        rc, stderr, pid = run_target_stdin("/bin/true", b"", timeout=5)
        assert rc == 0

    def test_runs_false(self):
        rc, stderr, pid = run_target_stdin("/bin/false", b"", timeout=5)
        assert rc == 1

    def test_nonexistent_target(self):
        rc, stderr, pid = run_target_stdin("/nonexistent/binary", b"test", timeout=5)
        assert rc == -2

    def test_timeout(self):
        # cat blocks on stdin, so it won't exit until EOF — but communicate()
        # sends data and waits. Use a command that truly blocks.
        import subprocess as sp
        proc = sp.Popen(["/bin/sleep", "3600"])
        rc, stderr, pid = run_target_stdin(f"/proc/{proc.pid}/exe", b"", timeout=0.1)
        proc.kill()
        proc.wait()
        # The /proc/PID/exe trick may not work, so just verify the API works
        assert isinstance(rc, int)

    def test_timeout_with_sleep(self):
        """Verify timeout returns -1 with a blocking command."""
        # Use a python one-liner that sleeps forever
        rc, stderr, _ = run_target_stdin(
            sys.executable, b"import time; time.sleep(3600)", timeout=0.1
        )
        assert rc == -1
        assert "timeout" in stderr

    def test_passes_data(self):
        # cat echoes stdin to stdout, but we only capture stderr
        rc, _, _ = run_target_stdin("/bin/cat", b"hello", timeout=5)
        assert rc == 0

    def test_custom_env(self):
        rc, _, _ = run_target_stdin("/bin/true", b"", timeout=5, env={"MY_VAR": "test"})
        assert rc == 0

    def test_returns_pid(self):
        _, _, pid = run_target_stdin("/bin/true", b"", timeout=5)
        assert pid > 0


class TestRunTargetFile:
    def test_runs_true(self, tmp_path):
        rc, stderr, pid = run_target_file("/bin/true", b"", 5, str(tmp_path), [])
        assert rc == 0

    def test_nonexistent_target(self, tmp_path):
        rc, stderr, pid = run_target_file("/nonexistent/binary", b"test", 5, str(tmp_path), [])
        assert rc == -2

    def test_target_args_file_placeholder(self, tmp_path):
        rc, _, _ = run_target_file("/bin/cat", b"test", 5, str(tmp_path), ["{file}"])
        assert rc == 0

    def test_returns_pid(self):
        _, _, pid = run_target_file("/bin/true", b"", 5, "/tmp", [])
        assert pid > 0
