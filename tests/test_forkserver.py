"""Unit tests for adapters/forkserver.py — forkserver state management."""

import os
import tempfile

import pytest

from fuzzer_tool.adapters.forkserver import ForkserverRunner, _ensure_compiled


class TestEnsureCompiled:
    def test_returns_none_when_no_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "fuzzer_tool.adapters.forkserver._FUZZ_LOADER_BIN",
            str(tmp_path / "nonexistent"),
        )
        assert _ensure_compiled() is None

    def test_returns_binary_when_already_compiled(self, tmp_path, monkeypatch):
        bin_path = tmp_path / "fuzz_loader"
        bin_path.write_bytes(b"\x7fELF")
        os.chmod(bin_path, 0o755)
        monkeypatch.setattr(
            "fuzzer_tool.adapters.forkserver._FUZZ_LOADER_BIN",
            str(bin_path),
        )
        assert _ensure_compiled() == str(bin_path)


class TestForkserverRunner:
    def test_initial_state(self):
        r = ForkserverRunner("/fake/target")
        assert r._ready is False
        assert r._proc is None
        assert r._last_bitmap is None

    def test_run_one_not_ready(self):
        r = ForkserverRunner("/fake/target")
        rc, bitmap = r.run_one(b"test")
        assert rc == -2
        assert bitmap is None

    def test_stop_without_start(self):
        r = ForkserverRunner("/fake/target")
        r.stop()  # should not raise
        assert r._ready is False
        assert r._proc is None

    def test_stderr_output_empty(self):
        r = ForkserverRunner("/fake/target")
        assert r.stderr_output() == ""

    def test_stderr_output_capped(self):
        r = ForkserverRunner("/fake/target")
        r._stderr_lines = [f"line_{i}" for i in range(30)]
        output = r.stderr_output()
        lines = output.split("\n")
        assert len(lines) == 20
        assert lines[0] == "line_10"

    def test_start_with_missing_binary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "fuzzer_tool.adapters.forkserver._FUZZ_LOADER_BIN",
            str(tmp_path / "nonexistent"),
        )
        r = ForkserverRunner("/fake/target")
        assert r.start() is False

    def test_bitmap_out_cleanup_on_stop(self, tmp_path):
        r = ForkserverRunner("/fake/target")
        fd, bmp = tempfile.mkstemp(suffix=".bmp")
        os.close(fd)
        r._bitmap_out = bmp
        r._proc = None  # never started
        r.stop()
        # stop() only cleans up _bitmap_out when _proc was set (process started)
        # Since _proc is None, bitmap_out is not cleaned — this is expected
        # Test that stop() at least doesn't crash
        assert r._ready is False

    def test_bitmap_out_cleanup_with_proc(self, tmp_path):
        r = ForkserverRunner("/fake/target")
        fd, bmp = tempfile.mkstemp(suffix=".bmp")
        os.close(fd)
        r._bitmap_out = bmp
        # Simulate a dead process
        class FakeProc:
            def poll(self): return 0
            class stdin:
                @staticmethod
                def write(data): pass
                @staticmethod
                def flush(): pass
            def wait(self, timeout=0): pass
        r._proc = FakeProc()
        r.stop()
        assert r._bitmap_out is None
        assert not os.path.exists(bmp)

    def test_restart_flag_prevents_recursive_restart(self):
        r = ForkserverRunner("/fake/target")
        r._restarting = True
        r._ready = False
        r._proc = None
        rc, bitmap = r.run_one(b"test")
        assert rc == -2
        assert bitmap is None
