"""Unit tests for adapters/forkserver.py — forkserver state management."""

import os
import tempfile

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
        src_path = tmp_path / "fuzz_loader.c"
        bin_path.write_bytes(b"\x7fELF")
        os.chmod(bin_path, 0o755)
        # Source older than the binary: nothing to rebuild.
        src_path.write_text("int main(void) { return 0; }\n")
        os.utime(src_path, (1_000_000, 1_000_000))
        os.utime(bin_path, (2_000_000, 2_000_000))
        monkeypatch.setattr(
            "fuzzer_tool.adapters.forkserver._FUZZ_LOADER_BIN",
            str(bin_path),
        )
        assert _ensure_compiled() == str(bin_path)

    def test_rebuilds_when_source_is_newer(self, tmp_path, monkeypatch):
        """A binary older than its source must be rebuilt, not handed back.

        The binary is gitignored, so it outlives every checkout and pull while
        fuzz_loader.c changes underneath it. Returning it on existence alone
        meant source edits were silently ignored for the life of the working
        tree -- the loader kept speaking its old protocol, and the only
        symptom was tests failing against assertions the current source
        already satisfies.
        """
        bin_path = tmp_path / "fuzz_loader"
        src_path = tmp_path / "fuzz_loader.c"
        bin_path.write_bytes(b"\x7fELF stale")
        os.chmod(bin_path, 0o755)
        src_path.write_text("int main(void) { return 0; }\n")
        # Binary predates the source.
        os.utime(bin_path, (1_000_000, 1_000_000))
        os.utime(src_path, (2_000_000, 2_000_000))
        monkeypatch.setattr(
            "fuzzer_tool.adapters.forkserver._FUZZ_LOADER_BIN",
            str(bin_path),
        )
        assert _ensure_compiled() == str(bin_path)
        assert bin_path.read_bytes() != b"\x7fELF stale", "stale binary was not rebuilt"
        assert os.path.getmtime(bin_path) >= os.path.getmtime(src_path)

    def test_keeps_binary_when_source_is_absent(self, tmp_path, monkeypatch):
        """An installed tree may ship the binary with no .c beside it."""
        bin_path = tmp_path / "fuzz_loader"
        bin_path.write_bytes(b"\x7fELF")
        os.chmod(bin_path, 0o755)
        os.utime(bin_path, (1_000_000, 1_000_000))
        monkeypatch.setattr(
            "fuzzer_tool.adapters.forkserver._FUZZ_LOADER_BIN",
            str(bin_path),
        )
        assert _ensure_compiled() == str(bin_path)
        assert bin_path.read_bytes() == b"\x7fELF"


class TestForkserverRunner:
    def test_initial_state(self):
        r = ForkserverRunner("/fake/target")
        assert r._ready is False
        assert r._proc is None
        assert r._last_stderr == ""

    def test_run_one_not_ready(self):
        r = ForkserverRunner("/fake/target")
        rc, stderr = r.run_one(b"test")
        assert rc == -2
        assert stderr == ""

    def test_stop_without_start(self):
        r = ForkserverRunner("/fake/target")
        r.stop()  # should not raise
        assert r._ready is False
        assert r._proc is None

    def test_start_with_missing_binary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "fuzzer_tool.adapters.forkserver._FUZZ_LOADER_BIN",
            str(tmp_path / "nonexistent"),
        )
        r = ForkserverRunner("/fake/target")
        assert r.start() is False

    def test_input_file_cleanup_on_stop(self, tmp_path):
        r = ForkserverRunner("/fake/target")
        fd, cur = tempfile.mkstemp(suffix=".cur")
        os.close(fd)
        r._input_file = cur
        r._proc = None  # never started
        r.stop()
        # stop() only cleans up _input_file when _proc was set (process started)
        # Since _proc is None, the input file is not cleaned — this is expected
        # Test that stop() at least doesn't crash
        assert r._ready is False

    def test_input_file_cleanup_with_proc(self, tmp_path):
        r = ForkserverRunner("/fake/target")
        fd, cur = tempfile.mkstemp(suffix=".cur")
        os.close(fd)
        r._input_file = cur

        class FakeStdin:
            def write(self, data):
                pass

            def flush(self):
                pass

        class FakeProc:
            def poll(self):
                return 0

            stdin = FakeStdin()

            def wait(self, timeout=0):
                pass

        r._proc = FakeProc()
        r.stop()
        assert r._input_file is None
        assert not os.path.exists(cur)

    def test_restart_flag_prevents_recursive_restart(self):
        r = ForkserverRunner("/fake/target")
        r._restarting = True
        r._ready = False
        r._proc = None
        rc, stderr = r.run_one(b"test")
        assert rc == -2
        assert stderr == ""
