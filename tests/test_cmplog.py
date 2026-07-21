"""Tests for core/cmplog.py — comparison logging collector."""

from unittest.mock import patch

from fuzzer_tool.core.cmplog import CmplogCollector


class TestCmplogCollector:
    def test_init(self):
        c = CmplogCollector()
        assert c.log_path is None
        assert c.tokens == []
        assert c._shim_path is None

    def test_collect_tokens_no_log(self):
        c = CmplogCollector()
        assert c.collect_tokens() == []

    def test_collect_tokens_with_file(self, tmp_path):
        c = CmplogCollector()
        log_file = tmp_path / "test.cmplog"
        log_file.write_text("CMP 48656c6c6f 576f726c64\nCMP 4142 4344\nOTHER line\n\n")
        c.log_path = str(log_file)
        tokens = c.collect_tokens()
        assert len(tokens) == 4  # Hello, World, AB, CD
        assert b"Hello" in tokens
        assert b"World" in tokens

    def test_collect_tokens_dedup(self, tmp_path):
        c = CmplogCollector()
        log_file = tmp_path / "test.cmplog"
        log_file.write_text("CMP 4142 4344\n")
        c.log_path = str(log_file)
        tokens1 = c.collect_tokens()
        assert len(tokens1) == 2  # AB, CD

        log_file.write_text("CMP 4142 4546\n")
        c.log_path = str(log_file)
        tokens2 = c.collect_tokens()
        assert len(tokens2) == 1  # only EF is new

    def test_collect_tokens_clears_log(self, tmp_path):
        c = CmplogCollector()
        log_file = tmp_path / "test.cmplog"
        log_file.write_text("CMP 4142 4344\n")
        c.log_path = str(log_file)
        c.collect_tokens()
        assert log_file.exists()  # file kept (truncated, not deleted)
        assert log_file.read_text() == ""  # content cleared

    def test_collect_tokens_corrupt_hex(self, tmp_path):
        c = CmplogCollector()
        log_file = tmp_path / "test.cmplog"
        log_file.write_text("CMP ZZZZ 4344\nCMP 4142 4344\n")
        c.log_path = str(log_file)
        tokens = c.collect_tokens()
        assert len(tokens) == 2  # CD, AB from second line; ZZZZ line skipped

    def test_collect_tokens_short_line(self, tmp_path):
        c = CmplogCollector()
        log_file = tmp_path / "test.cmplog"
        log_file.write_text("CMP only_one\n")
        c.log_path = str(log_file)
        tokens = c.collect_tokens()
        assert tokens == []

    def test_collect_tokens_os_error(self, tmp_path):
        c = CmplogCollector()
        c.log_path = "/nonexistent/path.cmplog"
        tokens = c.collect_tokens()
        assert tokens == []

    def test_get_tokens(self):
        c = CmplogCollector()
        c.tokens = [b"AB", b"CD"]
        assert c.get_tokens() == [b"AB", b"CD"]

    def test_stop_removes_log(self, tmp_path):
        c = CmplogCollector()
        log_file = tmp_path / "test.cmplog"
        log_file.write_text("data")
        c.log_path = str(log_file)
        c.stop()
        assert not log_file.exists()
        assert c.log_path is None

    def test_stop_no_log(self):
        c = CmplogCollector()
        c.stop()  # no-op

    def test_setup_env_no_shim(self):
        c = CmplogCollector()
        env = {"PATH": "/usr/bin"}
        result = c.setup_env(env)
        assert result == env  # unmodified

    def test_setup_env_with_shim(self, tmp_path):
        c = CmplogCollector()
        c._shim_path = "/tmp/fake_shim.so"
        env = {"PATH": "/usr/bin"}
        result = c.setup_env(env)
        assert "_CMPLOG_OUT" in result
        assert "LD_PRELOAD" in result
        assert "fake_shim" in result["LD_PRELOAD"]
        assert c.log_path is not None

    def test_setup_env_prepend_ld_preload(self, tmp_path):
        c = CmplogCollector()
        c._shim_path = "/tmp/fake_shim.so"
        env = {"LD_PRELOAD": "/existing/lib.so"}
        result = c.setup_env(env)
        assert result["LD_PRELOAD"] == "/tmp/fake_shim.so:/existing/lib.so"

    def test_start_no_shim_source(self):
        c = CmplogCollector()
        with patch(
            "fuzzer_tool.adapters.shim_factory._find_compiler", side_effect=Exception("no compiler")
        ):
            result = c.start()
        # If shim already cached, this returns True regardless
        if c._shim_path:
            assert result is True
        else:
            assert result is False
