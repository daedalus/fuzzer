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


class TestHashDetection:
    def test_hash_candidates_initially_empty(self):
        c = CmplogCollector()
        assert c.hash_candidates == set()

    def test_detect_short_pairs_not_flagged(self):
        c = CmplogCollector()
        n = c.detect_hash_candidates([(b"ab", b"cd")])
        assert n == 0  # too short (< _HASH_MIN_BYTES = 8)

    def test_detect_long_matching_pair_not_flagged(self):
        c = CmplogCollector()
        pair = (b"\x01\x02\x03\x04\x05\x06\x07\x08",
                b"\x01\x02\x03\x04\x05\x06\x07\x08")
        n = c.detect_hash_candidates([pair])
        assert n == 0  # exact match — not hash-like

    def test_detect_hash_like_pair(self):
        c = CmplogCollector()
        # 8 bytes, only 1 matching position — looks like a hash
        pair = (b"\x01\x02\x03\x04\x05\x06\x07\x08",
                b"\x01\xff\xfe\xfd\xfc\xfb\xfa\xf9")
        n = c.detect_hash_candidates([pair])
        assert n == 1
        assert pair in c.hash_candidates

    def test_is_hash_candidate(self):
        c = CmplogCollector()
        pair = (b"\x01\x02\x03\x04\x05\x06\x07\x08",
                b"\x01\xff\xfe\xfd\xfc\xfb\xfa\xf9")
        c.detect_hash_candidates([pair])
        assert c.is_hash_candidate(*pair)
        assert not c.is_hash_candidate(b"ab", b"cd")

    def test_hash_detection_integration(self, tmp_path):
        c = CmplogCollector()
        # Two 8-byte values that look hash-like (very few matching bytes)
        log_file = tmp_path / "hash.cmplog"
        hex_a = "0102030405060708"
        hex_b = "01fffefdfcfbfaf9"
        log_file.write_text(f"CMP {hex_a} {hex_b}\n")
        c.log_path = str(log_file)
        c.collect_tokens()
        assert len(c.hash_candidates) >= 1


class TestMultiRunCollection:
    def test_pair_occurrence_initially_empty(self):
        c = CmplogCollector()
        assert c._pair_occurrence == {}

    def test_collect_tokens_tracks_occurrence(self, tmp_path):
        c = CmplogCollector()
        log_file = tmp_path / "run.cmplog"
        log_file.write_text("CMP 4142 4344\n")
        c.log_path = str(log_file)
        c.collect_tokens()
        assert c._pair_occurrence.get((b"AB", b"CD")) == 1

    def test_high_confidence_pairs(self, tmp_path):
        c = CmplogCollector()
        # Simulate pair seen in 3 runs
        c._pair_occurrence[(b"AB", b"CD")] = 3
        c._pair_occurrence[(b"EF", b"GH")] = 1
        high = c.high_confidence_pairs(min_occurrences=2)
        assert (b"AB", b"CD") in high
        assert (b"EF", b"GH") not in high

    def test_pair_confidence(self):
        c = CmplogCollector()
        assert c.pair_confidence(b"AB", b"CD") == 0
        c._pair_occurrence[(b"AB", b"CD")] = 5
        assert c.pair_confidence(b"AB", b"CD") == 5
