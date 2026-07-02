"""Tests for services/tmin.py — crash minimization (error paths)."""


from fuzzer_tool.services.tmin import tmin


class TestTmin:
    def test_nonexistent_crash_file(self, tmp_path):
        result = tmin("/fake/target", str(tmp_path / "nonexistent.bin"))
        assert result is None

    def test_empty_crash_file(self, tmp_path):
        crash = tmp_path / "empty.bin"
        crash.write_bytes(b"")
        result = tmin("/fake/target", str(crash))
        assert result is None


class TestMain:
    def test_main_imports(self):
        """Verify main function exists and is callable."""
        from fuzzer_tool.services.tmin import main
        assert callable(main)
