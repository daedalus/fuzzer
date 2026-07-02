"""Tests for __main__.py — CLI entry point."""

from unittest.mock import patch

from fuzzer_tool.__main__ import entry


class TestMain:
    def test_entry_calls_main(self):
        with patch("fuzzer_tool.__main__.main", return_value=0):
            result = entry()
            assert result == 0
