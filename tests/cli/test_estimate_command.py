"""Tests for estimate CLI command."""

import subprocess
import sys


def test_estimate_help():
    result = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "estimate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "estimate" in result.stdout.lower()
