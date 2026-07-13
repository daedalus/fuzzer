"""Tests for calibration pass."""

import subprocess
import sys


def test_calibration_flag_in_help():
    """Verify --calibrate flag exists in fuzz subcommand."""
    result = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--calibrate" in result.stdout
