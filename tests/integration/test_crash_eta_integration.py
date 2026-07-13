"""Integration test for crash ETA estimation."""

import subprocess
import sys


def test_estimate_end_to_end():
    """Run estimate against the test target and verify output format."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fuzzer_tool",
            "estimate",
            "./targets/test_target",
            "--corpus",
            "./corpus",
            "--calibrate",
            "50",
        ],
        capture_output=True,
        text=True,
        cwd="/home/dclavijo/my_code/fuzzer",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Risky density" in result.stdout, result.stdout
    assert "Point estimate" in result.stdout, result.stdout
    assert "Range" in result.stdout, result.stdout
