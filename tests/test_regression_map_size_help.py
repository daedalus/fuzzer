"""Regression: `fuzz --help` called --map-size bytes.

The value is taken as entries (8 bytes each): `--map-size 1024` gave load
38% on fuzzgoat, which only fits 1,024 entries. The help also named branch
density as the auto source; sancov guard counts come first.
"""

from __future__ import annotations

import re
import subprocess
import sys


def _map_size_help() -> str:
    r = subprocess.run(
        [sys.executable, "-m", "fuzzer_tool", "fuzz", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr

    # The option's block runs until the next option line.
    m = re.search(r"\n  --map-size N(.*?)\n  -", r.stdout, re.S)
    assert m, "--map-size missing from fuzz --help"
    return " ".join(m.group(1).split())


def test_regression_map_size_help_says_entries():
    text = _map_size_help()

    assert "entries" in text
    assert "bytes" not in text.replace("8 bytes", "")
    assert "branch density" not in text
