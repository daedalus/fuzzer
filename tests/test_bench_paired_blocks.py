"""Bench metric ``blocks``: guard sites hit, from the stats line.

Edge ids move under IJON_STATE (state ^ edge) and synthetic channels; the
block bitmap counts guard sites, so it compares two builds of one target.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import _PLAIN_METRICS, _parse  # noqa: E402
from eval_set import TARGET_SETS  # noqa: E402

_LOG = (
    "[*] execs: 1000 | corpus: 3 | shm: 8 max: 9 sat: 86% blocks: 6/15 | ph: 0x1\n"
    "[*] execs: 2000 | corpus: 7 | shm: 13 max: 21 sat: 61% blocks: 8/15 | ph: 0x2\n"
)


def test_blocks_is_last_stats_line():
    """Cumulative bitmap: the last line is the campaign's figure."""
    assert _parse(_LOG)["blocks"] == 8


def test_adversarial_no_blocks_field_leaves_key_absent():
    """Absent, never 0: analyse skips the cell rather than scoring it as no coverage."""
    assert "blocks" not in _parse("[*] execs: 10 | corpus: 1\n")


def test_blocks_is_a_metric():
    assert "blocks" in _PLAIN_METRICS


def test_ffmpeg_ijon_set_targets_ijon_variant():
    targets = [t for t, _ in TARGET_SETS["ffmpeg_ijon"]]
    assert targets and all(t.endswith("_ijon.so") for t in targets)
