"""Regression: crash counting included sidecar files, inflating totals ~4x.

``services/report.py:_crash_analysis`` counted every file in ``crashes_dir``.
``save_crash`` writes one input (``<base>.bin``) plus up to three sidecars
(``.txt`` report, ``.sh`` repro script, ``.hex`` dump), so "Total crashes" read
about four times high.

The count was the visible symptom, not the whole defect: the same list feeds
the size histogram and the sample list, so sidecar TEXT was being reported as
crash input sizes. A campaign's headline number and its size distribution were
both wrong.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.services.report import CRASH_INPUT_SUFFIX, _crash_analysis


@pytest.fixture
def tmp_root():
    with tempfile.TemporaryDirectory(prefix="fuzz_crashcount_") as d:
        yield Path(d)


def _write_crash(d: Path, name: str, payload: bytes):
    """Mirror save_crash's on-disk shape: one input plus three sidecars."""
    (d / f"{name}.bin").write_bytes(payload)
    (d / f"{name}.txt").write_text("=" * 400 + "\nASAN report text\n")
    (d / f"{name}.sh").write_text("#!/bin/sh\n" + "# repro\n" * 40)
    (d / f"{name}.hex").write_text("00000000  41 41 41 41\n" * 40)


# ── 1. crash counting ─────────────────────────────────────────────────


class TestCrashCountExcludesSidecars:
    def test_counts_inputs_not_files(self, tmp_root):
        crashes = tmp_root / "crashes"
        crashes.mkdir()
        for i in range(3):
            _write_crash(crashes, f"crash_{i}", b"A" * (10 + i))

        # 3 crashes, 12 files on disk.
        assert len(list(crashes.iterdir())) == 12

        f = SimpleNamespace(crash_count=3)
        out = _crash_analysis(f, crashes)
        assert "Total crashes:   3" in out
        assert "Total crashes:   12" not in out

    def test_size_histogram_uses_input_sizes(self, tmp_root):
        # The sidecars are far larger than the inputs, so if they leaked into
        # the histogram the reported sizes would be wrong, not merely too many.
        crashes = tmp_root / "crashes"
        crashes.mkdir()
        _write_crash(crashes, "only", b"B" * 7)

        f = SimpleNamespace(crash_count=1)
        out = _crash_analysis(f, crashes)
        assert "x 1" in out
        # 7 B input; every sidecar is >100 B.
        assert "7 B" in out or "7B" in out.replace(" ", "")


class TestCrashCountFalsification:
    def test_zero_crash_count_short_circuits(self, tmp_root):
        crashes = tmp_root / "crashes"
        crashes.mkdir()
        _write_crash(crashes, "c", b"x")
        assert _crash_analysis(SimpleNamespace(crash_count=0), crashes) == ""

    def test_directory_of_only_sidecars_reports_nothing(self, tmp_root):
        # Falsification: if the filter were absent or wrong, a directory with
        # no .bin at all would still report crashes.
        crashes = tmp_root / "crashes"
        crashes.mkdir()
        (crashes / "orphan.txt").write_text("stale report")
        (crashes / "orphan.sh").write_text("#!/bin/sh")
        assert _crash_analysis(SimpleNamespace(crash_count=2), crashes) == ""

    def test_suffix_constant_matches_writer(self):
        # The constant must stay in sync with adapters/filesystem.py:save_crash.
        assert CRASH_INPUT_SUFFIX == ".bin"


class TestCrashCountAdversarial:
    def test_unknown_extensions_are_not_counted(self, tmp_root):
        crashes = tmp_root / "crashes"
        crashes.mkdir()
        _write_crash(crashes, "real", b"z")
        (crashes / "notes.md").write_text("hand-written")
        (crashes / "core.12345").write_bytes(b"\x7fELF")
        out = _crash_analysis(SimpleNamespace(crash_count=1), crashes)
        assert "Total crashes:   1" in out

    def test_subdirectory_is_ignored(self, tmp_root):
        crashes = tmp_root / "crashes"
        (crashes / "triaged").mkdir(parents=True)
        (crashes / "triaged" / "nested.bin").write_bytes(b"q")
        _write_crash(crashes, "top", b"w")
        out = _crash_analysis(SimpleNamespace(crash_count=1), crashes)
        assert "Total crashes:   1" in out

    def test_missing_directory_returns_empty(self, tmp_root):
        assert _crash_analysis(SimpleNamespace(crash_count=5), tmp_root / "nope") == ""
