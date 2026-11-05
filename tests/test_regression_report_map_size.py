"""Report printed the SHM map size (entries) with a "bytes" label: 8x off."""

from unittest.mock import MagicMock

from fuzzer_tool.adapters.shm import SIZEOF_ENTRY, ShmCoverage
from fuzzer_tool.services.report import _coverage_analysis

N = 1024


def _line(text: str) -> str:
    return next(ln for ln in text.splitlines() if "SHM map size" in ln)


def test_regression_report_map_size_entries():
    shm = ShmCoverage(size=N)
    try:
        f = MagicMock()
        f.shm_cov = shm
        f.corpus_dir = "/nonexistent-corpus-for-report"
        line = _line(_coverage_analysis(f))

        assert f"{N:,} entries" in line
        assert f"{N * SIZEOF_ENTRY:,} bytes" in line
        assert f"{N:,} bytes" not in line
    finally:
        shm.cleanup()
