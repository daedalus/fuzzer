"""Heap trim on every status line (``adapters/libc_mem.py``)."""

import ctypes
import importlib
from unittest.mock import patch

import pytest

from fuzzer_tool.adapters import libc_mem
from fuzzer_tool.services.stats import StatsReporter
from tests.test_stats_reporter import _mock_fuzzer

_STATUS_TICKS = 3
# Above pymalloc's 512 B cutoff, below glibc's 128 KiB mmap threshold: these
# land on the malloc heap, and freeing them leaves pages trim can release.
_BLOCK = 4096
_BLOCKS = 16_384
_KB = 1 << 10
_MB = 1 << 20


def test_trim_heap_releases_on_glibc():
    """Falsification: freed heap blocks come back as a positive byte count."""
    blocks = [bytes(_BLOCK) for _ in range(_BLOCKS)]
    del blocks

    assert libc_mem.trim_heap() > 0


@pytest.mark.parametrize(
    ("freed", "shown"),
    [(3 * _MB, "trim: 3MB"), (512 * _KB, "trim: 512KB"), (0, "trim: 0KB")],
)
def test_trim_per_status_line(freed, shown):
    """Falsification: one trim per print_stats(), its result on the line."""
    reporter = StatsReporter(_mock_fuzzer())

    with (
        patch("builtins.print") as mock_print,
        patch.object(libc_mem, "trim_heap", return_value=freed) as mock_trim,
    ):
        for _ in range(_STATUS_TICKS):
            reporter.print_stats()

    assert mock_trim.call_count == _STATUS_TICKS
    assert f"| {shown} |" in mock_print.call_args_list[0][0][0]


def test_trim_heap_rss_growth_clamped():
    """Adversarial: RSS rising across the trim (another thread) is not a
    negative release."""
    with patch.object(libc_mem, "_rss_bytes", side_effect=[_MB, 2 * _MB]):
        assert libc_mem.trim_heap() == 0


class _NoTrimLibc:
    """A libc without ``malloc_trim`` (musl, macOS)."""


def test_trim_heap_without_symbol():
    """Adversarial: a libc lacking the symbol imports and trims as a no-op."""
    try:
        with patch.object(ctypes, "CDLL", return_value=_NoTrimLibc()):
            importlib.reload(libc_mem)
        assert libc_mem._malloc_trim is None
        assert libc_mem.trim_heap() == 0
    finally:
        importlib.reload(libc_mem)
