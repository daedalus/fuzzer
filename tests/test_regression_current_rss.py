"""Regression: the memory pruner read peak RSS as if it were current RSS.

``services/fuzzer.py:_check_memory_and_prune`` read
``getrusage(RUSAGE_SELF).ru_maxrss`` -- the monotonic high-water mark -- into a
variable its own docstring called current RSS. A threshold check against a peak
can only ever latch on: one transient spike armed the corpus pruner
permanently, and the warning printed the stale peak as current usage.
"""

import os
import tempfile
from pathlib import Path

import pytest

from fuzzer_tool.services.fuzzer import _current_rss_kb


@pytest.fixture
def tmp_root():
    with tempfile.TemporaryDirectory(prefix="fuzz_rss_") as d:
        yield Path(d)


# ── 2. current RSS ────────────────────────────────────────────────────


class TestCurrentRss:
    def test_returns_plausible_value(self):
        rss = _current_rss_kb()
        assert rss is not None
        # A live CPython with numpy loaded is comfortably in this band.
        assert 1_000 < rss < 100_000_000

    def test_tracks_allocation(self):
        before = _current_rss_kb()
        ballast = bytearray(80 * 1024 * 1024)  # 80 MiB, touched below
        for i in range(0, len(ballast), 4096):
            ballast[i] = 1
        during = _current_rss_kb()
        assert during - before > 40_000, "current RSS did not follow a real allocation"
        del ballast


class TestCurrentRssFalsification:
    def test_is_not_the_getrusage_peak(self):
        # Falsification of the exact defect. Allocate and free a large block:
        # ru_maxrss keeps the peak forever, current RSS falls back. If
        # _current_rss_kb were still reading the peak, the two would agree
        # after the free.
        import resource

        ballast = bytearray(200 * 1024 * 1024)
        for i in range(0, len(ballast), 4096):
            ballast[i] = 1
        del ballast

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        current = _current_rss_kb()
        assert current is not None
        assert current < peak, (
            f"current RSS ({current} KiB) is not below the peak ({peak} KiB) -- "
            "still reading ru_maxrss"
        )

    def test_is_not_monotonic(self):
        # A peak reading can never decrease. A current reading must be able to.
        import resource

        readings = []
        for _ in range(3):
            block = bytearray(120 * 1024 * 1024)
            for i in range(0, len(block), 4096):
                block[i] = 1
            readings.append(_current_rss_kb())
            del block
            readings.append(_current_rss_kb())
        assert any(b < a for a, b in zip(readings, readings[1:], strict=True)), (
            f"RSS readings never decreased: {readings}"
        )
        # Sanity: the peak genuinely is monotonic, so the contrast is real.
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >= max(readings)


class TestCurrentRssAdversarial:
    def test_returns_none_when_proc_unreadable(self, monkeypatch):
        def _boom(*_a, **_k):
            raise OSError("no /proc")

        monkeypatch.setattr("builtins.open", _boom)
        assert _current_rss_kb() is None

    def test_returns_none_on_garbage_statm(self, monkeypatch, tmp_root):
        garbage = tmp_root / "statm"
        garbage.write_text("not numbers here\n")
        real_open = open

        def _fake_open(path, *a, **k):
            if str(path) == "/proc/self/statm":
                return real_open(garbage, *a, **k)
            return real_open(path, *a, **k)

        monkeypatch.setattr("builtins.open", _fake_open)
        assert _current_rss_kb() is None

    def test_page_size_conversion(self):
        # Result is KiB, not pages: cross-check against the raw field.
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        expected = pages * (os.sysconf("SC_PAGE_SIZE") // 1024)
        got = _current_rss_kb()
        # Allow drift between the two reads.
        assert abs(got - expected) < 5_000
