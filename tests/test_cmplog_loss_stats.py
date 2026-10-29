"""Stats line: shim CMPLOG_DROPPED / CMPLOG_WRITE_FAILED beside the c/a totals.

Dropped records explain a frozen pair pool (e.g. ffmpeg's 4999 evictions
that never grow), so the live line has to show them.
"""

from types import SimpleNamespace

from fuzzer_tool.core.shim_health import ShimField, cmplog_loss
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.stats import StatsReporter, _format_count


def _counters(dropped: int, failed: int) -> list[int]:
    counters = [0] * len(ShimField)
    counters[ShimField.ATTACHED] = 1
    counters[ShimField.CMPLOG_DROPPED] = dropped
    counters[ShimField.CMPLOG_WRITE_FAILED] = failed
    return counters


def _fuzzer(shim_counters) -> SimpleNamespace:
    cmplog = SimpleNamespace(
        tokens=[b"a"],
        pairs=[],
        get_eviction_stats=lambda: (0, 0),
        total_comparisons=lambda: (2_223_000_000_000, 555_800_000_000),
    )
    return SimpleNamespace(_cmplog=cmplog, _shim_counters=shim_counters)


def _line(f) -> str:
    return StatsReporter(f)._print_stats_cmplog_str(f)


def test_cmplog_loss_reads_both_fields():
    assert cmplog_loss(_counters(1_200_345, 77)) == (1_200_345, 77)


def test_stats_line_shows_loss_after_comparisons():
    dropped, failed = 1_200_345, 77
    line = _line(_fuzzer(_counters(dropped, failed)))

    cmp_part = f"{_format_count(2_223_000_000_000)}c/{_format_count(555_800_000_000)}a"
    loss_part = f"drop:{_format_count(dropped)} wfail:{_format_count(failed)}"
    assert line.endswith(f"{cmp_part} {loss_part}")


def test_stats_line_shows_zero_loss():
    """Zero is evidence too: it rules the shim's write path out."""
    assert _line(_fuzzer(_counters(0, 0))).endswith("drop:0 wfail:0")


def test_falsify_no_snapshot_no_fragment():
    """Subprocess modes never read the counters: print nothing, not zeros."""
    line = _line(_fuzzer(None))
    assert "drop:" not in line
    assert "wfail:" not in line


def test_adversarial_old_shim_short_snapshot():
    """A shim predating the fields exports a shorter tuple: zeros, no IndexError."""
    short = (1, 8192, 0)
    assert cmplog_loss(short) == (0, 0)
    assert _line(_fuzzer(short)).endswith("drop:0 wfail:0")


def test_read_shim_counters_keeps_snapshot():
    """The 1000-exec health read is the only source; the stats line reuses it."""
    snap = tuple(_counters(5, 2))
    f = Fuzzer.__new__(Fuzzer)
    f._init_shim_watch()
    assert f._shim_counters is None

    f._inprocess_runner = SimpleNamespace(shim_health=lambda: snap)
    f.use_coverage = False
    f._read_shim_counters()
    assert f._shim_counters == snap
