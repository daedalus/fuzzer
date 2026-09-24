"""Regression: ``ExecTimeCalibrator`` kept every exec time and re-sorted it.

``observe()`` appended to an unbounded list (32 B/exec) and ``threshold()``
ran ``statistics.median`` over all of it on every exec. Measured: 1.1 ms
per call at 10k execs, 17 ms at 100k, 235 ms at 1M -- above fuzzgoat's
0.67 ms raw exec time from 10k execs on.
"""

import statistics

import pytest

from fuzzer_tool.core.analyzers.analyzer_exec_time_anomaly import ExecTimeCalibrator
from fuzzer_tool.core.rand_pool import RandPool

_WINDOW = 64
_STREAM_LEN = _WINDOW * 5
# Few distinct values force ties, so eviction must remove one copy, not all.
_TIE_LEVELS = 7


def _stream(seed: int) -> list[float]:
    rng = RandPool(seed)
    return [rng.randint(1, _TIE_LEVELS) / 100 for _ in range(_STREAM_LEN)]


def test_regression_exec_time_anomaly_unbounded():
    """Falsification: retention is bounded by the window, not the exec count."""
    cal = ExecTimeCalibrator(min_samples=1, window=_WINDOW)
    for t in _stream(seed=1):
        cal.observe(t)

    assert len(cal._times) == _WINDOW
    assert len(cal._sorted) == _WINDOW
    assert cal.count == _STREAM_LEN


@pytest.mark.parametrize("window", [1, 2, _WINDOW - 1, _WINDOW])
def test_window_exact(window):
    """Adversarial: tied values, odd/even and size-1 windows -- the median
    must equal ``statistics.median`` of the last ``window`` samples at
    every step."""
    stream = _stream(seed=window)
    cal = ExecTimeCalibrator(min_samples=1, window=window)

    for i, t in enumerate(stream):
        cal.observe(t)
        expect = statistics.median(stream[max(0, i + 1 - window) : i + 1])
        assert cal.median == expect
        assert cal.threshold(mult=2.0) == 2.0 * expect


def test_window_forgets():
    """A regime change is fully absorbed once it fills the window."""
    cal = ExecTimeCalibrator(min_samples=1, window=_WINDOW)
    for _ in range(_WINDOW * 3):
        cal.observe(1.0)
    for _ in range(_WINDOW):
        cal.observe(0.05)

    assert cal.median == 0.05


def test_min_samples_total():
    """``min_samples`` above the window still arms on total observations."""
    cal = ExecTimeCalibrator(min_samples=_WINDOW * 2, window=_WINDOW)
    for _ in range(_WINDOW * 2 - 1):
        cal.observe(0.05)
    assert cal.threshold() is None

    cal.observe(0.05)
    assert cal.threshold() == pytest.approx(0.1)


def test_rejects_empty_window():
    with pytest.raises(ValueError):
        ExecTimeCalibrator(window=0)
