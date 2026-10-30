"""AdaptiveBatch: hill-climb the write batch size on observed seeds/s.

rate up or equal -> keep direction    rate down -> reverse
size *= FACTOR (up) or //= FACTOR (down), clamped to [lo, hi]
"""

import pytest

from fuzzer_tool.core.adaptive_batch import FACTOR, AdaptiveBatch


class _Clock:
    """Scripted monotonic clock: each call returns the next time."""

    def __init__(self, times):
        self._times = iter(times)

    def __call__(self) -> float:
        return next(self._times)


def _drive(batch: AdaptiveBatch, rates: list[float]) -> list[int]:
    """Feed one window per rate (seeds = current size); return sizes after each."""
    sizes = []
    for _ in rates:
        batch.observe(batch.size)
        sizes.append(batch.size)
    return sizes


def _clock_for(start: int, rates: list[float], lo: int, hi: int) -> _Clock:
    """Times that make window i run at rates[i] for the sizes the climb will pick.

    Sizes are re-derived here with the documented rule (not the code under test).
    """
    t, size, direction, prev = 0.0, start, 1, None
    times = [t]
    for r in rates:
        t += size / r
        times.append(t)
        if prev is not None and r < prev:
            direction = -direction
        prev = r
        size = min(hi, size * FACTOR) if direction > 0 else max(lo, size // FACTOR)
    return _Clock(times)


def test_grows_while_rate_improves():
    rates = [100.0, 200.0, 300.0, 400.0]
    b = AdaptiveBatch(start=8, lo=2, hi=1024, clock=_clock_for(8, rates, 2, 1024))
    assert _drive(b, rates) == [8 * FACTOR**k for k in range(1, 5)]


def test_reverses_when_rate_drops():
    """Falsification: a slower window must turn the climb around."""
    rates = [100.0, 200.0, 150.0, 120.0]
    b = AdaptiveBatch(start=8, lo=2, hi=1024, clock=_clock_for(8, rates, 2, 1024))
    s0 = 8 * FACTOR
    s1 = s0 * FACTOR
    # 100 -> up; 200 >= 100 keep up; 150 < 200 flip down; 120 < 150 flip up.
    assert _drive(b, rates) == [s0, s1, s1 // FACTOR, s1]


def test_clamped_to_bounds():
    """Adversarial: monotone improvement never exceeds hi; collapse never below lo."""
    up = [float(r) for r in range(1, 20)]
    b = AdaptiveBatch(start=4, lo=2, hi=16, clock=_clock_for(4, up, 2, 16))
    assert max(_drive(b, up)) == 16

    # One drop turns the climb down; rising rates then keep it going down.
    down = [10.0, 5.0] + [float(r) for r in range(6, 20)]
    b = AdaptiveBatch(start=4, lo=2, hi=16, clock=_clock_for(4, down, 2, 16))
    assert min(_drive(b, down)) == 2


def test_zero_elapsed_or_empty_window_ignored():
    """Adversarial: dt = 0 or no seeds gives no rate; size and state unchanged."""
    b = AdaptiveBatch(start=8, lo=2, hi=64, clock=_Clock([0.0, 0.0, 1.0]))
    b.observe(8)
    assert b.size == 8
    b.observe(0)
    assert b.size == 8


@pytest.mark.parametrize("lo,hi,start", [(0, 8, 4), (8, 4, 4), (2, 8, 16), (2, 8, 1)])
def test_bad_bounds_rejected(lo, hi, start):
    with pytest.raises(ValueError):
        AdaptiveBatch(start=start, lo=lo, hi=hi)


def test_pinned_when_bounds_equal():
    """lo == hi == start: the size never moves (how tests pin batches)."""
    rates = [100.0, 50.0, 200.0]
    b = AdaptiveBatch(start=4, lo=4, hi=4, clock=_clock_for(4, rates, 4, 4))
    assert _drive(b, rates) == [4, 4, 4]


def test_tolerance_ignores_small_drops():
    """A drop within tol keeps climbing; a drop past it reverses (falsification)."""
    small = [100.0, 95.0, 96.0]  # -5%: inside tol 0.1
    clock = _Clock([0.0, 8 / 100.0, 8 / 100.0 + 16 / 95.0, 8 / 100.0 + 16 / 95.0 + 32 / 96.0])
    b = AdaptiveBatch(start=8, lo=2, hi=1024, tol=0.1, clock=clock)
    assert _drive(b, small) == [16, 32, 64]

    big = [100.0, 80.0]  # -20%: past tol 0.1
    b = AdaptiveBatch(start=8, lo=2, hi=1024, tol=0.1, clock=_Clock([0.0, 0.08, 0.08 + 16 / 80.0]))
    assert _drive(b, big) == [16, 8]
