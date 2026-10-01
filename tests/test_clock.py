"""core/clock.py: wall vs virtual time for reproducible seeded runs."""

import time

import pytest

from fuzzer_tool.core.clock import (
    VIRTUAL_EPOCH,
    VIRTUAL_EXEC_S,
    WALL_CLOCK,
    Clock,
    ClockMode,
)

_TICKS = 7


def test_virtual_advances_only_on_tick():
    clock = Clock(ClockMode.VIRTUAL)

    assert clock.monotonic() == 0.0
    assert clock.time() == VIRTUAL_EPOCH

    for _ in range(_TICKS):
        clock.tick()

    assert clock.monotonic() == pytest.approx(_TICKS * VIRTUAL_EXEC_S)
    assert clock.time() == pytest.approx(VIRTUAL_EPOCH + _TICKS * VIRTUAL_EXEC_S)


def test_virtual_ignores_real_time(monkeypatch):
    """Adversarial: the host clock jumping must not move virtual time."""
    clock = Clock(ClockMode.VIRTUAL)
    before = (clock.monotonic(), clock.time())

    monkeypatch.setattr(time, "monotonic", lambda: 1e9)
    monkeypatch.setattr(time, "time", lambda: 1e9)

    assert (clock.monotonic(), clock.time()) == before


def test_wall_tracks_host_clock():
    """Falsification: wall mode is the host clock, tick or not."""
    clock = Clock(ClockMode.WALL)

    lo = time.monotonic()
    clock.tick()
    mid = clock.monotonic()
    hi = time.monotonic()

    assert lo <= mid <= hi
    assert abs(clock.time() - time.time()) < 1.0


def test_two_virtual_clocks_agree():
    """Control: same tick sequence, same readings."""
    a, b = Clock(ClockMode.VIRTUAL), Clock(ClockMode.VIRTUAL)
    for _ in range(_TICKS):
        a.tick()
        b.tick()
    assert (a.monotonic(), a.time()) == (b.monotonic(), b.time())


def test_default_is_wall():
    assert WALL_CLOCK.mode is ClockMode.WALL
    assert Clock().mode is ClockMode.WALL


def test_rejects_stray_state():
    with pytest.raises(AttributeError):
        Clock().undeclared = 1
