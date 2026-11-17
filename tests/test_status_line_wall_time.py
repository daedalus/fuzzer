"""Status line shows wall time and wall EPS; decisions keep the virtual clock.

Virtual time advances VIRTUAL_EXEC_S per exec, so under the default
``--clock virtual`` the line used to print ``eps: 1000`` and a time equal to
``execs / 1000`` regardless of reality.
"""

import re
import time
from array import array
from unittest.mock import patch

from fuzzer_tool.core.clock import VIRTUAL_EPOCH, VIRTUAL_EXEC_S, Clock, ClockMode
from fuzzer_tool.services.stats import StatsReporter
from tests.test_stats_reporter import _mock_fuzzer

EXECS = 1000
WALL_S = 50.0


def _virtual_fuzzer(wall_start: float):
    """Mock fuzzer on a virtual clock ticked EXECS times, wall-started at *wall_start*."""
    clock = Clock(ClockMode.VIRTUAL)
    for _ in range(EXECS):
        clock.tick()
    return _mock_fuzzer(
        _clock=clock,
        start_time=VIRTUAL_EPOCH,
        _wall_start=wall_start,
        exec_count=EXECS,
        _resume_baseline_exec=0,
        _eps_history=array("d"),
        _eps_history_max=10,
    )


def _status_line(f) -> str:
    with patch("builtins.print") as mock_print:
        StatsReporter(f).print_stats()
    return mock_print.call_args_list[0][0][0]


def _field(line: str, name: str) -> float:
    return float(re.search(rf"\| {name}: ([0-9.]+)", line).group(1))


def test_shows_wall_time_and_eps():
    """Time and EPS on the line come from the host clock."""
    f = _virtual_fuzzer(time.time() - WALL_S)
    line = _status_line(f)

    assert abs(_field(line, "time") - WALL_S) <= 1
    assert abs(_field(line, "eps") - EXECS / WALL_S) <= 1


def test_falsify_decisions_stay_virtual():
    """The decision EPS (``f._eps``) still reads the virtual clock."""
    f = _virtual_fuzzer(time.time() - WALL_S)
    _status_line(f)

    assert f._eps == EXECS / (EXECS * VIRTUAL_EXEC_S)


def test_adversarial_zero_wall_elapsed():
    """A wall start at or after now prints eps 0, not a ZeroDivisionError."""
    f = _virtual_fuzzer(time.time() + 3600)
    line = _status_line(f)

    assert _field(line, "eps") == 0
