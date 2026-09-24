"""--no-calibration: skip the verbatim seed pass before the fuzz loop."""

import ast
import inspect

from fuzzer_tool.cli import commands
from fuzzer_tool.services.fuzzer import Fuzzer
from tests.test_regression_calibration_target_edges import _fuzzer
from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

_EDGES = set(range(1_000_000, 1_000_100))


def test_disabled_runs_nothing():
    """Falsification: no execution, no recorded edges, no report."""
    f = _fuzzer([b"seed"], _EDGES)
    f._seed_calibration = False
    ran = []
    f._run_target = lambda data: ran.append(data) or (0, "")

    f._calibrate_seed_baselines()

    assert ran == []
    assert f._edge_tracker.seed_edges == {}


def test_enabled_still_calibrates():
    """Adversarial: the gate must not swallow the default path."""
    f = _fuzzer([b"seed"], _EDGES)
    f._seed_calibration = True

    f._calibrate_seed_baselines()

    assert f._edge_tracker.seed_edges


def test_default_is_enabled():
    assert inspect.signature(Fuzzer.__init__).parameters["seed_calibration"].default is True


def test_cli_flag_wired():
    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    assert calls
    assert all("seed_calibration" in {k.arg for k in c.keywords} for c in calls)
    assert "no_calibration" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
