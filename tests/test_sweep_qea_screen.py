"""``sweep_qea_grover_angle --screen`` wiring to tools/lib/factorial_design."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "tools" / "sweep_qea_grover_angle.py"
_spec = importlib.util.spec_from_file_location("sweep_qea", _PATH)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)


def _names(report: str) -> list[str]:
    return [ln.split()[0] for ln in report.splitlines()[1:]]


def test_ranks_every_factor_once():
    out = sweep.screen_report("c", trials=2, cap=200)
    assert sorted(_names(out)) == sorted(sweep._SCREEN_FACTORS)


def test_control_same_run_identical():
    a = sweep.screen_report("c", trials=2, cap=200)
    assert a == sweep.screen_report("c", trials=2, cap=200)


def test_constant_response_gives_zero_effects():
    # Adversarial: every run hits the cap -> no factor can rank above zero.
    out = sweep.screen_report("c", trials=1, cap=1)
    assert all(float(ln.split()[1]) == 0.0 for ln in out.splitlines()[1:])
