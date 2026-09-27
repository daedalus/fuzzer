"""Regression: the minimax Phase 3-5 fuzz flags reach Fuzzer and its state.

``--wall-order`` (Phase 3), ``--op-minimax`` (Phase 4) and
``--minimax-select`` (Phase 5) gate code that had no caller. Each must be
parsed, passed to ``Fuzzer`` and stored under the attribute its consumer
reads, and stay off by default.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.services.fuzzer import Fuzzer

# (CLI flag, Fuzzer kwarg, attribute the consumer reads)
FLAGS = (
    ("--wall-order", "wall_order", "_use_wall_order"),
    ("--op-minimax", "op_minimax", "_use_op_minimax"),
    ("--minimax-select", "minimax_select", "_use_minimax_select"),
)


def _fuzz_kwargs(monkeypatch, tmp_path, extra: list[str]) -> dict:
    target = tmp_path / "t"
    target.write_bytes(b"\x7fELF")
    target.chmod(0o755)
    captured: dict = {}
    monkeypatch.setattr(commands, "Fuzzer", lambda **kw: captured.update(kw) or MagicMock())
    argv = ["fuzzer-tool", "fuzz", str(target), "-d", str(tmp_path / "c"), *extra]
    monkeypatch.setattr(sys, "argv", argv)

    commands.main()
    return captured


def test_flags_reach_fuzzer(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, [flag for flag, _, _ in FLAGS])

    assert all(kw[name] is True for _, name, _ in FLAGS)


def test_flags_default_off(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, [])

    assert all(kw[name] is False for _, name, _ in FLAGS)


@pytest.mark.parametrize(("_flag", "name", "attr"), FLAGS)
def test_fuzzer_stores_consumer_attribute(_flag, name, attr):
    # Fuzzer() needs a live target; the assignment is checked in source.
    src = inspect.getsource(Fuzzer.__init__)

    assert f"self.{attr} = bool({name})" in src


# bench arm -> the dest it adds over its declared baseline
ARM_DEST = {
    "elo-op-minimax": "op_minimax",
    "wall-order": "wall_order",
    "minimax-select": "minimax_select",
}


@pytest.mark.parametrize(("arm", "dest"), sorted(ARM_DEST.items()))
def test_bench_arm_is_baseline_plus_flag(monkeypatch, arm, dest):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))
    from bench_paired import ARM_BASELINES, ARMS

    from tests.test_bench_paired_arms import _parse

    base = ARMS[ARM_BASELINES[arm]]
    added = [f for f in ARMS[arm] if f not in base]

    assert ARMS[arm][: len(base)] == base
    assert len(added) == 1
    assert getattr(_parse(monkeypatch, ARMS[arm]), dest) is True


def test_hail_mary_enables_minimax_flags(monkeypatch, tmp_path):
    # op_minimax is inert without mc_bandit; hail-mary must set both.
    kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--hail-mary"])

    assert all(kw[name] is True for _, name, _ in FLAGS)
    assert kw["mc_bandit"] is True
