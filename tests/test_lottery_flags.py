"""The lottery-port fuzz flags reach Fuzzer and its state, off by default."""

from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.services.fuzzer import Fuzzer

FLAGS = (
    ("--seed-stride-cost", "seed_stride_cost"),
    ("--inverse-lottery-select", "inverse_lottery_select"),
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
    kw = _fuzz_kwargs(monkeypatch, tmp_path, [flag for flag, _ in FLAGS])

    assert all(kw[name] is True for _, name in FLAGS)


def test_flags_default_off(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, [])

    assert all(kw[name] is False for _, name in FLAGS)


def test_not_in_hail_mary_until_measured(monkeypatch, tmp_path):
    """Unbenchmarked flags stay out of --hail-mary (docs/AB_PENDING.md)."""
    kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--hail-mary"])

    assert all(kw[name] is False for _, name in FLAGS)


@pytest.mark.parametrize(
    "needle",
    [
        "self._use_inverse_lottery_select = bool(inverse_lottery_select)",
        "cost_aware=bool(seed_stride_cost)",
    ],
)
def test_fuzzer_stores_consumer_state(needle):
    # Fuzzer() needs a live target; the assignment is checked in source.
    assert needle in inspect.getsource(Fuzzer.__init__)
