"""CLI plumbing for the static-CFG flags: --cut-damage, --waypoints, --prove-dead."""

import sys
from unittest.mock import MagicMock

from fuzzer_tool.cli import commands

FLAGS = (
    ("--cut-damage", "cut_damage"),
    ("--waypoints", "waypoints"),
    ("--prove-dead", "prove_dead"),
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
    kw = _fuzz_kwargs(monkeypatch, tmp_path, ["--hail-mary"])

    assert all(kw[name] is False for _, name in FLAGS)
