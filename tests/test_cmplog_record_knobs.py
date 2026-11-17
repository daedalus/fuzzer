"""Cmplog record knobs: ``--cmplog-site-cap`` / ``--cmplog-dedup-scope`` wiring.

Shim behaviour: tests/test_cmplog_site_cap_scope.py. This pins CLI ->
Fuzzer -> CmplogCollector -> env, the I2S probe's rescope, and the bench
arms (same discipline as tests/test_cmplog_prune_env.py).
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.cmplog import CmplogCollector, CmplogRecords, DedupScope
from fuzzer_tool.services.fuzzer import Fuzzer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "lib"))

from bench_paired import ARM_BASELINES, ARMS  # noqa: E402

_CAP = "__AFL_CMPLOG_SITE_CAP"
_SCOPE = "__AFL_CMPLOG_DEDUP_SCOPE"


def _collector(tmp_path, **kw) -> CmplogCollector:
    c = CmplogCollector(workdir=str(tmp_path), **kw)
    c._shim_path = str(tmp_path / "fuzz_cmplog_shim.so")
    return c


# ── Collector -> env ────────────────────────────────────────────────────


class TestSetupEnv:
    def test_default_sets_neither(self, tmp_path):
        env = _collector(tmp_path).setup_env({})
        assert _CAP not in env and _SCOPE not in env

    def test_knobs_appear(self, tmp_path):
        env = _collector(tmp_path, site_cap=32, dedup_scope=DedupScope.CAMPAIGN).setup_env({})
        assert env[_CAP] == "32"
        assert env[_SCOPE] == "campaign"

    @pytest.mark.parametrize("cap", [0, -5])
    def test_adversarial_non_positive_cap_is_off(self, tmp_path, cap):
        assert _CAP not in _collector(tmp_path, site_cap=cap).setup_env({})


class TestEnvForRun:
    def test_set_then_restored_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv(_CAP, raising=False)
        monkeypatch.delenv(_SCOPE, raising=False)
        c = _collector(tmp_path, site_cap=8, dedup_scope=DedupScope.CAMPAIGN)
        c.setup_env_for_run()
        assert (os.environ[_CAP], os.environ[_SCOPE]) == ("8", "campaign")
        c.restore_env()
        assert _CAP not in os.environ and _SCOPE not in os.environ

    def test_preexisting_value_survives_restore(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_CAP, "3")
        c = _collector(tmp_path, site_cap=8)
        c.setup_env_for_run()
        c.restore_env()
        assert os.environ[_CAP] == "3"


# ── CLI -> Fuzzer -> collector ──────────────────────────────────────────


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
    flags = ["--cmplog-site-cap", "32", "--cmplog-dedup-scope", "campaign"]
    kw = _fuzz_kwargs(monkeypatch, tmp_path, flags)
    assert kw["cmplog_site_cap"] == 32
    assert kw["cmplog_dedup_scope"] is DedupScope.CAMPAIGN


def test_falsify_defaults_are_current_behaviour(monkeypatch, tmp_path):
    kw = _fuzz_kwargs(monkeypatch, tmp_path, [])
    assert kw["cmplog_site_cap"] == 0
    assert kw["cmplog_dedup_scope"] is DedupScope.DRAIN


def test_adversarial_unknown_scope_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _fuzz_kwargs(monkeypatch, tmp_path, ["--cmplog-dedup-scope", "forever"])


def test_fuzzer_hands_knobs_to_collector():
    # Fuzzer() needs a live target; the hand-off is checked in source.
    src = inspect.getsource(Fuzzer.__init__)
    assert "site_cap=cmplog_site_cap" in src
    assert "dedup_scope=cmplog_dedup_scope" in src


# ── I2S probe rescope ───────────────────────────────────────────────────


class _Lib:
    """ctypes handle stand-in logging shim calls in order."""

    def __init__(self, log: list, symbols=("__cmplog_pause", "__cmplog_rescope")):
        self._log = log
        self._symbols = symbols

    def __getattr__(self, name):
        if name not in self._symbols:
            raise AttributeError(name)
        return lambda *a: self._log.append(name)


def _probe(lib, log: list) -> None:
    f = Fuzzer.__new__(Fuzzer)
    f._inprocess_runner = SimpleNamespace(direct_lite=True, _lib=lib)
    f._cmplog_records_mode = CmplogRecords.ON
    f._cmplog = SimpleNamespace(collect_tokens=lambda: [], last_pairs=[])
    f._runner = SimpleNamespace(run_target=lambda data: log.append("run"))
    f.exec_count = 0
    f._add_dict_tokens = lambda tokens: None
    f._rewind_cmplog_shim = lambda: None
    f._reset_cmplog = lambda: None
    f._i2s_probe(b"x")


def test_i2s_probe_rescopes_before_run():
    """Campaign scope would hide repeats; the probe needs one exec's full set."""
    log: list = []
    _probe(_Lib(log), log)
    assert "__cmplog_rescope" in log
    assert log.index("__cmplog_rescope") < log.index("run")


def test_adversarial_old_build_without_rescope_still_probes():
    log: list = []
    _probe(_Lib(log, symbols=("__cmplog_pause",)), log)
    assert "run" in log


# ── Bench arms ──────────────────────────────────────────────────────────


def _parse(monkeypatch, flags: list[str]):
    captured: dict[str, object] = {}
    monkeypatch.setattr(commands, "cmd_fuzz", lambda args: captured.update(args=args) or 0)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *flags])
    assert commands.main() == 0
    return captured["args"]


def test_arm_sets_both_knobs(monkeypatch):
    assert ARM_BASELINES["cmp-cap-dedup"] == "baseline"
    args = _parse(monkeypatch, ARMS["cmp-cap-dedup"])
    assert args.cmplog_site_cap > 0
    assert args.cmplog_dedup_scope is DedupScope.CAMPAIGN
    base = _parse(monkeypatch, ARMS["baseline"])
    assert (base.cmplog_site_cap, base.cmplog_dedup_scope) == (0, DedupScope.DRAIN)


def test_control_is_baseline_exactly():
    """A/A control (Rule 46)."""
    assert ARMS["cmp-cap-dedup-ctl"] == ARMS["baseline"]
    assert ARM_BASELINES["cmp-cap-dedup-ctl"] == "baseline"
