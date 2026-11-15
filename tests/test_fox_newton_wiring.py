"""--fox-newton: registry gate, operator handler, Fuzzer, round feed, CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fuzzer_tool.core.cmplog import CountsDrain
from fuzzer_tool.core.fox_newton import SecantLog
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.operators import OperatorEngine
from tests import test_commands_extended
from tests.support.scripted_rng import ScriptedRng
from tests.test_fox_newton import CONST, _affine, _cond, _field, _input, _log_runs
from tests.test_regression_operator_registry import _MockFuzzer

OP = "fox_newton"


def _available(**attrs) -> set[str]:
    fuzzer = _MockFuzzer()
    for name, value in attrs.items():
        setattr(fuzzer, name, value)
    return set(REGISTRY.available(fuzzer, b"seed"))


# ── registry ─────────────────────────────────────────────────────────


def test_registered_in_adaptive_band():
    assert OP in REGISTRY.names()
    assert REGISTRY.category_of(OP) == "adaptive"


def test_gated_on_fittable_site():
    assert OP not in _available()
    assert OP not in _available(_secant_log=SecantLog())  # adversarial: empty log
    assert OP in _available(_secant_log=_log_runs([10, 20]))


# ── handler ──────────────────────────────────────────────────────────


def _engine(log, choice_idxs):
    ctx = SimpleNamespace(max_len=4096, _rng=ScriptedRng(choice_idxs=choice_idxs))
    eng = SimpleNamespace(ctx=ctx, f=SimpleNamespace(_secant_log=log))
    eng._op_declined = MagicMock(return_value=None)
    return eng


def test_handler_writes_secant_root():
    eng = _engine(_log_runs([10, 20]), [0, 0, 0])
    out = OperatorEngine._op_fox_newton(eng, bytearray(_input(10)), 0, None)
    assert abs(_affine(_field(bytes(out))) - CONST) <= 1


def test_handler_declines_without_fit():
    # Falsification: no log -> no change, and the decline is recorded.
    eng = _engine(None, [])
    buf = bytearray(_input(10))
    assert OperatorEngine._op_fox_newton(eng, buf, 0, None) is None
    eng._op_declined.assert_called_once_with(OP, buf)


def test_handler_clamps_to_max_len():
    eng = _engine(_log_runs([10, 20]), [0, 0, 0])
    eng.ctx.max_len = 4
    out = OperatorEngine._op_fox_newton(eng, bytearray(_input(10)), 0, None)
    assert out is None or len(out) <= 4


# ── Fuzzer ───────────────────────────────────────────────────────────


def _fuzzer(**kwargs) -> Fuzzer:
    with (
        tempfile.TemporaryDirectory(prefix="fox_newton_") as tmp,
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(Path(tmp) / "corpus"),
            crashes_dir=str(Path(tmp) / "crashes"),
            **kwargs,
        )


def test_fuzzer_builds_log_only_when_asked():
    assert isinstance(_fuzzer(fox_newton=True)._secant_log, SecantLog)
    assert _fuzzer()._secant_log is None


# ── round feed ───────────────────────────────────────────────────────


def _round(log):
    f = MagicMock()
    f._secant_log = log
    f._cmplog_skip_counter = 0
    f._cmplog.pairs = []
    f._cmplog.collect_tokens.return_value = []
    f._cmplog.last_conds = [_cond(_affine(10))]
    r = FuzzRound(f, b"seed")
    r._mutated = _input(10)
    return f, r


def _collect(r, collect_due: bool) -> None:
    with patch.object(FuzzRound, "_collect_due", return_value=collect_due):
        r._collect_tokens()


def test_collect_round_feeds_executed_input():
    log = MagicMock()
    f, r = _round(log)
    _collect(r, collect_due=True)
    f._cmplog.collect_tokens.assert_called_once_with(CountsDrain.SKIP)
    log.observe.assert_called_once_with(_input(10), f._cmplog.last_conds)


def test_skipped_round_feeds_nothing():
    # Records of a skipped round are not this exec's alone: never attribute them.
    log = MagicMock()
    _, r = _round(log)
    _collect(r, collect_due=False)
    log.observe.assert_not_called()


def test_round_without_log():
    _, r = _round(None)
    _collect(r, collect_due=True)


# ── CLI ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("flag", [True, False])
def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path, flag):
    from fuzzer_tool.cli.commands import cmd_fuzz

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    args.fox_newton = flag
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
    assert cmd_fuzz(args) == 0
    assert seen["fox_newton"] is flag


def test_cli_help_lists_flag():
    import subprocess
    import sys

    cmd = [sys.executable, "-c", "from fuzzer_tool.cli.commands import main; main()"]
    r = subprocess.run([*cmd, "fuzz", "--help"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--fox-newton" in r.stdout


def test_hail_mary_arms_the_flag():
    from fuzzer_tool.cli.commands import _HAIL_MARY_FLAGS

    assert "fox_newton" in _HAIL_MARY_FLAGS
