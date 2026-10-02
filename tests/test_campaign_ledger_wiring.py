"""--preset-ledger wiring: gap tracking, LedgerSession, CLI preset/stall application."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.campaign_ledger import CampaignLedger
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.ledger import (
    MAX_GAP_KEY,
    MIN_HISTORY,
    STALL_FLOOR,
    LedgerSession,
    build_key,
    ledger_dir,
)

TARGET = "/bin/true"


class _Perplexity:
    def note_new_edge(self) -> None:
        return None


def _fake_round(f) -> SimpleNamespace:
    return SimpleNamespace(
        _f=f, _attribute_edges=lambda new: None, _cmplog_found=False, _smt_found=False
    )


def _fake_fuzzer(**kw) -> SimpleNamespace:
    base = dict(
        exec_count=0,
        _last_new_edge_exec=0,
        _max_edge_gap=0,
        _exec_perplexity=_Perplexity(),
        _novel_input_count=0,
        _matrix_substrate=None,
        op_edges={},
        _stall_note_coverage=lambda n: None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _done(edges: int | None, gap: int) -> SimpleNamespace:
    cov = None if edges is None else SimpleNamespace(cumulative_edges=edges)
    return SimpleNamespace(shm_cov=cov, _max_edge_gap=gap, exec_count=10_000)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class TestGapTracking:
    def test_longest_broken_silence(self) -> None:
        f = _fake_fuzzer()
        rnd = _fake_round(f)
        execs = (100, 150, 1150, 1200)
        for e in execs:
            f.exec_count = e
            FuzzRound._on_new_edges(rnd, {e}, [e])
        gaps = [b - a for a, b in zip((0, *execs), execs, strict=False)]
        assert f._max_edge_gap == max(gaps)


class TestLedgerSession:
    def test_ledger_lives_under_home_fuzzing(self, home: Path) -> None:
        assert ledger_dir(TARGET) == home / "fuzzing" / "true" / "ledger"

    def test_build_key_stable_and_unknown_for_missing(self, tmp_path: Path) -> None:
        assert build_key(TARGET) == build_key(TARGET)
        assert build_key(str(tmp_path / "missing")) == "unknown"

    def test_finish_then_choose_sees_history(self, tmp_path: Path) -> None:
        s = LedgerSession(TARGET, tmp_path)
        assert s.choose(["a", "b"]) == "a"
        assert s.finish(_done(42, 500), "a")

        again = LedgerSession(TARGET, tmp_path)
        assert again.choose(["a", "b"]) == "b"

    def test_stall_prior_needs_history(self, tmp_path: Path) -> None:
        """Falsification: the prior moves with history, and only with enough of it."""
        default = 1000
        gaps = [3000 + 100 * i for i in range(MIN_HISTORY)]
        for gap in gaps[:-1]:
            LedgerSession(TARGET, tmp_path).finish(_done(10, gap), "a")
        assert LedgerSession(TARGET, tmp_path).stall_prior(default) == default

        LedgerSession(TARGET, tmp_path).finish(_done(10, gaps[-1]), "a")
        assert LedgerSession(TARGET, tmp_path).stall_prior(default) == min(gaps)

    def test_adversarial_no_coverage_and_tiny_gaps(self, tmp_path: Path) -> None:
        s = LedgerSession(str(tmp_path / "missing"), tmp_path)
        # No SHM coverage: nothing comparable to record.
        assert not s.finish(_done(None, 50), "a")
        assert len(_loaded(tmp_path)) == 0

        for _ in range(MIN_HISTORY):
            LedgerSession(TARGET, tmp_path).finish(_done(10, 1), "a")
        # A degenerate gap clamps to the floor instead of thrashing recovery.
        assert LedgerSession(TARGET, tmp_path).stall_prior(1000) == STALL_FLOOR

        # A campaign that discovered nothing is no evidence: back to the default.
        LedgerSession(TARGET, tmp_path).finish(_done(10, 0), "a")
        assert LedgerSession(TARGET, tmp_path).stall_prior(1000) == 1000


def _parse(monkeypatch: pytest.MonkeyPatch, *argv: str):
    seen = {}

    def spy(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(commands, "cmd_fuzz", spy)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", TARGET, *argv])
    commands.main()
    return seen["args"]


class TestCliPresets:
    def test_off_by_default(self, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        args = _parse(monkeypatch)
        assert args.ledger_preset is None
        assert not ledger_dir(TARGET).exists()

    def test_first_campaign_runs_first_preset(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = _parse(monkeypatch, "--preset-ledger")
        assert args.ledger_preset == next(iter(commands._LEDGER_PRESETS))

    def test_untried_preset_flags_applied(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names = list(commands._LEDGER_PRESETS)
        session = LedgerSession(TARGET)
        session.finish(_done(10, 5), names[0])

        args = _parse(monkeypatch, "--preset-ledger")
        assert args.ledger_preset == names[1]
        assert args.elo and args.mc_bandit and args.mopt

    def test_explicit_flags_win(self, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        names = list(commands._LEDGER_PRESETS)
        optimal = names.index("optimal")
        for name in names[:optimal]:
            LedgerSession(TARGET).finish(_done(10, 5000), name)

        args = _parse(monkeypatch, "--preset-ledger", "--markov-order", "2", "--stall", "777")
        assert args.ledger_preset == "optimal"
        assert args.markov and args.replicator
        assert args.markov_order == "2"
        assert args.stall == 777

    def test_stall_prior_applied_at_default(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for _ in range(MIN_HISTORY):
            LedgerSession(TARGET).finish(_done(10, 4321), "baseline")
        args = _parse(monkeypatch, "--preset-ledger")
        assert args.stall == 4321

    def test_record_skipped_on_resume(self, home: Path) -> None:
        args = SimpleNamespace(targets=[TARGET], ledger_preset="baseline", resume=True)
        commands._record_ledger(args, _done(10, 5))
        assert len(_loaded(ledger_dir(TARGET))) == 0

        args.resume = False
        commands._record_ledger(args, _done(10, 5))
        assert len(_loaded(ledger_dir(TARGET))) == 1


def _loaded(path: Path) -> CampaignLedger:
    led = CampaignLedger(path)
    led.load()
    return led


def test_max_gap_key_is_recorded(tmp_path: Path) -> None:
    LedgerSession(TARGET, tmp_path).finish(_done(10, 999), "a")
    assert _loaded(tmp_path).min_value(MAX_GAP_KEY, -1, 1, 1) == 999
