"""--lightweight: execs/s mode. Coverage, crashes and admission only.

Covers the CLI preset (_apply_lightweight), the lean round (RunMode.LITE),
and the two switches it adds: --no-markov and --no-auto-minimize.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.operator_registry import _AVAILABLE
from fuzzer_tool.core.run_mode import RunMode, Switch
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.fuzzer import Fuzzer

_SEGV = (139, "segv")
_CLEAN = (0, "")
_SEED = b"SEEDDATA-0123456789-ABCDEFGHIJ"
_MUTANT = b"MUTANT-0123456789-abcdefghij-XYZ"


# ── CLI ──────────────────────────────────────────────────────────────


def _cli(monkeypatch, *argv: str) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    """Run main() on ``fuzz /bin/true *argv``; return parsed args + fuzz parser."""
    captured: dict[str, object] = {}
    real_apply = commands._apply_lightweight

    def _spy_apply(args, fuzz_parser):
        captured["parser"] = fuzz_parser
        real_apply(args, fuzz_parser)

    def _spy_fuzz(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(commands, "_apply_lightweight", _spy_apply)
    monkeypatch.setattr(commands, "cmd_fuzz", _spy_fuzz)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *argv])
    assert commands.main() == 0
    return captured["args"], captured.get("parser")


def test_preset_turns_features_off(monkeypatch):
    """Falsification: --lightweight sets every listed dest to its off value."""
    args, _ = _cli(monkeypatch, "--lightweight")

    for dest, off in commands._LIGHTWEIGHT_OFF.items():
        assert getattr(args, dest) == off, f"--lightweight left {dest!r} on"
    assert args.no_cmplog is True
    assert args.no_markov is True
    assert args.no_auto_minimize is True


def test_normal_mode_untouched(monkeypatch):
    """Control: without --lightweight the same dests keep their defaults."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(commands, "cmd_fuzz", lambda a: captured.setdefault("a", a) and 0)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true"])
    commands.main()
    args = captured["a"]

    assert args.lightweight is False
    assert args.no_cmplog is False
    assert args.no_markov is False
    assert args.no_auto_minimize is False
    assert args.trace is True


def test_explicit_flag_wins(monkeypatch):
    """Adversarial: an explicit True-default flag survives the preset."""
    args, _ = _cli(monkeypatch, "--lightweight", "--resize-map-on-stall")

    assert args.resize_map_on_stall is True
    assert args.no_cmplog is True


@pytest.mark.parametrize("extra", [("--hail-mary",), ("--elo", "all")])
def test_rejects_heavy_presets(monkeypatch, extra):
    """Adversarial: --lightweight cannot be stacked with the everything-on presets."""
    monkeypatch.setattr(commands, "cmd_fuzz", lambda a: 0)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", "--lightweight", *extra])
    with pytest.raises(SystemExit):
        commands.main()


def _default_on_dests(parser: argparse.ArgumentParser) -> set[str]:
    # Features on by default: True defaults, plus `--no-*`-only store_true opt-outs.
    dests = set()
    for action in parser._actions:
        if action.default is True:
            dests.add(action.dest)
            continue
        longs = [s for s in action.option_strings if s.startswith("--")]
        opt_out = longs and all(s.startswith("--no-") for s in longs)
        if isinstance(action, argparse._StoreTrueAction) and opt_out:
            dests.add(action.dest)
    return dests


def test_every_default_on_classified(monkeypatch):
    """Gate: each default-on feature is either switched off or kept on purpose."""
    _, parser = _cli(monkeypatch, "--lightweight")
    classified = set(commands._LIGHTWEIGHT_OFF) | commands._LIGHTWEIGHT_KEEP

    missing = _default_on_dests(parser) - classified
    assert not missing, f"classify in _LIGHTWEIGHT_OFF or _LIGHTWEIGHT_KEEP: {missing}"
    assert not set(commands._LIGHTWEIGHT_OFF) & commands._LIGHTWEIGHT_KEEP


# ── Fuzzer ───────────────────────────────────────────────────────────


def _make(tmp: str, **kwargs) -> Fuzzer:
    corpus = Path(tmp) / "corpus"
    corpus.mkdir()
    (corpus / "seed").write_bytes(_SEED)
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(corpus),
            crashes_dir=str(Path(tmp) / "crashes"),
            max_len=256,
            timeout=1,
            **kwargs,
        )


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory(prefix="lightweight_") as d:
        yield d


def _round(f: Fuzzer, result: tuple[int, str], crash: bool, new_cov: bool) -> bool:
    with (
        patch.object(f, "_dedup_mutate", return_value=_MUTANT),
        patch.object(f, "_run_target", return_value=result),
        patch.object(f, "_is_crash", return_value=crash),
        patch.object(f, "_is_interesting", return_value=False),
        patch.object(FuzzRound, "_scan_coverage", _fake_scan(new_cov)),
        patch.object(FuzzRound, "_lite_scan", _fake_scan(new_cov)),
    ):
        return f.fuzz_one(f.corpus[0])


def _fake_scan(new_cov: bool):
    def scan(self):
        self._has_new_coverage = new_cov
        self._f._current_edges_cache = {1, 2, 3}

    return scan


def test_lite_round_skips_credit(tmp):
    """Falsification: a LITE round admits without op credit, lifetimes or trim."""
    f = _make(tmp, run_mode=RunMode.LITE)
    with (
        patch.object(FuzzRound, "_credit_ops") as credit,
        patch.object(f._edge_tracker, "record_edge_lifetimes") as lifetimes,
        patch.object(f, "_trim_new_coverage") as trim,
    ):
        assert _round(f, _CLEAN, crash=False, new_cov=True) is True

    assert _MUTANT in f.corpus
    credit.assert_not_called()
    lifetimes.assert_not_called()
    trim.assert_not_called()


def test_full_round_credits(tmp):
    """Control: the same round in FULL mode does credit the operators."""
    f = _make(tmp)
    with patch.object(FuzzRound, "_credit_ops") as credit:
        assert _round(f, _CLEAN, crash=False, new_cov=True) is True
    credit.assert_called_once()


def test_lite_crash_saved(tmp):
    """Adversarial: LITE still counts and saves a crash, and does not admit it."""
    f = _make(tmp, run_mode=RunMode.LITE)
    before = f.crash_count

    assert _round(f, _SEGV, crash=True, new_cov=True) is True

    assert f.crash_count == before + 1
    assert any(Path(f.crashes_dir).iterdir())
    assert _MUTANT not in f.corpus


def test_lite_boring_round(tmp):
    """Adversarial: a LITE round with nothing new admits nothing."""
    f = _make(tmp, run_mode=RunMode.LITE)
    size = len(f.corpus)

    assert _round(f, _CLEAN, crash=False, new_cov=False) is False
    assert len(f.corpus) == size


def test_markov_off_never_trains(tmp):
    """Falsification: Switch.OFF leaves the chain untrained through admissions."""
    f = _make(tmp, markov=Switch.OFF)
    _round(f, _CLEAN, crash=False, new_cov=True)

    assert f.markov.is_trained() is False
    assert _AVAILABLE["markov_bytes"](f, _SEED) is False


def test_markov_on_trains(tmp):
    """Control: the default trains on the seed corpus and arms markov_bytes."""
    f = _make(tmp)

    assert f.markov.is_trained() is True
    assert _AVAILABLE["markov_bytes"](f, _SEED) is True


def test_no_auto_minimize_trim(tmp):
    """Falsification: Switch.OFF skips trimming on a FULL-mode admission."""
    f = _make(tmp, auto_minimize=Switch.OFF)
    with patch.object(f, "_trim_new_coverage") as trim:
        assert _round(f, _CLEAN, crash=False, new_cov=True) is True
    trim.assert_not_called()


def test_auto_minimize_trims(tmp):
    """Control: by default the same admission is trimmed."""
    f = _make(tmp)
    with patch.object(f, "_trim_new_coverage") as trim:
        _round(f, _CLEAN, crash=False, new_cov=True)
    trim.assert_called_once()


def test_no_auto_minimize_prune(tmp):
    """Adversarial: a far-oversized corpus is not pruned when auto-minimize is off."""
    f = _make(tmp, auto_minimize=Switch.OFF)
    f.corpus.extend(bytes([i % 256, i // 256]) * 8 for i in range(12000))
    f.exec_count = 10_000
    with patch.object(f, "_auto_minimize_corpus") as prune:
        f._check_corpus_size_and_prune()
    prune.assert_not_called()


def test_auto_minimize_prunes(tmp):
    """Control: the same oversized corpus is pruned by default."""
    f = _make(tmp)
    f.corpus.extend(bytes([i % 256, i // 256]) * 8 for i in range(12000))
    f.exec_count = 10_000
    with patch.object(f, "_auto_minimize_corpus") as prune:
        f._check_corpus_size_and_prune()
    prune.assert_called_once()


def _bloat(f: Fuzzer) -> None:
    # A history pinned at max_len trips seed_size_bloat on the next admission.
    f._corpus_size_history = [f.max_len] * 200
    f.exec_count = 1000
    f._last_bloat_warn_exec = 0
    f._corpus_manager._after_admit(b"\x00" * f.max_len)


def test_no_auto_minimize_bloat(tmp):
    """Adversarial: a bloated corpus does not schedule a minimize when off."""
    f = _make(tmp, auto_minimize=Switch.OFF)
    _bloat(f)
    assert f._minimize_pending is False


def test_auto_minimize_on_bloat(tmp):
    """Control: by default the same bloat schedules a minimize."""
    f = _make(tmp)
    _bloat(f)
    assert f._minimize_pending is True


# ── ICFG / K-Scheduler node channel ──────────────────────────────────


@pytest.mark.parametrize(
    ("mode", "builds"),
    [(RunMode.LITE, 0), (RunMode.FULL, 1)],
)
def test_lite_skips_icfg_build(tmp, mode, builds):
    """Falsification: LITE never builds the ICFG; FULL still tries (control)."""
    from fuzzer_tool.services.katz_channel import KatzChannel

    with patch.object(KatzChannel, "build", return_value=None) as build:
        f = _make(tmp, run_mode=mode)

    assert build.call_count == builds
    assert f._katz_channel is None


# ── --cmplog (explicit counterpart of --no-cmplog) ───────────────────


def test_cmplog_flag_overrides_lightweight(monkeypatch):
    """--lightweight turns cmplog off, but an explicit --cmplog keeps it on."""
    args, _ = _cli(monkeypatch, "--lightweight", "--cmplog")

    assert args.no_cmplog is False
    assert args.no_markov is True  # the rest of the preset still applies


def test_cmplog_flag_is_noop_by_default(monkeypatch):
    """Control: --cmplog alone equals the default (cmplog on)."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(commands, "cmd_fuzz", lambda a: captured.setdefault("a", a) and 0)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", "--cmplog"])
    commands.main()

    assert captured["a"].no_cmplog is False


@pytest.mark.parametrize(
    ("flags", "expected_no_cmplog"),
    [
        (("--no-cmplog", "--cmplog"), False),
        (("--cmplog", "--no-cmplog"), True),
        (("--no-cmplog",), True),
    ],
)
def test_cmplog_vs_no_cmplog_last_wins(monkeypatch, flags, expected_no_cmplog):
    """Adversarial: both flags share one dest, so the last one given decides."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(commands, "cmd_fuzz", lambda a: captured.setdefault("a", a) and 0)
    monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "/bin/true", *flags])
    commands.main()

    assert captured["a"].no_cmplog is expected_no_cmplog


def test_default_still_comes_from_no_cmplog(monkeypatch):
    """--cmplog must not change the parser default of the shared dest."""
    _, parser = _cli(monkeypatch, "--lightweight")

    assert parser.get_default("no_cmplog") is False
