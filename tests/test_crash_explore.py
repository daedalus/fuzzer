"""Crash exploration mode (AFL ``-C``; lcamtuf, Nov 2014).

Seed with crashing inputs. A mutant is kept only if it still crashes *and*
reaches a crash path not seen before; non-crashing mutants are dropped. The
result is a corpus of related crash variants whose fault addresses show how
much control the input has over the fault.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fuzzer_tool.core.crash_explore import CrashExplorer
from fuzzer_tool.services.fuzz_round import FuzzRound
from tests import test_commands_extended


class TestExplorer:
    def test_first_crash_path_is_new(self):
        assert CrashExplorer().observe({1, 2, 3})

    def test_same_path_is_not_new(self):
        ex = CrashExplorer()
        ex.observe({1, 2, 3})
        assert not ex.observe({1, 2, 3})

    def test_subset_is_not_new(self):
        ex = CrashExplorer()
        ex.observe({1, 2, 3})
        assert not ex.observe({2})

    def test_one_new_edge_is_new(self):
        ex = CrashExplorer()
        ex.observe({1, 2, 3})
        assert ex.observe({1, 2, 3, 4})

    def test_variant_count_tracks_new_paths_only(self):
        ex = CrashExplorer()
        for edges in ({1}, {1}, {1, 2}, {2}):
            ex.observe(edges)
        assert ex.variants == 2

    def test_empty_trace_is_never_new(self):
        # Adversarial: a crash with no readable trace (SHM unavailable,
        # crash before the first edge) must not flood the corpus.
        ex = CrashExplorer()
        assert not ex.observe(set())
        assert not ex.observe(())
        assert ex.variants == 0

    def test_bitmap_reads_nonzero_indices(self):
        # Adversarial: ptrace hands a byte bitmap, not edge ids. Iterating
        # it raw would record hit *counts* as edge ids.
        ex = CrashExplorer()
        assert ex.observe(bytes([0, 3, 0, 1]))
        assert not ex.observe({1, 3})
        assert ex.observe({0})

    def test_accepts_any_int_iterable(self):
        ex = CrashExplorer()
        assert ex.observe(iter([5, 6]))
        assert not ex.observe([6, 5])


# ── FuzzRound gate ───────────────────────────────────────────────────────


def _round(explorer, *, crash: bool, new_cov: bool, edges=frozenset({1, 2})):
    f = MagicMock()
    f._crash_explorer = explorer
    f._current_edges_cache = set(edges)
    r = FuzzRound(f, b"seed")
    r._is_crash = crash
    r._has_new_coverage = new_cov
    r._is_interesting = new_cov
    r._is_new_max = new_cov
    r._is_cmp_progress = new_cov
    r._is_new_valid_coverage = new_cov
    r._is_slow = new_cov
    return r


class TestGate:
    def test_non_crash_with_new_coverage_is_dropped(self):
        r = _round(CrashExplorer(), crash=False, new_cov=True)
        r._gate_explore()
        r._judge()
        assert not r._is_crash
        assert not r._admits()
        assert not r._success

    def test_crash_on_new_path_survives(self):
        r = _round(CrashExplorer(), crash=True, new_cov=False)
        r._gate_explore()
        r._judge()
        assert r._is_crash and r._success

    def test_crash_on_known_path_is_dropped(self):
        ex = CrashExplorer()
        ex.observe({1, 2})
        r = _round(ex, crash=True, new_cov=True)
        r._gate_explore()
        r._judge()
        assert not r._is_crash
        assert not r._success

    def test_gate_is_inert_when_mode_off(self):
        # Falsification: normal fuzzing must keep every signal.
        r = _round(None, crash=False, new_cov=True)
        r._gate_explore()
        assert r._has_new_coverage and r._admits()


class TestRouting:
    def _stub_pipeline(self, monkeypatch, crash: bool):
        for name in (
            "_begin",
            "_execute",
            "_mine_cmplog",
            "_periodic",
            "_count_ops",
            "_scan_coverage",
            "_observe",
            "_credit_seed",
            "_feed_models",
            "_record_edges",
            "_learn_format",
            "_track_edges",
            "_credit_ops",
        ):
            monkeypatch.setattr(FuzzRound, name, lambda self: None)

        def classify(self):
            self._is_crash = crash

        monkeypatch.setattr(FuzzRound, "_classify", classify)
        monkeypatch.setattr(FuzzRound, "_on_crash", lambda self: True)

    def test_new_crash_variant_enters_corpus(self, monkeypatch):
        self._stub_pipeline(monkeypatch, crash=True)
        f = MagicMock()
        f._crash_explorer = CrashExplorer()
        f._current_edges_cache = {7}
        f.corpus = []
        r = FuzzRound(f, b"seed")
        r._mutated = b"variant"
        assert r.run()
        f.save_to_corpus.assert_called_once_with(b"variant", parent=b"seed")

    def test_crash_outside_mode_not_queued(self, monkeypatch):
        self._stub_pipeline(monkeypatch, crash=True)
        f = MagicMock()
        f._crash_explorer = None
        f._current_edges_cache = {7}
        r = FuzzRound(f, b"seed")
        assert r.run()
        f.save_to_corpus.assert_not_called()


# ── CLI wiring ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("flag", [True, False])
def test_cli_flag_reaches_fuzzer(monkeypatch, tmp_path, flag):
    from fuzzer_tool.cli.commands import cmd_fuzz

    args = test_commands_extended.TestCmdFuzzConstruction()._make_default_args(tmp_path)
    args.crash_explore = flag
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake)
    assert cmd_fuzz(args) == 0
    assert seen["crash_explore"] is flag


TARGET = str(Path(__file__).resolve().parent.parent / "targets" / "test_target")


def test_fuzzer_builds_explorer_only_when_asked(tmp_path):
    from fuzzer_tool.services.fuzzer import Fuzzer

    built = []
    for i, flag in enumerate((False, True)):
        corpus = tmp_path / f"c{i}"
        (corpus / "seeds").mkdir(parents=True)
        (corpus / "seeds" / "s").write_bytes(b"x")
        f = Fuzzer(
            target=TARGET,
            corpus_dir=str(corpus),
            crashes_dir=str(tmp_path / f"k{i}"),
            crash_explore=flag,
        )
        built.append(f._crash_explorer)
    assert built[0] is None
    assert isinstance(built[1], CrashExplorer)
