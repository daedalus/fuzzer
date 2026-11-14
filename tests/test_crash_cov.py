"""Crash-only virgin map (Fuzzilli cov_evaluate_crash)."""

from __future__ import annotations

import numpy as np
import pytest

from fuzzer_tool.core.crash_cov import DENSE_MAX, CrashCoverage, SigNovelty
from fuzzer_tool.services.fuzz_round import FuzzRound
from fuzzer_tool.services.stats import StatsReporter
from tests.test_dirichlet_wiring import build  # noqa: F401
from tests.test_regression_enabled_features_entropy_deviation import _make_fake_fuzzer


def _ids(*xs: int) -> np.ndarray:
    return np.array(xs, dtype=np.uint64)


class TestRecord:
    def test_first_crash_edges_are_new(self):
        c = CrashCoverage()
        assert c.record(_ids(3, 4, 4), SigNovelty.NEW) == 2
        assert c.summary() == {"crashes": 1, "cov_new": 1, "sig_dup_cov_new": 0, "edges": 2}

    def test_falsification_repeat_adds_nothing(self):
        # Same edges twice: the second crash must not count as coverage-new.
        c = CrashCoverage()
        c.record(_ids(3, 4), SigNovelty.NEW)
        assert c.record(_ids(4, 3), SigNovelty.DUP) == 0
        assert c.summary()["cov_new"] == 1
        assert c.summary()["sig_dup_cov_new"] == 0

    def test_sig_dup_with_new_edge_flagged(self):
        c = CrashCoverage()
        c.record(_ids(3), SigNovelty.NEW)
        assert c.record(_ids(3, 9), SigNovelty.DUP) == 1
        assert c.summary()["sig_dup_cov_new"] == 1

    def test_empty_crash_counts_but_adds_nothing(self):
        c = CrashCoverage()
        assert c.record(_ids(), SigNovelty.NEW) == 0
        assert c.summary() == {"crashes": 1, "cov_new": 0, "sig_dup_cov_new": 0, "edges": 0}

    def test_adversarial_ids_past_dense_cap(self):
        # Synthetic ids (bit 31) and anything >= DENSE_MAX go to the sparse side.
        c = CrashCoverage()
        big = DENSE_MAX + 5
        synth = 1 << 31
        assert c.record(_ids(1, big, synth, big), SigNovelty.NEW) == 3
        assert c.record(_ids(big, synth), SigNovelty.DUP) == 0
        assert c.summary()["edges"] == 3


class TestPersistence:
    def test_round_trip(self):
        c = CrashCoverage()
        c.record(_ids(2, DENSE_MAX + 1), SigNovelty.NEW)
        c.record(_ids(2, 8), SigNovelty.DUP)
        d = CrashCoverage()
        d.from_dict(c.to_dict())
        assert d.summary() == c.summary()
        assert d.record(_ids(2, 8, DENSE_MAX + 1), SigNovelty.DUP) == 0

    @pytest.mark.parametrize(
        "payload",
        [
            {"version": 999},
            {"version": 1, "edges": [1], "counts": [1, 2]},
            {"version": 1, "edges": [-1], "counts": [1, 1, 0]},
            {"version": 1, "edges": "x", "counts": [1, 1, 0]},
            "garbage",
        ],
    )
    def test_adversarial_malformed_ignored(self, payload):
        c = CrashCoverage()
        c.record(_ids(5), SigNovelty.NEW)
        before = c.summary()
        c.from_dict(payload)
        assert c.summary() == before


class TestWiring:
    def _crash(self, f, monkeypatch, edges, name):
        monkeypatch.setattr(f, "_get_current_edge_array", lambda: _ids(*edges))
        monkeypatch.setattr(f, "save_crash", lambda *a: name)
        r = FuzzRound(f, b"seed")
        r._mutated = b"boom"
        r._returncode = -11
        return r

    def test_on_crash_records_edges(self, build, monkeypatch):  # noqa: F811
        f = build()
        self._crash(f, monkeypatch, {1, 2}, "c1")._on_crash()
        self._crash(f, monkeypatch, {1, 2, 3}, False)._on_crash()
        assert f._crash_cov.summary() == {
            "crashes": 2,
            "cov_new": 2,
            "sig_dup_cov_new": 1,
            "edges": 3,
        }

    def test_lite_crash_records_edges(self, build, monkeypatch):  # noqa: F811
        f = build()
        self._crash(f, monkeypatch, {4}, "c1")._lite_crash()
        assert f._crash_cov.summary()["edges"] == 1

    def test_resume_round_trip(self, build):  # noqa: F811
        f = build()
        f._crash_cov.record(_ids(7), SigNovelty.NEW)
        f._save_learned()
        g = build()
        g.resume = True
        g._state_store = f._state_store
        g._load_learned()
        assert g._crash_cov.summary() == f._crash_cov.summary()

    def test_summary_line(self, capsys):
        c = CrashCoverage()
        c.record(_ids(1), SigNovelty.NEW)
        c.record(_ids(1, 2), SigNovelty.DUP)
        fake = _make_fake_fuzzer(_crash_cov=c)
        StatsReporter(fake)._print_summary_crash_cov(fake)
        out = capsys.readouterr().out
        assert "Crash coverage" in out and "2 crashes" in out and "1 sig-dup" in out

    def test_summary_silent_without_crashes(self, capsys):
        fake = _make_fake_fuzzer(_crash_cov=CrashCoverage())
        StatsReporter(fake)._print_summary_crash_cov(fake)
        assert capsys.readouterr().out == ""
