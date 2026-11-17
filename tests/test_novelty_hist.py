"""New-edges-per-admitted-seed histogram (Fuzzilli corpus diagnostic)."""

from __future__ import annotations

import pytest

from fuzzer_tool.core.novelty_hist import BUCKET_LABELS, NoveltyHist
from fuzzer_tool.services.stats import StatsReporter
from tests.test_dirichlet_wiring import build  # noqa: F401
from tests.test_regression_enabled_features_entropy_deviation import _make_fake_fuzzer


def _hist(*counts: int) -> NoveltyHist:
    h = NoveltyHist()
    for n in counts:
        h.add(n)
    return h


class TestBuckets:
    @pytest.mark.parametrize(
        ("n", "label"),
        [(0, "0"), (1, "1"), (2, "2-5"), (5, "2-5"), (6, "6-10"), (10, "6-10"), (11, "11+")],
    )
    def test_boundaries(self, n, label):
        counts = _hist(n).counts()
        assert counts[BUCKET_LABELS.index(label)] == 1
        assert sum(counts) == 1

    def test_shares_sum_to_one(self):
        h = _hist(1, 1, 3, 7, 40)
        assert sum(h.shares()) == pytest.approx(1.0)
        assert h.total() == 5

    def test_falsification_empty_has_no_shares(self):
        # A fresh histogram must not divide by zero or invent mass.
        h = NoveltyHist()
        assert h.total() == 0
        assert h.shares() == tuple(0.0 for _ in BUCKET_LABELS)

    def test_adversarial_negative_and_huge(self):
        # Negative counts are impossible upstream; clamp to the 0 bucket, never index -1.
        h = _hist(-5, 10**18)
        assert h.counts()[0] == 1
        assert h.counts()[-1] == 1


class TestPersistence:
    def test_round_trip(self):
        h = _hist(0, 1, 4, 9, 12)
        g = NoveltyHist()
        g.from_dict(h.to_dict())
        assert g.counts() == h.counts()

    @pytest.mark.parametrize(
        "payload",
        [
            {"version": 999, "counts": [1, 2, 3, 4, 5]},
            {"version": 1, "counts": [1, 2]},
            {"version": 1, "counts": [1, 2, 3, 4, -1]},
            {"version": 1, "counts": "abcde"},
            "garbage",
        ],
    )
    def test_adversarial_malformed_ignored(self, payload):
        h = _hist(1)
        h.from_dict(payload)
        assert h.counts() == _hist(1).counts()


class TestWiring:
    def test_admit_records_last_new_edge_count(self, build):  # noqa: F811
        f = build()
        f._last_new_edge_count = 7
        f._corpus_manager.save_to_corpus(b"novel-seed-7")
        assert f._novelty_hist.counts()[BUCKET_LABELS.index("6-10")] == 1

    def test_duplicate_not_counted(self, build):  # noqa: F811
        f = build()
        f._last_new_edge_count = 1
        f._corpus_manager.save_to_corpus(b"dup")
        f._corpus_manager.save_to_corpus(b"dup")
        assert f._novelty_hist.total() == 1

    def test_resume_round_trip(self, build):  # noqa: F811
        f = build()
        f._novelty_hist.add(3)
        f._save_learned()
        g = build()
        g.resume = True
        g._state_store = f._state_store
        g._load_learned()
        assert g._novelty_hist.counts() == f._novelty_hist.counts()

    def test_summary_line(self, capsys):
        fake = _make_fake_fuzzer(_novelty_hist=_hist(1, 1, 3, 20))
        StatsReporter(fake)._print_summary_novelty(fake)
        out = capsys.readouterr().out
        assert "New edges/seed" in out and "1:2 (50%)" in out

    def test_summary_silent_when_empty(self, capsys):
        fake = _make_fake_fuzzer(_novelty_hist=NoveltyHist())
        StatsReporter(fake)._print_summary_novelty(fake)
        assert capsys.readouterr().out == ""
