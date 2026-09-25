"""The seed-overhead (RSS/seed) stats fragment must never take down
print_stats.

Same hazard as test_regression_stats_gini_fragments.py: a stand-in that
answers every attribute (MagicMock, a partially restored state) must still
degrade to an empty string rather than raising out of print_stats. This is
a pure diagnostic -- it must never be read by any scheduler or scoring
path, only checked here for safe formatting.
"""

from unittest.mock import MagicMock

from fuzzer_tool.services.stats import StatsReporter


def _reporter():
    return StatsReporter.__new__(StatsReporter)


class TestSeedOverheadFragment:
    def test_no_rss_yields_empty(self):
        f = MagicMock()
        f._peak_rss = 0
        f.corpus = [b"a", b"b"]
        assert _reporter()._print_stats_seed_overhead_str(f) == ""

    def test_absent_rss_yields_empty(self):
        f = object()
        assert _reporter()._print_stats_seed_overhead_str(f) == ""

    def test_empty_corpus_yields_empty(self):
        f = MagicMock()
        f._peak_rss = 2646 * 1024
        f.corpus = []
        assert _reporter()._print_stats_seed_overhead_str(f) == ""

    def test_absent_corpus_yields_empty(self):
        f = MagicMock()
        f._peak_rss = 2646 * 1024
        del f.corpus
        f.corpus = None
        assert _reporter()._print_stats_seed_overhead_str(f) == ""

    def test_computes_mb_per_seed(self):
        f = MagicMock()
        f._peak_rss = 2646 * 1024  # 2646MB, matches the status-line example
        f.corpus = [b"x"] * 949
        out = _reporter()._print_stats_seed_overhead_str(f)
        assert out == " | seed-ovh: 2.79MB/seed"

    def test_single_seed_corpus(self):
        f = MagicMock()
        f._peak_rss = 1024  # 1MB
        f.corpus = [b"only"]
        out = _reporter()._print_stats_seed_overhead_str(f)
        assert out == " | seed-ovh: 1.00MB/seed"
