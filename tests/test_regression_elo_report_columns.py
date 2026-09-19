"""Regression tests for the Stddev/Rpi/K-factor columns in the Elo ranking
table (services/report.py::_elo_ratings).

Covers both tracker backends since they source these columns differently:
EloTracker has no rating posterior (Stddev falls back to reward-moment
stddev), while BayesianEloTracker has a per-operator posterior sigma_sq
(Stddev is sqrt(sigma_sq)). Rpi (expected score vs the pool mean) is
computed the same way for both, straight from the rating.
"""

import math
from types import SimpleNamespace

from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker, EloTracker
from fuzzer_tool.services.report import _elo_ratings


def _fake_fuzzer(elo):
    return SimpleNamespace(_use_elo=True, _elo=elo, mc=None)


class TestEloTrackerColumns:
    def test_header_lists_new_columns_in_order(self):
        elo = EloTracker(k_factor=24.0, min_matches=1)
        elo.record_round(["a", "b"], {"a"})
        report = _elo_ratings(_fake_fuzzer(elo))
        header_line = next(line for line in report.splitlines() if "Rank" in line)
        # Columns must appear in the order the user asked for: Rating,
        # then Stddev, then Rpi, then K-factor.
        assert header_line.index("Rating") < header_line.index("Stddev")
        assert header_line.index("Stddev") < header_line.index("Rpi")
        assert header_line.index("Rpi") < header_line.index("K-fctr")

    def test_k_factor_column_matches_tracker_constant(self):
        elo = EloTracker(k_factor=24.0, min_matches=1)
        elo.record_round(["a", "b"], {"a"})
        report = _elo_ratings(_fake_fuzzer(elo))
        row_lines = [
            line for line in report.splitlines() if line.strip().startswith(("1 ", "2 "))
        ]
        assert row_lines, report
        for line in row_lines:
            assert "24.0" in line

    def test_stddev_uses_reward_moment_spread(self):
        elo = EloTracker(k_factor=16.0, min_matches=1)
        # Give "a" a mix of wins/losses so its reward stddev is nonzero and
        # computable (needs >1 sample), while "b" only ever loses.
        elo.record_match("a", "b", score_a=1.0)
        elo.record_match("b", "a", score_a=1.0)  # b beats a this time
        report = _elo_ratings(_fake_fuzzer(elo))
        moments_a = elo.get_reward_moments("a")
        assert moments_a is not None and moments_a.count > 1
        expected_sd = f"{moments_a.stddev:.1f}"
        a_line = next(line for line in report.splitlines() if " a " in f" {line} ")
        assert expected_sd in a_line

    def test_rpi_is_expected_score_against_pool_mean(self):
        elo = EloTracker(k_factor=16.0, min_matches=1)
        elo.record_round(["a", "b"], {"a"})
        report = _elo_ratings(_fake_fuzzer(elo))
        ranking = elo.get_ranking()
        pool_mean = sum(r for _, r in ranking) / len(ranking)
        for op, rating in ranking:
            expected_rpi = 100.0 / (1.0 + 10.0 ** ((pool_mean - rating) / 400.0))
            line = next(line for line in report.splitlines() if f" {op:<22s}" in line)
            assert f"{expected_rpi:.1f}%" in line

    def test_footnote_names_reward_moment_stddev(self):
        elo = EloTracker(k_factor=16.0, min_matches=1)
        elo.record_round(["a", "b"], {"a"})
        report = _elo_ratings(_fake_fuzzer(elo))
        assert "match-score stddev" in report


class TestBayesianEloTrackerColumns:
    def test_stddev_uses_posterior_sigma(self):
        elo = BayesianEloTracker(min_matches=1)
        elo.record_round(["a", "b"], {"a"})
        report = _elo_ratings(_fake_fuzzer(elo))
        expected_sd_a = math.sqrt(elo.sigma_sq["a"])
        a_line = next(line for line in report.splitlines() if " a " in f" {line} ")
        assert f"{expected_sd_a:.1f}" in a_line

    def test_k_factor_column_uses_effective_k(self):
        elo = BayesianEloTracker(min_matches=1)
        elo.record_round(["a", "b"], {"a"})
        report = _elo_ratings(_fake_fuzzer(elo))
        eff_k = elo._effective_k()
        row_lines = [
            line for line in report.splitlines() if line.strip().startswith(("1 ", "2 "))
        ]
        assert row_lines, report
        for line in row_lines:
            assert f"{eff_k:.1f}" in line

    def test_footnote_names_posterior_sigma(self):
        elo = BayesianEloTracker(min_matches=1)
        elo.record_round(["a", "b"], {"a"})
        report = _elo_ratings(_fake_fuzzer(elo))
        assert "posterior rating sigma" in report
