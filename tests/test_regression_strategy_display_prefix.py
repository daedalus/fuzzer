"""Regression tests for op_/seed_ display names and the per-strategy
Stddev/Rpi/K/Wins columns of the Elo meta-scheduler.

40cbf8a / f13ef9a renamed the scheduler modules to op_*.py and seed_*.py
but the report's strategy section, the live status line and the end-of-run
convergence tables still printed bare names (``bandit``, ``weighted``), and
the strategy section showed only rating and matches while the operator
ranking table above it had gained Stddev/Rpi/K-factor in 3edc370.

The Elo keys themselves are deliberately NOT renamed: they are the arm names
persisted in the state store, so a rename would orphan every saved rating.
"""

import math
from types import SimpleNamespace

from fuzzer_tool.core.analyzers.analyzer_elo import (
    BayesianEloTracker,
    EloTracker,
    seed_strategy_display_name,
    strategy_display_name,
)
from fuzzer_tool.services.report import _elo_ratings, strategy_table_lines
from fuzzer_tool.services.stats import _elo_status_str


def _strategy_section(elo) -> str:
    report = _elo_ratings(SimpleNamespace(_use_elo=True, _elo=elo, mc=None))
    return report[report.index("Meta-scheduler operator strategies (Elo):") :]


def _row(lines: list[str], name: str) -> list[str]:
    return next(line.split() for line in lines if line.split()[:1] == [name])


class TestDisplayName:
    def test_operator_keys_get_op_prefix(self):
        assert strategy_display_name("bandit") == "op_bandit"
        assert strategy_display_name("kl_ducb") == "op_kl_ducb"

    def test_seed_keys_are_shown_as_keyed(self):
        assert strategy_display_name("seed_katz") == "seed_katz"
        assert seed_strategy_display_name("katz") == "seed_katz"

    def test_idempotent(self):
        assert strategy_display_name("op_bandit") == "op_bandit"

    def test_stall_marker_is_not_dressed_as_a_scheduler(self):
        assert strategy_display_name("random_stall") == "random_stall"
        assert seed_strategy_display_name("random_stall") == "random_stall"


class TestReportStrategySection:
    def _elo(self, cls):
        elo = cls(min_matches=1)
        elo.record_round(["bit_flip", "byte_flip"], {"bit_flip"})
        for _ in range(4):
            elo.record_strategy_match("replicator", "bandit", 1.0)
            elo.record_strategy_match("seed_ga", "seed_pareto", 0.2)
        return elo

    def test_names_are_prefixed(self):
        section = _strategy_section(self._elo(BayesianEloTracker))
        op_block, seed_block = section.split("Seed strategies (Elo):")
        assert "op_replicator" in op_block and "op_bandit" in op_block
        assert "seed_ga" in seed_block and "seed_pareto" in seed_block
        # No bare operator name left as a row label.
        assert not any(line.split()[:1] == ["bandit"] for line in op_block.splitlines())

    def test_header_has_new_columns_next_to_rating_wins_matches(self):
        section = _strategy_section(self._elo(BayesianEloTracker))
        header = next(line for line in section.splitlines() if "Strategy" in line)
        cols = header.split()
        for col in ("Rating", "Stddev", "Sigmas", "Rpi", "K", "Wins", "Matches"):
            assert col in cols

    def test_bayesian_row_values(self):
        elo = self._elo(BayesianEloTracker)
        lines = strategy_table_lines(elo, ["replicator", "bandit"], "")
        rep = _row(lines, "op_replicator")
        st = elo.strategy_stats("replicator")
        assert float(rep[3]) == round(math.sqrt(elo._strategy_sigma_sq["replicator"]), 1)
        assert rep[7] == "4" and rep[8] == "4"  # 4 wins of 4 matches
        assert _row(lines, "op_bandit")[7] == "0"
        pool_mean = (st["rating"] + elo.strategy_stats("bandit")["rating"]) / 2
        rpi = 100.0 / (1.0 + 10.0 ** ((pool_mean - st["rating"]) / 400.0))
        assert rep[5] == f"{rpi:.1f}%"

    def test_bayesian_k_is_the_step_the_update_actually_takes(self):
        """K must predict the next rating move, not just the tracker-wide K."""
        elo = BayesianEloTracker(min_matches=1)
        for _ in range(3):
            elo.record_strategy_match("a", "b", 1.0)
        elo.record_strategy_match("c", "b", 1.0)  # fresh c: larger sigma
        k_a = elo.strategy_stats("a")["k"]
        k_c = elo.strategy_stats("c")["k"]
        assert k_c > k_a  # per-strategy, not one figure for every row
        mu_a, mu_b = elo._strategy_mu["a"], elo._strategy_mu["b"]
        expected = elo._expected_score(mu_a, mu_b)
        elo.record_strategy_match("a", "b", 1.0)
        assert math.isclose(elo._strategy_mu["a"] - mu_a, k_a * (1.0 - expected))

    def test_plain_elo_backend_has_no_stddev_but_fixed_k(self):
        elo = self._elo(EloTracker)
        lines = strategy_table_lines(elo, ["replicator", "bandit"], "")
        rep = _row(lines, "op_replicator")
        assert rep[3] == "-"
        assert rep[4] == "-"  # no posterior, no sigmas
        assert rep[6] == f"{elo.k_factor:.2f}"
        assert rep[7] == "4"

    def test_draw_is_nobodys_win(self):
        elo = BayesianEloTracker(min_matches=1)
        elo.record_strategy_match("a", "b", 0.5)
        assert elo.strategy_stats("a")["wins"] == 0
        assert elo.strategy_stats("b")["wins"] == 0

    def test_seed_convergence_keys_render_with_seed_prefix(self):
        elo = self._elo(BayesianEloTracker)
        lines = strategy_table_lines(elo, ["seed_ga", "seed_pareto"], "    ")
        assert _row(lines, "seed_ga")[8] == "4"


class TestWinCountPersistence:
    def test_round_trip_both_backends(self):
        for cls in (BayesianEloTracker, EloTracker):
            elo = cls(min_matches=1)
            elo.record_strategy_match("a", "b", 0.9)
            restored = cls(min_matches=1)
            restored.from_dict(elo.to_dict())
            assert restored.strategy_stats("a")["wins"] == 1

    def test_old_snapshot_without_field_loads_as_zero(self):
        elo = BayesianEloTracker(min_matches=1)
        elo.record_strategy_match("a", "b", 0.9)
        data = elo.to_dict()
        del data["strategy_win_count"]
        restored = BayesianEloTracker(min_matches=1)
        restored.from_dict(data)
        assert restored.strategy_stats("a")["wins"] == 0
        restored.record_strategy_match("a", "b", 1.0)
        assert restored.strategy_stats("a")["wins"] == 1


class TestLiveStatusLine:
    def test_prefixed_names(self):
        elo = BayesianEloTracker(min_matches=1)
        elo.record_strategy_match("corral", "bandit", 1.0)
        elo.record_strategy_match("seed_katz", "seed_tang", 1.0)
        f = SimpleNamespace(_use_elo=True, _elo=elo, _meta_strategy="corral", _seed_strategy="katz")
        line = _elo_status_str(f)
        assert "meta=op_corral" in line
        assert "seed=seed_katz" in line
        assert "top_op=op_corral(" in line
        assert "top_seed=seed_katz(" in line

    def test_stall_and_unset(self):
        elo = BayesianEloTracker(min_matches=1)
        f = SimpleNamespace(
            _use_elo=True, _elo=elo, _meta_strategy="random_stall", _seed_strategy=None
        )
        line = _elo_status_str(f)
        assert "meta=random_stall" in line and "seed=?" in line

    def test_off(self):
        assert _elo_status_str(SimpleNamespace(_use_elo=False, _elo=None)) == ""


def test_sigmas_is_pool_delta_over_posterior_stddev():
    elo = BayesianEloTracker(min_matches=1)
    for _ in range(4):
        elo.record_strategy_match("replicator", "bandit", 1.0)
    lines = strategy_table_lines(elo, ["replicator", "bandit"], "")
    stats = {k: elo.strategy_stats(k) for k in ("replicator", "bandit")}
    pool_mean = sum(s["rating"] for s in stats.values()) / 2
    for key, name in (("replicator", "op_replicator"), ("bandit", "op_bandit")):
        st = stats[key]
        expected = (st["rating"] - pool_mean) / st["stddev"]
        assert _row(lines, name)[4] == f"{expected:+.2f}"
    header = next(line for line in lines if "Strategy" in line).split()
    assert header.index("Stddev") + 1 == header.index("Sigmas")
