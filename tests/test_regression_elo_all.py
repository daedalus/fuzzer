"""Regression tests: --elo all must enable every scheduler (operator meta-
schedulers, seed schedulers, and the mutation scheduling stack), and the
convergence reports must show only schedulers actually used — never
enabled-but-unused ones or stall-recovery pseudo-strategies.
"""

from types import SimpleNamespace

from fuzzer_tool.core.elo import BayesianEloTracker
from fuzzer_tool.core.schedulers import MonteCarloScheduler
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.operators import OperatorEngine
from tests.test_commands_extended import TestCmdFuzzConstruction


class TestEloAllEnablesAllSchedulers:
    """--elo all must flip every scheduler flag, not just list it as available."""

    def test_elo_all_flips_all_scheduler_flags(self, monkeypatch, tmp_path):
        args = TestCmdFuzzConstruction()._make_default_args(tmp_path)
        args.elo = "all"
        captured = {}

        def fake_fuzzer(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(run=lambda iterations: 0)

        monkeypatch.setattr("fuzzer_tool.cli.commands.Fuzzer", fake_fuzzer)
        from fuzzer_tool.cli.commands import cmd_fuzz

        assert cmd_fuzz(args) == 0
        assert captured["elo"]  # "all" passes through as truthy to Fuzzer
        for flag in (
            # operator meta-schedulers (Elo-arbitrated)
            "mc_bandit",
            "mc_cem",
            "mopt",
            "replicator",
            "exp3",
            "eps_greedy",
            "hierarchical_bandit",
            "gp_ucb",
            # seed schedulers
            "ga",
            "qea",
            "bayesian",
            "boltzmann",
            "markov_generate",
            # mutation scheduling stack (non-Elo features)
            "metropolis",
            "shapley",
            "mi_guided",
            "secretary",
            "wfc",
            "lineage",
        ):
            assert captured[flag] is True, f"{flag} should be enabled by --elo all"
        assert captured["schedule"] == "fast"


class _StubFuzzer:
    """Minimal fuzzer exposing only the attrs the record/convergence helpers touch."""


class TestSeedEloRecordsPoolOnly:
    def _make(self, strategy, pool, used, elo=True):
        f = _StubFuzzer()
        f._use_elo = elo
        f._elo = BayesianEloTracker() if elo else None
        f._seed_strategy = strategy
        f._seed_strategy_pool = list(pool)
        f._seed_strategies_used = set(used)
        return f

    def test_records_only_pool_members(self):
        f = self._make("weighted", ["weighted", "pareto"], {"weighted"})
        Fuzzer._record_seed_strategy_matches(f, 1.0)
        assert f._elo._strategy_match_count == {"seed_weighted": 1, "seed_pareto": 1}

    def test_strategy_outside_pool_records_nothing(self):
        f = self._make("random_stall", ["weighted", "pareto"], {"weighted"})
        Fuzzer._record_seed_strategy_matches(f, 1.0)
        assert f._elo._strategy_match_count == {}

    def test_unused_strategy_records_nothing(self):
        f = self._make("weighted", ["weighted"], set())
        Fuzzer._record_seed_strategy_matches(f, 1.0)
        assert f._elo._strategy_match_count == {}

    def test_no_elo_is_noop(self):
        f = self._make("weighted", ["weighted"], {"weighted"}, elo=False)
        Fuzzer._record_seed_strategy_matches(f, 1.0)
        assert f._elo is None

    def test_convergence_rows_only_used(self):
        f = self._make("weighted", ["weighted", "pareto"], {"weighted", "pareto"})
        Fuzzer._record_seed_strategy_matches(f, 1.0)
        rows = Fuzzer._seed_convergence_rows(f)
        assert {r[0] for r in rows} == {"weighted", "pareto"}
        for _name, rating, delta, count in rows:
            assert rating > 0
            assert count >= 1
            assert abs(delta - (rating - f._elo.initial_mu)) < 1e-6

    def test_unused_strategy_excluded_from_report(self):
        f = self._make("weighted", ["weighted", "pareto"], {"weighted"})
        Fuzzer._record_seed_strategy_matches(f, 1.0)
        assert [r[0] for r in Fuzzer._seed_convergence_rows(f)] == ["weighted"]


class TestOperatorEloRecordsUsedOnly:
    def _make(self, strategy, used, elo=True):
        f = _StubFuzzer()
        f._use_elo = elo
        f._elo = BayesianEloTracker() if elo else None
        f._meta_strategy = strategy
        f._meta_strategy_used = set(used)
        f._use_replicator = False
        f._replicator = None
        f.mc = SimpleNamespace(cem_fitted=False)
        f.mc_bandit = True
        f.mc_cem = False
        f._use_mopt = True
        f._mopt = object()
        f._exp3 = False
        f._eps_greedy = False
        f._hierarchical = False
        f._gp_ucb = False
        return f

    def test_records_against_enabled_schedulers_only(self):
        f = self._make("bandit", {"bandit", "mopt"})
        Fuzzer._record_operator_strategy_matches(f, 1.0)
        assert f._elo._strategy_match_count == {"bandit": 1, "mopt": 1}

    def test_disabled_scheduler_not_an_opponent(self):
        # mopt is enabled by _make (an opponent); exp3 is disabled and must
        # not appear as an opponent
        f = self._make("bandit", {"bandit"})
        Fuzzer._record_operator_strategy_matches(f, 1.0)
        assert "exp3" not in f._elo._strategy_match_count
        assert set(f._elo._strategy_match_count) == {"bandit", "mopt"}

    def test_random_stall_records_nothing(self):
        f = self._make("random_stall", set())
        Fuzzer._record_operator_strategy_matches(f, 1.0)
        assert f._elo._strategy_match_count == {}

    def test_convergence_rows_only_used(self):
        f = self._make("bandit", {"bandit", "mopt"})
        Fuzzer._record_operator_strategy_matches(f, 1.0)
        rows = Fuzzer._operator_convergence_rows(f)
        assert {r[0] for r in rows} == {"bandit", "mopt"}
        for _name, rating, _delta, count in rows:
            assert rating > 0
            assert count >= 1

    def test_enabled_but_never_selected_excluded_from_report(self):
        # gp_ucb is enabled (so it can be an Elo opponent) but was never
        # selected as the active scheduler → must not appear in the report
        f = self._make("bandit", {"bandit"})
        f._gp_ucb = True
        Fuzzer._record_operator_strategy_matches(f, 1.0)
        assert [r[0] for r in Fuzzer._operator_convergence_rows(f)] == ["bandit"]


class _FakeBandit:
    def __init__(self):
        self.cem_fitted = False

    def select_op(self, ops, prev_op=None):
        return "bit_flip"


class _FakeFuzzerForSelectOp:
    """Attribute surface select_op (operators.py) reads from the fuzzer."""

    def __init__(self, elo=True):
        self._stall_recovery_active = False
        self._meta_strategy = None
        self._meta_strategy_cached = None
        self._meta_strategy_used = set()
        self._use_replicator = False
        self._replicator = None
        self.mc = None
        self.mc_bandit = False
        self.mc_cem = False
        self._use_mopt = False
        self._mopt = None
        self._use_exp3 = False
        self._exp3 = None
        self._use_eps_greedy = False
        self._eps_greedy = None
        self._use_hierarchical = False
        self._hierarchical = None
        self._use_gp_ucb = False
        self._gp_ucb = None
        self._use_elo = elo
        self._elo = BayesianEloTracker() if elo else None
        import random

        self._rand_pool = random
        self._last_mopt_particles = []
        self._prev_bandit_op = None


class TestSelectOpTracksUsedSchedulers:
    def test_single_arm_records_the_arm(self):
        f = _FakeFuzzerForSelectOp()
        f.mc = _FakeBandit()
        f.mc_bandit = True
        engine = OperatorEngine(f)
        engine.select_op(["bit_flip", "byte_flip"])
        assert f._meta_strategy_used == {"bandit"}
        assert f._meta_strategy == "bandit"

    def test_multi_arm_tracks_every_selected_scheduler(self):
        f = _FakeFuzzerForSelectOp()
        f.mc = _FakeBandit()
        f.mc_bandit = True
        f._use_gp_ucb = True
        f._gp_ucb = SimpleNamespace(select_op=lambda ops: "byte_flip")

        class _Mopt:
            def select_op(self, ops):
                return ("byte_flip", 0)

        f._use_mopt = True
        f._mopt = _Mopt()
        engine = OperatorEngine(f)
        for _ in range(150):
            f._meta_strategy_cached = None  # mutate() resets this each exec
            engine.select_op(["bit_flip", "byte_flip"])
        assert len(f._meta_strategy_used) >= 1
        assert f._meta_strategy_used <= {"bandit", "mopt", "gp_ucb"}
        # every recorded strategy is one of the enabled schedulers
        assert not (f._meta_strategy_used - {"bandit", "mopt", "gp_ucb"})

    def test_random_stall_not_tracked(self):
        f = _FakeFuzzerForSelectOp()
        f.mc = _FakeBandit()
        f.mc_bandit = True
        f._stall_recovery_active = True
        engine = OperatorEngine(f)
        engine.select_op(["bit_flip"])
        assert f._meta_strategy_used == set()


class TestBanditStatsReportBasis:
    """The bandit convergence block shows only arms with real evidence;
    bandit_stats() subtracts priors so never-tried arms read (0, 0)."""

    def test_never_tried_arm_reports_zero_evidence(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.init_arm("byte_flip")
        stats = mc.bandit_stats()
        assert stats == {"bit_flip": (0.0, 0.0), "byte_flip": (0.0, 0.0)}

    def test_tried_arm_reports_positive_evidence(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.record("bit_flip", 1.0)
        stats = mc.bandit_stats()
        assert stats["bit_flip"][0] > 0  # success evidence after a win
