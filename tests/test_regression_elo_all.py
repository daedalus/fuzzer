"""Regression tests: --elo all must enable every scheduler (operator meta-
schedulers, seed schedulers, and the mutation scheduling stack), and the
convergence reports must show only schedulers actually used — never
enabled-but-unused ones or stall-recovery pseudo-strategies.
"""

from types import SimpleNamespace

from fuzzer_tool.core.elo import BayesianEloTracker
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import MonteCarloScheduler
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.operators import OperatorEngine
from tests.support.operator_env import install_scheduler_surface
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
            "ducb",
            "kl_ducb",
            "swucb",
            "kl_swucb",
            "cucb",
            "cusum_ucb",
            "c2ucb",
            "fpl",
            "consolidated",
            "moss",
            "contextual",
            "invasion",
            "cmaes",
            "round_robin",
            # seed schedulers
            "ga",
            "qea",
            "bayesian",
            "boltzmann",
            "ecofuzz",
            "markov_generate",
            "mcts",
            "alphabeta",
            "tang",
            "kruskal_count",
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
        # _record_strategy_matches walks every scheduler to build the Elo
        # ballot, so the stub needs the whole surface even though this test
        # is about seed strategies.
        install_scheduler_surface(f)
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
        # Whole surface first, then the handful this test deliberately turns
        # on -- install_scheduler_surface never overwrites what is set.
        install_scheduler_surface(f)
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
        # cmaes joined the opponent ballot when the asymmetry was fixed: it
        # had a dispatch branch and appeared on the select_op ballot, but was
        # absent from all_strategies, so a cmaes-vs-other match was recorded
        # only when cmaes was the *selected* strategy.
        f._cmaes = False
        f._contextual = None
        # The recency/combinatorial family joins the same opponent ballot.
        f._ducb = None
        f._swucb = None
        f._cucb = None
        f._use_invasion = False
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

    def test_canary_included_even_though_never_selected(self):
        # canary is designed to be selected the least of anyone (see
        # core/schedulers/op_canary.py) -- unlike gp_ucb above, it must still
        # show up in the report whenever it's enabled and has accrued match
        # data as an opponent, even though it was never f._meta_strategy.
        f = self._make("bandit", {"bandit"})
        f._use_canary = True
        f._canary = object()
        Fuzzer._record_operator_strategy_matches(f, 1.0)
        assert "canary" not in f._meta_strategy_used
        assert f._elo._strategy_match_count.get("canary", 0) > 0
        assert {r[0] for r in Fuzzer._operator_convergence_rows(f)} == {"bandit", "canary"}

    def test_convergence_rows_sorted_by_rating_descending(self):
        # bandit wins every recorded match, so it should end up rated above
        # its opponents -- and the report should reflect that ranking, not
        # alphabetical order.
        f = self._make("bandit", {"bandit", "mopt"})
        f._use_exp3 = True
        f._exp3 = True
        for _ in range(20):
            Fuzzer._record_operator_strategy_matches(f, 1.0)
        rows = Fuzzer._operator_convergence_rows(f)
        names = [r[0] for r in rows]
        assert names[0] == "bandit"
        ratings = [r[1] for r in rows]
        assert ratings == sorted(ratings, reverse=True)


class TestSeedAndOpArenasNeverCrossCompete:
    """Op-mutator schedulers and seed schedulers share one BayesianEloTracker
    instance but must never be matched against each other: the "seed_"
    prefix keeps them in disjoint keyspaces, ``_record_operator_strategy_matches``
    and ``_record_seed_strategy_matches`` each only ever pass names from their
    own pool to ``record_strategy_match``, and both reporting surfaces
    (``report.py``'s convergence block and ``stats.py``'s live status line)
    must present two separate rankings rather than one blended leaderboard.
    """

    def _make(self):
        f = _StubFuzzer()
        install_scheduler_surface(f)
        f._use_elo = True
        f._elo = BayesianEloTracker()
        f._meta_strategy = "bandit"
        f._meta_strategy_used = {"bandit", "mopt"}
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
        f._cmaes = False
        f._contextual = None
        f._ducb = None
        f._swucb = None
        f._cucb = None
        f._use_invasion = False
        f._seed_strategy = "weighted"
        f._seed_strategy_pool = ["weighted", "pareto"]
        f._seed_strategies_used = {"weighted", "pareto"}
        return f

    def test_no_key_ever_crosses_arenas(self):
        f = self._make()
        for _ in range(5):
            Fuzzer._record_operator_strategy_matches(f, 1.0)
            Fuzzer._record_seed_strategy_matches(f, 1.0)
        op_keys = {"bandit", "mopt"}
        seed_keys = {"seed_weighted", "seed_pareto"}
        assert set(f._elo._strategy_match_count) == op_keys | seed_keys
        # Every op key only ever matched another op key, and vice versa --
        # if a cross-arena match had ever been recorded, an op name would
        # show up with a "seed_" match count contribution it never earned
        # (or the two pools' match counts would fail to partition cleanly).
        assert f._elo._strategy_match_count["bandit"] == 5  # 1 opponent (mopt)
        assert f._elo._strategy_match_count["seed_weighted"] == 5  # 1 opponent (seed_pareto)

    def test_report_partitions_both_pools_with_no_overlap(self):
        f = self._make()
        for _ in range(10):
            Fuzzer._record_operator_strategy_matches(f, 1.0)
            Fuzzer._record_seed_strategy_matches(f, 1.0)
        ranking = f._elo.get_strategy_ranking()
        op_strategies = [p for p in ranking if not p[0].startswith("seed_")]
        seed_strategies = [p for p in ranking if p[0].startswith("seed_")]
        assert {n for n, _ in op_strategies} == {"bandit", "mopt"}
        assert {n for n, _ in seed_strategies} == {"seed_weighted", "seed_pareto"}
        # Partition is total: nothing dropped, nothing double-counted.
        assert len(op_strategies) + len(seed_strategies) == len(ranking)

    def test_live_status_line_reports_separate_arena_leaders(self):
        f = self._make()
        for _ in range(10):
            Fuzzer._record_operator_strategy_matches(f, 1.0)
            Fuzzer._record_seed_strategy_matches(f, 1.0)
        ranking = f._elo.get_strategy_ranking()
        op_ranking = [p for p in ranking if not p[0].startswith("seed_")]
        seed_ranking = [p for p in ranking if p[0].startswith("seed_")]
        assert op_ranking and seed_ranking
        # The two arenas must not be flattened into a single "top" pick --
        # exercise the same split stats.py's live status line performs.
        top_op = op_ranking[0][0]
        top_seed = seed_ranking[0][0][len("seed_") :]
        assert top_op in {"bandit", "mopt"}
        assert top_seed in {"weighted", "pareto"}


class _FakeBandit:
    def __init__(self):
        self.cem_fitted = False

        install_scheduler_surface(self)

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
        install_scheduler_surface(self)
        self._use_invasion = False
        self._use_elo = elo
        self._elo = BayesianEloTracker() if elo else None
        self._rng = RandPool(seed=0)
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
