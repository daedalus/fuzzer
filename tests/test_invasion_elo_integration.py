"""Regression tests: invasion percolation on the Elo meta-strategy ballot.

Percolation handover Module 4, wired into ``operators.select_op()``.
Mirrors ``test_regression_cmaes_elo_ballot.py``'s shape: the ballot
(``available``), the dispatch branch, and the opponent ballot fed to
``record_strategy_match`` (``all_strategies`` in fuzzer.py) all have to
list a new strategy, or it's registered and rated but never actually
selects anything -- exactly the cmaes bug that file documents.

Two things are specific to invasion rather than a copy of that file:
- it has no scheduler object of its own -- it reads ``f.mc.bandit_stats()``
  directly, gated on ``_use_invasion`` and ``mc_bandit`` alone;
- ``bandit_stats()`` covers every registered arm, not just the current
  call's candidate ``ops``, so the dispatch branch must filter it down
  before calling ``invasion_select`` -- an unfiltered call could return an
  operator outside ``ops``, which no other branch here would ever do.
"""

from __future__ import annotations

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.operators import OperatorEngine


class _FakeMC:
    def __init__(self, stats: dict[str, tuple[float, float]], cem_fitted: bool = False):
        self._stats = stats
        self.cem_fitted = cem_fitted
        self.calls = 0

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        return dict(self._stats)

    def select_op(self, ops, prev_op=None) -> str:
        self.calls += 1
        return "op_bandit"


class _FakeElo:
    """Records every ballot offered, and picks invasion whenever it is on one."""

    def __init__(self):
        self.ballots: list[list[str]] = []

    def select_strategy(self, available, temperature=None):
        self.ballots.append(list(available))
        return "invasion" if "invasion" in available else available[0]


class _RecordingRandPool:
    def __init__(self):
        self.choices: list[list[str]] = []

    def choice(self, seq):
        self.choices.append(list(seq))
        return seq[0]


class _FakeFuzzer:
    """Minimal stand-in exposing exactly the attributes select_op() reads."""

    def __init__(self):
        self._stall_recovery_active = False
        self._rand_pool = RandPool()
        self._last_mopt_particles: list = []
        self._prev_bandit_op = None
        self._meta_strategy = None
        self._meta_strategy_cached = None
        self._meta_strategy_used: set[str] = set()
        self._op_time_ema: dict = {}
        self.mc = None
        self.mc_bandit = False
        self.mc_cem = False
        for flag, attr in (
            ("_use_replicator", "_replicator"),
            ("_use_mopt", "_mopt"),
            ("_use_exp3", "_exp3"),
            ("_use_eps_greedy", "_eps_greedy"),
            ("_use_hierarchical", "_hierarchical"),
            ("_use_gp_ucb", "_gp_ucb"),
            ("_use_cmaes", "_cmaes"),
            ("_use_contextual", "_contextual"),
            ("_use_ducb", "_ducb"),
            ("_use_swucb", "_swucb"),
            ("_use_cucb", "_cucb"),
        ):
            setattr(self, flag, False)
            setattr(self, attr, None)
        self._use_invasion = False
        self._use_elo = False
        self._elo = None


class TestInvasionReachableUnderElo:
    def test_invasion_appears_on_the_elo_ballot_and_selects_lowest_resistance(self):
        f = _FakeFuzzer()
        f.mc_bandit = True
        f.mc = _FakeMC({"bit_flip": (9.0, 1.0), "byte_flip": (1.0, 9.0)})
        f._use_invasion = True
        f._use_elo = True
        f._elo = _FakeElo()

        op = OperatorEngine(f).select_op(["bit_flip", "byte_flip"])

        assert "invasion" in f._elo.ballots[0], f"missing from ballot {f._elo.ballots[0]}"
        assert op == "bit_flip"
        assert f.mc.calls == 0  # invasion reads bandit_stats(), not select_op()

    def test_invasion_absent_from_ballot_without_mc_bandit(self):
        """invasion has no effect without --mc-bandit -- it must not even be
        offered, rather than being offered and silently doing nothing."""
        f = _FakeFuzzer()
        f._use_invasion = True
        # Two other schedulers so the ballot has >=2 entries and actually
        # goes through select_strategy -- a single-entry ballot takes a
        # different path (available[0] with no select_strategy call at
        # all) and wouldn't test the ballot-membership condition.
        f._use_exp3 = True
        f._use_eps_greedy = True

        class _FakeScheduler:
            def select_op(self, ops):
                return ops[0]

        f._exp3 = _FakeScheduler()
        f._eps_greedy = _FakeScheduler()
        f._use_elo = True
        f._elo = _FakeElo()

        OperatorEngine(f).select_op(["bit_flip", "byte_flip"])

        assert "invasion" not in f._elo.ballots[0]

    def test_dispatch_filters_bandit_stats_to_candidate_ops(self):
        """bandit_stats() can carry arms outside this call's candidate list
        (e.g. a format-gated operator not offered this round). The result
        must still come from `ops`, never from a better-looking arm outside
        it."""
        f = _FakeFuzzer()
        f.mc_bandit = True
        f.mc = _FakeMC(
            {
                "bit_flip": (1.0, 9.0),
                "byte_flip": (2.0, 8.0),
                "havoc": (99.0, 1.0),  # not a candidate this round
            }
        )
        f._use_invasion = True
        f._use_elo = True
        f._elo = _FakeElo()

        op = OperatorEngine(f).select_op(["bit_flip", "byte_flip"])

        assert op == "byte_flip"  # better of the two actual candidates

    def test_stuck_falls_back_to_random(self):
        """Every candidate at/above the resistance threshold -> invasion_select
        returns None -> falls back to the random terminal, not a crash."""
        f = _FakeFuzzer()
        f.mc_bandit = True
        f.mc = _FakeMC({"bit_flip": (0.0, 10.0), "byte_flip": (0.0, 10.0)})
        f._use_invasion = True
        f._use_elo = True
        f._elo = _FakeElo()
        pool = _RecordingRandPool()
        f._rand_pool = pool

        op = OperatorEngine(f).select_op(["bit_flip", "byte_flip"])

        assert op == "bit_flip"  # pool.choice returns seq[0]
        assert pool.choices == [["bit_flip", "byte_flip"]]

    def test_invasion_not_reachable_without_elo(self):
        """No fallback-chain branch by design: without Elo, the plain
        "bandit" branch (same f.mc/mc_bandit condition, earlier in the
        chain) already covers this case -- invasion must not be consulted
        a second time."""
        f = _FakeFuzzer()
        f.mc_bandit = True
        f.mc = _FakeMC({"bit_flip": (9.0, 1.0)})
        f._use_invasion = True

        op = OperatorEngine(f).select_op(["bit_flip", "byte_flip"])

        assert op == "op_bandit"
        assert f.mc.calls == 1


class TestInvasionOpponentBallot:
    """invasion must also be on the *opponent* ballot record_strategy_match
    iterates over (fuzzer.py's `all_strategies`), or its Elo rating only
    moves on the exec where it was the winner -- the exact bug documented
    in test_regression_cmaes_elo_ballot.py for cmaes."""

    def _stub(self, meta_strategy, used, invasion, mc_bandit):
        f = _FakeFuzzer()
        f._meta_strategy = meta_strategy
        f._meta_strategy_used = set(used)
        f._use_invasion = invasion
        f.mc_bandit = mc_bandit
        f.mc = _FakeMC({})
        f._use_elo = True

        class _RatingElo:
            def __init__(self):
                self._strategy_match_count: dict[str, int] = {}

            def record_strategy_match(self, a, b, score):
                self._strategy_match_count[a] = self._strategy_match_count.get(a, 0) + 1
                self._strategy_match_count[b] = self._strategy_match_count.get(b, 0) + 1

        f._elo = _RatingElo()
        return f

    def test_invasion_is_an_opponent_when_another_strategy_wins(self):
        f = self._stub("bandit", {"bandit"}, invasion=True, mc_bandit=True)
        Fuzzer._record_operator_strategy_matches(f, 1.0)
        assert f._elo._strategy_match_count.get("invasion", 0) == 1

    def test_invasion_not_an_opponent_without_mc_bandit(self):
        f = self._stub("bandit", {"bandit"}, invasion=True, mc_bandit=False)
        Fuzzer._record_operator_strategy_matches(f, 1.0)
        assert "invasion" not in f._elo._strategy_match_count
