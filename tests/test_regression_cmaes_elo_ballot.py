"""Regression tests: CMA-ES must be a candidate on the Elo meta-strategy ballot.

``operators.select_op()`` builds ``available`` -- the list of scheduling
strategies Elo arbitrates between -- and omitted ``cmaes``, while carrying a
live ``strategy == "cmaes"`` dispatch branch just below it and a
``_use_cmaes`` branch in the no-Elo fallback chain below that.

The effect was not a preference, it was a disappearance. Elo names a strategy
from ``available``; the chain then matches that strategy's branch and returns,
so the fallback chain -- the only place ``_use_cmaes`` is ever read -- is not
evaluated at all. With ``--cma-es --elo`` and any second scheduler enabled,
CMA-ES had its arms registered from ``REGISTRY.names()`` and was fed
``record()`` on every outcome (``fuzzer.py`` includes it in the
``record(op, success, weight=...)`` fan-out), while selecting nothing,
forever. It was a scheduler learning full-time and driving nothing.

Note the shape of what made this invisible: every individual piece looked
correct in isolation. The arm registration was there, the reward wiring was
there, the dispatch branch was there, and ``_FALLBACK_PRECEDENCE`` in
``test_regression_scheduler_fallback_precedence.py`` already listed ``cmaes``
in the right position. Only the ballot was missing, and nothing asserted what
went onto the ballot.
"""

from __future__ import annotations

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine


class _RecordingScheduler:
    def __init__(self, label: str):
        self.label = label
        self.calls = 0

    def select_op(self, ops):
        self.calls += 1
        return f"op_{self.label}"


class _FakeMC:
    def __init__(self):
        self.cem_fitted = False
        self.calls = 0

    def select_op(self, ops, prev_op=None):
        self.calls += 1
        return "op_bandit"


class _FakeElo:
    """Records every ballot offered, and picks cmaes whenever it is on one."""

    def __init__(self):
        self.ballots: list[list[str]] = []

    def select_strategy(self, available):
        self.ballots.append(list(available))
        return "cmaes" if "cmaes" in available else available[0]


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
        self.mc = _FakeMC()
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


class TestCMAESReachableUnderElo:
    def test_cmaes_appears_on_the_elo_ballot(self):
        """Before the fix the ballot was ['bandit'] and cmaes never ran."""
        f = _FakeFuzzer()
        f.mc_bandit = True
        f._use_cmaes = True
        f._cmaes = _RecordingScheduler("cmaes")
        f._use_elo = True
        f._elo = _FakeElo()

        op = OperatorEngine(f).select_op(["bit_flip", "byte_flip"])

        assert "cmaes" in f._elo.ballots[0], f"cmaes missing from Elo ballot {f._elo.ballots[0]}"
        assert op == "op_cmaes"
        assert f._cmaes.calls == 1
        assert f.mc.calls == 0

    def test_cmaes_with_elo_and_no_other_scheduler_is_not_shut_out(self):
        """len(available) < 2 takes the single-strategy Elo path, which reads
        the same list -- cmaes must be a candidate there too rather than
        losing by absence and falling through."""
        f = _FakeFuzzer()
        f._use_cmaes = True
        f._cmaes = _RecordingScheduler("cmaes")
        f._use_elo = True
        f._elo = _FakeElo()

        op = OperatorEngine(f).select_op(["bit_flip", "byte_flip"])
        assert op == "op_cmaes"
        assert f._meta_strategy == "cmaes"

    def test_ballot_position_matches_fallback_precedence(self):
        """cmaes sits between gp_ucb and contextual, matching the documented
        fallback order in test_regression_scheduler_fallback_precedence."""
        f = _FakeFuzzer()
        for flag, attr, label in (
            ("_use_gp_ucb", "_gp_ucb", "gp_ucb"),
            ("_use_cmaes", "_cmaes", "cmaes"),
            ("_use_contextual", "_contextual", "contextual"),
        ):
            setattr(f, flag, True)
            setattr(f, attr, _RecordingScheduler(label))
        f._use_elo = True
        f._elo = _FakeElo()

        OperatorEngine(f).select_op(["bit_flip", "byte_flip"])
        assert f._elo.ballots[0] == ["gp_ucb", "cmaes", "contextual"]

    def test_cmaes_still_reachable_with_elo_off(self):
        """The no-Elo fallback chain was already correct -- pin it so the
        ballot fix cannot regress the path that was working."""
        f = _FakeFuzzer()
        f._use_cmaes = True
        f._cmaes = _RecordingScheduler("cmaes")

        assert OperatorEngine(f).select_op(["bit_flip"]) == "op_cmaes"
        assert f._cmaes.calls == 1
