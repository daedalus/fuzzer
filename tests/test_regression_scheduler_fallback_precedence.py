"""Regression tests pinning select_op() fallback precedence (Elo off).

With Elo arbitration disabled, operators.select_op() falls back to a fixed
chain (operators.py:1632–1656): replicator → mopt → bandit → exp3 →
eps_greedy → hierarchical → gp_ucb → cmaes → contextual → ducb → swucb →
cucb → random. These tests pin that contract
so future scheduler additions/removals cannot silently change which
scheduler wins, and document that cem is reachable only via Elo.
"""

import pytest

from fuzzer_tool.core.elo import BayesianEloTracker
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine
from fuzzer_tool.services.report import _elo_ratings

# Fallback precedence, highest first. "random" is the terminal fallback (the
# chain's else-branch, represented by _rand_pool.choice, not a scheduler).
_FALLBACK_PRECEDENCE = [
    "replicator",
    "mopt",
    "bandit",
    "exp3",
    "eps_greedy",
    "hierarchical",
    "gp_ucb",
    "cmaes",
    "contextual",
    "ducb",
    "swucb",
    "cucb",
]


class _FakeMC:
    """Stand-in for the MonteCarloScheduler: carries cem_fitted and a
    recording select_op (bandit dispatch calls f.mc.select_op)."""

    def __init__(self, cem_fitted: bool = False):
        self.cem_fitted = cem_fitted
        self.calls = 0

    def select_op(self, ops: list[str], prev_op=None) -> str:
        self.calls += 1
        return "op_bandit"


class _RecordingScheduler:
    """Fake scheduler that records consultations and returns a marker op."""

    def __init__(self, name: str):
        self.name = name
        self.calls = 0

    def select_op(self, ops: list[str]) -> str:
        self.calls += 1
        return f"op_{self.name}"


class _RecordingMopt(_RecordingScheduler):
    """MOpt's select_op returns (op, particle_id) — the chain unpacks it."""

    def select_op(self, ops: list[str]) -> tuple[str, int]:
        self.calls += 1
        return "op_mopt", 7


class _RecordingContextual(_RecordingScheduler):
    """LinUCB's select_op takes a second (context) argument — either a
    shared vector or a per-arm callable; the chain always passes a
    callable (OperatorEngine._context_vector)."""

    def select_op(self, ops: list[str], context) -> str:
        self.calls += 1
        assert callable(context), "contextual dispatch must pass a per-arm context callable"
        return "op_contextual"


class _FakeFuzzer:
    """Minimal fuzzer stand-in exposing exactly the attrs select_op() reads."""

    _SCHEDULER_ATTRS = {
        "replicator": ("_use_replicator", "_replicator"),
        "mopt": ("_use_mopt", "_mopt"),
        "exp3": ("_use_exp3", "_exp3"),
        "eps_greedy": ("_use_eps_greedy", "_eps_greedy"),
        "hierarchical": ("_use_hierarchical", "_hierarchical"),
        "gp_ucb": ("_use_gp_ucb", "_gp_ucb"),
        "cmaes": ("_use_cmaes", "_cmaes"),
        "contextual": ("_use_contextual", "_contextual"),
        "ducb": ("_use_ducb", "_ducb"),
        "swucb": ("_use_swucb", "_swucb"),
        "cucb": ("_use_cucb", "_cucb"),
    }

    def __init__(self):
        self._stall_recovery_active = False
        self._use_elo = False
        self._elo = None
        self._last_mopt_particles: list = []
        self._prev_bandit_op = None
        self._meta_strategy = None
        self._meta_strategy_cached = None
        self._meta_strategy_used: set[str] = set()
        self._rand_pool = RandPool()
        self.mc = _FakeMC()
        self.mc_bandit = False
        self.mc_cem = False
        self._op_time_ema: dict = {}
        for flag, attr in self._SCHEDULER_ATTRS.values():
            setattr(self, flag, False)
            setattr(self, attr, None)

    def enable(self, name: str) -> _RecordingScheduler | _FakeMC:
        """Enable scheduler *name* and return a recording fake for it."""
        if name == "bandit":
            self.mc_bandit = True
            self.mc = _FakeMC()
            return self.mc
        fake: _RecordingScheduler
        if name == "mopt":
            fake = _RecordingMopt(name)
        elif name == "contextual":
            fake = _RecordingContextual(name)
        else:
            fake = _RecordingScheduler(name)
        flag, attr = self._SCHEDULER_ATTRS[name]
        setattr(self, flag, True)
        setattr(self, attr, fake)
        return fake


class _RecordingRandPool:
    """Records random-fallback choices (RandPool uses __slots__, so it can't
    be monkeypatched in place)."""

    def __init__(self):
        self.choices: list[str] = []

    def choice(self, seq):
        self.choices.append("random")
        return seq[0]


class TestFallbackPrecedence:
    def test_highest_priority_enabled_wins(self):
        """All 8 schedulers enabled → only replicator is consulted."""
        f = _FakeFuzzer()
        fakes = {name: f.enable(name) for name in _FALLBACK_PRECEDENCE}
        op = OperatorEngine(f).select_op(["bit_flip", "byte_flip"])
        assert op == "op_replicator"
        assert fakes["replicator"].calls == 1
        for name, fake in fakes.items():
            if name != "replicator":
                assert fake.calls == 0, f"{name} consulted despite replicator enabled"

    @pytest.mark.parametrize(
        ("enabled", "expected"),
        [
            (["replicator"], "replicator"),
            (["mopt"], "mopt"),
            (["bandit"], "bandit"),
            (["exp3"], "exp3"),
            (["eps_greedy"], "eps_greedy"),
            (["hierarchical"], "hierarchical"),
            (["gp_ucb"], "gp_ucb"),
            (["contextual"], "contextual"),
            # Prefix-off subsets: the first enabled in precedence order wins.
            (["gp_ucb", "exp3"], "exp3"),
            (["contextual", "gp_ucb"], "gp_ucb"),
            (["contextual", "exp3"], "exp3"),
            (["hierarchical", "bandit"], "bandit"),
            (["mopt", "replicator", "eps_greedy"], "replicator"),
            (
                [
                    "contextual",
                    "gp_ucb",
                    "hierarchical",
                    "eps_greedy",
                    "exp3",
                    "bandit",
                    "mopt",
                    "replicator",
                ],
                "replicator",
            ),
        ],
    )
    def test_precedence_order(self, enabled, expected):
        f = _FakeFuzzer()
        fakes = {name: f.enable(name) for name in enabled}
        op = OperatorEngine(f).select_op(["bit_flip"])
        assert op == f"op_{expected}"
        assert fakes[expected].calls == 1
        for name, fake in fakes.items():
            if name != expected:
                assert fake.calls == 0, f"{name} consulted before {expected}"

    def test_cem_alone_without_elo_falls_to_random(self):
        """cem is reachable only via Elo — with Elo off it must not be
        consulted and selection falls through to the random terminal."""
        f = _FakeFuzzer()
        f.mc_cem = True
        f.mc = _FakeMC(cem_fitted=True)
        pool = _RecordingRandPool()
        f._rand_pool = pool
        op = OperatorEngine(f).select_op(["bit_flip", "byte_flip"])
        assert op == "bit_flip"
        assert f.mc.calls == 0, "cem consulted without Elo"
        assert pool.choices == ["random"]

    def test_mopt_particle_appended(self):
        """MOpt's (op, particle_id) tuple must be unpacked and the particle
        appended to _last_mopt_particles."""
        f = _FakeFuzzer()
        f.enable("mopt")
        op = OperatorEngine(f).select_op(["bit_flip"])
        assert op == "op_mopt"
        assert f._last_mopt_particles == [7]


class TestReportGroupSeparation:
    def test_ranking_report_separates_operator_and_seed_groups(self):
        """The strategy-ranking report must split plain-key operator
        strategies from seed_*-keyed seed strategies into separate blocks —
        operator strategy names must never appear in the seed block and vice
        versa (they are disjoint Elo keyspaces, not one group)."""
        elo = BayesianEloTracker(min_matches=1)
        # Operator-level match so the operator ranking section is non-empty.
        elo.record_round(["bit_flip", "byte_flip"], {"bit_flip"})
        # Strategy-level matches: one plain pair, one seed_*-keyed pair.
        elo.record_strategy_match("replicator", "bandit", 1.0)
        elo.record_strategy_match("seed_ga", "seed_pareto", 1.0)

        f = _FakeFuzzer()
        f._use_elo = True
        f._elo = elo
        f.mc = None  # skip the bandit-comparison section

        report = _elo_ratings(f)
        assert "Meta-scheduler operator strategies (Elo):" in report
        assert "Seed strategies (Elo):" in report
        op_block, seed_block = report.split("Seed strategies (Elo):")
        assert "replicator" in op_block and "bandit" in op_block
        assert "seed_ga" not in op_block and "seed_pareto" not in op_block
        assert "seed_ga" in seed_block and "seed_pareto" in seed_block
        assert "replicator" not in seed_block and "bandit" not in seed_block
