"""Classic op schedulers: HRRN, Q-learning, tabu search, hill climbing.

Unit behavior with scripted RNGs (Hard Rule 39) plus Fuzzer / ballot / CLI wiring.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_hill_climb import OpHillClimbScheduler
from fuzzer_tool.core.schedulers.op_hrrn import OpHRRNScheduler
from fuzzer_tool.core.schedulers.op_qlearn import OpQLearnScheduler
from fuzzer_tool.core.schedulers.op_tabu import OpTabuScheduler
from fuzzer_tool.services.operators import operator_strategy_pool
from tests.support.operator_env import install_scheduler_surface
from tests.support.scripted_rng import ScriptedRng

OP_ARMS = {
    "op_hrrn": OpHRRNScheduler,
    "op_qlearn": OpQLearnScheduler,
    "op_tabu": OpTabuScheduler,
    "op_hill_climb": OpHillClimbScheduler,
}


def _build(cls):
    try:
        return cls(rng=RandPool(seed=3))
    except TypeError:
        return cls()


def _beta_mean(s: float, f: float) -> float:
    """Beta(1, 1) posterior mean, derived independently of ArmCounts."""
    return (s + 1.0) / (s + f + 2.0)


def _drive(sched, ops, outcome, steps):
    """Select/record loop; *outcome* maps op -> success."""
    picks = []
    for _ in range(steps):
        op = sched.select_op(ops)
        sched.record(op, outcome[op])
        picks.append(op)
    return picks


# --- Interface --------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(OP_ARMS))
class TestInterface:
    def test_empty_and_single(self, name):
        s = _build(OP_ARMS[name])

        assert s.select_op([]) == ""
        assert s.select_op(["only"]) == "only"

    def test_supports_priors_declared_false(self, name):
        assert OP_ARMS[name].__dict__["supports_priors"] is False

    def test_record_clamps(self, name):
        s = _build(OP_ARMS[name])
        s.record("x", True, weight=5.0)
        s.record("x", False)

        assert s.bandit_stats() == {"x": (1.0, 1.0)}

    def test_result_is_a_candidate(self, name):
        """Adversarial: a recorded op outside the candidate list is never returned."""
        s = _build(OP_ARMS[name])
        for _ in range(5):
            s.record("zzz", True)

        assert s.select_op(["a", "b"]) in ("a", "b")

    def test_nan_weight_is_failure(self, name):
        s = _build(OP_ARMS[name])
        s.record("x", True, weight=float("nan"))

        assert s.bandit_stats() == {"x": (0.0, 1.0)}


# --- HRRN -------------------------------------------------------------------


class TestHRRN:
    def test_unseen_ties_pick_first(self):
        s = OpHRRNScheduler()
        s.init_arm("a")
        s.init_arm("b")

        assert s.select_op(["a", "b"]) == "a"

    def test_ratio_is_one_plus_wait_times_mean(self):
        s = OpHRRNScheduler()
        s.record("a", True)  # tick 1
        s.record("b", False)  # tick 2
        s.record("c", True)  # tick 3

        # ratio = 1 + wait * mean, wait counting the slot being scheduled:
        # a waits 3 at 2/3, b waits 2 at 1/3, c waits 1 at 2/3.
        ratios = {
            "a": 1 + 3 * _beta_mean(1, 0),
            "b": 1 + 2 * _beta_mean(0, 1),
            "c": 1 + 1 * _beta_mean(1, 0),
        }
        assert s.ratios(["a", "b", "c"]) == pytest.approx(ratios)
        assert s.select_op(["a", "b", "c"]) == "a"

    def test_share_follows_mean_without_starvation(self):
        """Falsification: argmax-of-mean would never serve 'bad'; HRRN must."""
        picks = _drive(OpHRRNScheduler(), ["good", "bad"], {"good": True, "bad": False}, 60)

        assert picks.count("good") > picks.count("bad") > 0

    def test_select_is_pure(self):
        s = OpHRRNScheduler()
        s.record("a", True)
        s.record("b", True)

        assert s.select_op(["a", "b"]) == s.select_op(["a", "b"])


# --- Q-learning -------------------------------------------------------------


class TestQLearn:
    def test_requires_rng(self):
        with pytest.raises(ValueError, match="RandPool"):
            OpQLearnScheduler(rng=None)

    @pytest.mark.parametrize(
        "kw", [{"alpha": 0.0}, {"alpha": 1.5}, {"gamma": 1.0}, {"gamma": -0.1}, {"epsilon": 1.5}]
    )
    def test_rejects_bad_params(self, kw):
        with pytest.raises(ValueError):
            OpQLearnScheduler(rng=RandPool(seed=1), **kw)

    def test_bellman_update_uses_next_state_max(self):
        alpha, gamma = 0.5, 0.9
        s = OpQLearnScheduler(rng=RandPool(seed=1), alpha=alpha, gamma=gamma)
        s.record("a", True)  # Q[start,a] = alpha * 1
        s.record("b", True)  # Q[a,b] = alpha * 1
        s.record("a", False)  # Q[b,a] = alpha * (0 + gamma * max Q[a,.])

        assert s.q_value("a", "b") == pytest.approx(alpha)
        assert s.q_value("b", "a") == pytest.approx(alpha * gamma * alpha)

    def test_exploit_picks_best_successor_of_last_op(self):
        s = OpQLearnScheduler(rng=ScriptedRng(randoms=[0.5]), epsilon=0.1)
        s.record("a", True)
        s.record("b", True)
        s.record("a", False)  # state is now "a"; Q[a,b] > Q[a,a] = 0

        assert s.select_op(["a", "b"]) == "b"

    def test_explore_draws_uniform(self):
        s = OpQLearnScheduler(rng=ScriptedRng(randoms=[0.05], choice_idxs=[2]), epsilon=0.1)
        s.record("a", True)

        assert s.select_op(["a", "b", "c"]) == "c"

    def test_state_is_context_not_global_mean(self):
        """Falsification: a context-free bandit would pick 'a' (higher mean) after 'a'."""
        s = OpQLearnScheduler(rng=ScriptedRng(randoms=[0.5]), epsilon=0.1, gamma=0.0)
        for _ in range(3):
            s.record("b", False)
            s.record("a", True)  # b -> a pays
            s.record("b", True)  # a -> b pays
        s.record("a", True)  # state "a"; never seen a -> a

        assert s.select_op(["a", "b"]) == "b"

    def test_epsilon_zero_never_draws(self):
        """Adversarial: ScriptedRng with no values raises if any draw happens."""
        s = OpQLearnScheduler(rng=ScriptedRng(), epsilon=0.0)
        s.record("a", True)

        assert s.select_op(["a", "b"]) == "a"


# --- Tabu search ------------------------------------------------------------


def _ranked_tabu(tenure):
    s = OpTabuScheduler(tenure=tenure)
    s.record("a", True)
    s.record("a", True)
    s.record("b", True)
    s.record("c", False)
    return s


class TestTabu:
    def test_rejects_negative_tenure(self):
        with pytest.raises(ValueError):
            OpTabuScheduler(tenure=-1)

    def test_tenure_zero_is_greedy(self):
        """Falsification: without a tabu list, best mean wins every time."""
        s = _ranked_tabu(0)

        assert [s.select_op(["a", "b", "c"]) for _ in range(4)] == ["a"] * 4

    def test_recent_picks_are_tabu(self):
        s = _ranked_tabu(1)

        assert [s.select_op(["a", "b", "c"]) for _ in range(4)] == ["a", "b", "a", "b"]

    def test_longer_tenure_reaches_worse_ops(self):
        s = _ranked_tabu(2)

        assert [s.select_op(["a", "b", "c"]) for _ in range(6)] == ["a", "b", "c"] * 2

    def test_all_tabu_falls_back_to_least_recent(self):
        """Adversarial: tenure longer than the candidate list must not stall."""
        s = _ranked_tabu(10)

        assert [s.select_op(["a", "b"]) for _ in range(4)] == ["a", "b", "a", "b"]


# --- Hill climbing ----------------------------------------------------------


def _ranked_climber(rng, restart=0.05):
    s = OpHillClimbScheduler(rng=rng, restart=restart)
    s.record("a", True)
    s.record("b", False)
    return s


class TestHillClimb:
    def test_requires_rng(self):
        with pytest.raises(ValueError, match="RandPool"):
            OpHillClimbScheduler(rng=None)

    @pytest.mark.parametrize("restart", [-0.1, 1.5])
    def test_rejects_bad_restart(self, restart):
        with pytest.raises(ValueError):
            OpHillClimbScheduler(rng=RandPool(seed=1), restart=restart)

    def test_first_pick_takes_the_neighbour(self):
        s = _ranked_climber(ScriptedRng(choice_idxs=[1]))

        assert s.select_op(["a", "b"]) == "b"

    def test_moves_uphill(self):
        s = _ranked_climber(ScriptedRng(choice_idxs=[1, 0]))
        s.select_op(["a", "b"])  # current = b

        assert s.select_op(["a", "b"]) == "a"

    def test_rejects_downhill_without_restart(self):
        """Falsification: a uniform picker would follow the neighbour to 'b'."""
        s = _ranked_climber(ScriptedRng(choice_idxs=[0, 1], randoms=[0.9]))
        s.select_op(["a", "b"])  # current = a

        assert s.select_op(["a", "b"]) == "a"

    def test_restart_accepts_downhill(self):
        s = _ranked_climber(ScriptedRng(choice_idxs=[0, 1], randoms=[0.01]))
        s.select_op(["a", "b"])

        assert s.select_op(["a", "b"]) == "b"

    def test_current_dropped_from_candidates(self):
        """Adversarial: the held op vanishing from the ballot moves to the neighbour."""
        s = _ranked_climber(ScriptedRng(choice_idxs=[0, 0]))
        s.select_op(["a", "b"])  # current = a

        assert s.select_op(["b", "c"]) == "b"


# --- Fuzzer wiring ----------------------------------------------------------


def _real_fuzzer(tmp_path, **kw):
    from fuzzer_tool.services.fuzzer import Fuzzer

    corpus, crashes = tmp_path / "c", tmp_path / "k"
    corpus.mkdir(parents=True)
    crashes.mkdir(parents=True)
    return Fuzzer(
        target="targets/test_target",
        corpus_dir=str(corpus),
        crashes_dir=str(crashes),
        max_len=4096,
        **kw,
    )


class TestFuzzer:
    @pytest.mark.parametrize("name", sorted(OP_ARMS))
    def test_elo_only(self, name):
        from fuzzer_tool.services.fuzzer import _OPERATOR_STRATEGY_NAMES
        from fuzzer_tool.services.operators import _FALLBACK_PRECEDENCE

        assert name not in _OPERATOR_STRATEGY_NAMES
        assert name not in _FALLBACK_PRECEDENCE

    def test_kwargs_default_off_and_appended(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        params = inspect.signature(Fuzzer.__init__).parameters
        names = list(params)
        for kw in OP_ARMS:
            assert params[kw].default is False
            assert names.index(kw) > names.index("seed_drr_scheduler")

    def test_off_by_default(self, tmp_path):
        f = _real_fuzzer(tmp_path)

        for name in OP_ARMS:
            assert getattr(f, f"_{name}") is None
            assert getattr(f, f"_use_{name}") is False

    def test_flags_build_the_arms(self, tmp_path):
        f = _real_fuzzer(tmp_path, **dict.fromkeys(OP_ARMS, True))

        for name, cls in OP_ARMS.items():
            assert isinstance(getattr(f, f"_{name}"), cls)

    def test_registration_and_record_fanout(self):
        from fuzzer_tool.services import fuzzer as fz
        from fuzzer_tool.services.fuzz_round import FuzzRound

        src = inspect.getsource(fz.Fuzzer)
        record_loop = inspect.getsource(FuzzRound)
        tuple_src = record_loop[record_loop.index("for scheduler in (") :]
        tuple_src = tuple_src[: tuple_src.index("):")]
        for name in OP_ARMS:
            assert f"_register_arms(self._{name})" in src
            assert f"f._{name}," in tuple_src


@pytest.mark.parametrize("name", sorted(OP_ARMS))
class TestOpBallot:
    def test_on_ballot_when_enabled(self, name):
        f = SimpleNamespace(mc=None)
        install_scheduler_surface(f)
        setattr(f, f"_use_{name}", True)
        setattr(f, f"_{name}", _build(OP_ARMS[name]))

        assert name in operator_strategy_pool(f)

    def test_absent_when_off(self, name):
        f = SimpleNamespace(mc=None)
        install_scheduler_surface(f)

        assert name not in operator_strategy_pool(f)

    def test_elo_dispatch_uses_the_arm(self, name):
        from fuzzer_tool.services.operators import _PLAIN_STRATEGIES

        assert _PLAIN_STRATEGIES[name] == f"_{name}"


def test_cli_flags_reach_fuzzer_not_hail_mary():
    """Unmeasured arms: on the parser and passed through, kept out of --hail-mary."""
    from fuzzer_tool.cli import commands
    from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

    tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
    ]
    dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))

    for kw in OP_ARMS:
        assert all(kw in {k.arg for k in c.keywords} for c in calls)
        assert kw in dests
        assert kw not in commands._HAIL_MARY_FLAGS
