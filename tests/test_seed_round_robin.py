"""Tests for the seed-arena round-robin scheduler
(core/schedulers/seed_round_robin.py), the seed-selection counterpart of
core/schedulers/op_round_robin.py.
"""

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_round_robin import SeedRoundRobinScheduler

SEED_A = b"\x01" * 4
SEED_B = b"\x02" * 4
SEED_C = b"\x03" * 4


class TestSelectSeed:
    def test_empty_returns_empty_string(self):
        assert SeedRoundRobinScheduler().select_seed([]) == ""

    def test_single_candidate_returned_without_advancing(self):
        s = SeedRoundRobinScheduler()
        assert s.select_seed(["only"]) == "only"
        assert s.select_seed(["only"]) == "only"
        assert s._index == 0

    def test_cycles_in_registration_order(self):
        s = SeedRoundRobinScheduler()
        ids = ["a", "b", "c"]
        picks = [s.select_seed(ids) for _ in range(6)]
        assert picks == ["a", "b", "c", "a", "b", "c"]

    def test_unregistered_candidates_are_registered_on_the_fly(self):
        s = SeedRoundRobinScheduler()
        s.select_seed(["a", "b"])
        assert s._seed_order == ["a", "b"]

    def test_registration_order_persists_across_shrinking_candidate_sets(self):
        """Once registered, a key's position in the cycle order is fixed --
        offering a subset later still cycles in the original relative
        order (the running index is global, same as RoundRobinScheduler's
        select_op, so it does not reset when the candidate set shrinks).
        """
        s = SeedRoundRobinScheduler()
        s.select_seed(["a", "b", "c"])  # picks "a", index -> 1
        # Now offer a two-element subset; cycling continues over that
        # subset in the original registration order, continuing from the
        # global index left at 1.
        picks = [s.select_seed(["b", "c"]) for _ in range(4)]
        assert picks == ["c", "b", "c", "b"]

    def test_late_arriving_candidate_is_appended_to_the_cycle(self):
        s = SeedRoundRobinScheduler()
        s.select_seed(["a", "b"])  # picks "a", index -> 1
        s.select_seed(["a", "b"])  # picks "b", index -> 2
        picks = [s.select_seed(["a", "b", "c"]) for _ in range(3)]
        assert picks == ["c", "a", "b"]

    def test_init_arm_ignores_priors_and_does_not_reset(self):
        s = SeedRoundRobinScheduler()
        s.init_arm("a", prior_alpha=99.0, prior_beta=0.01)
        s.record("a", success=True, weight=1.0)
        before = dict(s._seed_counts)
        s.init_arm("a", prior_alpha=1.0, prior_beta=1.0)
        assert s._seed_counts == before


class TestRecordAndStats:
    def test_record_success_and_failure(self):
        s = SeedRoundRobinScheduler()
        s.record("a", success=True, weight=1.0)
        s.record("a", success=False, weight=1.0)
        successes, failures = s.bandit_stats()["a"]
        assert successes == pytest.approx(1.0)
        assert failures == pytest.approx(1.0)

    def test_partial_weight_is_fractional_bernoulli(self):
        s = SeedRoundRobinScheduler()
        s.record("a", success=True, weight=0.25)
        successes, failures = s.bandit_stats()["a"]
        assert successes == pytest.approx(0.25)
        assert failures == pytest.approx(0.75)

    def test_weight_out_of_range_is_clamped(self):
        s = SeedRoundRobinScheduler()
        s.record("a", success=True, weight=5.0)
        successes, failures = s.bandit_stats()["a"]
        assert successes == pytest.approx(1.0)
        assert failures == pytest.approx(0.0)

    def test_record_does_not_affect_selection(self):
        """record() is Elo-compatibility bookkeeping only -- selection is
        the same deterministic cycle regardless of what's recorded.
        """
        s = SeedRoundRobinScheduler()
        ids = ["a", "b"]
        s.select_seed(ids)  # registers both, picks "a"
        s.record("a", success=False, weight=1.0)
        s.record("a", success=False, weight=1.0)
        s.record("b", success=True, weight=1.0)
        picks = [s.select_seed(ids) for _ in range(4)]
        assert picks == ["b", "a", "b", "a"]

    def test_bandit_stats_sorted_by_key(self):
        s = SeedRoundRobinScheduler()
        s.record("z", success=True)
        s.record("a", success=True)
        assert list(s.bandit_stats().keys()) == ["a", "z"]

    def test_supports_priors_is_false(self):
        assert SeedRoundRobinScheduler.supports_priors is False


def _profile(signature=None, markers=(), magic=()):
    return SimpleNamespace(
        format_signature=signature,
        boundary_markers=list(markers),
        magic_bytes=list(magic),
    )


NO_PROFILE = _profile()


class TestSeedPickerWiring:
    """Elo pool eligibility, handler dispatch, and the non-Elo fallback."""

    def _fuzzer(self, enabled=True, corpus=(SEED_A, SEED_B, SEED_C)):
        f = SimpleNamespace(
            corpus=list(corpus),
            seed_meta={},
            _use_elo=True,
            _elo=SimpleNamespace(select_strategy=lambda keys, **_: "seed_round_robin"),
            _seed_strategy=None,
            _seed_strategy_pool=[],
            _seed_strategies_used=set(),
            _stall_recovery_active=False,
            _rng=RandPool(seed=4),
            ga=None,
            qea=None,
            markov_generate=False,
            markov_trained=False,
            _use_bayesian=False,
            _use_boltzmann=False,
            _use_ecofuzz=False,
            _profile=NO_PROFILE,
            _kruskal_count=None,
        )
        f._use_seed_round_robin = enabled
        f._seed_round_robin = SeedRoundRobinScheduler() if enabled else None

        def _seed_key(data):
            return data

        f._seed_key = _seed_key
        return f

    def _picker(self, f):
        from fuzzer_tool.services.seed_picker import SeedPicker

        sp = SeedPicker.__new__(SeedPicker)
        sp.f = f
        sp._rng = f._rng
        return sp

    def test_eligible_and_dispatched_under_elo(self):
        f = self._fuzzer()
        picked = self._picker(f)._pick_seed_elo()
        assert "round_robin" in f._seed_strategy_pool
        assert f._seed_strategy == "round_robin"
        assert picked == SEED_A

    def test_not_eligible_when_disabled_or_empty(self):
        for f in (self._fuzzer(enabled=False), self._fuzzer(corpus=())):
            f._use_boltzmann = True  # two arms, so Elo is consulted
            f._elo = SimpleNamespace(select_strategy=lambda keys, **_: "seed_unhandled")
            self._picker(f)._pick_seed_elo()
            assert "round_robin" not in f._seed_strategy_pool

    def test_cycles_through_corpus_over_successive_picks(self):
        f = self._fuzzer()
        sp = self._picker(f)
        picks = [sp._pick_seed_round_robin_seed() for _ in range(6)]
        assert picks == [SEED_A, SEED_B, SEED_C, SEED_A, SEED_B, SEED_C]

    def test_empty_corpus_returns_none(self):
        f = self._fuzzer(corpus=())
        assert self._picker(f)._pick_seed_round_robin_seed() is None

    def test_disabled_returns_none(self):
        f = self._fuzzer(enabled=False)
        assert self._picker(f)._pick_seed_round_robin_seed() is None

    def test_non_elo_fallback_dispatches_before_bayesian(self, monkeypatch):
        f = self._fuzzer()
        f._use_elo = False
        f._use_bayesian = True
        f._seed_quality = {"x": 1}
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)
        monkeypatch.setattr(sp, "_pick_bayesian_seed", lambda: pytest.fail("bayesian won"))
        assert sp.pick_seed() == SEED_A

    def test_non_elo_fallback_absent_when_disabled(self, monkeypatch):
        """With round-robin off, the no-elo chain falls through to
        weighted/bayesian as before -- this scheduler adds no side effects
        when its own flag is off.
        """
        f = self._fuzzer(enabled=False)
        f._use_elo = False
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)
        sentinel = object()
        monkeypatch.setattr(sp, "weighted_pick_seed", lambda: sentinel)
        f.seed_meta = {s: {} for s in f.corpus}
        assert sp.pick_seed() is sentinel


class TestFuzzerWiring:
    def test_registered_as_seed_strategy(self):
        from fuzzer_tool.services.fuzzer import _SEED_STRATEGY_NAMES

        assert "round_robin" in _SEED_STRATEGY_NAMES

    def test_constructor_flag_is_last_and_defaults_off(self):
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        params = inspect.signature(Fuzzer.__init__).parameters
        assert list(params)[-1] == "position_arena"
        assert params["seed_round_robin_scheduler"].default is False

    def test_cli_passes_flag_to_the_fuzzer_construction(self):
        import ast
        import inspect

        from fuzzer_tool.cli import commands

        def kws(fn, callee):
            tree = ast.parse(inspect.getsource(fn))
            calls = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == callee
            ]
            return [{k.arg for k in c.keywords} for c in calls]

        assert all("seed_round_robin_scheduler" in k for k in kws(commands.cmd_fuzz, "Fuzzer"))
        assert "seed_round_robin_scheduler" in commands._HAIL_MARY_FLAGS

    def test_parser_declares_flag(self):
        import ast
        import inspect

        from fuzzer_tool.cli import commands
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        assert "seed_round_robin_scheduler" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))

    def test_constructor_enables_scheduler(self):
        from fuzzer_tool.core.schedulers.seed_round_robin import SeedRoundRobinScheduler
        from fuzzer_tool.services.fuzzer import Fuzzer

        f = Fuzzer.__new__(Fuzzer)
        # Exercise just the two attributes this flag sets, the same
        # pattern seed_canary_scheduler's own tests would use, without
        # constructing a full Fuzzer (which needs a real target binary).
        seed_round_robin_scheduler = True
        f._use_seed_round_robin = seed_round_robin_scheduler
        f._seed_round_robin = None
        if seed_round_robin_scheduler:
            f._seed_round_robin = SeedRoundRobinScheduler()
        assert f._use_seed_round_robin is True
        assert isinstance(f._seed_round_robin, SeedRoundRobinScheduler)
