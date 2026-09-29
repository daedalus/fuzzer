"""Seed-arena deficit round robin (core/schedulers/seed_drr.py)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_drr import FAVORED_WEIGHT, SeedDRRScheduler

SEED_A = b"\x01" * 4
SEED_B = b"\x02" * 4
SEED_C = b"\x03" * 4
NAN = float("nan")


def _unit(_key):
    return 1.0


class TestSelectSeed:
    def test_empty_returns_empty_string(self):
        assert SeedDRRScheduler().select_seed([]) == ""

    def test_single_candidate_returned_as_is(self):
        assert SeedDRRScheduler().select_seed(["only"]) == "only"

    def test_flat_cost_is_round_robin(self):
        """Falsification: unit cost and weight reduce DRR to plain cycling."""
        s = SeedDRRScheduler()
        ids = ["a", "b", "c"]

        assert [s.select_seed(ids) for _ in range(6)] == ["a", "b", "c"] * 2

    def test_costly_seed_picked_less_often_equal_time(self):
        """Cost 4 vs 1: the slow seed gets a quarter of the picks, same wall time."""
        cost = {"fast": 1.0, "slow": 4.0}
        s = SeedDRRScheduler(quantum=4.0)
        picks = [s.select_seed(["fast", "slow"], cost.get, _unit) for _ in range(500)]

        assert picks.count("fast") == 4 * picks.count("slow")

    def test_favored_weight_gets_more_time(self):
        w = {"fav": FAVORED_WEIGHT, "std": 1.0}
        s = SeedDRRScheduler()
        picks = [s.select_seed(["fav", "std"], _unit, w.get) for _ in range(300)]

        assert picks.count("fav") == pytest.approx(
            FAVORED_WEIGHT * picks.count("std"), abs=FAVORED_WEIGHT
        )

    def test_garbage_cost_and_weight_do_not_hang_or_raise(self):
        """Adversarial: NaN / negative / inf from a corrupted ledger."""
        s = SeedDRRScheduler()
        bad = {"a": NAN, "b": -1.0, "c": float("inf")}
        picks = [s.select_seed(list(bad), bad.get, bad.get) for _ in range(30)]

        assert set(picks) <= set(bad)
        assert len(set(picks)) == 3

    def test_unregistered_candidates_register_on_the_fly(self):
        s = SeedDRRScheduler()
        s.select_seed(["a", "b"])

        assert set(s.bandit_stats()) == {"a", "b"}

    def test_removed_seed_never_picked(self):
        s = SeedDRRScheduler()
        for _ in range(6):
            s.select_seed(["a", "b", "c"])

        assert "b" not in [s.select_seed(["a", "c"]) for _ in range(20)]


class TestBanditInterface:
    def test_record_and_stats(self):
        s = SeedDRRScheduler()
        s.record("a", success=True)
        s.record("a", success=False)

        assert s.bandit_stats() == {"a": (1.0, 1.0)}

    def test_weight_is_clamped(self):
        s = SeedDRRScheduler()
        s.record("a", success=True, weight=9.0)

        assert s.bandit_stats()["a"] == (1.0, 0.0)

    def test_record_does_not_affect_selection(self):
        a, b = SeedDRRScheduler(), SeedDRRScheduler()
        b.record("a", success=True)

        assert [a.select_seed(["a", "b"]) for _ in range(6)] == [
            b.select_seed(["a", "b"]) for _ in range(6)
        ]

    def test_init_arm_ignores_priors_and_never_resets(self):
        s = SeedDRRScheduler()
        s.record("a", success=True)
        s.init_arm("a", prior_alpha=9.0, prior_beta=9.0)

        assert s.bandit_stats()["a"] == (1.0, 0.0)

    def test_supports_priors_is_false(self):
        assert SeedDRRScheduler.supports_priors is False


def _profile():
    return SimpleNamespace(format_signature=None, boundary_markers=[], magic_bytes=[])


class TestSeedPickerWiring:
    """Elo eligibility, dispatch, cost/weight plumbing, no-elo chain."""

    def _fuzzer(self, enabled=True, corpus=(SEED_A, SEED_B, SEED_C), meta=None, favored=()):
        f = SimpleNamespace(
            corpus=list(corpus),
            seed_meta=meta if meta is not None else {},
            _use_elo=True,
            _elo=SimpleNamespace(select_strategy=lambda keys, **_: "seed_drr"),
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
            _profile=_profile(),
            _kruskal_count=None,
            _favored=set(favored),
            mean_exec_time=lambda: 0.001,
        )
        f._use_seed_drr = enabled
        f._seed_drr = SeedDRRScheduler() if enabled else None
        f._seed_key = lambda data: data
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

        assert "drr" in f._seed_strategy_pool
        assert f._seed_strategy == "drr"
        assert picked == SEED_A

    def test_not_eligible_when_disabled_or_empty(self):
        for f in (self._fuzzer(enabled=False), self._fuzzer(corpus=())):
            f._use_boltzmann = True
            f._elo = SimpleNamespace(select_strategy=lambda keys, **_: "seed_unhandled")
            self._picker(f)._pick_seed_elo()

            assert "drr" not in f._seed_strategy_pool

    def test_empty_or_disabled_returns_none(self):
        assert self._picker(self._fuzzer(corpus=()))._pick_seed_drr_seed() is None
        assert self._picker(self._fuzzer(enabled=False))._pick_seed_drr_seed() is None

    def test_slow_seed_is_picked_less_from_the_cost_ledger(self):
        """Falsification: cost comes from meta total_time/cost_samples, relative to the mean."""
        meta = {
            SEED_A: {"total_time": 0.001, "cost_samples": 1},
            SEED_B: {"total_time": 0.004, "cost_samples": 1},
        }
        f = self._fuzzer(corpus=(SEED_A, SEED_B), meta=meta)
        sp = self._picker(f)
        picks = [sp._pick_seed_drr_seed() for _ in range(500)]

        assert picks.count(SEED_A) > 3 * picks.count(SEED_B)

    def test_favored_seed_gets_more_picks(self):
        f = self._fuzzer(corpus=(SEED_A, SEED_B), favored={SEED_A})
        sp = self._picker(f)
        picks = [sp._pick_seed_drr_seed() for _ in range(300)]

        assert picks.count(SEED_A) > picks.count(SEED_B)

    def test_missing_meta_and_zero_mean_are_neutral(self):
        """Adversarial: unmeasured seeds and an empty ledger degrade to plain cycling."""
        f = self._fuzzer(meta={})
        f.mean_exec_time = lambda: 0.0
        sp = self._picker(f)

        assert [sp._pick_seed_drr_seed() for _ in range(6)] == [SEED_A, SEED_B, SEED_C] * 2

    def test_non_elo_fallback_dispatches_before_round_robin(self, monkeypatch):
        f = self._fuzzer()
        f._use_elo = False
        f._use_seed_round_robin = True
        f._seed_round_robin = SimpleNamespace(
            select_seed=lambda ids: pytest.fail("round_robin won")
        )
        sp = self._picker(f)
        monkeypatch.setattr(sp, "_update_temperature", lambda: None)

        assert sp.pick_seed() == SEED_A

    def test_non_elo_fallback_absent_when_disabled(self, monkeypatch):
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

        assert "drr" in _SEED_STRATEGY_NAMES

    def test_constructor_flag_defaults_off_and_is_last(self):
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        params = inspect.signature(Fuzzer.__init__).parameters

        assert list(params)[-1] == "seed_drr_scheduler"
        assert params["seed_drr_scheduler"].default is False

    def test_cli_passes_flag_and_parser_declares_it(self):
        import ast
        import inspect

        from fuzzer_tool.cli import commands
        from tests.test_regression_cli_fuzzer_kwargs import _fuzz_parser_dests

        tree = ast.parse(inspect.getsource(commands.cmd_fuzz))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Fuzzer"
        ]

        assert all("seed_drr_scheduler" in {k.arg for k in c.keywords} for c in calls)
        assert "seed_drr_scheduler" in _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
        assert "seed_drr_scheduler" in commands._HAIL_MARY_FLAGS
