"""Position-selection schedulers and their Elo arena.

Covers core/schedulers/pos_base.py, pos_burn_front.py and
services/position_arena.py, plus the pos_ keyspace helpers in
core/analyzers/analyzer_elo.py.
"""

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.analyzers.analyzer_elo import (
    Arena,
    BayesianEloTracker,
    strategy_arena,
    strategy_display_name,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import (
    CallablePosition,
    Outcome,
    PositionScheduler,
    UniformPosition,
)
from fuzzer_tool.core.schedulers.pos_burn_front import (
    COOL_EVERY,
    FUEL_FLOOR,
    KERNEL_RADIUS,
    MAX_BINS,
    MAX_HOT_BINS,
    MAX_SEEDS,
    SPARK_RATE,
    BurnFrontPositionScheduler,
)
from fuzzer_tool.services.operators import OperatorEngine
from fuzzer_tool.services.position_arena import POSITION_STRATEGY_NAMES, PositionArena

SEED = bytes(1000)
NO_SPARK = 0.99  # random() draw above SPARK_RATE


class ScriptedRng:
    """Deterministic stand-in: scripted random(), argmax weighted_choice."""

    def __init__(self, randoms=()):
        self._randoms = list(randoms)

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_SPARK

    def randint(self, a, b):
        return a

    def weighted_choice(self, seq, weights):
        return seq[max(range(len(seq)), key=weights.__getitem__)]


def _bf(rng=None):
    return BurnFrontPositionScheduler(rng or ScriptedRng())


class TestProtocol:
    def test_burn_front_and_uniform_satisfy_the_protocol(self):
        assert isinstance(_bf(), PositionScheduler)
        assert isinstance(UniformPosition(RandPool(seed=1)), PositionScheduler)

    def test_callable_adapter_forwards_and_ignores_record(self):
        arm = CallablePosition("x", lambda d, n: 7)
        assert arm.propose(SEED, 10) == 7
        arm.record(SEED, [1], Outcome.GAIN)  # no-op, must not raise

    def test_uniform_stays_in_bounds(self):
        u = UniformPosition(RandPool(seed=3))
        assert all(0 <= u.propose(SEED, 5) < 5 for _ in range(200))

    def test_uniform_empty_buffer(self):
        assert UniformPosition(RandPool(seed=3)).propose(b"", 0) == 0


class TestBurnFront:
    def test_cold_seed_declines(self):
        assert _bf().propose(SEED, len(SEED)) is None

    def test_gain_lights_the_offset(self):
        s = _bf()
        s.record(SEED, [100], Outcome.GAIN)
        assert s.propose(SEED, len(SEED)) == 100

    def test_miss_deposits_nothing(self):
        # FALSIFICATION: a MISS that heated a bin would make every
        # execution a gain.
        s = _bf()
        s.record(SEED, [100], Outcome.MISS)
        assert s.hot_bins(SEED) == {}
        assert s.propose(SEED, len(SEED)) is None

    def test_heat_spreads_to_neighbours_and_falls_with_distance(self):
        s = _bf()
        s.record(SEED, [100], Outcome.GAIN)
        heat = s.hot_bins(SEED)
        assert set(heat) == set(range(100 - KERNEL_RADIUS, 100 + KERNEL_RADIUS + 1))
        assert heat[100] > heat[101] > heat[102]
        assert heat[99] == pytest.approx(heat[101])

    def test_weight_is_split_across_offsets(self):
        one, two = _bf(), _bf()
        one.record(SEED, [100], Outcome.GAIN, weight=1.0)
        two.record(SEED, [100, 500], Outcome.GAIN, weight=1.0)
        assert two.hot_bins(SEED)[100] == pytest.approx(one.hot_bins(SEED)[100] / 2)

    def test_fuel_burns_and_front_moves_to_the_neighbour(self):
        # FALSIFICATION: without fuel burn the argmax bin never changes and
        # the front never advances.
        s = _bf()
        s.record(SEED, [100], Outcome.GAIN)
        picks = [s.propose(SEED, len(SEED)) for _ in range(40)]
        assert picks[0] == 100
        assert set(picks) != {100}
        assert any(abs(p - 100) <= KERNEL_RADIUS for p in picks)

    def test_gain_refuels_a_burnt_bin(self):
        s = _bf()
        s.record(SEED, [100], Outcome.GAIN)
        for _ in range(40):
            s.propose(SEED, len(SEED))
        assert s.fuel_of(SEED, 100) < 1.0
        s.record(SEED, [100], Outcome.GAIN)
        assert s.fuel_of(SEED, 100) == 1.0

    def test_fuel_never_reaches_zero(self):
        # ADVERSARIAL: a zero-weight front would make weighted_choice raise.
        s = BurnFrontPositionScheduler(RandPool(seed=5))
        s.record(SEED, [100], Outcome.GAIN)
        for _ in range(5000):
            s.propose(SEED, len(SEED))
        assert s.fuel_of(SEED, 100) >= FUEL_FLOOR

    def test_spark_escapes_the_hot_region(self):
        s = BurnFrontPositionScheduler(ScriptedRng([SPARK_RATE / 2]))
        s.record(SEED, [500], Outcome.GAIN)
        assert s.propose(SEED, len(SEED)) == 0  # ScriptedRng.randint -> lo

    def test_cooling_eventually_extinguishes_the_front(self):
        s = _bf()
        s.record(SEED, [100], Outcome.GAIN)
        for _ in range(COOL_EVERY * 400):
            s.propose(SEED, len(SEED))
        assert s.hot_bins(SEED) == {}
        assert s.propose(SEED, len(SEED)) is None

    def test_position_is_clamped_to_a_shrunken_buffer(self):
        # ADVERSARIAL: earlier ops in the round may already have shrunk the
        # buffer below the hot offset.
        s = _bf()
        s.record(SEED, [900], Outcome.GAIN)
        for _ in range(50):
            assert 0 <= s.propose(SEED, 10) < 10

    def test_empty_buffer_declines(self):
        s = _bf()
        s.record(SEED, [1], Outcome.GAIN)
        assert s.propose(SEED, 0) is None

    def test_offsets_past_the_seed_end_are_accepted(self):
        # ADVERSARIAL: the buffer may have grown past the parent seed.
        s = _bf()
        s.record(SEED, [len(SEED) + 500], Outcome.GAIN)
        assert 0 <= s.propose(SEED, len(SEED) + 600) < len(SEED) + 600

    def test_conduction_stops_at_the_seed_end(self):
        # REGRESSION: the kernel's right tail used to heat bins past the
        # seed, and propose() clamps them all onto the last byte.
        s = _bf()
        seed = bytes(32)
        s.record(seed, [31], Outcome.GAIN)
        heat = s.hot_bins(seed)
        assert max(heat) == 31
        assert set(heat) == set(range(31 - KERNEL_RADIUS, 32))

    def test_gain_past_the_seed_end_lights_only_its_own_bin_beyond(self):
        s = _bf()
        s.record(SEED, [len(SEED) + 500], Outcome.GAIN)
        beyond = [b for b in s.hot_bins(SEED) if b >= len(SEED)]
        assert beyond == [len(SEED) + 500]

    def test_negative_offsets_are_ignored(self):
        s = _bf()
        s.record(SEED, [-5], Outcome.GAIN)
        assert s.hot_bins(SEED) == {}

    def test_large_seed_uses_at_most_max_bins(self):
        big = bytes(MAX_BINS * 64)
        s = _bf()
        s.record(big, [len(big) - 1], Outcome.GAIN)
        assert max(s.hot_bins(big)) <= MAX_BINS + KERNEL_RADIUS
        assert 0 <= s.propose(big, len(big)) < len(big)

    def test_seed_table_is_lru_bounded(self):
        s = _bf()
        for i in range(MAX_SEEDS + 50):
            s.record(i.to_bytes(4, "big") * 4, [1], Outcome.GAIN)
        assert s.seed_count() == MAX_SEEDS
        assert s.hot_bins((0).to_bytes(4, "big") * 4) == {}  # oldest evicted

    def test_hot_bins_are_capped_and_keep_the_hottest(self):
        s = _bf()
        wide = bytes(MAX_HOT_BINS * 100 * 2)
        s.record(wide, [0], Outcome.GAIN, weight=10.0)
        for i in range(1, MAX_HOT_BINS * 2):
            s.record(wide, [i * 50], Outcome.GAIN, weight=0.1)
        heat = s.hot_bins(wide)
        assert len(heat) <= MAX_HOT_BINS
        assert 0 in heat


class _Fuzzer:
    """Mock exposing the position sources the arena consults."""

    def __init__(self, **enabled):
        self._rng = RandPool(seed=11)
        self._use_elo = True
        self._elo = BayesianEloTracker(min_matches=2, rng=RandPool(seed=12))
        self.seed_meta = {}
        self._use_sensitivity = enabled.get("sensitivity", False)
        self._sensitivity = SimpleNamespace(get_weighted_position=lambda d, n: 11)
        self._use_transfer_entropy = enabled.get("te", False)
        self._te = object()
        self._use_mi = enabled.get("mi", False)
        self._mi = SimpleNamespace(weighted_position=lambda n: 33)
        self._crash_mi = None
        self._use_region_profile = enabled.get("region", False)
        self.phase_calls = []

    def _get_te_weighted_position(self, _n):
        return 22

    def _get_phase_weighted_position(self, n, stride):
        self.phase_calls.append((n, stride))
        return None if stride is None else 44


def _arena(f=None, burn_front=None, region=lambda d, n: 55):
    f = f or _Fuzzer(sensitivity=True, te=True)
    return f, PositionArena(f, region_fn=region, burn_front=burn_front)


def _force(f, name):
    f._elo.select_strategy = lambda keys: f"pos_{name}"


class TestKeyspace:
    def test_pos_keys_form_their_own_arena(self):
        assert strategy_arena("pos_mi") is Arena.POSITION
        assert strategy_arena("seed_ga") is Arena.SEED
        assert strategy_arena("bandit") is Arena.OPERATOR
        assert strategy_arena("canary") is Arena.OPERATOR

    def test_display_name_is_not_dressed_as_an_operator(self):
        assert strategy_display_name("pos_mi") == "pos_mi"

    def test_canary_floor_stays_inside_its_arena(self):
        # ADVERSARIAL: pos_ keys used to fall into the operator arena's
        # "not seed_" filter and would be flagged against op canary.
        elo = BayesianEloTracker(min_matches=1)
        for k, mu in (("canary", 1500), ("pos_x", 1400), ("bandit", 1600)):
            elo._strategy_mu[k] = mu
            elo._strategy_match_count[k] = 5
        flagged = [s for s, *_ in elo.strategies_below_canary("canary")]
        assert "pos_x" not in flagged

    def test_position_floor_flags_a_proposer_below_uniform(self):
        elo = BayesianEloTracker(min_matches=1)
        for k, mu in (("pos_uniform", 1500), ("pos_mi", 1400), ("pos_te", 1600)):
            elo._strategy_mu[k] = mu
            elo._strategy_match_count[k] = 5
        elo._strategy_mu["seed_ga"] = 1000
        elo._strategy_match_count["seed_ga"] = 5
        flagged = [s for s, *_ in elo.strategies_below_canary("pos_uniform")]
        assert flagged == ["pos_mi"]


class TestPool:
    def test_uniform_is_first_and_only_enabled_arms_follow(self):
        _, arena = _arena()
        assert arena.pool()[0] == "uniform"
        assert set(arena.pool()) == {"uniform", "sensitivity", "te", "phase"}

    def test_disabled_arm_is_absent(self):
        # FALSIFICATION: a static pool would list arms whose tracker is off.
        _, arena = _arena(_Fuzzer())
        assert arena.pool() == ["uniform"]

    def test_crash_mi_waits_for_min_observations(self):
        f = _Fuzzer()
        f._crash_mi = SimpleNamespace(
            total_execs=1, min_observations=5, weighted_position=lambda n: 66
        )
        _, arena = _arena(f)
        assert "crash_mi" not in arena.pool()
        f._crash_mi.total_execs = 5
        assert "crash_mi" in arena.pool()

    def test_burn_front_joins_when_supplied(self):
        _, arena = _arena(burn_front=_bf())
        assert "burn_front" in arena.pool()

    def test_every_pool_name_is_registered(self):
        f = _Fuzzer(sensitivity=True, te=True, mi=True, region=True)
        f._crash_mi = SimpleNamespace(
            total_execs=9, min_observations=1, weighted_position=lambda n: 1
        )
        _, arena = _arena(f, burn_front=_bf())
        assert set(arena.pool()) == set(POSITION_STRATEGY_NAMES)


class TestSelect:
    def test_elo_choice_selects_the_arm(self):
        f, arena = _arena()
        _force(f, "sensitivity")
        assert arena.select(SEED, len(SEED)) == 11

    def test_elo_is_offered_prefixed_keys(self):
        f, arena = _arena()
        seen = []
        f._elo.select_strategy = lambda keys: seen.append(list(keys)) or keys[0]
        arena.select(SEED, len(SEED))
        assert seen[0][0] == "pos_uniform"
        assert all(k.startswith("pos_") for k in seen[0])

    def test_declining_arm_gets_a_uniform_offset_but_is_charged_itself(self):
        f, arena = _arena(_Fuzzer(te=True))  # phase declines: no stride
        _force(f, "phase")
        pos = arena.select(SEED, len(SEED))
        assert 0 <= pos < len(SEED)
        assert arena.used() == ["phase"]

    def test_a_decliner_cannot_outrate_uniform(self):
        # REGRESSION: a decline used to be charged to uniform, so an arm that
        # never proposed never served, only ever played as an opponent, and
        # won every miss round. With 5% gains it rated ~560 above uniform and
        # the uniform floor could never flag it. Charged to itself it *is*
        # uniform, so the two must end up level. Real Elo, no forcing.
        import random

        f = _Fuzzer(sensitivity=True)
        f._elo = BayesianEloTracker(
            initial_mu=1500, initial_sigma=350, beta=200, tau=5.0,
            min_matches=10, rng=RandPool(seed=12),
        )  # fmt: skip
        f._sensitivity = SimpleNamespace(get_weighted_position=lambda d, n: None)
        _, arena = _arena(f)
        draw = random.Random(0)
        for _ in range(4000):
            arena.select(SEED, len(SEED))
            gain = draw.random() < 0.05
            outcome = Outcome.GAIN if gain else Outcome.MISS
            arena.settle(SEED, [], outcome, weight=1.0, score=1.0 if gain else 0.0)
        mu = f._elo._strategy_mu
        assert abs(mu["pos_sensitivity"] - mu["pos_uniform"]) < 50

    def test_phase_receives_the_parent_stride(self):
        f, arena = _arena(_Fuzzer(te=True))
        f.seed_meta[SEED] = {"record_stride": 8}
        _force(f, "phase")
        assert arena.select(SEED, len(SEED)) == 44
        assert f.phase_calls == [(len(SEED), 8)]

    def test_out_of_range_proposals_are_clamped(self):
        # ADVERSARIAL: a stale tracker can name an offset past the buffer.
        f, arena = _arena(_Fuzzer(mi=True))
        _force(f, "mi")
        assert arena.select(SEED, 5) == 4

    def test_single_member_pool_skips_elo(self):
        f, arena = _arena(_Fuzzer())

        def boom(keys):
            raise AssertionError("no arbitration for a one-arm pool")

        f._elo.select_strategy = boom
        assert 0 <= arena.select(SEED, 10) < 10


class TestSettle:
    def _played(self, picked="sensitivity", **kw):
        f, arena = _arena(**kw)
        _force(f, picked)
        arena.select(SEED, len(SEED))
        return f, arena

    def test_picked_arm_plays_every_unpicked_member(self):
        f, arena = self._played()
        arena.settle(SEED, [], Outcome.MISS, weight=0.0, score=0.0)
        counts = f._elo._strategy_match_count
        assert counts["pos_sensitivity"] == 3  # vs uniform, te, phase
        assert counts["pos_uniform"] == counts["pos_te"] == counts["pos_phase"] == 1

    def test_score_orients_the_match(self):
        f, arena = self._played()
        arena.settle(SEED, [], Outcome.GAIN, weight=1.0, score=1.0)
        mu = f._elo._strategy_mu
        assert mu["pos_sensitivity"] > mu["pos_uniform"]

    def test_round_state_resets_after_settle(self):
        f, arena = self._played()
        arena.settle(SEED, [], Outcome.MISS, weight=0.0, score=0.0)
        n = f._elo._strategy_match_count["pos_sensitivity"]
        arena.settle(SEED, [], Outcome.MISS, weight=0.0, score=0.0)
        assert f._elo._strategy_match_count["pos_sensitivity"] == n

    def test_all_pool_members_used_means_no_opponents(self):
        f, arena = _arena(_Fuzzer())
        arena.select(SEED, 10)
        arena.settle(SEED, [], Outcome.MISS, weight=0.0, score=0.0)
        assert f._elo._strategy_match_count == {}

    def test_begin_round_discards_a_rerolled_mutant(self):
        # REGRESSION: _dedup_mutate re-rolls a seen mutant with a fresh
        # mutate(); the arm that served the discarded one must not share the
        # executed round's outcome.
        f, arena = self._played(picked="sensitivity")
        arena.begin_round()
        _force(f, "te")
        arena.select(SEED, len(SEED))
        arena.settle(SEED, [], Outcome.GAIN, weight=1.0, score=1.0)
        counts = f._elo._strategy_match_count
        assert counts["pos_te"] == 3
        assert counts["pos_sensitivity"] == 1  # as te's opponent only

    def test_mutate_starts_a_position_round(self, monkeypatch):
        # Wiring: OperatorEngine.mutate must reset the arena before anything
        # else, so every dedup re-roll starts clean. Stop mutate right after
        # the reset by failing the context refresh that follows it.
        from fuzzer_tool.services import operators as ops_mod

        class _Stop(Exception):
            pass

        def stop(_f):
            raise _Stop

        monkeypatch.setattr(ops_mod.MutationContext, "from_fuzzer", staticmethod(stop))
        f, arena = self._played()
        f._position_arena = arena
        with pytest.raises(_Stop):
            OperatorEngine(f).mutate(SEED)
        assert arena.used() == []

    def test_burn_front_is_credited_off_policy(self):
        # The picker was sensitivity, not burn_front; the front still learns.
        bf = _bf()
        f, arena = self._played(burn_front=bf)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert bf.hot_bins(SEED)[100] > 0

    def test_elo_off_still_credits_burn_front(self):
        bf = _bf()
        f, arena = _arena(burn_front=bf)
        f._use_elo = False
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert bf.hot_bins(SEED)
        assert f._elo._strategy_match_count == {}


class TestSelectPositionWiring:
    def _engine(self, f):
        return OperatorEngine(f)

    def test_arena_replaces_the_uniform_candidate_pick(self):
        f, arena = _arena()
        f._position_arena = arena
        _force(f, "te")
        assert self._engine(f).select_position(bytearray(SEED), SEED) == 22
        assert arena.used() == ["te"]

    def test_arena_ignored_when_elo_is_off(self):
        # Legacy path: uniform choice among candidates, no arena state.
        f, arena = _arena()
        f._position_arena = arena
        f._use_elo = False
        pos = self._engine(f).select_position(bytearray(SEED), SEED)
        assert pos in {11, 22}
        assert arena.used() == []

    def test_legacy_path_unchanged_without_an_arena(self):
        f = _Fuzzer(te=True)
        drawn = {self._engine(f).select_position(bytearray(SEED), SEED) for _ in range(20)}
        assert drawn == {22}

    def test_burn_front_is_a_legacy_candidate(self):
        # FALSIFICATION: if the proposer were computed but dropped from the
        # candidate list, its hot offset could never be returned.
        f = _Fuzzer(te=True)
        f._burn_front = _bf()
        f._burn_front.record(SEED, [100], Outcome.GAIN)
        engine = self._engine(f)
        drawn = {engine.select_position(bytearray(SEED), SEED) for _ in range(200)}
        assert 100 in drawn
        assert 22 in drawn

    def test_cold_burn_front_adds_no_candidate(self):
        f = _Fuzzer(te=True)
        f._burn_front = _bf()
        engine = self._engine(f)
        assert {engine.select_position(bytearray(SEED), SEED) for _ in range(50)} == {22}


class TestRegistration:
    def test_elo_activation_preregisters_pos_keys(self):
        from fuzzer_tool.core.analyzer_registry import _activate_elo

        f = SimpleNamespace(_state_store=SimpleNamespace(get=lambda k: None))
        _activate_elo(f)
        for name in POSITION_STRATEGY_NAMES:
            assert f"pos_{name}" in f._elo._strategy_mu


class TestFuzzerWiring:
    def test_flags_are_the_last_constructor_params(self):
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        params = inspect.signature(Fuzzer.__init__).parameters
        assert list(params)[-2:] == ["burn_front", "position_arena"]
        assert params["burn_front"].default is False
        assert params["position_arena"].default is False

    def test_cli_passes_flags_and_excludes_them_from_hail_mary(self):
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
        assert calls
        for c in calls:
            kw = {k.arg for k in c.keywords}
            assert {"burn_front", "position_arena"} <= kw
        dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
        assert {"burn_front", "position_arena"} <= dests
        assert not {"burn_front", "position_arena"} & set(commands._HAIL_MARY_FLAGS)

    def test_settle_skips_delocalised_ops(self):
        from fuzzer_tool.services.fuzzer import Fuzzer

        bf = _bf()
        f = SimpleNamespace(
            _position_arena=None,
            _burn_front=bf,
            _last_parent_seed=SEED,
            _last_ops_with_sites=[("byte_shuffle", 300), ("bit_flip", 100)],
        )
        Fuzzer._settle_positions(f, outcome=Outcome.GAIN, weight=1.0)
        assert 100 in bf.hot_bins(SEED)
        assert 300 not in bf.hot_bins(SEED)
