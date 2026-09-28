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
from fuzzer_tool.core.schedulers.pos_boundary import PositionBoundaryScheduler
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
from fuzzer_tool.core.schedulers.pos_canary import PositionCanaryScheduler
from fuzzer_tool.core.schedulers.pos_cmplog import PositionCmplogScheduler
from fuzzer_tool.core.schedulers.pos_context import PositionContextScheduler
from fuzzer_tool.core.schedulers.pos_fibonacci import PositionFibonacciScheduler
from fuzzer_tool.core.schedulers.pos_fractal import PositionFractalScheduler
from fuzzer_tool.core.schedulers.pos_kl_ducb import PositionKLDUCBScheduler
from fuzzer_tool.core.schedulers.pos_levy import PositionLevyScheduler
from fuzzer_tool.core.schedulers.pos_lineage import PositionLineageScheduler
from fuzzer_tool.core.schedulers.pos_round_robin import PositionRoundRobinScheduler
from fuzzer_tool.services.operators import OperatorEngine
from fuzzer_tool.services.position_arena import (
    POSITION_STRATEGY_NAMES,
    PositionArena,
    parse_arena_arms,
)

SEED = bytes(1000)
NO_SPARK = 0.99  # random() draw above SPARK_RATE


class ScriptedRng:
    """Deterministic stand-in: scripted random(), argmax weighted_choice."""

    def __init__(self, randoms=()):
        self._randoms = list(randoms)

    def random(self):
        return self._randoms.pop(0) if self._randoms else NO_SPARK

    def randint(self, a, b):
        # The spark coin is randint(0, 9) < 1; route it through the scripted
        # random() so NO_SPARK / SPARK_RATE / 2 keep their meaning.
        if (a, b) == (0, 9):
            return int(self.random() * 10)
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


class TestPositionKLDUCB:
    def _s(self, rng=None):
        return PositionKLDUCBScheduler(rng or RandPool(seed=7))

    def test_satisfies_the_protocol(self):
        assert isinstance(self._s(), PositionScheduler)

    def test_name_is_kl_ducb(self):
        assert self._s().name == "kl_ducb"

    def test_empty_buffer_declines(self):
        assert self._s().propose(b"", 0) is None

    def test_cold_seed_proposes_a_bin_start(self):
        # No evidence yet -> every bin is "unpulled"; whichever the (seeded,
        # deterministic) rng opens first, the offset lands on a bin start.
        s = self._s()
        pos = s.propose(SEED, len(SEED))
        assert pos is not None
        assert 0 <= pos < len(SEED)

    def test_empty_offsets_is_a_no_op(self):
        s = self._s()
        s.record(SEED, [], Outcome.GAIN)
        assert s.bandit_stats(SEED) == {}

    def test_negative_offsets_are_ignored(self):
        s = self._s()
        s.record(SEED, [-5], Outcome.GAIN)
        assert s.bandit_stats(SEED) == {}

    def test_a_bin_with_more_gains_is_preferred_once_every_bin_is_open(self):
        # FALSIFICATION: select_op opens every never-pulled arm first, so
        # with any bin still unpulled a reinforced bin could never win by
        # chance alone. Open all 4 bins of a small seed first, then load
        # bin 0 with gains: it must be reachable afterwards.
        small = bytes(4)  # width 1 under MAX_BINS -> 4 bins
        s = self._s()
        for b in range(4):
            s.record(small, [b], Outcome.MISS)
        for _ in range(30):
            s.record(small, [0], Outcome.GAIN)
        picks = {s.propose(small, len(small)) for _ in range(20)}
        assert 0 in picks

    def test_repeated_misses_do_not_outrank_a_gaining_bin(self):
        small = bytes(4)
        s = self._s()
        for b in range(4):
            s.record(small, [b], Outcome.MISS)
        for _ in range(30):
            s.record(small, [0], Outcome.GAIN)
            s.record(small, [1], Outcome.MISS)
        picks = [s.propose(small, len(small)) for _ in range(50)]
        assert picks.count(0) >= picks.count(1)

    def test_weight_is_shared_across_offsets_as_separate_pulls(self):
        # Each offset in a round is its own bandit pull with a fractional
        # (weight-shared) reward, not a fractional pull -- pinning that
        # convention so a future change doesn't silently alter it.
        one, two = self._s(), self._s()
        one.record(SEED, [100], Outcome.GAIN, weight=1.0)
        two.record(SEED, [100, 500], Outcome.GAIN, weight=1.0)
        assert one.bandit_stats(SEED)["kl_ducb_pulls"] == 1
        assert one.bandit_stats(SEED)["kl_ducb_arms"] == 1
        assert two.bandit_stats(SEED)["kl_ducb_pulls"] == 2
        assert two.bandit_stats(SEED)["kl_ducb_arms"] == 2

    def test_offsets_past_the_seed_end_are_clamped_to_the_last_bin(self):
        # ADVERSARIAL: the buffer may have grown past the parent seed.
        s = self._s()
        s.record(SEED, [len(SEED) + 500], Outcome.GAIN)
        assert s.bandit_stats(SEED)["kl_ducb_arms"] == 1

    def test_position_is_clamped_to_a_shrunken_buffer(self):
        s = self._s()
        s.record(SEED, [900], Outcome.GAIN)
        for _ in range(50):
            pos = s.propose(SEED, 10)
            assert pos is None or 0 <= pos < 10

    def test_seed_table_is_lru_bounded(self):
        from fuzzer_tool.core.schedulers.pos_kl_ducb import MAX_SEEDS as KLD_MAX_SEEDS

        s = self._s()
        for i in range(KLD_MAX_SEEDS + 50):
            s.record(i.to_bytes(4, "big") * 4, [1], Outcome.GAIN)
        assert s.seed_count() == KLD_MAX_SEEDS
        assert s.bandit_stats((0).to_bytes(4, "big") * 4) == {}  # oldest evicted

    def test_bandit_stats_unknown_seed_is_empty(self):
        assert self._s().bandit_stats(SEED) == {}


class TestPositionCanary:
    def test_satisfies_the_protocol(self):
        assert isinstance(PositionCanaryScheduler(), PositionScheduler)

    def test_name_is_canary(self):
        assert PositionCanaryScheduler().name == "canary"

    def test_cold_seed_proposes_the_first_bin(self):
        # No data yet -> every bin ties at the Beta(1,1) prior; deterministic
        # tie-break picks the lowest-indexed (leftmost) bin.
        assert PositionCanaryScheduler().propose(SEED, len(SEED)) == 0

    def test_targets_the_bin_with_the_worst_posterior(self):
        s = PositionCanaryScheduler()
        # bin 0 (offset 0) gains every time; bin width for a 1000-byte seed
        # under MAX_BINS is 1, so offset 100 is its own bin.
        for _ in range(10):
            s.record(SEED, [0], Outcome.GAIN)
        s.record(SEED, [100], Outcome.MISS)
        assert s.propose(SEED, len(SEED)) == 100

    def test_miss_lowers_the_posterior_mean(self):
        s = PositionCanaryScheduler()
        s.record(SEED, [50], Outcome.MISS)
        a, b = s.bandit_stats(SEED)[50]
        assert a == 1.0
        assert b == 2.0

    def test_gain_raises_the_posterior_mean(self):
        s = PositionCanaryScheduler()
        s.record(SEED, [50], Outcome.GAIN, weight=1.0)
        a, b = s.bandit_stats(SEED)[50]
        assert a == 2.0
        assert b == 1.0

    def test_empty_offsets_is_a_no_op(self):
        s = PositionCanaryScheduler()
        s.record(SEED, [], Outcome.GAIN)
        assert s.bandit_stats(SEED) == {}

    def test_empty_buffer_declines(self):
        assert PositionCanaryScheduler().propose(b"", 0) is None

    def test_ties_go_to_the_lowest_bin_index(self):
        s = PositionCanaryScheduler()
        s.record(SEED, [0, 200], Outcome.MISS)  # both bins now tied at 1/3
        assert s.propose(SEED, len(SEED)) == 0

    def test_seed_table_is_lru_bounded(self):
        from fuzzer_tool.core.schedulers.pos_canary import MAX_SEEDS as CANARY_MAX_SEEDS

        s = PositionCanaryScheduler()
        for i in range(CANARY_MAX_SEEDS + 50):
            s.record(i.to_bytes(4, "big") * 4, [1], Outcome.GAIN)
        assert s.seed_count() == CANARY_MAX_SEEDS
        assert s.bandit_stats((0).to_bytes(4, "big") * 4) == {}  # oldest evicted


class TestPositionRoundRobin:
    def test_satisfies_the_protocol(self):
        assert isinstance(PositionRoundRobinScheduler(), PositionScheduler)

    def test_name_is_round_robin(self):
        assert PositionRoundRobinScheduler().name == "round_robin"

    def test_cycles_through_bins_in_order(self):
        s = PositionRoundRobinScheduler()
        picks = [s.propose(SEED, len(SEED)) for _ in range(3)]
        assert picks == [0, 1, 2]  # width 1 for a 1000-byte seed under MAX_BINS

    def test_wraps_around_after_the_last_bin(self):
        small = bytes(3)
        s = PositionRoundRobinScheduler()
        picks = [s.propose(small, len(small)) for _ in range(4)]
        assert picks == [0, 1, 2, 0]

    def test_record_does_not_perturb_the_cycle(self):
        s = PositionRoundRobinScheduler()
        s.propose(SEED, len(SEED))
        s.record(SEED, [999], Outcome.GAIN)
        assert s.propose(SEED, len(SEED)) == 1

    def test_empty_buffer_declines(self):
        assert PositionRoundRobinScheduler().propose(b"", 0) is None

    def test_seed_table_is_lru_bounded(self):
        from fuzzer_tool.core.schedulers.pos_round_robin import (
            MAX_SEEDS as RR_MAX_SEEDS,
        )

        s = PositionRoundRobinScheduler()
        for i in range(RR_MAX_SEEDS + 50):
            s.propose(i.to_bytes(4, "big") * 4, 16)
        assert s.seed_count() == RR_MAX_SEEDS


INV_PHI = (5**0.5 - 1) / 2  # 1/phi, derived independently of pos_fibonacci


def _max_gap(picks, num_bins):
    """Largest circular gap between distinct picked bins."""
    pts = sorted(set(picks))
    gaps = [b - a for a, b in zip(pts, pts[1:], strict=False)]
    return max(gaps + [num_bins - pts[-1] + pts[0]])


class TestPositionFibonacci:
    def test_satisfies_the_protocol(self):
        assert isinstance(PositionFibonacciScheduler(), PositionScheduler)

    def test_name_is_fibonacci(self):
        assert PositionFibonacciScheduler().name == "fibonacci"

    def test_follows_the_golden_ratio_sequence(self):
        # bin_n = floor(frac(n / phi) * num_bins); width 1 for a 1000-byte seed.
        s = PositionFibonacciScheduler()
        picks = [s.propose(SEED, len(SEED)) for _ in range(8)]
        assert picks == [int((n * INV_PHI) % 1.0 * len(SEED)) for n in range(8)]

    def test_any_prefix_spreads_evenly(self):
        # Falsification: the same bound round-robin must fail, else it is vacuous.
        data = bytes(MAX_BINS)
        for k in (8, 34, 89):
            fib, rr = PositionFibonacciScheduler(), PositionRoundRobinScheduler()
            bound = 3 * MAX_BINS // k
            assert _max_gap([fib.propose(data, len(data)) for _ in range(k)], MAX_BINS) <= bound
            assert _max_gap([rr.propose(data, len(data)) for _ in range(k)], MAX_BINS) > bound

    def test_long_seed_lands_on_bin_starts(self):
        data = bytes(3 * MAX_BINS)
        s = PositionFibonacciScheduler()
        assert all(s.propose(data, len(data)) % 3 == 0 for _ in range(64))

    def test_clamps_to_a_shrunk_buffer(self):
        s = PositionFibonacciScheduler()
        assert all(s.propose(SEED, 10) <= 9 for _ in range(64))

    def test_empty_buffer_declines(self):
        assert PositionFibonacciScheduler().propose(b"", 0) is None

    def test_regression_shrunk_buffer_does_not_pile_on_last_byte(self):
        # Bins sized from the parent seed, clamped to a shrunk buffer, sent
        # ~99% of picks to buf_len-1 (paired png run: 5/20 cells collapsed).
        s = PositionFibonacciScheduler()
        buf_len, k = 10, 100
        picks = [s.propose(SEED, buf_len) for _ in range(k)]
        assert set(picks) == set(range(buf_len))
        assert picks.count(buf_len - 1) <= 2 * k // buf_len

    def test_grown_buffer_reaches_past_the_seed(self):
        # Adversarial: bins sized from the seed never reach inserted tail bytes.
        s = PositionFibonacciScheduler()
        grown = 2 * len(SEED)
        assert max(s.propose(SEED, grown) for _ in range(16)) >= len(SEED)

    def test_empty_seed_with_live_buffer_proposes_zero(self):
        # Zero first; bins follow the live buffer, so an empty parent still
        # sweeps it rather than pinning every pick to 0.
        s = PositionFibonacciScheduler()
        picks = [s.propose(b"", 4) for _ in range(3)]
        assert picks == [int((n * INV_PHI) % 1.0 * 4) for n in range(3)]

    def test_record_does_not_perturb_the_sequence(self):
        a, b = PositionFibonacciScheduler(), PositionFibonacciScheduler()
        a.propose(SEED, len(SEED))
        b.propose(SEED, len(SEED))
        a.record(SEED, [999], Outcome.GAIN)
        assert a.propose(SEED, len(SEED)) == b.propose(SEED, len(SEED))

    def test_counter_wrap_stays_in_range(self):
        # Adversarial: the 64-bit counter wraps without leaving [0, len).
        s = PositionFibonacciScheduler()
        s._n = (1 << 64) - 2
        picks = [s.propose(SEED, len(SEED)) for _ in range(4)]
        assert all(0 <= p < len(SEED) for p in picks)
        assert s._n < 1 << 64

    def test_reaches_the_tail_under_seed_churn(self):
        # Adversarial: more distinct seeds than round-robin's LRU holds are
        # fuzzed between two visits of SEED. Round-robin (control) forgets
        # its cycle and restarts at bin 0; fibonacci keeps no per-seed state.
        from fuzzer_tool.core.schedulers.pos_round_robin import (
            MAX_SEEDS as RR_MAX_SEEDS,
        )

        fib, rr = PositionFibonacciScheduler(), PositionRoundRobinScheduler()
        fib_picks, rr_picks = [], []
        for _ in range(16):
            fib_picks.append(fib.propose(SEED, len(SEED)))
            rr_picks.append(rr.propose(SEED, len(SEED)))
            for i in range(RR_MAX_SEEDS + 1):
                other = i.to_bytes(4, "big") * 4
                fib.propose(other, len(other))
                rr.propose(other, len(other))

        assert set(rr_picks) == {0}
        assert max(fib_picks) >= 3 * len(SEED) // 4


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
        self._format_learner = (
            SimpleNamespace(clusters={"x": object()}, weighted_position=lambda d, n: 77)
            if enabled.get("field", False)
            else None
        )
        self._cmplog = object() if enabled.get("cmplog", False) else None
        self._use_lineage = enabled.get("lineage", False)
        self.phase_calls = []

    def _get_te_weighted_position(self, _n):
        return 22

    def _get_phase_weighted_position(self, n, stride):
        self.phase_calls.append((n, stride))
        return None if stride is None else 44


def _arena(
    f=None,
    burn_front=None,
    kl_ducb=None,
    canary=None,
    round_robin=None,
    fibonacci=None,
    fractal=None,
    cmplog=None,
    lineage=None,
    context=None,
    levy=None,
    boundary=None,
    region=lambda d, n: 55,
):
    f = f or _Fuzzer(sensitivity=True, te=True)
    return f, PositionArena(
        f,
        region_fn=region,
        burn_front=burn_front,
        kl_ducb=kl_ducb,
        canary=canary,
        round_robin=round_robin,
        fibonacci=fibonacci,
        fractal=fractal,
        cmplog=cmplog,
        lineage=lineage,
        context=context,
        levy=levy,
        boundary=boundary,
    )


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

    def test_context_joins_when_supplied(self):
        _, arena = _arena(context=PositionContextScheduler(RandPool(seed=1)))
        assert "context" in arena.pool()

    def test_context_absent_when_not_supplied(self):
        _, arena = _arena()
        assert "context" not in arena.pool()

    def test_levy_joins_when_supplied(self):
        _, arena = _arena(levy=PositionLevyScheduler(RandPool(seed=1)))
        assert "levy" in arena.pool()

    def test_boundary_joins_when_supplied(self):
        _, arena = _arena(boundary=PositionBoundaryScheduler(RandPool(seed=1)))
        assert "boundary" in arena.pool()

    def test_boundary_absent_when_not_supplied(self):
        _, arena = _arena()
        assert "boundary" not in arena.pool()

    def test_levy_absent_when_not_supplied(self):
        _, arena = _arena()
        assert "levy" not in arena.pool()

    def test_fractal_joins_when_supplied(self):
        _, arena = _arena(fractal=PositionFractalScheduler(RandPool(seed=1)))
        assert "fractal" in arena.pool()

    def _cmplog_arm(self, meta):
        f = _Fuzzer(cmplog=True)
        f.seed_meta = {SEED: meta}
        sched = PositionCmplogScheduler(
            RandPool(seed=1), meta_of=f.seed_meta.get, smap_of=lambda d: None
        )
        return f, sched

    def test_cmplog_joins_only_while_cmplog_is_live(self):
        f, sched = self._cmplog_arm({"redqueen_offsets": [40]})
        _, arena = _arena(f, cmplog=sched)
        assert "cmplog" in arena.pool()
        f._cmplog = None  # collector gone (start failed / cmplog off)
        assert "cmplog" not in arena.pool()

    def test_cmplog_absent_when_not_supplied(self):
        _, arena = _arena(_Fuzzer(cmplog=True))
        assert "cmplog" not in arena.pool()

    def test_cmplog_serves_a_redqueen_offset(self):
        f, _ = self._cmplog_arm({})
        f.seed_meta = {SEED: {"redqueen_offsets": [40]}}

        class _NoEscape:  # never the uniform escape, no jitter
            random = staticmethod(lambda: 0.99)
            randint = staticmethod(lambda a, b: 0 if a <= 0 <= b else a)
            weighted_choice = staticmethod(lambda seq, w: seq[0])

        sched = PositionCmplogScheduler(
            _NoEscape(), meta_of=f.seed_meta.get, smap_of=lambda d: None
        )
        _, arena = _arena(f, cmplog=sched)
        _force(f, "cmplog")
        assert arena.select(SEED, len(SEED)) == 40
        assert arena.used() == ["cmplog"]

    def test_cmplog_declines_to_uniform_but_stays_charged(self):
        f, sched = self._cmplog_arm({})  # no cmplog data on this seed
        _, arena = _arena(f, cmplog=sched)
        _force(f, "cmplog")
        assert 0 <= arena.select(SEED, len(SEED)) < len(SEED)
        assert arena.used() == ["cmplog"]

    def test_cmplog_is_not_an_off_policy_extra(self):
        # Passive tracker-style arm: settle() must not feed it.
        f, sched = self._cmplog_arm({"redqueen_offsets": [40]})
        seen = []
        sched.record = lambda *a, **k: seen.append(a)
        _, arena = _arena(f, cmplog=sched)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert seen == []

    def _lineage_arm(self, meta, rng=None):
        f = _Fuzzer(lineage=True)
        f.seed_meta = {SEED: meta}
        sched = PositionLineageScheduler(
            rng or RandPool(seed=1), meta_of=f.seed_meta.get, delocalised=()
        )
        return f, sched

    def test_lineage_joins_only_while_lineage_is_on(self):
        f, sched = self._lineage_arm({"parent_sites": [40]})
        _, arena = _arena(f, lineage=sched)
        assert "lineage" in arena.pool()
        f._use_lineage = False  # no --lineage: nothing records parent_sites
        assert "lineage" not in arena.pool()

    def test_lineage_absent_when_not_supplied(self):
        _, arena = _arena(_Fuzzer(lineage=True))
        assert "lineage" not in arena.pool()

    def test_lineage_serves_a_parent_site(self):
        class _NoEscapeNoJitter:  # never the escape, jitter magnitude 0
            random = staticmethod(lambda: 0.99)
            randint = staticmethod(lambda a, b: a)

        f, sched = self._lineage_arm({"parent_sites": [40]}, rng=_NoEscapeNoJitter())
        _, arena = _arena(f, lineage=sched)
        _force(f, "lineage")
        assert arena.select(SEED, len(SEED)) == 40
        assert arena.used() == ["lineage"]

    def test_lineage_declines_to_uniform_but_stays_charged(self):
        f, sched = self._lineage_arm({})  # initial-corpus seed: no parent_sites
        _, arena = _arena(f, lineage=sched)
        _force(f, "lineage")
        assert 0 <= arena.select(SEED, len(SEED)) < len(SEED)
        assert arena.used() == ["lineage"]

    def test_lineage_is_not_an_off_policy_extra(self):
        # Passive tracker-style arm: settle() must not feed it.
        f, sched = self._lineage_arm({"parent_sites": [40]})
        seen = []
        sched.record = lambda *a, **k: seen.append(a)
        _, arena = _arena(f, lineage=sched)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert seen == []

    def test_kl_ducb_joins_when_supplied(self):
        _, arena = _arena(kl_ducb=PositionKLDUCBScheduler(RandPool(seed=1)))
        assert "kl_ducb" in arena.pool()

    def test_canary_joins_when_supplied(self):
        _, arena = _arena(canary=PositionCanaryScheduler())
        assert "canary" in arena.pool()

    def test_round_robin_joins_when_supplied(self):
        _, arena = _arena(round_robin=PositionRoundRobinScheduler())
        assert "round_robin" in arena.pool()

    def test_fibonacci_joins_when_supplied(self):
        _, arena = _arena(fibonacci=PositionFibonacciScheduler())
        assert "fibonacci" in arena.pool()

    def test_field_waits_for_a_hypothesis_cluster(self):
        # FALSIFICATION: a static pool would list the arm the moment
        # --learn-format is on, before the learner has anything to propose.
        f = _Fuzzer()
        f._format_learner = SimpleNamespace(clusters={}, weighted_position=lambda d, n: 1)
        _, arena = _arena(f)
        assert "field" not in arena.pool()
        f._format_learner.clusters["x"] = object()
        assert "field" in arena.pool()

    def test_field_absent_without_format_learner(self):
        _, arena = _arena(_Fuzzer())
        assert "field" not in arena.pool()

    def test_every_pool_name_is_registered(self):
        f = _Fuzzer(
            sensitivity=True, te=True, mi=True, region=True, field=True, cmplog=True, lineage=True
        )
        f._crash_mi = SimpleNamespace(
            total_execs=9, min_observations=1, weighted_position=lambda n: 1
        )
        _, arena = _arena(
            f,
            burn_front=_bf(),
            kl_ducb=PositionKLDUCBScheduler(RandPool(seed=1)),
            canary=PositionCanaryScheduler(),
            round_robin=PositionRoundRobinScheduler(),
            fibonacci=PositionFibonacciScheduler(),
            fractal=PositionFractalScheduler(RandPool(seed=1)),
            cmplog=PositionCmplogScheduler(
                RandPool(seed=1), meta_of=lambda d: None, smap_of=lambda d: None
            ),
            lineage=PositionLineageScheduler(RandPool(seed=1), meta_of=lambda d: None),
            context=PositionContextScheduler(RandPool(seed=1)),
            levy=PositionLevyScheduler(RandPool(seed=1)),
            boundary=PositionBoundaryScheduler(RandPool(seed=1)),
        )
        assert set(arena.pool()) == set(POSITION_STRATEGY_NAMES)


class TestSelect:
    def test_elo_choice_selects_the_arm(self):
        f, arena = _arena()
        _force(f, "sensitivity")
        assert arena.select(SEED, len(SEED)) == 11

    def test_field_arm_selects_through_the_format_learner(self):
        f = _Fuzzer(field=True)
        _, arena = _arena(f)
        _force(f, "field")
        assert arena.select(SEED, len(SEED)) == 77

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

    def test_boundary_is_credited_off_policy(self):
        # record() is a documented no-op: settle must feed it without
        # raising or changing what it proposes for the same seed.
        seed = b"ab,cd" * 20
        bnd = PositionBoundaryScheduler(RandPool(seed=1))
        ref = PositionBoundaryScheduler(RandPool(seed=1))
        f, arena = self._played(boundary=bnd)
        picks = []
        bnd.record = lambda *a, **k: picks.append(a)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert len(picks) == 1
        assert bnd.propose(seed, len(seed)) == ref.propose(seed, len(seed))

    def test_levy_is_credited_off_policy(self):
        # The picker was sensitivity, not levy; its anchor still moves.
        levy = PositionLevyScheduler(RandPool(seed=1))
        f, arena = self._played(levy=levy)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert levy.anchor(SEED) == 100

    def test_levy_misses_are_credited_and_age_the_anchor(self):
        # FALSIFICATION: MISS rounds must reach the arm, or the anchor never
        # goes stale and levy orbits a dead site forever.
        from fuzzer_tool.core.schedulers.pos_levy import STALE

        levy = PositionLevyScheduler(RandPool(seed=1))
        f, arena = self._played(levy=levy)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        for _ in range(STALE):
            arena.settle(SEED, [100], Outcome.MISS, weight=1.0, score=0.0)
        assert levy.anchor(SEED) is None

    def test_context_is_credited_off_policy(self):
        # The picker was sensitivity, not context; its table still learns.
        ctx = PositionContextScheduler(RandPool(seed=1))
        f, arena = self._played(context=ctx)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert ctx.obs == 1
        assert ctx.context_counts(SEED, 100) == (1.0, 0.0)

    def test_context_miss_is_credited_as_a_fail(self):
        # FALSIFICATION: MISS rounds must reach the table as failures.
        ctx = PositionContextScheduler(RandPool(seed=1))
        f, arena = self._played(context=ctx)
        arena.settle(SEED, [100], Outcome.MISS, weight=1.0, score=0.0)
        assert ctx.context_counts(SEED, 100) == (0.0, 1.0)

    def test_fractal_is_credited_off_policy(self):
        # The picker was sensitivity, not fractal; its tree still heats.
        frac = PositionFractalScheduler(RandPool(seed=1))
        f, arena = self._played(fractal=frac)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert frac.cell_state(SEED)[0] == 1.0

    def test_kl_ducb_is_credited_off_policy(self):
        # The picker was sensitivity, not kl_ducb; its bandit still learns.
        kld = PositionKLDUCBScheduler(RandPool(seed=1))
        f, arena = self._played(kl_ducb=kld)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert kld.bandit_stats(SEED)["kl_ducb_pulls"] == 1

    def test_canary_is_credited_off_policy(self):
        # The picker was sensitivity, not canary; canary's posterior still learns.
        canary = PositionCanaryScheduler()
        f, arena = self._played(canary=canary)
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert canary.bandit_stats(SEED)[100] == (2.0, 1.0)

    def test_round_robin_is_credited_off_policy(self):
        # settle() calls round_robin.record(), which is a documented no-op;
        # this just pins that calling it does not raise or perturb state.
        rr = PositionRoundRobinScheduler()
        f, arena = self._played(round_robin=rr)
        first = rr.propose(SEED, len(SEED))
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert rr.propose(SEED, len(SEED)) == first + 1

    def test_fibonacci_is_credited_off_policy(self):
        # record() is a documented no-op: settle must not advance the sequence.
        fib, ref = PositionFibonacciScheduler(), PositionFibonacciScheduler()
        f, arena = self._played(fibonacci=fib)
        fib.propose(SEED, len(SEED))
        ref.propose(SEED, len(SEED))
        arena.settle(SEED, [100], Outcome.GAIN, weight=1.0, score=1.0)
        assert fib.propose(SEED, len(SEED)) == ref.propose(SEED, len(SEED))


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

    def test_fibonacci_is_the_sole_legacy_candidate(self):
        # Falsification: dropped from the candidate list, the pick would be
        # the uniform fallback, not the golden-ratio sequence.
        f = _Fuzzer()
        f._pos_fibonacci = PositionFibonacciScheduler()
        ref = PositionFibonacciScheduler()
        engine = self._engine(f)
        drawn = [engine.select_position(bytearray(SEED), SEED) for _ in range(8)]
        assert drawn == [ref.propose(SEED, len(SEED)) for _ in range(8)]

    def test_fibonacci_shares_the_pick_with_other_trackers(self):
        # Adversarial: next to TE it is one candidate of two, not a takeover.
        f = _Fuzzer(te=True)
        f._pos_fibonacci = PositionFibonacciScheduler()
        engine = self._engine(f)
        drawn = {engine.select_position(bytearray(SEED), SEED) for _ in range(200)}
        assert 22 in drawn
        assert len(drawn - {22}) >= 2

    def test_field_is_a_legacy_candidate(self):
        # FALSIFICATION: if field_pos were computed but dropped from the
        # candidate list, the format learner's offset could never be
        # returned by the non-arena (uniform-choice) path.
        f = _Fuzzer(te=True, field=True)
        f._format_learner.weighted_position = lambda d, n: 100
        engine = self._engine(f)
        drawn = {engine.select_position(bytearray(SEED), SEED) for _ in range(200)}
        assert 100 in drawn
        assert 22 in drawn

    def test_cold_field_learner_adds_no_candidate(self):
        f = _Fuzzer(te=True, field=True)
        f._format_learner.weighted_position = lambda d, n: None
        engine = self._engine(f)
        assert {engine.select_position(bytearray(SEED), SEED) for _ in range(50)} == {22}

    def test_field_absent_without_format_learner_legacy_path(self):
        # No _format_learner attribute at all (feature off) must not raise.
        f = _Fuzzer(te=True)
        engine = self._engine(f)
        assert {engine.select_position(bytearray(SEED), SEED) for _ in range(20)} == {22}


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
        # Contiguous block; later params (e.g. target_schedule) may follow.
        names = list(params)
        start = names.index("burn_front")
        assert names[start : start + 13] == [
            "burn_front",
            "position_arena",
            "pos_canary",
            "pos_round_robin",
            "pos_fibonacci",
            "pos_kl_ducb",
            "pos_fractal",
            "pos_cmplog",
            "pos_lineage",
            "pos_context",
            "pos_levy",
            "pos_arena_arms",
            "pos_boundary",
        ]
        assert params["burn_front"].default is False
        assert params["position_arena"].default is False
        assert params["pos_canary"].default is False
        assert params["pos_round_robin"].default is False
        assert params["pos_fibonacci"].default is False
        assert params["pos_kl_ducb"].default is False
        assert params["pos_fractal"].default is False
        assert params["pos_cmplog"].default is False
        assert params["pos_lineage"].default is False
        assert params["pos_context"].default is False
        assert params["pos_levy"].default is False
        assert params["pos_boundary"].default is False

    def test_cli_passes_flags_and_lists_them_for_hail_mary(self):
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
        expected = {
            "burn_front",
            "position_arena",
            "pos_canary",
            "pos_round_robin",
            "pos_fibonacci",
            "pos_kl_ducb",
            "pos_fractal",
            "pos_cmplog",
            "pos_lineage",
            "pos_context",
            "pos_levy",
            "pos_boundary",
        }
        for c in calls:
            kw = {k.arg for k in c.keywords}
            assert expected <= kw
        dests = _fuzz_parser_dests(ast.parse(inspect.getsource(commands)))
        assert expected <= dests
        assert expected <= set(commands._HAIL_MARY_FLAGS)

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


class TestRealConstruction:
    """Build a real Fuzzer: every other test here uses the _Fuzzer mock,
    which is how a constructor that could never run went unnoticed."""

    @staticmethod
    def _build(tmp_path, **kw):
        from fuzzer_tool.services.fuzzer import Fuzzer

        (tmp_path / "c").mkdir()
        (tmp_path / "x").mkdir()
        return Fuzzer(
            "/bin/true", corpus_dir=str(tmp_path / "c"), crashes_dir=str(tmp_path / "x"), **kw
        )

    def test_position_arena_constructs(self, tmp_path):
        # REGRESSION: the arena block read self._use_elo ~500 lines before
        # the constructor assigns it -> AttributeError on every
        # --position-arena run.
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._position_arena, PositionArena)

    def test_position_arena_implies_burn_front(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._burn_front, BurnFrontPositionScheduler)
        assert "burn_front" in f._position_arena.pool()

    def test_position_arena_implies_pos_canary_and_round_robin(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._pos_canary, PositionCanaryScheduler)
        assert isinstance(f._pos_round_robin, PositionRoundRobinScheduler)
        assert "canary" in f._position_arena.pool()
        assert "round_robin" in f._position_arena.pool()

    def test_position_arena_implies_fibonacci(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._pos_fibonacci, PositionFibonacciScheduler)
        assert "fibonacci" in f._position_arena.pool()

    def test_position_arena_implies_kl_ducb(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._pos_kl_ducb, PositionKLDUCBScheduler)
        assert "kl_ducb" in f._position_arena.pool()

    def test_position_arena_implies_fractal(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._pos_fractal, PositionFractalScheduler)
        assert "fractal" in f._position_arena.pool()

    def test_position_arena_implies_context(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._pos_context, PositionContextScheduler)
        assert "context" in f._position_arena.pool()

    def test_position_arena_implies_levy(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._pos_levy, PositionLevyScheduler)
        assert "levy" in f._position_arena.pool()

    def test_position_arena_implies_boundary(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._pos_boundary, PositionBoundaryScheduler)
        assert "boundary" in f._position_arena.pool()

    def test_pos_boundary_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_boundary=True)
        assert isinstance(f._pos_boundary, PositionBoundaryScheduler)
        assert f._position_arena is None

    def test_pos_levy_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_levy=True)
        assert isinstance(f._pos_levy, PositionLevyScheduler)
        assert f._position_arena is None

    def test_levy_state_survives_save_and_load(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        f._pos_levy.record(SEED, [100], Outcome.GAIN)
        f._pos_levy.record(SEED, [260], Outcome.GAIN)
        f._save_learned()
        (tmp_path / "g").mkdir()
        g = self._build(tmp_path / "g", elo="all", position_arena=True)
        g._state_store = f._state_store
        g.resume = True  # _load_learned is a no-op on a fresh run
        g._load_learned()
        assert f._pos_levy.anchor(SEED) == 260
        assert g._pos_levy.to_dict() == f._pos_levy.to_dict()
        assert g._pos_levy.walk_state(SEED) == (260, 0, [160])

    def test_pos_context_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_context=True)
        assert isinstance(f._pos_context, PositionContextScheduler)
        assert f._position_arena is None

    def test_context_state_survives_save_and_load(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        for _ in range(3):
            f._pos_context.record(SEED, [100], Outcome.GAIN)
        f._save_learned()
        (tmp_path / "g").mkdir()
        g = self._build(tmp_path / "g", elo="all", position_arena=True)
        g._state_store = f._state_store
        g.resume = True  # _load_learned is a no-op on a fresh run
        g._load_learned()
        assert f._pos_context.obs == 3
        assert g._pos_context.to_dict() == f._pos_context.to_dict()

    def test_position_arena_implies_cmplog_scheduler(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._pos_cmplog, PositionCmplogScheduler)
        # In the pool exactly when the cmplog collector is live.
        assert ("cmplog" in f._position_arena.pool()) == (f._cmplog is not None)

    def test_cmplog_arm_end_to_end_on_a_real_fuzzer(self, tmp_path):
        # Real Fuzzer, real OperatorEngine._weizz_structure_map, real
        # StructureMap RLE round trip: the arm serves a flagged span.
        from fuzzer_tool.core.weizz_tags import ByteTag, StructureMap, TagFlags

        f = self._build(tmp_path, elo="all", position_arena=True)
        seed = bytes(range(64))
        tags = [ByteTag() for _ in range(64)]
        for i in range(20, 24):
            tags[i] = ByteTag(cmp_id=1, flags=TagFlags.IS_LEN)
        f.seed_meta[seed] = {
            "weizz_tags_rle": StructureMap(tags=tags).to_rle(),
            "weizz_tags_len": 64,
        }
        f._cmplog = object()  # collector live (real start needs the shim)
        assert "cmplog" in f._position_arena.pool()
        assert "cmplog" in __import__(
            "fuzzer_tool.services.fuzzer", fromlist=["x"]
        )._active_position_schedulers(f)
        _force(f, "cmplog")
        hits = [f._position_arena.select(seed, 64) for _ in range(400)]
        in_span = sum(20 <= h < 24 for h in hits)
        # ~90% served from the span (10% uniform escape); uniform alone gives ~6%.
        assert in_span > 300
        f._cmplog = None
        assert "cmplog" not in f._position_arena.pool()

    def test_position_arena_implies_lineage_scheduler(self, tmp_path):
        f = self._build(tmp_path, elo="all", position_arena=True)
        assert isinstance(f._pos_lineage, PositionLineageScheduler)
        # Pooled exactly when lineage tracking is on (it is off here).
        assert "lineage" not in f._position_arena.pool()
        (tmp_path / "l").mkdir()
        f = self._build(tmp_path / "l", elo="all", position_arena=True, lineage=True)
        assert "lineage" in f._position_arena.pool()

    def test_lineage_arm_end_to_end_on_a_real_fuzzer(self, tmp_path):
        # Real Fuzzer, real _DELOCALISED_OPS injection: the arm lands near a
        # recorded site and ignores a delocalised operator's site.
        f = self._build(tmp_path, elo="all", position_arena=True, lineage=True)
        seed = bytes(range(256)) * 2
        f.seed_meta[seed] = {
            "parent_ops": ["byte_shuffle", "bitflip"],
            "parent_sites": [30, 400],
            "lineage_depth": 2,
        }
        assert "lineage" in f._position_arena.pool()
        assert "lineage" in __import__(
            "fuzzer_tool.services.fuzzer", fromlist=["x"]
        )._active_position_schedulers(f)
        _force(f, "lineage")
        hits = [f._position_arena.select(seed, len(seed)) for _ in range(400)]
        near = sum(abs(h - 400) <= 40 for h in hits)
        # ~80% served near site 400 (20% uniform escape, jitter mean 8 bytes);
        # uniform alone gives ~16%. Site 30 is byte_shuffle's, so excluded.
        assert near > 250
        f._use_lineage = False
        assert "lineage" not in f._position_arena.pool()

    def test_pos_lineage_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_lineage=True)
        assert isinstance(f._pos_lineage, PositionLineageScheduler)
        assert f._position_arena is None

    def test_pos_cmplog_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_cmplog=True)
        assert isinstance(f._pos_cmplog, PositionCmplogScheduler)
        assert f._position_arena is None

    def test_pos_fractal_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_fractal=True)
        assert isinstance(f._pos_fractal, PositionFractalScheduler)
        assert f._position_arena is None

    def test_burn_front_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, burn_front=True)
        assert isinstance(f._burn_front, BurnFrontPositionScheduler)
        assert f._position_arena is None

    def test_pos_canary_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_canary=True)
        assert isinstance(f._pos_canary, PositionCanaryScheduler)
        assert f._position_arena is None

    def test_pos_round_robin_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_round_robin=True)
        assert isinstance(f._pos_round_robin, PositionRoundRobinScheduler)
        assert f._position_arena is None

    def test_pos_fibonacci_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_fibonacci=True)
        assert isinstance(f._pos_fibonacci, PositionFibonacciScheduler)
        assert f._position_arena is None

    def test_pos_kl_ducb_alone_does_not_build_an_arena(self, tmp_path):
        f = self._build(tmp_path, pos_kl_ducb=True)
        assert isinstance(f._pos_kl_ducb, PositionKLDUCBScheduler)
        assert f._position_arena is None

    def test_hail_mary_enables_the_arena_and_burn_front(self, monkeypatch):
        import sys

        from fuzzer_tool.cli import commands

        seen = {}
        monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "t", "--hail-mary"])
        monkeypatch.setattr(commands, "cmd_fuzz", lambda a: seen.setdefault("a", a) and 0)
        commands.main()
        args = seen["a"]
        assert args.position_arena is True
        assert args.burn_front is True
        assert args.pos_canary is True
        assert args.pos_round_robin is True
        assert args.pos_fibonacci is True
        assert args.pos_kl_ducb is True
        assert args.pos_fractal is True
        assert args.pos_cmplog is True
        assert args.pos_lineage is True
        assert args.pos_context is True
        assert args.pos_levy is True
        assert args.pos_boundary is True
        assert args.lineage is True  # the lineage arm needs the metadata it records
        assert args.elo == "all"  # the arena needs it; hail-mary sets it

    def test_position_arena_without_elo_warns_and_constructs(self, tmp_path, caplog):
        f = self._build(tmp_path, position_arena=True)
        assert "no effect without --elo" in caplog.text
        assert isinstance(f._position_arena, PositionArena)


class _Spy(PositionScheduler):
    """Off-policy arm that counts what it is fed and always proposes 5."""

    def __init__(self, name):
        self.name = name
        self.recorded = 0

    def propose(self, data, buf_len):
        return 5

    def record(self, data, offsets, outcome, weight=1.0):
        self.recorded += 1


class TestArenaArmSubset:
    def _all(self, arms):
        f = _Fuzzer(sensitivity=True, te=True, mi=True, region=True, field=True)
        spies = {n: _Spy(n) for n in ("burn_front", "kl_ducb", "fractal", "levy", "canary")}
        arena = PositionArena(f, region_fn=lambda d, n: 55, arms=arms, **spies)
        return arena, spies

    def test_none_means_every_armed_feature(self):
        arena, _ = self._all(None)
        assert arena.enabled_arms is None
        assert {"uniform", "sensitivity", "burn_front", "fractal", "canary"} <= set(arena.pool())

    def test_subset_keeps_only_named_arms_plus_uniform(self):
        arena, _ = self._all(["fractal"])
        assert arena.pool() == ["uniform", "fractal"]
        assert arena.enabled_arms == {"uniform", "fractal"}

    def test_boundary_can_be_isolated_by_subset(self):
        f = _Fuzzer()
        spy = _Spy("boundary")
        other = _Spy("levy")
        arena = PositionArena(
            f, region_fn=lambda d, n: 55, arms=["uniform", "boundary"], boundary=spy, levy=other
        )
        assert arena.pool() == ["uniform", "boundary"]
        arena.select(SEED, len(SEED))
        arena.settle(SEED, [3], Outcome.GAIN, 1.0, 1.0)
        assert (spy.recorded, other.recorded) == (1, 0)

    def test_uniform_only_is_a_single_member_pool(self):
        arena, _ = self._all(["uniform"])
        assert arena.pool() == ["uniform"]

    def test_tracker_arms_are_filtered_too(self):
        arena, _ = self._all(["mi"])
        assert arena.pool() == ["uniform", "mi"]

    def test_excluded_arm_is_never_selected_or_credited(self):
        # Adversarial: an excluded off-policy arm must not be fed settled
        # rounds; it would accrue state it can never be rated on.
        arena, spies = self._all(["fractal"])
        for _ in range(20):
            arena.select(SEED, len(SEED))
            arena.settle(SEED, [3], Outcome.GAIN, 1.0, 1.0)
        assert spies["fractal"].recorded == 20
        assert all(s.recorded == 0 for n, s in spies.items() if n != "fractal")
        assert {n for n in arena.used()} == set()  # settled

    def test_excluded_arm_never_plays_elo_matches(self):
        arena, spies = self._all(["fractal"])
        f = arena._f
        for _ in range(60):
            arena.select(SEED, len(SEED))
            arena.settle(SEED, [3], Outcome.GAIN, 1.0, 1.0)
        rated = {k for k in f._elo._strategy_match_count if f._elo._strategy_match_count[k]}
        assert rated <= {"pos_uniform", "pos_fractal"}

    def test_ungated_arm_in_subset_still_waits_for_its_feature(self):
        # cmplog named but cmplog not live: subset must not force it in.
        f = _Fuzzer()
        arena = PositionArena(
            f, region_fn=lambda d, n: 55, cmplog=_Spy("cmplog"), arms=["uniform", "cmplog"]
        )
        assert arena.pool() == ["uniform"]

    def test_unknown_arm_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            PositionArena(_Fuzzer(), region_fn=lambda d, n: 1, arms=["uniform", "bogus"])


class TestParseArenaArms:
    def test_parses_dedupes_and_normalises_hyphens(self):
        assert parse_arena_arms("uniform, kl-ducb,fractal,uniform") == (
            "uniform",
            "kl_ducb",
            "fractal",
        )

    @pytest.mark.parametrize("bad", ["", " , ", "nope", "uniform,nope"])
    def test_rejects_empty_and_unknown(self, bad):
        with pytest.raises(ValueError):
            parse_arena_arms(bad)

    def test_every_registered_name_parses(self):
        assert parse_arena_arms(",".join(POSITION_STRATEGY_NAMES)) == POSITION_STRATEGY_NAMES


class TestBannerHonoursSubset:
    def test_dropped_arms_are_not_listed(self):
        from fuzzer_tool.services.fuzzer import _active_position_schedulers

        f = _Fuzzer()
        f._burn_front, f._pos_kl_ducb, f._pos_fractal = _Spy("b"), _Spy("k"), _Spy("f")
        f._position_arena = PositionArena(
            f,
            region_fn=lambda d, n: 1,
            burn_front=f._burn_front,
            kl_ducb=f._pos_kl_ducb,
            fractal=f._pos_fractal,
            arms=["uniform", "kl-ducb"],
        )
        assert _active_position_schedulers(f) == ["kl-ducb"]
        f._position_arena = None
        assert _active_position_schedulers(f) == ["burn-front", "kl-ducb", "fractal"]
