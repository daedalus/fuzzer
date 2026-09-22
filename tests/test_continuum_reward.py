"""``--continuum-reward``: neighbourhood-scarcity shaping of the shared operator reward.

``surprisal_weight`` (``1 - bitmap_density``) is one global scalar, so it cannot tell a
discovery in scarce territory from one that fills a gap in a saturated path.  The factor
here can: it is the mean pressure (``core/analyzers/analyzer_navier_stokes.py``) of the
edges co-hit with the new ones, with pressure taken from how many corpus seeds own each
edge.  Pinned here:

* the arithmetic of ``frontier_weight``, on hand-computable ownership (N=3 seeds: an edge
  owned by 0/1/3 seeds has pressure exactly 1.0/0.5/0.0, no logarithm needed to check it);
* the guard set in ``Fuzzer._continuum_reward_shape``.  A round that succeeded without an
  edge (crash, hang, new max) must be left alone or every such reward is zeroed;
* the wiring at both ends of ``fuzz_one``, the flag, and the bench arm.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from fuzzer_tool.core.analyzers.analyzer_navier_stokes import frontier_weight
from fuzzer_tool.services.fuzzer import Fuzzer, _apply_reward_shape
from tests.test_bench_paired_arms import _parse

N_SEEDS = 3
SCARCE = 1  # pressure 0.5 at N_SEEDS
SATURATED = N_SEEDS  # pressure 0.0 at N_SEEDS


def _owners(table):
    return lambda edge: table.get(edge, 0)


class TestFrontierWeight:
    def test_mean_pressure_of_the_co_hit_edges(self):
        owners = _owners({11: SCARCE, 12: SATURATED})
        # (0.5 + 0.0) / 2
        assert frontier_weight([10], {10, 11, 12}, owners, N_SEEDS) == pytest.approx(0.25)

    def test_falsification_scarce_territory_outpays_a_saturated_gap(self):
        # Same new edge, same trace shape; only the ownership of the neighbours differs.
        # A constant factor (the global-scalar behaviour this replaces) fails this.
        scarce = frontier_weight([10], {10, 11, 12}, _owners({11: 1, 12: 1}), N_SEEDS)
        gap = frontier_weight([10], {10, 11, 12}, _owners({11: 3, 12: 3}), N_SEEDS)
        assert scarce == pytest.approx(0.5)
        assert gap == 0.0
        assert scarce > gap

    def test_a_new_edge_is_not_its_own_neighbour(self):
        # Ownership of the new edge itself must not move the answer.
        a = frontier_weight([10], {10, 11}, _owners({10: 0, 11: SCARCE}), N_SEEDS)
        b = frontier_weight([10], {10, 11}, _owners({10: SATURATED, 11: SCARCE}), N_SEEDS)
        assert a == b == pytest.approx(0.5)

    def test_new_edges_missing_from_the_trace_are_harmless(self):
        owners = _owners({11: SCARCE})
        assert frontier_weight([99], {11}, owners, N_SEEDS) == pytest.approx(0.5)

    def test_a_trace_of_only_new_edges_has_no_neighbourhood(self):
        assert frontier_weight([10, 11], {10, 11}, _owners({}), N_SEEDS) is None

    def test_an_empty_trace_has_no_neighbourhood(self):
        assert frontier_weight([10], set(), _owners({}), N_SEEDS) is None

    def test_no_seeds_means_no_scale_to_measure_against(self):
        assert frontier_weight([10], {10, 11}, _owners({11: 1}), 0) is None
        assert frontier_weight([10], {10, 11}, _owners({11: 1}), -4) is None

    def test_the_floor_clamps_from_below_and_only_from_below(self):
        gap = _owners({11: SATURATED})
        scarce = _owners({11: SCARCE})
        assert frontier_weight([10], {10, 11}, gap, N_SEEDS, floor=0.5) == 0.5
        assert frontier_weight([10], {10, 11}, scarce, N_SEEDS, floor=0.25) == pytest.approx(0.5)

    def test_the_floor_cannot_leave_the_reward_contract(self):
        # op_rewards is documented [0, 1]; several consumers assume it.
        owners = _owners({11: SCARCE})
        assert frontier_weight([10], {10, 11}, owners, N_SEEDS, floor=7.0) == 1.0
        assert frontier_weight([10], {10, 11}, owners, N_SEEDS, floor=-3.0) == pytest.approx(0.5)

    def test_same_inputs_same_answer(self):
        # Control (Hard Rule 46): a pure function must agree with a second run of itself.
        owners = _owners({11: SCARCE, 12: SATURATED})
        first = frontier_weight([10], {10, 11, 12}, owners, N_SEEDS)
        assert frontier_weight([10], {10, 11, 12}, owners, N_SEEDS) == first

    def test_a_list_trace_with_duplicates_equals_the_set(self):
        owners = _owners({11: SCARCE, 12: SATURATED})
        as_list = frontier_weight([10], [10, 11, 11, 12, 12, 12], owners, N_SEEDS)
        assert as_list == frontier_weight([10], {10, 11, 12}, owners, N_SEEDS)


class TestFrontierWeightAdversarial:
    """Ownership numbers the tracker should never produce, but a pruned or resized one can."""

    def test_ownership_above_the_seed_count_pays_zero_not_negative(self):
        assert frontier_weight([10], {10, 11}, _owners({11: 10_000}), N_SEEDS) == 0.0

    def test_negative_ownership_is_clamped_not_trusted(self):
        # log1p(-1) raises and log1p(<-1) is nan; a negative count reads as "unowned".
        assert frontier_weight([10], {10, 11}, _owners({11: -5}), N_SEEDS) == 1.0
        assert frontier_weight([10], {10, 11}, _owners({11: -1}), N_SEEDS) == 1.0

    @pytest.mark.parametrize("owned", [0, 1, 2, 3, 4, 10**9])
    @pytest.mark.parametrize("seeds", [1, 2, 3, 10**6])
    def test_always_inside_the_unit_interval(self, owned, seeds):
        w = frontier_weight([10], {10, 11, 12}, _owners({11: owned, 12: owned}), seeds)
        assert w is not None and 0.0 <= w <= 1.0

    def test_a_large_trace_stays_bounded(self):
        trace = set(range(50_000))
        owners = lambda e: e % 7  # noqa: E731
        w = frontier_weight([0], trace, owners, N_SEEDS)
        assert w is not None and 0.0 <= w <= 1.0


class TestShapeApplication:
    REWARDS = [("havoc", True, 0.8), ("splice", False, 0.0)]

    def test_factors_compose_by_multiplication(self):
        both = _apply_reward_shape(_apply_reward_shape(self.REWARDS, 0.5), 0.5)
        assert both == [("havoc", True, pytest.approx(0.2)), ("splice", False, 0.0)]

    def test_composition_is_a_discount_never_a_promotion(self):
        both = _apply_reward_shape(_apply_reward_shape(self.REWARDS, 1.0), 1.0)
        assert all(w <= 1.0 for _, _, w in both)


def _fuzzer(**overrides):
    """Stand-in exposing exactly what ``_continuum_reward_shape`` reads."""
    owned = {11: SCARCE, 12: SATURATED}
    tracker = SimpleNamespace(
        edge_owner_count=_owners(owned),
        seed_edges={"a": {10, 11, 12}, "b": {11, 12}, "c": {12}},
    )
    base = dict(
        _continuum_reward=True,
        _continuum_reward_floor=0.0,
        _edge_tracker=tracker,
        _last_new_edge_ids=[10],
        _last_trace_edges={10, 11, 12},
        _continuum_reward_rounds=0,
        _continuum_reward_neutral=0,
        _continuum_reward_factor_sum=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestShapeGuard:
    def test_shapes_a_real_discovery(self):
        f = _fuzzer()
        assert Fuzzer._continuum_reward_shape(f) == pytest.approx(0.25)
        assert f._continuum_reward_rounds == 1

    def test_flag_off_is_no_shaping_at_all(self):
        f = _fuzzer(_continuum_reward=False)
        assert Fuzzer._continuum_reward_shape(f) is None
        assert f._continuum_reward_rounds == 0

    def test_a_round_that_found_no_edge_is_left_alone(self):
        # `success` is a disjunction (crash, hang, new max, cmp progress, ...). A 0.0
        # here instead of None would zero every one of those rewards for every arm.
        f = _fuzzer(_last_new_edge_ids=[])
        assert Fuzzer._continuum_reward_shape(f) is None
        assert (f._continuum_reward_rounds, f._continuum_reward_neutral) == (0, 0)

    def test_a_stale_trace_cannot_shape_a_round_without_a_discovery(self):
        f = _fuzzer(_last_new_edge_ids=[], _last_trace_edges={10, 11, 12})
        assert Fuzzer._continuum_reward_shape(f) is None

    def test_no_edge_tracker_is_no_shaping(self):
        assert Fuzzer._continuum_reward_shape(_fuzzer(_edge_tracker=None)) is None

    def test_a_round_with_no_neighbourhood_is_left_alone_and_counted_apart(self):
        f = _fuzzer(_last_trace_edges={10})
        assert Fuzzer._continuum_reward_shape(f) is None
        assert (f._continuum_reward_rounds, f._continuum_reward_neutral) == (0, 1)

    def test_the_floor_reaches_the_shape(self):
        assert Fuzzer._continuum_reward_shape(_fuzzer(_continuum_reward_floor=0.6)) == 0.6

    def test_stats_average_only_the_rounds_that_were_shaped(self):
        f = _fuzzer()
        Fuzzer._continuum_reward_shape(f)  # 0.25
        f._last_trace_edges = {10, 11}
        Fuzzer._continuum_reward_shape(f)  # 0.5
        f._last_trace_edges = {10}
        Fuzzer._continuum_reward_shape(f)  # neutral, not averaged
        f._last_new_edge_ids = []
        Fuzzer._continuum_reward_shape(f)  # not a discovery, not counted
        stats = Fuzzer.continuum_reward_stats(f)
        assert stats["continuum_reward_rounds"] == 2
        assert stats["continuum_reward_neutral_rounds"] == 1
        assert stats["continuum_reward_mean_factor"] == pytest.approx((0.25 + 0.5) / 2)

    def test_stats_report_no_mean_when_nothing_was_shaped(self):
        assert Fuzzer.continuum_reward_stats(_fuzzer())["continuum_reward_mean_factor"] is None


class TestBanner:
    """A campaign whose rewards are being rescaled has to say so in the banner."""

    def _out(self, capsys, **overrides):
        from tests.test_regression_enabled_features_op_kuramoto import _make_fake_fuzzer

        Fuzzer._print_enabled_features(_make_fake_fuzzer(**overrides))
        return capsys.readouterr().out

    def test_named_when_on(self, capsys):
        out = self._out(capsys, _continuum_reward=True, _continuum_reward_floor=0.0)
        assert "Scheduling:" in out and "continuum-reward" in out

    def test_the_floor_is_part_of_the_name(self, capsys):
        out = self._out(capsys, _continuum_reward=True, _continuum_reward_floor=0.25)
        assert "continuum-reward(floor=0.25)" in out

    def test_silent_when_off(self, capsys):
        assert "continuum-reward" not in self._out(capsys)


class TestWiring:
    """Assertions with teeth: guard and arithmetic can be right while ``fuzz_one`` does nothing."""

    def test_fuzz_one_applies_the_shape(self):
        src = inspect.getsource(Fuzzer.fuzz_one)
        assert "_apply_reward_shape(op_rewards, self._continuum_reward_shape())" in src

    def test_fuzz_one_keeps_the_class_credit_shape(self):
        src = inspect.getsource(Fuzzer.fuzz_one)
        assert "_apply_reward_shape(op_rewards, self._credit_reward_shape())" in src

    def test_fuzz_one_captures_the_trace_beside_the_new_edge_ids(self):
        src = inspect.getsource(Fuzzer.fuzz_one)
        assert "self._last_new_edge_ids = list(new)" in src
        assert "self._last_trace_edges = hit_edges" in src

    def test_fuzz_one_resets_the_trace_every_round(self):
        src = inspect.getsource(Fuzzer.fuzz_one)
        assert "self._last_trace_edges: Collection[int] = ()" in src

    def test_fuzzer_takes_both_knobs_with_inert_defaults(self):
        params = inspect.signature(Fuzzer.__init__).parameters
        assert params["continuum_reward"].default is False
        assert params["continuum_reward_floor"].default == 0.0


class TestCli:
    def test_flags_reach_the_real_parser(self, monkeypatch):
        args = _parse(monkeypatch, ["--continuum-reward", "--continuum-reward-floor", "0.25"])
        assert args.continuum_reward is True
        assert args.continuum_reward_floor == 0.25

    def test_off_by_default(self, monkeypatch):
        args = _parse(monkeypatch, [])
        assert args.continuum_reward is False
        assert args.continuum_reward_floor == 0.0

    def test_independent_of_continuum(self, monkeypatch):
        # One variable per paired run: --continuum also re-ranks invasion operators.
        args = _parse(monkeypatch, ["--continuum-reward"])
        assert args.continuum is False

    def test_a_typo_is_rejected(self, monkeypatch):
        with pytest.raises(SystemExit):
            _parse(monkeypatch, ["--continuum-rewardd"])


class TestBenchArm:
    def test_arm_is_the_elo_baseline_plus_the_knob(self, monkeypatch):
        from bench_paired import ARMS

        arm, base = ARMS["elo-continuum-reward"], ARMS["elo"]
        assert arm[: len(base)] == base and len(arm) > len(base)
        parsed, ref = vars(_parse(monkeypatch, arm)), vars(_parse(monkeypatch, base))
        changed = {k for k in ref if k != "func" and ref[k] != parsed[k]}
        assert changed == {"continuum_reward"}
