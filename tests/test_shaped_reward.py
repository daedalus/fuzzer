"""``--shaped-reward``: class-deduplicated shaping of the shared operator reward.

The arithmetic lives in ``core/schedulers/op_credit.shaped_weight`` and is covered
by ``tests/test_op_credit.py`` for the selector's own use of it. What is pinned
here is the part that is *not* the arithmetic:

* the guard set in ``Fuzzer._credit_reward_shape``, which decides when the class
  partition is allowed to speak at all. Each of its three "leave the reward
  alone" answers is a separate failure if it goes wrong, and the middle one --
  a round that succeeded without finding an edge -- would silently delete every
  crash, hang, new-max and cmp-progress reward in the fuzzer if the factor were
  applied unconditionally;
* ``--shaped-reward-floor``, the clamp that makes the two collapse modes (a long
  duplicate chain paying ``1/n``, a derived-only round paying 0) a measurable
  knob instead of a design bet baked into the code.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.edge_matrix import MatrixSubstrate
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_credit import OpCreditScheduler, shaped_weight
from fuzzer_tool.services.fuzzer import Fuzzer, _apply_reward_shape


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    """coverage_trust returns early for a target-less run, so gate tests need one."""
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")


class _Tracker:
    def __init__(self, profiles):
        self.seed_hit_counts = profiles
        self.seed_edges = {k: set(v) for k, v in profiles.items()}
        self.cumulative_edges = set().union(*self.seed_edges.values())


# Edges 10-12 are one straight-line chain (identical columns); 20 and 30 stand alone.
PROFILES = {"a": {10: 1, 11: 1, 12: 1}, "b": {10: 1, 11: 1, 12: 1, 20: 4}, "c": {20: 4, 30: 1}}


def _sub():
    sub = MatrixSubstrate(target="/x")
    sub.maybe_refit(_Tracker(PROFILES), 0)
    return sub


class TestShapedWeightFunction:
    """The module function: same answers as the method, without the selector."""

    def test_a_chain_pays_one_over_n(self):
        assert shaped_weight(_sub(), {10, 11, 12}) == pytest.approx(1 / 3)

    def test_independent_edges_pay_in_full(self):
        assert shaped_weight(_sub(), {10, 20, 30}) == 1.0

    def test_an_empty_round_pays_nothing_even_with_a_floor(self):
        # There is no round to shape, so the floor has nothing to clamp: a
        # non-zero answer here would invent a reward out of no discovery.
        assert shaped_weight(_sub(), set()) == 0.0
        assert shaped_weight(_sub(), [], floor=0.9) == 0.0

    def test_a_derived_only_round_pays_nothing(self):
        # `derived` is empty in production (P1-2 negative, handover F17); this
        # pins the contract should a graph-derived mask ever fill it.
        sub = _sub()
        sub.derived = {10, 11, 12}
        assert shaped_weight(sub, {10, 11, 12}) == 0.0

    def test_the_floor_clamps_from_below_and_only_from_below(self):
        sub = _sub()
        assert shaped_weight(sub, {10, 11, 12}, floor=0.5) == 0.5  # raw 1/3, lifted
        assert shaped_weight(sub, {10, 20, 30}, floor=0.5) == 1.0  # raw 1.0, untouched

    def test_floor_one_disables_the_shaping_without_unwiring_it(self):
        sub = _sub()
        assert shaped_weight(sub, {10, 11, 12}, floor=1.0) == 1.0

    def test_the_floor_cannot_push_the_factor_outside_the_reward_contract(self):
        # op_rewards is documented as [0, 1] and several consumers assume it.
        sub = _sub()
        assert shaped_weight(sub, {10, 11, 12}, floor=7.0) == 1.0
        assert shaped_weight(sub, {10, 11, 12}, floor=-3.0) == pytest.approx(1 / 3)

    def test_the_method_is_the_same_answer_bound_to_its_substrate(self):
        sub = _sub()
        s = OpCreditScheduler(RandPool(seed=1), sub)
        for edges in ({10, 11, 12}, {10, 20, 30}, set()):
            assert s.shaped_weight(edges) == shaped_weight(sub, edges)
        assert s.shaped_weight({10, 11, 12}, floor=0.5) == 0.5


def _make_banner_fuzzer(**overrides):
    """The banner fake, reused verbatim from the op_kuramoto banner regression.

    Imported rather than copied so the two stay in step: that module is where the
    list of attributes ``_print_enabled_features`` reads unconditionally is
    maintained.
    """
    from tests.test_regression_enabled_features_op_kuramoto import _make_fake_fuzzer

    base = dict(_shaped_reward=False, _shaped_reward_floor=0.0)
    base.update(overrides)
    return _make_fake_fuzzer(**base)


def _fuzzer(**overrides):
    """Minimal stand-in exposing exactly what _credit_reward_shape reads."""
    base = dict(
        _shaped_reward=True,
        _shaped_reward_floor=0.0,
        _matrix_substrate=_sub(),
        _last_new_edge_ids=[10, 11, 12],
        _shaped_reward_rounds=0,
        _shaped_reward_gated=0,
        _shaped_reward_factor_sum=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestShapeGuard:
    def test_shapes_a_real_discovery_when_the_gate_is_open(self):
        f = _fuzzer()
        assert Fuzzer._credit_reward_shape(f) == pytest.approx(1 / 3)
        assert f._shaped_reward_rounds == 1

    def test_flag_off_is_no_shaping_at_all(self):
        f = _fuzzer(_shaped_reward=False)
        assert Fuzzer._credit_reward_shape(f) is None
        assert f._shaped_reward_rounds == 0

    def test_no_substrate_is_no_shaping(self):
        assert Fuzzer._credit_reward_shape(_fuzzer(_matrix_substrate=None)) is None

    def test_a_round_that_found_no_edge_is_left_alone(self):
        # `success` is a disjunction -- crash, interesting, slow, new max, cmp
        # progress, new valid coverage. Returning 0.0 here rather than None would
        # zero every one of those rewards for every scheduler.
        f = _fuzzer(_last_new_edge_ids=[])
        assert Fuzzer._credit_reward_shape(f) is None
        assert f._shaped_reward_rounds == 0

    def test_a_closed_gate_is_left_alone_and_counted_apart(self):
        # F1/F11: under per-process ids every id is a singleton owned by one
        # seed, so the classes are noise. Counting gated rounds separately is
        # what lets a bench run tell "shaping was off" from "shaping was neutral".
        f = _fuzzer()
        f._matrix_substrate.set_stability(0.007)
        assert Fuzzer._credit_reward_shape(f) is None
        assert (f._shaped_reward_gated, f._shaped_reward_rounds) == (1, 0)

    def test_the_floor_reaches_the_shape(self):
        f = _fuzzer(_shaped_reward_floor=0.5)
        assert Fuzzer._credit_reward_shape(f) == 0.5

    def test_stats_average_only_the_rounds_that_were_shaped(self):
        f = _fuzzer()
        Fuzzer._credit_reward_shape(f)  # 1/3
        f._last_new_edge_ids = [10, 20, 30]
        Fuzzer._credit_reward_shape(f)  # 1.0
        f._last_new_edge_ids = []
        Fuzzer._credit_reward_shape(f)  # not shaped, not counted
        stats = Fuzzer.shaped_reward_stats(f)
        assert stats["shaped_reward_rounds"] == 2
        assert stats["shaped_reward_mean_factor"] == pytest.approx((1 / 3 + 1.0) / 2)

    def test_stats_report_no_mean_when_nothing_was_shaped(self):
        assert Fuzzer.shaped_reward_stats(_fuzzer())["shaped_reward_mean_factor"] is None


class TestBanner:
    """A campaign whose rewards are being rescaled has to say so in the banner.

    Drives the real ``_print_enabled_features`` (the fake is the one from
    ``tests/test_regression_enabled_features_op_kuramoto.py``, which is the
    established way to run that method without standing up a Fuzzer): a test
    that re-implemented the two lines would pass with the feature reverted.
    """

    def _out(self, capsys, **overrides):
        fake = _make_banner_fuzzer(**overrides)
        Fuzzer._print_enabled_features(fake)
        return capsys.readouterr().out

    def test_named_when_on(self, capsys):
        out = self._out(capsys, _shaped_reward=True)
        assert "Scheduling:" in out and "shaped-reward" in out

    def test_the_floor_is_part_of_the_name(self, capsys):
        out = self._out(capsys, _shaped_reward=True, _shaped_reward_floor=0.25)
        assert "shaped-reward(floor=0.25)" in out

    def test_silent_when_off(self, capsys):
        assert "shaped-reward" not in self._out(capsys)


class TestTheFactorReachesTheRewardList:
    """The assertion with teeth.

    The guard and the arithmetic can both be right while nothing in ``fuzz_one``
    multiplies anything -- a feature wired at one end and silent at the other,
    with a green suite. That is the forkserver failure this repo has already paid
    for once (``docs/handover/handover_done_2026-09-06.md`` §10). The shaping is
    therefore a pure transform over the finished ``op_rewards`` list, applied at
    exactly one call site, and this drives it directly.
    """

    REWARDS = [("havoc", True, 0.8), ("splice", False, 0.0), ("bitflip", True, 1.0)]

    def test_successes_are_scaled(self):
        assert _apply_reward_shape(self.REWARDS, 0.5) == [
            ("havoc", True, 0.4),
            ("splice", False, 0.0),
            ("bitflip", True, 0.5),
        ]

    def test_none_leaves_the_list_untouched(self):
        assert _apply_reward_shape(self.REWARDS, None) == self.REWARDS

    def test_failures_are_never_shaped(self):
        # A failure's weight describes no discovery, so a discovery's class count
        # has no business scaling it. Today every failure carries 0.0 and the
        # branch is arithmetic on a zero; it is explicit so that a future
        # non-zero failure weight does not inherit the factor by accident.
        assert _apply_reward_shape([("op", False, 0.25)], 0.5) == [("op", False, 0.25)]

    def test_shaping_is_a_discount_never_a_promotion(self):
        # The loop caps at 1.0 before this runs and the factor is in [0, 1], so
        # the [0, 1] contract every consumer of op_rewards documents survives.
        assert all(w <= 1.0 for _, _, w in _apply_reward_shape(self.REWARDS, 1.0))

    def test_the_call_site_passes_the_gate_through(self):
        # Pins the wiring, not the transform: fuzz_one must hand
        # _credit_reward_shape's answer to _apply_reward_shape, not a literal.
        import inspect

        src = inspect.getsource(Fuzzer.fuzz_one)
        assert "_apply_reward_shape(op_rewards, self._credit_reward_shape())" in src
