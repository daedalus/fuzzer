"""Target arena (``--target-arena``): Elo over target schedulers, ``tgt_`` keys."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fuzzer_tool.cli import commands
from fuzzer_tool.core.analyzers.analyzer_elo import (
    TGT_STRATEGY_PREFIX,
    Arena,
    strategy_arena,
    strategy_display_name,
)
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.schedulers.tgt_auction import AuctionTarget
from fuzzer_tool.core.schedulers.tgt_base import (
    WARMUP_EXECS,
    RoundRobinTarget,
    TargetRound,
    WeightedTarget,
    WfqTarget,
    WrrTarget,
)
from fuzzer_tool.core.schedulers.tgt_gale_shapley import GaleShapleyTarget, seat_quotas
from fuzzer_tool.core.target_schedule import TargetSchedule
from fuzzer_tool.services.fuzzer import Fuzzer
from fuzzer_tool.services.seed_picker import TARGET_MATCH
from fuzzer_tool.services.stats import _arena_leaders
from fuzzer_tool.services.target_arena import (
    AUCTION,
    GALE_SHAPLEY,
    TARGET_STRATEGY_NAMES,
    TargetArena,
)
from tests.support.scripted_rng import ScriptedRng
from tests.test_dirichlet_wiring import _TARGET, _fuzzer_call_kwargs, _parse, build  # noqa: F401

TARGETS = ["t0", "t1", "t2"]
A, B = b"seed-a", b"seed-b"


def _rnd(seed, idx, outcome=Outcome.MISS, cost=0.0):
    return TargetRound(seed=seed, idx=idx, outcome=outcome, weight=1.0, cost=cost)


def _feed(arm, seed, idx, gains, tries):
    """``tries`` rounds of (seed, idx), the first ``gains`` of them gains."""
    for i in range(tries):
        arm.record(_rnd(seed, idx, Outcome.GAIN if i < gains else Outcome.MISS))


# --- ported schedules ---------------------------------------------------------


def test_round_robin_cycles():
    arm = RoundRobinTarget()

    assert [arm.pick(3) for _ in range(5)] == [0, 1, 2, 0, 1]


def test_weighted_round_robins_during_warmup():
    """Falsification: no RNG draw before the warm-up ends (empty script would raise)."""
    arm = WeightedTarget(ScriptedRng(), lambda: [1.0, 1.0, 1.0], lambda: 0)

    assert [arm.pick(3) for _ in range(4)] == [0, 1, 2, 0]


def test_weighted_draws_from_inverse_edges_after_warmup():
    weights = [1.0, 0.001, 0.001]
    r_in_t0 = 0.5 * weights[0] / sum(weights)
    arm = WeightedTarget(ScriptedRng(randoms=[r_in_t0]), lambda: weights, lambda: WARMUP_EXECS + 1)

    assert arm.pick(3) == 0


def test_weighted_top_of_range_lands_on_last():
    """Adversarial: r == 1.0 must not fall off the CDF (float sum < total)."""
    arm = WeightedTarget(ScriptedRng(randoms=[1.0]), lambda: [0.1] * 3, lambda: WARMUP_EXECS + 1)

    assert arm.pick(3) == 2


def test_wrr_exact_shares():
    arm = WrrTarget("wrr", lambda: [4.0, 2.0, 1.0])
    picks = [arm.pick(3) for _ in range(70)]

    assert [picks.count(i) for i in range(3)] == [40, 20, 10]


def test_wfq_charge_comes_from_record():
    """A round's cost is charged to the target that ran, whoever picked it."""
    arm = WfqTarget(lambda: [1.0, 1.0])
    assert arm.pick(2) == 0  # tie -> first flow
    arm.record(_rnd(A, 0, cost=10.0))

    assert arm.pick(2) == 1


def test_wfq_negative_cost_is_neutral():
    """Adversarial: clock step back -> neutral charge, no crash, no negative time."""
    arm = WfqTarget(lambda: [1.0, 1.0])
    arm.record(_rnd(A, 0, cost=-5.0))

    assert arm.pick(2) in (0, 1)


# --- Gale-Shapley arm ---------------------------------------------------------


def _gs(seeds, weights=(1.0, 1.0), batch=8, max_seeds=4096):
    calls = []

    def seeds_fn(k):
        calls.append(k)
        return list(seeds)[:k]

    arm = GaleShapleyTarget(
        len(weights), seeds_fn, lambda: list(weights), lambda d: d, batch, max_seeds
    )
    return arm, calls


def test_gs_seeds_get_their_best_target():
    """Falsification: A pays on t1 only, B on t0 only -> A->t1, B->t0."""
    arm, _ = _gs([A, B])
    _feed(arm, A, 1, gains=3, tries=3)
    _feed(arm, A, 0, gains=0, tries=3)
    _feed(arm, B, 0, gains=3, tries=3)
    _feed(arm, B, 1, gains=0, tries=3)

    first = (arm.pick(2), arm.take_hint())
    second = (arm.pick(2), arm.take_hint())

    assert first == (1, A)
    assert second == (0, B)


def test_gs_target_prefers_the_less_tried_seed():
    """Both seeds want t0 (quota 1); t0 keeps B, run there 1x, over A, run there 5x.

    Yields (Laplace): A (5+1)/(5+2) > B (1+1)/(1+2) > untried t1 (0+1)/(0+2),
    so seed taste alone would hand t0 to A. The target side decides.
    """
    arm, _ = _gs([A, B])
    _feed(arm, A, 0, gains=5, tries=5)
    _feed(arm, B, 0, gains=1, tries=1)

    plan = {}
    for _ in range(2):
        idx = arm.pick(2)
        plan[arm.take_hint()] = idx

    assert plan == {A: 1, B: 0}


def test_gs_empty_corpus_declines():
    """Adversarial: no seeds -> a valid index, no hint, no crash."""
    arm, _ = _gs([])

    assert arm.pick(2) in (0, 1)
    assert arm.take_hint() is None


def test_gs_rematches_when_plan_runs_out():
    arm, calls = _gs([A, B])
    for _ in range(3):
        arm.pick(2)

    assert len(calls) == 2


def test_gs_hint_is_consumed_once():
    arm, _ = _gs([A])
    arm.pick(2)

    assert arm.take_hint() == A
    assert arm.take_hint() is None


def test_gs_memory_bounded():
    """Adversarial: far more seeds than the cap -> tracked rows never exceed it."""
    arm, _ = _gs([], max_seeds=4)
    for i in range(10):
        arm.record(_rnd(bytes([i]), 0, Outcome.GAIN))

    assert arm.tracked <= 4


def test_gs_ignores_out_of_range_target():
    """Adversarial: a stale target index (fewer targets now) is dropped, not an IndexError."""
    arm, _ = _gs([A])
    arm.record(_rnd(A, 7))

    assert arm.tracked == 0


def test_seat_quotas_cover_every_seed():
    assert seat_quotas(3, [1.0, 1.0, 1.0]) == [1, 1, 1]
    assert seat_quotas(4, [3.0, 1.0]) == [3, 1]
    for k in range(0, 20):
        for w in ([1.0, 2.0, 7.0], [0.3, 0.3], [1e-9, 1.0, 1e9]):
            q = seat_quotas(k, w)
            assert sum(q) >= k and all(x >= 0 for x in q)


def test_seat_quotas_bad_weights_fall_back_to_equal():
    """Adversarial: NaN / zero / negative weights -> equal seats, never a crash."""
    assert seat_quotas(2, [math.nan, 0.0]) == [1, 1]
    assert seat_quotas(4, [-1.0, math.inf]) == [2, 2]


# --- auction arm ----------------------------------------------------------------


class DrawRng:
    """Scripted Thompson step: returns ``draws`` and records the Beta parameters."""

    def __init__(self, draws):
        self._draws = np.array(draws, dtype=float)
        self.params = []

    def betavariate_array(self, alphas, betas):
        self.params.append((np.array(alphas), np.array(betas)))
        return self._draws


def _auc(seeds, draws, weights=(1.0, 1.0)):
    rng = DrawRng(draws)
    arm = AuctionTarget(
        rng, len(weights), lambda k: list(seeds)[:k], lambda: list(weights), lambda d: d
    )
    return arm, rng


def _plan(arm, n, rounds):
    out = {}
    for _ in range(rounds):
        idx = arm.pick(n)
        out[arm.take_hint()] = idx
    return out


def test_auction_maximizes_total_draw():
    """Falsification: seed-greedy would give A t0 (0.9 + 0.1); the optimum is 0.8 + 0.85."""
    arm, _ = _auc([A, B], [[0.9, 0.8], [0.85, 0.1]])

    assert _plan(arm, 2, 2) == {A: 1, B: 0}


def test_auction_draws_from_yield_posterior():
    """Beta(gains+1, misses+1) per (seed, target), derived from the fed rounds."""
    arm, rng = _auc([A, B], [[0.5, 0.5], [0.5, 0.5]])
    _feed(arm, A, 0, gains=2, tries=5)
    _feed(arm, B, 1, gains=1, tries=1)

    arm.pick(2)
    alphas, betas = rng.params[0]

    assert alphas.tolist() == [[2 + 1, 0 + 1], [0 + 1, 1 + 1]]
    assert betas.tolist() == [[3 + 1, 0 + 1], [0 + 1, 0 + 1]]


def test_auction_empty_corpus_declines():
    """Adversarial: no seeds -> a valid index, no hint, no draw."""
    arm, rng = _auc([], [])

    assert arm.pick(2) in (0, 1)
    assert arm.take_hint() is None
    assert rng.params == []


def test_auction_nan_draw_still_plans():
    """Adversarial: a broken draw must not crash or drop a seed from the plan."""
    arm, _ = _auc([A, B], [[np.nan, 0.2], [np.inf, 0.3]])

    assert set(_plan(arm, 2, 2)) == {A, B}


def test_auction_memory_bounded():
    """Adversarial: rows stay LRU-bounded like Gale-Shapley's."""
    rng = DrawRng([])
    arm = AuctionTarget(rng, 1, lambda k: [], lambda: [1.0], lambda d: d, max_seeds=4)
    for i in range(10):
        arm.record(_rnd(bytes([i]), 0, Outcome.GAIN))

    assert arm.tracked <= 4


# --- arena ----------------------------------------------------------------------


class FakeElo:
    def __init__(self, picks=()):
        self._picks = iter(picks)
        self.offered: list[list[str]] = []
        self.matches: list[tuple[str, str, float]] = []

    def select_strategy(self, keys):
        self.offered.append(list(keys))
        return next(self._picks)

    def record_strategy_match(self, a, b, score):
        self.matches.append((a, b, score))


def _arena(picks=(), clock=lambda: 0.0, use_elo=True):
    elo = FakeElo(picks)
    f = SimpleNamespace(
        multi_targets=list(TARGETS),
        _rng=RandPool(seed=0),
        exec_count=0,
        _inv_edge_weights=lambda: [1.0, 1.0, 1.0],
        _phi_weights=lambda: [1.0, 1.0, 1.0],
        corpus=[A, B],
        _seed_key=lambda d: d.hex(),
        _use_elo=use_elo,
        _elo=elo,
    )
    return TargetArena(f, clock=clock), elo


def _key(name):
    return TGT_STRATEGY_PREFIX + name


def test_pool_is_every_schedule_plus_gale_shapley():
    """Derived from the enum: a new TargetSchedule joins the arena or this fails."""
    arena, _ = _arena()
    expected = [s.value.replace("-", "_") for s in TargetSchedule] + [GALE_SHAPLEY, AUCTION]

    assert arena.pool() == expected == list(TARGET_STRATEGY_NAMES)
    assert arena.pool()[0] == TargetSchedule.WEIGHTED.value  # cold-start pick = old default


def test_select_offers_prefixed_pool():
    arena, elo = _arena(picks=[_key("round_robin")])

    assert arena.select() == 0
    assert elo.offered == [[_key(n) for n in arena.pool()]]


def test_settle_plays_served_against_rest():
    arena, elo = _arena(picks=[_key("wrr")])
    arena.select()
    arena.settle(A, 0, Outcome.GAIN, 0.7)

    assert elo.matches == [(_key("wrr"), _key(n), 0.7) for n in arena.pool() if n != "wrr"]


def test_settle_miss_scores_zero():
    arena, elo = _arena(picks=[_key("wrr")])
    arena.select()
    arena.settle(A, 0, Outcome.MISS, 0.7)

    assert {s for *_, s in elo.matches} == {0.0}


def test_settle_without_select_plays_nothing():
    """Adversarial: fuzz_one outside the loop (calibration, replay) has no round."""
    arena, elo = _arena()
    arena.settle(A, 0, Outcome.GAIN, 1.0)

    assert elo.matches == []


def test_unsettled_round_is_dropped():
    """Adversarial: two selects, one settle -> only the second arm is charged."""
    arena, elo = _arena(picks=[_key("wrr"), _key("wfq")])
    arena.select()
    arena.select()
    arena.settle(A, 0, Outcome.GAIN, 1.0)

    assert {a for a, *_ in elo.matches} == {_key("wfq")}
    assert len(elo.matches) == len(arena.pool()) - 1


def test_seed_hint_only_when_gale_shapley_served():
    arena, _ = _arena(picks=[_key(GALE_SHAPLEY), _key("wrr")])
    arena.select()
    hint = arena.seed_hint()
    arena.settle(A, 0, Outcome.MISS, 1.0)
    arena.select()

    assert hint in (A, B)
    assert arena.seed_hint() is None


def test_seed_hint_when_auction_served():
    arena, _ = _arena(picks=[_key(AUCTION)])
    arena.select()

    assert arena.seed_hint() in (A, B)
    assert arena.seed_hint() is None  # consumed once


def test_round_cost_is_select_to_settle():
    ticks = iter([10.0, 12.5])
    arena, _ = _arena(picks=[_key("round_robin")], clock=lambda: next(ticks))
    seen = []
    spy = SimpleNamespace(name="spy", pick=lambda n: 0, record=seen.append)
    arena._arms["spy"] = spy

    arena.select()
    arena.settle(A, 0, Outcome.MISS, 1.0)

    assert seen[0].cost == pytest.approx(2.5)


def test_every_arm_is_credited_each_round():
    """Off-policy: the round reaches every arm, not just the one that served."""
    arena, _ = _arena(picks=[_key("wrr")])
    seen = []
    arena._arms["spy"] = SimpleNamespace(name="spy", pick=lambda n: 0, record=seen.append)
    arena.select()
    arena.settle(B, 2, Outcome.GAIN, 1.0)

    assert [(r.seed, r.idx, r.outcome) for r in seen] == [(B, 2, Outcome.GAIN)]


def test_elo_off_never_asks_elo():
    """Adversarial: no Elo -> the first arm serves and no match is played."""
    arena, elo = _arena(use_elo=False)
    arena.select()
    arena.settle(A, 0, Outcome.GAIN, 1.0)

    assert elo.offered == [] and elo.matches == []


# --- Elo keyspace ---------------------------------------------------------------


def test_tgt_keys_are_their_own_arena():
    assert strategy_arena(_key("wrr")) is Arena.TARGET
    assert strategy_display_name(_key("wrr")) == _key("wrr")
    assert strategy_display_name(TARGET_MATCH) == TARGET_MATCH


def test_stats_leaders_show_top_target():
    out = _arena_leaders([(_key("wfq"), 1600.0), ("bandit", 1500.0)])

    assert f"top_tgt={_key('wfq')}(1600)" in out
    assert "top_op=op_bandit" in out


# --- fuzzer wiring --------------------------------------------------------------


def _stub():
    f = Fuzzer.__new__(Fuzzer)
    f.multi_targets = list(TARGETS)
    f._active_target_idx = 0
    f.target = TARGETS[0]
    return f


def test_select_next_target_defers_to_arena():
    f = _stub()
    f._target_arena = SimpleNamespace(select=lambda: 2)

    Fuzzer._select_next_target(f)

    assert (f._active_target_idx, f.target) == (2, "t2")


def test_settle_targets_forwards_round():
    f = _stub()
    f._active_target_idx = 1
    f._last_parent_seed = A
    got = []
    f._target_arena = SimpleNamespace(settle=lambda *a: got.append(a))

    Fuzzer._settle_targets(f, True, 0.4)

    assert got == [(A, 1, Outcome.GAIN, 0.4)]


def test_settle_targets_without_arena_is_noop():
    f = _stub()
    f._target_arena = None

    Fuzzer._settle_targets(f, True, 0.4)


def test_seed_picker_takes_the_matched_seed(build):  # noqa: F811
    f = build()
    f.corpus = [A, B]
    f._target_arena = SimpleNamespace(seed_hint=lambda: B)

    assert f._seed_picker.pick_seed() == B
    assert f._seed_strategy == TARGET_MATCH


def test_stall_recovery_beats_the_hint(build):  # noqa: F811
    """Adversarial: stall recovery is a global override; the hint must not mask it."""
    f = build()
    f.corpus = [A]
    f._stall_recovery_active = True
    f._target_arena = SimpleNamespace(seed_hint=lambda: B)

    assert f._seed_picker.pick_seed() == A
    assert f._seed_strategy == "random_stall"


def test_single_target_builds_no_arena(build):  # noqa: F811
    assert build(target_arena=True, elo=True)._target_arena is None


def test_multi_target_builds_arena(build):  # noqa: F811
    f = build(target_arena=True, elo=True, multi_targets=[_TARGET, _TARGET])

    assert isinstance(f._target_arena, TargetArena)


def test_arena_needs_elo(build):  # noqa: F811
    assert build(target_arena=True, multi_targets=[_TARGET, _TARGET])._target_arena is None


def test_cli_flag(monkeypatch):
    assert _parse(monkeypatch).target_arena is False
    assert _parse(monkeypatch, "--target-arena").target_arena is True


def test_cmd_fuzz_forwards():
    assert all("target_arena" in k for k in _fuzzer_call_kwargs())


def test_hail_mary_enables_it():
    assert "target_arena" in commands._HAIL_MARY_FLAGS


def test_report_has_target_block():
    """Report splits ``tgt_`` keys into their own table, out of the operator one."""
    from fuzzer_tool.core.analyzers.analyzer_elo import BayesianEloTracker
    from fuzzer_tool.services.report import _elo_strategy_lines

    elo = BayesianEloTracker(min_matches=1)
    elo.record_strategy_match(_key("wrr"), _key(GALE_SHAPLEY), 1.0)
    lines: list[str] = []

    _elo_strategy_lines(SimpleNamespace(_use_elo=True, _elo=elo), lines)
    text = "\n".join(lines)

    assert "Target strategies (Elo):" in text
    assert "operator strategies" not in text
