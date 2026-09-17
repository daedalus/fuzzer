"""Tests for ``core/schedulers/op_corral.py`` -- log-barrier OMD over operators.

Three carry the module's argument and are the ones to look at first:

- ``test_baseline_estimator_is_unbiased`` -- the loss shift that makes the
  default ``baseline=True`` defensible does not bias the estimate.
- ``test_vectorised_solver_matches_the_scalar_oracle`` -- the numpy solve
  and the plain-Python one agree. The vectorisation was a 7.7x hot-path
  optimisation (819 us -> 106 us per round at 155 arms), so it needs an
  oracle, not a smoke test.
- ``test_record_is_on_policy`` -- an operator this scheduler did not draw is
  dropped rather than weighted against a probability that never applied.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_corral import MIN_PROB, CorralScheduler


def _sched(**kw) -> CorralScheduler:
    kw.setdefault("rng", RandPool(seed=1234))
    return CorralScheduler(**kw)


def _with_arms(n: int, **kw) -> tuple[CorralScheduler, list[str]]:
    s = _sched(**kw)
    arms = [f"op{i}" for i in range(n)]
    for a in arms:
        s.init_arm(a)
    return s, arms


# --------------------------------------------------------------------------
# Scheduler contract (Hard Rules 1 and 40)
# --------------------------------------------------------------------------


def test_declares_no_prior_support():
    assert CorralScheduler.supports_priors is False


def test_has_the_operator_scheduler_interface():
    s = _sched()
    for method in ("init_arm", "select_op", "record", "bandit_stats"):
        assert callable(getattr(s, method))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eta": 0.0},
        {"eta": -1.0},
        {"horizon": 2},  # <= e: one underflow would scale a rate by >= e
        {"horizon": 1},
        {"mix": -0.1},
        {"mix": 0.5},
        {"mix": 1.0},
    ],
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        _sched(**kwargs)


def test_beta_is_corrals_doubling_factor():
    assert _sched(horizon=100_000).beta == pytest.approx(math.exp(1.0 / math.log(100_000)))


# --------------------------------------------------------------------------
# Arm registration
# --------------------------------------------------------------------------


def test_init_arm_keeps_the_simplex():
    s = _sched()
    for i in range(6):
        s.init_arm(f"op{i}")
        assert float(s._pv.sum()) == pytest.approx(1.0)
    assert np.allclose(s._pv, 1.0 / 6)


def test_init_arm_is_idempotent():
    s, _ = _with_arms(3)
    before = s._pv.copy()
    s.init_arm("op1")
    assert np.array_equal(s._pv, before)


def test_arms_registered_mid_run_are_not_dropped():
    """``REGISTRY.register_mutator`` adds operators after import -- the case
    Hierarchical silently discarded."""
    s, arms = _with_arms(3)
    for i in range(50):
        op = s.select_op(arms)
        s.record(op, i % 10 == 0)
    arms.append("late_op")
    picks = {s.select_op(arms) for _ in range(400)}
    assert "late_op" in s._idx
    assert "late_op" in picks


def test_empty_candidate_list_is_not_a_crash():
    s = _sched()
    assert s.probabilities([]) == {}
    assert s.select_op([]) == ""


def test_select_op_registers_unseen_operators():
    s = _sched()
    chosen = s.select_op(["a", "b", "c"])
    assert chosen in {"a", "b", "c"}
    assert set(s._idx) == {"a", "b", "c"}


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_select_op_returns_a_candidate():
    s, arms = _with_arms(5)
    for i in range(200):
        op = s.select_op(arms)
        assert op in arms
        s.record(op, i % 7 == 0)


def test_probabilities_restrict_to_the_offered_subset():
    s, arms = _with_arms(8)
    probs = s.probabilities(arms[:3])
    assert set(probs) == set(arms[:3])
    assert sum(probs.values()) == pytest.approx(1.0)


def test_mixing_floor_guarantees_every_offered_arm_a_share():
    s, arms = _with_arms(4, mix=0.2)
    # Drive one arm down hard, then check the floor still holds.
    for _ in range(300):
        op = s.select_op(arms)
        s.record(op, op == arms[0])
    probs = s.probabilities(arms)
    assert all(q >= 0.2 / 4 for q in probs.values()), probs
    assert sum(probs.values()) == pytest.approx(1.0)


def test_mix_zero_is_the_unmixed_distribution():
    s, arms = _with_arms(4, mix=0.0)
    probs = s.probabilities(arms)
    assert probs == pytest.approx({a: float(s._pv[s._idx[a]]) for a in arms})


# --------------------------------------------------------------------------
# On-policy contract
# --------------------------------------------------------------------------


def test_record_is_on_policy():
    """Only the arm this scheduler drew may be credited.

    ``Fuzzer._record_outcome`` gates the fan-out on ``selector == "corral"``,
    but the scheduler must not rely on that: dividing another scheduler's
    outcome by a probability that never applied to it is worse than dropping
    the round.
    """
    s, arms = _with_arms(3)
    # Warm the baseline first: see
    # test_a_first_round_success_carries_no_information for why an
    # unwarmed success is a legitimate no-op and would hide the assertion.
    for _ in range(5):
        s.record(s.select_op(arms), False)

    drawn = s.select_op(arms)
    other = next(a for a in arms if a != drawn)
    before = s._pv.copy()

    s.record(other, True, weight=1.0)
    assert np.array_equal(s._pv, before)
    assert s.bandit_stats()["orphan_records"] == 1

    s.record(drawn, True, weight=1.0)
    assert not np.array_equal(s._pv, before)


def test_a_second_reward_for_the_same_draw_is_dropped():
    s, arms = _with_arms(3)
    drawn = s.select_op(arms)
    s.record(drawn, True, weight=1.0)
    after_first = s._pv.copy()
    s.record(drawn, True, weight=1.0)
    assert np.array_equal(s._pv, after_first)
    assert s.bandit_stats()["orphan_records"] == 1


def test_record_without_a_draw_is_dropped():
    s, _ = _with_arms(3)
    s.record("op0", True, weight=1.0)
    assert s.bandit_stats()["rounds"] == 0
    assert s.bandit_stats()["orphan_records"] == 1


def test_several_draws_in_one_round_each_keep_their_own_probability():
    """An operator stack draws more than once before any reward arrives, so
    the pending probability cannot collapse to a single 'last pick'."""
    s, arms = _with_arms(6)
    first = s.select_op(arms)
    second = next(s.select_op(arms) for _ in range(1) if True)
    while second == first:
        second = s.select_op(arms)
    assert first in s._pending and second in s._pending
    s.record(second, True, weight=1.0)
    s.record(first, False)
    assert s.bandit_stats()["rounds"] == 2
    assert s.bandit_stats()["orphan_records"] == 0


def test_unconsumed_draws_do_not_grow_without_bound():
    s = _sched()
    arms = [f"op{i}" for i in range(600)]
    for _ in range(600):
        s.select_op(arms)
    stats = s.bandit_stats()
    assert len(s._pending) <= 256
    assert stats["expired_draws"] > 0


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------


def test_baseline_estimator_is_unbiased():
    """E[est_j] == the true loss for every arm, baseline or not.

    Asserted on the formula rather than through ``record``, because what is
    claimed is a property of the arithmetic: the shift cancels in
    expectation, so it cuts variance without introducing bias. If this
    fails, the justification for defaulting ``baseline=True`` is gone.
    """
    rng = RandPool(seed=7)
    probs = {"a": 0.7, "b": 0.3}
    true_loss = {"a": 0.9, "b": 0.4}
    b = 0.85  # stand-in for the running mean

    totals = {"a": 0.0, "b": 0.0}
    draws = 200_000
    for _ in range(draws):
        played = "a" if rng.random() <= probs["a"] else "b"
        for arm in totals:
            est = b + ((true_loss[arm] - b) / probs[arm] if arm == played else 0.0)
            totals[arm] += est

    for arm, total in totals.items():
        assert total / draws == pytest.approx(true_loss[arm], abs=0.02)


def test_a_productive_operator_gains_probability():
    s, arms = _with_arms(4)
    good = arms[0]
    for _ in range(400):
        op = s.select_op(arms)
        s.record(op, op == good, weight=1.0)
    probs = s.probabilities(arms)
    assert probs[good] == max(probs.values())
    assert probs[good] > 2.0 * min(probs.values())


def test_concentrates_at_fuzzing_realistic_rates():
    """10% against 1% -- the regime where the loss shift matters.

    ``core/schedulers/op_consolidated.py`` records that the Elo fan-out gave a
    ten-times-more-productive arm only 54% of the picks. This is the bar
    that motivated the family.
    """
    rng = RandPool(seed=99)
    s = CorralScheduler(rng=rng)
    arms = ["good", "bad"]
    rates = {"good": 0.10, "bad": 0.01}
    for _ in range(4000):
        op = s.select_op(arms)
        s.record(op, rng.random() < rates[op], weight=1.0)
    assert s.probabilities(arms)["good"] > 0.70


def test_a_first_round_success_carries_no_information():
    """An unwarmed success is a legitimate no-op, and it should stay one.

    On round one ``_loss_count`` is 0, so the baseline ``b`` is 0. A success
    gives ``loss = 0``, hence ``est_j = 0 + (0 - 0)/p = 0`` for the drawn arm
    and ``0`` for every other -- a uniform estimate vector, which the
    log-barrier step correctly leaves alone. There is no *relative*
    information in "the first thing I tried worked" until something else has
    been tried. Pinned because a future change that makes round one move the
    distribution is making up a gradient it does not have.
    """
    s, arms = _with_arms(3)
    before = s._pv.copy()
    s.record(s.select_op(arms), True, weight=1.0)
    assert np.array_equal(s._pv, before)
    assert s.bandit_stats()["rounds"] == 1  # it counted; it just did not move


def test_equal_operators_produce_spurious_concentration():
    """Known-bad behaviour, pinned so a fix shows up here as a failure.

    With genuinely equal arms the distribution should stay flat, and it does
    not: the importance-weighted estimate has variance ``~1/p``, and at the
    default eta each lucky round moves the winner far enough that it gets
    sampled more, which compounds. Measured over 20 seeds, 3000 rounds, all
    arms at 3%: normalised entropy 4 arms 0.300/0.547/0.926 (min/median/max)
    and 12 arms 0.559/0.672/0.836. At eta=0.3 the same runs give 0.490/0.828
    and 0.694/0.889, so it is the learning rate, not the estimator alone.

    This costs no regret when the arms really are equal -- but it is the same
    mechanism as the lock-in measured in ``core/schedulers/op_corral.py``, and
    it is why the scheduler is Elo-only rather than a fallback selector.
    Asserted as an upper bound on entropy in the spirit of the convergence
    harness's STUCK set: information the operator needs, not an aspiration.
    """
    rng = RandPool(seed=5)
    s = CorralScheduler(rng=rng)
    arms = [f"op{i}" for i in range(4)]
    for _ in range(3000):
        op = s.select_op(arms)
        s.record(op, rng.random() < 0.03, weight=1.0)
    entropy = s.bandit_stats()["entropy"]
    assert entropy < 0.95, f"entropy {entropy:.3f}: spurious concentration gone -- update the docs"


def test_failure_is_zero_reward_whatever_the_weight():
    s, arms = _with_arms(3)
    op = s.select_op(arms)
    s.record(op, False, weight=1.0)
    hi = s._pv.copy()

    s2, arms2 = _with_arms(3)
    op2 = s2.select_op(arms2)
    s2.record(op2, False, weight=0.0)
    assert np.allclose(hi, s2._pv)


def test_reward_weight_is_clamped():
    s, arms = _with_arms(3)
    op = s.select_op(arms)
    s.record(op, True, weight=17.0)
    assert s.bandit_stats()["mean_loss"] == pytest.approx(0.0)
    assert float(s._pv.sum()) == pytest.approx(1.0)


def test_textbook_estimator_punishes_whoever_played():
    """Falsifies the alternative, so the default is a measured choice.

    With ``baseline=False`` the estimate is ``loss / p`` and loss is ~1 every
    round, so the arm that just played takes the hit. That is churn, not
    learning -- the same pathology as the Elo fan-out preferring whoever
    played least recently.
    """
    s, arms = _with_arms(4, baseline=False, mix=0.0)
    op = s.select_op(arms)
    before = s.probabilities(arms)[op]
    s.record(op, False)
    assert s.probabilities(arms)[op] < before


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------


def _scalar_omd(p: dict[str, float], eta: dict[str, float], est: dict[str, float]):
    """Plain-Python bisection oracle for the log-barrier normaliser.

    Deliberately the slow, obvious implementation: bracket the root of
    ``sum_i 1/(1/p_i + eta_i (est_i - lam)) - 1`` and halve 200 times. The
    shipped solver is numpy plus safeguarded Newton, which is where a bug
    would hide.
    """
    arms = list(p)
    inv = {a: 1.0 / p[a] for a in arms}
    hi = min(est[a] + inv[a] / eta[a] for a in arms) - 1e-12
    lo = min(est[a] for a in arms) - max(inv[a] / eta[a] for a in arms) - 1.0

    def total(lam):
        s = 0.0
        for a in arms:
            d = inv[a] + eta[a] * (est[a] - lam)
            if d <= 0.0:
                return math.inf
            s += 1.0 / d
        return s

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) > 1.0:
            hi = mid
        else:
            lo = mid
    lam = 0.5 * (lo + hi)
    out = {}
    for a in arms:
        d = inv[a] + eta[a] * (est[a] - lam)
        out[a] = max(MIN_PROB, 1.0 / d) if d > 0.0 else MIN_PROB
    norm = sum(out.values())
    return {a: q / norm for a, q in out.items()}


@pytest.mark.parametrize("n_arms", [2, 5, 40])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_vectorised_solver_matches_the_scalar_oracle(n_arms, seed):
    s, arms = _with_arms(n_arms, rng=RandPool(seed))
    rng = RandPool(seed + 100)

    for _ in range(60):
        op = s.select_op(arms)
        p_before = {a: float(s._pv[s._idx[a]]) for a in arms}
        eta_before = {a: float(s._etav[s._idx[a]]) for a in arms}
        p_drawn = s._pending[op]

        success = rng.random() < 0.2
        b = (
            (s._loss_sum / s._loss_count)
            if (s.baseline and s._loss_count)
            else 0.0  # matches record()'s baseline
        )
        loss = 0.0 if success else 1.0
        est = dict.fromkeys(arms, b)
        est[op] = b + (loss - b) / p_drawn
        expected = _scalar_omd(p_before, eta_before, est)

        s.record(op, success, weight=1.0)
        got = {a: float(s._pv[s._idx[a]]) for a in arms}
        for a in arms:
            assert got[a] == pytest.approx(expected[a], rel=1e-7, abs=1e-12)


def test_probabilities_stay_a_distribution_over_a_long_run():
    rng = RandPool(seed=11)
    s, arms = _with_arms(6, eta=1.0)
    for _ in range(2000):
        op = s.select_op(arms)
        s.record(op, rng.random() < 0.05, weight=1.0)
        assert float(s._pv.sum()) == pytest.approx(1.0)
        assert float(s._pv.min()) >= MIN_PROB


def test_no_arm_is_driven_to_zero():
    rng = RandPool(seed=13)
    s, arms = _with_arms(4, mix=0.0)
    for _ in range(3000):
        op = s.select_op(arms)
        s.record(op, op == arms[0], weight=1.0)
    assert float(s._pv.min()) > 0.0
    assert s.bandit_stats()["rate_increases"] > 0
    assert rng.random() >= 0.0  # keep the pool touched; no assertion intended


def test_the_doubling_trick_only_raises_the_starved_arms_rate():
    s, arms = _with_arms(4, mix=0.0)
    for _ in range(600):
        op = s.select_op(arms)
        s.record(op, op == arms[0], weight=1.0)
    winner_eta = float(s._etav[s._idx[arms[0]]])
    starved_eta = max(float(s._etav[s._idx[a]]) for a in arms[1:])
    assert starved_eta > winner_eta


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_bandit_stats_reports_over_the_live_ballot_only():
    """Regression: entropy must describe the arms that could be drawn.

    ~155 operators are registered but a round offers the handful that passed
    the sniffers. Averaging in arms that cannot be drawn pins the entropy
    near 1.0 and makes the one diagnostic this scheduler needs a statement
    about dead weight instead of about its own preference.
    """
    s, arms = _with_arms(20, eta=1.0)
    live = arms[:2]
    for _ in range(800):
        op = s.select_op(live)
        s.record(op, op == live[0], weight=1.0)

    stats = s.bandit_stats()
    assert stats["arms"] == 2
    assert stats["registered"] == 20
    assert set(stats["probs"]) == set(live)
    assert sum(stats["probs"].values()) == pytest.approx(1.0)
    assert stats["entropy"] < 0.9
    assert stats["concentration"] > 0.5


def test_bandit_stats_entropy_is_zero_for_a_single_arm():
    s, _ = _with_arms(1)
    s.select_op(["op0"])
    assert s.bandit_stats()["entropy"] == 0.0


def test_bandit_stats_counts_pulls_and_wins():
    s, arms = _with_arms(3)
    wins = 0
    for i in range(90):
        op = s.select_op(arms)
        success = i % 3 == 0
        wins += 1 if success else 0
        s.record(op, success, weight=1.0)
    stats = s.bandit_stats()
    assert sum(stats["pulls"].values()) == 90
    assert sum(stats["wins"].values()) == pytest.approx(wins)
    assert stats["rounds"] == 90
    assert stats["mean_loss"] == pytest.approx(1.0 - wins / 90)


def test_bandit_stats_is_safe_before_any_round():
    stats = _sched().bandit_stats()
    assert stats["rounds"] == 0
    assert stats["arms"] == 0
    assert stats["probs"] == {}
    assert stats["entropy"] == 0.0
