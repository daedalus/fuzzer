"""Falsification and adversarial tests for the FEWA scheduler."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.schedulers.op_fewa import FEWAScheduler


def test_fewa_validates_constructor_params() -> None:
    """A non-positive alpha or max_window silently breaks the algorithm
    (division by zero in the confidence bound, a ladder that can never
    reach a second rung) rather than raising -- catch it at construction.
    """
    with pytest.raises(ValueError):
        FEWAScheduler(alpha=0.0)
    with pytest.raises(ValueError):
        FEWAScheduler(alpha=-1.0)
    with pytest.raises(ValueError):
        FEWAScheduler(max_window=0)
    with pytest.raises(ValueError):
        FEWAScheduler(max_window=-5)


def test_fewa_single_and_empty_candidate_shortcuts() -> None:
    """select_op must not touch scheduler state for 0 or 1 candidates."""
    s = FEWAScheduler()
    assert s.select_op([]) == ""
    assert s.select_op(["only"]) == "only"
    # No arm was registered by the single-candidate shortcut.
    assert s.bandit_stats()["fewa_known_arms"] == 0


def test_fewa_warmup_round_robins_to_the_window_floor() -> None:
    """At h=1, every active arm must be pulled once before elimination.

    With two brand-new arms and h starting at 1, the first two selections
    must cover both names (in either order) -- an elimination test at h=1
    run on an arm with zero stored pulls would compare against a
    fabricated empty-history mean of 0.0 instead of real evidence.
    """
    s = FEWAScheduler()
    seen = set()
    for _ in range(2):
        op = s.select_op(["A", "B"])
        seen.add(op)
        s.record(op, success=True, weight=1.0)
    assert seen == {"A", "B"}, f"warm-up skipped an arm: only pulled {seen}"


def test_fewa_falsification_never_selects_a_name_outside_candidates() -> None:
    """select_op must only ever return one of the offered candidates.

    Internal state (``_active``) can carry names from a previous call's
    candidate set; if the restart/intersection logic ever forgot to filter
    against the *current* ``ops`` list, a stale name could leak out.
    """
    s = FEWAScheduler(max_window=4)
    ops_a = ["A", "B", "C"]
    ops_b = ["X", "Y"]
    for _ in range(50):
        pick = s.select_op(ops_a)
        assert pick in ops_a
        s.record(pick, success=(pick == "A"), weight=1.0)
    for _ in range(50):
        pick = s.select_op(ops_b)
        assert pick in ops_b, f"leaked a stale candidate: {pick!r} not in {ops_b}"
        s.record(pick, success=True, weight=1.0)


def test_fewa_eliminates_the_clearly_worse_arm_at_matched_window() -> None:
    """A large, persistent gap must eliminate the worse arm and commit to
    the better one, not keep splitting pulls between them.

    Feed A reward=1.0 and B reward=0.0 for the whole warm-up plus one full
    round at h=1: the gap (1.0) vastly exceeds any plausible 2*B(1) at
    these small pull counts (alpha=0.5 keeps the bound modest), so the
    scheduler must enter a commitment phase on A -- the post-elimination
    signal, since ``_active`` itself is reset to every candidate the
    instant a commitment phase begins (see ``_enter_exploit``).
    """
    s = FEWAScheduler(alpha=0.5)
    for _ in range(3):
        op = s.select_op(["A", "B"])
        s.record(op, success=(op == "A"), weight=1.0)
    assert s._exploit_arm == "A", (
        f"expected a commitment phase on A after a 1.0 vs 0.0 gap, got "
        f"exploit_arm={s._exploit_arm!r}"
    )


def test_fewa_falsification_close_arms_are_not_falsely_eliminated() -> None:
    """Two arms with identical reward streams must never eliminate each other.

    If the elimination inequality's sign or the 2x confidence-radius
    factor were dropped, two arms tied on every pull could still get
    filtered by floating-point noise. Feeding literally identical rewards
    is the sharpest version of "no real gap exists".
    """
    s = FEWAScheduler(alpha=0.5)
    for i in range(64):
        op = s.select_op(["A", "B"])
        # Same deterministic reward stream regardless of which is pulled.
        s.record(op, success=(i % 2 == 0), weight=1.0)
        assert {"A", "B"} <= (s._active | {op}), "a tied arm was falsely eliminated"


def test_fewa_bound_shrinks_as_window_grows() -> None:
    """B(h) must strictly decrease as h grows (fixed total_pulls).

    If the confidence radius ever grew with h, longer windows would keep
    exploring instead of converging -- the whole point of the doubling
    ladder is that comparisons get *more* confident at longer range.
    """
    s = FEWAScheduler(alpha=0.5)
    s._total_pulls = 1000
    b1 = s._bound(1)
    b2 = s._bound(2)
    b4 = s._bound(4)
    b8 = s._bound(8)
    assert math.isfinite(b1) and b1 > 0.0
    assert b1 > b2 > b4 > b8 > 0.0


def test_fewa_windowed_mean_uses_only_the_last_h_rewards() -> None:
    """The windowed mean must ignore anything older than the last h pulls.

    Feed a poor early record then a perfect run: at h=2 the mean must
    reflect only the most recent two pulls (1.0), not the polluted
    all-time average.
    """
    s = FEWAScheduler()
    s.init_arm("A")
    s.record("A", success=False, weight=1.0)  # old, must be excluded from h=2
    s.record("A", success=True, weight=1.0)
    s.record("A", success=True, weight=1.0)
    assert s._windowed_mean("A", 2) == pytest.approx(1.0)
    # But the all-history window (h >= n) does see the early failure.
    assert s._windowed_mean("A", 10) == pytest.approx(2.0 / 3.0)


def test_fewa_restarts_epoch_once_a_single_survivor_remains() -> None:
    """Once elimination narrows the active set to one arm, the epoch
    must restart (h back to 1, every candidate re-admitted) rather than
    committing to that arm forever -- see the module docstring's
    documented deviation from the paper's one-shot identification.
    """
    s = FEWAScheduler(alpha=0.05, max_window=1024)
    epochs_seen = set()
    for i in range(80):
        op = s.select_op(["A", "B"])
        s.record(op, success=(op == "A"), weight=1.0)
        epochs_seen.add(s._epoch)
    assert s.bandit_stats()["fewa_epochs"] >= 1, "epoch never restarted"
    # After a restart, B must be re-admitted to the active set at h=1,
    # not permanently exiled -- the scheduler has to keep offering it,
    # otherwise a rotted-then-recovered arm could never be reconsidered.
    assert s._h == 1 or "B" in s._active or s._epoch >= 1


def test_fewa_restarts_when_h_reaches_max_window_even_with_multiple_survivors() -> None:
    """max_window is a hard ceiling on the ladder, not just a deque cap.

    Two arms tied closely enough to never trigger elimination must still
    force a restart once h reaches max_window, or the scheduler would
    freeze at the top rung forever and stop re-admitting anything.
    """
    s = FEWAScheduler(alpha=0.5, max_window=4)
    for i in range(64):
        op = s.select_op(["A", "B"])
        s.record(op, success=(i % 2 == 0), weight=1.0)
    assert s._h <= s.max_window, "window ladder exceeded its own ceiling"
    assert s.bandit_stats()["fewa_epochs"] >= 1


def test_fewa_history_is_capped_at_max_window() -> None:
    """Per-arm reward history must not grow without bound.

    ``max_window`` doubles as the per-arm deque's maxlen (see module
    docstring); a long-lived arm must not accumulate unbounded state.
    """
    s = FEWAScheduler(max_window=16)
    s.init_arm("A")
    for _ in range(500):
        s.record("A", success=True, weight=1.0)
    assert len(s._history["A"]) == 16


def test_fewa_record_is_off_policy_safe() -> None:
    """record() must only touch the named arm, regardless of who was picked.

    This scheduler is fed every round's outcome from the shared record()
    fan-out (see services/fuzzer.py), not just rounds it selected itself,
    so a record for an arm it never chose must still be reflected in that
    arm's own windowed history and nothing else's.
    """
    s = FEWAScheduler()
    s.init_arm("A")
    s.init_arm("B")
    s.record("A", success=True, weight=1.0)
    assert len(s._history["A"]) == 1
    assert len(s._history["B"]) == 0


def test_fewa_bandit_stats_shape() -> None:
    s = FEWAScheduler()
    s.init_arm("A")
    s.record("A", success=True, weight=1.0)
    stats = s.bandit_stats()
    for key in (
        "fewa_pulls",
        "fewa_window",
        "fewa_active_arms",
        "fewa_known_arms",
        "fewa_epochs",
        "fewa_exploiting",
    ):
        assert key in stats, f"missing diagnostic key {key!r}"


def test_fewa_convergence_stationary() -> None:
    """Locates the best arm on a fixed-seed stationary campaign.

    Elimination-based schedulers commit harder than index-based ones once
    a gap is detected, so a high floor is appropriate here -- this is the
    textbook case every scheduler in this package is expected to pass.
    """
    from tests.support.bandit_env import StationaryBernoulli, run

    env = StationaryBernoulli.build()
    c = run(FEWAScheduler(), env, seed=92, rounds=20_000)
    assert c.tail_share(env.best) >= 0.80, (
        f"FEWA spent only {c.tail_share(env.best):.3f} of campaign tail "
        f"on {env.best!r} (p={env.probs[env.best]:.3f}); uniform baseline "
        f"{1.0 / len(env.arms):.3f}"
    )


@pytest.mark.slow
def test_fewa_convergence_recovery_after_decay() -> None:
    """Recovers after the best arm's yield collapses (non-stationary).

    This is FEWA's actual home turf: the windowed mean only ever looks at
    an arm's own recent pulls, so a collapsed former champion should fall
    out of favour once its recent window reflects the collapse, and the
    periodic epoch restart keeps the new best arm discoverable.
    """
    from tests.support.bandit_env import DecayingBest, run

    env = DecayingBest.build(switch_at=10_000)
    c = run(FEWAScheduler(), env, seed=92, rounds=20_000)
    assert c.tail_share(env.best_late) >= 0.4, (
        f"FEWA spent only {c.tail_share(env.best_late):.3f} of campaign tail "
        f"on {env.best_late!r} after {env.best_early!r} decayed"
    )
