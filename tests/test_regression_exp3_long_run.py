"""Regression: EXP3 stays numerically sound over long campaigns.

The decay factor was stored as a plain float multiplied by ``window_decay``
on every record, and the blowup guard tested ``decay * max_relative``. A
factor common to every arm cannot change ``w_i / sum(w)``, but it did hide
relative growth from the guard: the relative weights grew until they
overflowed to inf, the mixture turned to NaN, and the roulette walk (every
``r <= NaN`` comparison false) returned ``ops[-1]`` forever. Measured with
K=20 and the default ``window_decay=0.999``: NaN at pull 691,465.

The tests below reach the same state in a few thousand records by using a
faster decay; the mechanism is identical.
"""

import math

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import Exp3Scheduler


def _drive(sched: Exp3Scheduler, records: int) -> None:
    for _ in range(records):
        sched._last_probs = {"good": 1.0}
        sched.record("good", success=True)


def test_relative_weights_stay_finite_under_decay():
    # step = gamma * (1/p) / K = 0.1 / 2 = 0.05 per record; exp(0.05)^n
    # overflows at n ~ 14,200, while 0.9^n drives decay*max_relative to 0.
    sched = Exp3Scheduler(gamma=0.1, window_decay=0.9, rng=RandPool(seed=1))
    sched.init_arm("good")
    sched.init_arm("bad")
    _drive(sched, 16_000)
    assert all(math.isfinite(w) and w > 0.0 for w in sched.weights.values())
    assert sched._max_relative <= 1e9


def test_selection_law_survives_long_run():
    sched = Exp3Scheduler(gamma=0.1, window_decay=0.9, rng=RandPool(seed=2))
    ops = ["good", "bad"]
    for op in ops:
        sched.init_arm(op)
    _drive(sched, 16_000)
    picks = [sched.select_op(ops) for _ in range(2000)]
    probs = sched.last_selection_probs()
    assert all(math.isfinite(p) for p in probs.values())
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)
    # gamma/K exploration floor for the bad arm, the rest to the good one.
    assert probs["good"] > 0.9
    assert picks.count("good") > 1700


def test_decay_underflow_does_not_break_record():
    """0.5**3000 is 0.0 as a float; the old default ``1.0 / decay`` for an
    arm record() had not seen then raised ZeroDivisionError."""
    sched = Exp3Scheduler(gamma=0.1, window_decay=0.5, rng=RandPool(seed=3))
    sched.init_arm("a")
    for _ in range(3000):
        sched._last_probs = {"a": 0.5}
        sched.record("a", success=False)
    sched._last_probs = {"new": 0.5}
    sched.record("new", success=True)
    assert math.isfinite(sched.weights["new"])
    assert sched.bandit_stats()["exp3_max_weight"] >= 0.0


def test_unseen_arm_starts_at_init_weight():
    """record() on an arm it has not seen must start from the same relative
    weight init_arm() assigns and select_op() assumes (1.0), not 1/decay."""
    sched = Exp3Scheduler(gamma=0.1, window_decay=0.5, rng=RandPool(seed=4))
    sched.init_arm("a")
    for _ in range(40):
        sched._last_probs = {"a": 1.0}
        sched.record("a", success=False)
    sched._last_probs = {"late": 1.0}
    sched.record("late", success=False)
    assert sched.weights["late"] == 1.0


def test_pathological_reward_weight_does_not_raise():
    sched = Exp3Scheduler(gamma=0.5, window_decay=1.0, rng=RandPool(seed=5))
    sched.init_arm("a")
    sched._last_probs = {"a": 1e-9}
    sched.record("a", success=True, weight=1e6)  # exp(5e14) would overflow
    assert math.isfinite(sched.weights["a"])
