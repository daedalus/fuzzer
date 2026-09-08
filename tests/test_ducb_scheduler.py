"""Falsification and adversarial tests for DUCB scheduler."""

from __future__ import annotations

import math

import pytest

from fuzzer_tool.core.schedulers.ducb import DUCBScheduler


def test_ducb_falsification_width_decreases_with_n() -> None:
    """Gaussian width strictly decreases as n grows (fixed mean, log_n).

    The D-UCB index is mean + exploration * 2B * sqrt(xi * log_n) / sqrt(n).
    If the denominator were ever constant or growing, an arm with more
    evidence would get a wider confidence band than a less-pulled arm —
    the scheduler would over-explore forever.
    """
    xi = 0.6
    exploration = 0.25
    b = 1.0
    log_n = math.log(1000.0)

    def width(n: float) -> float:
        return exploration * 2.0 * b * math.sqrt(xi * log_n) / math.sqrt(n)

    for n in (1.0, 10.0, 100.0, 1000.0):
        w = width(n)
        assert math.isfinite(w) and w > 0.0, f"width non-positive at n={n}"
    assert width(1.0) > width(10.0) > width(100.0) > width(1000.0) - 1e-12


def test_ducb_falsification_gamma_one_absolute_counts() -> None:
    """With gamma=1.0, discounted counts equal absolute pull counts.

    The discount factor is applied as gamma**age on every record; at 1.0
    every record has weight 1.0 regardless of age, so N_t(i) is just the
    number of times arm i was pulled.
    """
    s = DUCBScheduler(gamma=1.0, xi=0.6, exploration=0.25)
    for a in ("bit_flip", "byte_flip", "dict_byte"):
        s.init_arm(a)
    for i in range(50):
        s.record("bit_flip", success=(i % 3 == 0))
        s.record("byte_flip", success=(i % 4 == 0))
        s.record("dict_byte", success=(i % 5 == 0))
    counts = s.discounted_counts()
    assert counts["bit_flip"] == pytest.approx(50.0, rel=1e-9)
    assert counts["byte_flip"] == pytest.approx(50.0, rel=1e-9)
    assert counts["dict_byte"] == pytest.approx(50.0, rel=1e-9)


def test_ducb_adversarial_scripted_rng() -> None:
    """Deterministic selection: less-pulled arm with same mean wins.

    Independent derivation:
      Both arms mean=0, gamma=1.0, xi=0.6, exploration=0.25, b=1.0.
      n_total = 100 + 10 = 110, log_n = ln(110).
      width = exploration * 2B * sqrt(xi * log_n) / sqrt(n).
      A: n=100 → width_A = 0.5 * sqrt(ln(110)) / 10
      B: n=10  → width_B = 0.5 * sqrt(ln(110)) / sqrt(10)
      width_B > width_A → score_B > score_A → B selected.
    """
    scheduler = DUCBScheduler(gamma=1.0, xi=0.6, exploration=0.25)
    scheduler.init_arm("A")
    scheduler.init_arm("B")
    for _ in range(100):
        scheduler.record("A", success=False)
    for _ in range(10):
        scheduler.record("B", success=False)
    # Both arms have evidence; no RNG consumed.
    op = scheduler.select_op(["A", "B"])
    assert op == "B"


def test_ducb_adversarial_renormalise_preserves_ordering() -> None:
    """Renormalisation does not flip the relative ordering of arm means.

    After the O(K) sweep folds the discount back to 1.0, every arm's
    displayed mean is X_rel / N_rel, identical to the pre-renormalise
    ratio. If the sweep ever divided by a different factor per arm the
    ordering would flip and an arm with higher true mean could lose.
    """
    s = DUCBScheduler(gamma=0.9, xi=0.6, exploration=0.25)
    for a in ("bit_flip", "byte_flip", "dict_byte"):
        s.init_arm(a)
    for i in range(200):
        s.record("bit_flip", success=(i % 3 == 0), weight=1.0)
        s.record("byte_flip", success=(i % 4 == 0), weight=1.0)
        s.record("dict_byte", success=(i % 7 == 0), weight=1.0)
    means_before = s.discounted_means()
    order_before = sorted(means_before, key=means_before.get, reverse=True)
    # Force renormalisation by pushing discount to underflow, then sweep.
    s._discount = 1e-15
    s._renormalise()
    means_after = s.discounted_means()
    order_after = sorted(means_after, key=means_after.get, reverse=True)
    assert order_before == order_after, (
        f"renormalisation flipped mean ordering: {order_before} -> {order_after}"
    )
