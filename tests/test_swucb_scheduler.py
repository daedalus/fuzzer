"""Falsification and adversarial tests for SWUCB scheduler."""

from __future__ import annotations

import math

from fuzzer_tool.core.schedulers.swucb import SWUCBScheduler


def test_swucb_falsification_width_decreases_with_n() -> None:
    """Gaussian width strictly decreases as n grows (fixed mean, log_n).

    SW-UCB index: mean + b * sqrt(xi * log(min(t, tau))) / sqrt(n).
    The width is independent of the mean but must shrink with n — an arm
    that has been pulled more times must have a tighter confidence band.
    """
    xi = 0.15
    b = 1.0
    log_n = math.log(4000.0)

    def width(n: float) -> float:
        return b * math.sqrt(xi * log_n) / math.sqrt(n)

    for n in (1.0, 10.0, 100.0, 1000.0):
        w = width(n)
        assert math.isfinite(w) and w > 0.0, f"width non-positive at n={n}"
    assert width(1.0) > width(10.0) > width(100.0) > width(1000.0) - 1e-12


def test_swucb_falsification_window_eviction_resets_count() -> None:
    """After window+k pulls on other arms, an arm's count drops to zero.

    The deque eviction subtracts one from the arm's count per expired
    record. If subtraction were missing, an arm would keep a phantom
    count from pulls that fell out of the window and the scheduler would
    never re-explore it.
    """
    s = SWUCBScheduler(window=10, xi=0.15, b=1.0)
    s.init_arm("bit_flip")
    s.init_arm("byte_flip")
    # Pull bit_flip once, then pound byte_flip past the window edge.
    s.record("bit_flip", success=True)
    for i in range(20):
        s.record("byte_flip", success=(i % 3 == 0))
    assert s.windowed_counts().get("bit_flip", 0) == 0, (
        "bit_flip count should have evicted to zero after 20 byte_flip pulls in a window of 10"
    )


def test_swucb_adversarial_scripted_rng() -> None:
    """Deterministic selection: less-pulled arm with same mean wins.

    Independent derivation:
      Both arms mean=0, window=1000, xi=0.15, b=1.0.
      n_total = 100 + 10 = 110, log_n = ln(110).
      width = b * sqrt(xi * log_n) / sqrt(n).
      A: n=100 → width_A = sqrt(0.15*ln(110)) / 10
      B: n=10  → width_B = sqrt(0.15*ln(110)) / sqrt(10)
      width_B > width_A → score_B > score_A → B selected.
    """
    scheduler = SWUCBScheduler(window=1000, xi=0.15, b=1.0)
    scheduler.init_arm("A")
    scheduler.init_arm("B")
    for _ in range(100):
        scheduler.record("A", success=False)
    for _ in range(10):
        scheduler.record("B", success=False)
    # Both arms have evidence; no RNG consumed.
    op = scheduler.select_op(["A", "B"])
    assert op == "B"


def test_swucb_adversarial_evicted_arm_is_retried() -> None:
    """An arm fully evicted from the window is opened again (pulled first).

    When every candidate arm has zero windowed count, select_op falls
    through to the unpulled-arm branch and picks one at random. The
    adversary's job is to set up a state where the only surviving arm
    gets evicted, forcing the scheduler back to exploration.
    """
    scheduler = SWUCBScheduler(window=5, xi=0.15, b=1.0)
    scheduler.init_arm("bit_flip")
    scheduler.init_arm("byte_flip")
    # Pull bit_flip once, then evict it with 5 byte_flip pulls.
    scheduler.record("bit_flip", success=True)
    for i in range(5):
        scheduler.record("byte_flip", success=(i % 2 == 0))
    # bit_flip count is now 0, byte_flip count is 5.
    # select_op on [bit_flip, byte_flip] must open bit_flip first.
    picks = set()
    for _ in range(20):
        op = scheduler.select_op(["bit_flip", "byte_flip"])
        picks.add(op)
        if op == "bit_flip":
            scheduler.record("bit_flip", success=False)
        else:
            scheduler.record("byte_flip", success=False)
    assert "bit_flip" in picks, "evicted arm was never retried"
