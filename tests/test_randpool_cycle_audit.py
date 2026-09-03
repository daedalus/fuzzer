"""Regression coverage for the RandPool cycle audit (tools/audit_randpool_cycle.py).

Two things need checking, and they cut in opposite directions:

1. The real generator (numpy PCG64, what RandPool actually uses) must
   NOT be flagged within a reasonable bound — a false positive here
   would mean the audit is broken, not that PCG64 is.
2. The audit technique must actually be *capable* of catching the bug
   class it exists for. Since we can't wait out PCG64's real period to
   prove a true positive, this uses the same state-stepper approach
   against a deliberately tiny-period generator (a small-modulus LCG,
   not RandPool) standing in for "a broken/degenerate generator got
   wired in somehow" — if the detector can't catch a maximally obvious
   short cycle, it can't be trusted to catch a subtle one either.
"""

import numpy as np
import pytest

from fuzzer_tool.core.cycle_detect import floyd_detect


def test_pcg64_no_short_cycle():
    """RandPool's actual backing generator: no cycle within a bound that
    would already be a very bad sign if hit (real period is 2**128).
    """
    probe = np.random.default_rng(12345)
    x0 = probe.bit_generator.state

    def step(state):
        probe.bit_generator.state = state
        probe.integers(0, 2**32, dtype=np.uint32)
        return probe.bit_generator.state

    result = floyd_detect(x0, step, is_close=lambda a, b: a == b, max_steps=200_000)
    assert result is None, (
        f"PCG64 state cycled within 200k draws (mu={result.mu}, "
        f"period={result.period}) — this would mean the generator is "
        "badly broken, not a false positive to wave away."
    )


@pytest.mark.parametrize("modulus", [17, 100, 251])
def test_detector_catches_a_deliberately_weak_generator(modulus):
    """Same technique (Floyd's over a state-stepper), applied to a tiny
    LCG standing in for 'the real generator got swapped for something
    broken'. Proves the detector fires when it should, not just that it
    stays quiet on the healthy case above.
    """
    a, c = 5, 3  # arbitrary small LCG multiplier/increment

    def step(state):
        return (a * state + c) % modulus

    result = floyd_detect(0, step, is_close=lambda x, y: x == y, max_steps=10_000)
    assert result is not None
    assert result.period <= modulus
