#!/usr/bin/env python3
"""Floyd cycle-detection audit for RandPool's backing generator.

RandPool (core/rand_pool.py) delegates all randomness to numpy's default
Generator, PCG64. PCG64's actual period is 2**128 — nothing short of
running the universe again can exhaustively verify that period, and this
script does not try to. What it *does* check is the real bug class this
project can actually introduce: a generator that degenerates into a
*short* repeating state cycle, which would happen if e.g. a reseed
silently no-ops, two pool instances end up sharing effective state after
a fork, or someone swaps in a weaker generator down the line. Floyd's
algorithm (core/cycle_detect.py) finds that in O(1) extra memory instead
of hashing and storing every state seen, which matters here because
"short" for this audit still means walking millions of draws before
concluding "no cycle" — see module docstring in cycle_detect.py for the
general tradeoff.

The "state" being walked is PCG64's actual internal state
(``rng.bit_generator.state``, a plain nested dict of Python ints), not
the drawn values — the drawn uint32 stream alone doesn't determine the
next draw, so it can't be used as Floyd's state directly. Re-assigning
``bit_generator.state`` before each draw makes the per-state step a pure
function, which is what Floyd's requires (it must be able to replay from
an arbitrary earlier state, several times, from different starting
points).

Usage:
    python3 tools/audit_randpool_cycle.py [--max-steps N] [--seed N]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from fuzzer_tool.core.cycle_detect import floyd_detect

DEFAULT_MAX_STEPS = 2_000_000


def state_stepper(seed):
    """Return (x0, step) where step(state) draws one uint32 from a PCG64
    initialized to `state` and returns the resulting state — a pure
    function of `state` alone, suitable for Floyd's algorithm."""
    probe = np.random.default_rng(seed)
    x0 = probe.bit_generator.state

    def step(state):
        probe.bit_generator.state = state
        probe.integers(0, 2**32, dtype=np.uint32)
        return probe.bit_generator.state

    return x0, step


def audit(seed: int, max_steps: int) -> None:
    x0, step = state_stepper(seed)
    start = time.monotonic()
    result = floyd_detect(x0, step, is_close=lambda a, b: a == b, max_steps=max_steps)
    elapsed = time.monotonic() - start

    if result is None:
        print(
            f"seed={seed}: no cycle found within {max_steps:,} draws "
            f"({elapsed:.2f}s) — expected for PCG64, nothing to report."
        )
        return

    # A hit here means the generator is degenerate — this should never
    # happen for numpy's real PCG64 and is exactly the regression this
    # script exists to catch.
    print(
        f"seed={seed}: CYCLE DETECTED — tail={result.mu} draws, "
        f"period={result.period} draws ({elapsed:.2f}s). "
        "This is not expected for PCG64; investigate the generator setup."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, action="append", default=None)
    args = parser.parse_args()
    seeds = args.seed or [0, 1, 42]
    for seed in seeds:
        audit(seed, args.max_steps)


if __name__ == "__main__":
    main()
