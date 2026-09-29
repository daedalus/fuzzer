#!/usr/bin/env python3
"""Benchmark: non-adaptive group testing vs level-order binary splitting.

Synthetic OR oracle with d random defectives among n items. Reports mean
oracle calls (cost) and sequential rounds (latency) over --trials seeds.
Control (Hard Rule 46): both methods must be exact on every trial.
"""

from __future__ import annotations

import argparse
import random
import statistics

from fuzzer_tool.core import group_testing as gt

_GRID_N = (256, 4096)
_GRID_D = (1, 2, 4, 8, 16)


def _trial(n: int, d: int, seed: int) -> tuple[gt.Result, gt.Result]:
    rng = random.Random(seed)
    defective = set(rng.sample(range(n), d))

    def oracle(pool: frozenset[int]) -> bool:
        return bool(pool & defective)

    a = gt.identify(n, oracle, d=d, rng=random.Random(seed + 1))
    b = gt.split_search(n, oracle)
    if a.defective != defective or b.defective != defective:
        raise SystemExit(f"inexact result n={n} d={d} seed={seed}: oracle control failed")
    return a, b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()

    print(
        f"{'n':>6} {'d':>3} | {'GT tests':>9} {'GT rnds':>8} | {'split tests':>11} {'split rnds':>10}"
    )
    for n in _GRID_N:
        for d in _GRID_D:
            runs = [_trial(n, d, s) for s in range(args.trials)]
            gt_t = statistics.mean(a.tests for a, _ in runs)
            gt_r = statistics.mean(a.rounds for a, _ in runs)
            sp_t = statistics.mean(b.tests for _, b in runs)
            sp_r = statistics.mean(b.rounds for _, b in runs)
            print(f"{n:>6} {d:>3} | {gt_t:>9.1f} {gt_r:>8.2f} | {sp_t:>11.1f} {sp_r:>10.2f}")


if __name__ == "__main__":
    main()
