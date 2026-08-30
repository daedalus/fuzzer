#!/usr/bin/env python3
"""Paired analysis of the replicated Boltzmann A/B.

Reports per target, never pooled-only: two of the six `direct_lite` targets
are no-ops by construction and the informative ones sit on different edge
scales, so a pooled McNemar lets one dilute or manufacture the other.

Also reports what the design could have resolved. A null from an underpowered
matrix and a null from a real absence of effect read identically in the test
statistic, and only the second is a finding. The bound here is computed from
the measured replicate noise rather than assumed.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

A, B = "boltzmann-count", "boltzmann-cost"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def sign_test_power(sd_cell: float, reps: int, n_cells: int, effect: float) -> float:
    """P(detect an *effect*-sized shift) for the paired sign test at 0.05.

    The per-cell comparison is a difference of two medians of `reps`
    replicates, so its noise is roughly sd_cell * sqrt(2 / reps). An effect
    is seen in a cell with probability p = P(noise < effect), and the sign
    test needs enough cells to fall the same way.
    """
    se = sd_cell * math.sqrt(2.0 / reps)
    if se <= 0:
        return 1.0
    # P(a single cell shows the true sign)
    z = effect / (se * math.sqrt(2.0))
    p = 0.5 * (1.0 + math.erf(z))
    # Two-sided sign test at 0.05: need k of n on one side.
    crit = None
    for k in range(n_cells, -1, -1):
        if mcnemar(k, n_cells - k) > 0.05:
            crit = k + 1
            break
    if crit is None or crit > n_cells:
        return 0.0
    return sum(
        math.comb(n_cells, i) * p**i * (1 - p) ** (n_cells - i)
        for i in range(crit, n_cells + 1)
    )


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/paired/boltzmann_replicated.json"
    rows = json.load(open(path))
    bad = [r for r in rows if not r["coverage_attached"]]
    if bad:
        print(f"[!] {len(bad)} runs without coverage -- excluded")
        rows = [r for r in rows if r["coverage_attached"]]

    by: dict = {}
    for r in rows:
        by.setdefault((r["target"], r["seed"]), {}).setdefault(r["arm"], []).append(r["edges"])

    all_d = []
    for target in sorted({t for t, _ in by}):
        print(f"\n{'=' * 66}\n{Path(target).name}\n{'=' * 66}")
        print(f"{'seed':>4} {'count':>18} {'cost':>18} {'med Δ':>7}")
        ds, sds = [], []
        for (t, s), arms in sorted(by.items()):
            if t != target:
                continue
            a, b = sorted(arms.get(A, [])), sorted(arms.get(B, []))
            if len(a) < 2 or len(b) < 2:
                continue
            d = statistics.median(b) - statistics.median(a)
            ds.append(d)
            sds += [statistics.stdev(a), statistics.stdev(b)]
            print(f"{s:>4} {str(a):>18} {str(b):>18} {d:>+7.1f}")

        if not ds:
            continue
        all_d += ds
        w = sum(1 for d in ds if d > 0)
        loss = sum(1 for d in ds if d < 0)
        tie = sum(1 for d in ds if d == 0)
        p = mcnemar(w, loss)
        sd = statistics.mean(sds)
        print(
            f"\n  cost wins {w}, loses {loss}, ties {tie}"
            f" | median Δ {statistics.median(ds):+.1f}"
            f" | McNemar p = {p:.3f}"
        )
        print(f"  within-cell sd {sd:.1f} edges; per-cell comparison se {sd * math.sqrt(2 / 3):.1f}")
        line = "  detectable (80% power): "
        for eff in (2, 5, 10, 20):
            pw = sign_test_power(sd, 3, len(ds), eff)
            line += f"{eff}e:{pw:.0%}  "
        print(line)

    w = sum(1 for d in all_d if d > 0)
    loss = sum(1 for d in all_d if d < 0)
    tie = sum(1 for d in all_d if d == 0)
    print(f"\n{'=' * 66}\npooled (for reference only -- scales differ)")
    print(
        f"  cost wins {w}, loses {loss}, ties {tie}"
        f" | median Δ {statistics.median(all_d):+.1f} | McNemar p = {mcnemar(w, loss):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
