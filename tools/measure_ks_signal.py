#!/usr/bin/env python3
"""Measure whether KS tracks a different signal than Wasserstein on real corpora.

Wasserstein (L1) integrates the CDF difference across the whole hit-count axis;
KS (Linf) is the single largest CDF gap.  Both are computed for free by
``EdgeTracker._cdf_walk`` on every call, but only Wasserstein was being
returned -- KS was discarded at ``_wasserstein_vs_aggregate``.

This script builds a synthetic 30-60 seed corpus (the size the JS-divergence
comment in edge_tracker.py used) and reports the correlation of each metric
with loopiness and edge count.  If KS is redundant with Wasserstein on real
corpora, do not wire it into ``compute_hitcount_diversity_weight``; if it
tracks a genuinely different signal, that wiring is the next step.

Usage::

    python tools/measure_ks_signal.py [--seeds 45] [--seed 7]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzzer_tool.core.edge_tracker import EdgeTracker


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (sx * sy) ** 0.5


def build_corpus(n_seeds: int, seed: int) -> EdgeTracker:
    """A tracker with loud (loop-heavy) and quiet seeds, like the ground tests."""
    rng = random.Random(seed)
    locs = [rng.getrandbits(16) for _ in range(400)]
    et = EdgeTracker()
    for i in range(n_seeds):
        key = f"s{i}"
        edges = {(rng.choice(locs) ^ rng.choice(locs)) | 1 for _ in range(rng.randint(20, 120))}
        loud = i % 5 == 0
        hits = {e: (rng.choice([200, 400, 800]) if loud else rng.choice([1, 2, 3])) for e in edges}
        et.record_edges(key, edges, hit_counts=hits)
    return et


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=45, help="number of synthetic seeds")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed")
    args = ap.parse_args()

    et = build_corpus(args.seeds, args.seed)
    keys = [f"s{i}" for i in range(args.seeds)]

    wass = [et._aggregate_norms(et.seed_hit_counts[k])[0] for k in keys]
    ks = [et.ks_vs_aggregate(et.seed_hit_counts[k]) for k in keys]
    loopiness = [
        max(et.seed_hit_counts[k].values()) if et.seed_hit_counts[k] else 0.0 for k in keys
    ]
    edge_count = [len(et.seed_hit_counts[k]) for k in keys]

    print(f"corpus: {args.seeds} seeds, seed={args.seed}")
    print(f"{'metric':<12} {'vs loopiness':>14} {'vs edge_count':>14}")
    print("-" * 44)
    for name, values in (("Wasserstein", wass), ("KS", ks)):
        print(
            f"{name:<12} {_pearson(values, loopiness):>14.4f} {_pearson(values, edge_count):>14.4f}"
        )

    print()
    print("Wasserstein vs KS correlation:", f"{_pearson(wass, ks):.4f}")
    if abs(_pearson(wass, ks)) > 0.95:
        print("RESULT: KS is redundant with Wasserstein on this corpus -- do not wire")
        print("        it into compute_hitcount_diversity_weight yet.")
    else:
        print("RESULT: KS tracks a different signal than Wasserstein -- wiring it into")
        print("        compute_hitcount_diversity_weight is the next step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
