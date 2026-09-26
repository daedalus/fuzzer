#!/usr/bin/env python3
"""Measure the intra-arm replicate spread of a bench_paired cell.

bench_paired pairs cells by (target, seed) and calls a cell a win or a loss
by comparing two arms' edge counts. That is only meaningful if re-running a
cell under the *same* arm reproduces its edge count. It does not: the fuzzer
carries wall-clock state (``age = now - added_at``, and now ``mean_exec``),
so a cell is not a fixed function of its seed.

This runs the same arm on the same cell N times and reports the spread, so
the A/B's discordant pairs can be read against a measured null instead of an
assumed one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_paired import run_cell  # noqa: E402
from eval_set import TARGET_SETS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="boltzmann-cost")
    ap.add_argument("--set", dest="set_name", default="direct_lite")
    ap.add_argument("--targets", default="png_read.so,jpeg_read.so,zlib_read.so,lz4_read.so")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="results/paired/noise_probe.json")
    args = ap.parse_args()

    want = set(args.targets.split(","))
    spec = {Path(t).name: (t, f) for t, f in TARGET_SETS[args.set_name]}
    chosen = [spec[n] for n in want if n in spec]
    missing = want - set(spec)
    if missing:
        print(f"[!] not in set {args.set_name}: {', '.join(sorted(missing))}", file=sys.stderr)

    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    have = {(r["target"], r["seed"], r["rep"]) for r in rows}

    total = len(chosen) * len(seeds) * args.reps
    done = 0
    for target, flags in chosen:
        for seed in seeds:
            for rep in range(args.reps):
                done += 1
                if (target, seed, rep) in have:
                    continue
                row = run_cell(args.arm, target, flags, seed, args.iters, args.timeout)
                row["rep"] = rep
                rows.append(row)
                print(
                    f"  [{done:>3}/{total}] {Path(target).name:<16} seed={seed} "
                    f"rep={rep} edges={row['edges']:<5} {row['secs']:>6.1f}s",
                    flush=True,
                )
                tmp = out.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(rows, indent=1))
                tmp.replace(out)

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'cell':<26} {'n':>2} {'edges':<22} {'min':>4} {'max':>4} {'rng':>4} {'sd':>6} {'CV':>6}")
    by = {}
    for r in rows:
        by.setdefault((Path(r["target"]).name, r["seed"]), []).append(r["edges"])
    for (name, seed), vals in sorted(by.items()):
        vals = sorted(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mean = statistics.mean(vals)
        cv = sd / mean if mean else 0.0
        rng = vals[-1] - vals[0]
        print(
            f"{name + ' s' + str(seed):<26} {len(vals):>2} {str(vals):<22} "
            f"{vals[0]:>4} {vals[-1]:>4} {rng:>4} {sd:>6.2f} {cv:>6.3f}"
        )

    # The null discordance rate: over all ordered pairs of replicates of the
    # same cell, how often do they disagree at all? Every such disagreement
    # is a discordant pair McNemar would count as evidence if the two
    # replicates had carried different arm labels.
    disc = tie = 0
    for vals in by.values():
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[i] == vals[j]:
                    tie += 1
                else:
                    disc += 1
    n = disc + tie
    if n:
        print(f"\nnull discordance over replicate pairs: {disc}/{n} = {disc / n:.1%}")
        print("(a same-arm pair that McNemar would score as a win or a loss)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
