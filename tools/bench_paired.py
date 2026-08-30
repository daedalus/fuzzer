#!/usr/bin/env python3
"""Paired A/B benchmark harness over a locked (target, seed) matrix.

The existing harnesses run one unseeded campaign per arm and compare
aggregate edge counts. That cannot distinguish an arm from a lucky draw:
a fuzzing campaign is a stochastic process, and the between-run variance
on a single target routinely exceeds the effect sizes these arms are
supposed to have.

This runs every arm over the *same* frozen ``(target, seed)`` cells and
analyses the result pairwise:

* per-cell outcome is a win, a loss, or a tie against the baseline arm
* significance is McNemar's exact test on the discordant pairs, which is
  the correct test for paired binary outcomes -- an unpaired Fisher test
  on aggregate totals throws away the pairing and is reported alongside
  only for comparison
* effect size is the per-cell edge delta, reported as a median with an
  interquartile range rather than a mean, since edge counts across
  different targets are not on a common scale

Usage::

    # define arms in a JSON file, or use the built-in ones
    tools/bench_paired.py run --arms baseline,cbh-reanchor --set cmplog
    tools/bench_paired.py analyse results/paired/*.json

Raw per-run JSON is written to ``results/paired/`` so an analysis can be
rerun, or a later arm compared against an earlier arm's recorded cells,
without re-executing anything.

Budget note: the default matrix is 6 targets x 20 seeds x 10k execs per
arm. That is 120 campaigns per arm. Use ``--set cmplog`` (3 targets) and
``--seeds`` to cut it down while prototyping, and say which set a reported
number came from.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_set import DEFAULT_ITERS, SEEDS, TARGET_SETS, cells  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "paired"

# ── Arms ───────────────────────────────────────────────────────────────
# An arm is a name and the extra CLI flags that define it. Keep arms
# single-variable against the baseline: an arm that changes two things
# cannot attribute its result to either.

ARMS: dict[str, list[str]] = {
    "baseline": [],
    # Ports under test. Each differs from baseline in exactly one knob.
    "qea": ["--qea"],
    "qea-elite-reset": ["--qea", "--qea-elite-reset", "8"],
    "qea-no-rotation": ["--qea", "--qea-rotation-angle", "0.0"],
    "qea-no-bias": ["--qea", "--qea-strong-bias", "0.5"],
    # Boltzmann seed energy. Both arms pass the same flag and differ only in
    # a code edit (see UNWIRED_ARMS), so they must be run in separate
    # invocations with the source swapped between them. --boltzmann without
    # --elo makes _pick_boltzmann_seed the sole seed strategy, which is what
    # keeps this single-variable: under --elo it would be one arbitrated arm
    # among several and most picks would not go through the code under test.
    "boltzmann-count": ["--boltzmann"],
    "boltzmann-cost": ["--boltzmann"],
}

# Arms that are compile-time rather than flag-driven still belong here, as
# a note, so the arm list stays the single record of what has been tested.
# cbh site re-anchoring is one: it is a `max_sites` default in
# core/mb_cbh.py with no CLI surface, because giving every internal search
# constant a flag is how the flag space stops being reviewable. To test it,
# edit _CBH_MAX_SITES and record the arm name by hand.
UNWIRED_ARMS = {
    "cbh-reanchor": "core/mb_cbh.py:_CBH_MAX_SITES = 4",
    # The energy term in SeedPicker._pick_boltzmann_seed. "cost" is the
    # shipped code; "count" is the pre-ab07835 form, restored by hand:
    #     n = max(meta.get("fuzz_count", 1), 1)
    # in place of the effective_fuzz_count call. No flag, deliberately: the
    # arm is a question about which quantity is right, not a knob to keep.
    "boltzmann-count": 'services/seed_picker.py: n = max(meta.get("fuzz_count", 1), 1)',
    "boltzmann-cost": "services/seed_picker.py: n = effective_fuzz_count(meta, mean_exec)",
}

# ── Running ────────────────────────────────────────────────────────────

_EDGES = re.compile(r"Edges discovered:\s+(\d+)")
_CORPUS = re.compile(r"Corpus:\s+(\d+)")
_CRASHES = re.compile(r"Crashes:\s+(\d+)")


def _parse(log: str) -> dict:
    def one(rx, default=0):
        m = rx.search(log)
        return int(m.group(1)) if m else default

    return {
        "edges": one(_EDGES),
        "corpus": one(_CORPUS),
        "crashes": one(_CRASHES),
        "coverage_attached": _EDGES.search(log) is not None and one(_EDGES) > 0,
    }


def run_cell(arm: str, target: str, flags: str, seed: int, iters: int, timeout: int) -> dict:
    """Run one campaign and return its parsed outcome."""
    workdir = Path(tempfile.mkdtemp(prefix=f"paired_{arm}_"))
    cmd = [
        sys.executable,
        "-m",
        "fuzzer_tool",
        "fuzz",
        target,
        "-d",
        str(workdir),
        "-c",
        "-n",
        str(iters),
        "-s",
        str(seed),
        *flags.split(),
        *ARMS[arm],
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False
        )
        log = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        log = (exc.stdout or b"").decode(errors="replace") if exc.stdout else ""
        rc = -1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    row = {
        "arm": arm,
        "target": target,
        "seed": seed,
        "iters": iters,
        "secs": round(time.time() - t0, 2),
        "rc": rc,
        **_parse(log),
    }
    return row


def cmd_run(args: argparse.Namespace) -> int:
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    arms = args.arms.split(",")
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        print(f"unknown arm(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(ARMS)}", file=sys.stderr)
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    matrix = list(cells(args.set, seeds))
    total = len(matrix) * len(arms)
    print(f"[*] {len(arms)} arms x {len(matrix)} cells = {total} campaigns @ {args.iters} execs")

    missing = sorted({t for t, _, _ in matrix if not (REPO / t).exists()})
    if missing:
        # Skipped, not scored zero: a target that failed to build is a hole
        # in the matrix, and recording it as an outcome would let a build
        # problem masquerade as an arm difference.
        print(f"[!] not built, cells skipped: {', '.join(Path(m).name for m in missing)}")
        print("    build them with tools/build_targets.sh before quoting a result")
        matrix = [c for c in matrix if c[0] not in set(missing)]
        total = len(matrix) * len(arms)
        if not matrix:
            print("[!] no targets available", file=sys.stderr)
            return 1

    done = 0
    for arm in arms:
        out = RESULTS / f"{args.set}_{arm}.json"
        # Resume from whatever is already on disk. A full matrix is hours of
        # compute and the results file used to be written once, after the last
        # cell of an arm -- so an interrupted run lost every cell it had
        # completed. Cells are keyed by (target, seed), which is the same key
        # the pairing uses, so a resumed file is indistinguishable from one
        # produced in a single pass.
        rows = []
        if out.exists() and not args.restart:
            try:
                rows = json.loads(out.read_text())
            except (OSError, json.JSONDecodeError):
                rows = []
        have = {(r["target"], r["seed"]) for r in rows}
        if have:
            print(f"[*] {arm}: resuming, {len(have)} cells already recorded in {out.name}")

        for target, flags, seed in matrix:
            done += 1
            if (target, seed) in have:
                continue
            row = run_cell(arm, target, flags, seed, args.iters, args.timeout)
            rows.append(row)
            flag = "" if row["coverage_attached"] else "  [NO COVERAGE]"
            print(
                f"  [{done:>4}/{total}] {arm:<18} {Path(target).name:<18} "
                f"seed={seed:<3} edges={row['edges']:<6} {row['secs']:>6.1f}s{flag}",
                flush=True,
            )
            # Checkpoint after every cell, via a temp file and an atomic
            # rename so an interruption mid-write cannot truncate the results.
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rows, indent=1))
            tmp.replace(out)
        print(f"[*] wrote {out} ({len(rows)} cells)")
    return 0


# ── Analysis ───────────────────────────────────────────────────────────


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts *b* and *c*.

    Under the null the discordant pairs split Binomial(b + c, 0.5). Exact
    rather than the chi-square approximation because the discordant count
    is routinely under 25 at these matrix sizes, where the approximation
    is anticonservative.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]].

    Reported only as a contrast: it treats the two arms as independent
    samples and so discards the pairing the matrix was built to create.
    Where the two disagree, McNemar is the one to believe.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0

    def prob(x):
        return math.comb(a + b, x) * math.comb(c + d, a + c - x) / math.comb(n, a + c)

    lo = max(0, a + c - (c + d))
    hi = min(a + b, a + c)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def _key(row: dict) -> tuple[str, int]:
    return row["target"], row["seed"]


def compare(base: list[dict], test: list[dict], metric: str = "edges") -> dict:
    """Pair two arms' rows by (target, seed) and score them."""
    b_by = {_key(r): r for r in base}
    t_by = {_key(r): r for r in test}
    shared = sorted(b_by.keys() & t_by.keys())

    wins = losses = ties = 0
    deltas = []
    dropped = 0
    for k in shared:
        rb, rt = b_by[k], t_by[k]
        if not (rb["coverage_attached"] and rt["coverage_attached"]):
            dropped += 1
            continue
        d = rt[metric] - rb[metric]
        deltas.append(d)
        if d > 0:
            wins += 1
        elif d < 0:
            losses += 1
        else:
            ties += 1

    n = wins + losses + ties
    return {
        "cells": n,
        "dropped_no_coverage": dropped,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mcnemar_p": _mcnemar_exact(wins, losses),
        "fisher_p": _fisher_exact(wins, ties + losses, losses, ties + wins),
        "median_delta": statistics.median(deltas) if deltas else 0,
        "iqr": (
            (
                round(statistics.quantiles(deltas, n=4)[0], 1),
                round(statistics.quantiles(deltas, n=4)[2], 1),
            )
            if len(deltas) >= 4
            else None
        ),
        "base_total": sum(b_by[k][metric] for k in shared),
        "test_total": sum(t_by[k][metric] for k in shared),
    }


def cmd_analyse(args: argparse.Namespace) -> int:
    loaded: dict[str, list[dict]] = {}
    for path in args.files:
        rows = json.loads(Path(path).read_text())
        if rows:
            loaded.setdefault(rows[0]["arm"], []).extend(rows)

    if args.baseline not in loaded:
        print(f"baseline arm '{args.baseline}' not among {list(loaded)}", file=sys.stderr)
        return 2

    base = loaded.pop(args.baseline)
    print(f"baseline: {args.baseline}  ({len(base)} cells)\n")
    hdr = f"{'arm':<20} {'cells':>5} {'W':>4} {'L':>4} {'T':>4} {'McNemar':>9} {'Fisher':>9} {'med Δ':>7}"
    print(hdr)
    print("-" * len(hdr))
    for arm, rows in sorted(loaded.items()):
        r = compare(base, rows, args.metric)
        print(
            f"{arm:<20} {r['cells']:>5} {r['wins']:>4} {r['losses']:>4} {r['ties']:>4} "
            f"{r['mcnemar_p']:>9.3g} {r['fisher_p']:>9.3g} {r['median_delta']:>+7.1f}"
        )
        if r["dropped_no_coverage"]:
            print(f"{'':<20} dropped {r['dropped_no_coverage']} cells: coverage did not attach")

    if args.by_target:
        # A pooled row is not enough to read an arm whose effect is expected on
        # some targets and absent by construction on others: cells that cannot
        # move dilute a real effect, and a target near coverage saturation
        # contributes ties that read as agreement. Break the same pairing out
        # per target so the shape of the result is visible, not just its sum.
        for arm, rows in sorted(loaded.items()):
            print(f"\nper-target: {arm} vs {args.baseline}")
            sub = f"{'target':<20} {'cells':>5} {'W':>4} {'L':>4} {'T':>4} {'McNemar':>9} {'med Δ':>7}"
            print(sub)
            print("-" * len(sub))
            targets = sorted({r["target"] for r in base} | {r["target"] for r in rows})
            for tgt in targets:
                b = [r for r in base if r["target"] == tgt]
                t = [r for r in rows if r["target"] == tgt]
                if not b or not t:
                    continue
                r = compare(b, t, args.metric)
                print(
                    f"{Path(tgt).name:<20} {r['cells']:>5} {r['wins']:>4} {r['losses']:>4} "
                    f"{r['ties']:>4} {r['mcnemar_p']:>9.3g} {r['median_delta']:>+7.1f}"
                )

    print(
        "\nMcNemar is the test to read: the matrix is paired by construction. "
        "Fisher is shown only to make the cost of discarding the pairing visible."
    )
    print(
        "A ~10-point difference in per-cell win rate needs roughly 100 paired cells "
        "to resolve; check the cells column before believing a p-value."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run arms over the locked matrix")
    r.add_argument("--arms", default="baseline", help=f"comma-separated; known: {','.join(ARMS)}")
    # Derived from TARGET_SETS rather than hardcoded: the choices tuple had
    # drifted from eval_set.py and silently rejected a set that exists.
    r.add_argument("--set", default="locked", choices=tuple(TARGET_SETS))
    r.add_argument("--seeds", default=None, help="comma-separated seed override")
    r.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    r.add_argument("--timeout", type=int, default=900, help="per-campaign timeout, seconds")
    r.add_argument(
        "--restart",
        action="store_true",
        help="discard any recorded cells for these arms and re-run the matrix from scratch",
    )
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyse", help="paired analysis of recorded runs")
    a.add_argument("files", nargs="+")
    a.add_argument("--baseline", default="baseline")
    a.add_argument("--metric", default="edges", choices=("edges", "corpus", "crashes"))
    a.add_argument(
        "--by-target",
        action="store_true",
        help="also break the pairing out per target, not just pooled",
    )
    a.set_defaults(func=cmd_analyse)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
