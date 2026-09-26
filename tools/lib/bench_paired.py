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
    tools/lib/bench_paired.py run --arms baseline,cbh-reanchor --set cmplog
    tools/lib/bench_paired.py analyse results/paired/*.json

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
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_lock  # noqa: E402
from eval_set import DEFAULT_ITERS, SEEDS, TARGET_SETS, cells  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
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
    # GARCH volatility. Single-variable against `baseline`: the model only
    # feeds the regime detector's new branch, nothing else changes.
    "garch": ["--garch"],
    # Continuum flux ranking. Its baseline is `invasion`, not `baseline`:
    # the flux map only reaches invasion_select, so measuring it against a
    # run with no invasion arm at all would attribute invasion's own effect
    # to the continuum.
    "invasion": ["--mc-bandit", "--invasion"],
    "continuum": ["--mc-bandit", "--invasion", "--continuum"],
    # Matrix arms (docs/handover/handover_matrix_schedulers_2026-09-19.md). Both
    # only mean anything under Elo arbitration, so each has its own baseline with
    # --elo on and nothing else: measuring against plain `baseline` would attribute
    # the arbiter's effect to the arm. Neither has a threshold yet -- the maintainer
    # freezes one BEFORE the first run (P3-3, P3-4).
    "elo": ["--elo", "--mc-bandit"],
    "elo-seed-residual": ["--elo", "--mc-bandit", "--seed-residual"],
    "elo-op-credit": ["--elo", "--mc-bandit", "--op-credit"],
    # The reward-shaping form of the same class arithmetic, without the selector
    # change: --shaped-reward rescales the reward EVERY arm reads, so this pairs
    # against "elo" and isolates the reward, which is what makes it the cheapest
    # single-variable test the edge-id handover names. Deliberately not combined
    # with --op-credit: that would move the selector and the reward at once.
    "elo-shaped-reward": ["--elo", "--mc-bandit", "--shaped-reward"],
    # The clamp on the two ways that factor collapses. Registered because it was
    # measured (docs/learnings/2026-09-20-shaped-reward-ab-result.md), so the arm
    # list stays the record of what has been tested.
    "elo-shaped-reward-floor25": [
        "--elo",
        "--mc-bandit",
        "--shaped-reward",
        "--shaped-reward-floor",
        "0.25",
    ],
    # Prices *where* a discovery landed (frontier pressure) rather than *how
    # duplicated* it was (shaped-reward, above). Pairs against "elo" for the
    # same single-variable reason; deliberately not combined with
    # --shaped-reward or --op-credit.
    "elo-continuum-reward": ["--elo", "--mc-bandit", "--continuum-reward"],
    # Generation group (docs/handover/handover_generators_2026-09-20.md G0).
    # Each pairs against the baseline named in ARM_BASELINES.
    #
    # wfc: --wfc turns on WFC chunk reordering for png/jpeg/bmp and the learned
    # wfc_reorder_learned operator for isobmff/webp/riff/gif. Only targets that
    # parse one of those formats can move: png_read and jpeg_read qualify,
    # zlib/lz4/gzip are bit-for-bit deterministic and cannot produce a
    # discordant pair (the handover's power note).
    "wfc": ["--wfc"],
    # mcts / alphabeta only mean anything under Elo arbitration, so they carry
    # `elo-lineage` as their baseline: the same --elo/--mc-bandit stack plus
    # --lineage, which both flags imply, so lineage bookkeeping is not
    # attributed to the tree policy. Compare mcts against alphabeta directly
    # (the E3 question) with `analyse --baseline elo-mcts`.
    "elo-lineage": ["--elo", "--mc-bandit", "--lineage"],
    "elo-mcts": ["--elo", "--mc-bandit", "--lineage", "--mcts"],
    "elo-alphabeta": ["--elo", "--mc-bandit", "--lineage", "--alphabeta"],
    # bootstrap runs only inside corpus minimization, and the recorded `edges`
    # is the tracker's cumulative count, which minimization does not shrink.
    # The arm can therefore only move `edges` through the corpus it leaves
    # behind; check "Bootstrap percolation removed" appears in a cell's log
    # before reading a null as evidence.
    "bootstrap": ["--bootstrap"],
    # Strata A0-A4 (docs/handover/handover_strata_schedulers_2026-09-19.md §6).
    # a0-ctl is a0 again: the control that must not reject (Rule 46). op_strata
    # is Elo-only, so A3/A4 pair against a1-elo, not a1 as §6 wrote: pairing
    # against a1 would credit --elo to the arm.
    "strata-a0": [],
    "strata-a0-ctl": [],
    "strata-a1": ["--confirm-novelty"],
    "strata-a2": ["--confirm-novelty", "--strata"],
    "strata-a1-elo": ["--confirm-novelty", "--elo", "--mc-bandit"],
    "strata-a3": ["--confirm-novelty", "--elo", "--mc-bandit", "--op-strata"],
    "strata-a4": ["--confirm-novelty", "--elo", "--mc-bandit", "--strata", "--op-strata"],
    # Standalone position policies, no arena: with no other tracker on, the
    # scheduler is select_position's only candidate. Pair each against
    # baseline (uniform offsets) and against each other.
    "pos-round-robin": ["--pos-round-robin"],
    "pos-fibonacci": ["--pos-fibonacci"],
}

STRATA_ARMS = (
    "strata-a0",
    "strata-a0-ctl",
    "strata-a1",
    "strata-a2",
    "strata-a1-elo",
    "strata-a3",
    "strata-a4",
)

# The arms added for the generation group, in the order the handover lists them.
GENERATION_ARMS = ("wfc", "elo-mcts", "elo-alphabeta", "bootstrap")

POSITION_ARMS = ("pos-round-robin", "pos-fibonacci")

# Which arm each one is paired against. `analyse --baseline` takes one name;
# this records the intended pairing so a reviewer does not have to reverse it
# from comments. Only arms whose baseline is not plain `baseline` need care,
# but every generation arm is listed so a test can hold them to one rule: the
# arm is its baseline plus added flags.
ARM_BASELINES: dict[str, str] = {
    "wfc": "baseline",
    "elo-mcts": "elo-lineage",
    "elo-alphabeta": "elo-lineage",
    "bootstrap": "baseline",
    "strata-a1": "strata-a0",
    "strata-a2": "strata-a1",
    "strata-a1-elo": "strata-a1",
    "strata-a3": "strata-a1-elo",
    "strata-a4": "strata-a1-elo",
    "pos-round-robin": "baseline",
    "pos-fibonacci": "baseline",
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
# Campaign summary prints "Avg eps:"; the report section "Avg throughput:".
_EPS = re.compile(r"(?:Avg eps|Avg throughput):\s+([\d.]+)")


def _parse(log: str) -> dict:
    def one(rx, default=0):
        m = rx.search(log)
        return int(m.group(1)) if m else default

    return {
        "edges": one(_EDGES),
        "corpus": one(_CORPUS),
        "crashes": one(_CRASHES),
        "eps": float(m.group(1)) if (m := _EPS.search(log)) else 0.0,
        "coverage_attached": _EDGES.search(log) is not None and one(_EDGES) > 0,
    }


def run_cell(
    arm: str, target: str, flags: str, seed: int, iters: int, timeout: int, rep: int = 0
) -> dict:
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
        "rep": rep,
        "iters": iters,
        "secs": round(time.time() - t0, 2),
        "rc": rc,
        **_parse(log),
    }
    return row


def cmd_run(args: argparse.Namespace) -> int:
    # Refuse to share the machine with another campaign when asked to:
    # contention perturbs mean_exec, which the Boltzmann arm reads.
    lock = bench_lock.engage(getattr(args, "lock_single_thread", False))
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    arms = args.arms.split(",")
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        print(f"unknown arm(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"known: {', '.join(ARMS)}", file=sys.stderr)
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    matrix = list(cells(args.set, seeds))
    if args.targets:
        want = set(args.targets.split(","))
        matrix = [c for c in matrix if Path(c[0]).name in want]
        unknown = want - {Path(c[0]).name for c in list(cells(args.set, seeds))}
        if unknown:
            print(f"[!] not in set {args.set}: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
    total = len(matrix) * len(arms) * args.reps
    print(
        f"[*] {len(arms)} arms x {len(matrix)} cells x {args.reps} reps = "
        f"{total} campaigns @ {args.iters} execs"
    )

    missing = sorted({t for t, _, _ in matrix if not (REPO / t).exists()})
    if missing:
        # Skipped, not scored zero: a target that failed to build is a hole
        # in the matrix, and recording it as an outcome would let a build
        # problem masquerade as an arm difference.
        print(f"[!] not built, cells skipped: {', '.join(Path(m).name for m in missing)}")
        print("    build them with tools/build_targets.sh before quoting a result")
        matrix = [c for c in matrix if c[0] not in set(missing)]
        total = len(matrix) * len(arms) * args.reps
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
        have = {(r["target"], r["seed"], r.get("rep", 0)) for r in rows}
        if have:
            print(f"[*] {arm}: resuming, {len(have)} cells already recorded in {out.name}")

        for target, flags, seed in matrix:
            for rep in range(args.reps):
                done += 1
                if (target, seed, rep) in have:
                    continue
                row = run_cell(arm, target, flags, seed, args.iters, args.timeout, rep)
                rows.append(row)
                flag = "" if row["coverage_attached"] else "  [NO COVERAGE]"
                print(
                    f"  [{done:>4}/{total}] {arm:<18} {Path(target).name:<18} "
                    f"seed={seed:<3} rep={rep} edges={row['edges']:<6} "
                    f"{row['secs']:>6.1f}s{flag}",
                    flush=True,
                )
                # Checkpoint after every cell, via a temp file and an atomic
                # rename so an interruption mid-write cannot truncate the
                # results.
                tmp = out.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(rows, indent=1))
                tmp.replace(out)
        print(f"[*] wrote {out} ({len(rows)} cells)")
    if lock:
        lock.release()
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


def _wilcoxon_signed_rank(deltas: list[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired deltas.

    McNemar binarises the same deltas (sign only) for the binary outcome; this
    uses the magnitude rank instead, giving more power at the same cell count
    because it is the right test for a paired continuous measurement. Zeros
    carry no sign information and are dropped before ranking.

    Returns ``(statistic, p_value)`` where *statistic* is the smaller of the
    two rank sums (W- by convention). Exact permutation distribution for
    n <= 20, normal approximation with continuity correction above.
    """
    ranked = sorted((abs(d), i) for i, d in enumerate(deltas) if d != 0.0)
    if not ranked:
        return 0.0, 1.0

    # Average ranks for tied absolute values so the null is exact under H0.
    ranks = [0.0] * len(ranked)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    signs = [1.0 if deltas[idx] > 0 else -1.0 for _, idx in ranked]
    w_pos = sum(r for r, s in zip(ranks, signs, strict=False) if s > 0)
    w_neg = sum(r for r, s in zip(ranks, signs, strict=False) if s < 0)
    stat = min(w_pos, w_neg)
    total = sum(ranks)

    n = len(ranked)
    if n <= 20:
        # Enumerate all 2^n sign assignments; W+ under the null is the sum of
        # a random subset of the ranks, so every assignment is equally likely.
        count_le = count_ge = 0
        for signs in product((-1, 1), repeat=n):
            w = sum(r for r, s in zip(ranks, signs, strict=False) if s > 0)
            if w <= stat + 1e-12:
                count_le += 1
            if w >= total - stat - 1e-12:
                count_ge += 1
        p = min(1.0, 2.0 * min(count_le, count_ge) / (2**n))
        return stat, p

    # Normal approximation with continuity correction.
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (abs(w_pos - mean) - 0.5) / math.sqrt(var)
    if z < 0:
        return stat, 1.0
    # Standard normal tail via the error function.
    p = math.erfc(z / math.sqrt(2.0))
    return stat, min(1.0, p)


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


def _collapse(rows: list[dict], metric: str) -> dict[tuple[str, int], dict]:
    """Reduce a cell's replicates to one row, using the median metric.

    A cell is not a fixed function of its seed -- see the noise floor
    recorded in eval_set.DIRECT_LITE_SIGNAL_TARGETS -- so with --reps the
    unit of analysis is the cell's median across replicates, not a single
    draw. Median rather than mean because the replicate distribution on
    png is visibly skewed by occasional low outliers (45 against a 57-63
    cluster), and one bad draw should not decide a paired cell.

    ``coverage_attached`` is required of *every* replicate: a cell where
    coverage failed to attach even once is not one whose median can be
    trusted, and dropping it is the same choice the single-replicate path
    already makes.
    """
    by: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        by.setdefault(_key(r), []).append(r)
    out = {}
    for k, group in by.items():
        merged = dict(group[0])
        merged[metric] = statistics.median(r[metric] for r in group)
        merged["coverage_attached"] = all(r["coverage_attached"] for r in group)
        merged["reps"] = len(group)
        merged["spread"] = max(r[metric] for r in group) - min(r[metric] for r in group)
        out[k] = merged
    return out


def compare(base: list[dict], test: list[dict], metric: str = "edges") -> dict:
    """Pair two arms' rows by (target, seed) and score them."""
    b_by = _collapse(base, metric)
    t_by = _collapse(test, metric)
    shared = sorted(b_by.keys() & t_by.keys())

    wins = losses = ties = 0
    deltas = []
    dropped_base = dropped_test = 0
    for k in shared:
        rb, rt = b_by[k], t_by[k]
        # A cell is only scorable when BOTH arms attached coverage. Which arm
        # failed is the MCAR/MNAR diagnostic: if the drop rate differs between
        # base and test, the reported win rate is optimistic for whichever arm
        # drops less, because McNemar/Wilcoxon are computed on the survivors.
        if not rb["coverage_attached"]:
            dropped_base += 1
        if not rt["coverage_attached"]:
            dropped_test += 1
        if not (rb["coverage_attached"] and rt["coverage_attached"]):
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
    wilcoxon_stat, wilcoxon_p = _wilcoxon_signed_rank(deltas)
    spreads = [b_by[k]["spread"] for k in shared] + [t_by[k]["spread"] for k in shared]
    reps = [b_by[k]["reps"] for k in shared] + [t_by[k]["reps"] for k in shared]
    return {
        "cells": n,
        "reps": min(reps) if reps else 0,
        # The within-arm spread is the yardstick the median delta has to be
        # read against: an effect smaller than the noise it sits in is not a
        # result, however the p-value comes out.
        "median_spread": statistics.median(spreads) if spreads else 0,
        "dropped_no_coverage": dropped_base + dropped_test,
        "dropped_base": dropped_base,
        "dropped_test": dropped_test,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mcnemar_p": _mcnemar_exact(wins, losses),
        "fisher_p": _fisher_exact(wins, ties + losses, losses, ties + wins),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
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


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order.

    Example: [.04, .01, .03] -> sorted .01, .03, .04 scaled x3, x2, x1 ->
    .03, .06, .04 -> running max .03, .06, .06.
    """
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


# Pre-registered strata thresholds (§6).
STRATA_ALPHA = 0.05
STRATA_EPS_FLOOR = 0.98  # A1 may lose < 2% execs/s
STRATA_TESTS = ("strata-a2", "strata-a3", "strata-a4")


def _eps_ratio(base: list[dict], test: list[dict]) -> float:
    """Median per-cell execs/s ratio test/base over cells both measured."""
    b = {_key(r): r["eps"] for r in base if r.get("eps")}
    t = {_key(r): r["eps"] for r in test if r.get("eps")}
    shared = b.keys() & t.keys()
    return statistics.median(t[k] / b[k] for k in shared) if shared else 0.0


def strata_verdict(loaded: dict[str, list[dict]]) -> dict:
    """§6 analysis: control first, then A1 gates, then Holm over A2-A4.

    A control that rejects means the oracle is broken, and nothing after it
    is reported (Rule 46).
    """
    ctl = compare(loaded["strata-a0"], loaded["strata-a0-ctl"])
    out: dict = {"control": ctl, "control_ok": ctl["mcnemar_p"] >= STRATA_ALPHA, "comparisons": {}}
    if not out["control_ok"]:
        return out

    a1 = compare(loaded["strata-a0"], loaded["strata-a1"])
    worse = a1["losses"] > a1["wins"] and a1["mcnemar_p"] < STRATA_ALPHA
    ratio = _eps_ratio(loaded["strata-a0"], loaded["strata-a1"])
    out.update(
        a1=a1, a1_noninferior=not worse, a1_eps_ratio=ratio, a1_eps_ok=ratio >= STRATA_EPS_FLOOR
    )

    rows = {arm: compare(loaded[ARM_BASELINES[arm]], loaded[arm]) for arm in STRATA_TESTS}
    adj = holm([rows[a]["mcnemar_p"] for a in STRATA_TESTS])
    for arm, p in zip(STRATA_TESTS, adj, strict=True):
        rows[arm]["holm_p"] = p
    out["comparisons"] = rows
    return out


def cmd_strata(args: argparse.Namespace) -> int:
    loaded: dict[str, list[dict]] = {}
    for path in args.files:
        rows = json.loads(Path(path).read_text())
        if rows:
            loaded.setdefault(rows[0]["arm"], []).extend(rows)
    missing = [a for a in STRATA_ARMS if a not in loaded]
    if missing:
        print(f"missing arms: {', '.join(missing)}", file=sys.stderr)
        return 2

    v = strata_verdict(loaded)
    c = v["control"]
    print(
        f"control a0 vs a0-ctl: W{c['wins']} L{c['losses']} T{c['ties']} McNemar {c['mcnemar_p']:.3g}"
    )
    if not v["control_ok"]:
        print("[!] control rejected: the comparison is broken, not the arms. Stop.")
        return 1

    a1 = v["a1"]
    print(
        f"A1 vs A0: W{a1['wins']} L{a1['losses']} McNemar {a1['mcnemar_p']:.3g} "
        f"non-inferior={v['a1_noninferior']} execs/s ratio {v['a1_eps_ratio']:.3f} "
        f"(floor {STRATA_EPS_FLOOR}) ok={v['a1_eps_ok']}"
    )
    for arm, r in v["comparisons"].items():
        print(
            f"{arm} vs {ARM_BASELINES[arm]}: W{r['wins']} L{r['losses']} T{r['ties']} "
            f"McNemar {r['mcnemar_p']:.3g} Holm {r['holm_p']:.3g} "
            f"med Δ {r['median_delta']:+.1f} IQR {r['iqr']}"
        )
    return 0


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
    hdr = (
        f"{'arm':<20} {'cells':>5} {'rep':>3} {'W':>4} {'L':>4} {'T':>4} "
        f"{'McNemar':>9} {'Fisher':>9} {'med Δ':>7} {'noise':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for arm, rows in sorted(loaded.items()):
        r = compare(base, rows, args.metric)
        print(
            f"{arm:<20} {r['cells']:>5} {r['reps']:>3} {r['wins']:>4} {r['losses']:>4} "
            f"{r['ties']:>4} {r['mcnemar_p']:>9.3g} {r['fisher_p']:>9.3g} "
            f"{r['median_delta']:>+7.1f} {r['median_spread']:>6.1f}"
        )
        if r["dropped_no_coverage"]:
            db, dt = r["dropped_base"], r["dropped_test"]
            if db == dt:
                print(f"{'':<20} dropped {r['dropped_no_coverage']} cells: coverage did not attach")
            else:
                print(
                    f"{'':<20} dropped base {db} / test {dt}: coverage did not attach "
                    f"(MCAR check -- unequal drop rates bias the surviving-cell win rate)"
                )

    if args.by_target:
        # A pooled row is not enough to read an arm whose effect is expected on
        # some targets and absent by construction on others: cells that cannot
        # move dilute a real effect, and a target near coverage saturation
        # contributes ties that read as agreement. Break the same pairing out
        # per target so the shape of the result is visible, not just its sum.
        for arm, rows in sorted(loaded.items()):
            print(f"\nper-target: {arm} vs {args.baseline}")
            sub = (
                f"{'target':<20} {'cells':>5} {'rep':>3} {'W':>4} {'L':>4} {'T':>4} "
                f"{'McNemar':>9} {'Wilcox':>8} {'med Δ':>7} {'noise':>6}"
            )
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
                    f"{Path(tgt).name:<20} {r['cells']:>5} {r['reps']:>3} {r['wins']:>4} "
                    f"{r['losses']:>4} {r['ties']:>4} {r['mcnemar_p']:>9.3g} "
                    f"{r['wilcoxon_p']:>8.3g} {r['median_delta']:>+7.1f} "
                    f"{r['median_spread']:>6.1f}"
                )

    # Compute and output risk matrix if requested
    if args.risk_matrix:
        risk_matrix = compute_risk_matrix(loaded)
        print("\nRisk Matrix (for minimax-robust scheduler selection):")
        print("Format: {scheduler: {target: worst_case_regret}}")
        print(json.dumps(risk_matrix, indent=2))
        print("\nThis can be used to update the EloTracker's risk matrix.")

    print(
        "\nMcNemar is the test to read: the matrix is paired by construction. "
        "Fisher is shown only to make the cost of discarding the pairing visible."
    )
    print(
        "A ~10-point difference in per-cell win rate needs roughly 100 paired cells "
        "to resolve; check the cells column before believing a p-value."
    )
    print(
        "noise is the median within-arm spread across a cell's replicates -- the "
        "yardstick med \u0394 has to beat. At rep=1 it reads 0 because a single "
        "draw cannot show its own spread, not because the cell is reproducible."
    )
    return 0


def compute_risk_matrix(loaded: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Compute risk matrix from benchmark results.

    For each arm (scheduler) and target, compute the worst-case regret across seeds.
    Regret = 1.0 - average_score, where average_score is the arm's average performance
    in head-to-head matchups against other arms on the same (target, seed) cell.
    Worst-case regret = maximum regret across seeds for that target.
    """
    if not loaded:
        return {}

    # Get all arms and targets
    arms = list(loaded.keys())
    if not arms:
        return {}

    # Get all targets from the first arm's results
    first_arm_results = loaded[arms[0]]
    if not first_arm_results:
        return {}

    targets = sorted(set(r["target"] for r in first_arm_results))
    if not targets:
        return {}

    # Initialize risk matrix: arm -> target -> list of regrets per seed
    risk_matrix: dict[str, dict[str, list[float]]] = {
        arm: {target: [] for target in targets} for arm in arms
    }

    # For each target and seed, collect results from all arms
    for target in targets:
        # Group results by (arm, seed) for this target
        arm_seed_results: dict[tuple[str, int], dict] = {}
        for arm in arms:
            for result in loaded[arm]:
                if result["target"] == target:
                    seed = result["seed"]
                    arm_seed_results[(arm, seed)] = result

        # For each seed, compute head-to-head performance
        seeds = set(seed for (_, seed) in arm_seed_results)
        for seed in seeds:
            # Get results for all arms on this (target, seed)
            arm_results: dict[str, dict] = {}
            for arm in arms:
                if (arm, seed) in arm_seed_results:
                    arm_results[arm] = arm_seed_results[(arm, seed)]

            if len(arm_results) < 2:
                # Need at least two arms to compare
                continue

            # For each arm, compute its average score against all other arms
            for arm, result in arm_results.items():
                scores = []
                for other_arm, other_result in arm_results.items():
                    if arm == other_arm:
                        continue
                    # Determine who won based on edge count
                    edges_a = result["edges"]
                    edges_b = other_result["edges"]
                    if edges_a > edges_b:
                        score_a = 1.0  # arm won
                    elif edges_a < edges_b:
                        score_a = 0.0  # arm lost
                    else:
                        score_a = 0.5  # tie
                    scores.append(score_a)

                if scores:
                    avg_score = sum(scores) / len(scores)
                    regret = 1.0 - avg_score
                    risk_matrix[arm][target].append(regret)

    # Compute worst-case regret (maximum regret across seeds) for each arm-target pair
    worst_case_risk_matrix: dict[str, dict[str, float]] = {}
    for arm in arms:
        worst_case_risk_matrix[arm] = {}
        for target in targets:
            regrets = risk_matrix[arm][target]
            if regrets:
                worst_case_risk_matrix[arm][target] = max(regrets)
            else:
                worst_case_risk_matrix[arm][target] = 0.0  # No data, assume no regret

    return worst_case_risk_matrix


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
    r.add_argument(
        "--targets",
        default=None,
        help=(
            "comma-separated basenames to slice this invocation down to, e.g. "
            "png_read.so,jpeg_read.so. Slices a set, it does not redefine one: "
            "the results file is still keyed by the set, and cells left out here "
            "are filled in by a later resuming run"
        ),
    )
    r.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    r.add_argument(
        "--reps",
        type=int,
        default=1,
        help=(
            "replicates per cell; the cell's median is what the pairing uses. "
            "1 reproduces the old single-draw behaviour"
        ),
    )
    r.add_argument("--timeout", type=int, default=900, help="per-campaign timeout, seconds")
    r.add_argument(
        "--restart",
        action="store_true",
        help="discard any recorded cells for these arms and re-run the matrix from scratch",
    )
    bench_lock.add_argument(r)
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyse", help="paired analysis of recorded runs")
    a.add_argument("files", nargs="+")
    a.add_argument("--baseline", default="baseline")
    a.add_argument("--metric", default="edges", choices=("edges", "corpus", "crashes", "eps"))
    a.add_argument(
        "--by-target",
        action="store_true",
        help="also break the pairing out per target, not just pooled",
    )
    a.add_argument(
        "--risk-matrix",
        action="store_true",
        help="output risk matrix for minimax-robust scheduler selection",
    )
    a.set_defaults(func=cmd_analyse)

    st = sub.add_parser("strata", help="pre-registered strata A0-A4 analysis (§6)")
    st.add_argument("files", nargs="+")
    st.set_defaults(func=cmd_strata)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
