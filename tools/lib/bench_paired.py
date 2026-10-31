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
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import bench_lock  # noqa: E402
from eval_set import DEFAULT_ITERS, SEED_DIRS, SEEDS, TARGET_SETS, cells  # noqa: E402

from fuzzer_tool.core.exact_tests import (  # noqa: E402
    fisher_exact,
    mcnemar_exact,
    wilcoxon_signed_rank,
)

REPO = Path(__file__).resolve().parents[2]
# $FUZZ_BENCH_RESULTS keeps results out of the source tree (AGENTS.md Rule 17).
RESULTS = Path(os.environ.get("FUZZ_BENCH_RESULTS", REPO / "results" / "paired"))

# ── Arms ───────────────────────────────────────────────────────────────
# An arm is a name and the extra CLI flags that define it. Keep arms
# single-variable against the baseline: an arm that changes two things
# cannot attribute its result to either.

# Arena arms that can be A/B'd against uniform with no extra feature flag.
ARENA_TESTABLE = (
    "burn_front",
    "kl_ducb",
    "fractal",
    "context",
    "levy",
    "boundary",
    "round_robin",
    "fibonacci",
    "token",
    "chunk",
    "changed",
    "rare_mask",
    "centroid",
    "good_turing",
    "consolidated",
)
_ARENA = ["--elo", "--mc-bandit", "--position-arena"]

# Growing Tree sweep (docs/handover/handover_maze_algorithms_2026-09-24.md items
# 2-3): p_newest in percent, 0 = uniform random (Prim), 100 = pure newest (DFS).
# Percent integers keep arm names free of dots.
SEED_NEWEST_PCTS = (0, 30, 60, 100)

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
    # Tang seed arm, paper score vs popularity-removed (docs/TODO.md r1_tang).
    "elo-tang": ["--elo", "--mc-bandit", "--tang"],
    "elo-tang-residual": ["--elo", "--mc-bandit", "--tang", "--tang-mode", "residual"],
    "elo-op-credit": ["--elo", "--mc-bandit", "--op-credit"],
    # Reactive gradient (core/schedulers/op_reactive.py) vs its static parent.
    # Both pair against "elo"; read reactive with `analyse --baseline elo-gradient`.
    "elo-gradient": ["--elo", "--mc-bandit", "--gradient"],
    "elo-reactive": ["--elo", "--mc-bandit", "--reactive"],
    # The control that must not reject (Rule 46): elo-gradient again.
    "elo-gradient-ctl": ["--elo", "--mc-bandit", "--gradient"],
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
    # Repeat crashes stop paying operators (fuzz_round._judge). Metric: unique.
    "novel-crash-reward": ["--novel-crash-reward"],
    # The control that must not reject (Rule 46): baseline again.
    "novel-crash-reward-ctl": [],
    # Same on a target with a dominant shallow crash: FFmpeg asserts count as
    # crashes only with --abort-is-crash, so every arm of this pair carries it.
    "abort-crash": ["--abort-is-crash"],
    "novel-crash-reward-abort": ["--abort-is-crash", "--novel-crash-reward"],
    "abort-crash-ctl": ["--abort-is-crash"],
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
    # Line-graph 'branch' seed arm; needs --set fuzzgoat_katz (trace-pc), where
    # Katz is on in both. ctl is elo again: the A/A control (Rule 46).
    "seed-branch": ["--elo", "--mc-bandit", "--seed-branch-scheduler"],
    "seed-branch-ctl": ["--elo", "--mc-bandit"],
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
    # Growing Tree seed arms (see SEED_NEWEST_PCTS). No --elo: the arm is the
    # sole seed strategy, so each differs from `baseline` by that one picker.
    "seed-round-robin": ["--seed-round-robin-scheduler"],
    # DPP seed arm (docs/handover/handover_dpp_seed_selector_2026-10-22.md):
    # sole seed strategy without --elo, same single-knob pairing as above.
    "seed-dpp": ["--seed-dpp-scheduler"],
    **{
        f"seed-newest-p{pct}": ["--seed-newest-scheduler", "--seed-newest-p", str(pct / 100)]
        for pct in SEED_NEWEST_PCTS
    },
    # Good-Turing discovery-probability arms (handover entropy §7.1). Both are
    # Elo arms, so they pair against `elo`, not `baseline`. Prior left at its
    # default: the sweep is a separate A/B.
    "elo-good-turing-seed": ["--elo", "--mc-bandit", "--good-turing-seed"],
    # Rarity per incidence class + near-miss bonus (core/edge_clusters.py).
    # Acts on the default weighted picker, so it pairs against plain baseline.
    "edge-clusters": ["--edge-clusters"],
    "elo-op-good-turing": ["--elo", "--mc-bandit", "--op-good-turing"],
    # Arena arm subsets (--pos-arena-arms). The control is the arena with
    # uniform as its only member: same Elo/bandit stack and same arena
    # bookkeeping as every arm below, no proposer. Each pos-arena-<x> is that
    # control plus one arm, so it isolates <x> from the arena machinery;
    # pos-arena-all is the full pool (the ensemble question). cmplog, lineage
    # and the tracker arms (sensitivity/te/mi/...) are left out on purpose:
    # they join only while their own feature is on, so an arm here would have
    # to carry that flag and no longer differ from the control by one knob.
    "pos-arena-uniform": _ARENA + ["--pos-arena-arms", "uniform"],
    **{
        f"pos-arena-{a.replace('_', '-')}": _ARENA + ["--pos-arena-arms", f"uniform,{a}"]
        for a in ARENA_TESTABLE
    },
    "pos-arena-all": _ARENA,
    # Saliency (NEUZZ-style net) is gated on its own flag, so it cannot be A/B'd by
    # subset alone: this arm differs from pos-arena-uniform by --pos-saliency plus
    # the subset that lets it into the pool.
    "pos-arena-saliency": _ARENA + ["--pos-saliency", "--pos-arena-arms", "uniform,saliency"],
    # Centroid count sweep: --cluster-jaccard 0.60 / 0.90 (default 0.75 is
    # pos-arena-centroid). Lower J -> fewer, looser clusters, more gaps.
    **{
        f"pos-arena-centroid-j{j}": _ARENA
        + ["--pos-arena-arms", "uniform,centroid", "--cluster-jaccard", f"0.{j}"]
        for j in ("60", "90")
    },
    # Harmonic is opt-in like saliency: --pos-harmonic plus the subset that admits it.
    "pos-arena-harmonic": _ARENA + ["--pos-harmonic", "--pos-arena-arms", "uniform,harmonic"],
    # Kadane is opt-in like saliency: --pos-kadane plus the subset that admits it.
    "pos-arena-kadane": _ARENA + ["--pos-kadane", "--pos-arena-arms", "uniform,kadane"],
    # PCA is opt-in like kadane; the byteflip pass it reads is on by default.
    "pos-arena-pca": _ARENA + ["--pos-pca", "--pos-arena-arms", "uniform,pca"],
    # LearnAFL path mask is opt-in like kadane.
    "pos-arena-learn": _ARENA + ["--pos-learn", "--pos-arena-arms", "uniform,learn"],
    # Gravity splice donor (core/gravity.py). Only the seven corpus-crossing
    # operators change; read "Gravity splice: ... hits, refits" in a cell's
    # log before trusting a null -- a closed fit gate means prior exponents.
    "splice-gravity": ["--splice-donor", "gravity"],
    # Minimax Phases 3-5 (docs/handover/handover_minimax_implementation_
    # 2026-09-01.md). op-minimax only replaces the bandit's pick, so it pairs
    # against "elo". minimax-select only acts inside auto_minimize_corpus,
    # which is off unless scheduled, hence its own minimizing baseline.
    "elo-op-minimax": ["--elo", "--mc-bandit", "--op-minimax"],
    "wall-order": ["--wall-order"],
    "minimize-2k": ["--minimize-every-execs", "2000"],
    "minimax-select": ["--minimize-every-execs", "2000", "--minimax-select"],
    # Lottery-scheduling ports (docs/handover/handover_lottery_scheduling_\
    # 2026-10-27.md). seed-stride-cost pairs against plain seed-stride, which
    # pairs against baseline; judge it on edges at fixed WALL time, since at
    # fixed execs compensation cannot show an effect. inverse-lottery-select
    # only acts inside auto_minimize_corpus, so it shares minimize-2k.
    "seed-stride": ["--seed-stride-scheduler"],
    "seed-stride-cost": ["--seed-stride-scheduler", "--seed-stride-cost"],
    "inverse-lottery-select": ["--minimize-every-execs", "2000", "--inverse-lottery-select"],
    # The lottery arm (core/schedulers/seed_lottery.py). Each option pairs against
    # the arm before it: lottery vs stride (same compensation, random vs
    # deterministic draw), then currency and transfer vs plain lottery.
    # seed-lottery-transfer needs --lineage to have any parent to credit, so it
    # pairs against seed-lottery-lineage, not seed-lottery: --lineage alone is a
    # change and must not be credited to the transfer.
    "seed-lottery": ["--seed-lottery-scheduler"],
    "seed-lottery-lineage": ["--seed-lottery-scheduler", "--lineage"],
    "seed-lottery-currency": ["--seed-lottery-scheduler", "--seed-lottery-currency", "sqrt"],
    "seed-lottery-transfer": [
        "--seed-lottery-scheduler",
        "--lineage",
        "--seed-lottery-transfer",
        "0.5",
    ],
    # Viterbi field spans (core/field_hmm.py): hmm_segment op + span-start positions.
    "field-hmm": ["--field-hmm"],
    # A/A control (Rule 46): baseline again, must not reject.
    "field-hmm-ctl": [],
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

# Growing Tree sweep arms, in the order the handover lists them. Each pairs
# against `baseline` (default weighted picker, ARM_BASELINES); to pair one
# against round robin instead, run `analyse --baseline seed-round-robin`.
GROWING_TREE_ARMS = (
    "seed-round-robin",
    *(f"seed-newest-p{pct}" for pct in SEED_NEWEST_PCTS),
)

GOOD_TURING_ARMS = ("elo-good-turing-seed", "elo-op-good-turing")

REACTIVE_ARMS = ("elo-gradient", "elo-reactive")

# Arena subset group (pos-arena-uniform is the control the rest pair against).
POSITION_ARENA_ARMS = (
    "pos-arena-uniform",
    *(f"pos-arena-{a.replace('_', '-')}" for a in ARENA_TESTABLE),
    "pos-arena-all",
)

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
    **{arm: "baseline" for arm in GROWING_TREE_ARMS},
    "seed-dpp": "baseline",
    "novel-crash-reward-abort": "abort-crash",
    "abort-crash-ctl": "abort-crash",
    "seed-branch": "elo",
    "elo-tang": "elo",
    "elo-tang-residual": "elo-tang",
    "seed-branch-ctl": "elo",
    "elo-good-turing-seed": "elo",
    "elo-op-good-turing": "elo",
    **{arm: "elo" for arm in REACTIVE_ARMS},
    "edge-clusters": "baseline",
    "pos-arena-uniform": "elo",
    **{f"pos-arena-{a.replace('_', '-')}": "pos-arena-uniform" for a in ARENA_TESTABLE},
    "pos-arena-all": "pos-arena-uniform",
    "pos-arena-saliency": "pos-arena-uniform",
    "pos-arena-centroid-j60": "pos-arena-uniform",
    "pos-arena-centroid-j90": "pos-arena-uniform",
    "pos-arena-harmonic": "pos-arena-uniform",
    "pos-arena-kadane": "pos-arena-uniform",
    "pos-arena-pca": "pos-arena-uniform",
    "pos-arena-learn": "pos-arena-uniform",
    "splice-gravity": "baseline",
    "elo-op-minimax": "elo",
    "wall-order": "baseline",
    "minimax-select": "minimize-2k",
    "seed-stride": "baseline",
    "seed-stride-cost": "seed-stride",
    "inverse-lottery-select": "minimize-2k",
    "seed-lottery": "seed-stride-cost",
    "seed-lottery-currency": "seed-lottery",
    "seed-lottery-lineage": "seed-lottery",
    "seed-lottery-transfer": "seed-lottery-lineage",
    "field-hmm": "baseline",
    "field-hmm-ctl": "baseline",
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
    # Off arm: a tree without the op (no per-op CLI switch). 2026-10-17 used a master worktree.
    "literal-mismatch": "core/operator_registry.py: drop literal_mismatch from _CATEGORIES['dict']",
}

# ── Running ────────────────────────────────────────────────────────────

_EDGES = re.compile(r"Edges discovered:\s+(\d+)")
_CORPUS = re.compile(r"Corpus:\s+(\d+)")
_CRASHES = re.compile(r"Crashes:\s+(\d+)")
# "Crashes:  7 (4 unique signatures)": distinct bugs, not crashing execs.
_UNIQUE = re.compile(r"Crashes:\s+\d+ \((\d+) unique")
# Campaign summary prints "Avg eps:"; the report section "Avg throughput:".
_EPS = re.compile(r"(?:Avg eps|Avg throughput):\s+([\d.]+)")
# Run summary "Executions:        10,000": execs actually run.
_EXECS = re.compile(r"Executions:\s+([\d,]+)")


def _parse(log: str) -> dict:
    def one(rx, default=0):
        m = rx.search(log)
        return int(m.group(1)) if m else default

    return {
        "edges": one(_EDGES),
        "corpus": one(_CORPUS),
        "crashes": one(_CRASHES),
        "unique": one(_UNIQUE),
        "eps": float(m.group(1)) if (m := _EPS.search(log)) else 0.0,
        "execs": int(m.group(1).replace(",", "")) if (m := _EXECS.search(log)) else 0,
        "coverage_attached": _EDGES.search(log) is not None and one(_EDGES) > 0,
    }


def _seed_workdir(target: str, workdir: Path) -> None:
    """Copy *target*'s seed pack (eval_set.SEED_DIRS) into *workdir*.

    A copy: cells write their corpus there. A listed pack that is missing
    raises rather than running the cell unseeded.
    """
    pack = SEED_DIRS.get(target)
    if pack is None:
        return
    try:
        shutil.copytree(pack / "seeds", workdir / "seeds")
    except FileNotFoundError:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


def run_cell(
    arm: str, target: str, flags: str, seed: int, iters: int, timeout: int, rep: int = 0
) -> dict:
    """Run one campaign and return its parsed outcome."""
    workdir = Path(tempfile.mkdtemp(prefix=f"paired_{arm}_"))
    _seed_workdir(target, workdir)
    cmd = [
        sys.executable,
        "-m",
        "fuzzer_tool",
        "fuzz",
        target,
        "-d",
        str(workdir),
        "-c",
        "--max-execs",
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
        # Timeout output is bytes even under text=True; stderr holds sanitizer reports.
        log = b"".join(x for x in (exc.stdout, exc.stderr) if x).decode(errors="replace")
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
    # Short of the budget: not a result. Keep the log for diagnosis.
    if row["execs"] < iters:
        row["log"] = log
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
    matrix = _target_matrix(args, seeds)
    if matrix is None:
        return 2
    total = len(matrix) * len(arms) * args.reps
    print(
        f"[*] {len(arms)} arms x {len(matrix)} cells x {args.reps} reps = "
        f"{total} campaigns @ {args.iters} execs"
    )

    built = _drop_unbuilt(matrix)
    if len(built) != len(matrix):
        matrix = built
        total = len(matrix) * len(arms) * args.reps
        if not matrix:
            print("[!] no targets available", file=sys.stderr)
            return 1

    done = 0
    for arm in arms:
        done = _run_arm(args, arm, matrix, done, total)
    if lock:
        lock.release()
    return 0


def _target_matrix(args: argparse.Namespace, seeds: list[int]) -> list | None:
    """Cells of ``args.set``, narrowed to ``--targets``; None if a name is not in the set."""
    matrix = list(cells(args.set, seeds))
    if not args.targets:
        return matrix
    want = set(args.targets.split(","))
    matrix = [c for c in matrix if Path(c[0]).name in want]
    unknown = want - {Path(c[0]).name for c in list(cells(args.set, seeds))}
    if unknown:
        print(f"[!] not in set {args.set}: {', '.join(sorted(unknown))}", file=sys.stderr)
        return None
    return matrix


def _drop_unbuilt(matrix: list) -> list:
    """Remove cells whose target binary is missing, reporting them."""
    missing = sorted({t for t, _, _ in matrix if not (REPO / t).exists()})
    if not missing:
        return matrix
    # Skipped, not scored zero: a target that failed to build is a hole
    # in the matrix, and recording it as an outcome would let a build
    # problem masquerade as an arm difference.
    print(f"[!] not built, cells skipped: {', '.join(Path(m).name for m in missing)}")
    print("    build them with tools/build_targets.sh before quoting a result")
    return [c for c in matrix if c[0] not in set(missing)]


def _load_rows(out: Path, restart: bool) -> list:
    """Previously recorded cells from *out*, or [] on restart / unreadable file."""
    if not out.exists() or restart:
        return []
    try:
        return json.loads(out.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def _keep_early(out: Path, row: dict) -> None:
    """Save the log of a cell that stopped short of ``iters``; it is not checkpointed.

    Unrecorded, so a resume re-runs it instead of pairing a partial campaign.
    """
    early = out.with_name(f"{out.stem}_early")
    early.mkdir(exist_ok=True)
    name = f"{Path(row['target']).name}_s{row['seed']}_r{row['rep']}.log"
    (early / name).write_text(row["log"])
    print(
        f"  [!] {row['arm']} {Path(row['target']).name} seed={row['seed']} rep={row['rep']}: "
        f"{row['execs']}/{row['iters']} execs rc={row['rc']}, not recorded; log in {early / name}",
        flush=True,
    )


def _run_arm(args: argparse.Namespace, arm: str, matrix: list, done: int, total: int) -> int:
    """Run every missing cell of one arm, checkpointing each; returns updated *done*."""
    out = RESULTS / f"{args.set}_{arm}.json"
    # Resume from whatever is already on disk. A full matrix is hours of
    # compute and the results file used to be written once, after the last
    # cell of an arm -- so an interrupted run lost every cell it had
    # completed. Cells are keyed by (target, seed), which is the same key
    # the pairing uses, so a resumed file is indistinguishable from one
    # produced in a single pass.
    rows = _load_rows(out, args.restart)
    have = {(r["target"], r["seed"], r.get("rep", 0)) for r in rows}
    if have:
        print(f"[*] {arm}: resuming, {len(have)} cells already recorded in {out.name}")

    for target, flags, seed in matrix:
        for rep in range(args.reps):
            done += 1
            if (target, seed, rep) in have:
                continue
            row = run_cell(arm, target, flags, seed, args.iters, args.timeout, rep)
            if "log" in row:
                _keep_early(out, row)
                continue
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
    return done


# ── Analysis ───────────────────────────────────────────────────────────


# Exact tests live in core (single source of truth); private names kept for
# tools/ab_synergy_multi_ffmpeg.py and tests/test_bench_paired_stats.py.
_mcnemar_exact = mcnemar_exact
_fisher_exact = fisher_exact
_wilcoxon_signed_rank = wilcoxon_signed_rank


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

    deltas, dropped_base, dropped_test = _paired_deltas(b_by, t_by, shared, metric)
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - wins - losses

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
        "iqr": _iqr(deltas),
        "base_total": sum(b_by[k][metric] for k in shared),
        "test_total": sum(t_by[k][metric] for k in shared),
    }


def _paired_deltas(b_by: dict, t_by: dict, shared: list, metric: str) -> tuple[list, int, int]:
    """Test-minus-base deltas over cells where both arms attached coverage, plus per-arm drops."""
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
        deltas.append(rt[metric] - rb[metric])
    return deltas, dropped_base, dropped_test


def _iqr(deltas: list) -> tuple[float, float] | None:
    """(Q1, Q3) of *deltas* rounded to 0.1; None below 4 samples."""
    if len(deltas) < 4:
        return None
    q = statistics.quantiles(deltas, n=4)
    return (round(q[0], 1), round(q[2], 1))


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


def _add_holm(results: list[dict]) -> None:
    """Set ``holm_p`` on each comparison: Holm over the family a reader scans.

    A table of m raw p-values shows ~m*alpha below alpha by chance; the
    smallest one is the row the eye picks.
    """
    adj = holm([r["mcnemar_p"] for r in results])
    for r, p in zip(results, adj, strict=True):
        r["holm_p"] = p


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


def _print_pooled(arm: str, r: dict) -> None:
    """One pooled comparison row, plus the coverage-drop (MCAR) note when cells were dropped."""
    print(
        f"{arm:<20} {r['cells']:>5} {r['reps']:>3} {r['wins']:>4} {r['losses']:>4} "
        f"{r['ties']:>4} {r['mcnemar_p']:>9.3g} {r['holm_p']:>9.3g} {r['fisher_p']:>9.3g} "
        f"{r['median_delta']:>+7.1f} {r['median_spread']:>6.1f}"
    )
    if not r["dropped_no_coverage"]:
        return
    db, dt = r["dropped_base"], r["dropped_test"]
    if db == dt:
        print(f"{'':<20} dropped {r['dropped_no_coverage']} cells: coverage did not attach")
        return
    print(
        f"{'':<20} dropped base {db} / test {dt}: coverage did not attach "
        f"(MCAR check -- unequal drop rates bias the surviving-cell win rate)"
    )


def _target_results(base: list, loaded: dict, metric: str) -> list[tuple[str, str, dict]]:
    """(arm, target, comparison) for every arm x target both sides ran."""
    out = []
    for arm, rows in sorted(loaded.items()):
        targets = sorted({r["target"] for r in base} | {r["target"] for r in rows})
        for tgt in targets:
            b = [r for r in base if r["target"] == tgt]
            t = [r for r in rows if r["target"] == tgt]
            if not b or not t:
                continue
            out.append((arm, tgt, compare(b, t, metric)))
    return out


def _print_by_target(baseline: str, results: list[tuple[str, str, dict]]) -> None:
    """Per-target breakdown, Holm-adjusted over all arm x target rows."""
    _add_holm([r for _, _, r in results])
    sub = (
        f"{'target':<20} {'cells':>5} {'rep':>3} {'W':>4} {'L':>4} {'T':>4} "
        f"{'McNemar':>9} {'Holm':>9} {'Wilcox':>8} {'med Δ':>7} {'noise':>6}"
    )
    last_arm = None
    for arm, tgt, r in results:
        if arm != last_arm:
            print(f"\nper-target: {arm} vs {baseline}")
            print(sub)
            print("-" * len(sub))
            last_arm = arm
        print(
            f"{Path(tgt).name:<20} {r['cells']:>5} {r['reps']:>3} {r['wins']:>4} "
            f"{r['losses']:>4} {r['ties']:>4} {r['mcnemar_p']:>9.3g} "
            f"{r['holm_p']:>9.3g} {r['wilcoxon_p']:>8.3g} {r['median_delta']:>+7.1f} "
            f"{r['median_spread']:>6.1f}"
        )


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
        f"{'McNemar':>9} {'Holm':>9} {'Fisher':>9} {'med Δ':>7} {'noise':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    pooled = {arm: compare(base, rows, args.metric) for arm, rows in sorted(loaded.items())}
    _add_holm(list(pooled.values()))
    for arm, r in pooled.items():
        _print_pooled(arm, r)

    if args.by_target:
        # A pooled row is not enough to read an arm whose effect is expected on
        # some targets and absent by construction on others: cells that cannot
        # move dilute a real effect, and a target near coverage saturation
        # contributes ties that read as agreement. Break the same pairing out
        # per target so the shape of the result is visible, not just its sum.
        _print_by_target(args.baseline, _target_results(base, loaded, args.metric))

    # Compute and output risk matrix if requested
    if args.risk_matrix:
        risk_matrix = compute_risk_matrix({**loaded, args.baseline: base})
        print("\nRisk Matrix (for minimax-robust scheduler selection):")
        print("Format: {scheduler: {target: mean_regret}}")
        print(json.dumps(risk_matrix, indent=2))
        arm, regret, held = holdout_minimax({**loaded, args.baseline: base})
        held_s = "n/a" if held is None else f"{held:.3f}"
        print(f"\nminimax-robust arm: {arm} (worst-target regret {regret:.3f}, held-out {held_s})")

    print(
        "\nMcNemar is the test to read: the matrix is paired by construction. "
        "Fisher is shown only to make the cost of discarding the pairing visible."
    )
    print(
        "Holm adjusts McNemar over every row of its table: read it, not the raw p, "
        "when scanning for the best arm or target."
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


def _target_regrets(loaded: dict, arms: list, target: str) -> dict[str, list[float]]:
    """Per-arm regrets on *target*, one per seed where at least two arms ran."""
    # Group results by (arm, seed) for this target
    arm_seed_results: dict[tuple[str, int], dict] = {}
    for arm in arms:
        for result in loaded[arm]:
            if result["target"] == target:
                arm_seed_results[(arm, result["seed"])] = result

    out: dict[str, list[float]] = {}
    seeds = set(seed for (_, seed) in arm_seed_results)
    for seed in seeds:
        arm_results = {
            a: arm_seed_results[(a, seed)] for a in arms if (a, seed) in arm_seed_results
        }
        if len(arm_results) < 2:
            # Need at least two arms to compare
            continue
        for arm, regret in _h2h_regrets(arm_results).items():
            out.setdefault(arm, []).append(regret)
    return out


def _h2h_regrets(arm_results: dict[str, dict]) -> dict[str, float]:
    """Regret = 1 - mean head-to-head score (win 1, tie 0.5, loss 0 on edges) per arm."""
    regrets = {}
    for arm, result in arm_results.items():
        scores = []
        for other_arm, other_result in arm_results.items():
            if arm == other_arm:
                continue
            edges_a, edges_b = result["edges"], other_result["edges"]
            scores.append(1.0 if edges_a > edges_b else 0.0 if edges_a < edges_b else 0.5)
        if scores:
            regrets[arm] = 1.0 - sum(scores) / len(scores)
    return regrets


def compute_risk_matrix(loaded: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Compute risk matrix from benchmark results.

    Regret = 1.0 - average_score, where average_score is the arm's average performance
    in head-to-head matchups against other arms on the same (target, seed) cell.
    Each (arm, target) entry is the mean regret over seeds: the adversary a
    minimax pick guards against is the target, not seed noise (with two arms
    a per-seed regret is 0, 0.5 or 1, so a max over seeds is ~1.0 for every
    arm). An arm with no cell on a target gets regret 1.0: no evidence is
    the least favourable case, not a perfect score.
    """
    arms = list(loaded)
    targets = sorted({r["target"] for rows in loaded.values() for r in rows})
    if not arms or not targets:
        return {}

    regrets: dict[str, dict[str, list[float]]] = {arm: {t: [] for t in targets} for arm in arms}
    for target in targets:
        for arm, per_seed in _target_regrets(loaded, arms, target).items():
            regrets[arm][target].extend(per_seed)

    return {
        arm: {t: statistics.fmean(v) if v else 1.0 for t, v in regrets[arm].items()} for arm in arms
    }


def minimax_arm(risk_matrix: dict[str, dict[str, float]]) -> tuple[str, float]:
    """Arm minimizing its worst-target regret, and that regret ("", 0.0 if empty).

    Delegates to ``EloTracker.select_minimax_scheduler`` (argmin_i max_j R_ij)
    so the offline pick and the tracker share one definition.
    """
    if not risk_matrix:
        return "", 0.0

    from fuzzer_tool.core.analyzers.analyzer_elo import EloTracker

    tracker = EloTracker()
    tracker.from_dict({"use_minimax": True, "risk_matrix": risk_matrix})
    targets = sorted({t for row in risk_matrix.values() for t in row})
    arm = tracker.select_minimax_scheduler(list(risk_matrix), targets)
    return arm, max(risk_matrix[arm].values())


def _seed_fold(loaded: dict[str, list[dict]], seeds: set) -> dict[str, list[dict]]:
    """Rows of *loaded* restricted to *seeds*."""
    return {arm: [r for r in rows if r["seed"] in seeds] for arm, rows in loaded.items()}


def holdout_minimax(loaded: dict[str, list[dict]]) -> tuple[str, float, float | None]:
    """Minimax arm, its in-sample worst regret, and a 2-fold held-out regret.

    Picking and scoring on the same seeds reports the winner's own noise as
    skill (winner's curse). Held-out: pick on even-indexed seeds, score on
    odd, and vice versa; mean of the two. E.g. two equal arms that alternate
    wins: each fold's pick shows regret 0, scores 1 held out. None below 2
    seeds.
    """
    arm, regret = minimax_arm(compute_risk_matrix(loaded))
    seeds = sorted({r["seed"] for rows in loaded.values() for r in rows})
    if len(seeds) < 2:
        return arm, regret, None

    folds = (set(seeds[0::2]), set(seeds[1::2]))
    held = []
    for pick, score in (folds, folds[::-1]):
        picked, _ = minimax_arm(compute_risk_matrix(_seed_fold(loaded, pick)))
        risk = compute_risk_matrix(_seed_fold(loaded, score))
        held.append(max(risk[picked].values()))
    return arm, regret, statistics.fmean(held)


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
    a.add_argument(
        "--metric", default="edges", choices=("edges", "corpus", "crashes", "unique", "eps")
    )
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
