#!/usr/bin/env python3
"""A/B: does fuzzing several FFmpeg versions together beat fuzzing each alone?

Arms, all on the same seeds (paired):

    MULTI    all V versions in one multi-target campaign, N execs total
    SPLIT    each version alone, N/V execs            (same total compute)
    FULL     each version alone, N execs              (per-version ceiling)
    CONTROL  SPLIT again on a disjoint seed           (A/A, Hard Rule 46)

Edge ids are per binary, so campaign-reported counts do not compare. Every
final corpus is replayed on every version binary instead:

    campaign(cell) ──> corpus ──> replay on v1..vV ──> {v: edge ids} ──> rows.pkl
                                                                           │
    analyse:  per (seed, v)  MULTI            vs SPLIT_UNION  (primary)    │
                             MULTI            vs SPLIT_OWN / FULL_OWN  <───┘
                             CONTROL_OWN      vs SPLIT_OWN    (must be null)

SPLIT_UNION pools the V split corpora (three separate campaigns, merged
afterwards): synergy means the joint campaign covers v better than that.
Wilcoxon signed-rank per version, Holm across comparisons. A significant
control means the comparison is broken, and analyse exits 2.

Usage::

    tools/ab_synergy_multi_ffmpeg.py run --seed-corpus ~/fuzzing/ab_synergy/seeds \\
        --seeds 10 --budget 6000
    tools/ab_synergy_multi_ffmpeg.py analyse ~/fuzzing/ab_synergy/rows.pkl

Cost: seeds x (1 + 3V) campaigns; at ~3 eps, 10 seeds x 6k execs is ~22 h.
Budget must divide by V so SPLIT's total equals MULTI's exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import enum
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from array import array
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bench_paired import _wilcoxon_signed_rank, holm  # noqa: E402

FUZZ_ROOT = Path.home() / "fuzzing"
DEFAULT_OUT = FUZZ_ROOT / "ab_synergy" / "rows.pkl"
DEFAULT_WORK = FUZZ_ROOT / "ab_synergy" / "work"
DEFAULT_GLOB = "ffmpeg_read_*_asan"
DEFAULT_SEEDS = 10
DEFAULT_BUDGET = 6000  # divisible by 2, 3, 4 and 6 versions

CONTROL_SEED_OFFSET = 10_000  # CONTROL seeds never collide with SPLIT seeds
ALPHA = 0.05
REPLAY_MAP_SIZE = 1 << 18  # campaign auto-sizes ffmpeg to 262,144 entries
REPLAY_TIMEOUT_S = 5.0  # per input; ASAN ffmpeg decodes are slow
STARTUP_S = 900  # campaign startup (profile, ICFG) before the first exec
EXEC_S = 2.0  # generous per-exec wall budget; ~3 eps measured multi-target
# --jobs gate: a new campaign starts only with this much memory available.
# Measured multi-target RSS: 1.8 GB at 2k execs, 3.1 GB at 6k, 5.5 GB at 12k.
MIN_FREE_MB = 4096
MEM_POLL_S = 5.0

Replay = Callable[[str, Path], Iterable[int]]


class CampaignError(RuntimeError):
    """A campaign exited non-zero: its corpus is not a valid cell."""


class Arm(enum.Enum):
    MULTI = "multi"
    SPLIT = "split"
    FULL = "full"
    CONTROL = "control"


class View(enum.Enum):
    """One per-(seed, version) edge count derived from replay results."""

    MULTI = "multi"
    SPLIT_UNION = "split_union"
    SPLIT_OWN = "split_own"
    FULL_OWN = "full_own"
    CONTROL_OWN = "control_own"


# (test, base) pairs; the last is the A/A control.
COMPARISONS = [
    (View.MULTI, View.SPLIT_UNION),
    (View.MULTI, View.SPLIT_OWN),
    (View.MULTI, View.FULL_OWN),
]
CONTROL = (View.CONTROL_OWN, View.SPLIT_OWN)


@dataclass(frozen=True)
class Cell:
    """One campaign: *targets* fuzzed for *execs* with RNG seed *run_seed*."""

    arm: Arm
    seed: int
    run_seed: int
    targets: tuple[str, ...]
    execs: int

    @property
    def key(self) -> tuple[Arm, int, str | None]:
        # MULTI has no owner version; single-target arms are keyed by it.
        owner = self.targets[0] if self.arm is not Arm.MULTI else None
        return self.arm, self.seed, owner


# ── Plan ──────────────────────────────────────────────────────────────


def plan(versions: list[str], seeds: Iterable[int], budget: int) -> list[Cell]:
    """Every campaign of the experiment, SPLIT budget = budget // V each."""
    if not versions:
        raise ValueError("no target versions")

    share = budget // len(versions)
    if share < 1:
        raise ValueError(f"budget {budget} leaves no execs per version ({len(versions)} versions)")
    if budget % len(versions):
        raise ValueError(f"budget {budget} not divisible by {len(versions)} versions")

    cells = []
    for s in seeds:
        cells.append(Cell(Arm.MULTI, s, s, tuple(versions), share * len(versions)))
        for v in versions:
            cells.append(Cell(Arm.SPLIT, s, s, (v,), share))
            cells.append(Cell(Arm.FULL, s, s, (v,), share * len(versions)))
            cells.append(Cell(Arm.CONTROL, s, s + CONTROL_SEED_OFFSET, (v,), share))
    return cells


# ── Execution ─────────────────────────────────────────────────────────


def campaign_cmd(cell: Cell, corpus: Path, crashes: Path) -> list[str]:
    """fuzzer-tool invocation for *cell* (exec budget, not iterations)."""
    return [
        sys.executable,
        "-m",
        "fuzzer_tool",
        "fuzz",
        *cell.targets,
        "-d",
        str(corpus),
        "--crashes",
        str(crashes),
        "--max-execs",
        str(cell.execs),
        "-s",
        str(cell.run_seed),
    ]


def make_campaign(seed_corpus: Path) -> Callable[[Cell, Path], Path]:
    """Real campaign runner: fresh corpus seeded from *seed_corpus* per cell."""

    def campaign(cell: Cell, workdir: Path) -> Path:
        corpus = workdir / "corpus"
        shutil.copytree(seed_corpus, corpus / "seeds")
        cmd = campaign_cmd(cell, corpus, workdir / "crashes")
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=STARTUP_S + EXEC_S * cell.execs,
            check=False,
        )
        print(f"    {cell.arm.value} seed={cell.seed} rc={proc.returncode} {time.time() - t0:.0f}s")
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            raise CampaignError(f"{cell.arm.value} seed={cell.seed} exited {proc.returncode}")
        return corpus

    return campaign


def replay(target: str, corpus: Path) -> frozenset[int]:
    """Union of shim edge ids *target* hits over every input in *corpus*."""
    from fuzzer_tool.adapters.filesystem import load_corpus
    from fuzzer_tool.adapters.shm import ShmCoverage

    inputs, _, _ = load_corpus(corpus, add_default=False)
    if not inputs:
        return frozenset()

    # FUZZER_KEEP_ASLR=1: base-relative context ids, stable across processes.
    cov = ShmCoverage(size=REPLAY_MAP_SIZE)
    env = dict(
        os.environ,
        __AFL_SHM_ID=cov.env_id,
        AFL_MAP_SIZE=str(REPLAY_MAP_SIZE),
        FUZZER_KEEP_ASLR="1",
    )
    seen: set[int] = set()
    try:
        with tempfile.NamedTemporaryFile(prefix="ab_synergy_") as fh:
            _run_input(target, fh, inputs[0], cov, env)  # warm-up, discarded
            for data in inputs:
                seen |= _run_input(target, fh, data, cov, env)
    finally:
        cov.cleanup()
    return frozenset(seen)


def _run_input(target, fh, data: bytes, cov, env) -> set[int]:
    fh.seek(0)
    fh.truncate()
    fh.write(data)
    fh.flush()
    cov.reset_edge_map()
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run([target, fh.name], env=env, capture_output=True, timeout=REPLAY_TIMEOUT_S)
    return cov.get_edge_ids()


def run(
    cells: list[Cell],
    versions: list[str],
    workroot: Path,
    out: Path,
    campaign: Callable[[Cell, Path], Path],
    replayer: Replay,
    manifest: dict,
    jobs: int = 1,
    mem_ok: Callable[[], bool] | None = None,
) -> dict:
    """Run every cell not yet in *out*, up to *jobs* at once; replay each corpus on every version.

    Saved after each cell, so an interrupted run resumes where it stopped.
    Resuming requires the same *manifest* (versions, budget, binary and seed
    digests): rows from another experiment are never mixed in. A cell starts
    only while *mem_ok()* holds or nothing else runs, so parallel campaigns
    cannot exhaust memory. Results are saved by this thread only.

        pending cells ──submit (≤ jobs, mem_ok)──> pool ──(key, ids)──> results ──> out
    """
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")

    mem_ok = mem_ok or _mem_ok
    results: dict = {}
    if out.exists():
        stored, results = _read(out)
        if stored != manifest:
            raise ValueError(f"{out}: manifest differs from this run; use another --out")

    todo = [(i, c) for i, c in enumerate(cells, 1) if c.key not in results]
    errors: list[BaseException] = []
    pending: set[Future] = set()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, cell in todo:
            # Wait for a slot: pool full, or memory short while something still runs.
            while pending and (len(pending) >= jobs or not mem_ok()):
                pending = _drain(pending, results, manifest, out, errors, MEM_POLL_S)
            if errors:
                break

            print(
                f"[{i}/{len(cells)}] {cell.arm.value} seed={cell.seed} {cell.targets}", flush=True
            )
            pending.add(pool.submit(_run_cell, i, cell, versions, workroot, campaign, replayer))
        while pending:
            pending = _drain(pending, results, manifest, out, errors, None)

    if errors:
        raise errors[0]
    return results


def _run_cell(i: int, cell: Cell, versions, workroot: Path, campaign, replayer) -> tuple:
    """Campaign + replay for one cell (worker thread); returns its result row."""
    workdir = workroot / f"{cell.arm.value}_{cell.seed}_{i}"
    shutil.rmtree(workdir, ignore_errors=True)
    try:
        corpus = campaign(cell, workdir)
        # array('I'): 4 B per id, vs ~70 B for a set member (Hard Rule 54).
        return cell.key, {v: array("I", sorted(replayer(v, corpus))) for v in versions}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _drain(pending: set, results: dict, manifest: dict, out: Path, errors: list, timeout) -> set:
    """Collect finished cells: save each success, record each failure."""
    done, rest = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
    for fut in done:
        exc = fut.exception()
        if exc is not None:
            errors.append(exc)
            continue
        key, row = fut.result()
        results[key] = row
        _save(manifest, results, out)
    return rest


def _mem_ok() -> bool:
    """MemAvailable >= MIN_FREE_MB (Linux /proc/meminfo; True where unreadable)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024 >= MIN_FREE_MB
    except OSError:
        return True
    return True


def fingerprint(paths: Iterable[str]) -> dict[str, str]:
    """Content digest per file: a rebuilt binary at the same path is a new experiment."""
    from fuzzer_tool.adapters.filesystem import hash_data

    return {p: hash_data(Path(p).read_bytes()) for p in paths}


def _save(manifest: dict, results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    blob = {"manifest": manifest, "cells": results}
    tmp.write_bytes(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(out)


def _read(path: Path) -> tuple[dict, dict]:
    blob = pickle.loads(path.read_bytes())
    return blob["manifest"], blob["cells"]


def load(path: Path) -> dict:
    """Recorded cells: {(arm, seed, owner): {version: edge ids}}."""
    return _read(path)[1]


# ── Analysis ──────────────────────────────────────────────────────────


def score(results: dict, versions: list[str]) -> dict[tuple[View, int, str], int]:
    """Edge counts per (view, seed, version); a view with a missing cell is skipped."""
    out: dict[tuple[View, int, str], int] = {}
    seeds = {seed for _, seed, _ in results}
    own = {Arm.SPLIT: View.SPLIT_OWN, Arm.FULL: View.FULL_OWN, Arm.CONTROL: View.CONTROL_OWN}

    for s in seeds:
        multi = results.get((Arm.MULTI, s, None))
        split = [results.get((Arm.SPLIT, s, o)) for o in versions]
        for v in versions:
            if multi is not None:
                out[(View.MULTI, s, v)] = len(multi[v])
            if all(r is not None for r in split):
                out[(View.SPLIT_UNION, s, v)] = len(set().union(*(r[v] for r in split)))
            for arm, view in own.items():
                r = results.get((arm, s, v))
                if r is not None:
                    out[(view, s, v)] = len(r[v])
    return out


def _compare(scores: dict, test: View, base: View, version: str) -> dict:
    """Paired deltas over the seeds both views have for *version*."""
    t = {s: n for (vw, s, v), n in scores.items() if vw is test and v == version}
    b = {s: n for (vw, s, v), n in scores.items() if vw is base and v == version}
    deltas = [t[s] - b[s] for s in sorted(t.keys() & b.keys())]
    _, p = _wilcoxon_signed_rank(deltas)
    return {
        "n": len(deltas),
        "wins": sum(d > 0 for d in deltas),
        "losses": sum(d < 0 for d in deltas),
        "median_delta": statistics.median(deltas) if deltas else 0,
        "p_raw": p,
    }


def analyse(scores: dict, versions: list[str]) -> dict:
    """Synergy comparisons (Holm-adjusted ``p``) plus the A/A control verdict."""
    rep: dict = {}
    keys = [(t, b, v) for t, b in COMPARISONS for v in versions]
    rows = [_compare(scores, t, b, v) for t, b, v in keys]
    for key, row, p in zip(keys, rows, holm([r["p_raw"] for r in rows]), strict=True):
        rep[key] = {**row, "p": p}

    # Unadjusted on purpose: the control must not pass on a technicality.
    ctrl = [_compare(scores, *CONTROL, v) for v in versions]
    for v, row in zip(versions, ctrl, strict=True):
        rep[(*CONTROL, v)] = {**row, "p": row["p_raw"]}
    rep["control_ok"] = all(r["p_raw"] >= ALPHA for r in ctrl)
    return rep


def _print(rep: dict, versions: list[str]) -> None:
    print(f"{'test':<12} {'base':<12} {'version':<28} {'n':>3} {'W/L':>7} {'med Δ':>8} {'p':>7}")
    for t, b in [*COMPARISONS, CONTROL]:
        for v in versions:
            r = rep[(t, b, v)]
            wl = f"{r['wins']}/{r['losses']}"
            print(
                f"{t.value:<12} {b.value:<12} {Path(v).name:<28} {r['n']:>3} {wl:>7} "
                f"{r['median_delta']:>8} {r['p']:>7.4f}"
            )
    if not rep["control_ok"]:
        print("CONTROL FAILED: A/A differs; the comparison is invalid, not the arms.")


# ── CLI ───────────────────────────────────────────────────────────────


def _versions(build_root: Path, pattern: str) -> list[str]:
    return [str(p) for p in sorted(build_root.glob(pattern)) if os.access(p, os.X_OK)]


def cmd_run(args: argparse.Namespace) -> int:
    versions = _versions(args.build_root, args.glob)
    if not versions:
        print(f"no executable {args.glob} under {args.build_root}", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print("--jobs must be >= 1", file=sys.stderr)
        return 2
    if not args.seed_corpus.is_dir():
        print(f"seed corpus not found: {args.seed_corpus}", file=sys.stderr)
        return 2

    try:
        cells = plan(versions, range(args.seeds), args.budget)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    seeds = sorted(str(p) for p in args.seed_corpus.iterdir() if p.is_file())
    manifest = {
        "versions": versions,
        "budget": args.budget,
        "binaries": fingerprint(versions),
        "seed_corpus": fingerprint(seeds),
    }
    print(f"[*] {len(versions)} versions, {len(cells)} campaigns -> {args.out}")
    campaign = make_campaign(args.seed_corpus)
    run(cells, versions, args.work, args.out, campaign, replay, manifest, jobs=args.jobs)
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    results = load(args.rows)
    versions = sorted({v for r in results.values() for v in r})
    rep = analyse(score(results, versions), versions)
    _print(rep, versions)
    return 0 if rep["control_ok"] else 2


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the arms and replay every corpus")
    r.add_argument(
        "--seed-corpus", type=Path, required=True, help="frozen initial seeds (flat dir)"
    )
    r.add_argument(
        "--build-root",
        type=Path,
        default=Path(os.environ.get("FUZZ_BUILD_ROOT", FUZZ_ROOT / "builds")),
    )
    r.add_argument("--glob", default=DEFAULT_GLOB, help="version binaries under --build-root")
    r.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    r.add_argument(
        "--budget", type=int, default=DEFAULT_BUDGET, help="MULTI / FULL execs; SPLIT gets 1/V"
    )
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    r.add_argument("--work", type=Path, default=DEFAULT_WORK)
    r.add_argument("--jobs", type=int, default=1, help="campaigns in parallel (memory-gated)")
    r.set_defaults(fn=cmd_run)

    a = sub.add_parser("analyse", help="paired comparison + A/A control")
    a.add_argument("rows", type=Path, nargs="?", default=DEFAULT_OUT)
    a.set_defaults(fn=cmd_analyse)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
