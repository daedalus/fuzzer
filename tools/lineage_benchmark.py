#!/usr/bin/env python3
"""Benchmark --lineage: does the mutation lineage tree pay for itself?

The lineage tree records parent/child relationships between corpus seeds
and feeds a diversity multiplier into seed selection (seed_picker.py:676 —
seeds whose lineage is distant from their peers get weighted up). It costs
memory and per-insert bookkeeping, so the question is whether the resulting
seed choices actually discover more edges.

Measures, with lineage ON vs OFF:
  * distinct edges discovered under a fixed execution budget
  * throughput (eps), to price the bookkeeping
  * corpus size and observed lineage depth

Isolation matters and is easy to get wrong:
  * each trial runs in a fresh interpreter — coverage SHM attaches once
    per process, so a second in-process Fuzzer silently reports 0 edges;
  * each trial gets a pristine copy of the seed corpus — a shared corpus
    accumulates across trials, so later runs start richer and score higher
    regardless of the setting being tested.

Run: python3 tools/lineage_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_worker(
    target: str,
    corpus: str,
    func: str,
    execs: int,
    seed: int,
    lineage: bool,
    backtrack: bool = False,
) -> dict:
    from fuzzer_tool.services.fuzzer import Fuzzer

    t0 = time.time()
    f = Fuzzer(
        target=target,
        corpus_dir=corpus,
        crashes_dir=tempfile.mkdtemp(prefix="lin_crash_"),
        use_coverage=True,
        inprocess=True,
        inprocess_func=func,
        inprocess_direct=True,
        seed=seed,
        lineage=lineage,
        lineage_backtrack=backtrack,
        quiet_stats=True,
    )
    f.run(iterations=execs)
    elapsed = time.time() - t0

    edges = 0
    if getattr(f, "shm_cov", None):
        edges = getattr(f.shm_cov, "_peak_cumulative_edges", 0) or 0
    depths = [m.get("lineage_depth", 0) for m in getattr(f, "seed_meta", {}).values()]
    tree = getattr(f, "_lineage", None)
    return {
        "lineage": lineage,
        "backtrack": backtrack,
        "seed": seed,
        "edges": edges,
        "execs": getattr(f, "exec_count", execs),
        "eps": round(getattr(f, "exec_count", execs) / elapsed, 1) if elapsed else 0,
        "secs": round(elapsed, 1),
        "corpus": len(getattr(f, "corpus", []) or []),
        "max_depth": max(depths) if depths else 0,
        "tree_nodes": len(tree) if tree is not None else 0,
    }


def run_isolated(target, corpus, func, execs, seed, lineage, backtrack=False) -> dict:
    workdir = tempfile.mkdtemp(prefix="linbench_")
    trial_corpus = str(Path(workdir) / "corpus")
    shutil.copytree(corpus, trial_corpus)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker",
                "--target",
                target,
                "--corpus",
                trial_corpus,
                "--func",
                func,
                "--execs",
                str(execs),
                "--seed",
                str(seed),
            ]
            + (["--lineage"] if lineage else [])
            + (["--backtrack"] if backtrack else []),
            capture_output=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT ") :])
        return {
            "lineage": lineage,
            "seed": seed,
            "edges": 0,
            "eps": 0,
            "corpus": 0,
            "max_depth": 0,
            "tree_nodes": 0,
            "error": proc.stderr[-300:],
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _summarize(rows, key):
    on = [r[key] for r in rows if r["lineage"]]
    off = [r[key] for r in rows if not r["lineage"]]
    return on, off


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--target", default="/tmp/png_scov.so")
    ap.add_argument("--corpus", default="/tmp/png_seed")
    ap.add_argument("--func", default="fuzz_shm_run")
    ap.add_argument("--execs", type=int, default=10000)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--lineage", action="store_true")
    ap.add_argument("--backtrack", action="store_true")
    args = ap.parse_args()

    if args.worker:
        r = run_worker(
            args.target, args.corpus, args.func, args.execs, args.seed, args.lineage, args.backtrack
        )
        print("RESULT " + json.dumps(r))
        return 0

    arms = [("baseline", False, False), ("lineage", True, False), ("backtrack", True, True)]
    rows = []
    for label, lineage, backtrack in arms:
        for t in range(args.trials):
            r = run_isolated(
                args.target, args.corpus, args.func, args.execs, 1000 + t, lineage, backtrack
            )
            r["arm"] = label
            rows.append(r)
            print(
                f"  {label:9s} seed={r['seed']} edges={r['edges']:4d} "
                f"eps={r['eps']:7.1f} corpus={r['corpus']:3d} "
                f"depth={r['max_depth']} nodes={r['tree_nodes']}",
                flush=True,
            )

    print("\n=== arms ===")
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    for key in ("edges", "eps", "corpus", "max_depth"):
        print(f"-- {key}")
        base = [r[key] for r in by_arm.get("baseline", [])]
        for arm, rs in by_arm.items():
            v = [r[key] for r in rs]
            m = statistics.mean(v)
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            extra = ""
            if base and arm != "baseline":
                mb = statistics.mean(base)
                sb = statistics.stdev(base) if len(base) > 1 else 0.0
                se = ((sd**2 / len(v)) + (sb**2 / len(base))) ** 0.5
                extra = f"  delta={m - mb:+7.1f}  {abs(m - mb) / se if se else 0:.1f} SE"
            print(f"   {arm:10s} {m:8.1f} (sd {sd:5.1f}){extra}")

    Path("/tmp/lineage_benchmark.json").write_text(json.dumps(rows, indent=2))
    print("\nraw results: /tmp/lineage_benchmark.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
