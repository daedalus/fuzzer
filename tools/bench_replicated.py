#!/usr/bin/env python3
"""Replicated, time-paired A/B over a (target, seed) matrix.

Why this exists rather than ``bench_paired.py --arms a,b``:

**Replication.** A cell is not a fixed function of its seed. Measured with
``tools/noise_probe.py`` on the shipped arm: ``png_read.so`` re-runs of the
same cell span 39-66 edges (CV 0.19) and ``jpeg_read.so`` spans 196-209.
46% of same-arm replicate pairs would be scored a win or a loss. Single-shot
cells therefore hand McNemar mostly noise, and a null out of that design
cannot be told from a false negative. Each cell is run ``--reps`` times and
summarised by its median before the arms are compared.

The noise cannot be engineered away. ``effective_fuzz_count`` is
``total_time / mean_exec``, so under constant per-execution cost it reduces
to ``fuzz_count`` exactly -- the identity that disqualifies the ``locked``
set in the Boltzmann handover, and the reason a virtual clock would collapse
the two arms into the same computation. The quantity under test and the
noise are the same phenomenon, so the only lever is repetition.

**Pairing in time.** ``bench_paired.py`` runs one arm to completion, then the
other. With hours between them, any drift in machine state is confounded with
the arm. Here the two arms run back to back within a replicate, so a drift
has to happen inside a single pair to bias anything.

**Two source trees.** The arms differ by a code edit, not a flag
(``UNWIRED_ARMS``), so each arm names a source tree and is selected by
``PYTHONPATH`` rather than by editing a file between runs. That removes the
mislabelling risk the handover warns about -- an arm cannot be misrecorded
because the tree it ran from is what defines it -- and it is what makes
interleaving affordable at all.

Targets are named explicitly. ``zlib``, ``lz4`` and ``gzip`` were measured
bit-for-bit deterministic and saturated (12, 12 and 36 edges on every
replicate), so they contribute no discordant pairs in either direction and
their cells are budget spent on a guaranteed tie.
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

import bench_lock  # noqa: E402
from eval_set import TARGET_SETS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "paired"

_EDGES = re.compile(r"Edges discovered:\s+(\d+)")
_CORPUS = re.compile(r"Corpus:\s+(\d+)")
_CRASHES = re.compile(r"Crashes:\s+(\d+)")


def run_one(src: str, target: str, flags: str, seed: int, iters: int, timeout: int) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix="rep_"))
    env = dict(os.environ)
    env["PYTHONPATH"] = src
    cmd = [
        sys.executable, "-m", "fuzzer_tool", "fuzz", target,
        "-d", str(workdir), "-c", "-n", str(iters), "-s", str(seed),
        *flags.split(), "--boltzmann",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO, capture_output=True, text=True,
            timeout=timeout, check=False, env=env,
        )
        log = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        log = (exc.stdout or b"").decode(errors="replace") if exc.stdout else ""
        rc = -1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    def one(rx):
        m = rx.search(log)
        return int(m.group(1)) if m else 0

    return {
        "secs": round(time.time() - t0, 2),
        "rc": rc,
        "edges": one(_EDGES),
        "corpus": one(_CORPUS),
        "crashes": one(_CRASHES),
        # A cell that lost coverage is a broken run, not a zero-edge arm
        # result, and must not be pooled with real outcomes.
        "coverage_attached": _EDGES.search(log) is not None and one(_EDGES) > 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", default="direct_lite")
    ap.add_argument("--targets", required=True, help="comma-separated .so basenames")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--src-a", required=True, help="source tree for arm A")
    ap.add_argument("--src-b", required=True, help="source tree for arm B")
    ap.add_argument("--name-a", default="boltzmann-count")
    ap.add_argument("--name-b", default="boltzmann-cost")
    ap.add_argument("--out", default="results/paired/replicated.json")
    bench_lock.add_argument(ap)
    args = ap.parse_args()

    # Held for the whole run. Taken before any target resolution so a
    # second harness is refused immediately rather than after doing work.
    lock = bench_lock.engage(args.lock_single_thread)

    spec = {Path(t).name: (t, f) for t, f in TARGET_SETS[args.set_name]}
    chosen = []
    for n in args.targets.split(","):
        if n not in spec:
            print(f"[!] {n} not in set {args.set_name}", file=sys.stderr)
            return 2
        if not (REPO / spec[n][0]).exists():
            print(f"[!] {n} not built", file=sys.stderr)
            return 2
        chosen.append(spec[n])

    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    have = {(r["arm"], r["target"], r["seed"], r["rep"]) for r in rows}

    arms = [(args.name_a, args.src_a), (args.name_b, args.src_b)]
    total = len(chosen) * len(seeds) * args.reps * 2
    done = 0
    for target, flags in chosen:
        for seed in seeds:
            for rep in range(args.reps):
                # Arms run back to back inside the replicate so that machine
                # drift cannot separate them.
                for arm, src in arms:
                    done += 1
                    if (arm, target, seed, rep) in have:
                        continue
                    r = run_one(src, target, flags, seed, args.iters, args.timeout)
                    r.update(arm=arm, target=target, seed=seed, rep=rep, src=src)
                    rows.append(r)
                    flag = "" if r["coverage_attached"] else "  [NO COVERAGE]"
                    print(
                        f"  [{done:>4}/{total}] {arm:<17} {Path(target).name:<15} "
                        f"s={seed:<3} r={rep} edges={r['edges']:<5} "
                        f"{r['secs']:>6.1f}s{flag}",
                        flush=True,
                    )
                    tmp = out.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(rows, indent=1))
                    tmp.replace(out)

    print(f"\n[*] wrote {out} ({len(rows)} runs)")
    summarise(rows, args.name_a, args.name_b)
    if lock:
        lock.release()
    return 0


def summarise(rows: list[dict], name_a: str, name_b: str) -> None:
    by: dict = {}
    for r in rows:
        by.setdefault((r["target"], r["seed"]), {}).setdefault(r["arm"], []).append(r["edges"])

    print(f"\n{'target':<16} {'seed':>4} {name_a:>16} {name_b:>16} {'delta':>7}")
    deltas = []
    for (target, seed), arms in sorted(by.items()):
        a, b = arms.get(name_a, []), arms.get(name_b, [])
        if not a or not b:
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        d = mb - ma
        deltas.append((target, seed, ma, mb, d))
        print(f"{Path(target).name:<16} {seed:>4} {ma:>16.1f} {mb:>16.1f} {d:>+7.1f}")

    if not deltas:
        return
    for target in sorted({t for t, _, _, _, _ in deltas}):
        sub = [d for t, _, _, _, d in deltas if t == target]
        wins = sum(1 for d in sub if d > 0)
        losses = sum(1 for d in sub if d < 0)
        ties = sum(1 for d in sub if d == 0)
        print(
            f"\n{Path(target).name}: {name_b} wins {wins}, loses {losses}, ties {ties}"
            f" | median delta {statistics.median(sub):+.1f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
