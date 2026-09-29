#!/usr/bin/env python3
"""Does COMPCOV solve coverage that cmplog does not? (targets/compcov_gates.c)

Arms (same target source, same seed corpus, same exec budget):
  plain    trace-pc-guard build, no cmplog shim           -> edge coverage only
  cmplog   cmplog build, --compcov-level 0                -> input-to-state only
  cc1      cmplog build, --compcov-level 1                -> + const-compare ladders
  cc2      cmplog build, --compcov-level 2                -> + libc byte-walk ladders

Scoring never trusts fuzzer feedback: every corpus file is replayed through
targets/compcov_gates_check (gcc -O0, no shim) which prints
    <solved-mask> d1 d2 d3 d4 d5
where d* is the ground-truth ladder depth reached on each gate. Reported per
run: union mask over the corpus (which gates were solved) and the max depth
per gate (how far each ladder was climbed even when unsolved).

Usage:
    python tools/compcov_gates_experiment.py --execs 12000 --seeds 1 2 3
Results: /tmp/compcov_gates/results.json (appended per run, resumable).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path("/tmp/compcov_gates")
CHECK = REPO / "targets" / "compcov_gates_check"
ARMS = {
    "plain": ("compcov_gates_plain.so", None),
    "cmplog": ("compcov_gates_cmplog.so", 0),
    "cc1": ("compcov_gates_cmplog.so", 1),
    "cc2": ("compcov_gates_cmplog.so", 2),
}
GATES = ["G0 byte", "G1 memcmp", "G2 u32*K", "G3 sum16", "G4 avalanche", "G5 rot13 strncmp"]


def score(corpus: Path) -> dict:
    mask, depth = 0, [0, 0, 0, 0, 0]
    n = 0
    for root, _, files in os.walk(corpus):
        if "deltas" in Path(root).parts:
            continue
        for f in files:
            if f.endswith((".pkl.gz", ".json")):
                continue
            r = subprocess.run([str(CHECK), os.path.join(root, f)], capture_output=True, text=True)
            try:
                v = [int(x) for x in r.stdout.split()]
            except ValueError:
                continue
            if len(v) != 6:
                continue
            n += 1
            mask |= v[0]
            depth = [max(a, b) for a, b in zip(depth, v[1:], strict=True)]
    return {"files": n, "mask": mask, "depth": depth}


def run_one(arm: str, seed: int, execs: int) -> dict:
    so, level = ARMS[arm]
    work = OUT / f"{arm}_s{seed}"
    shutil.rmtree(work, ignore_errors=True)
    (work / "corpus").mkdir(parents=True)
    (work / "corpus" / "seed0").write_bytes(bytes(64))
    cmd = [
        "fuzzer-tool", "fuzz", str(REPO / "targets" / so),
        "--inprocess", "--inprocess-func", "fuzz_shm_run",
        "-d", str(work / "corpus"), "--max-execs", str(execs), "-s", str(seed),
    ]
    if level is not None:
        cmd += ["--compcov-level", str(level)]
    t0 = time.time()
    with open(work / "log.txt", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900, cwd=work)
    res = score(work / "corpus")
    res.update(arm=arm, seed=seed, execs=execs, secs=round(time.time() - t0, 1))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execs", type=int, default=12000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rj = OUT / "results.json"
    done = json.loads(rj.read_text()) if rj.exists() else []
    seen = {(r["arm"], r["seed"], r["execs"]) for r in done}
    for seed in a.seeds:  # seed-major so partial results are balanced across arms
        for arm in a.arms:
            if (arm, seed, a.execs) in seen:
                continue
            r = run_one(arm, seed, a.execs)
            done.append(r)
            rj.write_text(json.dumps(done, indent=1))
            solved = [GATES[g].split()[0] for g in range(6) if r["mask"] >> g & 1]
            print(f"{arm:7s} s{seed} solved={solved} depth={r['depth']} "
                  f"files={r['files']} {r['secs']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
