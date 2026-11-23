#!/usr/bin/env python3
"""Can the fuzzer find the planted bugs in targets/p6_flags_diff.c, and stay quiet without them?

Arms (same source, same seeds, same exec budget):
  control  no planted bug            -> must report 0 crashes (Hard Rule 46)
  bug1..4  -DP6_BUG=n                -> must report > 0 crashes

A crash is a replayable MISMATCH abort; the control arm is what proves a crash
means a model/hardware disagreement and not a harness fault.

Usage:
    python tools/p6_flags_experiment.py --execs 20000 --seed 1
Exit status 1 when the control crashes or a bug arm finds nothing.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "targets" / "p6_flags_diff.c"
SHIM = REPO / "src" / "fuzzer_tool" / "adapters" / "afl_shim.c"
OUT = Path.home() / "p6_flags_experiment"
CRASH_ROOT = Path.home() / "fuzzing"
ARMS = {"control": 0, "bug1": 1, "bug2": 2, "bug3": 3, "bug4": 4}
NOPS = 25
REC = "<BBBBII"
SEED_CTL = 2 | 1 << 4  # 32-bit, rm=eax, reg=ecx
SEED_A, SEED_B = 0x80000001, 0x7FFFFFFF


def build(arm: str, bug: int) -> Path:
    exe = OUT / f"p6_{arm}"
    cmd = [
        "clang", "-O1", "-g", "-fsanitize-coverage=trace-pc-guard", "-include", str(SHIM),
        f"-DP6_BUG={bug}", "-o", str(exe), str(SRC),
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"build {arm} failed: {r.stderr[:400]}")
    return exe


def seeds(d: Path) -> None:
    d.mkdir(parents=True)
    for op in range(NOPS):
        (d / f"op{op}").write_bytes(struct.pack(REC, op, SEED_CTL, 0, 3, SEED_A, SEED_B))


def crashes(exe: Path) -> int:
    d = CRASH_ROOT / exe.name / "crashes"
    return len(list(d.glob("*.bin"))) if d.is_dir() else 0


def run_arm(arm: str, bug: int, execs: int, seed: int) -> int:
    exe = build(arm, bug)
    work = OUT / arm
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(CRASH_ROOT / exe.name, ignore_errors=True)
    seeds(work / "corpus")
    cmd = ["fuzzer-tool", "fuzz", str(exe), "-d", str(work / "corpus"),
           "--max-execs", str(execs), "-s", str(seed)]  # fmt: skip
    with open(work / "log.txt", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900, cwd=work)
    return crashes(exe)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    bad = 0
    for arm in args.arms:
        n = run_arm(arm, ARMS[arm], args.execs, args.seed)
        ok = n == 0 if arm == "control" else n > 0
        bad += not ok
        print(f"{arm:8s} crashes={n:5d} {'ok' if ok else 'FAIL'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
