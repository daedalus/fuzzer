#!/usr/bin/env python3
"""Measure how the byte-position sampling distribution affects edge discovery.

Isolates ONE variable: which byte offset a mutation targets.

`OperatorEngine.select_position` normally returns a position from the
MI / transfer-entropy / byte-sensitivity trackers when they have data, and
falls back to `randint(0, len-1)` — uniform — otherwise. This harness
monkeypatches that method so every run uses exactly one distribution, then
counts distinct edges discovered under a fixed execution budget.

Distributions:
  uniform   — the current fallback; every offset equally likely.
  head      — Zipf-like bias toward low offsets. Container formats put
              magic/length/type fields near the front, so header bytes
              gate the most branches.
  edges     — U-shaped (beta(0.5,0.5)); favours both extremes. Trailers
              (checksums, IEND) matter as much as headers in many formats.
  gauss     — locality: cluster near the previously chosen offset, so
              multi-byte fields get hit together rather than one byte at
              a time.
  native    — no patch; the real tracker-driven behaviour, as a control.

Run: python3 tools/distribution_experiment.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fuzzer_tool.services.fuzzer import Fuzzer  # noqa: E402
from fuzzer_tool.services.operators import OperatorEngine  # noqa: E402

_ORIGINAL_SELECT = OperatorEngine.select_position


def _make_uniform():
    def select(self, buf, data):
        n = len(buf)
        return self.f._rand_pool.randint(0, n - 1) if n else 0

    return select


def _make_head(exponent: float = 1.5):
    """Zipf-like: P(i) ~ 1/(i+1)^exponent, normalised over the buffer."""

    def select(self, buf, data):
        n = len(buf)
        if not n:
            return 0
        u = self.f._rand_pool.random()
        # inverse-CDF of a bounded power law
        return min(n - 1, int((1 - u) ** (-1.0 / exponent) - 1))

    return select


def _make_edges():
    """U-shaped beta(0.5, 0.5) via its closed-form inverse CDF."""

    def select(self, buf, data):
        n = len(buf)
        if not n:
            return 0
        u = self.f._rand_pool.random()
        x = math.sin(0.5 * math.pi * u) ** 2  # arcsine distribution
        return min(n - 1, int(x * n))

    return select


def _make_gauss(sigma_frac: float = 0.05):
    """Locality: sample near the last chosen offset."""

    def select(self, buf, data):
        n = len(buf)
        if not n:
            return 0
        last = getattr(self, "_dist_last_pos", n // 2)
        sigma = max(1.0, sigma_frac * n)
        pos = int(random.gauss(last, sigma)) % n
        self._dist_last_pos = pos
        return pos

    return select


DISTRIBUTIONS = {
    "native": None,
    "uniform": _make_uniform(),
    "head": _make_head(),
    "edges": _make_edges(),
    "gauss": _make_gauss(),
}


def run_one(target: str, corpus: str, func: str, execs: int, seed: int, dist: str) -> dict:
    patch = DISTRIBUTIONS[dist]
    OperatorEngine.select_position = patch or _ORIGINAL_SELECT
    random.seed(seed)

    t0 = time.time()
    f = Fuzzer(
        target=target,
        corpus_dir=corpus,
        crashes_dir="/tmp/dist_crashes",
        use_coverage=True,
        inprocess=True,
        inprocess_func=func,
        inprocess_direct=True,
        seed=seed,
        quiet_stats=True,
    )
    try:
        f.run(iterations=execs)
    finally:
        OperatorEngine.select_position = _ORIGINAL_SELECT

    edges = 0
    if getattr(f, "shm_cov", None):
        edges = getattr(f.shm_cov, "_peak_cumulative_edges", 0) or 0
    return {
        "dist": dist,
        "seed": seed,
        "edges": edges,
        "execs": getattr(f, "exec_count", execs),
        "secs": round(time.time() - t0, 1),
        "corpus": len(getattr(f, "corpus", []) or []),
    }


def _run_in_subprocess(target, corpus, func, execs, seed, dist) -> dict:
    """Each trial gets a fresh interpreter.

    Coverage SHM is attached once per process; a second Fuzzer built in the
    same process silently reports zero edges. Isolating trials keeps the
    measurement honest.
    """
    import shutil
    import subprocess
    import tempfile

    # Each trial gets a pristine copy of the seed corpus. Sharing one
    # directory lets seeds accumulate across trials, so later runs start
    # from a richer corpus and score higher regardless of distribution —
    # which silently turns the experiment into a measure of run order.
    workdir = tempfile.mkdtemp(prefix="distexp_")
    trial_corpus = str(Path(workdir) / "corpus")
    shutil.copytree(corpus, trial_corpus)

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
            "--dists",
            dist,
        ],
        capture_output=True,
        text=True,
    )
    try:
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[len("RESULT ") :])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {
        "dist": dist,
        "seed": seed,
        "edges": 0,
        "execs": 0,
        "secs": 0,
        "corpus": 0,
        "error": proc.stderr[-300:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--target", default="/tmp/png_scov.so")
    ap.add_argument("--corpus", default="/tmp/png_corpus")
    ap.add_argument("--func", default="fuzz_shm_run")
    ap.add_argument("--execs", type=int, default=6000)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--dists", default=",".join(DISTRIBUTIONS))
    args = ap.parse_args()

    if args.worker:
        r = run_one(args.target, args.corpus, args.func, args.execs, args.seed, args.dists)
        print("RESULT " + json.dumps(r))
        return 0

    rows = []
    for dist in args.dists.split(","):
        for trial in range(args.trials):
            r = _run_in_subprocess(
                args.target, args.corpus, args.func, args.execs, 1000 + trial, dist
            )
            rows.append(r)
            print(
                f"  {dist:8s} seed={r['seed']} edges={r['edges']:4d} "
                f"corpus={r['corpus']:3d} {r['secs']}s",
                flush=True,
            )

    print("\n=== distribution vs edges discovered ===")
    print(f"{'dist':10s} {'mean':>7s} {'median':>7s} {'min':>5s} {'max':>5s}")
    summary = {}
    for dist in args.dists.split(","):
        vals = [r["edges"] for r in rows if r["dist"] == dist]
        if not vals:
            continue
        summary[dist] = vals
        print(
            f"{dist:10s} {statistics.mean(vals):7.1f} "
            f"{statistics.median(vals):7.1f} {min(vals):5d} {max(vals):5d}"
        )

    base = summary.get("uniform")
    if base:
        bm = statistics.mean(base)
        print(f"\nrelative to uniform (mean {bm:.1f}):")
        for dist, vals in summary.items():
            if dist == "uniform":
                continue
            m = statistics.mean(vals)
            print(f"  {dist:10s} {m / bm if bm else 0:5.2f}x  ({m:.1f})")

    Path("/tmp/dist_experiment.json").write_text(json.dumps(rows, indent=2))
    print("\nraw results: /tmp/dist_experiment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
