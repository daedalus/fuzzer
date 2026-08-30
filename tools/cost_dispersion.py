#!/usr/bin/env python3
"""Per-seed execution-cost dispersion on a target, from the cost ledger.

The Boltzmann arm reads its energy from ``effective_fuzz_count``, which is
``total_time / mean_exec``. Under *uniform* per-execution cost that reduces
to ``fuzz_count`` exactly, so the cost and count arms become the same
computation and the A/B returns a null whether or not the change has an
effect -- a false negative dressed as a result. That identity is what
disqualifies the ``locked`` set, and it is a property of the target, not of
the harness, so it has to be checked per target before a cell is spent.

This runs one campaign, reads the persisted cost ledger, and reports how much
per-seed mean execution cost actually varies. A target whose dispersion is
near 1.0x cannot resolve this arm no matter how many replicates it is given.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--flags", default="")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="disp_"))
    cmd = [
        sys.executable, "-m", "fuzzer_tool", "fuzz", args.target,
        "-d", str(workdir), "-c", "-n", str(args.iters), "-s", str(args.seed),
        *args.flags.split(), "--boltzmann",
    ]
    print(f"[*] {Path(args.target).name}: running {args.iters} execs to fill the ledger")
    subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                   timeout=args.timeout, check=False)

    from fuzzer_tool.core.cost_ledger import cost_samples
    from fuzzer_tool.core.state_store import StateStore

    store = StateStore(workdir)
    state = store.get("corpus") or {}
    meta = state.get("seed_meta", {})
    if not meta:
        print("[!] no seed_meta persisted -- cannot measure", file=sys.stderr)
        return 1

    costs = []
    for m in meta.values():
        n = cost_samples(m)
        if n <= 0:
            continue
        total = float(m.get("total_time", 0.0) or 0.0)
        if total > 0:
            costs.append(total / n)

    if len(costs) < 5:
        print(f"[!] only {len(costs)} seeds carry cost samples -- too few", file=sys.stderr)
        return 1

    costs.sort()
    p10 = costs[int(0.10 * (len(costs) - 1))]
    p90 = costs[int(0.90 * (len(costs) - 1))]
    mean = statistics.mean(costs)
    cv = statistics.stdev(costs) / mean if mean else 0.0

    print(f"\n{Path(args.target).name}: {len(costs)} seeds with cost samples")
    print(f"  mean exec   {mean * 1e6:.1f} us")
    print(f"  p90/p10     {p90 / p10 if p10 else float('nan'):.2f}x")
    print(f"  max/min     {costs[-1] / costs[0] if costs[0] else float('nan'):.1f}x")
    print(f"  CV          {cv:.3f}")
    verdict = (
        "can carry the arm" if cv > 0.3
        else "MARGINAL -- arms nearly degenerate here" if cv > 0.1
        else "CANNOT resolve this arm (cost is flat)"
    )
    print(f"  verdict:    {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
