#!/usr/bin/env python3
"""Sweep: constant vs Grover M/N-adaptive QEA rotation angle on a synthetic problem.

Problem: n input bits, M of them "live" with a hidden target value; fitness is
the number of live bits matching the target, dead bits are irrelevant. This is
the minimal model of "few input bits matter" the M/N angle assumes. It is NOT
a coverage benchmark -- it isolates the step-size rule from everything else.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from factorial_design import screen  # noqa: E402

from fuzzer_tool.core.qea import _uniform_amplitudes, collapse, rotation_gate
from fuzzer_tool.core.qea_grover import LiveFractionTracker
from fuzzer_tool.core.rand_pool import RandPool


def run(n_bytes: int, m_live: int, mode: str, delta: float, seed: int, cap: int) -> int | None:
    rng = np.random.default_rng(seed)
    pool = RandPool(seed=seed)
    n = n_bytes * 8
    live = rng.choice(n, size=m_live, replace=False)
    target = rng.integers(0, 2, size=n)

    def fit(data: bytes) -> int:
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return int((bits[live] == target[live]).sum())

    amps = _uniform_amplitudes(n)
    best = collapse(amps, pool)
    best_f = fit(best)
    tracker = LiveFractionTracker(n)
    for evals in range(1, cap + 1):
        x = collapse(amps, pool)
        fx = fit(x)
        if fx == m_live:
            return evals
        if mode == "grover" and fx > best_f:
            tracker.observe_improvement(int.from_bytes(best, "big"), int.from_bytes(x, "big"))
        d = tracker.angle(delta) if mode == "grover" else delta
        improved = fx >= best_f
        rotation_gate(amps, x, improved=improved, best=best, delta=d)
        if improved:
            best, best_f = x, fx
    return None


# Two-level ranges for --screen; response is median evals-to-solve.
_SCREEN_FACTORS = {
    "delta": (0.02, 0.20),
    "n_bytes": (16, 64),
    "m_live": (2, 32),
}


def _median_evals(cfg: dict[str, float], mode: str, trials: int, cap: int) -> float:
    runs = [
        run(int(cfg["n_bytes"]), int(cfg["m_live"]), mode, cfg["delta"], s, cap)
        for s in range(trials)
    ]
    return statistics.median(cap if r is None else r for r in runs)


def screen_report(mode: str, trials: int, cap: int) -> str:
    """Rank delta / n_bytes / m_live by main effect on median evals (Plackett-Burman)."""
    ranked = screen(_SCREEN_FACTORS, lambda c: _median_evals(c, mode, trials, cap))
    lines = [f"screen mode={mode} trials={trials} cap={cap} (effect = high - low, evals)"]
    lines += [f"{e.name:>8} {e.effect:>+10.0f}" for e in ranked]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bytes", type=int, default=32)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--cap", type=int, default=20000)
    ap.add_argument("--m-live", type=int, nargs="+", default=[2, 8, 32, 128])
    ap.add_argument(
        "--screen",
        choices=("c", "grover"),
        help="rank delta/n_bytes/m_live by main effect (4 runs) instead of the grid",
    )
    a = ap.parse_args()
    if a.screen:
        print(screen_report(a.screen, a.trials, a.cap))
        return
    n = a.n_bytes * 8
    print(f"n_bits={n} trials={a.trials} cap={a.cap} (median evals to solve; fail=cap)")
    print(
        f"{'M':>4} {'M/N':>7} | "
        + " ".join(f"{k:>12}" for k in ("c=0.02", "c=0.05", "c=0.20", "grover"))
    )
    for m in a.m_live:
        cells = []
        for mode, d in (("c", 0.02), ("c", 0.05), ("c", 0.20), ("grover", 0.05)):
            res = [run(a.n_bytes, m, mode, d, s, a.cap) for s in range(a.trials)]
            fails = sum(r is None for r in res)
            med = statistics.median(a.cap if r is None else r for r in res)
            cells.append(f"{med:>7.0f}/{fails:<3}f")
        print(f"{m:>4} {m / n:>7.4f} | " + " ".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    main()
