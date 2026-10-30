#!/usr/bin/env python3
"""Benchmark: detect_periodicity false-positive rate on known nulls.

Regenerates the raw vs pre-whitened table in ``detect_periodicity``'s
docstring. Each replicate draws one series and scores it both ways, so the
two columns are paired. Every (null, n) cell gets its own seeded RandPool.

    null              n      raw  pre-whitened
    white           512    0.046         0.046
    ar1 phi=0.7     512    0.936         0.069
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fuzzer_tool.core.periodicity import detect_periodicity  # noqa: E402
from fuzzer_tool.core.rand_pool import RandPool  # noqa: E402
from fuzzer_tool.core.synthetic_series import ar1, cox, garch, pink  # noqa: E402

MIN_SAMPLES = 50  # same gate as tests/test_regression_periodicity_null.py
DEFAULT_NS = (256, 512)
DEFAULT_REPS = 500
DEFAULT_SEED = 20260913
DEFAULT_ALPHA = 0.05

Gen = Callable[[RandPool, int], np.ndarray]

# label -> series generator; the docstring table's rows plus GARCH.
NULLS: tuple[tuple[str, Gen], ...] = (
    ("white", lambda r, n: r.normal_array(n)),
    ("cox (OU rate)", cox),
    ("ar1 phi=0.7", lambda r, n: ar1(r, n, 0.7)),
    ("ar1 phi=0.9", lambda r, n: ar1(r, n, 0.9)),
    ("ar1 phi=-0.6", lambda r, n: ar1(r, n, -0.6)),
    ("pink 1/f", lambda r, n: pink(r, n, 1.0)),
    ("pink 1/f^2", lambda r, n: pink(r, n, 2.0)),
    ("garch .05/.15/.80", lambda r, n: garch(r, n, 0.05, 0.15, 0.80)),
)


def null_rates(gen: Gen, n: int, reps: int, seed: int, alpha: float) -> tuple[float, float]:
    """(raw, pre-whitened) false-positive rates of *gen* over *reps* series."""
    rng = RandPool(seed=seed)
    raw = pw = 0
    for _ in range(reps):
        series = gen(rng, n).tolist()
        raw += detect_periodicity(
            series, min_samples=MIN_SAMPLES, alpha=alpha, prewhiten_series=False
        ).significant
        pw += detect_periodicity(series, min_samples=MIN_SAMPLES, alpha=alpha).significant
    return raw / reps, pw / reps


def _at_least(lo: int) -> Callable[[str], int]:
    """argparse type: int >= *lo*."""

    def parse(text: str) -> int:
        value = int(text)
        if value < lo:
            raise argparse.ArgumentTypeError(f"must be >= {lo}, got {value}")
        return value

    return parse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reps", type=_at_least(1), default=DEFAULT_REPS)
    ap.add_argument("--n", type=_at_least(MIN_SAMPLES), nargs="+", default=list(DEFAULT_NS))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = ap.parse_args(argv)

    width = max(len(label) for label, _ in NULLS)
    print(f"{'null':<{width}} {'n':>5} {'raw':>7} {'pre-whitened':>13}")

    # Cell k gets seed + k: cells are independent, reruns are exact.
    cell = 0
    for label, gen in NULLS:
        for n in args.n:
            raw, pw = null_rates(gen, n, args.reps, args.seed + cell, args.alpha)
            print(f"{label:<{width}} {n:>5} {raw:>7.3f} {pw:>13.3f}", flush=True)
            cell += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
