#!/usr/bin/env python3
"""Fixed AFL count ladder vs exact 1-D k-means buckets on real hit counts.

Replays a corpus, histograms the per-exec count of every (edge, exec) pair
(saturated at 255 like the byte map), then compares the 9 AFL classes
(1 2 3 4-7 8-15 16-31 32-63 64-127 128+) with the optimal ``k`` contiguous
buckets from ``core/kmeans1d.py``. Report only: ``--count-aware`` ids are
``id<<8|tag``, so adopting other boundaries is an id-scheme bump.

The DP wins at equal ``k`` by construction. The informative numbers are the
share of variance each leaves unexplained and ``k_match``: the fewest DP
buckets that match the ladder's error.

Usage:
    python tools/count_buckets.py TARGET -d CORPUS [-o hist.pkl] [--scale log|linear]
    python tools/count_buckets.py --load hist.pkl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fuzzer_tool.adapters.edge_pickle import EdgePickleWriter, read_edges_pickle
from fuzzer_tool.core.count_class import classify_single
from fuzzer_tool.core.kmeans1d import Scale, optimal_buckets, partition_cost, to_scale

MAX_COUNT = 255
COUNTS_KEY = "counts"
FREQ_KEY = "freq"
COST_RTOL = 1e-9  # vs total variance: singleton buckets cost ~1e-13 from rounding


class Histogram:
    """How many (edge, exec) pairs hit each per-exec count 1..255."""

    def __init__(self) -> None:
        self._freq = np.zeros(MAX_COUNT + 1, dtype=np.uint64)

    def add(self, counts: dict[int, int]) -> None:
        """Fold one exec's {edge_id: count}."""
        if not counts:
            return
        vals = np.fromiter(counts.values(), dtype=np.int64, count=len(counts))
        vals = np.minimum(vals, MAX_COUNT)
        self._freq += np.bincount(vals, minlength=MAX_COUNT + 1).astype(np.uint64)
        self._freq[0] = 0

    def freq(self) -> np.ndarray:
        return self._freq

    def save(self, path: Path) -> None:
        """Replace *path*: the writer appends and skips keys already present."""
        path.unlink(missing_ok=True)
        writer = EdgePickleWriter(path)
        writer.open()
        writer.write(
            [
                (COUNTS_KEY, np.arange(MAX_COUNT + 1, dtype=np.uint32)),
                (FREQ_KEY, self._freq),
            ]
        )
        writer.close()

    @classmethod
    def load(cls, path: Path) -> Histogram:
        data = read_edges_pickle(path)
        h = cls()
        h._freq[data[COUNTS_KEY].astype(np.intp)] = data[FREQ_KEY]
        return h


def ladder_starts(counts: np.ndarray) -> np.ndarray:
    """Index of the first observed count of each AFL class."""
    classes = np.array([classify_single(int(c)) for c in counts])
    return np.flatnonzero(np.concatenate([[True], classes[1:] != classes[:-1]]))


@dataclass(frozen=True)
class Comparison:
    counts: np.ndarray  # observed distinct per-exec counts
    weights: np.ndarray
    scale: Scale
    k: int
    ladder_starts: np.ndarray
    ladder_cost: float
    dp_starts: np.ndarray
    dp_cost: float
    total_cost: float  # one bucket: all the variance there is
    k_match: int  # fewest DP buckets with cost <= ladder_cost

    def unexplained(self, cost: float) -> float:
        return cost / self.total_cost if self.total_cost > 0 else 0.0


def compare(freq: np.ndarray, scale: Scale, k: int | None = None) -> Comparison:
    """Ladder vs DP on a count histogram; ``k`` defaults to the ladder's."""
    counts = np.flatnonzero(freq[1:]) + 1
    if counts.size == 0:
        raise ValueError("empty histogram")
    weights = freq[counts].astype(np.float64)
    values = to_scale(counts, scale)

    lad = ladder_starts(counts)
    k = lad.size if k is None else k
    if k < 1:
        raise ValueError("k must be >= 1")

    ladder_cost = partition_cost(values, weights, lad)
    dp_starts, dp_cost = optimal_buckets(values, weights, k)
    total = optimal_buckets(values, weights, 1)[1]
    bar = ladder_cost + COST_RTOL * total
    k_match = next(
        m for m in range(1, counts.size + 1) if optimal_buckets(values, weights, m)[1] <= bar
    )
    return Comparison(
        counts, weights, scale, k, lad, ladder_cost, dp_starts, dp_cost, total, k_match
    )


def ranges(counts: np.ndarray, starts: np.ndarray) -> list[str]:
    """Bucket labels 'lo-hi' over the observed counts."""
    ends = np.append(starts[1:], counts.size) - 1
    return [
        str(int(counts[a])) if a == b else f"{int(counts[a])}-{int(counts[b])}"
        for a, b in zip(starts, ends, strict=True)
    ]


def report(r: Comparison) -> None:
    print(
        f"\npairs {int(r.weights.sum())}  distinct counts {r.counts.size}  "
        f"scale {r.scale.value}  k {r.k}"
    )
    print(f"ladder  {' '.join(ranges(r.counts, r.ladder_starts))}")
    print(f"dp      {' '.join(ranges(r.counts, r.dp_starts))}")
    print(
        f"unexplained  ladder {r.unexplained(r.ladder_cost):.4%}  dp {r.unexplained(r.dp_cost):.4%}"
    )
    print(f"k_match {r.k_match}  (DP buckets needed to match the ladder's error)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", nargs="?")
    ap.add_argument("-d", "--corpus", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="save the histogram pickle")
    ap.add_argument("--load", type=Path, help="compare a saved histogram, no replay")
    ap.add_argument("-t", "--timeout", type=float, default=1.0)
    ap.add_argument("-k", type=int, default=None, help="DP buckets (default: ladder's)")
    ap.add_argument("--scale", choices=[s.value for s in Scale], default=Scale.LOG.value)
    args = ap.parse_args()

    if args.load:
        hist = Histogram.load(args.load)
    else:
        if not (args.target and args.corpus):
            ap.error("TARGET and -d CORPUS, or --load")
        from edge_hit_spectrum import Ctx, replay

        hist = Histogram()
        replay(args.target, args.corpus, args.timeout, Ctx.OFF, tap=hist.add)
    if args.out:
        hist.save(args.out)
    report(compare(hist.freq(), Scale(args.scale), args.k))


if __name__ == "__main__":
    main()
