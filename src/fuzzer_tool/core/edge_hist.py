"""Seeds-per-edge-count histogram and edges/seed ratio (``--edge-hist``).

Per-seed edge counts span orders of magnitude, so bins are powers of two:

    edges per seed:   0   1   2-3   4-7   8-15 ...
    bin index:        0   1   2     3     4

    counts [0, 1, 3, 4] -> one seed in each of bins 0, 1, 2, 3
    ratio = unique edges / seeds
"""

from collections.abc import Iterable

import numpy as np

BAR_WIDTH = 40
BAR_CHAR = "#"
FUZZ_TITLE = "fuzz"

# Bin i >= 1 holds counts in [2**(i-1), 2**i); the bin of n is
# searchsorted(_POW2, n, "right"). 2**62 is the largest int64 power of two.
_POW2 = np.left_shift(1, np.arange(63, dtype=np.int64))


def bin_counts(counts: Iterable[int]) -> np.ndarray:
    """Seeds per power-of-two bin; index = ``bin_label`` order."""
    arr = np.fromiter(counts, dtype=np.int64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.min() < 0:
        raise ValueError("negative edge count")

    return np.bincount(np.searchsorted(_POW2, arr, side="right"))


def bin_label(idx: int) -> str:
    """``0``, ``1``, ``2-3``, ``4-7``, ..."""
    if idx < 2:
        return str(idx)
    return f"{1 << (idx - 1)}-{(1 << idx) - 1}"


def _bar(n: int, peak: int) -> str:
    """Bar of n/peak * BAR_WIDTH chars; a nonempty bin is never invisible."""
    if n == 0:
        return ""
    return BAR_CHAR * max(1, round(n / peak * BAR_WIDTH))


def hist_report(
    counts: list[int],
    unique_edges: int,
    title: str,
    population: int | None = None,
) -> str:
    """Report text; *population* is the corpus size when *counts* cover less."""
    lines = ["", f"--- Edge Histogram ({title}) ---"]
    if not counts:
        lines.append("  No seeds with edge data.")
        return "\n".join(lines)

    arr = np.asarray(counts, dtype=np.int64)
    seeds = f"{len(counts)}"
    if population is not None and population > len(counts):
        seeds += f" (of {population} in corpus)"

    lines += [
        f"  {'Seeds:':<16s}{seeds}",
        f"  {'Unique edges:':<16s}{unique_edges}",
        f"  {'Edges/seed:':<16s}{unique_edges / len(counts):.2f}",
        f"  {'Per-seed edges:':<16s}mean {arr.mean():.1f}  "
        f"median {np.median(arr):g}  max {arr.max()}",
        "",
        f"  {'Edges':>10s}  {'Seeds':>6s}",
    ]

    bins = bin_counts(counts)
    peak = int(bins.max())
    first = int(np.flatnonzero(bins)[0])
    for i in range(first, len(bins)):
        n = int(bins[i])
        lines.append(f"  {bin_label(i):>10s}  {n:>6d}  {_bar(n, peak)}".rstrip())

    return "\n".join(lines)


def fuzz_hist(fuzzer) -> str:
    """Histogram over the edge tracker's per-seed edge sets.

    The tracker holds at most ``MAX_TRACKED_SEEDS`` seeds, so the corpus
    size is passed as *population* to flag a partial view.
    """
    seed_edges = fuzzer._edge_tracker.seed_edges
    counts = [len(edges) for edges in seed_edges.values()]
    unique = len(set().union(*seed_edges.values()))

    return hist_report(counts, unique, FUZZ_TITLE, population=len(fuzzer.corpus))
