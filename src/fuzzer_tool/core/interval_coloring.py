"""Optimal coloring of half-open byte intervals (interval graphs).

Two spans [s, e) conflict when they share a byte. Interval graphs are
perfect, so greedy by left endpoint, reusing the lowest freed color, uses
exactly as many colors as the deepest point (the max clique).

    [0,4) [2,6) [8,12)     depth 2 at bytes 2-3
    color 0     1     0    -> classes [[0, 2], [1]]

A class is a set of pairwise disjoint spans: writes into them commute, so
one exec can apply a whole class without one write clobbering another.

Sequential by nature (each choice depends on what is still open); the
spans per call are few (redqueen keeps <= 50 matches per seed), so a
heap loop beats numpy call overhead.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence


def color_intervals(starts: Sequence[int], ends: Sequence[int]) -> list[int]:
    """Color per span, input order; overlapping spans never share one.

    Raises:
        ValueError: lengths differ, or a span is empty (end <= start).
    """
    n = len(starts)
    if len(ends) != n:
        raise ValueError("starts and ends differ in length")
    if any(e <= s for s, e in zip(starts, ends, strict=True)):
        raise ValueError("empty interval: end must exceed start")

    colors = [0] * n
    busy: list[tuple[int, int]] = []  # (end, color) of open spans
    free: list[int] = []  # colors whose span has ended
    for i in sorted(range(n), key=starts.__getitem__):
        # Release every span that ended at or before this start.
        s = starts[i]
        while busy and busy[0][0] <= s:
            heapq.heappush(free, heapq.heappop(busy)[1])

        c = heapq.heappop(free) if free else len(busy)
        colors[i] = c
        heapq.heappush(busy, (ends[i], c))
    return colors


def color_classes(starts: Sequence[int], ends: Sequence[int]) -> list[list[int]]:
    """Span indices grouped by color, ascending within each class."""
    colors = color_intervals(starts, ends)
    classes: list[list[int]] = [[] for _ in range(max(colors, default=-1) + 1)]
    for i, c in enumerate(colors):
        classes[c].append(i)
    return classes
