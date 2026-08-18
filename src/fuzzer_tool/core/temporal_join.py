"""K-way sliding-window temporal join for independently-timestamped streams.

Port of LaurieWired/tailslayer's ``discovery/benchmark/benchmark.cpp``
``pair_samples_n`` (lines 206-257), per handover item 7.

Greedy single-pass join: one read pointer per stream; match when the
spread across all current timestamps is within ``max_gap``; otherwise
advance only laggard pointers.

Output is always a list of matched tuples, one element per stream, so
callers see every aligned payload rather than a caller-specific collapse.
A ``value_fn`` post-processing step is intentionally omitted from the
core join because the correct representative depends on domain semantics
(min latency for replicated sensors, first-stream-wins for heterogeneous
channels, median for noisy data, etc.) and is cheaper to apply after
joining than to re-run the join with different collapse policies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def join_streams(
    streams: Sequence[Sequence[tuple[float, T]]],
    max_gap: float,
) -> list[tuple[T, ...]]:
    """Greedy K-way join of timestamped streams within a tolerance window.

    Args:
        streams: One timestamped stream per sequence element. Each stream
            must be sorted by timestamp ascending. Timestamps are floats
            in any comparable unit; ``max_gap`` is interpreted in the same
            unit.
        max_gap: Maximum allowed spread between the earliest and latest
            current timestamp across all streams for a match to be
            accepted.

    Returns:
        A list of matched tuples, one per accepted window. Each tuple
        contains one payload per stream, in stream order. Streams that
        never fall within ``max_gap`` of the others produce no output.

    Notes:
        - An empty or single-stream input returns ``[]`` immediately: there
          is nothing to join.
        - Any stream that is empty or out of order produces undefined
          behavior. The caller owns sorting and validation.
        - The algorithm is **greedy and non-optimal**: a laggard advance
          can skip a better match for the fast stream. This is documented
          rather than hidden because the source's own ``n_paired``
          accounting makes the limitation observable.
    """
    if len(streams) <= 1:
        return []

    for stream in streams:
        if not stream:
            return []

    pointers = [0] * len(streams)
    results: list[tuple[T, ...]] = []

    while True:
        for i, stream in enumerate(streams):
            if pointers[i] >= len(stream):
                return results

        best_ts = streams[0][pointers[0]][0]
        for i in range(1, len(streams)):
            ts = streams[i][pointers[i]][0]
            if ts < best_ts:
                best_ts = ts

        current_ts = [streams[i][pointers[i]][0] for i in range(len(streams))]
        current_vals = [streams[i][pointers[i]][1] for i in range(len(streams))]

        spread = max(current_ts) - min(current_ts)
        if spread <= max_gap:
            results.append(tuple(current_vals))
            for i in range(len(streams)):
                pointers[i] += 1
        else:
            for i in range(len(streams)):
                if current_ts[i] < best_ts + max_gap:
                    pointers[i] += 1
