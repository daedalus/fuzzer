#!/usr/bin/env python3
"""Benchmark the havoc sub-mutation sampler against the uniform draw.

Two measurements:

1. *Isolated* — the selection block only (draw -> branch index -> counters),
   with no mutation work attached. This is what picked the implementation:
   a precomputed 256-slot inverse-CDF table beat a bisect over an 11-float
   CDF, and plain int lists beat ``array("d")`` counters at this size.
2. *End to end* — full ``_apply_single_mutation`` calls with the flag on and
   off, which is the number that matters for a throughput decision: the
   selection cost is amortized against whatever the chosen branch does.

The end-to-end delta is the tax the better branch mix has to repay. Run
``tools/bench.sh`` with and without ``--no-adaptive-havoc`` for the actual
edges-per-hour comparison; this script only measures the mutation loop.

Usage:
    PYTHONPATH=. python tools/bench_havoc_subop.py
"""

from __future__ import annotations

import bisect
import statistics
import sys
import time
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_new_operators import _make_minimal_fuzzer  # noqa: E402

from fuzzer_tool.services.operators import _HAVOC_N, OperatorEngine  # noqa: E402

REPEATS = 7
INV_SCALE = 1.0 / (1 << 30)


def _median(fn, repeats: int = REPEATS) -> float:
    return statistics.median([fn() for _ in range(repeats)])


# ── 1. isolated selection ────────────────────────────────────────────────


def _bench_isolated(n: int = 2_000_000) -> None:
    draws = [(i * 2654435761) % (1 << 30) for i in range(4096)]
    cdf = [(i + 1) / _HAVOC_N for i in range(_HAVOC_N)]
    table = [min(int(x * _HAVOC_N / 256), _HAVOC_N - 1) for x in range(256)]
    bits = [1 << i for i in range(_HAVOC_N)]

    def timed(body) -> float:
        t0 = time.perf_counter()
        body(n)
        return (time.perf_counter() - t0) / n * 1e9

    def uniform(count: int) -> None:
        mask = 0
        for i in range(count):
            mask |= draws[i & 4095] % _HAVOC_N

    def with_bisect(count: int) -> None:
        trials = [0] * _HAVOC_N
        mask = 0
        for i in range(count):
            op = bisect.bisect_right(cdf, draws[i & 4095] * INV_SCALE)
            if op >= _HAVOC_N:
                op = _HAVOC_N - 1
            trials[op] += 1
            mask |= bits[op]

    def with_table(counters) -> None:
        def body(count: int) -> None:
            trials = counters()
            mask = 0
            for i in range(count):
                op = table[draws[i & 4095] & 255]
                trials[op] += 1
                mask |= bits[op]

        return body

    print(f"isolated selection ({n:,} draws, median of {REPEATS})")
    for label, body in (
        ("uniform r[0] % 11", uniform),
        ("bisect + int list", with_bisect),
        ("table + int list", with_table(lambda: [0] * _HAVOC_N)),
        ("table + array('d')", with_table(lambda: array("d", [0.0] * _HAVOC_N))),
    ):
        print(f"  {label:20s} {_median(lambda b=body: timed(b)):7.1f} ns/draw")


# ── 2. end to end ────────────────────────────────────────────────────────


def _bench_end_to_end(n: int = 200_000) -> None:
    def run(adaptive: bool, buf_len: int) -> float:
        fuzzer = _make_minimal_fuzzer()
        fuzzer._adaptive_havoc = adaptive
        engine = OperatorEngine(fuzzer)
        base = bytes((i * 37) % 256 for i in range(buf_len))
        buf = bytearray(base)
        t0 = time.perf_counter()
        for _ in range(n):
            # Branches grow and shrink buf; reset so the size distribution
            # stays comparable across arms instead of drifting.
            if not (buf_len // 2 <= len(buf) <= buf_len * 2):
                buf = bytearray(base)
            engine._apply_single_mutation(buf)
        return (time.perf_counter() - t0) / n * 1e9

    # Arms are interleaved and reported as minimums, not medians. Both arms
    # run the same work modulo the selection block, so any excess over the
    # floor is scheduler noise -- and on a shared machine the median absorbs
    # that noise into the comparison and produced deltas that contradicted
    # the isolated measurement by an order of magnitude.
    print(f"\n_apply_single_mutation ({n:,} calls, min of {REPEATS} interleaved)")
    for buf_len in (64, 1024, 16384):
        off, on = [], []
        for _ in range(REPEATS):
            off.append(run(False, buf_len))
            on.append(run(True, buf_len))
        best_off, best_on = min(off), min(on)
        spread = (max(off) - best_off) / best_off * 100
        delta_pct = (best_on / best_off - 1) * 100
        print(
            f"  buf={buf_len:6d}B  uniform {best_off:8.1f} ns  adaptive {best_on:8.1f} ns  "
            f"delta {best_on - best_off:+7.1f} ns ({delta_pct:+.1f}%)"
            f"   [baseline noise {spread:.1f}%]"
        )
        if spread > abs(delta_pct):
            print(
                "      ^ noise exceeds the delta: this row measures nothing. "
                "Trust the isolated number above, or rerun on a quiet machine."
            )


if __name__ == "__main__":
    _bench_isolated()
    _bench_end_to_end()
