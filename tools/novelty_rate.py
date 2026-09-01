#!/usr/bin/env python3
"""Campaign-level novelty rate (docs/port-backlog.md §C2).

The metric: the fraction of generated inputs that exercised at least one
previously-unseen edge. The claim from the source this was merged from is
that it converges much faster than total edge count, so it is a shorter,
lower-variance cell for `tools/bench_paired.py` than raw coverage.

Computed entirely offline from an already-running campaign's coverage log
(``--coverage-log``, one row per stats tick: elapsed, exec_count,
cumulative_edges, corpus_size, crash_count, novel_input_count). Nothing in
the hot loop changes to produce this report -- `novel_input_count` is a
single counter incremented where the fuzzer already detects new edges
(`services/fuzzer.py`'s `if new:` branch in `fuzz_one`); this script only
reads the log.

Usage:
    tools/novelty_rate.py run1/cov.csv [run2/cov.csv ...]

With one file: prints the cumulative rate and a windowed time series between
consecutive stats ticks. With several: also reports the spread of the final
rate across runs, next to the spread of final cumulative-edge counts, which
is the comparison that supports (or refutes) the faster-convergence claim
for this specific campaign set.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fuzzer_tool.core.plotting import read_coverage_log  # noqa: E402


def cumulative_novelty_rate(rows: list[dict]) -> float | None:
    """Fraction of all executions in the log that found >=1 new edge.

    None when the log predates the novel_input_count column, so a caller
    can distinguish "rate is 0" from "not measured".
    """
    if not rows:
        return None
    last = rows[-1]
    if last.get("novel_input_count") is None:
        return None
    if last["exec_count"] <= 0:
        return None
    return last["novel_input_count"] / last["exec_count"]


def windowed_novelty_rate(rows: list[dict]) -> list[tuple[float, float]]:
    """(elapsed, rate) between each consecutive pair of stats ticks.

    Rate here is (novel_input_count delta) / (exec_count delta) for that
    window -- the same fraction, just localized to one interval instead of
    the whole campaign, so a caller can see whether novelty is still
    climbing or has flattened.
    """
    out = []
    prev = None
    for row in rows:
        if row.get("novel_input_count") is None:
            prev = None
            continue
        if prev is not None:
            d_execs = row["exec_count"] - prev["exec_count"]
            d_novel = row["novel_input_count"] - prev["novel_input_count"]
            if d_execs > 0:
                out.append((row["elapsed"], d_novel / d_execs))
        prev = row
    return out


def _report_one(path: Path) -> tuple[list[dict], float | None]:
    rows = read_coverage_log(path)
    if not rows:
        print(f"{path}: no rows (missing or empty log)")
        return rows, None
    rate = cumulative_novelty_rate(rows)
    if rate is None:
        print(f"{path}: {len(rows)} rows, but no novel_input_count column " "(log predates C2)")
        return rows, None
    final = rows[-1]
    print(
        f"{path}: {final['exec_count']} execs, "
        f"{final['cumulative_edges']} edges, "
        f"novelty_rate={rate:.4%} "
        f"({final['novel_input_count']} novel inputs)"
    )
    window = windowed_novelty_rate(rows)
    if window:
        recent = window[-min(5, len(window)) :]
        recent_avg = statistics.mean(r for _, r in recent)
        print(f"  windowed rate, last {len(recent)} ticks: avg={recent_avg:.4%}")
    return rows, rate


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    paths = [Path(a) for a in argv]
    rates: list[float] = []
    edges: list[int] = []
    for path in paths:
        rows, rate = _report_one(path)
        if rate is not None:
            rates.append(rate)
        if rows:
            edges.append(rows[-1]["cumulative_edges"])

    if len(paths) > 1 and len(rates) > 1:
        print()
        print(f"across {len(rates)} runs with a measured rate:")
        print(
            f"  novelty_rate  min={min(rates):.4%} max={max(rates):.4%} "
            f"stdev={statistics.stdev(rates):.4%}"
        )
        if len(edges) == len(rates):
            edge_mean = statistics.mean(edges)
            edge_stdev = statistics.stdev(edges) if len(edges) > 1 else 0.0
            rate_mean = statistics.mean(rates)
            rate_stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
            print(f"  cumulative_edges  min={min(edges)} max={max(edges)} stdev={edge_stdev:.1f}")
            if edge_mean and rate_mean:
                print(
                    f"  relative stdev  edges={edge_stdev / edge_mean:.4%}  "
                    f"novelty_rate={rate_stdev / rate_mean:.4%}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
