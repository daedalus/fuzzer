"""Explainability report generator for fuzzer runs.

Produces a structured text report covering coverage, mutation effectiveness,
seed contribution analysis, and crash triage. Can output to stdout or a file.
"""

import json
import os
from collections import Counter
from pathlib import Path


def generate_report(fuzzer, corpus_dir: str, crashes_dir: str) -> str:
    """Build a full explainability report from a Fuzzer instance after a run."""
    sections = []
    sections.append(_header(fuzzer))
    sections.append(_run_summary(fuzzer))
    sections.append(_coverage_analysis(fuzzer))
    sections.append(_mutation_effectiveness(fuzzer))
    sections.append(_seed_contribution(fuzzer))
    sections.append(_corpus_overview(fuzzer, corpus_dir))
    sections.append(_crash_analysis(fuzzer, crashes_dir))
    sections.append(_edge_map_analysis(fuzzer))
    return "\n".join(s for s in sections if s)


def _header(fuzzer) -> str:
    target = os.path.basename(fuzzer.target)
    line = "=" * 72
    return line + "\n  FUZZING REPORT: " + target + "\n" + line


def _run_summary(f) -> str:
    execs = f.exec_count
    crashes = f.crash_count
    corpus_size = len(f.corpus)
    timeouts = f.timeout_count
    lines = [
        "",
        "--- Run Summary ---",
        f"  Target:          {f.target}",
        f"  Executions:      {execs:,}",
        f"  Corpus size:     {corpus_size}",
        f"  Crashes:         {crashes}",
        f"  Timeouts:        {timeouts}",
        f"  Max input len:   {f.max_len}",
        f"  Timeout:         {f.timeout}s",
        f"  Coverage mode:   {'SHM bitmap' if f.shm_cov else 'ptrace' if f.ptrace_cov else 'none'}",
        f"  In-process:      {f._inprocess_runner is not None}",
    ]
    if execs > 0:
        lines.append(f"  Crash rate:      {crashes/execs*100:.4f}%")
        lines.append(f"  Timeout rate:    {timeouts/execs*100:.4f}%")
    return "\n".join(lines)


def _coverage_analysis(f) -> str:
    if not f.shm_cov:
        return ""
    cov = f.shm_cov
    seen = getattr(cov, "_seen", bytearray(cov.size))
    total_seen = sum(1 for b in seen if b)
    density = total_seen / cov.size * 100 if cov.size else 0

    # Cluster analysis: group edges into 256-byte buckets
    buckets = Counter()
    for i in range(cov.size):
        if seen[i]:
            buckets[i // 256] += 1

    lines = [
        "",
        "--- Coverage Analysis ---",
        f"  SHM map size:    {cov.size:,} bytes",
        f"  Unique edges:    {total_seen}",
        f"  Coverage density:{density:.4f}%",
    ]

    if buckets:
        lines.append(f"  Edge buckets:    {len(buckets)}")
        lines.append("  Top clusters (256-byte buckets):")
        for bucket, count in sorted(buckets.items(), key=lambda x: -x[1])[:10]:
            addr = bucket * 256
            lines.append(f"    0x{addr:04x}-0x{addr+255:04x}: {count:3d} edges")

    # Coverage growth timeline from edge tracker
    et_path = Path(f.corpus_dir) / "edge_tracker.json"
    if et_path.exists():
        with open(et_path) as fobj:
            et = json.load(fobj)
        cum = et.get("cumulative_edges", [])
        if cum:
            # Show coverage at milestones
            total = len(cum)
            milestones = [100, 200, 500, 1000, 2000, 5000, 10000]
            lines.append("  Coverage growth:")
            shown = set()
            for m in milestones:
                if m <= total:
                    lines.append(f"    iter {m:>5d}: {m} edges")
                    shown.add(m)
            if total not in shown:
                lines.append(f"    iter {total:>5d}: {total} edges (final)")

    return "\n".join(lines)


def _mutation_effectiveness(f) -> str:
    counts = f.op_counts
    successes = f.op_success
    if not counts:
        return ""

    total = sum(counts.values())
    total_success = sum(successes.values())

    lines = [
        "",
        "--- Mutation Effectiveness ---",
        f"  {'Operation':<22s} {'Count':>7s} {'Success':>8s} {'Rate':>7s}",
        f"  {'-'*22} {'-'*7} {'-'*8} {'-'*7}",
    ]

    for op, count in sorted(counts.items(), key=lambda x: -x[1]):
        succ = successes.get(op, 0)
        rate = succ / count * 100 if count else 0
        lines.append(f"  {op:<22s} {count:>7d} {succ:>8d} {rate:>6.1f}%")

    lines.append(f"  {'TOTAL':<22s} {total:>7d} {total_success:>8d} {total_success/total*100:>6.1f}%" if total else "")
    return "\n".join(lines)


def _seed_contribution(f) -> str:
    if not f.seed_meta:
        return ""

    # Seeds ranked by coverage contribution
    ranked = []
    for seed, meta in f.seed_meta.items():
        ce = meta.get("coverage_edges", 0)
        fc = meta.get("fuzz_count", 0)
        if ce > 0:
            name = seed.decode(errors="replace") if isinstance(seed, bytes) else str(seed)
            name = name[:40] + ("..." if len(name) > 40 else "")
            ranked.append((name, ce, fc))

    if not ranked:
        return ""

    ranked.sort(key=lambda x: -x[1])
    total_edges = f.shm_cov.cumulative_edges if f.shm_cov else 0

    lines = [
        "",
        "--- Seed Contribution (coverage) ---",
    ]

    top_n = min(15, len(ranked))
    lines.append(f"  Top {top_n} seeds by unique edges discovered:")
    for i, (name, ce, fc) in enumerate(ranked[:top_n], 1):
        pct = ce / total_edges * 100 if total_edges else 0
        lines.append(f"    {i:>2d}. [{ce:>3d} edges, {pct:>5.1f}%] fuzzed {fc:>3d}x  {name}")

    total_cov_seeds = len(ranked)
    lines.append(f"\n  {total_cov_seeds} of {len(f.corpus)} seeds contributed new coverage")
    return "\n".join(lines)


def _corpus_overview(f, corpus_dir) -> str:
    p = Path(corpus_dir)
    if not p.exists():
        return ""

    pngs = [f for f in p.iterdir() if f.is_file() and not f.name.endswith((".json",))]

    if not pngs:
        return ""

    sizes = sorted([f.stat().st_size for f in pngs])
    total_size = sum(sizes)

    lines = [
        "",
        "--- Corpus Overview ---",
        f"  Files:           {len(pngs)}",
        f"  Total size:      {_human_size(total_size)}",
        f"  Smallest:        {_human_size(sizes[0])}",
        f"  Median:          {_human_size(sizes[len(sizes)//2])}",
        f"  Largest:         {_human_size(sizes[-1])}",
    ]

    # Size distribution
    buckets = {"<100B": 0, "100B-1KB": 0, "1KB-10KB": 0, "10KB-100KB": 0, ">100KB": 0}
    for s in sizes:
        if s < 100:
            buckets["<100B"] += 1
        elif s < 1024:
            buckets["100B-1KB"] += 1
        elif s < 10240:
            buckets["1KB-10KB"] += 1
        elif s < 102400:
            buckets["10KB-100KB"] += 1
        else:
            buckets[">100KB"] += 1

    lines.append("  Size distribution:")
    for bucket, count in buckets.items():
        bar = "#" * min(count, 40)
        lines.append(f"    {bucket:<12s} {count:>4d} {bar}")

    return "\n".join(lines)


def _crash_analysis(f, crashes_dir) -> str:
    p = Path(crashes_dir)
    if not p.exists():
        return ""

    crashes = [f for f in p.iterdir() if f.is_file()]
    if not crashes:
        return ""

    lines = [
        "",
        "--- Crash Analysis ---",
        f"  Total crashes:   {len(crashes)}",
    ]

    # Group by size
    size_groups = Counter()
    for c in crashes:
        size = c.stat().st_size
        size_groups[size] += 1

    lines.append("  Unique crash sizes:")
    for size, count in sorted(size_groups.items())[:10]:
        lines.append(f"    {_human_size(size):>8s} x {count}")

    # Show top 5 crashes by filename
    lines.append("  Sample crashes:")
    for c in sorted(crashes, key=lambda x: x.name)[:5]:
        lines.append(f"    {c.name} ({_human_size(c.stat().st_size)})")

    return "\n".join(lines)


def _edge_map_analysis(f) -> str:
    if not f.shm_cov:
        return ""
    cov = f.shm_cov
    seen = getattr(cov, "_seen", bytearray(cov.size))
    if not any(seen):
        return ""

    # Find contiguous regions
    regions = []
    start = None
    for i in range(cov.size):
        if seen[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                regions.append((start, i - 1))
                start = None
    if start is not None:
        regions.append((start, cov.size - 1))

    if not regions:
        return ""

    lines = [
        "",
        "--- Edge Map Regions ---",
        f"  Contiguous regions: {len(regions)}",
    ]
    for s, e in regions[:10]:
        span = e - s + 1
        filled = sum(1 for i in range(s, e + 1) if seen[i])
        pct = filled / span * 100
        lines.append(f"    0x{s:04x}-0x{e:04x}: {filled}/{span} bytes ({pct:.1f}% filled)")

    return "\n".join(lines)


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    elif n < 1024 * 1024:
        return f"{n/1024:.1f}KB"
    else:
        return f"{n/1024/1024:.1f}MB"
