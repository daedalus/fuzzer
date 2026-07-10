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
    sections.append(_runtime_performance(fuzzer))
    sections.append(_good_turing(fuzzer))
    sections.append(_coverage_analysis(fuzzer))
    sections.append(_mutation_effectiveness(fuzzer))
    sections.append(_operator_diversity(fuzzer))
    sections.append(_elo_ratings(fuzzer))
    sections.append(_bandit_calibration(fuzzer))
    sections.append(_fuzzing_strategy(fuzzer))
    sections.append(_execution_time_analysis(fuzzer))
    sections.append(_mdl_codelength(fuzzer))
    sections.append(_seed_contribution(fuzzer))
    sections.append(_edge_rarity(fuzzer))
    sections.append(_corpus_health(fuzzer))
    sections.append(_corpus_overview(fuzzer, corpus_dir))
    sections.append(_crash_analysis(fuzzer, crashes_dir))
    sections.append(_crash_exploitability(fuzzer, crashes_dir))
    sections.append(_crash_reproducibility(fuzzer))
    sections.append(_crash_rate_trend(fuzzer))
    sections.append(_disk_footprint(corpus_dir))
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
        lines.append(f"  Crash rate:      {crashes / execs * 100:.4f}%")
        lines.append(f"  Timeout rate:    {timeouts / execs * 100:.4f}%")
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
            lines.append(f"    0x{addr:04x}-0x{addr + 255:04x}: {count:3d} edges")

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
        f"  {'-' * 22} {'-' * 7} {'-' * 8} {'-' * 7}",
    ]

    for op, count in sorted(counts.items(), key=lambda x: -x[1]):
        succ = successes.get(op, 0)
        rate = succ / count * 100 if count else 0
        lines.append(f"  {op:<22s} {count:>7d} {succ:>8d} {rate:>6.1f}%")

    lines.append(
        f"  {'TOTAL':<22s} {total:>7d} {total_success:>8d} {total_success / total * 100:>6.1f}%"
        if total
        else ""
    )
    return "\n".join(lines)


def _mdl_codelength(f) -> str:
    """MDL codelength + perplexity analysis: how surprising is the corpus to the Markov model."""
    if not hasattr(f, "markov") or not f.markov.is_trained():
        return ""
    if not f.corpus:
        return ""

    pp_stats = f.markov.corpus_perplexity(f.corpus)
    if pp_stats["mean"] == 0:
        return ""

    ratios = []
    for seed in f.corpus[:200]:
        ratios.append(f.markov.codelength_ratio(seed))

    if not ratios:
        return ""

    avg_cl = sum(ratios) / len(ratios)
    s = sorted(ratios)

    lines = [
        "",
        "--- Markov Model Quality ---",
        f"  Perplexity:        mean={pp_stats['mean']:.1f}  "
        f"p10={pp_stats['p10']:.1f}  p90={pp_stats['p90']:.1f}",
        f"  Well-predicted:    {pp_stats['low_surprise_count']} seeds (PP < 10)",
        f"  Model lost:        {pp_stats['high_surprise_count']} seeds (PP > 200)",
        f"  Avg codelength:    {avg_cl:.2f} bits/byte  range=[{s[0]:.2f}, {s[-1]:.2f}]",
    ]

    # NCD between most surprising seeds
    if len(ratios) >= 2:
        indexed = list(enumerate(ratios))
        indexed.sort(key=lambda x: -x[1])
        top_i = indexed[0][0]
        second_i = indexed[1][0]
        if top_i < len(f.corpus) and second_i < len(f.corpus):
            from fuzzer_tool.core.edge_tracker import normalized_compression_distance

            ncd = normalized_compression_distance(f.corpus[top_i], f.corpus[second_i])
            lines.append(f"  NCD (top 2):       {ncd:.4f} (0=same structure, 1=unrelated)")

    return "\n".join(lines)

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

    entries = [f for f in p.iterdir() if f.is_file() and not f.name.endswith((".json",))]

    if not entries:
        return ""

    sizes = sorted([f.stat().st_size for f in entries])
    total_size = sum(sizes)

    lines = [
        "",
        "--- Corpus Overview ---",
        f"  Files:           {len(entries)}",
        f"  Total size:      {_human_size(total_size)}",
        f"  Smallest:        {_human_size(sizes[0])}",
        f"  Median:          {_human_size(sizes[len(sizes) // 2])}",
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


def _good_turing(f) -> str:
    if not hasattr(f, "_edge_tracker"):
        return ""
    gt = f._edge_tracker.good_turing_estimate()
    if gt["n"] == 0:
        return ""
    lines = [
        "",
        "--- Good-Turing Coverage Estimation ---",
        f"  Edges observed:       {gt['n']}",
        f"  Singletons (1x):     {gt['n1']}",
        f"  Doubletons (2x):     {gt['n2']}",
        f"  Est. undiscovered:   {gt['estimated_undiscovered']}",
        f"  Saturation:          {gt['saturation']:.1%}",
        f"  Confidence:          {gt['confidence']}",
    ]
    if f.discovery_rate() > 0:
        lines.append(f"  Discovery rate:       {f.discovery_rate():.1f} edges/1k execs")
    return "\n".join(lines)


def _crash_reproducibility(f) -> str:
    if not f._crash_replays:
        return ""
    lines = ["", "--- Crash Reproducibility ---"]
    total = 0
    reproducible = 0
    for sig, replays in f._crash_replays.items():
        if len(replays) >= f.replay_n:
            total += 1
            rate = sum(1 for r in replays if r >= 0) / len(replays)
            reproducible += rate
            lines.append(f"  {sig[:40]}: {rate:.0%} ({len(replays)} replays)")
    if total > 0:
        avg = reproducible / total
        lines.insert(2, f"  Overall repro rate:   {avg:.0%} ({total} crashes replayed)")
    return "\n".join(lines)


def _disk_footprint(corpus_dir: str) -> str:
    p = Path(corpus_dir)
    if not p.exists():
        return ""
    entries = [f for f in p.iterdir() if f.is_file() and not f.name.endswith((".json",))]
    if not entries:
        return ""
    total_size = sum(f.stat().st_size for f in entries)
    lines = [
        "",
        "--- Disk Footprint ---",
        f"  Corpus files:    {len(entries)}",
        f"  Total size:      {_human_size(total_size)}",
    ]
    # Delta vs full analysis: check if any files are very small (< 100 bytes) vs large
    small = [f for f in entries if f.stat().st_size < 100]
    large = [f for f in entries if f.stat().st_size >= 100]
    if small:
        lines.append(f"  Small (<100B):   {len(small)} files (potential deltas)")
        lines.append(f"  Large (>=100B):  {len(large)} files")
    return "\n".join(lines)


def _bandit_calibration(f) -> str:
    if not f.mc or not f.mc_bandit:
        return ""
    brier = f.mc.brier_score()
    if brier == 0:
        return ""
    lines = [
        "",
        "--- Bandit Calibration (Brier Score) ---",
        f"  Brier score:       {brier:.4f} (0=perfect, 0.25=random, 0.5=worst)",
    ]
    cal = f.mc.calibration_report()
    if cal:
        lines.append("  Calibration by predicted probability bin:")
        lines.append(f"    {'Bin':<12s} {'Predicted':>10s} {'Actual':>10s} {'Samples':>8s}")
        for bin_label, (pred, actual) in cal.items():
            lines.append(f"    {bin_label:<12s} {pred:>10.3f} {actual:>10.3f}")
    return "\n".join(lines)


def _execution_time_analysis(f) -> str:
    tracker = f._exec_time_tracker
    if tracker.count < 10:
        return ""
    lines = [
        "",
        "--- Execution Time Analysis ---",
        f"  Observations:   {tracker.count}",
        f"  p50:            {tracker.p50 * 1000:.1f}ms",
        f"  p99:            {tracker.p99 * 1000:.1f}ms",
        f"  Suggested timeout: {tracker.suggested_timeout():.2f}s",
        f"  CRPS (mean):    {tracker.mean_crps():.6f}",
        f"  CRPS trend:     {tracker.crps_trend():.6f} (+ = degrading)",
    ]
    if tracker.crps_trend() > 0.001:
        lines.append("  WARNING: CRPS rising — target runtime behavior is drifting")
    return "\n".join(lines)


def _corpus_health(f) -> str:
    """Corpus health: entropy, lineage depth, duplicate rate."""
    if not f.seed_meta:
        return ""
    lines = ["", "--- Corpus Health ---"]

    # Lineage depth distribution
    depths = [m.get("lineage_depth", 0) for m in f.seed_meta.values()]
    if depths:
        avg_d = sum(depths) / len(depths)
        lines.append(f"  Lineage depth:     min={min(depths)} avg={avg_d:.1f} max={max(depths)}")

    # Input size distribution
    if f._corpus_size_history:
        s = sorted(f._corpus_size_history)
        lines.append(
            f"  Input sizes:       min={s[0]} p50={s[len(s) // 2]} p90={s[-len(s) // 10]} max={s[-1]}"
        )

    # Duplicate rejection rate
    if f._total_corpus_attempts > 0:
        dup_rate = f._duplicate_reject_count / f._total_corpus_attempts * 100
        lines.append(
            f"  Dup rejection:     {dup_rate:.1f}% ({f._duplicate_reject_count}/{f._total_corpus_attempts})"
        )

    # Shannon entropy of corpus byte distribution
    byte_freq = [0] * 256
    total_bytes = 0
    for seed in f.corpus:
        for b in seed[:4096]:  # cap per-seed to avoid huge corpus bias
            byte_freq[b] += 1
            total_bytes += 1
    if total_bytes > 0:
        entropy = 0.0
        for count in byte_freq:
            if count > 0:
                p = count / total_bytes
                entropy -= p * __import__("math").log2(p)
        lines.append(f"  Byte entropy:      {entropy:.2f} bits (max=8.0)")
    return "\n".join(lines)


def _crash_exploitability(f, crashes_dir: str) -> str:
    """Exploitability tier distribution from crash metadata."""
    p = Path(crashes_dir)
    if not p.exists():
        return ""
    # Scan .json metadata files for exploitability
    tiers: dict[str, int] = {}
    for meta_file in p.glob("*.json"):
        try:
            data = json.loads(meta_file.read_text())
            tier = data.get("exploitability", "UNKNOWN")
            tiers[tier] = tiers.get(tier, 0) + 1
        except Exception:
            continue
    if not tiers:
        return ""
    lines = [
        "",
        "--- Crash Exploitability ---",
    ]
    for tier, count in sorted(tiers.items(), key=lambda x: -x[1]):
        lines.append(f"  {tier:<12s}: {count}")
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


def _runtime_performance(f) -> str:
    """Wall-clock time, memory, throughput, and corpus growth."""
    import resource as _resource
    import time

    elapsed = time.time() - f.start_time
    eps = f.exec_count / elapsed if elapsed > 0 else 0
    rss_kb = f._peak_rss
    rss_str = f"{rss_kb // 1024}MB" if rss_kb >= 1024 else f"{rss_kb}KB"

    lines = [
        "",
        "--- Runtime Performance ---",
        f"  Duration:         {_format_duration(elapsed)}",
        f"  Executions:       {f.exec_count:,}",
        f"  Avg throughput:   {eps:.1f} execs/sec",
        f"  Peak throughput:  {f._peak_eps:.1f} execs/sec",
        f"  Peak RSS:         {rss_str}",
        f"  Map size:         {f.map_size:,} bytes",
    ]

    # Corpus growth
    added = f._total_corpus_attempts
    rejected = f._duplicate_reject_count
    pruned = f._pruned_count
    lines.append(f"  Seeds added:      {added}")
    lines.append(f"  Duplicates:       {rejected} rejected")
    if pruned > 0:
        lines.append(f"  Seeds pruned:     {pruned}")

    # Dup rejection rate
    if f._total_corpus_attempts > 0:
        dup_rate = rejected / f._total_corpus_attempts * 100
        lines.append(f"  Dup rejection:    {dup_rate:.1f}%")

    # Input size distribution
    if f._corpus_size_history:
        s = sorted(f._corpus_size_history)
        lines.append(
            f"  Input sizes:      min={s[0]} p50={s[len(s) // 2]} "
            f"p90={s[-len(s) // 10]} max={s[-1]}"
        )

    return "\n".join(lines)


def _operator_diversity(f) -> str:
    """Operator usage diversity — entropy of the operator distribution."""
    if not f.op_counts:
        return ""

    total = sum(f.op_counts.values())
    if total == 0:
        return ""

    # Shannon entropy of operator distribution
    import math

    entropy = 0.0
    for count in f.op_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    max_entropy = math.log2(len(f.op_counts)) if f.op_counts else 0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    lines = [
        "",
        "--- Operator Diversity ---",
        f"  Operators used:   {len(f.op_counts)}",
        f"  Shannon entropy:  {entropy:.2f} bits (max={max_entropy:.2f})",
        f"  Normalized:       {norm_entropy:.2%} (1.0=uniform, 0.0=single op)",
    ]

    # Most/least used
    sorted_ops = sorted(f.op_counts.items(), key=lambda x: -x[1])
    if sorted_ops:
        lines.append(f"  Most used:        {sorted_ops[0][0]} ({sorted_ops[0][1]}x)")
        lines.append(f"  Least used:       {sorted_ops[-1][0]} ({sorted_ops[-1][1]}x)")

    # Effective operators (those that found new coverage or crashes)
    effective = [op for op, c in f.op_success.items() if c > 0]
    lines.append(f"  Effective ops:    {len(effective)}/{len(f.op_counts)} produced results")

    return "\n".join(lines)


def _fuzzing_strategy(f) -> str:
    """Active scheduling strategies and their states."""
    lines = ["", "--- Fuzzing Strategy ---"]

    strategies = []

    # MC bandit
    if f.mc and f.mc_bandit:
        strategies.append(f"  MC Bandit:        Thompson sampling, {len(f.mc.arm_alpha)} arms")
        if f.mc.brier_score() > 0:
            strategies.append(f"    Brier score:    {f.mc.brier_score():.4f}")

    # MC CEM
    if f.mc and f.mc_cem:
        strategies.append(
            f"  MC CEM:           elite_frac={f.mc.elite_frac}, elite_set={len(f.mc.elite_set)}"
        )

    # MOpt
    if f._mopt:
        strategies.append(
            f"  MOpt PSO:         {f._mopt.n_particles} particles, window={f._mopt.window_size}"
        )

    # Replicator
    if f._replicator:
        strategies.append(
            f"  Replicator:       window={f._replicator.window_size}, eta={f._replicator.eta}"
        )

    # Markov
    if f.markov_trained:
        if hasattr(f.markov, "chains"):
            orders = ",".join(str(o) for o in f.markov.orders)
            strategies.append(f"  Markov ensemble:  orders=[{orders}]")
        else:
            strategies.append(f"  Markov chain:     order={f.markov.order}")
        strategies.append(f"    Generation:     {'enabled' if f.markov_generate else 'disabled'}")

    # MI guided
    if f._use_mi and f._mi:
        strategies.append(f"  MI-guided:        max_positions={f._mi.max_positions}")

    # Transfer entropy
    if f._use_transfer_entropy and f._te:
        strategies.append(f"  Transfer entropy: history={f._te.k}")

    # Secretary
    if f._secretary:
        strategies.append(
            f"  Secretary:        window={f._secretary_window}, "
            f"exploration={f._secretary_exploration:.0%}"
        )
        if f._corpus_secretary:
            stop, reason = f._corpus_secretary.should_stop()
            status = f"STOP ({reason})" if stop else "active"
            strategies.append(f"    Corpus status:  {status}")

    # Annealing
    if f._anneal_budget > 0:
        strategies.append(
            f"  Annealing:        budget={f._anneal_budget}, progress={f._anneal_progress:.1%}"
        )

    # Grammar
    if f.grammar:
        strategies.append(f"  Grammar:          {len(f.grammar.rules)} rules")

    # Dictionary
    if f.dictionary:
        strategies.append(f"  Dictionary:       {len(f.dictionary)} tokens")

    if not strategies:
        strategies.append("  Mode:             random mutation (no scheduling)")

    lines.extend(strategies)
    return "\n".join(lines)


def _edge_rarity(f) -> str:
    """Edge rarity distribution and seed irreplaceability."""
    if not hasattr(f, "_edge_tracker"):
        return ""
    rarity = f._edge_tracker.edge_rarity_stats()
    if rarity["total"] == 0:
        return ""

    lines = [
        "",
        "--- Edge Rarity ---",
        f"  Total edges:      {rarity['total']}",
        f"  Singleton (1x):   {rarity['singleton']}",
        f"  Cold (2-5x):      {rarity['cold']}",
        f"  Warm (6-20x):     {rarity['warm']}",
        f"  Hot (>20x):       {rarity['hot']}",
        f"  Avg seeds/edge:   {rarity['avg_seeds_per_edge']:.1f}",
    ]

    # Seed irreplaceability
    uniqueness = f._edge_tracker.seed_uniqueness()
    if uniqueness:
        irreplaceable = sum(1 for v in uniqueness.values() if v > 0)
        lines.append(f"  Irreplaceable:    {irreplaceable} seeds cover singleton edges")

    # Top co-occurring edges
    cooccur = f._edge_tracker.edge_cooccurrence(top_k=3)
    if cooccur:
        pairs_str = ", ".join(f"e{a}<->e{b}({j:.0%})" for a, b, j in cooccur)
        lines.append(f"  Co-occurrence:    {pairs_str}")

    return "\n".join(lines)


def _crash_rate_trend(f) -> str:
    """Crash rate over time."""
    if not f._crash_rate_history or len(f._crash_rate_history) < 2:
        return ""

    lines = ["", "--- Crash Rate Trend ---"]

    # Sample at milestones
    history = f._crash_rate_history
    milestones = [100, 500, 1000, 5000, 10000]
    shown = set()
    for m in milestones:
        for exec_c, crash_c in history:
            if exec_c >= m and m not in shown:
                rate = crash_c / exec_c * 100 if exec_c > 0 else 0
                lines.append(f"  iter {m:>5d}: {crash_c:>5d} crashes ({rate:.1f}%)")
                shown.add(m)
                break

    # Final
    if history:
        last_exec, last_crash = history[-1]
        rate = last_crash / last_exec * 100 if last_exec > 0 else 0
        if last_exec not in shown:
            lines.append(f"  iter {last_exec:>5d}: {last_crash:>5d} crashes ({rate:.1f}%)")

    return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def _elo_ratings(f) -> str:
    """Elo operator rankings and comparison with bandit rankings."""
    if not f._use_elo or not f._elo or not f._elo.ratings:
        return ""

    ranking = f._elo.get_ranking()
    unrated = f._elo.get_unrated()
    lines = [
        "",
        "--- Elo Operator Ratings ---",
        f"  K-factor:        {f._elo.k_factor}",
        f"  Decay:           {f._elo.decay}",
        f"  Min matches:     {f._elo.min_matches}",
        f"  Total matches:   {sum(f._elo._match_count.values()) // 2}",
        f"  Rated:           {len(ranking)} operators",
        f"  Unrated:         {len(unrated)} operators (< {f._elo.min_matches} matches)",
    ]

    # Top 10 and bottom 5 of rated operators
    if ranking:
        lines.append(f"  {'Rank':<6s} {'Operator':<22s} {'Rating':>8s} {'Matches':>8s}")
        lines.append(f"  {'-' * 6} {'-' * 22} {'-' * 8} {'-' * 8}")
        for i, (op, rating) in enumerate(ranking[:10], 1):
            matches = f._elo._match_count.get(op, 0)
            lines.append(f"  {i:<6d} {op:<22s} {rating:>8.0f} {matches:>8d}")
        if len(ranking) > 10:
            lines.append(f"  {'...':<6s}")
            for i, (op, rating) in enumerate(ranking[-5:], len(ranking) - 4):
                matches = f._elo._match_count.get(op, 0)
                lines.append(f"  {i:<6d} {op:<22s} {rating:>8.0f} {matches:>8d}")

    # Unrated operators
    if unrated:
        lines.append("")
        lines.append(f"  Not yet rated ({len(unrated)} operators):")
        unrated_sample = unrated[:8]
        lines.append(f"    {', '.join(unrated_sample)}")
        if len(unrated) > 8:
            lines.append(f"    ... and {len(unrated) - 8} more")

    # Crash-specific Elo if available
    if f._elo.crash_track and f._elo.crash_ratings:
        crash_ranking = f._elo.get_ranking(crash=True)
        if crash_ranking and crash_ranking[0][1] != f._elo.default_rating:
            lines.append("")
            lines.append("  Crash-specific Elo:")
            for i, (op, rating) in enumerate(crash_ranking[:5], 1):
                delta = rating - f._elo.default_rating
                sign = "+" if delta >= 0 else ""
                lines.append(f"    {i}. {op:<20s} {rating:>7.0f} ({sign}{delta:.0f})")

    # Meta-scheduler strategy ranking (bandit vs MOpt)
    if f._use_meta_elo and f._elo:
        strategy_ranking = f._elo.get_strategy_ranking()
        if strategy_ranking:
            lines.append("")
            lines.append("  Meta-scheduler (bandit vs MOpt):")
            for s, rating in strategy_ranking:
                delta = rating - f._elo.default_rating
                sign = "+" if delta >= 0 else ""
                matches = f._elo._strategy_match_count.get(s, 0)
                lines.append(f"    {s:<12s} {rating:>7.0f} ({sign}{delta:.0f}, {matches} matches)")

    # Compare with bandit if available
    if f.mc and f.mc_bandit and f.mc.arm_alpha:
        bandit_ranking = sorted(
            f.mc.arm_alpha.items(),
            key=lambda x: -x[1] / (x[1] + f.mc.arm_beta.get(x[0], 1)),
        )
        elo_rank = {op: i for i, (op, _) in enumerate(ranking)}
        bandit_rank = {op: i for i, (op, _) in enumerate(bandit_ranking)}
        if elo_rank and bandit_rank:
            common = set(elo_rank) & set(bandit_rank)
            if common:
                rank_diffs = [abs(elo_rank[op] - bandit_rank[op]) for op in common]
                avg_diff = sum(rank_diffs) / len(rank_diffs)
                max_diff = max(rank_diffs)
                lines.append("")
                lines.append(
                    f"  Elo vs Bandit:    avg rank diff={avg_diff:.1f}, "
                    f"max={max_diff} ({len(common)} rated operators)"
                )

    return "\n".join(lines)


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    else:
        return f"{n / 1024 / 1024:.1f}MB"
