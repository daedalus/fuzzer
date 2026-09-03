"""Explainability report generator for fuzzer runs.

Produces a structured text report covering coverage, mutation effectiveness,
seed contribution analysis, and crash triage. Can output to stdout or a file.
"""

import contextlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from fuzzer_tool.core.temporal_join import join_streams

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# Shared milestone ladder for every "over time" table in the report. Coverage
# growth and the crash-rate trend used different ladders (100/200/500/1000/2000
# vs 100/500/1000/5000), so the two sections could not be read against each
# other.
MILESTONES = (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)

# A binary Brier score is bounded by 1.0 (confident and always wrong).
# 0.25 is the uninformative constant-0.5 predictor, not "random" in the
# sense of the worst case.
_BRIER_LEGEND = "(0=perfect, 0.25=uninformative, 1.0=worst)"

# Extension save_crash() gives the crash INPUT; `.txt`/`.sh`/`.hex` alongside it
# are human-readable sidecars, not crashes. Keep in sync with
# adapters/filesystem.py:save_crash.
CRASH_INPUT_SUFFIX = ".bin"


def _confidence_interval(n, success_count=None):
    """Compute ±1σ, ±2σ, ±3σ confidence intervals.

    For binomial proportions (success_count is not None):
        se = sqrt(p * (1-p) / n)
        ci_k = p ± k * se

    For continuous values (success_count is None):
        Returns (0, 0, 0, 0) — caller must provide std.

    Returns (mean, se, ci_1, ci_2, ci_3) where ci_k is the half-width.
    """
    if n <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if success_count is not None:
        p = success_count / n
        se = math.sqrt(p * (1 - p) / n) if n > 1 else 0.0
        return (p, se, se, se * 2, se * 3)
    return (0.0, 0.0, 0.0, 0.0, 0.0)


def _format_ci_inline(mean, ci_1, ci_2, ci_3, fmt=".1f", pct=False, lo_clamp=None):
    """Format as: mean ±1σ: lo-hi  ±2σ: lo-hi  ±3σ: lo-hi

    lo_clamp: floor applied to every lower bound after scaling. Throughput and
    counts cannot go negative, but a symmetric Wald band around a small mean
    happily prints one (the ffmpeg_read_nosan run reported a ±3σ throughput
    lower bound of -5.5 execs/sec).
    """
    # Convert to float to handle MagicMock objects in tests
    m = float(mean)
    s1, s2, s3 = float(ci_1), float(ci_2), float(ci_3)
    if pct:
        m, s1, s2, s3 = m * 100, s1 * 100, s2 * 100, s3 * 100

    def lo(half):
        v = m - half
        return max(v, lo_clamp) if lo_clamp is not None else v

    return (
        f"{m:{fmt}}  "
        f"±1σ: {lo(s1):{fmt}}-{m + s1:{fmt}}  "
        f"±2σ: {lo(s2):{fmt}}-{m + s2:{fmt}}  "
        f"±3σ: {lo(s3):{fmt}}-{m + s3:{fmt}}"
    )


def _wilson_interval(n: int, k: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, clamped to [0, 1].

    The Wald interval that _confidence_interval computes degenerates at the
    boundaries: at k=0 it returns a zero-width band (the ffmpeg_read_nosan run
    reported a crash rate of "0.0000-0.0000" at ±3σ from 0 crashes in 1,305
    executions, claiming certainty it did not have), and near p=0 it produces
    negative lower bounds. Wilson stays inside [0, 1] and keeps non-zero width
    at k=0, where its upper bound approximates the rule of three (~3/n).
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _format_proportion(n: int, k: int, label_pct: bool = True) -> str:
    """Format a rate with Wilson ±1σ/±2σ/±3σ bands.

    Emitted as a percentage with an explicit '%' so the unit is unambiguous.
    The previous formatting passed pct=True (scale by 100) together with a
    '.4f' format and no '%' suffix, so 4 timeouts in 1,305 executions printed
    as "0.3065" — a value that reads as a fraction but is really a percent,
    off by exactly 100x from the 0.0031 a reader would compute.
    """
    p = (k / n) if n > 0 else 0.0
    bands = []
    for z in (1.0, 2.0, 3.0):
        lo, hi = _wilson_interval(n, k, z)
        bands.append(f"±{int(z)}σ: {lo * 100:.4f}%-{hi * 100:.4f}%")
    scale = 100 if label_pct else 1
    suffix = "%" if label_pct else ""
    return f"{p * scale:.4f}{suffix}  " + "  ".join(bands)


def generate_report(fuzzer, corpus_dir: str, crashes_dir: str) -> str:
    """Build a full explainability report from a Fuzzer instance after a run."""
    sections = []
    sections.append(_header(fuzzer))
    sections.append(_run_summary(fuzzer))
    sections.append(_configuration(fuzzer))
    sections.append(_runtime_performance(fuzzer))
    sections.append(_good_turing(fuzzer))
    sections.append(_coverage_analysis(fuzzer))
    sections.append(_mutation_effectiveness(fuzzer))
    sections.append(_mutation_edge_attribution(fuzzer))
    sections.append(_operator_diversity(fuzzer))
    sections.append(_entropy_metrics(fuzzer))
    sections.append(_format_learning(fuzzer))
    sections.append(_elo_ratings(fuzzer))
    sections.append(_bandit_calibration(fuzzer))
    sections.append(_fuzzing_strategy(fuzzer))
    sections.append(_execution_time_analysis(fuzzer))
    sections.append(_distribution_diagnostics(fuzzer))
    sections.append(_spectral_diagnostics(fuzzer))
    sections.append(_temporal_correlation(fuzzer))
    sections.append(_mdl_codelength(fuzzer))
    sections.append(_comparison_profile(fuzzer))
    sections.append(_smt_solver_activity(fuzzer))
    sections.append(_seed_contribution(fuzzer))
    sections.append(_edge_rarity(fuzzer))
    sections.append(_corpus_health(fuzzer))
    sections.append(_corpus_overview(fuzzer, corpus_dir))
    sections.append(_crash_analysis(fuzzer, crashes_dir))
    sections.append(_crash_signatures(fuzzer))
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


def _target_exec_line(f) -> str:
    """Reconstruct the target invocation as the runner builds it.

    File mode: target + args with {file} -> @@ (or a bare @@ appended when
    target_args is empty, mirroring run_target_file). Stdin mode: target only.
    """
    args = list(getattr(f, "target_args", None) or [])
    if getattr(f, "file_mode", False):
        parts = [f.target] + [a.replace("{file}", "@@") for a in args] if args else [f.target, "@@"]
    else:
        parts = [f.target] + args
    return " ".join(parts)


def _run_summary(f) -> str:
    execs = f.exec_count
    crashes = f.crash_count
    corpus_size = len(f.corpus)
    timeouts = f.timeout_count
    lines = [
        "",
        "--- Run Summary ---",
        f"  Target:          {f.target}",
        f"  Exec line:       {_target_exec_line(f)}",
    ]
    inv = getattr(f, "invocation", "")
    if inv:
        lines.append(f"  Invocation:      {inv}")
    # Only interesting when it differs: on a fresh run the two are equal, and
    # printing the same string twice reads as a bug in the report.
    orig = getattr(f, "original_invocation", "")
    if orig and orig != inv:
        lines.append(f"  Started as:      {orig}")
    # In-process runners hand the buffer to the target function directly (via
    # shared memory for --inprocess-direct); neither stdin nor an input file is
    # involved, so reporting "stdin" there described a path the run never took.
    if getattr(f, "_inprocess_runner", None) is not None:
        direct = bool(getattr(f._inprocess_runner, "direct", False))
        input_mode = "in-process (direct/SHM)" if direct else "in-process"
    elif getattr(f, "file_mode", False):
        input_mode = "file"
    else:
        input_mode = "stdin"
    lines.append(f"  Input mode:      {input_mode}")
    if getattr(f, "target_args", None):
        lines.append(f"  Target args:     {' '.join(f.target_args)}")
    lines.extend(
        [
            f"  Executions:      {execs:,}",
            f"  Corpus size:     {corpus_size}",
            f"  Crashes:         {crashes}",
            f"  Timeouts:        {timeouts}",
            f"  Max input len:   {f.max_len}",
            f"  Timeout:         {f.timeout}s",
            f"  Coverage mode:   {'SHM bitmap' if f.shm_cov else 'ptrace' if f.ptrace_cov else 'none'}",
            f"  In-process:      {f._inprocess_runner is not None}",
        ]
    )
    if f._cmplog is not None:
        n_tok = len(f._cmplog.tokens)
        n_prs = len(f._cmplog.pairs)
        lines.append(f"  Cmplog:          enabled ({n_tok}t {n_prs}p)")
    else:
        lines.append("  Cmplog:          disabled")
    if getattr(f, "_use_poisson_disk_admission", False):
        rej = getattr(f, "_poisson_reject_count", 0)
        near = getattr(f, "_poisson_near_dup_admit_count", 0)
        n_admitted = len(getattr(f, "_admitted_keys", set()) or ())
        n_buckets = len(getattr(f, "_poisson_occupied_buckets", set()) or ())
        lines.append("  Poisson disk:     enabled")
        lines.append(f"  Admitted seeds:   {n_admitted}")
        lines.append(f"  Poisson rejects:  {rej} (near-dup admitted: {near})")
        lines.append(f"  Occupied buckets: {n_buckets}")
    smt_enabled = False
    with contextlib.suppress(AttributeError):
        smt_enabled = (
            object.__getattribute__(f, "_enable_smt_z3")
            and object.__getattribute__(f, "_smt_solver") is not None
        )
    lines.append(f"  SMT:             {'enabled' if smt_enabled else 'disabled'}")
    if execs > 0:
        lines.append(f"  Crash rate:      {_format_proportion(execs, crashes)}")
        lines.append(f"  Timeout rate:    {_format_proportion(execs, timeouts)}")
    return "\n".join(lines)


def _configuration(f) -> str:
    """Run configuration knobs — how this fuzzing session was set up."""
    rows = []

    def row(label, value):
        if value is not None and value != "" and value != []:
            rows.append(f"  {label:<22s} {value}")

    row("Seed", getattr(f, "seed", None))
    row("Schedule", getattr(f, "_power_schedule", getattr(f, "schedule", None)))
    row("Mutations/input", getattr(f, "mutations_per_input", None))
    row("Resume", getattr(f, "resume", False))
    row("Max input len", f.max_len)
    row("Timeout", f"{f.timeout}s")
    row("Map size", f"{getattr(f, 'map_size', 0):,} bytes")
    multi = getattr(f, "multi_targets", None)
    if multi:
        row("Multi-target", f"{len(multi)} targets")
    row("In-process", f._inprocess_runner is not None)
    if getattr(f, "_inprocess_runner", None) is not None:
        row("Direct-lite", bool(getattr(f._inprocess_runner, "direct", False)))
        row("Persistent", getattr(f._inprocess_runner, "_persistent", None) is not None)
    codes = getattr(f, "extra_crash_codes", None)
    if codes:
        row("Extra crash codes", ",".join(str(c) for c in sorted(codes)))
    row("ASAN target", getattr(f, "asan_target", None))
    row("UBSAN target", getattr(f, "ubsan_target", None))
    row(
        "Cmplog",
        None
        if f._cmplog is None
        else f"enabled ({len(f._cmplog.tokens)}t {len(f._cmplog.pairs)}p)",
    )
    row("Dictionary", None if not f.dictionary else f"{len(f.dictionary)} tokens")
    row("Grammar", None if f.grammar is None else f"{len(f.grammar.rules)} rules")
    row("Markov", None if not getattr(f, "markov_trained", False) else "trained")
    if getattr(f, "markov_generate", False):
        row("Markov gen", "enabled")
    row("Operators", None if not f.op_counts else f"{len(f.op_counts)} exercised")

    if not rows:
        return ""
    return "\n".join([""] + ["--- Configuration ---"] + rows)


def _crash_signatures(f) -> str:
    """Crash signature histogram from f.crash_sigs (SanitizerReport signatures).

    Also groups signatures into stack-similarity clusters (Levenshtein
    distance over normalized frame sequences, via
    :func:`fuzzer_tool.core.crash_metadata.cluster_crashes`) so distinct
    signatures caused by the same underlying bug -- differing only in
    inlined frames or instruction offsets -- are reported as one root
    cause instead of inflating the apparent bug count.
    """
    sigs = getattr(f, "crash_sigs", None)
    if not sigs:
        return ""
    frames = getattr(f, "crash_frames", None) or {}
    lines = ["", "--- Crash Signatures ---"]
    for sig, count in sorted(sigs.items(), key=lambda x: -x[1]):
        line = f"  {count:>4d}x  {sig}"
        if sig in frames and frames[sig]:
            line += "  " + " -> ".join(str(x) for x in frames[sig][:2])
        lines.append(line)

    if len(sigs) > 1:
        from fuzzer_tool.core.crash_metadata import cluster_crashes

        sig_list = list(sigs.keys())
        frame_lists = [frames.get(s, []) for s in sig_list]
        clusters = cluster_crashes(sig_list, frame_lists=frame_lists)
        multi = [c for c in clusters if len(c) > 1]
        if multi:
            lines.append("")
            lines.append(
                f"  Clustered by stack similarity: {len(sigs)} signature(s) -> "
                f"{len(clusters)} likely distinct bug(s)"
            )
            for cluster in sorted(multi, key=len, reverse=True):
                total = sum(sigs[sig_list[i]] for i in cluster)
                members = ", ".join(sig_list[i] for i in cluster)
                lines.append(f"    [{total:>4d}x] {members}")

    return "\n".join(lines)


def _coverage_analysis(f) -> str:
    if not f.shm_cov:
        return ""
    cov = f.shm_cov
    total_seen = cov.cumulative_edges
    density = total_seen / cov.size * 100 if cov.size else 0

    # Cluster analysis: group edges into 256-byte buckets from bitmap.
    # Skip if the coverage object does not expose the raw bitmap.
    buckets = Counter()
    seen = getattr(cov, "_seen", None)
    if seen is not None:
        if _HAS_NUMPY:
            seen_arr = np.frombuffer(seen, dtype=np.uint8)
            bucket_indices = np.flatnonzero(seen_arr) // 256
            for b in bucket_indices:
                buckets[b] += 1
        else:
            for i in range(cov.size):
                if seen[i]:
                    buckets[i // 256] += 1

    lines = [
        "",
        "--- Coverage Analysis ---",
        f"  SHM map size:    {cov.size:,} bytes",
        f"  Unique edges:    {total_seen}",
        f"  Coverage density: {density:.4f}%",
    ]
    dropped = int(cov.read_dropped_edges())
    if dropped:
        lines.append(f"  Dropped edges:   {dropped:,}")

    if buckets:
        lines.append(f"  Edge buckets:    {len(buckets)}")
        lines.append("  Top clusters (256-byte buckets):")
        for bucket, count in sorted(buckets.items(), key=lambda x: -x[1])[:10]:
            addr = bucket * 256
            lines.append(f"    0x{addr:04x}-0x{addr + 255:04x}: {count:3d} edges")

    # Coverage growth timeline from edge tracker (via StateStore)
    from fuzzer_tool.core.state_store import StateStore

    store = StateStore(f.corpus_dir)
    et_data = store.get("edge_tracker")
    # "cumulative_edges" is sorted(self.cumulative_edges) -- the set of edge
    # IDs, not a growth series. Reading it here meant len() was the edge total
    # and every row printed the milestone as both the iteration and the edge
    # count ("iter 100: 100 edges"), with a "final" row claiming iteration 3566
    # on a 1,305-execution run. The growth series is "coverage_timeline":
    # [[exec_count, edge_count], ...] recorded by record_coverage_snapshot.
    timeline = [
        (int(pt[0]), int(pt[1]))
        for pt in (et_data.get("coverage_timeline", []) if et_data else [])
        if len(pt) >= 2
    ]
    if timeline:
        timeline.sort(key=lambda p: p[0])
        final_exec, final_edges = timeline[-1]
        rows: list[tuple[int, int]] = []
        for m in MILESTONES:
            if m > final_exec:
                break  # never reached; do not invent a data point for it
            # Last snapshot at or before the milestone.
            edges_at = None
            for exec_c, edge_c in timeline:
                if exec_c <= m:
                    edges_at = edge_c
                else:
                    break
            if edges_at is not None:
                rows.append((m, edges_at))
        rows.append((final_exec, final_edges))

        # Dedupe on iteration, keeping the last value seen for it, and emit the
        # final marker exactly once. The old loop appended the final row from
        # inside the milestone loop, so it repeated once per milestone.
        seen: dict[int, int] = {}
        for it, ed in rows:
            seen[it] = ed
        lines.append("  Coverage growth:")
        for it in sorted(seen):
            marker = " (final)" if it == final_exec else ""
            lines.append(f"    iter {it:>6,d}: {seen[it]:>6,d} edges{marker}")

    return "\n".join(lines)


def _applicable_counts(f) -> dict:
    """``f.op_applicable`` when it is really a dict.

    isinstance, not getattr-with-default: report tests drive these
    functions with MagicMock fuzzers, where every attribute exists and is
    truthy, so a plain getattr yields a Mock that sails through the `or`
    and then fails arithmetic several frames later.
    """
    applicable = getattr(f, "op_applicable", None)
    return applicable if isinstance(applicable, dict) else {}


def _mutation_effectiveness(f) -> str:
    counts = f.op_counts
    successes = f.op_success
    if not counts:
        return ""

    # A sniffer-gated operator runs in one of two regimes. Handed a file of
    # its own format it mutates that file; handed anything else it
    # synthesises a fresh file of its format from scratch. Both can find
    # edges, so neither set of selections can simply be dropped -- but
    # pooling them makes the reported rate depend on how much of the corpus
    # happens to be that format, which is a property of the corpus and not
    # of the operator. Two mechanisms in _format_available keep feeding the
    # second regime: the bootstrap trickle offers a never-seen format on
    # non-matching input, and once a format has been seen even once the
    # live-format short circuit offers it on *every* input thereafter.
    #
    # So both pairs are reported. Count/Success/Rate is the raw picture --
    # what the budget was spent on and what came back. Applic/SuccA/RateA
    # is the same question restricted to the mutate-this-file regime.
    # For an ungated operator the two pairs are identical by construction,
    # which makes the invariant visible on every line.
    applicable = _applicable_counts(f)
    succ_applicable = getattr(f, "op_success_applicable", None)
    if not isinstance(succ_applicable, dict):
        succ_applicable = {}

    total = sum(counts.values())
    total_success = sum(successes.values())

    lines = [
        "",
        "--- Mutation Effectiveness ---",
        f"  {'Operation':<22s} {'Count':>7s} {'Success':>7s} {'Rate':>6s} "
        f"{'Applic':>7s} {'SuccA':>6s} {'RateA':>6s}  "
        f"{'±1σ':>7s} {'±2σ':>7s} {'±3σ':>7s}",
        f"  {'-' * 22} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 7} {'-' * 6} {'-' * 6}  "
        f"{'-' * 7} {'-' * 7} {'-' * 7}",
    ]

    split_regimes = False
    for op, count in sorted(counts.items(), key=lambda x: -x[1]):
        succ = successes.get(op, 0)
        rate = succ / count * 100 if count else 0.0
        # Missing means unknown -- a state file written before this was
        # tracked -- and falls back to the raw pair. Present-and-zero means
        # the operator genuinely never saw its own format.
        appl = applicable.get(op, count)
        # Default 0 when applicability is known for this op (it simply
        # never succeeded in that regime) and `succ` only when it is not
        # tracked at all -- defaulting to succ in the known case is how a
        # zero-denominator row still showed a numerator.
        succ_a = succ_applicable.get(op, 0 if op in applicable else succ)
        if appl < count:
            split_regimes = True
        # Intervals follow the restricted pair: it is the one making a
        # claim about the operator rather than about the corpus.
        _, se, c1, c2, c3 = _confidence_interval(appl, succ_a)
        # n/a, not 0.0%, when the operator never met its own format:
        # undefined is not zero, and printing zero is what made a working
        # operator read as broken -- same reasoning as Edges/Success below.
        rate_a = f"{'n/a':>6s}" if appl <= 0 else f"{succ_a / appl * 100:>5.1f}%"
        lines.append(
            f"  {op:<22s} {count:>7d} {succ:>7d} {rate:>5.1f}% "
            f"{appl:>7d} {succ_a:>6d} {rate_a}  "
            f"{c1 * 100:>6.1f}% {c2 * 100:>6.1f}% {c3 * 100:>6.1f}%"
        )

    lines.append(
        f"  {'TOTAL':<22s} {total:>7d} {total_success:>7d} {total_success / total * 100:>5.1f}%"
        if total
        else ""
    )
    if split_regimes:
        lines += [
            "  Applic/SuccA/RateA restrict to selections where the operator's own",
            "  format matched the input it was handed -- i.e. it mutated a file of",
            "  that format rather than synthesising one from scratch. RateA is n/a",
            "  where that never happened; the raw Rate still covers those runs.",
        ]
    return "\n".join(lines)


def _comparison_profile(f) -> str:
    """Per-callback comparison counts from the shim's counter channel.

    Answers what the CMP record stream structurally cannot: how many
    comparisons each callback actually executed, and how many of those were
    satisfied. The records carry no callback identity, the collector dedups
    multiplicity away, and satisfied layer-1 comparisons are never written at
    all -- the record writer drops ``result == 0`` to keep solved compares
    out of the input-to-state pair pool.

    Read it as a solve-progress signal: a site with a high fire count and a
    near-zero assert rate is a comparison the campaign keeps reaching and
    keeps failing, which is where cmplog-driven mutation has room to work.
    """
    cmplog = getattr(f, "_cmplog", None)
    if cmplog is None:
        return ""
    stats = cmplog.comparison_stats()
    if not stats:
        return ""

    total_fired, total_asserted = cmplog.total_comparisons()
    walls = cmplog.comparison_walls()
    lines = [
        "",
        "--- Comparison Profile ---",
        f"  {'Callback':<20s} {'Fired':>14s} {'Asserted':>14s} {'Rate':>7s}",
        f"  {'-' * 20} {'-' * 14} {'-' * 14} {'-' * 7}",
    ]
    for name, (fired, asserted) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        rate = f"{asserted / fired * 100:>6.1f}%" if fired else f"{'n/a':>7s}"
        mark = ""
        if name in walls:
            trend = walls[name][2]
            mark = f"  <- wall ({trend})" if trend != "unknown" else "  <- wall"
        lines.append(f"  {name:<20s} {fired:>14,d} {asserted:>14,d} {rate}{mark}")

    total_rate = f"{total_asserted / total_fired * 100:>6.1f}%" if total_fired else f"{'n/a':>7s}"
    lines.append(f"  {'TOTAL':<20s} {total_fired:>14,d} {total_asserted:>14,d} {total_rate}")
    lines.append("  Fired = callback entered; Asserted = the predicate held (operands")
    lines.append("  equal, needle found, non-empty span, or a matched switch case).")
    lines.append("  A switch dispatch counts once, not once per case.")
    if walls:
        lines.append("  A wall is a family reached constantly and passed essentially never. The")
        lines.append("  trend is its fire rate: rising means the campaign still reaches it and")
        lines.append("  keeps failing; falling means it stopped reaching it. Family, not site --")
        lines.append("  two call sites share one bucket.")
    return "\n".join(lines)


def _mutation_edge_attribution(f) -> str:
    """Per-operator new-edge attribution report.

    Shows how many new edges each mutation operator discovered (proportional
    attribution when multiple operators run per iteration). Cmplog is shown
    separately as cumulative involvement — it overlaps with mutation ops.
    """
    edges = f.op_edges
    if not edges:
        return ""

    # Filter out cmplog for the proportional table
    op_edges = {k: v for k, v in edges.items() if k != "cmplog"}
    if not op_edges:
        return ""

    total = sum(op_edges.values())
    # Raw op_counts, deliberately, not the applicable denominator used for
    # the rate column above: a format operator handed input of the wrong
    # format synthesises a file of its own format instead, and edges found
    # that way are real edges found on that selection. Every use is an
    # opportunity to find an edge, whichever regime it ran in.
    counts = f.op_counts
    successes = f.op_success
    cmplog_edge_count = edges.get("cmplog", 0)

    lines = [
        "",
        "--- Mutation Edge Attribution ---",
        f"  {'Operation':<22s} {'Edges':>8s} {'%Total':>7s} "
        f"{'Edges/Use':>10s} {'Edges/Success':>13s}",
        f"  {'-' * 22} {'-' * 8} {'-' * 7} {'-' * 10} {'-' * 13}",
    ]

    for op, edge_val in sorted(op_edges.items(), key=lambda x: -x[1]):
        count = counts.get(op, 0)
        succ = successes.get(op, 0)
        pct = edge_val / total * 100 if total > 0 else 0
        edges_per_use = edge_val / count if count > 0 else 0
        # With zero successes the ratio is undefined, not zero. Printing 0.00
        # made operators that were credited with edges (byte_shuffle: 27.1)
        # look like they produced nothing per success.
        eps_str = f"{edge_val / succ:>12.2f}" if succ > 0 else f"{'n/a':>12s}"
        lines.append(f"  {op:<22s} {edge_val:>8.1f} {pct:>6.1f}%  {edges_per_use:>9.2f}  {eps_str}")

    lines.append(f"  {'TOTAL':<22s} {total:>8.1f} {'':>7s}")

    # Cmplog involvement line (cumulative, overlaps with mutations above)
    if cmplog_edge_count > 0:
        cmplog_pct = cmplog_edge_count / (total + cmplog_edge_count) * 100
        lines.append("")
        lines.append(
            f"  Cmplog-involved edges: {cmplog_edge_count:.0f} ({cmplog_pct:.1f}% of "
            f"total edge discoveries involved cmplog token extraction)"
        )
        lines.append(
            "  (cmplog is a signal source, not a mutation operator — "
            "its count overlaps with the per-op attribution above)"
        )

    # SMT solver involvement line (cumulative, overlaps with mutations above)
    smt_edge_count = edges.get("smt_solver", 0)
    if smt_edge_count > 0:
        smt_pct = smt_edge_count / (total + smt_edge_count) * 100
        lines.append("")
        lines.append(
            f"  SMT-involved edges: {smt_edge_count:.0f} ({smt_pct:.1f}% of "
            f"total edge discoveries involved SMT constraint solving)"
        )
        lines.append(
            "  (SMT is a signal source, not a mutation operator — "
            "its count overlaps with the per-op attribution above)"
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


def _smt_solver_activity(f) -> str:
    """SMT solver activity: queries attempted, solved, timed out."""
    try:
        solver = object.__getattribute__(f, "_smt_solver")
    except AttributeError:
        return ""
    if solver is None:
        return ""
    lines = ["", "--- SMT Solver Activity ---"]
    try:
        stats = solver.stats
        if not isinstance(stats, dict):
            lines.append("  Solver enabled (stats unavailable)")
            return "\n".join(lines)
        attempted = stats.get("queries_attempted", 0)
        solved = stats.get("queries_solved", 0)
        timed_out = stats.get("queries_failed", 0)
        solved_pct = solved / attempted * 100 if attempted > 0 else 0.0
    except (TypeError, AttributeError):
        lines.append("  Solver enabled")
        return "\n".join(lines)
    lines.extend(
        [
            f"  Queries attempted: {attempted}",
            f"  Queries solved:    {solved} ({solved_pct:.1f}%)",
            f"  Queries failed:   {timed_out}",
        ]
    )
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
            name = _preview(seed, 40)
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


def _scan_corpus_files(corpus_dir) -> list[Path]:
    """Every file load_corpus() would treat as a seed.

    Seeds live under subdirectories of corpus_dir (recursively), excluding
    pruned/ and delta_*.json. A flat iterdir() on corpus_dir itself never
    descends into seeds/ and instead picks up state.pkl.gz -- the unified
    fuzzer state file, several MB by the end of a run -- which is not a seed
    and was never loaded as an input.
    """
    p = Path(corpus_dir)
    if not p.exists():
        return []
    entries: list[Path] = []
    for sub in p.iterdir():
        if not sub.is_dir() or sub.name == "pruned":
            continue
        for entry in sub.rglob("*"):
            if not entry.is_file():
                continue
            if "pruned" in entry.relative_to(sub).parts:
                continue  # nested pruned/ (e.g. seeds/pruned/), same as load_corpus
            if entry.suffix == ".json" and entry.name.startswith("delta_"):
                continue
            entries.append(entry)
    return entries


def _corpus_overview(f, corpus_dir) -> str:
    """Summarize on-disk corpus seed files.

    Must mirror what load_corpus() actually treats as a seed: every file
    under a subdirectory of corpus_dir (recursively), excluding pruned/
    and delta_*.json. Previously this did a flat p.iterdir() on
    corpus_dir itself, which never descends into seeds/ at all -- so it
    reported 0 real seeds and instead picked up whatever sits directly at
    the corpus root, namely state.pkl.gz (the unified fuzzer state file,
    several MB by the end of a run with GA/QEA/Markov/etc. active). That
    showed up as "1 corpus file, N MB" and was mistaken for an oversized
    seed -- state.pkl.gz was never loaded as an input by load_corpus, it
    was just the only thing this scan could see.
    """
    entries = _scan_corpus_files(corpus_dir)
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
    # Scale to the largest bucket. min(count, 40) clamps every bucket over 40
    # to a full-width bar, so 123 / 187 / 109 all rendered identically and the
    # histogram carried no information at all.
    peak = max(buckets.values()) if buckets else 0
    for bucket, count in buckets.items():
        bar = "#" * round(count / peak * 40) if peak else ""
        lines.append(f"    {bucket:<12s} {count:>4d} {bar}")

    return "\n".join(lines)


def _crash_analysis(f, crashes_dir) -> str:
    if f.crash_count == 0:
        return ""

    p = Path(crashes_dir)
    if not p.exists():
        return ""

    # save_crash() writes ONE input (`<base>.bin`) plus up to three sidecars
    # (`.txt` report, `.sh` repro script, `.hex` dump) per crash, so counting
    # every file in the directory reported ~4x the real crash count, and fed
    # the size histogram and the sample list with sidecar text as if it were
    # crash input. Count inputs only.
    crashes = [f for f in p.iterdir() if f.is_file() and f.suffix == CRASH_INPUT_SUFFIX]
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
        "--- Chao2 Coverage Estimation (incidence) ---",
        f"  Edges observed:      {gt['n']}",
        f"  Seeds sampled:       {gt.get('m', 0)}",
        f"  Q1 (1 seed):         {gt['n1']}",
        f"  Q2 (2 seeds):        {gt['n2']}",
        f"  Est. undiscovered:   {gt['estimated_undiscovered']}",
        f"  Total richness:      {gt.get('ci_low', 0):.0f} - {gt.get('ci_high', 0):.0f} (95% CI)",
        f"  Saturation:          {gt['saturation']:.1%}",
        f"  P(next seed is new): {gt.get('discovery_probability', 0.0):.2%}",
        f"  Confidence:          {gt['confidence']}",
    ]
    if f.shm_cov:
        dropped = int(f.shm_cov.read_dropped_edges())
        if dropped:
            lines.append(f"  Dropped edges:       {dropped:,}")
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
    """On-disk size profile of the seed corpus.

    Uses the same scan as _corpus_overview. This used to do a flat
    p.iterdir() on corpus_dir, which sees none of the seeds (they live under
    subdirectories) and instead picks up state.pkl.gz -- so the section
    reported "6 corpus files, 1.7MB" for a 419-file, 330KB corpus, with the
    multi-megabyte state file counted as a single oversized seed.
    """
    entries = _scan_corpus_files(corpus_dir)
    if not entries:
        return ""
    sizes = [e.stat().st_size for e in entries]
    total_size = sum(sizes)
    lines = [
        "",
        "--- Disk Footprint ---",
        f"  Corpus files:    {len(entries)}",
        f"  Total size:      {_human_size(total_size)}",
    ]
    # Delta vs full analysis: check if any files are very small (< 100 bytes) vs large
    small = sum(1 for sz in sizes if sz < 100)
    large = len(sizes) - small
    if small:
        lines.append(f"  Small (<100B):   {small} {_plural(small, 'file')} (potential deltas)")
        lines.append(f"  Large (>=100B):  {large} {_plural(large, 'file')}")
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
    ]

    # Brier score CI from individual prediction errors
    brier_history = f.mc._brier_history if hasattr(f.mc, "_brier_history") else []
    if brier_history and len(brier_history) >= 10:
        n = len(brier_history)
        mean = sum(brier_history) / n
        if _HAS_NUMPY:
            var = float(np.var(brier_history, ddof=1))
        else:
            var = sum((x - mean) ** 2 for x in brier_history) / (n - 1) if n > 1 else 0
        se = (var / n) ** 0.5
        ci1, ci2, ci3 = se, se * 2, se * 3
        lines.append(
            f"  Brier score:       {_format_ci_inline(mean, ci1, ci2, ci3, '.4f', lo_clamp=0.0)} "
            f"{_BRIER_LEGEND}"
        )
    else:
        lines.append(f"  Brier score:       {brier:.4f} {_BRIER_LEGEND}")
    cal = f.mc.calibration_report()
    if cal:
        lines.append("  Calibration by predicted probability bin:")
        lines.append(f"    {'Bin':<12s} {'Predicted':>10s} {'Actual':>10s} {'Samples':>8s}")
        for bin_label, entry in cal.items():
            # calibration_report now returns (predicted, actual, n). Tolerate
            # the old 2-tuple so a stale pickled scheduler doesn't crash the
            # report.
            pred, actual = entry[0], entry[1]
            n = entry[2] if len(entry) > 2 else None
            n_str = f"{n:>8d}" if n is not None else f"{'?':>8s}"
            lines.append(f"    {bin_label:<12s} {pred:>10.3f} {actual:>10.3f} {n_str}")
    return "\n".join(lines)


def _execution_time_analysis(f) -> str:
    tracker = f._exec_time_tracker
    if tracker.count < 10:
        return ""
    lines = [
        "",
        "--- Execution Time Analysis ---",
        f"  Observations:   {tracker.count} sampled of {f.exec_count:,} executions",
        f"  p50:            {tracker.p50 * 1000:.1f}ms",
        f"  p99:            {tracker.p99 * 1000:.1f}ms",
        f"  Suggested timeout: {tracker.suggested_timeout():.2f}s",
    ]
    # Whether that suggestion did anything. It was print-only for its whole
    # existence, so a reader had no way to tell the two cases apart.
    retunes = getattr(f, "_timeout_retunes", [])
    if getattr(f, "_adaptive_timeout", False) or retunes:
        lines.append(f"  Active timeout: {f.timeout:.3f}s (adaptive, {len(retunes)} retunes)")
        for at_exec, old, new in retunes[-5:]:
            lines.append(f"    exec {at_exec:>7,d}: {old:.3f}s -> {new:.3f}s")
    else:
        lines.append(f"  Active timeout: {f.timeout:.3f}s (fixed; --adaptive-timeout to retune)")
    lines += [
        f"  CRPS (mean):    {tracker.mean_crps():.6f}",
        f"  CRPS trend:     {tracker.crps_trend():.6f} (+ = degrading)",
    ]
    if tracker.crps_trend() > 0.001:
        lines.append("  WARNING: CRPS rising — target runtime behavior is drifting")
    return "\n".join(lines)


def _distribution_diagnostics(f) -> str:
    """Distribution diagnostics: stddev, skewness, kurtosis for key signal sources."""
    lines = ["", "--- Distribution Diagnostics ---"]
    has_data = False

    # Execution time moments
    try:
        tracker = f._exec_time_tracker
        if tracker and int(tracker.count) >= 3:
            has_data = True
            m = tracker._moments
            lines.append(
                f"  Exec time:       mean={float(m.mean) * 1000:.1f}ms  "
                f"stddev={float(m.stddev) * 1000:.1f}ms  "
                f"skew={float(m.skewness):.2f}  kurt={float(m.kurtosis):.2f}"
            )
            if tracker.tail_risk:
                lines.append("                    TAIL_RISK: heavy right skew detected")
    except (TypeError, AttributeError):
        pass

    # Discovery rate moments (from CSD detector)
    try:
        csd = f._csd
        if csd and hasattr(csd, "_history") and len(csd._history) >= 3:
            has_data = True
            from fuzzer_tool.core.running_stats import RunningMoments

            dr_moments = RunningMoments()
            for v in csd._history:
                dr_moments.update(float(v))
            lines.append(
                f"  Discovery rate:  mean={dr_moments.mean:.2f}  "
                f"stddev={dr_moments.stddev:.2f}  "
                f"skew={dr_moments.skewness:.2f}  kurt={dr_moments.kurtosis:.2f}"
            )
    except (TypeError, AttributeError):
        pass

    # Per-operator reward moments (from Elo tracker)
    try:
        elo = f._elo
        if elo and hasattr(elo, "_reward_moments") and elo._reward_moments:
            has_data = True
            lines.append("  Operator rewards:")
            for op, rm in sorted(elo._reward_moments.items(), key=lambda x: -(x[1].mean)):
                if rm.count >= 3:
                    lines.append(
                        f"    {op:<22s} mean={rm.mean:.3f}  "
                        f"stddev={rm.stddev:.3f}  "
                        f"skew={rm.skewness:.2f}  kurt={rm.kurtosis:.2f}"
                    )
    except (TypeError, AttributeError):
        pass

    # Seed size moments (corpus bloat indicator)
    try:
        seed_moments = f._seed_size_moments
        if seed_moments and int(seed_moments.count) >= 10:
            has_data = True
            lines.append(
                f"  Seed sizes:      mean={float(seed_moments.mean):.0f}B  "
                f"stddev={float(seed_moments.stddev):.0f}B  "
                f"skew={float(seed_moments.skewness):.2f}  kurt={float(seed_moments.kurtosis):.2f}"
            )
            if float(seed_moments.skewness) > 2.0:
                lines.append("                    BLOAT WARNING: rising right tail in seed sizes")
    except (TypeError, AttributeError):
        pass

    if not has_data:
        return ""
    return "\n".join(lines)


def _peak_intervals(series: list[float]) -> list[float]:
    """Return peak-to-peak intervals for strict 1-sample local maxima."""
    if len(series) < 3:
        return []
    intervals: list[float] = []
    last_peak: float | None = None
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            if last_peak is not None:
                intervals.append(series[i] - last_peak)
            last_peak = series[i]
    return intervals


def _spectral_diagnostics(f) -> str:
    """Spectral (FFT) diagnostics: periodic components in key time series.

    Two scans, both via rfft magnitude spectra:
    - Exec-time series (one sample per recorded execution) — a genuine
      periodic component would be unusual and worth flagging.
    - Discovery-rate series (first-differences of cumulative edges per sync
      interval) — a peak at a short period suggests corpus-sync artifacts
      imprinting fake discovery waves, not real coverage breakthroughs.
    """
    if not _HAS_NUMPY:
        return ""
    from fuzzer_tool.core.periodicity import (
        classify_periodicity,
        detect_periodicity,
        harmonic_fraction,
    )

    lines = ["", "--- Spectral Diagnostics ---"]
    has_data = False

    # Execution-time series (samples = one per recorded execution)
    try:
        tracker = f._exec_time_tracker
        times = list(getattr(tracker, "_times", []) or [])
        if len(times) >= 50:
            has_data = True
            res = detect_periodicity([float(t) for t in times], min_samples=50)
            if res.significant:
                lines.append(
                    f"  Exec time:      PERIODIC — dominant period {res.dominant_period:.1f} "
                    f"samples (g={res.peak_strength:.3f}, p={res.p_value:.2e} at bin {res.peak_bin})"
                )
                if res.dominant_period is not None:
                    intervals = _peak_intervals([float(t) for t in times])
                    if len(intervals) >= 2:
                        fracs = harmonic_fraction(intervals, res.dominant_period)
                        verdict = classify_periodicity(fracs["total"])
                        lines.append(
                            f"  Exec time:      Harmonic confirmation — {fracs['total']:.0%} of "
                            f"peak intervals on harmonics of {res.dominant_period:.1f} ({verdict})"
                        )
            else:
                lines.append("  Exec time:      no significant periodic component")
    except (TypeError, AttributeError):
        pass

    # Discovery-rate series: first-differences of cumulative edges per sync interval
    try:
        edges_series = f._discovery_edges
        if len(edges_series) >= 51:
            has_data = True
            deltas = [b - a for a, b in zip(edges_series[:-1], edges_series[1:], strict=True)]
            res = detect_periodicity([float(d) for d in deltas], min_samples=50)
            if res.significant:
                lines.append(
                    f"  Discovery rate: PERIODIC — dominant period {res.dominant_period:.1f} "
                    f"sync intervals (g={res.peak_strength:.3f}, p={res.p_value:.2e} at bin "
                    f"{res.peak_bin}); possible corpus-sync artifact"
                )
                if res.dominant_period is not None:
                    intervals = _peak_intervals([float(d) for d in deltas])
                    if len(intervals) >= 2:
                        fracs = harmonic_fraction(intervals, res.dominant_period)
                        verdict = classify_periodicity(fracs["total"])
                        lines.append(
                            f"  Discovery rate: Harmonic confirmation — {fracs['total']:.0%} of "
                            f"peak intervals on harmonics of {res.dominant_period:.1f} ({verdict})"
                        )
            else:
                lines.append("  Discovery rate: no significant periodic component")
    except (TypeError, AttributeError):
        pass

    if not has_data:
        return ""
    return "\n".join(lines)


def _temporal_correlation(f) -> str:
    """Correlate coverage and discovery snapshots via temporal join."""
    try:
        et = f._edge_tracker
        cov_ts = getattr(et, "_coverage_timestamps", None)
        cov_execs = getattr(et, "_coverage_execs", None)
        cov_edges = getattr(et, "_coverage_edges", None)
        disc_ts = getattr(f, "_discovery_timestamps", None)
        disc_execs = getattr(f, "_discovery_execs", None)
        disc_edges = getattr(f, "_discovery_edges", None)
        if not (cov_ts and disc_ts):
            return ""
        n_cov = min(len(cov_ts), len(cov_execs), len(cov_edges))
        n_disc = min(len(disc_ts), len(disc_execs), len(disc_edges))
        if n_cov < 2 or n_disc < 2:
            return ""
        coverage_stream = [
            (float(cov_ts[i]), (int(cov_execs[i]), int(cov_edges[i]))) for i in range(n_cov)
        ]
        discovery_stream = [
            (float(disc_ts[i]), (int(disc_execs[i]), int(disc_edges[i]))) for i in range(n_disc)
        ]
        joined = join_streams([coverage_stream, discovery_stream], max_gap=5.0)
        if not joined:
            return ""
        deltas = []
        for cov, disc in joined:
            cov_exec, cov_edge = cov
            disc_exec, disc_edge = disc
            edge_delta = cov_edge - disc_edge
            exec_delta = cov_exec - disc_exec
            if exec_delta > 0:
                deltas.append(edge_delta / exec_delta * 1000)
            elif exec_delta == 0:
                deltas.append(0.0)
        if not deltas:
            return ""
        avg_rate = sum(deltas) / len(deltas)
        return (
            "\n  Temporal correlation: coverage-vs-discovery aligned at "
            f"{len(joined)} sync points (avg edge-rate delta={avg_rate:.2f} ed/kexec)"
        )
    except (TypeError, AttributeError, ValueError):
        return ""


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
    _ent = _corpus_byte_entropy(f.corpus)
    if _ent is not None:
        lines.append(f"  Byte entropy:      {_ent:.2f} bits (max=8.0)")
    return "\n".join(lines)


def _crash_exploitability(f, crashes_dir: str) -> str:
    """Exploitability tier distribution from crash metadata."""
    if f.crash_count == 0:
        return ""

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
    seen = getattr(cov, "_seen", None)
    if not seen or not any(seen):
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
    ]

    # Throughput with CI
    tracker = f._exec_time_tracker
    if tracker and tracker.count >= 2:
        # Throughput CI: uses std of execution times to estimate throughput variance
        # SE(throughput) ≈ throughput * (std / mean) / sqrt(n)
        mean_t = tracker.p50  # use median as robust mean proxy
        std_t = tracker.std
        n = tracker.count
        if mean_t > 0 and n > 1:
            cv = std_t / mean_t  # coefficient of variation
            se_eps = eps * cv / math.sqrt(n)
            ci1, ci2, ci3 = se_eps, se_eps * 2, se_eps * 3
            lines.append(
                f"  Avg throughput:   {_format_ci_inline(eps, ci1, ci2, ci3, '.1f')} execs/sec"
            )
        else:
            lines.append(f"  Avg throughput:   {eps:.1f} execs/sec")
    else:
        lines.append(f"  Avg throughput:   {eps:.1f} execs/sec")

    lines.append(f"  Peak throughput:  {f._peak_eps:.1f} execs/sec")
    lines.append(f"  Peak RSS:         {rss_str}")
    lines.append(f"  Map size:         {f.map_size:,} bytes")

    # Corpus growth
    attempts = f._total_corpus_attempts
    rejected = f._duplicate_reject_count
    pruned = f._pruned_count
    # _total_corpus_attempts counts candidate insertions, not accepted seeds;
    # labelling it "Seeds added" put it next to a corpus size it could never
    # reconcile with (188 "added" against a 489-seed corpus).
    lines.append(f"  Corpus attempts:  {attempts}")
    lines.append(f"  Seeds accepted:   {max(attempts - rejected, 0)}")
    lines.append(f"  Duplicates:       {rejected} rejected")
    if pruned > 0:
        lines.append(f"  Seeds pruned:     {pruned}")

    # Dup rejection rate
    if attempts > 0:
        dup_rate = rejected / attempts * 100
        lines.append(f"  Dup rejection:    {dup_rate:.1f}% of attempts")

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


def _entropy_metrics(f) -> str:
    """Entropy and diversity metrics for coverage and corpus analysis."""
    import math

    lines = ["", "--- Entropy & Diversity Metrics ---"]

    # Shannon entropy of edge hits
    if f._edge_tracker and f._edge_tracker._global_edge_hits:
        try:
            ent = float(f._edge_tracker.shannon_entropy_global())
            simp = float(f._edge_tracker.simpson_diversity_global())
            n_edges = len(f._edge_tracker._global_edge_hits)
            max_ent = math.log2(n_edges) if n_edges > 1 else 0
            lines.append(f"  Edge entropy:     {ent:.2f} bits (max={max_ent:.2f})")
            lines.append(f"  Simpson diversity: {simp:.4f} (0=monoculture, 1=uniform)")
        except (TypeError, ValueError):
            lines.append("  Edge entropy:     n/a")
            lines.append("  Simpson diversity: n/a")
    else:
        lines.append("  Edge entropy:     n/a (no coverage data)")
        lines.append("  Simpson diversity: n/a")

    # Coverage uniformity via Rényi spectrum
    if f._edge_tracker and f._edge_tracker._global_edge_hits:
        try:
            hits = f._edge_tracker._global_edge_hits
            total = sum(hits.values())
            if total > 0:
                max_hit = max(hits.values())
                h_inf = -math.log2(max_hit / total) if max_hit > 0 else 0
                h_0 = math.log2(len(hits))
                uniformity = h_inf / h_0 if h_0 > 0 else 1.0
                lines.append(f"  Coverage uniformity: {uniformity:.4f} (1.0=perfectly uniform)")
        except (TypeError, ValueError):
            pass

    # Entropy rate of change
    if hasattr(f, "_entropy_execs") and len(f._entropy_execs) >= 2:
        try:
            recent = list(zip(f._entropy_execs[-10:], f._entropy_vals[-10:], strict=True))
            if len(recent) >= 2:
                dt = recent[-1][0] - recent[0][0]
                if dt > 0:
                    dS = recent[-1][1] - recent[0][1]
                    rate = dS / dt
                    label = "rising" if rate > 0.001 else ("falling" if rate < -0.001 else "flat")
                    lines.append(f"  Entropy rate (dS/dt): {rate:+.6f} ({label})")
                    lines.append(
                        f"  Entropy samples:     {len(f._entropy_execs)} (window={recent[-1][0] - recent[0][0]} execs)"
                    )
        except (TypeError, IndexError):
            pass
    else:
        lines.append("  Entropy rate:     n/a (insufficient samples)")

    # Byte entropy of corpus (same helper as Corpus Health -- one metric, one value)
    if f.corpus and isinstance(f.corpus, list):
        _ent = _corpus_byte_entropy(f.corpus)
        if _ent is not None:
            lines.append(f"  Corpus byte entropy: {_ent:.2f} bits (max=8.0)")

    return "\n".join(lines)


def _format_learning(f) -> str:
    """Format structure learning results (schema-harness methodology)."""
    fl = getattr(f, "_format_learner", None)
    if fl is None:
        return ""

    lines = ["", "--- Format Structure Learning ---"]

    try:
        if not fl.hypotheses and not fl.timeline:
            lines.append("  Status:          not enabled or no data collected")
            return "\n".join(lines)
    except (TypeError, AttributeError):
        lines.append("  Status:          not available (mock object)")
        return "\n".join(lines)

    try:
        # Overview
        summary = fl.get_format_summary()
        lines.append(f"  Timeline:        {summary['timeline_size']} transitions recorded")
        lines.append(
            f"  Hypotheses:      {summary['hypotheses']} total, {summary['classified']} classified"
        )
        lines.append(f"  Model version:   {summary['model_version']}")
        lines.append(
            f"  Backtest:        {summary['backtest_passes']} passes, {summary['backtest_fails']} fails"
        )

        # Learned format (magic at offset 0)
        magic_fields = [f for f in summary["fields"] if f.get("type") == "magic"]
        if magic_fields:
            magic = magic_fields[0]
            lines.append(f"  Learned format:  {magic['width']}-byte identifier at offset 0")

        # Backtest verdict
        passes = int(summary["backtest_passes"])
        fails = int(summary["backtest_fails"])
        if passes > 0 and fails == 0:
            lines.append("  Model status:    CERTIFIED (all backtests passed)")
        elif fails > 0:
            lines.append("  Model status:    UNCERTIFIED (backtest failures detected)")
        else:
            lines.append("  Model status:    PENDING (no backtest run yet)")

        # Field map
        if summary["fields"]:
            lines.append("")
            lines.append("  Inferred format fields:")
            lines.append(
                f"    {'Offset':>8s}  {'Width':>5s}  {'Type':>10s}  {'Conf':>5s}  {'Obs':>4s}  {'Edges':>6s}  {'Sensitive ops'}"
            )
            lines.append(
                f"    {'------':>8s}  {'-----':>5s}  {'----':>10s}  {'----':>5s}  {'---':>4s}  {'-----':>6s}  {'-------------'}"
            )
            for field in summary["fields"]:
                ops = ", ".join(sorted(field["sensitive_ops"].keys())[:4])
                if len(field["sensitive_ops"]) > 4:
                    ops += f" +{len(field['sensitive_ops']) - 4}"
                lines.append(
                    f"    {field['offset']:>8d}  {field['width']:>5d}  {field['type']:>10s}  "
                    f"{field['confidence']:>5.2f}  {field['observations']:>4d}  {field['controlled_edges']:>6d}  {ops}"
                )

            # Field type legend
            lines.append("")
            lines.append("  Field types:")
            lines.append("    magic    — file identifier / signature bytes (offset 0)")
            lines.append("    length   — size/length field (changing it alters coverage patterns)")
            lines.append("    crc      — checksum / integrity field (broad coverage sensitivity)")
            lines.append("    data     — payload or content bytes (many observations)")
            lines.append("    unknown  — structure detected, type not yet classified")

        # Discriminating probe suggestion
        try:
            probe = fl.suggest_discriminating_mutation(
                list(f.op_counts.keys()) if f.op_counts else []
            )
            if probe and isinstance(probe, tuple) and len(probe) == 2:
                op, offset = probe
                lines.append(f"\n  Suggested probe: {op} at offset {offset}")
        except (TypeError, AttributeError, ValueError):
            pass

    except (TypeError, AttributeError):
        lines.append("  Status:          not available")

    # PPMD corpus compression stats
    ppmd = getattr(f, "_ppmd", None)
    if ppmd and getattr(ppmd, "enabled", False) and f.corpus:
        try:
            stats = ppmd.compute_corpus_stats(f.corpus)
            lines.append("")
            lines.append("  PPMD Compression:")
            lines.append(f"    Corpus ratio:   {stats['corpus_ratio']:.4f} (lower=more novel)")
            lines.append(f"    Mean ratio:     {stats['mean_ratio']:.4f}")
            lines.append(f"    Median ratio:   {stats['median_ratio']:.4f}")
            lines.append(f"    Min ratio:      {stats['min_ratio']:.4f} (most novel)")
            lines.append(f"    Max ratio:      {stats['max_ratio']:.4f} (most redundant)")
            lines.append(f"    Total raw:      {stats['total_raw']:,} bytes")
            lines.append(f"    Total compressed:{stats['total_compressed']:,} bytes")
        except (TypeError, AttributeError):
            pass

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

    # MC Floyd cycle detection (opt-in; --mc-cycle-detect). Only shown once
    # the check has actually run at least once -- cycle_checks stays 0 both
    # when the flag was never passed and when every power iteration so far
    # converged normally without needing the check, so this line only
    # appears when there's something to report.
    if f.mc and getattr(f, "mc_cycle_detect", False) and f.mc.cycle_checks > 0:
        stats = f.mc.cycle_stats()
        strategies.append(
            f"  MC Cycle-detect:  {stats['checks']} check(s), "
            f"{stats['detections']} periodic chain(s) found"
        )
        if stats["detections"] > 0:
            strategies.append(
                f"    Period:         last={stats['last_period']}, max={stats['max_period']}"
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

    # Labels are built from the thresholds the stats actually applied. They
    # used to be hardcoded as 2-5/6-20/>20 while the non-Morris code path
    # bucketed at 2-3/4-10/>10, so the legend contradicted the numbers.
    cold_hi, warm_hi = rarity.get("bounds", (3, 10))
    lines = [
        "",
        "--- Edge Rarity ---",
        f"  Total edges:      {rarity['total']}",
        f"  {'Singleton (1 seed):':<18s}{rarity['singleton']}",
        f"  {f'Cold (2-{cold_hi} seeds):':<18s}{rarity['cold']}",
        f"  {f'Warm ({cold_hi + 1}-{warm_hi} seeds):':<18s}{rarity['warm']}",
        f"  {f'Hot (>{warm_hi} seeds):':<18s}{rarity['hot']}",
        f"  {'Avg seeds/edge:':<18s}{rarity['avg_seeds_per_edge']:.1f}",
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
    if len(f._crash_rate_execs) < 2:
        return ""

    lines = ["", "--- Crash Rate Trend ---"]

    # Sample at milestones
    execs = f._crash_rate_execs
    counts = f._crash_rate_counts
    shown = set()
    for m in MILESTONES:
        for i, exec_c in enumerate(execs):
            if exec_c >= m and m not in shown:
                crash_c = counts[i]
                rate = crash_c / exec_c * 100 if exec_c > 0 else 0
                lines.append(f"  iter {m:>5d}: {crash_c:>5d} crashes ({rate:.1f}%)")
                shown.add(m)
                break

    # Final
    if execs:
        last_exec = execs[-1]
        last_crash = counts[-1]
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
    if not f._use_elo or not f._elo:
        return ""
    ranking = f._elo.get_ranking()
    if not ranking:
        return ""

    # BayesianEloTracker has no k_factor/decay -- it has beta (rating scale)
    # and tau (per-match system noise), and its learning rate is the adaptive
    # _effective_k(). Printing beta=200 under the label "K-factor" invited the
    # reading that a single match could move a rating by 100 points.
    unrated = f._elo.get_unrated()
    lines = [
        "",
        "--- Elo Operator Ratings ---",
    ]
    if hasattr(f._elo, "k_factor"):
        lines.append(f"  K-factor:        {f._elo.k_factor}")
        lines.append(f"  Decay:           {f._elo.decay}")
    else:
        eff_k = f._elo._effective_k() if hasattr(f._elo, "_effective_k") else None
        lines.append("  Model:           Bayesian (Gaussian posteriors)")
        lines.append(f"  Beta (scale):    {getattr(f._elo, 'beta', '?')}")
        lines.append(f"  Tau (noise):     {getattr(f._elo, 'tau', '?')}")
        if eff_k is not None:
            lines.append(f"  Effective K:     {eff_k:.1f} (adaptive)")
    lines += [
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

    # Crash-specific Elo if available (EloTracker only; BayesianEloTracker
    # does not maintain separate crash posteriors)
    if hasattr(f._elo, "crash_track") and f._elo.crash_track:
        crash_ranking = f._elo.get_ranking(crash=True)
        if crash_ranking and crash_ranking[0][1] != getattr(
            f._elo, "initial_mu", getattr(f._elo, "default_rating", 1500)
        ):
            lines.append("")
            lines.append("  Crash-specific Elo:")
            elo_mu = getattr(f._elo, "initial_mu", getattr(f._elo, "default_rating", 1500))
            for i, (op, rating) in enumerate(crash_ranking[:5], 1):
                delta = rating - elo_mu
                sign = "+" if delta >= 0 else ""
                lines.append(f"    {i}. {op:<20s} {rating:>7.0f} ({sign}{delta:.0f})")

    # Meta-scheduler strategy ranking — operator (plain keys) and seed
    # (seed_* keys) strategies are disjoint keyspaces; show them separately
    # so they don't look like one group.
    if f._use_elo and f._elo:
        strategy_ranking = f._elo.get_strategy_ranking()
        if strategy_ranking:
            elo_mu = getattr(f._elo, "initial_mu", getattr(f._elo, "default_rating", 1500))
            op_strategies = [p for p in strategy_ranking if not p[0].startswith("seed_")]
            seed_strategies = [p for p in strategy_ranking if p[0].startswith("seed_")]

            def _strategy_block(title, group):
                # Deltas are measured against the pool mean, not the constant
                # initial_mu. The Bayesian update scales each side by its own
                # sigma^2/(sigma^2+beta^2), so a match moves the two players by
                # different amounts and the pool is not zero-sum: on the
                # ffmpeg_read_nosan run both strategy pools had drifted about
                # -65 points in aggregate, which made every strategy look like
                # a loser against a fixed 1500 baseline. Relative standing
                # within the pool is the meaningful quantity.
                pool_mean = sum(r for _, r in group) / len(group)
                width = max(len(name) for name, _ in group)
                lines.append("")
                lines.append(f"  {title}")
                lines.append(f"    (pool mean {pool_mean:.0f}, initial {elo_mu:.0f})")
                for name, rating in group:
                    delta = rating - pool_mean
                    sign = "+" if delta >= 0 else ""
                    matches = f._elo._strategy_match_count.get(name, 0)
                    lines.append(
                        f"    {name:<{width}s}  {rating:>7.0f} "
                        f"({sign}{delta:.0f} vs pool, {matches} matches)"
                    )

            if op_strategies:
                _strategy_block("Meta-scheduler operator strategies (Elo):", op_strategies)
            if seed_strategies:
                _strategy_block("Seed strategies (Elo):", seed_strategies)

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
                # Re-rank inside the intersection. The raw indices come from
                # two differently-sized lists (Elo ranks only rated operators,
                # the bandit ranks every arm), so subtracting them could report
                # a max difference of 110 across 101 rated operators -- larger
                # than the 100 that is arithmetically possible.
                elo_common = sorted(common, key=lambda op: elo_rank[op])
                bandit_common = sorted(common, key=lambda op: bandit_rank[op])
                er = {op: i for i, op in enumerate(elo_common)}
                br = {op: i for i, op in enumerate(bandit_common)}
                rank_diffs = [abs(er[op] - br[op]) for op in common]
                avg_diff = sum(rank_diffs) / len(rank_diffs)
                max_diff = max(rank_diffs)
                lines.append("")
                lines.append(
                    f"  Elo vs Bandit:    avg rank diff={avg_diff:.1f}, "
                    f"max={max_diff} (over {len(common)} operators ranked by both)"
                )

    return "\n".join(lines)


def _preview(data, width: int = 40) -> str:
    """Printable, single-line preview of seed bytes.

    Seed identifiers are raw input bytes. decode(errors="replace") emits
    control characters and embedded newlines straight into the report, which
    broke the Seed Contribution table across a dozen lines and corrupted the
    file for any downstream parser. Escape to printable ASCII and hard-truncate.
    """
    if isinstance(data, str):
        data = data.encode("utf-8", errors="replace")
    if not isinstance(data, bytes | bytearray):
        data = str(data).encode("utf-8", errors="replace")
    out = []
    for b in data[:width]:
        if 0x20 <= b < 0x7F and b != 0x5C:
            out.append(chr(b))
        elif b == 0x5C:
            out.append("\\\\")
        else:
            out.append(f"\\x{b:02x}")
        if len(out) >= width:
            break
    text = "".join(out)[:width]
    return text + ("..." if len(data) > width else "")


def _corpus_byte_entropy(corpus, cap: int = 4096) -> float | None:
    """Shannon entropy of the corpus byte distribution, in bits/byte.

    Single implementation shared by Corpus Health and Entropy & Diversity.
    These were two separate loops with different per-seed caps (one sliced
    seed[:4096], the other consumed whole seeds), so the same metric printed
    two values in the same report -- 6.48 and 6.49 bits.
    """
    if not corpus:
        return None
    byte_freq = [0] * 256
    total = 0
    try:
        if _HAS_NUMPY:
            for seed in corpus:
                chunk = np.frombuffer(bytes(seed[:cap]), dtype=np.uint8)
                if chunk.size == 0:
                    continue
                counts = np.bincount(chunk, minlength=256)
                for i, c in enumerate(counts):
                    byte_freq[i] += int(c)
                total += int(chunk.size)
        else:
            for seed in corpus:
                for b in seed[:cap]:
                    byte_freq[b] += 1
                    total += 1
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    ent = 0.0
    for count in byte_freq:
        if count > 0:
            pr = count / total
            ent -= pr * math.log2(pr)
    return ent


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    else:
        return f"{n / 1024 / 1024:.1f}MB"
