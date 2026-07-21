"""Statistics, reporting, and coverage logging.

Extracted from Fuzzer class (~lines 3101-3613, 3435-3508). Contains:
- _print_run_summary() — session-level summary at exit
- _dump_stats() — JSON stats file writer
- _dump_coverage_report() — coverage report writer
- _append_coverage_log() — CSV coverage log appender
- _format_elapsed() — human-readable elapsed time
- print_stats() — live stats line
- _record_discovery_snapshot() — edge discovery recording
- _run_calibration() — bootstrap coverage before main loop
- discovery_rate() — edges per 1k execs
- _run_crash_replays() — crash reproducibility checking
- _update_te_causal_map() — transfer entropy updates
- _get_te_weighted_position() — TE-based mutation position
- _get_current_edge_bitmap() — read current coverage bitmap
"""

import json
import logging
import os
import random
import time

from fuzzer_tool.services.stats_reporter import (
    discovery_rate as _discovery_rate,
)
from fuzzer_tool.services.stats_reporter import (
    format_elapsed as _format_elapsed_fn,
)
from fuzzer_tool.services.stats_reporter import (
    record_discovery_snapshot as _record_discovery_snapshot_fn,
)
from fuzzer_tool.services.stats_reporter import (
    run_crash_replays as _run_crash_replays_fn,
)
from fuzzer_tool.services.te_position import (
    get_te_weighted_position,
    update_te_causal_map,
)

log = logging.getLogger(__name__)


class StatsReporter:
    """Manages statistics collection and reporting.

    Holds a reference to the Fuzzer instance for accessing shared state.
    """

    def __init__(self, fuzzer):
        self.f = fuzzer

    def record_discovery_snapshot(self):
        f = self.f
        _record_discovery_snapshot_fn(
            f.exec_count,
            f.shm_cov,
            f.ptrace_cov,
            f._discovery_history,
        )

    def run_calibration(self, max_execs: int = 1000) -> None:
        from fuzzer_tool.core.mutations import byte_insert

        f = self.f
        print(f"[*] Calibration: running {max_execs} execs to bootstrap coverage stats...")
        if not f.corpus:
            print("[*] Calibration: no seeds found, skipping")
            return

        exec_count = 0
        report_interval = max(100, max_execs // 10)
        seeds = list(f.corpus)

        for seed in seeds:
            if exec_count >= max_execs:
                break
            f._runner.run_target(seed)
            f.exec_count += 1
            exec_count += 1
            edge_bitmap = self.get_current_edge_bitmap()
            if edge_bitmap:
                f._edge_tracker.record_edges(f._seed_key(seed), edge_bitmap)

        while exec_count < max_execs:
            seed = random.choice(seeds)
            if random.random() < 0.5:
                mutated = bytearray(seed)
                if mutated:
                    mutated[random.randint(0, len(mutated) - 1)] ^= 1 << random.randint(0, 7)
                mutated = bytes(mutated)
            else:
                mutated = byte_insert(seed)
            f._runner.run_target(mutated)
            f.exec_count += 1
            exec_count += 1
            edge_bitmap = self.get_current_edge_bitmap()
            if edge_bitmap:
                f._edge_tracker.record_edges(f._seed_key(mutated), edge_bitmap)
            if exec_count % report_interval == 0:
                edges = len(f._edge_tracker._global_edge_hits)
                print(
                    f"\r[*] Calibration: {exec_count}/{max_execs} execs, {edges} edges discovered",
                    end="",
                    flush=True,
                )

        self.record_discovery_snapshot()
        edges = len(f._edge_tracker._global_edge_hits)
        gt = f._edge_tracker.good_turing_estimate()
        dr = self.discovery_rate()
        print(
            f"\r[*] Calibration done: {exec_count} execs, {edges} edges discovered, "
            f"GT confidence={gt['confidence']}, discovery_rate={dr:.1f}/1k execs   "
        )

        if f.corpus and not f._frameshift.relations:
            seed0 = f.corpus[0]

            def _exec_fn(data: bytes) -> int:
                f._runner.run_target(data)
                bm = self.get_current_edge_bitmap()
                return sum(bm) if bm else 0

            n_rels = f._frameshift.discover_relations(
                seed0, _exec_fn, max_relations=8, max_execs=200
            )
            if n_rels > 0:
                print(f"[*] FrameShift: discovered {n_rels} length-field relations")

        from fuzzer_tool.core.crash_eta import estimate_execs_to_first_crash

        eta = estimate_execs_to_first_crash(f._profile, gt, dr, exec_count, f._crash_mi)
        print(
            f"[*] ETA to first crash: ~{eta.edges_to_crash:,} risky edges, "
            f"~{eta.point_est:,} execs "
            f"(range: {eta.low:,} - {eta.high:,}, confidence: {eta.confidence})"
        )

    def discovery_rate(self) -> float:
        return _discovery_rate(self.f._discovery_history)

    def run_crash_replays(self, budget_ms: float = 200):
        f = self.f
        _run_crash_replays_fn(
            f.crashes_dir,
            f.target,
            f.timeout,
            f._crash_replays,
            f.replay_n,
            f._seed_key,
            budget_ms,
        )

    def print_run_summary(self):
        f = self.f
        elapsed = time.time() - f.start_time
        eps = f.exec_count / elapsed if elapsed > 0 else 0

        print(f"\n{'=' * 60}")
        print("  RUN SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Duration:          {elapsed:.0f}s")
        print(f"  Executions:        {f.exec_count:,}")
        print(f"  Avg eps:           {eps:.1f}")
        print(f"  Peak eps:          {f._peak_eps:.1f}")

        added = f._total_corpus_attempts
        rejected = f._duplicate_reject_count
        print(f"  Corpus:            {len(f.corpus)} entries")
        print(f"  Seeds added:       {added}")
        print(f"  Duplicates rejected: {rejected}")
        if f._pruned_count > 0:
            print(f"  Seeds pruned:      {f._pruned_count}")

        if f._stall_recovery_count > 0:
            print(f"  Recovery entries:  {f._stall_recovery_count}")
            print(
                f"  Recovery execs:    {f._stall_recovery_execs:,} "
                f"({f._stall_recovery_execs / max(1, f.exec_count) * 100:.1f}%)"
            )

        shm_edges = f.shm_cov.cumulative_edges if f.shm_cov else 0
        et_edges = f._edge_tracker.get_cumulative_edge_count()
        edges = shm_edges if shm_edges else et_edges
        if f.ptrace_cov:
            edges = f.ptrace_cov.cumulative_edges
        density = f._edge_tracker.bitmap_density() * 100
        collision_risk = f._edge_tracker.birthday_collision_risk() * 100
        if shm_edges and et_edges and shm_edges != et_edges:
            print(f"  Edges discovered:  {shm_edges} (SHM unique positions)")
            print(f"  ET positions:      {et_edges} (includes stale positions after resize)")
        else:
            print(f"  Edges discovered:  {edges}")
        print(f"  Map density:       {density:.2f}%")
        print(f"  Collision risk:    {collision_risk:.2f}% (birthday paradox)")
        rec = f._edge_tracker.recommended_map_size()
        if rec:
            print(f"  Recommended map:   {rec:,} bytes (current: {f.map_size:,})")

        gt = f._edge_tracker.good_turing_estimate()
        if gt["n"] > 0:
            print(f"  Est. remaining:    {gt['estimated_undiscovered']} edges")
            print(f"  Saturation:        {gt['saturation']:.1%} ({gt['confidence']} confidence)")

        # Coverage growth model
        growth = f._edge_tracker.coverage_growth_model()
        if growth["confidence"] > 0.1:
            print(f"  Growth rate:       {growth['current_rate']:.4f} edges/exec")
            print(f"  Projected total:   {growth['projected_total']} edges")
            if growth["time_to_plateau"] > 0:
                print(f"  Plateau in:        ~{growth['time_to_plateau']:,} execs")

        if f.seed_meta:
            depths = [m.get("lineage_depth", 0) for m in f.seed_meta.values()]
            if depths:
                print(f"  Max lineage depth: {max(depths)}")
                avg_depth = sum(depths) / len(depths)
                print(f"  Avg lineage depth: {avg_depth:.1f}")

        if f.seed_meta:
            edges_per_seed = [m.get("coverage_edges", 0) for m in f.seed_meta.values()]
            productive = sum(1 for e in edges_per_seed if e > 0)
            stale = sum(
                1
                for m in f.seed_meta.values()
                if m.get("fuzz_count", 0) >= 50 and m.get("coverage_edges", 0) == 0
            )
            total_seeds = len(f.seed_meta)
            print(f"  Productive seeds:  {productive}/{total_seeds} discovered edges")
            print(f"  Stale seeds:       {stale}/{total_seeds} (50+ fuzzes, 0 edges)")

            # Seed classification
            classifications = f._edge_tracker.classify_seeds()
            keystone = sum(1 for c in classifications.values() if c["classification"] == "keystone")
            parasitic = sum(
                1 for c in classifications.values() if c["classification"] == "parasitic"
            )
            if keystone > 0 or parasitic > 0:
                print(f"  Keystone seeds:    {keystone} (cover unique edges)")
                print(f"  Parasitic seeds:   {parasitic} (fully subsumed)")

            # Redundant seeds (dominance tree)
            redundant = f._edge_tracker.find_redundant_seeds()
            if redundant:
                print(f"  Dominated seeds:   {len(redundant)} (removable)")

        rarity = f._edge_tracker.edge_rarity_stats()
        if rarity["total"] > 0:
            print(
                f"  Edge rarity:       {rarity['singleton']} singleton / "
                f"{rarity['cold']} cold / {rarity['warm']} warm / {rarity['hot']} hot"
            )
            print(f"  Avg seeds/edge:    {rarity['avg_seeds_per_edge']:.1f}")

            uniqueness = f._edge_tracker.seed_uniqueness()
            if uniqueness:
                irreplaceable = sum(1 for v in uniqueness.values() if v > 0)
                print(f"  Irreplaceable:     {irreplaceable} seeds cover singleton edges")

            cooccur = f._edge_tracker.edge_cooccurrence(top_k=3)
            if cooccur:
                pairs_str = ", ".join(f"e{a}↔e{b}({j:.0%})" for a, b, j in cooccur)
                print(f"  Edge co-occurrence:{pairs_str}")

    def dump_stats(self):
        f = self.f
        if not f.stats_file:
            return
        elapsed = time.time() - f.start_time
        eps = f.exec_count / elapsed if elapsed > 0 else 0
        stats = {
            "timestamp": time.time(),
            "exec_count": f.exec_count,
            "crash_count": f.crash_count,
            "timeout_count": f.timeout_count,
            "corpus_size": len(f.corpus),
            "unique_crash_sigs": len(f.crash_sigs),
            "eps": round(eps, 1),
            "elapsed_sec": round(elapsed, 1),
            "peak_rss_kb": f._peak_rss,
            "op_counts": dict(f.op_counts),
            "op_success": dict(f.op_success),
        }
        if f.mc and f.mc_bandit:
            stats["bandit_stats"] = {
                k: {"successes": v[0], "failures": v[1]} for k, v in f.mc.bandit_stats().items()
            }
        if f.mc and f.mc_cem:
            stats["cem_elite_size"] = len(f.mc.elite_set)
            stats["cem_fitted"] = f.mc.cem_fitted
        if f._use_replicator and f._replicator:
            stats["replicator"] = {
                "distribution": f._replicator.population_distribution(),
                "converged": f._replicator.is_converged(),
                "dominant": f._replicator.dominant_operator(),
            }
        if f._use_shapley and f._shapley:
            sv = f._shapley.shapley_values()
            stats["shapley"] = {k: round(v, 4) for k, v in sv.items()}
        if f._use_mi and f._mi:
            stats["mi"] = {
                "observations": f._mi.total_observations,
                "top_positions": [
                    {"pos": p, "mi_bits": round(v, 4)}
                    for p, v in f._mi.top_positions(k=5, input_length=f.max_len)
                ],
            }
        if f._use_renyi_weight:
            edge_hits = (
                dict(f._edge_tracker._global_edge_hits)
                if hasattr(f._edge_tracker, "_global_edge_hits")
                else {}
            )
            if edge_hits:
                from fuzzer_tool.core.renyi import RenyiEntropy

                renyi = RenyiEntropy()
                stats["renyi"] = {
                    "uniformity": round(renyi.coverage_uniformity(list(edge_hits.values())), 4),
                    "min_entropy": round(renyi.min_entropy(list(edge_hits.values())), 4),
                    "spectrum": {
                        k: round(v, 4)
                        for k, v in renyi.entropy_spectrum(list(edge_hits.values())).items()
                    },
                }
        if f._use_transfer_entropy:
            stats["transfer_entropy"] = {
                "history_len": len(f._te_input_history),
                "causal_positions": len(f._te_byte_edges),
            }
        try:
            f.stats_file.parent.mkdir(parents=True, exist_ok=True)
            f.stats_file.write_text(json.dumps(stats, indent=2))
        except OSError:
            log.debug("Failed to write stats to %s", f.stats_file, exc_info=True)

    def dump_coverage_report(self):
        f = self.f
        if not f.coverage_report:
            return
        edge_map = None
        if f.shm_cov:
            edge_map = f.shm_cov._seen
        elif f.ptrace_cov:
            edge_map = f.ptrace_cov.edge_map
        if edge_map is None:
            print("[!] No coverage data available for report")
            return

        hit_edges = []
        cumulative = 0
        for i, val in enumerate(edge_map):
            if val:
                hit_edges.append(i)
                cumulative += 1

        report = {
            "map_size": len(edge_map),
            "cumulative_edges": cumulative,
            "hit_edges": hit_edges,
            "coverage_pct": round(cumulative / len(edge_map) * 100, 4),
            "exec_count": f.exec_count,
            "corpus_size": len(f.corpus),
        }
        f.coverage_report.parent.mkdir(parents=True, exist_ok=True)
        f.coverage_report.write_text(json.dumps(report, indent=2))
        print(
            f"\n[*] Coverage report: {f.coverage_report} "
            f"({cumulative}/{len(edge_map)} edges, {report['coverage_pct']}%)"
        )

    def append_coverage_log(self):
        f = self.f
        if not f.coverage_log:
            return
        cumulative = 0
        if f.shm_cov:
            cumulative = f.shm_cov.cumulative_edges
        elif f.ptrace_cov:
            cumulative = f.ptrace_cov.cumulative_edges
        elif hasattr(f, "_edge_tracker"):
            cumulative = f._edge_tracker.get_cumulative_edge_count()
        elapsed = time.time() - f.start_time
        line = f"{elapsed:.1f},{f.exec_count},{cumulative},{len(f.corpus)},{f.crash_count}\n"
        with open(f.coverage_log, "a") as fh:
            fh.write(line)

    def update_te_causal_map(self):
        f = self.f
        update_te_causal_map(
            f._te,
            f._te_input_history,
            f._te_edge_history,
            f.map_size,
            f._te_byte_edges,
        )

    def get_te_weighted_position(self, input_length: int) -> int | None:
        return get_te_weighted_position(self.f._te_byte_edges, input_length)

    def get_current_edge_bitmap(self) -> bytes | None:
        f = self.f
        if f.multi_targets:
            active_shm = f._target_shm_covs.get(f.target)
            if active_shm:
                return bytes(active_shm._map)
        if f.shm_cov:
            return bytes(f.shm_cov._map)
        if f.ptrace_cov:
            return bytes(f.ptrace_cov.edge_map)
        return None

    def format_elapsed(self) -> str:
        return _format_elapsed_fn(self.f.start_time)

    def print_stats(self):
        f = self.f
        elapsed = time.time() - f.start_time
        eps = f.exec_count / elapsed if elapsed > 0 else 0
        f._eps = eps
        dict_str = f" | dict: {len(f.dictionary)}" if f.dictionary else ""
        markov_str = " | markov: trained" if f.markov_trained else ""
        cmplog_str = ""
        if f._cmplog is not None:
            n_tok = len(f._cmplog.tokens)
            n_prs = len(f._cmplog.pairs)
            cmplog_str = f" | cmplog: {n_tok}t {n_prs}p"
        if f.markov_generate:
            markov_str += "+gen"
        cov_str = ""
        if f.multi_targets and f._target_shm_covs:
            parts = []
            for t in f._target_shm_covs:
                shm = f._target_shm_covs[t]
                name = os.path.basename(t)
                parts.append(f"{name}:{shm.cumulative_edges}")
            cov_str = " | targets: " + " ".join(parts)
        elif f.shm_cov:
            shm_edges = f.shm_cov.cumulative_edges
            gt = f._edge_tracker.good_turing_estimate()
            max_edges = gt["n"] + gt["estimated_undiscovered"]
            sat = gt["saturation"] * 100 if max_edges > 0 else 0
            cov_str = f" | shm: {shm_edges} max: {max_edges} sat: {sat:.0f}%"
        elif f.ptrace_cov:
            cov_str = (
                f" | edges: {f.ptrace_cov.cumulative_edges} hits: {f.ptrace_cov.total_bp_hits}"
            )
            if f.ptrace_cov.deep_coverage:
                cov_str += f" bps:{len(f.ptrace_cov.original_bytes)}"
        mc_str = ""
        if f.mc:
            parts = []
            if f.mc_bandit:
                parts.append("bandit")
            if f.mc_cem:
                parts.append(f"cem:{len(f.mc.elite_set)}")
            if parts:
                mc_str = " | mc: " + "+".join(parts)
        sig_str = f"({len(f.crash_sigs)}sigs)" if f.crash_sigs else ""
        timeout_pct = f.timeout_count / f.exec_count * 100 if f.exec_count else 0
        timeout_str = f" | timeouts: {f.timeout_count} ({timeout_pct:.1f}%)"
        rss_kb = f._peak_rss
        rss_str = f" | rss: {rss_kb // 1024}MB" if rss_kb >= 1024 else f" | rss: {rss_kb}KB"
        ops_str = ""
        if f._last_ops_used:
            recent = list(dict.fromkeys(reversed(f._last_ops_used)))[:3]
            ops_str = " | ops: " + " ".join(recent)
        div_str = ""
        if len(f._edge_tracker.seed_hit_counts) >= 2:
            diversity = f._edge_tracker.compute_corpus_diversity()
            div_str = f" | div: {diversity:.0f}"
        jac_str = ""
        if len(f._edge_tracker.seed_hit_counts) >= 2:
            avg_jac = f._edge_tracker.compute_average_jaccard()
            jac_str = f" | jac: {avg_jac:.2f}"
        dr = self.discovery_rate()
        dr_str = f" | rate: {dr:.1f} ed/kexec" if f.exec_count > 100 else ""
        if f.exec_count > 100:
            f._csd.observe(dr)
            detected, csd_reason = f._csd.is_approaching_transition()
            if detected:
                dr_str += f" [CSD: {csd_reason}]"
        density = f._edge_tracker.bitmap_density() * 100
        collision_risk = f._edge_tracker.birthday_collision_risk() * 100
        density_str = f" | map: {density:.1f}%"
        if collision_risk > 10:
            density_str += f" (collision: {collision_risk:.0f}%)"
            if collision_risk > f._max_collision_risk and f.shm_cov:
                current = f.shm_cov.size
                new_size = min(1048576, current * 2)
                if new_size > current:
                    print(
                        f"\n[*] Collision risk {collision_risk:.0f}% — resizing bitmap "
                        f"{current:,} → {new_size:,} bytes"
                    )
                    f.shm_cov.resize(new_size)
                    f.map_size = new_size
                    f._edge_tracker.map_size = new_size
                    f._edge_tracker.reset_after_resize()
                    # Update env vars BEFORE patching target so __afl_map_shm()
                    # reads the correct __AFL_SHM_ID and AFL_MAP_SIZE.
                    os.environ["__AFL_SHM_ID"] = f.shm_cov.env_id
                    os.environ["AFL_MAP_SIZE"] = str(new_size)
                    # Update inprocess runner's SHM pointers — the target's
                    # __afl_area still points to the old (detached) SHM
                    if f._inprocess_runner:
                        f._inprocess_runner.update_shm_after_resize(
                            f.shm_cov._ptr, new_size, f.shm_cov.env_id
                        )
                    density_str = f" | map: {f._edge_tracker.bitmap_density() * 100:.1f}% (collision: {collision_risk:.0f}%)"
        repro_str = ""
        if f._crash_replays:
            done = [v for v in f._crash_replays.values() if len(v) >= f.replay_n]
            if done:
                avg_repro = (
                    sum(sum(1 for r in replays if r >= 0) / len(replays) for replays in done)
                    / len(done)
                    * 100
                )
                repro_str = f" | repro: {avg_repro:.0f}%"
        brier_str = ""
        if f.mc and f.mc_bandit and f.mc.brier_score() > 0:
            brier_str = f" | brier: {f.mc.brier_score():.3f}"
        crps_str = ""
        if f._exec_time_tracker.count > 20:
            crps_str = f" | crps: {f._exec_time_tracker.mean_crps():.4f}"
        # Shannon entropy and Simpson's diversity of edge hit distribution
        ent_str = ""
        simp_str = ""
        if f._edge_tracker._global_edge_hits:
            sh = f._edge_tracker.shannon_entropy_global()
            simp = f._edge_tracker.simpson_diversity_global()
            ent_str = f" | ent: {sh:.2f}"
            simp_str = f" | simp: {simp:.2f}"
        # Entropy rate of change
        rate_str = ""
        if hasattr(f, "_entropy_history") and len(f._entropy_history) >= 2:
            recent = f._entropy_history[-10:]
            if len(recent) >= 2:
                dt = recent[-1][0] - recent[0][0]
                if dt > 0:
                    dS = recent[-1][1] - recent[0][1]
                    entropy_rate = dS / dt
                    rate_str = f" | dS/dt: {entropy_rate:+.4f}"
        # Format learner summary
        fmt_str = ""
        if getattr(f, "_format_learner", None) and f._format_learner.hypotheses:
            fl = f._format_learner
            classified = sum(1 for h in fl.hypotheses if h.field_type != "unknown")
            fmt_str = f" | fmt: {classified}/{len(fl.hypotheses)} fields v{fl.format_model_version}"
        line = (
            f"[*] execs: {f.exec_count} | corpus: {len(f.corpus)} | "
            f"crashes: {f.crash_count}{sig_str}{timeout_str} | eps: {eps:.0f} | "
            f"time: {elapsed:.0f}s{rss_str}{ops_str}{dict_str}{markov_str}{cmplog_str}{cov_str}{mc_str}{div_str}{jac_str}{dr_str}{density_str}{repro_str}{brier_str}{crps_str}{ent_str}{simp_str}{rate_str}{fmt_str}"
        )
        # Add coverage growth model to stats line
        growth = f._edge_tracker.coverage_growth_model()
        if growth["confidence"] > 0.1:
            growth_str = f" | gr: {growth['current_rate']:.3f}e/x proj: {growth['projected_total']} plateau: ~{growth['time_to_plateau']:,}"
            line += growth_str
        print(line, flush=True)
