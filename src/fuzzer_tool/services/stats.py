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

    def _print_summary_coverage(self, f) -> None:
        """Print coverage-related summary lines."""
        shm_edges = f.shm_cov._peak_cumulative_edges if f.shm_cov else 0
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

        growth = f._edge_tracker.coverage_growth_model()
        if growth["confidence"] > 0.1:
            print(f"  Growth rate:       {growth['current_rate']:.4f} edges/exec")
            print(f"  Projected total:   {growth['projected_total']} edges")
            if growth["time_to_plateau"] > 0:
                print(f"  Plateau in:        ~{growth['time_to_plateau']:,} execs")

        # Bayesian coverage model (richer output when available)
        bayes = f._edge_tracker.bayesian_coverage_growth_model()
        if "p_stalled" in bayes and bayes["p_stalled"] is not None:
            print(f"  Bayesian — P(stalled): {bayes['p_stalled']:.1%} "
                  f"P(growth): {1 - bayes['p_stalled']:.1%}"
                  f" {'[STALLED]' if bayes['p_stalled'] > 0.5 else ''}")

    def _print_summary_seeds(self, f) -> None:
        """Print seed-related summary lines."""
        if not f.seed_meta:
            return
        depths = [m.get("lineage_depth", 0) for m in f.seed_meta.values()]
        if depths:
            print(f"  Max lineage depth: {max(depths)}")
            print(f"  Avg lineage depth: {sum(depths) / len(depths):.1f}")

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

        self._print_summary_classification(f)

    def _print_summary_classification(self, f) -> None:
        """Print seed classification and redundancy."""
        classifications = f._edge_tracker.classify_seeds()
        keystone = sum(1 for c in classifications.values() if c["classification"] == "keystone")
        parasitic = sum(1 for c in classifications.values() if c["classification"] == "parasitic")
        if keystone > 0 or parasitic > 0:
            print(f"  Keystone seeds:    {keystone} (cover unique edges)")
            print(f"  Parasitic seeds:   {parasitic} (fully subsumed)")
        redundant = f._edge_tracker.find_redundant_seeds()
        if redundant:
            print(f"  Dominated seeds:   {len(redundant)} (removable)")

    def _print_summary_rarity(self, f) -> None:
        """Print edge rarity summary lines."""
        rarity = f._edge_tracker.edge_rarity_stats()
        if rarity["total"] <= 0:
            return
        print(
            f"  Edge rarity:       {rarity['singleton']} singleton / {rarity['cold']} cold / {rarity['warm']} warm / {rarity['hot']} hot"
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
        print(f"  Corpus:            {len(f.corpus)} entries")
        print(f"  Seeds added:       {f._total_corpus_attempts}")
        print(f"  Duplicates rejected: {f._duplicate_reject_count}")
        if f._pruned_count > 0:
            print(f"  Seeds pruned:      {f._pruned_count}")
        if f._stall_recovery_count > 0:
            print(f"  Recovery entries:  {f._stall_recovery_count}")
            print(
                f"  Recovery execs:    {f._stall_recovery_execs:,} ({f._stall_recovery_execs / max(1, f.exec_count) * 100:.1f}%)"
            )
        self._print_summary_coverage(f)
        self._print_summary_seeds(f)
        self._print_summary_rarity(f)

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

    def _print_stats_cov_str(self, f) -> str:
        """Format coverage string."""
        if f.multi_targets and f._target_shm_covs:
            parts = [
                f"{os.path.basename(t)}:{shm.cumulative_edges}"
                for t, shm in f._target_shm_covs.items()
            ]
            return " | targets: " + " ".join(parts)
        if f.shm_cov:
            shm_edges = f.shm_cov.cumulative_edges
            gt = f._edge_tracker.good_turing_estimate()
            max_edges = gt["n"] + gt["estimated_undiscovered"]
            sat = gt["saturation"] * 100 if max_edges > 0 else 0
            return f" | shm: {shm_edges} max: {max_edges} sat: {sat:.0f}%"
        if f.ptrace_cov:
            s = f" | edges: {f.ptrace_cov.cumulative_edges} hits: {f.ptrace_cov.total_bp_hits}"
            if f.ptrace_cov.deep_coverage:
                s += f" bps:{len(f.ptrace_cov.original_bytes)}"
            return s
        return ""

    def _print_stats_density_str(self, f) -> str:
        """Format map density string with optional collision-induced resize."""
        density = f._edge_tracker.bitmap_density() * 100
        collision_risk = f._edge_tracker.birthday_collision_risk() * 100
        s = f" | map: {density:.1f}%"
        if collision_risk > 10:
            s += f" (collision: {collision_risk:.0f}%)"
            if collision_risk > f._max_collision_risk and f.shm_cov:
                s += self._maybe_resize_bitmap(f, collision_risk)
        return s

    def _maybe_resize_bitmap(self, f, collision_risk: float) -> str:
        """Resize bitmap when collision risk exceeds threshold."""
        current = f.shm_cov.size
        new_size = min(1048576, current * 2)
        if new_size <= current:
            return ""
        print(
            f"\n[*] Collision risk {collision_risk:.0f}% — resizing bitmap {current:,} → {new_size:,} bytes"
        )
        f.shm_cov.resize(new_size)
        f.map_size = new_size
        f._edge_tracker.map_size = new_size
        f._edge_tracker.reset_after_resize()
        os.environ["__AFL_SHM_ID"] = f.shm_cov.env_id
        os.environ["AFL_MAP_SIZE"] = str(new_size)
        if f._inprocess_runner:
            f._inprocess_runner.update_shm_after_resize(f.shm_cov._ptr, new_size, f.shm_cov.env_id)
        return f" | map: {f._edge_tracker.bitmap_density() * 100:.1f}% (collision: {collision_risk:.0f}%)"

    def _print_stats_smt_str(self, f) -> str:
        """Format SMT solver string."""
        if f._smt_solver is None or f._smt_solver.queries_attempted <= 0:
            return ""
        s = f._smt_solver
        inc_pct = s.batch_solved / max(s.batch_attempted, 1) * 100
        tot_pct = s.queries_solved / max(s.queries_attempted, 1) * 100
        cache_str = f" ch:{s.cache_hits}" if s.cache_hits else ""
        return f" | smt: {s.batch_solved}/{s.batch_attempted} ({inc_pct:.0f}%) tot: {s.queries_solved}/{s.queries_attempted} ({tot_pct:.0f}%){cache_str}"

    def _print_stats_dr_str(self, f) -> str:
        """Format discovery rate string with CSD detection."""
        dr = self.discovery_rate()
        s = f" | rate: {dr:.1f} ed/kexec" if f.exec_count > 100 else ""
        if f.exec_count > 100:
            f._csd.observe(dr)
            detected, csd_reason = f._csd.is_approaching_transition()
            if detected:
                s += f" [CSD: {csd_reason}]"
        return s

    def print_stats(self):
        f = self.f
        elapsed = time.time() - f.start_time
        eps = f.exec_count / elapsed if elapsed > 0 else 0
        f._eps = eps

        dict_str = f" | dict: {len(f.dictionary)}" if f.dictionary else ""
        markov_str = " | markov: trained" if f.markov_trained else ""
        markov_str += "+gen" if f.markov_generate else ""

        cmplog_str = (
            f" | cmplog: {len(f._cmplog.tokens)}t {len(f._cmplog.pairs)}p"
            if f._cmplog is not None
            else ""
        )

        smt_str = self._print_stats_smt_str(f)

        cov_str = self._print_stats_cov_str(f)
        mc_str = ""
        if f.mc:
            parts = [
                p
                for p, cond in [("bandit", f.mc_bandit), (f"cem:{len(f.mc.elite_set)}", f.mc_cem)]
                if cond
            ]
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

        div_str = (
            f" | div: {f._edge_tracker.compute_corpus_diversity():.0f}"
            if len(f._edge_tracker.seed_hit_counts) >= 2
            else ""
        )
        jac_str = (
            f" | jac: {f._edge_tracker.compute_average_jaccard():.2f}"
            if len(f._edge_tracker.seed_hit_counts) >= 2
            else ""
        )

        dr_str = self._print_stats_dr_str(f)

        density_str = self._print_stats_density_str(f)

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

        brier_str = (
            f" | brier: {f.mc.brier_score():.3f}"
            if f.mc and f.mc_bandit and f.mc.brier_score() > 0
            else ""
        )
        crps_str = (
            f" | crps: {f._exec_time_tracker.mean_crps():.4f}"
            if f._exec_time_tracker.count > 20
            else ""
        )

        ent_str = simp_str = ""
        if f._edge_tracker._global_edge_hits:
            ent_str = f" | ent: {f._edge_tracker.shannon_entropy_global():.2f}"
            simp_str = f" | simp: {f._edge_tracker.simpson_diversity_global():.2f}"

        rate_str = ""
        if hasattr(f, "_entropy_history") and len(f._entropy_history) >= 2:
            recent = f._entropy_history[-10:]
            if len(recent) >= 2:
                dt = recent[-1][0] - recent[0][0]
                if dt > 0:
                    dS = recent[-1][1] - recent[0][1]
                    rate_str = f" | dS/dt: {dS / dt:+.4f}"

        fmt_str = ""
        fl = getattr(f, "_format_learner", None)
        if fl and fl.hypotheses:
            classified = sum(1 for h in fl.hypotheses if h.field_type != "unknown")
            fmt_str = f" | fmt: {classified}/{len(fl.hypotheses)} fields v{fl.format_model_version}"

        line = (
            f"[*] execs: {f.exec_count} | corpus: {len(f.corpus)} | "
            f"crashes: {f.crash_count}{sig_str}{timeout_str} | eps: {eps:.0f} | "
            f"time: {elapsed:.0f}s{rss_str}{ops_str}{dict_str}{markov_str}{cmplog_str}{smt_str}{cov_str}{mc_str}{div_str}{jac_str}{dr_str}{density_str}{repro_str}{brier_str}{crps_str}{ent_str}{simp_str}{rate_str}{fmt_str}"
        )
        growth = f._edge_tracker.coverage_growth_model()
        if growth["confidence"] > 0.1:
            line += f" | gr: {growth['current_rate']:.3f}e/x proj: {growth['projected_total']} plateau: ~{growth['time_to_plateau']:,}"
        # Bayesian stall probability when available
        bayes = f._edge_tracker.bayesian_coverage_growth_model()
        if bayes.get("p_stalled") is not None and bayes["p_stalled"] > 0.3:
            line += f" | P(stall): {bayes['p_stalled']:.0%}"
        print(line, flush=True)
