"""Corpus persistence, state management, and minimization.

Extracted from Fuzzer class (~lines 648-783, 1845-2231). Contains:
- load_corpus() — load corpus from disk
- init_seed_metadata() — initialize per-seed tracking
- seed_key() — hash a seed for tracking
- save_state() — persist fuzzer state for resume
- load_state() — restore fuzzer state
- save_crash() — save a crash with metadata
- save_to_corpus() — add a new seed to corpus

Signal name mapping for crash return codes.
"""

import hashlib
import logging
import os
import shutil
import time
from array import array
from pathlib import Path

import xxhash

from fuzzer_tool.adapters.filesystem import (
    load_corpus,
    save_crash,
    save_irreplaceable,
    save_to_corpus,
)
from fuzzer_tool.core.periodicity import estimate_record_size
from fuzzer_tool.core.running_stats import RunningMoments
from fuzzer_tool.services.operators import HAVOC_SUB_OPS

log = logging.getLogger(__name__)


def _gdb_crash_replay(f, data: bytes, returncode: int) -> str:
    """Best-effort GDB crash replay text for the report sidecar; '' when unavailable.

    Runs the crashing input once under GDB (only when gdb is installed and the
    target is traceable) so the final crash report carries a real backtrace,
    registers, and fault address. Cost is ~1s on the rare crashing input.
    """
    from fuzzer_tool.core.trace import CrashTracer

    try:
        tracer = CrashTracer(f.target, timeout=max(5, int(getattr(f, "timeout", 5))))
        if not tracer._has_gdb:
            return ""
        return tracer.gdb_replay(data, returncode).sidecar_block()
    except Exception:
        log.debug("GDB crash replay failed", exc_info=True)
        return ""


_use_xxhash = True

SIGNAL_NAMES = {
    6: "SIGABRT",
    7: "SIGBUS",
    8: "SIGFPE",
    11: "SIGSEGV",
    13: "SIGPIPE",
    14: "SIGALRM",
    15: "SIGTERM",
}


def _returncode_to_signal(returncode: int) -> tuple[str | None, int | None]:
    """Map a subprocess return code to (signal_name, signal_number).

    Returns (None, None) for non-signal exit codes.
    Handles both WIFSIGNALED-style codes (-signum) and
    exit-with-signal codes (128+signum).
    """
    if returncode < 0:
        signum = -returncode
        if signum in SIGNAL_NAMES:
            return SIGNAL_NAMES[signum], signum
    elif returncode >= 128:
        signum = returncode - 128
        if signum in SIGNAL_NAMES:
            return SIGNAL_NAMES[signum], signum
    return None, None


class CorpusManager:
    """Manages corpus persistence, state, and minimization.

    Holds a reference to the Fuzzer instance for accessing shared state.

    - trim_new_coverage() — minimize inputs hitting new edges
    - edges_subset_of() — check edge coverage containment
    - auto_minimize_corpus() — hash dedup + subsumption pruning
    - deprioritize_near_duplicates() — merge near-identical seeds
    """

    def __init__(self, fuzzer):
        self.f = fuzzer

    def load_corpus(self):
        f = self.f
        f.corpus, f.seen_hashes, f.irreplaceable_hashes = load_corpus(f.corpus_dir, f.bloom)
        # Ensure the irreplaceable/ directory exists inside seeds/ so seeds can be
        # promoted to irreplaceable without a late mkdir.
        (f.corpus_dir / "seeds" / "irreplaceable").mkdir(parents=True, exist_ok=True)

    def init_seed_metadata(self):
        f = self.f
        now = time.time()
        f.seed_meta: dict[bytes, dict] = {}
        for seed in f.corpus:
            f.seed_meta[seed] = {
                "fuzz_count": 0,
                "coverage_edges": 0,
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": now,
                "record_stride": estimate_record_size(seed),
                "seed_passed_det": False,
            }
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        morris_mode = os.environ.get("AFL_MORRIS", "1") != "0"
        f._edge_tracker = EdgeTracker(map_size=f.map_size, morris_mode=morris_mode)
        f._corpus_size_history: array = array("I")
        f._seed_size_moments = RunningMoments(window=200)

        if f.resume:
            self.load_state()

    def seed_key(self, data: bytes) -> str:
        if _use_xxhash:
            return xxhash.xxh64(data).hexdigest()[:16]
        return hashlib.sha256(data).hexdigest()[:16]

    def save_state(self):
        f = self.f
        state = {
            "exec_count": f.exec_count,
            "crash_count": f.crash_count,
            "timeout_count": f.timeout_count,
            "crash_sigs": f.crash_sigs,
            "op_counts": f.op_counts,
            "op_success": f.op_success,
            # getattr, unlike the counters above it: save_state() is called
            # with partially-built stand-ins in several tests, and a new
            # required attribute here turns "this counter is new" into
            # "state cannot be saved at all".
            "op_applicable": getattr(f, "op_applicable", {}),
            "op_success_applicable": getattr(f, "op_success_applicable", {}),
            "op_edges": f.op_edges,
            # Havoc sub-mutation credit. Kept as plain lists (the state file
            # is a sanitized pickle, but array("d") would still pin the
            # branch count into the on-disk format); load_state re-pairs
            # them with HAVOC_SUB_OPS by name so adding a branch does not
            # silently shift every count by one slot.
            "havoc_subop_stats": {
                name: (f._operators._havoc_hits[i], f._operators._havoc_trials[i])
                for i, name in enumerate(HAVOC_SUB_OPS)
            },
            "corpus_size_history": list(f._corpus_size_history[-500:]),
            "checksum_learner": getattr(f, "checksum_learner", None)
            and f.checksum_learner.to_dict()
            or None,
            "seed_meta": {},
            "crash_frames": f.crash_frames,
            "crash_min_sizes": f.crash_min_sizes,
        }
        for seed, meta in f.seed_meta.items():
            key = seed.hex()
            if len(key) >= 256:
                continue
            rm = meta.get("redqueen_matches", [])
            rm_ser = [[m[0], m[1].hex(), m[2].hex()] for m in rm]
            state["seed_meta"][key] = {
                "fuzz_count": meta["fuzz_count"],
                "coverage_edges": meta["coverage_edges"],
                "momentum": meta.get("momentum", 0.0),
                "redqueen_offsets": meta["redqueen_offsets"],
                "redqueen_matches": rm_ser,
                "added_at": meta["added_at"],
                "lineage_depth": meta.get("lineage_depth", 0),
                "hamming_distance": meta.get("hamming_distance", -1),
                "child_count": meta.get("child_count", 0),
                "timed_out": meta.get("timed_out", False),
                "parent_key": meta.get("parent_key"),
                "parent_ops": meta.get("parent_ops", []),
                "parent_sites": meta.get("parent_sites", []),
                "new_edge_count": meta.get("new_edge_count", 0),
                "coverage_edges_baseline": meta.get("coverage_edges_baseline", 0),
                "record_stride": meta.get("record_stride", None),
            }
        store = f._state_store
        store.set("corpus", state)
        store.set("edge_tracker", f._edge_tracker.to_dict())
        if f._use_elo and f._elo:
            store.set("elo", f._elo.to_dict())
        store.set("sensitivity", f._sensitivity.save())
        store.set("crash_mi", f._crash_mi.save())
        if hasattr(f, "_seed_quality"):
            store.set("seed_quality", f._seed_quality.state_dict())
        store.save()

    def load_state(self):
        f = self.f
        state = f._state_store.get("corpus")
        if not state:
            return
        f.exec_count = state.get("exec_count", 0)
        f._resume_baseline_exec = f.exec_count
        f._last_eps_count = f.exec_count
        f.crash_count = state.get("crash_count", 0)
        f.timeout_count = state.get("timeout_count", 0)
        f.crash_sigs = state.get("crash_sigs", {})
        f.crash_frames = state.get("crash_frames", {})
        f.crash_min_sizes = state.get("crash_min_sizes", {})
        f.op_counts = state.get("op_counts", {})
        f.op_success = state.get("op_success", {})
        # Absent in states written before applicability was tracked. Left
        # empty rather than backfilled from op_counts: the report treats
        # "no entry" as unknown and falls back to the raw count, which is
        # honest, whereas copying op_counts would assert every historic
        # selection was applicable.
        f.op_applicable = state.get("op_applicable", {})
        f.op_success_applicable = state.get("op_success_applicable", {})
        f.op_edges = state.get("op_edges", {})
        havoc_stats = state.get("havoc_subop_stats") or {}
        for i, name in enumerate(HAVOC_SUB_OPS):
            saved = havoc_stats.get(name)
            if not saved:
                continue
            hits, trials = saved
            # Trials must stay >= 1 or the ratio divides by zero; a corrupt
            # or hand-edited state file should degrade to the prior, not
            # crash the resume.
            if trials >= 1.0 and hits >= 0.0:
                f._operators._havoc_hits[i] = int(hits)
                f._operators._havoc_trials[i] = int(trials)
        f._operators._rebuild_havoc_table()
        f._corpus_size_history = array("I", state.get("corpus_size_history", []))
        saved_meta = state.get("seed_meta", {})
        for seed in f.corpus:
            key = seed.hex()
            if key in saved_meta and len(key) < 256:
                sm = saved_meta[key]
                f.seed_meta[seed].update(
                    {
                        "fuzz_count": sm.get("fuzz_count", 0),
                        "coverage_edges": sm.get("coverage_edges", 0),
                        "momentum": sm.get("momentum", 0.0),
                        "redqueen_offsets": sm.get("redqueen_offsets", []),
                        "added_at": sm.get("added_at", f.seed_meta[seed]["added_at"]),
                        "lineage_depth": sm.get("lineage_depth", 0),
                        "hamming_distance": sm.get("hamming_distance", -1),
                        "child_count": sm.get("child_count", 0),
                        "timed_out": sm.get("timed_out", False),
                        "parent_key": sm.get("parent_key"),
                        "parent_ops": sm.get("parent_ops", []),
                        "parent_sites": sm.get("parent_sites", []),
                        "new_edge_count": sm.get("new_edge_count", 0),
                        "coverage_edges_baseline": sm.get("coverage_edges_baseline", 0),
                        "record_stride": sm.get("record_stride", None),
                    }
                )
                rm_ser = sm.get("redqueen_matches", [])
                if rm_ser:
                    f.seed_meta[seed]["redqueen_matches"] = [
                        (m[0], bytes.fromhex(m[1]), bytes.fromhex(m[2])) for m in rm_ser
                    ]
        et_data = f._state_store.get("edge_tracker")
        if et_data is not None:
            f._edge_tracker.from_dict(et_data)
        # Restore checksum learner state
        cl_data = state.get("checksum_learner")
        if cl_data and hasattr(f, "checksum_learner") and f.checksum_learner is not None:
            from fuzzer_tool.core.checksum_learner import ChecksumLearner

            f.checksum_learner = ChecksumLearner.from_dict(f, cl_data)
        sens_data = f._state_store.get("sensitivity")
        if sens_data is not None:
            f._sensitivity.load(sens_data)
        if f.resume:
            print(
                f"[*] Resumed: {f.exec_count} execs, {f.crash_count} crashes, {len(f.corpus)} seeds"
            )
        sq_data = f._state_store.get("seed_quality")
        if sq_data is not None and hasattr(f, "_seed_quality"):
            f._seed_quality.load_state_dict(sq_data)
        log.info(
            "Fuzzer state loaded: execs=%d, crashes=%d, corpus=%d",
            f.exec_count,
            f.crash_count,
            len(f.corpus),
        )

    def save_crash(self, data: bytes, returncode: int, stderr: str) -> str | None:
        f = self.f
        from fuzzer_tool.adapters.filesystem import hash_data
        from fuzzer_tool.core.crash_metadata import CrashMetadata, find_nearest_corpus

        meta = CrashMetadata()
        meta.exec_count = f.exec_count
        meta.corpus_size = len(f.corpus)
        meta.target = f.target
        meta.mutation_ops = list(f._last_ops_used)
        meta.parent_sites = [s for _, s in getattr(f, "_last_ops_with_sites", [])]
        meta.elapsed = f._stats.format_elapsed()

        if f.corpus:
            parent = f._last_parent_seed if hasattr(f, "_last_parent_seed") else None
            if parent:
                meta.parent_seed_hash = hash_data(parent)

        if not hasattr(f, "_target_sha256"):
            try:
                f._target_sha256 = hashlib.sha256(Path(f.target).read_bytes()).hexdigest()[:16]
            except Exception:
                f._target_sha256 = "unknown"
        meta.target_sha256 = f._target_sha256

        if f.corpus:
            label, sim, diffs, _ = find_nearest_corpus(data, f.corpus)
            meta.nearest_corpus_file = label
            meta.nearest_similarity = sim
            meta.diff_bytes = diffs

        if hasattr(f, "_last_regs") and (f.ptrace_cov or f._last_regs):
            meta.rip = f._last_regs.get("rip", 0)
            meta.rsp = f._last_regs.get("rsp", 0)
            meta.rbp = f._last_regs.get("rbp", 0)
        fault_addr = getattr(f, "_last_fault_addr", None)
        if fault_addr is not None:
            meta.fault_addr = f"0x{fault_addr:x}"

        # Populate error_type from return code for subprocess/inprocess
        # mode where ptrace isn't available and sanitizer reports are absent.
        if not meta.error_type:
            sig_name, sig_num = _returncode_to_signal(returncode)
            if sig_name is not None:
                meta.error_type = sig_name

        from fuzzer_tool.core.sanitizer import SanitizerReport

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            sig = report.signature
            if sig not in f.crash_frames:
                f.crash_frames[sig] = report.frames
        del report  # free SanitizerReport object

        # Embed the GDB crash replay in the report sidecar (best-effort).
        meta.gdb_replay = _gdb_crash_replay(f, data, returncode)

        return save_crash(
            data,
            returncode,
            stderr,
            f.crashes_dir,
            f.crash_hashes,
            f.crash_sigs,
            metadata=meta,
            fault_addr=fault_addr,
            crash_blocklist=f.crash_blocklist if f.crash_blocklist else None,
            crash_allowlist=f.crash_allowlist if f.crash_allowlist else None,
            crash_min_sizes=f.crash_min_sizes if f.save_smaller else None,
        )

    def save_to_corpus(self, data: bytes, parent: bytes | None = None):
        f = self.f
        parent_depth = 0
        if parent is not None:
            parent_meta = f.seed_meta.get(parent)
            if parent_meta is not None:
                parent_depth = parent_meta.get("lineage_depth", 0)
                parent_meta["child_count"] = parent_meta.get("child_count", 0) + 1

        f._total_corpus_attempts += 1
        if save_to_corpus(
            data,
            f.corpus_dir,
            f.seen_hashes,
            f.bloom,
            parent=parent,
            lineage_depth=parent_depth,
        ):
            if (
                # Under Elo arbitration the corpus-based seed strategies
                # (weighted/pareto/bayesian/boltzmann) read f.corpus; if QEA's
                # bypass froze it, those strategies starve on the initial seeds
                # and the run stalls.  Keep the bypass only for standalone QEA,
                # where the QEA population is the sole seed source.
                not f.qea or getattr(f, "_use_elo", False)
            ):
                f.corpus.append(data)
            if f.ga:
                import hashlib as _hashlib

                from fuzzer_tool.core.ga import Individual

                if _use_xxhash:
                    seed_key = xxhash.xxh64(data).hexdigest()[:16]
                else:
                    seed_key = _hashlib.sha256(data).hexdigest()[:16]
                edge_count = len(f._edge_tracker.seed_edges.get(seed_key, set()))
                ind = Individual(
                    data=data,
                    edge_count=edge_count,
                    generation=f.ga.generation,
                    seed_key=seed_key,
                )
                f.ga.add_to_population(ind)
            f.seed_meta[data] = {
                "fuzz_count": 0,
                "coverage_edges": 0,  # will update below from edge tracker
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": time.time(),
                "lineage_depth": parent_depth + 1 if parent else 0,
                "hamming_distance": f._last_hamming_distance,
                "record_stride": estimate_record_size(data),
                "input_size": len(data),
            }
            # Lineage edge: parent key + the ops/sites that produced this seed.
            # Only recorded when a real parent exists (interesting/Metropolis
            # paths in fuzz_one); parallel-sync inserts are roots. Gated on
            # the flag so default runs stay byte-identical.
            if f._use_lineage and parent is not None:
                f.seed_meta[data].update(
                    {
                        "parent_key": self.seed_key(parent),
                        "parent_ops": list(getattr(f, "_last_ops_used", [])),
                        "parent_sites": [s for _, s in getattr(f, "_last_ops_with_sites", [])],
                        "new_edge_count": getattr(f, "_last_new_edge_count", 0),
                        "coverage_edges_baseline": 0,
                    }
                )
            # Propagate actual coverage_edges from EdgeTracker — when called
            # from fuzz_one, the seed's edges were already recorded by
            # record_edges before save_to_corpus.  For the parallel-sync path
            # (no prior fuzz_one), edge_count stays 0, which is correct.
            seed_key = self.seed_key(data)
            edge_count = len(f._edge_tracker.seed_edges.get(seed_key, set()))
            if edge_count > 0:
                f.seed_meta[data]["coverage_edges"] = edge_count
            f.markov.train(data)
            f.markov_trained = f.markov.is_trained()
            if f.markov.snapshot_and_check_plateau():
                log.info(
                    "Markov plateau detected (JS=%.4f) — reducing generation rate",
                    f.markov.last_js_divergence,
                )
            f._corpus_size_history.append(len(data))
            seed_moments = getattr(f, "_seed_size_moments", None)
            if seed_moments is not None:
                seed_moments.update(float(len(data)))
                # Bloat early-warning: rising right skew in seed sizes means
                # a few oversized seeds are growing relative to the median.
                # Rate-limited to once per 500 execs to avoid log spam.
                if (
                    seed_moments.count >= 50
                    and seed_moments.skewness > 2.0
                    and f.exec_count - f._last_bloat_warn_exec >= 500
                ):
                    f._last_bloat_warn_exec = f.exec_count
                    log.warning(
                        "Corpus bloat warning: seed-size skewness=%.2f "
                        "(rising right tail — minimizing)",
                        seed_moments.skewness,
                    )
                    f._defer_minimize()
            if len(f._corpus_size_history) > 1000:
                f._corpus_size_history = f._corpus_size_history[-500:]
            if f._corpus_secretary:
                dr = f._stats.discovery_rate()
                f._corpus_secretary.observe(dr)
                stop, _reason = f._corpus_secretary.should_stop()
                if stop:
                    log.info("Corpus secretary stopping: %s", _reason)
                    f._defer_minimize()
            if f.max_corpus > 0 and len(f.corpus) > f.max_corpus:
                f._defer_minimize()
            if len(f._corpus_size_history) >= 100:
                sorted_sizes = sorted(f._corpus_size_history)
                p90 = sorted_sizes[-len(sorted_sizes) // 10]
                # Track the p90 of recent seed sizes in both directions. This
                # was max(f.max_len, ...), a one-way ratchet: once a handful of
                # large seeds pushed p90 up, max_len never came back down, so
                # mutation kept producing larger seeds, which kept p90 up. That
                # is a positive feedback loop into exactly the bloat the
                # skewness warning below reports, and minimizing the corpus
                # could not undo it. The configured max_len is the floor.
                f.max_len = min(max(p90 * 2, f._max_len_floor), 65536)
        else:
            f._duplicate_reject_count += 1

    def trim_new_coverage(self, data: bytes, parent: bytes) -> None:
        f = self.f
        if len(data) <= 16:
            return

        if f.shm_cov:
            current_edges = f.shm_cov.get_edge_ids()
        elif f.ptrace_cov:
            bm = bytes(f.ptrace_cov.edge_map)
            current_edges = {i for i, v in enumerate(bm) if v}
        else:
            return

        trimmed = data[: len(data) // 2]
        rc, _ = f._runner.run_target(trimmed)
        if rc in (-2, -1):
            return

        if f.shm_cov:
            trimmed_edges = f.shm_cov.get_edge_ids()
        elif f.ptrace_cov:
            bm = bytes(f.ptrace_cov.edge_map)
            trimmed_edges = {i for i, v in enumerate(bm) if v}
        else:
            return

        if not trimmed_edges.issubset(current_edges):
            return

        seed_key = self.seed_key(data)
        orig_meta = f.seed_meta.get(data, {})
        if data in f.seed_meta:
            f.seed_meta.pop(data, None)
            f._agg_cache_valid = False  # corpus structure changed
        if data in f.corpus:
            idx = f.corpus.index(data)
            f.corpus[idx] = trimmed
            f.seed_meta[trimmed] = {
                "fuzz_count": 0,
                "coverage_edges": f._edge_tracker.get_seed_edge_count(seed_key),
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": time.time(),
                "lineage_depth": orig_meta.get("lineage_depth", 0) + 1,
            }
            # The trimmed seed inherits the original's lineage edge so the
            # crash-path chain stays intact across the trim point, with a
            # synthetic ("trim", cut_point) operation appended.
            if f._use_lineage:
                f.seed_meta[trimmed].update(
                    {
                        "parent_key": orig_meta.get("parent_key"),
                        "parent_ops": list(orig_meta.get("parent_ops", [])) + ["trim"],
                        "parent_sites": list(orig_meta.get("parent_sites", [])) + [len(data) // 2],
                        "new_edge_count": orig_meta.get("new_edge_count", 0),
                        "coverage_edges_baseline": orig_meta.get("coverage_edges_baseline", 0),
                    }
                )
            log.debug("Trimmed %d -> %d bytes", len(data), len(trimmed))

    def auto_minimize_corpus(self):
        f = self.f
        if f.ga or f.qea:
            return
        if not f.corpus:
            return

        from fuzzer_tool.adapters.filesystem import hash_data

        seen: set[str] = set()
        unique: list[bytes] = []
        for seed in f.corpus:
            h = hash_data(seed)
            if h not in seen:
                seen.add(h)
                unique.append(seed)
        del seen  # free intermediate seed-hash set

        # Irreplaceable seeds (loaded from corpus/seeds/irreplaceable/) are never pruned.
        # Separate them from the unique pool before minimization; re-add after.
        irreplaceable_seeds: list[bytes] = []
        if f.irreplaceable_hashes:
            for seed in unique[:]:  # iterate copy, mutate original
                if hash_data(seed) in f.irreplaceable_hashes:
                    irreplaceable_seeds.append(seed)
                    unique.remove(seed)

        # Fresh seeds (fuzz_count == 0) have never been picked by the seed picker.
        # Exclude them from minimization — we don't know their value yet.
        fresh_seeds: list[bytes] = []
        for seed in unique[:]:  # iterate copy, mutate original
            meta = f.seed_meta.get(seed)
            if meta and meta["fuzz_count"] == 0:
                fresh_seeds.append(seed)
                unique.remove(seed)

        stale_count = 0
        for seed in unique:
            meta = f.seed_meta.get(seed)
            if meta and meta["fuzz_count"] >= 50 and meta["coverage_edges"] == 0:
                stale_count += 1
        stale_ratio = stale_count / max(len(unique), 1)

        # Bayesian stale probability: P(stale | fuzz_count, 0_discoveries)
        # Uses Beta(1 + 0, 1 + fuzz_count) — the posterior probability that
        # a seed with `fuzz_count` attempts and 0 discoveries has discovery
        # probability below 0.01.
        #
        # Extreme-value asymptotics (order_statistics.py Part 4):
        # n * min(U1..Un) → Exp(1) as n → ∞. If a seed's discovery probability
        # is the minimum of n independent tries, P(discovery < ε) ≈ 1 - exp(-n*ε).
        # For n = fuzz_count and ε = 0.01, this gives a simpler approximation:
        #   P(stale) ≈ 1 - exp(-fuzz_count * 0.01)
        # which matches the Beta CDF asymptotically and avoids the Beta integral.
        bayesian_stale_ratio = stale_ratio
        if f._use_bayesian and f._seed_quality:
            bayesian_stale_count = 0
            for seed in unique:
                sk = self.seed_key(seed)
                meta = f.seed_meta.get(seed)
                if not meta:
                    continue
                fuzz = meta.get("fuzz_count", 0)
                if fuzz < 5:
                    continue
                # P(discovery_prob < 0.01 | 0 discoveries in fuzz_count attempts)
                # = Beta.cdf(0.01, alpha=1, beta=1+fuzz_count)
                a, b = 1.0, 1.0 + fuzz
                # Mean of Beta = a/(a+b). If the posterior mean is below 0.01,
                # the seed is likely stale.
                if a / (a + b) < 0.01:
                    bayesian_stale_count += 1
            bayesian_stale_ratio = bayesian_stale_count / max(len(unique), 1)
            # Use whichever stale ratio is higher (more conservative)
            stale_ratio = max(stale_ratio, bayesian_stale_ratio)

        if f.max_corpus > 0:
            target_size = f.max_corpus
        else:
            edges = 0
            if f.shm_cov:
                edges = f.shm_cov.cumulative_edges
            elif f.ptrace_cov:
                edges = f.ptrace_cov.cumulative_edges
            target_size = min(max(edges, 50), 5000)

        if stale_ratio > 0.3:
            if len(unique) > target_size:
                target_size = max(target_size, int(len(unique) * (1.0 - stale_ratio)))
            else:
                target_size = int(len(unique) * (1.0 - stale_ratio))

        # Greedy set-cover is O(n²) against the full seed list, so for large
        # corpora we first reduce the search space to one cheap candidate per
        # edge.  That bounds the inner loop by edge count rather than seed
        # count, while still preserving the actual minimal-cover result.
        et = f._edge_tracker
        all_edges = et.cumulative_edges if et and et.cumulative_edges else set()
        mandatory: set[int] = set()
        if all_edges and et.seed_edges:
            covered: set[int] = set()
            seed_edge_map: dict[int, set[int]] = {}
            for seed in unique:
                sk = self.seed_key(seed)
                s_edges = et.seed_edges.get(sk, set())
                if s_edges:
                    seed_edge_map[id(seed)] = s_edges
            if seed_edge_map:
                # Terminate against what these seeds can actually cover, not
                # against cumulative_edges. EdgeTracker._prune_tracked_seeds drops
                # entries from seed_edges once past max_tracked_seeds (200) but
                # never removes their edges from cumulative_edges, so on any run
                # past 200 seeds all_edges is a strict superset of anything the
                # loop can reach. `covered != all_edges` was therefore permanently
                # true: the loop never converged, always ran to best_gain == 0, and
                # selected every seed holding a unique edge — making `mandatory`,
                # and the target_size floor derived from it, meaningless.
                coverable = set().union(*seed_edge_map.values()) if seed_edge_map else set()
                candidate_ids: set[int] = set(seed_edge_map.keys())
                if len(candidate_ids) > 5000:
                    edge_to_seeds: dict[int, list[tuple[float, int]]] = {}
                    for seed in unique:
                        sid = id(seed)
                        if sid not in seed_edge_map:
                            continue
                        meta = f.seed_meta.get(seed, {})
                        exec_us = max(
                            1.0,
                            meta.get("total_time", 0.0)
                            / max(1, meta.get("fuzz_count", 1))
                            * 1_000_000,
                        )
                        input_size = max(1, meta.get("input_size", 1))
                        cost = exec_us * input_size
                        for edge in seed_edge_map[sid]:
                            edge_to_seeds.setdefault(edge, []).append((cost, sid))
                    candidate_ids = {min(candidates)[1] for candidates in edge_to_seeds.values()}
                while covered != coverable:
                    best_seed = None
                    best_gain = 0
                    for sid in candidate_ids:
                        if sid not in seed_edge_map:
                            continue
                        gain = len(seed_edge_map[sid] - covered)
                        if gain > best_gain:
                            best_gain = gain
                            best_seed = sid
                    if best_seed is None:
                        break
                    covered |= seed_edge_map[best_seed]
                    mandatory.add(best_seed)
                target_size = max(target_size, len(mandatory))
                del seed_edge_map  # free edge map after set-cover
        elif all_edges or f.corpus:
            productive = sum(
                1 for seed in unique if f.seed_meta.get(seed, {}).get("coverage_edges", 0) > 0
            )
            if productive > 0:
                target_size = max(target_size, productive)

        if len(unique) > target_size or (
            f.max_corpus_bytes > 0 and sum(len(s) for s in unique) > f.max_corpus_bytes
        ):
            # Split into mandatory (set-cover essential) and optional.
            mandatory_seeds = [s for s in unique if id(s) in mandatory]
            optional = [s for s in unique if id(s) not in mandatory]
            scored = []
            for seed in optional:
                seed_key = self.seed_key(seed)
                meta = f.seed_meta.get(seed)
                fuzz = meta["fuzz_count"] if meta else 0
                discovered = meta["coverage_edges"] if meta else 0

                # Bayesian seed score: use the posterior mean from
                # BayesianSeedQuality when available.
                if f._use_bayesian and f._seed_quality:
                    seed_key_in_bq = seed_key in f._seed_quality._alpha
                    if seed_key_in_bq:
                        mean = f._seed_quality.posterior_mean(seed_key)
                        # Scale posterior mean to a useful range for scoring:
                        # posterior mean ~ [0,1]. Multiply by discovered * 10
                        # to get a score on a comparable scale to the heuristic.
                        edge_score = mean * 10.0 + (discovered * 5.0 if discovered > 0 else 0.0)
                    else:
                        edge_score = discovered * 10
                        if fuzz > 0 and discovered == 0:
                            edge_score *= max(0.01, 1.0 / (1.0 + fuzz * 0.01))
                        else:
                            edge_score += 1.0 / max(fuzz, 1)
                else:
                    edge_score = discovered * 10
                    if fuzz > 0 and discovered == 0:
                        edge_score *= max(0.01, 1.0 / (1.0 + fuzz * 0.01))
                    else:
                        edge_score += 1.0 / max(fuzz, 1)

                wasserstein_weight = f._edge_tracker.compute_wasserstein_weight(seed_key)

                # PPMD novelty: incompressible seeds are more diverse
                ppmd_bonus = 1.0
                if getattr(f, "_ppmd", None) and f._ppmd.enabled:
                    ppmd_bonus = 1.0 + f._ppmd.compute_seed_novelty(seed) * 0.5

                score = edge_score * wasserstein_weight * ppmd_bonus
                scored.append((score, seed))
            scored.sort(key=lambda x: x[0], reverse=True)
            if f.max_corpus_bytes > 0:
                # Knapsack: sort optional seeds by value/weight density
                # (value = coverage score, weight = seed byte size).
                # Greedy density-ordering is a well-known 2-approximation
                # for 0/1 knapsack.
                scored.sort(
                    key=lambda x: x[0] / max(len(x[1]), 1),
                    reverse=True,
                )
                selected = []
                total_bytes = sum(len(s) for s in mandatory_seeds)
                for _score, seed in scored:
                    seed_bytes = len(seed)
                    if total_bytes + seed_bytes <= f.max_corpus_bytes:
                        selected.append(seed)
                        total_bytes += seed_bytes
                unique = mandatory_seeds + selected
            else:
                # Count-budget: keep top-K by score (original behavior)
                budget = target_size - len(mandatory_seeds)
                keep = min(budget, len(scored))
                unique = mandatory_seeds + [s for _, s in scored[:keep]]
            del scored  # free scored list after sorting

        # Save set-cover mandatory seeds to irreplaceable/ so they survive
        # future pruning cycles. Remove the original from seeds/ to avoid
        # duplicates on disk.
        if mandatory and f.corpus_dir:
            for seed in unique:
                if id(seed) in mandatory:
                    h = hash_data(seed)
                    if h not in f.irreplaceable_hashes:
                        save_irreplaceable(
                            seed,
                            f.corpus_dir,
                            f.seen_hashes,
                            f.irreplaceable_hashes,
                            f.bloom,
                        )
                        # Remove the original from seeds/ to avoid duplicate
                        seeds_sub = f.corpus_dir / "seeds" / h[:2] / f"id_{h}"
                        if seeds_sub.exists():
                            seeds_sub.unlink()

        # Re-add fresh seeds that were set aside before minimization.
        # They haven't been fuzzed yet and need a chance to prove their value.
        if fresh_seeds:
            unique = fresh_seeds + unique

        # Re-add irreplaceable seeds that were set aside before minimization.
        # They are never pruned.
        if irreplaceable_seeds:
            unique = unique + irreplaceable_seeds

        # Lineage branch pruning: a dropped seed whose subtree contributed
        # < 1.0 structural edge-weight and gained no coverage since the last
        # minimize is an unproductive branch — drop the whole subtree instead
        # of just the low-scoring seed. Mandatory/fresh/irreplaceable seeds
        # are protected (they were explicitly kept above).
        if f._use_lineage and getattr(f, "_lineage", None) is not None:
            key_to_seed = {self.seed_key(s): s for s in f.corpus}
            kept_keys = {self.seed_key(s) for s in unique}
            protected = {id(s) for s in fresh_seeds + irreplaceable_seeds}
            if mandatory:
                protected |= {id(s) for s in unique if id(s) in mandatory}

            def _coverage_fn(k: str) -> tuple[int, int]:
                seed = key_to_seed.get(k)
                if seed is None:
                    return (0, 0)
                meta = f.seed_meta.get(seed, {})
                return (
                    meta.get("coverage_edges", 0),
                    meta.get("coverage_edges_baseline", 0),
                )

            subtree_drops: set[str] = set()
            for seed in f.corpus:
                sk = self.seed_key(seed)
                if sk in kept_keys or sk in subtree_drops:
                    continue
                if (
                    f._lineage.recent_credit(sk, _coverage_fn) == 0.0
                    and f._lineage.subtree_weight(sk) < 1.0
                ):
                    for k in f._lineage.subtree_keys(sk):
                        s = key_to_seed.get(k)
                        if s is not None and id(s) not in protected and s in unique:
                            unique.remove(s)
                        subtree_drops.add(k)
            # Reset the credit clock: record current coverage per seed so the
            # next minimize measures the delta gained since this one.
            for seed in f.corpus:
                meta = f.seed_meta.get(seed)
                if meta is not None:
                    meta["coverage_edges_baseline"] = meta.get("coverage_edges", 0)

        # Post-pruning coverage verification: recover seeds whose unique edges
        # were dropped by scoring or lineage pruning.
        et = f._edge_tracker
        recovered_count = 0
        if et and et.cumulative_edges and unique:
            kept_coverage: set[int] = set()
            for seed in unique:
                sk = self.seed_key(seed)
                kept_coverage.update(et.seed_edges.get(sk, set()))
            uncovered = et.cumulative_edges - kept_coverage
            if uncovered:
                for seed in f.corpus:
                    if seed in unique:
                        continue
                    sk = self.seed_key(seed)
                    seed_edges = et.seed_edges.get(sk, set())
                    if seed_edges & uncovered:
                        unique.append(seed)
                        mandatory.add(id(seed))
                        if f.corpus_dir:
                            h = hash_data(seed)
                            if h not in f.irreplaceable_hashes:
                                save_irreplaceable(
                                    seed,
                                    f.corpus_dir,
                                    f.seen_hashes,
                                    f.irreplaceable_hashes,
                                    f.bloom,
                                )
                                seeds_sub = f.corpus_dir / "seeds" / h[:2] / f"id_{h}"
                                if seeds_sub.exists():
                                    seeds_sub.unlink()
                        recovered_count += 1
                if recovered_count:
                    log.warning(
                        "Recovered %d seeds to cover %d uncovered edges after minimization",
                        recovered_count,
                        len(uncovered),
                    )

        removed = len(f.corpus) - len(unique)
        if removed > 0:
            seeds_dir = f.corpus_dir / "seeds"
            deltas_dir = f.corpus_dir / "deltas"
            pruned_dir = seeds_dir / "pruned"
            pruned_dir.mkdir(parents=True, exist_ok=True)
            from fuzzer_tool.adapters.filesystem import hash_data as _hash

            kept_set = {_hash(s) for s in unique}
            # Prune full seeds — seeds are stored in two-digit hash
            # subdirectories (seeds/ab/id_abc...), so walk recursively.
            # Skip the irreplaceable/ subdirectory — those seeds are never pruned.
            for fh in seeds_dir.rglob("id_*"):
                if not fh.is_file():
                    continue
                # Skip files under seeds/irreplaceable/ (never pruned)
                if "irreplaceable" in fh.parts:
                    continue
                h = fh.name[3:]
                if h not in kept_set:
                    sub = pruned_dir / h[:2]
                    sub.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(fh), str(sub / fh.name))
            # Prune delta files
            if deltas_dir.exists():
                deltas_pruned_dir = deltas_dir / "pruned"
                for fh in deltas_dir.iterdir():
                    if not fh.is_file():
                        continue
                    if fh.suffix == ".json" and fh.name.startswith("delta_"):
                        h = fh.name[6:-5]
                    else:
                        continue
                    if h not in kept_set:
                        sub = deltas_pruned_dir / h[:2]
                        sub.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(fh), str(sub / fh.name))
            del kept_set  # free kept hashes after file pruning

            f.corpus = unique
            new_meta = {}
            for seed in unique:
                if seed in f.seed_meta:
                    new_meta[seed] = f.seed_meta[seed]
            f.seed_meta = new_meta
            f._agg_cache_valid = False
            f._weight_cache = None
            f._cached_weights = {}
            f._overlap_density_cache = {}
            f._last_minimize_exec = f.exec_count
            f._pruned_count += removed
            log.info(
                "Auto-minimized corpus: %d -> %d seeds -> pruned/ (stale_ratio=%.1f)",
                len(f.corpus) + removed,
                len(f.corpus),
                stale_ratio,
            )

    def deprioritize_near_duplicates(self):
        f = self.f
        if len(f.corpus) < 10:
            return

        near_dupes = f._edge_tracker.find_near_duplicate_seeds(max_hamming=0.05)
        if not near_dupes:
            return

        to_remove: set[bytes] = set()
        for key_a, key_b, _hdist in near_dupes:
            seed_a = None
            seed_b = None
            for s in f.corpus:
                if self.seed_key(s) == key_a:
                    seed_a = s
                elif self.seed_key(s) == key_b:
                    seed_b = s
                if seed_a and seed_b:
                    break
            if not seed_a or not seed_b:
                continue
            if seed_a in to_remove or seed_b in to_remove:
                continue

            meta_a = f.seed_meta.get(seed_a, {})
            meta_b = f.seed_meta.get(seed_b, {})
            edges_a = meta_a.get("coverage_edges", 0)
            edges_b = meta_b.get("coverage_edges", 0)

            if edges_a <= edges_b:
                to_remove.add(seed_a)
            else:
                to_remove.add(seed_b)

        if to_remove:
            f.corpus = [s for s in f.corpus if s not in to_remove]
            for s in to_remove:
                f.seed_meta.pop(s, None)
            f._agg_cache_valid = False
            f._weight_cache = None
            f._cached_weights = {}
            f._overlap_density_cache = {}
            log.info(
                "Deprioritized %d near-duplicate seeds (Hamming <= 0.05 on edge bitmaps)",
                len(to_remove),
            )
