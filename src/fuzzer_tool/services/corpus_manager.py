"""Corpus persistence, state management, and minimization.

Extracted from Fuzzer class (~lines 648-783, 1845-2231). Contains:
- load_corpus() — load corpus from disk
- init_seed_metadata() — initialize per-seed tracking
- seed_key() — hash a seed for tracking
- save_state() — persist fuzzer state for resume
- load_state() — restore fuzzer state
- save_crash() — save a crash with metadata
- save_to_corpus() — add a new seed to corpus
- trim_new_coverage() — minimize inputs hitting new edges
- edges_subset_of() — check edge coverage containment
- auto_minimize_corpus() — hash dedup + subsumption pruning
- deprioritize_near_duplicates() — merge near-identical seeds
"""

import contextlib
import hashlib
try:
    import xxhash
    _use_xxhash = True
except ImportError:
    _use_xxhash = False
import json
import logging
import shutil
import time
from pathlib import Path

from fuzzer_tool.adapters.filesystem import load_corpus, save_crash, save_to_corpus

log = logging.getLogger(__name__)


class CorpusManager:
    """Manages corpus persistence, state, and minimization.

    Holds a reference to the Fuzzer instance for accessing shared state.
    """

    def __init__(self, fuzzer):
        self.f = fuzzer

    def load_corpus(self):
        f = self.f
        f.corpus, f.seen_hashes = load_corpus(f.corpus_dir, f.bloom)

    def init_seed_metadata(self):
        f = self.f
        f._state_path = f.corpus_dir / "state.json"
        f._edge_tracker_path = f.corpus_dir / "edge_tracker.json"
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
            }
        from fuzzer_tool.core.edge_tracker import EdgeTracker

        f._edge_tracker = EdgeTracker(map_size=f.map_size)
        f._corpus_size_history: list[int] = []

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
            "corpus_size_history": f._corpus_size_history[-500:],
            "seed_meta": {},
            "crash_frames": f.crash_frames,
        }
        for seed, meta in f.seed_meta.items():
            key = seed.hex()
            # Skip corrupted/bloated keys (tracker JSON loaded as seed)
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
            }
        try:
            f._state_path.write_text(json.dumps(state, separators=(",", ":")))
        except OSError as e:
            log.debug("Failed to save state: %s", e)
        f._edge_tracker.save(str(f._edge_tracker_path))
        if f._use_elo and f._elo:
            f._elo.save(str(f._elo_path))
        sens_path = f.corpus_dir / "sensitivity.json"
        with contextlib.suppress(OSError):
            sens_path.write_text(json.dumps(f._sensitivity.save(), separators=(",", ":")))
        with contextlib.suppress(OSError):
            f._crash_mi_path.write_text(json.dumps(f._crash_mi.save(), separators=(",", ":")))

    def load_state(self):
        f = self.f
        if not f._state_path.exists():
            return
        try:
            state = json.loads(f._state_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            log.debug("Failed to load state: %s", e)
            return
        f.exec_count = state.get("exec_count", 0)
        f.crash_count = state.get("crash_count", 0)
        f.timeout_count = state.get("timeout_count", 0)
        f.crash_sigs = state.get("crash_sigs", {})
        f.crash_frames = state.get("crash_frames", {})
        f.op_counts = state.get("op_counts", {})
        f.op_success = state.get("op_success", {})
        f._corpus_size_history = state.get("corpus_size_history", [])
        saved_meta = state.get("seed_meta", {})
        # Skip corrupted entries: seed keys should be hex hashes (< 256 chars),
        # not full JSON blobs from tracker files loaded as corpus seeds.
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
                    }
                )
                rm_ser = sm.get("redqueen_matches", [])
                if rm_ser:
                    f.seed_meta[seed]["redqueen_matches"] = [
                        (m[0], bytes.fromhex(m[1]), bytes.fromhex(m[2])) for m in rm_ser
                    ]
        f._edge_tracker.load(str(f._edge_tracker_path))
        sens_path = f.corpus_dir / "sensitivity.json"
        if sens_path.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                f._sensitivity.load(json.loads(sens_path.read_text()))
        if f.resume:
            print(
                f"[*] Resumed: {f.exec_count} execs, "
                f"{f.crash_count} crashes, {len(f.corpus)} seeds"
            )
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
            label, sim, diffs = find_nearest_corpus(data, f.corpus)
            meta.nearest_corpus_file = label
            meta.nearest_similarity = sim
            meta.diff_bytes = diffs

        if f.ptrace_cov and hasattr(f, "_last_regs"):
            meta.rip = f._last_regs.get("rip", 0)
            meta.rsp = f._last_regs.get("rsp", 0)
            meta.rbp = f._last_regs.get("rbp", 0)

        from fuzzer_tool.core.sanitizer import SanitizerReport

        report = SanitizerReport.parse(stderr)
        if report and report.is_valid():
            sig = report.signature
            if sig not in f.crash_frames:
                f.crash_frames[sig] = report.frames

        return save_crash(
            data,
            returncode,
            stderr,
            f.crashes_dir,
            f.crash_hashes,
            f.crash_sigs,
            metadata=meta,
        )

    def save_to_corpus(self, data: bytes, parent: bytes | None = None):
        f = self.f
        parent_depth = 0
        if parent is not None:
            parent_meta = f.seed_meta.get(parent)
            if parent_meta is not None:
                parent_depth = parent_meta.get("lineage_depth", 0)

        f._total_corpus_attempts += 1
        if save_to_corpus(
            data,
            f.corpus_dir,
            f.seen_hashes,
            f.bloom,
            parent=parent,
            lineage_depth=parent_depth,
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
                "coverage_edges": 0,
                "momentum": 0.0,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
                "added_at": time.time(),
                "lineage_depth": parent_depth + 1 if parent else 0,
                "hamming_distance": f._last_hamming_distance,
            }
            f.markov.train(data)
            f.markov_trained = f.markov.is_trained()
            if f.markov.snapshot_and_check_plateau():
                log.info(
                    "Markov plateau detected (JS=%.4f) — reducing generation rate",
                    f.markov.last_js_divergence,
                )
            f._corpus_size_history.append(len(data))
            if len(f._corpus_size_history) > 1000:
                f._corpus_size_history = f._corpus_size_history[-500:]
            if f._corpus_secretary:
                dr = f._stats.discovery_rate()
                f._corpus_secretary.observe(dr)
                stop, _reason = f._corpus_secretary.should_stop()
                if stop:
                    log.info("Corpus secretary stopping: %s", _reason)
                    self.auto_minimize_corpus()
            if f.max_corpus > 0 and len(f.corpus) > f.max_corpus:
                self.auto_minimize_corpus()
            if len(f._corpus_size_history) >= 100:
                sorted_sizes = sorted(f._corpus_size_history)
                p90 = sorted_sizes[-len(sorted_sizes) // 10]
                f.max_len = max(f.max_len, min(p90 * 2, 65536))
        else:
            f._duplicate_reject_count += 1

    def trim_new_coverage(self, data: bytes, parent: bytes) -> None:
        f = self.f
        if len(data) <= 16:
            return

        if f.shm_cov:
            current_edges = f.shm_cov.read_bitmap()
        elif f.ptrace_cov:
            current_edges = bytes(f.ptrace_cov.edge_map)
        else:
            return

        trimmed = data[: len(data) // 2]
        rc, _ = f._runner.run_target(trimmed)
        if rc in (-2, -1):
            return

        if f.shm_cov:
            trimmed_edges = f.shm_cov.read_bitmap()
        elif f.ptrace_cov:
            trimmed_edges = bytes(f.ptrace_cov.edge_map)
        else:
            return

        if not self._edges_subset_of(trimmed_edges, current_edges):
            return

        seed_key = self.seed_key(data)
        if data in f.seed_meta:
            f.seed_meta.pop(data, None)
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
                "lineage_depth": f.seed_meta.get(data, {}).get("lineage_depth", 0) + 1,
            }
            log.debug("Trimmed %d -> %d bytes", len(data), len(trimmed))

    @staticmethod
    def _edges_subset_of(candidate: bytes, reference: bytes) -> bool:
        for i in range(min(len(candidate), len(reference))):
            if reference[i] and not candidate[i]:
                return False
        return True

    def auto_minimize_corpus(self):
        f = self.f
        if f.ga:
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

        stale_count = 0
        for seed in unique:
            meta = f.seed_meta.get(seed)
            if meta and meta["fuzz_count"] >= 50 and meta["coverage_edges"] == 0:
                stale_count += 1
        stale_ratio = stale_count / max(len(unique), 1)

        if f.max_corpus > 0:
            target_size = f.max_corpus
        else:
            edges = 0
            if f.shm_cov:
                edges = f.shm_cov.cumulative_edges
            elif f.ptrace_cov:
                edges = f.ptrace_cov.cumulative_edges
            target_size = max(edges, 50)

        if stale_ratio > 0.3:
            if len(unique) > target_size:
                target_size = max(target_size, int(len(unique) * (1.0 - stale_ratio)))
            else:
                target_size = int(len(unique) * (1.0 - stale_ratio))

        # Greedy set-cover: find minimum seeds that cover all discovered edges.
        et = f._edge_tracker
        all_edges = et.cumulative_edges if et and et.cumulative_edges else set()
        if all_edges and et.seed_edges:
            covered: set[int] = set()
            minimal = 0
            seed_edge_map: dict[int, set[int]] = {}
            for seed in unique:
                sk = self.seed_key(seed)
                s_edges = et.seed_edges.get(sk, set())
                if s_edges:
                    seed_edge_map[id(seed)] = s_edges
            while covered != all_edges:
                best_seed = None
                best_gain = 0
                for seed in unique:
                    sid = id(seed)
                    if sid not in seed_edge_map:
                        continue
                    gain = len(seed_edge_map[sid] - covered)
                    if gain > best_gain:
                        best_gain = gain
                        best_seed = seed
                if best_seed is None:
                    break
                covered |= seed_edge_map[id(best_seed)]
                minimal += 1
            target_size = max(target_size, minimal)
        else:
            productive = sum(
                1 for seed in unique
                if f.seed_meta.get(seed, {}).get("coverage_edges", 0) > 0
            )
            if productive > 0:
                target_size = max(target_size, productive)

        if len(unique) > target_size:
            scored = []
            for seed in unique:
                seed_key = self.seed_key(seed)
                f._edge_tracker.get_seed_edge_count(seed_key)
                meta = f.seed_meta.get(seed)
                fuzz = meta["fuzz_count"] if meta else 0
                discovered = meta["coverage_edges"] if meta else 0

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
            keep = min(target_size, len(scored))
            unique = [s for _, s in scored[:keep]]

        removed = len(f.corpus) - len(unique)
        if removed > 0:
            seeds_dir = f.corpus_dir / "seeds"
            pruned_dir = seeds_dir / "pruned"
            pruned_dir.mkdir(parents=True, exist_ok=True)
            from fuzzer_tool.adapters.filesystem import hash_data as _hash

            kept_set = {_hash(s) for s in unique}
            for fh in seeds_dir.iterdir():
                if not fh.is_file():
                    continue
                if fh.suffix == ".json" and fh.name.startswith("delta_"):
                    h = fh.name[6:-5]
                elif fh.name.startswith("id_"):
                    h = fh.name[3:]
                else:
                    continue
                if h not in kept_set:
                    shutil.move(str(fh), str(pruned_dir / fh.name))

            f.corpus = unique
            new_meta = {}
            for seed in unique:
                if seed in f.seed_meta:
                    new_meta[seed] = f.seed_meta[seed]
            f.seed_meta = new_meta
            f._weight_cache = None
            f._cached_weights = {}
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
            f._weight_cache = None
            f._cached_weights = {}
            log.info(
                "Deprioritized %d near-duplicate seeds (Hamming <= 0.05 on edge bitmaps)",
                len(to_remove),
            )
