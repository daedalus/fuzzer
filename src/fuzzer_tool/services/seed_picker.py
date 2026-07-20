"""Seed selection strategies.

Extracted from Fuzzer class (~lines 2232-2735). Contains:
- _pick_seed() — main entry point, dispatches to strategy
- _pick_markov_seed() — Markov chain generation
- _pick_pareto_only() — pure Pareto frontier selection
- _format_aware_seed() — format-specific seed generation
- _weighted_pick_seed() — weighted scoring with Pareto
- _compute_weights() — multi-signal seed scoring
- _pareto_front() — sliding-window Pareto dominance
- _pick_from_pareto_front() — frontier sampling
"""

import logging
import math
import random
import struct
import time

log = logging.getLogger(__name__)


class SeedPicker:
    """Manages seed selection strategies.

    Holds a reference to the Fuzzer instance for accessing shared state.
    """

    def __init__(self, fuzzer):
        self.f = fuzzer

    def pick_seed(self) -> bytes:
        f = self.f
        if f._stall_recovery_active and f.corpus:
            f._seed_strategy = "random_stall"
            return random.choice(f.corpus)

        if f._use_elo and f._elo:
            available = []
            if f.ga:
                available.append("ga")
            if f.qea:
                available.append("qea")
            available.append("weighted")
            if f.corpus and f.seed_meta:
                available.append("pareto")
            if f._profile.format_signature:
                available.append("format")

            if len(available) >= 2:
                strategy = f._elo.select_strategy(available)
                f._seed_strategy = strategy
            elif available:
                strategy = available[0]
                f._seed_strategy = strategy
            else:
                strategy = None

            if strategy == "ga" and f.ga:
                return f.ga.pick_seed()
            elif strategy == "qea" and f.qea:
                return f.qea.pick_seed()
            elif strategy == "pareto" and f.corpus and f.seed_meta:
                return self._pick_pareto_only()
            elif strategy == "format":
                return self._format_aware_seed()

        if f.qea:
            return f.qea.pick_seed()
        if f.ga:
            return f.ga.pick_seed()
        if f.markov_generate and f.markov_trained:
            return self._pick_markov_seed()
        if f.corpus and f.seed_meta:
            return self.weighted_pick_seed()
        if f.corpus:
            return random.choice(f.corpus)
        return self._format_aware_seed()

    def _pick_markov_seed(self) -> bytes:
        f = self.f
        from fuzzer_tool.core.edge_tracker import ks_significance_threshold

        plateau_threshold = ks_significance_threshold(max(1, f.markov._contexts_seen), alpha=0.05)
        gen_rate = 0.03 if f.markov.last_js_divergence < plateau_threshold else 0.15

        if not hasattr(self, "_last_corpus_pp"):
            self._last_corpus_pp = 256.0
        if f.exec_count % 500 == 0 and f.corpus:
            pp_stats = f.markov.corpus_perplexity(f.corpus)
            self._last_corpus_pp = pp_stats["mean"]
        if self._last_corpus_pp > 200:
            gen_rate = min(gen_rate * 2, 0.40)
        elif self._last_corpus_pp < 10:
            gen_rate = max(gen_rate * 0.3, 0.01)

        if random.random() < gen_rate:
            length = random.randint(1, min(256, f.max_len))
            for _ in range(3):
                candidate = f.markov.generate(length)
                pp = f.markov.perplexity(candidate)
                if pp < 512:
                    return candidate
            return candidate
        length = random.randint(1, min(256, f.max_len))
        return f.markov.generate(length)

    def _pick_pareto_only(self) -> bytes:
        f = self.f
        if len(f.corpus) < 3 or not f.seed_meta:
            return random.choice(f.corpus)
        now = time.time()
        weights = [1.0] * len(f.corpus)
        return self._pick_from_pareto_front(weights, now)

    def _format_aware_seed(self) -> bytes:
        f = self.f
        fmt = getattr(f._profile, "format_signature", None)
        if fmt == "png":
            import binascii

            ihdr_data = b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
            ihdr_chunk = b"IHDR" + ihdr_data
            ihdr_crc = struct.pack(">I", binascii.crc32(ihdr_chunk))
            iend_chunk = b"IEND"
            iend_crc = struct.pack(">I", binascii.crc32(iend_chunk))
            return (
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", len(ihdr_data))
                + ihdr_chunk
                + ihdr_crc
                + struct.pack(">I", 0)
                + iend_chunk
                + iend_crc
            )
        elif fmt == "jpeg":
            return (
                b"\xff\xd8"
                + b"\xff\xe0"
                + b"\x00\x10"
                + b"JFIF\x00"
                + b"\x01\x01"
                + b"\x00"
                + b"\x00\x01"
                + b"\x00\x01"
                + b"\x00\x00"
                + b"\xff\xd9"
            )
        elif fmt == "gif":
            return b"GIF89a" + struct.pack("<HH", 1, 1) + b"\xf7\x00\x00"
        elif fmt == "bmp":
            return (
                b"BM"
                + struct.pack("<I", 54)
                + b"\x00\x00\x00\x00"
                + struct.pack("<I", 54)
                + struct.pack("<I", 40)
                + struct.pack("<I", 40)
                + struct.pack("<H", 1)
                + struct.pack("<H", 24)
                + b"\x00" * 24
            )
        elif fmt == "zlib":
            import zlib

            return b"\x78\x9c" + zlib.compress(b"\x00")
        elif fmt == "gzip":
            import zlib

            return (
                b"\x1f\x8b"
                + b"\x08"
                + b"\x00"
                + b"\x00\x00\x00\x00"
                + b"\x00"
                + b"\x00"
                + zlib.compress(b"\x00")
                + struct.pack("<I", zlib.crc32(b"\x00"))
                + struct.pack("<I", 1)
            )
        # Generic: zero-filled random-length buffer
        length = random.randint(4, min(64, f.max_len))
        return bytes(random.randint(0, 255) for _ in range(length))

    def _compute_weights(self, now: float) -> list[float]:
        f = self.f
        weights = []
        pareto_scores: list[tuple[float, float, float]] = []

        # Cache classification across calls (recompute every 100 execs)
        if not hasattr(f, "_classify_cache") or f.exec_count % 100 == 0:
            f._classify_cache = f._edge_tracker.classify_seeds()
        classifications = f._classify_cache

        for seed in f.corpus:
            meta = f.seed_meta.get(seed)
            if meta is None:
                weights.append(1.0)
                pareto_scores.append((1.0, 1.0, 1.0))
                continue
            fuzz_count = max(meta["fuzz_count"], 1)
            coverage = meta["coverage_edges"]
            age = now - meta["added_at"]

            T = f._temperature

            explore_part = T * (1.0 / math.sqrt(fuzz_count))
            exploit_part = (1.0 + coverage * 0.5) / (1.0 + age * 0.01)
            w = explore_part * exploit_part

            momentum = meta.get("momentum", 0.0)
            w *= 1.0 + momentum * 2.0

            burst_factor = max(1.0, 1.0 + T * (5.0 - 1.0) - (age / 60.0) * T)

            staleness = fuzz_count / max(coverage + 1, 1)
            stale_threshold = 50.0 * T
            w *= 0.01 if staleness > stale_threshold else 1.0

            seed_key = f._seed_key(seed)

            if f._secretary and seed_key in f._seed_secretary:
                stop, _reason = f._seed_secretary[seed_key].should_stop()
                if stop:
                    w *= 0.01
            if seed_key not in f._cached_weights:
                if seed_key in f._edge_tracker.seed_edges and f._edge_tracker.seed_edges[seed_key]:
                    sub = f._edge_tracker.compute_subsumption_weight(seed_key)
                    div = f._edge_tracker.compute_hitcount_diversity_weight(seed_key)
                    spa = f._edge_tracker.compute_wasserstein_weight(seed_key)
                    cov = f._edge_tracker.compute_coverage_proximity(seed_key)
                    f._cached_weights[seed_key] = (sub, div, spa, cov)
                else:
                    f._cached_weights[seed_key] = (1.0, 1.0, 1.0, 0.5)
            sub, div, spa, cov = f._cached_weights[seed_key]
            w *= sub * div * spa
            w *= 0.5 + cov

            # Boost keystone seeds, penalize parasitic ones
            if seed_key in classifications:
                cls = classifications[seed_key]["classification"]
                if cls == "keystone":
                    w *= 2.0  # Double weight for irreplaceable seeds
                elif cls == "parasitic":
                    w *= 0.1  # Heavily penalize redundant seeds

            seed_edges = f._edge_tracker.seed_edges.get(seed_key, set())
            if seed_edges:
                rare_count = sum(
                    1 for e in seed_edges if f._edge_tracker._global_edge_hits.get(e, 0) <= 2
                )
                if rare_count > 0:
                    w *= 1.0 + rare_count * 0.5

            if seed_edges:
                total_hits = 0
                for e in seed_edges:
                    total_hits += f._edge_tracker._global_edge_hits.get(e, 0)
                mean_hits = total_hits / len(seed_edges) if seed_edges else 0
                if mean_hits > 3:
                    w *= 1.0 + (mean_hits - 3) * 0.1
                elif mean_hits < 1.5 and fuzz_count > 10:
                    w *= 0.7

            if seed_edges:
                gap_score = 0
                for e in seed_edges:
                    seed_count = f._edge_tracker._global_edge_hits.get(e, 0)
                    if seed_count <= 2:
                        gap_score += 1
                if gap_score > 0:
                    w *= 1.0 + gap_score * 0.3

            if seed_edges and hasattr(f, "_recent_seed_edges"):
                overlap = 0
                for recent in f._recent_seed_edges:
                    overlap += len(seed_edges & recent)
                if overlap > 0:
                    penalty = overlap / max(len(seed_edges), 1)
                    w *= max(0.3, 1.0 - penalty * 0.5)

            # Input-level Shannon entropy bonus: seeds whose hit-count
            # distribution has unusual entropy (relative to corpus mean)
            # are behaviorally distinct. Boost seeds with entropy far from
            # the mean — they exercise edges in an unusual pattern.
            seed_sh = f._edge_tracker.shannon_entropy_seed(seed_key)
            if seed_sh > 0 and len(f._edge_tracker.seed_hit_counts) >= 3:
                # Compute corpus mean entropy (cached)
                if not hasattr(self, "_mean_seed_entropy"):
                    self._mean_seed_entropy = 0.0
                    self._mean_entropy_cache_key = -1
                cache_key = len(f._edge_tracker.seed_hit_counts)
                if cache_key != self._mean_entropy_cache_key:
                    entropies = [
                        f._edge_tracker.shannon_entropy_seed(k)
                        for k in f._edge_tracker.seed_hit_counts
                        if f._edge_tracker.shannon_entropy_seed(k) > 0
                    ]
                    self._mean_seed_entropy = sum(entropies) / len(entropies) if entropies else 0.0
                    self._mean_entropy_cache_key = cache_key
                del entropies  # free entropy list after mean computed
                if self._mean_seed_entropy > 0:
                    # z-score like: boost if far from mean
                    deviation = abs(seed_sh - self._mean_seed_entropy) / max(
                        self._mean_seed_entropy, 0.01
                    )
                    # Boost up to 1.5x for high deviation
                    w *= 1.0 + min(deviation, 1.0) * 0.5

            if f._distance:
                seed_dist = meta.get("avg_distance", f._distance.max_distance)
                max_d = f._distance.max_distance
                norm_dist = min(seed_dist / max_d, 1.0) if max_d > 0 else 0.5
                alpha = min(f._anneal_progress * 2, 1.0)
                dist_weight = math.exp(-norm_dist * 5.0 * alpha)
                w *= (1.0 - alpha) + alpha * dist_weight

            # PPMD novelty: seeds that compress poorly are more diverse
            if getattr(f, "_ppmd", None) and f._ppmd.enabled:
                novelty = f._ppmd.compute_seed_novelty(seed)
                # Boost up to 1.5x for maximally novel seeds
                w *= 1.0 + novelty * 0.5

            if f._profile.hot_functions and f._profile.functions:
                hot_density = sum(
                    f._profile.functions[fn].branch_density
                    for fn in f._profile.hot_functions
                    if fn in f._profile.functions
                ) / max(len(f._profile.hot_functions), 1)
                all_density = sum(fi.branch_density for fi in f._profile.functions.values()) / max(
                    len(f._profile.functions), 1
                )
                if all_density > 0:
                    hotness_ratio = hot_density / all_density
                    if coverage > 0:
                        w *= 1.0 + (hotness_ratio - 1.0) * min(coverage / 50.0, 1.0)

            hd = meta.get("hamming_distance", -1)
            if hd == 0:
                w *= 0.1
            elif 0 < hd <= 2:
                w *= 0.5

            # Length-productivity bonus: boost seeds whose input length
            # has historically discovered new coverage edges.
            if hasattr(f, "_length_tracker") and f._length_tracker:
                seed_len = len(seed)
                prod = f._length_tracker.length_productivity(seed_len)
                w *= 0.5 + min(prod, 2.0) * 0.75  # [0.5, 2.0] range

            # Cross-target boost: seeds that found edges in the least-covered
            # target get a multiplier proportional to the coverage gap.
            if f.multi_targets and f._edge_tracker and f._edge_tracker.target_cumulative_edges:
                target_edges = f._edge_tracker.target_cumulative_edges
                if len(target_edges) > 1:
                    counts = {t: len(e) for t, e in target_edges.items()}
                    min_target = min(counts, key=counts.get)
                    max_target = max(counts, key=counts.get)
                    gap = counts[max_target] - counts[min_target]
                    if gap > 0:
                        sk = f._seed_key(seed)
                        seed_targets = f._edge_tracker.seed_target_edges.get(sk, {})
                        if min_target in seed_targets and seed_targets[min_target]:
                            w *= 1.0 + min(gap / max(counts[min_target], 1), 1.0)

            weights.append(max(w, 1e-6))

            novelty = sub
            freshness = burst_factor
            diversity = spa
            pareto_scores.append((novelty, freshness, diversity))

        if len(pareto_scores) >= 3:
            front = self._pareto_front(pareto_scores, window=100)
            for i in range(len(weights)):
                if i in front:
                    weights[i] *= 2.0
                else:
                    weights[i] *= 0.5

        return weights

    @staticmethod
    def _pareto_front(scores: list[tuple[float, float, float]], window: int = 100) -> set[int]:
        n = len(scores)
        start = max(0, n - window)
        front: list[int] = list(range(start, n))

        # Sort by first dimension for efficient domination check
        front.sort(key=lambda i: (-scores[i][0], -scores[i][1], -scores[i][2]))

        result = []
        max_b = max_c = float('-inf')
        for i in front:
            a, b, c = scores[i]
            if b > max_b or c > max_c:
                result.append(i)
                max_b = max(max_b, b)
                max_c = max(max_c, c)

        return set(result)

    def _pick_from_pareto_front(self, weights: list[float], now: float) -> bytes:
        f = self.f
        if len(f.corpus) < 3 or not f.seed_meta:
            return random.choices(f.corpus, weights=weights, k=1)[0]

        # Cache Pareto scores - recompute every 100 execs or when corpus changes
        cache_key = len(f.corpus)
        if not hasattr(f, "_pareto_cache") or f._pareto_cache_key != cache_key or f.exec_count % 100 == 0:
            pareto_scores: list[tuple[float, float, float]] = []
            for seed in f.corpus:
                meta = f.seed_meta.get(seed)
                if meta is None:
                    pareto_scores.append((1.0, 1.0, 1.0))
                    continue
                seed_key = f._seed_key(seed)
                sub, div, spa, _cov = f._cached_weights.get(seed_key, (1.0, 1.0, 1.0, 0.5))
                age = now - meta["added_at"]
                burst = max(1.0, 1.0 + f._temperature * (5.0 - 1.0) - (age / 60.0) * f._temperature)
                pareto_scores.append((sub, burst, spa))
            f._pareto_cache = pareto_scores
            f._pareto_cache_key = cache_key
            f._pareto_front_cache = self._pareto_front(pareto_scores, window=100)

        front = f._pareto_front_cache

        if len(front) >= 2:
            front_indices = sorted(front)
            front_weights = [weights[i] for i in front_indices]
            front_seeds = [f.corpus[i] for i in front_indices]
            return random.choices(front_seeds, weights=front_weights, k=1)[0]
        else:
            return random.choices(f.corpus, weights=weights, k=1)[0]

    def weighted_pick_seed(self) -> bytes:
        f = self.f
        now = time.time()

        if f._anneal_budget > 0:
            f._temperature = max(0.1, 1.0 - f.exec_count / f._anneal_budget)
        else:
            f._temperature = 1.0

        if not hasattr(f, "_recent_seed_edges"):
            f._recent_seed_edges: list[set[int]] = []
            f._recent_seed_max = 20

        corpus_version = len(f.corpus)
        edge_version = f.shm_cov.cumulative_edges if f.shm_cov else 0
        if not hasattr(f, "_weight_cache"):
            f._weight_cache = None
            f._weight_cache_key = (-1, -1)
            f._cached_weights = {}
        cache_key = (corpus_version, edge_version)
        if cache_key != f._weight_cache_key:
            edge_changed = f._weight_cache_key[1] != edge_version
            f._weight_cache_key = cache_key
            if (
                edge_changed
                or f._weight_cache is not None
                and len(f._weight_cache) != corpus_version
            ):
                f._weight_cache = None

        if f._weight_cache is not None:
            weights = f._weight_cache
        else:
            weights = self._compute_weights(now)
            f._weight_cache = weights

        selected = self._pick_from_pareto_front(weights, now)

        sel_key = f._seed_key(selected)
        sel_edges = f._edge_tracker.seed_edges.get(sel_key, set())
        if sel_edges:
            f._recent_seed_edges.append(sel_edges)
            if len(f._recent_seed_edges) > f._recent_seed_max:
                f._recent_seed_edges.pop(0)

        if f._ablation_file:
            meta = f.seed_meta.get(selected)
            if meta:
                seed_key = f._seed_key(selected)
                cached = f._cached_weights.get(seed_key, (1.0, 1.0, 1.0))
                fuzz_count = max(meta["fuzz_count"], 1)
                coverage = meta["coverage_edges"]
                age = now - meta["added_at"]
                base_w = (1.0 / math.sqrt(fuzz_count)) * (1.0 + coverage * 0.5) / (1.0 + age * 0.01)
                burst_factor = max(1.0, 5.0 - (age / 60.0))
                staleness = fuzz_count / max(coverage + 1, 1)
                penalty = 0.01 if staleness > 50 else 1.0
                w = base_w * burst_factor * penalty * cached[0] * cached[1] * cached[2]
                w *= 0.5 + cached[3]
                mdl_weight = 1.0
                if f.markov_trained:
                    cl_ratio = f.markov.codelength_ratio(selected)
                    mdl_weight = 1.0 + min(cl_ratio / 8.0, 1.0)
                    w *= mdl_weight
                f._last_pick_signals = {
                    "seed_idx": f.corpus.index(selected),
                    "seed_hash": selected[:4].hex(),
                    "fuzz_count": fuzz_count,
                    "coverage_edges": coverage,
                    "age_s": f"{age:.1f}",
                    "base_w": f"{base_w:.4f}",
                    "burst": f"{burst_factor:.2f}",
                    "penalty": f"{penalty:.2f}",
                    "subsumption": f"{cached[0]:.4f}",
                    "diversity": f"{cached[1]:.4f}",
                    "spatial": f"{cached[2]:.4f}",
                    "mdl": f"{mdl_weight:.2f}",
                    "final_w": f"{w:.6f}",
                }

        return selected
