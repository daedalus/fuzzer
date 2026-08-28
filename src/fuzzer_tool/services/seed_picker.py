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
from collections import Counter

from fuzzer_tool.core.crc32 import crc32
from fuzzer_tool.core.validity import VALID_SEED_BONUS

log = logging.getLogger(__name__)

# ── Edge rarity thresholds (units: distinct corpus seeds covering an edge) ──
# An edge reached by at most this many seeds counts as rare. Matches the
# singleton+cold buckets that EdgeTracker.edge_rarity_stats() reports.
RARE_EDGE_OWNERS = 3
# Multiplier on log2(1 + rare_count) for the rare-edge energy bonus.
RARE_EDGE_GAIN = 0.5
# Mean owners per edge above which a seed's coverage is considered crowded.
CROWDED_EDGE_OWNERS = 10.0


class SeedPicker:
    """Manages seed selection strategies.

    Holds a reference to the Fuzzer instance for accessing shared state.
    """

    def __init__(self, fuzzer):
        self.f = fuzzer

    def _pick_seed_elo(self) -> bytes | None:
        """Pick seed via Elo-arbitrated strategy selection. Returns None if fallback needed.

        Seed strategies are rated under ``seed_<name>`` keys (pre-registered
        in fuzzer.py, persisted in elo.json); selection must pass the
        prefixed keys, then strip the prefix for downstream use.
        """
        f = self.f
        if not f._use_elo or not f._elo:
            return None
        available = [s for s, cond in [("ga", f.ga), ("qea", f.qea)] if cond]
        available.append("weighted")
        if getattr(f, "_mcts", None) is not None and getattr(f, "_lineage", None):
            available.append("mcts")
        if f.corpus and f.seed_meta:
            available.append("pareto")
        if f._profile.format_signature:
            available.append("format")
        if getattr(f, "_use_bayesian", False) and f._seed_quality:
            available.append("bayesian")
        if f.markov_generate and f.markov_trained:
            available.append("markov")
        if getattr(f, "_use_boltzmann", False):
            available.append("boltzmann")
        if getattr(f, "_distance", None) is not None:
            available.append("aflgo")
        if getattr(f, "_katz_channel", None) is not None and f.corpus:
            available.append("katz")

        # Expose the eligible pool so the fuzzer records Elo matches only against
        # strategies that were actually selectable (no phantom opponents) and so
        # the convergence report can show only what was really arbitrated.
        f._seed_strategy_pool = list(available)

        if not available:
            return None
        # Select with the seed_<name>-prefixed keys (the keyspace elo.json
        # rates); strip the prefix so _seed_strategy/_seed_strategy_pool/
        # strategy_map/the convergence report stay plain-consistent.
        if len(available) >= 2:
            selected = f._elo.select_strategy([f"seed_{s}" for s in available])
            strategy = selected[5:] if selected.startswith("seed_") else selected
        else:
            strategy = available[0]
        f._seed_strategy = strategy
        f._seed_strategies_used.add(strategy)

        strategy_map = {
            "ga": lambda: f.ga.pick_seed() if f.ga else None,
            "qea": lambda: f.qea.pick_seed() if f.qea else None,
            "weighted": lambda: self.weighted_pick_seed(),
            "pareto": lambda: self._pick_pareto_only() if f.corpus and f.seed_meta else None,
            "format": lambda: self._format_aware_seed(),
            "bayesian": lambda: (
                self._pick_bayesian_seed() if f.corpus and f._seed_quality else None
            ),
            "markov": lambda: (
                self._pick_markov_seed() if f.markov_generate and f.markov_trained else None
            ),
            "boltzmann": lambda: self._pick_boltzmann_seed(),
            "aflgo": lambda: self._pick_aflgo_seed() if f._distance else None,
            "mcts": lambda: self._pick_mcts_seed(),
            "katz": lambda: self._pick_katz_seed(),
        }
        handler = strategy_map.get(strategy)
        return handler() if handler else None

    def _pick_mcts_seed(self) -> bytes | None:
        """MCTS/UCT descent over the lineage tree — the Elo-arbitrated 'mcts' arm.

        Every other strategy scores the corpus as a flat pool. This one walks
        the parent/child genealogy the lineage tree already maintains and
        picks a seed by UCT, so budget flows toward *regions* of the tree that
        are still producing coverage rather than toward individually
        high-scoring seeds.

        Returns None (caller falls through to another strategy) when the tree
        offers no live seed — e.g. before the first corpus insert, or when
        minimization has emptied the reachable subtrees.
        """
        f = self.f
        tree = getattr(f, "_lineage", None)
        mcts = getattr(f, "_mcts", None)
        if tree is None or mcts is None or not f.corpus:
            return None

        key_to_seed = {f._seed_key(s): s for s in f.corpus}
        selected = mcts.select(tree, set(key_to_seed))
        if selected is None:
            return None
        return key_to_seed.get(selected)

    def _pick_katz_seed(self) -> bytes | None:
        """Centrality-pure picker — the Elo-arbitrated 'katz' arm.

        Picks the corpus seed with the highest normalized Katz centrality
        over the horizon graph (rarely-hit horizons dominate beta). Falls
        back to energy-weighted sampling when scores tie at zero so the
        arm still explores; returns None only when nothing is scored.
        """
        f = self.f
        ch = getattr(f, "_katz_channel", None)
        if ch is None or not f.corpus:
            return None
        weights = []
        for seed in f.corpus:
            e = ch.seed_energy(f._seed_key(seed))
            weights.append(max(e, 1e-4) ** 2)
        total = sum(weights)
        if total <= 0:
            return None
        r = random.random() * total
        acc = 0.0
        for seed, w in zip(f.corpus, weights, strict=False):
            acc += w
            if acc >= r:
                return seed
        return f.corpus[-1]

    def _pick_aflgo_seed(self) -> bytes | None:
        """Distance-pure seed picker — the Elo-arbitrated 'aflgo' arm.

        P(seed) ∝ exp(-2 · norm_dist) with norm_dist = avg_distance /
        max_distance; seeds without distance data count as farthest.
        This is deliberately more aggressive about near-target seeds
        than the generic weighted arm (which blends distance with
        speed/size/entropy), giving Elo a distinct strategy to rate.
        """
        f = self.f
        if not f.corpus or not f._distance:
            return None
        max_d = f._distance.max_distance
        dists = []
        for seed in f.corpus:
            meta = f.seed_meta.get(seed, {})
            seed_dist = meta.get("avg_distance")
            norm = 1.0 if seed_dist is None or max_d <= 0 else min(seed_dist / max_d, 1.0)
            dists.append(math.exp(-2.0 * norm))
        total = sum(dists)
        if total <= 0:
            return None
        r = random.random() * total
        acc = 0.0
        for seed, d in zip(f.corpus, dists, strict=False):
            acc += d
            if acc >= r:
                return seed
        return f.corpus[-1] if f.corpus else None

    def pick_seed(self) -> bytes:
        f = self.f
        rng = f._rand_pool

        # Update SA temperature on every call, regardless of which
        # strategy Elo selects (bug: was inside weighted_pick_seed()
        # so temperature only cooled when 'weighted' won the bandit).
        if f._anneal_budget > 0:
            f._temperature = max(0.1, 1.0 - f.exec_count / f._anneal_budget)
        else:
            f._temperature = 1.0

        if f._stall_recovery_active and f.corpus:
            f._seed_strategy = "random_stall"
            return rng.choice(f.corpus)

        elo_pick = self._pick_seed_elo()
        if elo_pick is not None:
            return elo_pick

        if f.qea:
            return f.qea.pick_seed()
        if f.ga:
            return f.ga.pick_seed()
        if f.corpus and getattr(f, "_use_bayesian", False) and f._seed_quality:
            return self._pick_bayesian_seed()
        if f.corpus and f.seed_meta:
            if getattr(f, "_use_boltzmann", False):
                return self._pick_boltzmann_seed()
            return self.weighted_pick_seed()
        if f.corpus:
            return rng.choice(f.corpus)
        return self._format_aware_seed()

    def _pick_markov_seed(self) -> bytes:
        f = self.f
        rng = f._rand_pool
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

        if rng.random() < gen_rate:
            length = rng.randint(1, min(256, f.max_len))
            for _ in range(3):
                candidate = f.markov.generate(length)
                pp = f.markov.perplexity(candidate)
                if pp < 512:
                    return candidate
            return candidate
        length = rng.randint(1, min(256, f.max_len))
        return f.markov.generate(length)

    def _pick_boltzmann_seed(self) -> bytes:
        """Pick seed via Boltzmann distribution over rarity energy.

        P(seed) ∝ exp(-E/T) where E = log(fuzz_count + 1), so
        weight = (fuzz_count + 1)^(-1/T).  Rare seeds (low fuzz_count)
        dominate at cold T; all seeds are roughly uniform at hot T.
        Falls back to random choice if corpus or seed_meta is empty.
        """
        f = self.f
        rng = f._rand_pool
        if not f.corpus or not f.seed_meta:
            return self._format_aware_seed()
        T = max(f._temperature, 0.01)
        weights = []
        for seed in f.corpus:
            meta = f.seed_meta.get(seed)
            if meta is None:
                weights.append(1e-6)
                continue
            n = max(meta.get("fuzz_count", 1), 1)
            E = math.log(n + 1)
            w = math.exp(-E / T)
            weights.append(max(w, 1e-6))
        total = sum(weights)
        if total <= 0:
            return f._rand_pool.choice(f.corpus)
        r = rng.random() * total
        cumulative = 0.0
        for i, seed in enumerate(f.corpus):
            cumulative += weights[i]
            if r <= cumulative:
                return seed
        return f.corpus[-1]

    def _pick_pareto_only(self) -> bytes:
        f = self.f
        if len(f.corpus) < 3 or not f.seed_meta:
            return f._rand_pool.choice(f.corpus)
        now = time.time()
        weights = [1.0] * len(f.corpus)
        return self._pick_from_pareto_front(weights, now)

    def _pick_bayesian_seed(self) -> bytes:
        """Pick seed via Thompson sampling from BayesianSeedQuality posteriors.

        Thompson sampling from the Beta posterior of each registered seed
        naturally explores seeds with high uncertainty (few observations)
        while exploiting seeds with proven success rates.
        """
        f = self.f
        if not f.corpus:
            return self._format_aware_seed()
        if not f._seed_quality:
            return f._rand_pool.choice(f.corpus)

        # Build list of registered seed IDs (content hashes)
        seed_ids = [f._seed_key(s) for s in f.corpus]
        # Ensure all current corpus seeds are registered (new seeds may not be yet)
        for sid in seed_ids:
            if sid not in f._seed_quality._alpha:
                f._seed_quality.init_seed(sid)

        # Thompson sample: pick the seed with the highest posterior draw
        chosen_id = f._seed_quality.select_seed(seed_ids)
        # Map back to the seed bytes
        for s in f.corpus:
            if f._seed_key(s) == chosen_id:
                return s
        return f._rand_pool.choice(f.corpus)

    def _pick_by_similarity(self) -> bytes:
        """Pick a seed diverse from recently fuzzed ones using byte-level similarity.

        Uses find_nearest_bytes to avoid picking seeds that are similar to
        recently selected seeds, promoting exploration diversity.
        """
        f = self.f
        if len(f.corpus) < 2:
            return f._rand_pool.choice(f.corpus) if f.corpus else b""
        from fuzzer_tool.core.similarity import find_nearest_bytes

        recent = getattr(f, "_recent_seeds", [])
        if not recent:
            return f._rand_pool.choice(f.corpus)
        # Pick the seed least similar to any recently fuzzed seed
        worst_sim = 1.0
        worst_idx = 0
        for idx, seed in enumerate(f.corpus):
            idx_candidate, sim = find_nearest_bytes(seed, recent)
            if sim < worst_sim:
                worst_sim = sim
                worst_idx = idx
        return f.corpus[worst_idx]

    def _format_aware_seed(self) -> bytes:
        f = self.f
        fmt = getattr(f._profile, "format_signature", None)
        if fmt == "png":
            ihdr_data = b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
            ihdr_chunk = b"IHDR" + ihdr_data
            ihdr_crc = struct.pack(">I", crc32(ihdr_chunk))
            iend_chunk = b"IEND"
            iend_crc = struct.pack(">I", crc32(iend_chunk))
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
        elif fmt == "webp":
            from fuzzer_tool.core.mutations.webp import WebpMutator

            return WebpMutator()._generate_random_webp(max_len=256)
        elif fmt == "webm":
            from fuzzer_tool.core.mutations.webm import WebmMutator

            return WebmMutator()._generate_random_webm(max_len=256)
        elif fmt == "zip":
            from fuzzer_tool.core.mutations.zip import ZipMutator

            return ZipMutator()._generate_random_zip(max_len=256)
        elif fmt == "protobuf":
            from fuzzer_tool.core.mutations.protobuf import ProtobufMutator

            return ProtobufMutator()._generate_random_protobuf(max_len=256)
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
                + struct.pack("<I", crc32(b"\x00"))
                + struct.pack("<I", 1)
            )
        # Generic: zero-filled random-length buffer
        rng = f._rand_pool
        length = rng.randint(min(4, f.max_len), min(64, f.max_len))
        return bytes(rng.randint(0, 255) for _ in range(length))

    def _weight_exploit_parts(
        self, meta: dict, fuzz_count: int, coverage: int, age: float, T: float
    ) -> tuple[float, float]:
        """Compute base explore/exploit, momentum, burst, and staleness factors. Returns (weight, burst_factor)."""
        explore_part = T * (1.0 / math.sqrt(fuzz_count))
        exploit_part = (1.0 + coverage * 0.5) / (1.0 + age * 0.01)
        w = explore_part * exploit_part
        momentum = meta.get("momentum", 0.0)
        w *= 1.0 + momentum * 2.0
        burst_factor = max(1.0, 1.0 + T * (5.0 - 1.0) - (age / 60.0) * T)
        staleness = fuzz_count / max(coverage + 1, 1)
        stale_threshold = 50.0 * T
        w *= 0.01 if staleness > stale_threshold else 1.0
        return w, burst_factor

    def _weight_secretary_and_cached(
        self, seed_key: str, w: float, classifications: dict | None, f
    ) -> tuple[float, float, float]:
        """Apply secretary stopping rule and cached edge weights."""
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

        if seed_key in classifications:
            cls = classifications[seed_key]["classification"]
            if cls == "keystone":
                w *= 2.0
            elif cls == "parasitic":
                w *= 0.1
        return w, sub, spa

    @staticmethod
    def _recent_edge_counts(f) -> dict[int, int] | None:
        """Fold ``f._recent_seed_edges`` into one {edge_id: window occurrences} map.

        The overlap penalty below wants ``sum(len(seed_edges & recent) for
        recent in window)``, which is by definition the number of (edge,
        window-slot) incidences -- i.e. the sum over the seed's own edges of
        how many window slots contain each. Counting it that way turns 20 set
        intersections per seed into one dict lookup per edge, folded into the
        owner-count pass that already walks the same edges.

        The window is shared by every seed in a ``_compute_weights`` pass, so
        this is built once per pass and threaded through; the intersections it
        replaces were rebuilt per seed. At the measured shape of an FFmpeg run
        (197 seeds x ~460 edges x a 20-slot window) the pass goes from 71.4ms
        to 11.6ms, and ``len(a & b)`` stops allocating a throwaway set of the
        intersection purely to read its size.

        Returns None when the window is absent or empty, which is the same
        condition the old ``hasattr`` guard tested: no window means no penalty.
        """
        recent = getattr(f, "_recent_seed_edges", None)
        if not recent:
            return None
        counts: Counter[int] = Counter()
        for edges in recent:
            counts.update(edges)
        return counts

    def _weight_edge_penalties(
        self,
        seed_key: str,
        w: float,
        fuzz_count: int,
        f,
        recent_counts: dict[int, int] | None = None,
    ) -> float:
        """Apply rare edge bonus, crowding adjustment, and overlap penalty.

        Rarity is measured in *seeds* (``_edge_owner_count``), not in bucketed
        execution hit volume (``_global_edge_hits``). The two were conflated
        here: an edge inside a hot loop accumulates hundreds of counter units
        from a single execution, so it could never fall under the old
        ``hits <= 2`` test no matter how few seeds reached it, while an edge
        touched once by one seed passed the test whether or not it was rare.
        The same conflation was fixed in ``edge_rarity_stats()``; this was the
        remaining instance, and it is the one that steered energy.

        The bonus is also applied once rather than twice. ``rare_count`` and
        ``gap_score`` were incremented under the same condition and then
        multiplied in separately as ``(1 + 0.5r)(1 + 0.3r)``, a quadratic that
        reached ~77x at r=20 with no ceiling -- enough to starve the rest of
        the queue on a seed that happens to sit on a lot of fresh code.
        """
        tracker = f._edge_tracker
        seed_edges = tracker.seed_edges.get(seed_key, set())
        if not seed_edges:
            return w
        if recent_counts is None:
            recent_counts = self._recent_edge_counts(f)
        owners_get = tracker._edge_owner_count.get
        rare_count = 0
        total_owners = 0
        overlap = 0
        # Two loop bodies rather than a per-edge branch: this runs once per
        # corpus seed per weight pass, i.e. tens of millions of iterations
        # over a fuzzing session, and the branch is loop-invariant.
        if recent_counts:
            recent_get = recent_counts.get
            for e in seed_edges:
                n = owners_get(e, 0)
                total_owners += n
                if n <= RARE_EDGE_OWNERS:
                    rare_count += 1
                overlap += recent_get(e, 0)
        else:
            for e in seed_edges:
                n = owners_get(e, 0)
                total_owners += n
                if n <= RARE_EDGE_OWNERS:
                    rare_count += 1

        if rare_count > 0:
            # log2 rather than linear: the marginal value of the twentieth rare
            # edge on a seed is not twenty times that of the first, and a linear
            # bonus lets one seed monopolise the queue. Matches the scaling the
            # ENTROPIC schedule already uses for the same signal.
            w *= 1.0 + RARE_EDGE_GAIN * math.log2(1.0 + rare_count)

        # Crowding: how many other seeds already reach the same code. A seed
        # whose edges are reached by many others is redundant; one sitting on
        # thinly-covered edges is worth more. The old version read mean hit
        # *volume* and boosted seeds with high counts, which rewarded hot loops
        # -- the opposite of what a rarity-driven schedule wants.
        mean_owners = total_owners / len(seed_edges)
        if mean_owners > CROWDED_EDGE_OWNERS:
            w *= max(0.5, CROWDED_EDGE_OWNERS / mean_owners)
        elif mean_owners < 1.5 and fuzz_count > 10:
            # Thinly covered but already heavily fuzzed: diminishing returns.
            w *= 0.7

        if overlap > 0:
            w *= max(0.3, 1.0 - (overlap / max(len(seed_edges), 1)) * 0.5)
        return w

    def _weight_entropy_and_distance(
        self,
        seed: bytes,
        seed_key: str,
        meta: dict,
        w: float,
        f,
        entropy_map: dict | None = None,
        mean_entropy: float = 0.0,
        max_d: float = 0.0,
    ) -> float:
        """Apply Shannon entropy bonus and directed distance weight.

        Args:
            entropy_map: Pre-computed {seed_key: entropy} dict from _compute_weights.
            mean_entropy: Pre-computed mean entropy across all seeds.
            max_d: Pre-computed max_distance for normalization.
        """
        # Use pre-computed entropy if available, else compute on the fly
        if entropy_map is not None:
            seed_sh = entropy_map.get(seed_key, 0.0)
        else:
            seed_sh = f._edge_tracker.shannon_entropy_seed(seed_key)

        if seed_sh > 0 and len(f._edge_tracker.seed_hit_counts) >= 3:
            # Use pre-computed mean if available, else fall back to cached computation
            if mean_entropy > 0:
                effective_mean = mean_entropy
            else:
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
                effective_mean = self._mean_seed_entropy

            if effective_mean > 0:
                deviation = abs(seed_sh - effective_mean) / max(effective_mean, 0.01)
                w *= 1.0 + min(deviation, 1.0) * 0.5

        if f._distance:
            seed_dist = meta.get("avg_distance", max_d if max_d > 0 else f._distance.max_distance)
            if max_d <= 0:
                max_d = f._distance.max_distance
            norm_dist = min(seed_dist / max_d, 1.0) if max_d > 0 else 0.5
            alpha = min(f._anneal_progress * 2, 1.0)
            w *= (1.0 - alpha) + alpha * math.exp(-norm_dist * 5.0 * alpha)
        return w

    def _weight_static_features(self, seed: bytes, coverage: int, w: float, f) -> float:
        """Apply PPMD novelty and hot function density bonuses."""
        ppmd = getattr(f, "_ppmd", None)
        if ppmd and ppmd.enabled:
            w *= 1.0 + ppmd.compute_seed_novelty(seed) * 0.5

        if f._profile.hot_functions and f._profile.functions:
            # Cache hot/all density — they depend only on the profile, not
            # the seed. Without caching, these sums (over 691 functions)
            # are recomputed 1404×27 = 38K times per weight pass.
            if not hasattr(f, "_hot_density_cache"):
                f._hot_density_cache = {}
            cache_key = (id(f._profile.hot_functions), id(f._profile.functions))
            cached = f._hot_density_cache.get(cache_key)
            if cached is None:
                hot_density = sum(
                    f._profile.functions[fn].branch_density
                    for fn in f._profile.hot_functions
                    if fn in f._profile.functions
                ) / max(len(f._profile.hot_functions), 1)
                all_density = sum(fi.branch_density for fi in f._profile.functions.values()) / max(
                    len(f._profile.functions), 1
                )
                f._hot_density_cache[cache_key] = (hot_density, all_density)
            else:
                hot_density, all_density = cached
            if all_density > 0 and coverage > 0:
                hotness_ratio = hot_density / all_density
                w *= 1.0 + (hotness_ratio - 1.0) * min(coverage / 50.0, 1.0)
        return w

    def _weight_length_and_cross_target(self, seed: bytes, meta: dict, w: float, f) -> float:
        """Apply hamming, length-productivity, and cross-target bonuses."""
        hd = meta.get("hamming_distance", -1)
        if hd == 0:
            w *= 0.1
        elif 0 < hd <= 2:
            w *= 0.5

        if hasattr(f, "_length_tracker") and f._length_tracker:
            prod = f._length_tracker.length_productivity(len(seed))
            w *= 0.5 + min(prod, 2.0) * 0.75

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
        return w

    def _weight_overlap_density(self, seed_key: str, w: float, f) -> float:
        """Apply overlap-density-based weight modifier.

        Seeds with high pairwise edge-set overlap (similar coverage to many
        other seeds) are penalised; seeds with low overlap are boosted.

        Uses pre-computed overlap density from the FMM-clustered cache
        (``_overlap_density_cache``).  A no-op when the feature is disabled.
        """
        od = getattr(f, "_overlap_density_cache", None)
        if od is None or not getattr(f, "_use_overlap_density", False):
            return w
        density = od.get(seed_key)
        if density is None:
            return w
        blend = getattr(f, "_overlap_density_blend", 0.5)
        # Map: density 0 → modifier 1.5 (boost), density 1 → modifier 0.5 (penalty)
        modifier = 1.0 + blend * (0.5 - density)
        return w * max(modifier, 0.1)

    def _weight_validity(self, meta: dict, w: float, f) -> float:
        """Boost seeds the target accepted (Zest validity channel).

        A boost rather than a gate: an invalid seed is often the shortest
        path to a branch inside the parser, so validity ranks the corpus
        and never excludes from it. A no-op when --reject-code is unset --
        every seed is then unclassified and the metadata key is absent.
        """
        valid = meta.get("valid")
        if valid is None or not valid:
            return w
        return w * VALID_SEED_BONUS

    def _weight_lineage_backtrack(
        self, seed_key: str, w: float, fuzz_count: int, f, key_to_seed: dict
    ) -> float:
        """Back off to shallower lineage when a branch stops producing edges.

        Depth-first drift is a known failure mode for coverage-guided
        fuzzing: each new seed is a child of the last interesting one, so
        selection walks steadily deeper down one lineage branch. When that
        branch saturates, continued mutation refines an already-explored
        region while whole sibling branches go untouched.

        This detects an exhausted branch — a seed whose lineage subtree has
        gained no coverage since the last credit reset, and which has been
        fuzzed enough times that the absence is evidence rather than noise —
        and geometrically penalises it by depth. Weight therefore shifts back
        toward the root, so the next pick is more likely to come from a
        shallow seed with unexplored siblings. That is the "backtrack": not
        an explicit jump, but a bias that makes descending a dead branch
        progressively less attractive.

        Seeds whose subtree is still producing coverage are untouched, so a
        productive deep branch keeps its weight.

        No-op unless ``--lineage-backtrack`` is set (requires ``--lineage``).
        """
        if not getattr(f, "_use_lineage_backtrack", False):
            return w
        tree = getattr(f, "_lineage", None)
        if tree is None:
            return w
        node = tree.nodes.get(seed_key)
        if node is None or node.depth <= 0:
            return w
        # Require enough fuzzing before treating "no new edges" as signal.
        min_fuzz = getattr(f, "_lineage_backtrack_min_fuzz", 8)
        if fuzz_count < min_fuzz:
            return w

        def _coverage_fn(k: str) -> tuple[int, int]:
            seed = key_to_seed.get(k)
            meta = f.seed_meta.get(seed, {}) if seed is not None else {}
            return (
                meta.get("coverage_edges", 0),
                meta.get("coverage_edges_baseline", 0),
            )

        try:
            credit = tree.recent_credit(seed_key, _coverage_fn)
        except (AttributeError, KeyError, TypeError):
            return w
        if credit > 0.0:
            return w  # branch still productive — leave it alone

        decay = getattr(f, "_lineage_backtrack_decay", 0.7)
        return w * max(decay**node.depth, 0.05)

    def _compute_weights(self, now: float) -> list[float]:
        f = self.f
        corpus = f.corpus
        n = len(corpus)
        weights = [1.0] * n
        pareto_scores: list[tuple[float, float, float]] = [(1.0, 1.0, 1.0)] * n

        if not hasattr(f, "_classify_cache") or f.exec_count % 100 == 0:
            f._classify_cache = f._edge_tracker.classify_seeds()
        classifications = f._classify_cache

        T = f._temperature
        seed_meta = f.seed_meta

        # Phase 1: extract metadata into parallel arrays for vectorized math
        has_meta = [False] * n
        fuzz_arr = None
        cov_arr = None
        age_arr = None
        mom_arr = None

        # Pre-compute seed keys and entropy for all seeds in one pass
        seed_keys = [None] * n
        entropy_map = {}
        entropy_sum = 0.0
        entropy_count = 0

        try:
            import numpy as _np

            fuzz_list = []
            cov_list = []
            age_list = []
            mom_list = []
            meta_indices = []

            for i, seed in enumerate(corpus):
                meta = seed_meta.get(seed)
                if meta is None:
                    continue
                has_meta[i] = True
                meta_indices.append(i)
                fuzz_list.append(max(meta["fuzz_count"], 1))
                cov_list.append(meta["coverage_edges"])
                age_list.append(now - meta["added_at"])
                mom_list.append(meta.get("momentum", 0.0))

                # Pre-compute seed key and entropy in same pass
                sk = f._seed_key(seed)
                seed_keys[i] = sk
                ent = f._edge_tracker.shannon_entropy_seed(sk)
                if ent > 0:
                    entropy_map[sk] = ent
                    entropy_sum += ent
                    entropy_count += 1

            if meta_indices:
                fuzz_arr = _np.array(fuzz_list, dtype=_np.float64)
                cov_arr = _np.array(cov_list, dtype=_np.float64)
                age_arr = _np.array(age_list, dtype=_np.float64)
                mom_arr = _np.array(mom_list, dtype=_np.float64)

                # Vectorized _weight_exploit_parts
                explore = T * (1.0 / _np.sqrt(fuzz_arr))
                exploit = (1.0 + cov_arr * 0.5) / (1.0 + age_arr * 0.01)
                w_vec = explore * exploit
                w_vec *= 1.0 + mom_arr * 2.0
                burst_vec = _np.maximum(1.0, 1.0 + T * 4.0 - (age_arr / 60.0) * T)
                staleness = fuzz_arr / _np.maximum(cov_arr + 1, 1)
                stale_mask = staleness > 50.0 * T
                w_vec[stale_mask] *= 0.01

                # Write back vectorized results
                for j, idx in enumerate(meta_indices):
                    weights[idx] = float(w_vec[j])
                    pareto_scores[idx] = (1.0, float(burst_vec[j]), 1.0)
        except ImportError:
            pass

        # Compute FMM-clustered pairwise overlap density if enabled.
        # This runs before Phase 2 so the per-seed loop can consume it.
        if getattr(f, "_use_overlap_density", False) and n >= 3:
            all_keys: list[str] = []
            for s in corpus:
                all_keys.append(f._seed_key(s))
            od_result = f._edge_tracker.compute_overlap_density(
                all_keys, min_jaccard=getattr(f, "_overlap_min_jaccard", 0.25)
            )
            f._overlap_density_cache = od_result[0]
        else:
            f._overlap_density_cache = {}

        # Pre-compute mean entropy once
        mean_entropy = entropy_sum / entropy_count if entropy_count > 0 else 0.0

        # Pre-compute max_distance once (avoids repeated property access)
        max_d = f._distance.max_distance if f._distance else 0.0

        # LCA-based lineage diversity multipliers (Query 3): a seed whose
        # lineage subtree is far from a sampled set of peers gets a small
        # weight boost (mult = 1.0 + 0.5 * diversity, diversity in [0, 1]).
        lineage_div: dict[str, float] = {}
        if f._use_lineage and getattr(f, "_lineage", None) is not None and n >= 2:
            tree = f._lineage
            max_depth = max((node.depth for node in tree.nodes.values()), default=0)
            if max_depth > 0:
                # seed_keys was filled by the pre-compute pass above; only
                # seeds without metadata are still None, so re-hashing the
                # whole corpus here was redundant.
                all_sk = [
                    sk if sk is not None else f._seed_key(s)
                    # strict=True: seed_keys is built as [None] * len(corpus)
                    # at the top of this pass, so the lengths are structurally
                    # equal and a mismatch means the pre-compute pass has been
                    # broken. Silently truncating to the shorter of the two
                    # would drop seeds from the diversity pool without a
                    # symptom.
                    for sk, s in zip(seed_keys, corpus, strict=True)
                ]
                rng = getattr(f, "_rand_pool", None)
                sample_cap = 64
                # Draw the peer pool once per pass instead of rebuilding
                # `[k for k in all_sk if k != sk_i]` for every seed: that was
                # O(n^2) list construction to keep sample_cap entries of it
                # (81ms/pass at n=2000). Peers are now shared across seeds
                # within a pass rather than drawn independently per seed --
                # this is a diversity heuristic averaged over the pool, and
                # the pool is redrawn on the next pass, so the estimate stays
                # unbiased across passes while costing O(n).
                n_sk = len(all_sk)
                if rng is not None and n_sk > sample_cap + 1:
                    pool_idx = rng.sample(n_sk, sample_cap + 1)
                else:
                    pool_idx = range(n_sk)
                pool_idx = list(pool_idx)
                for i, sk_i in enumerate(all_sk):
                    sample = [all_sk[j] for j in pool_idx if j != i][:sample_cap]
                    valid = [d for d in (tree.lca_distance(sk_i, k) for k in sample) if d >= 0]
                    avg = sum(valid) / len(valid) if valid else 0.0
                    diversity = min(avg / (2.0 * max_depth), 1.0)
                    lineage_div[sk_i] = 1.0 + 0.5 * diversity

        # Phase 2: apply remaining per-seed weight functions (dict lookups, set ops)
        # Built once per pass: lineage backtracking resolves subtree keys back
        # to seeds to read their coverage meta.
        bt_key_to_seed: dict = {}
        if getattr(f, "_use_lineage_backtrack", False) and getattr(f, "_lineage", None):
            bt_key_to_seed = {(seed_keys[i] or f._seed_key(s)): s for i, s in enumerate(corpus)}
        # Window occurrence counts, folded once per pass rather than
        # re-intersected per seed. See _recent_edge_counts.
        recent_counts = self._recent_edge_counts(f)
        for i, seed in enumerate(corpus):
            if not has_meta[i]:
                continue
            meta = seed_meta.get(seed)
            fuzz_count = max(meta["fuzz_count"], 1)
            sk = seed_keys[i] or f._seed_key(seed)
            w = weights[i]

            w, sub, spa = self._weight_secretary_and_cached(sk, w, classifications, f)
            w = self._weight_edge_penalties(sk, w, fuzz_count, f, recent_counts)
            w = self._weight_entropy_and_distance(
                seed, sk, meta, w, f, entropy_map, mean_entropy, max_d
            )
            w = self._weight_static_features(seed, meta["coverage_edges"], w, f)
            w = self._weight_length_and_cross_target(seed, meta, w, f)
            w = self._weight_overlap_density(sk, w, f)
            w = self._weight_validity(meta, w, f)
            w *= lineage_div.get(sk, 1.0)
            w = self._weight_lineage_backtrack(sk, w, fuzz_count, f, bt_key_to_seed)

            weights[i] = max(w, 1e-6)
            bf = pareto_scores[i][1]
            is_pareto4d = (
                getattr(f, "_use_overlap_density", False)
                and getattr(f, "_overlap_mode", "") == "pareto4d"
            )
            if is_pareto4d:
                od = f._overlap_density_cache.get(sk, 0.5)
                pareto_scores[i] = (sub, bf, spa, od)
            else:
                pareto_scores[i] = (sub, bf, spa)

        if len(pareto_scores) >= 3:
            front = self._pareto_front(pareto_scores, window=100)
            front_set = front  # already a set
            for i in range(len(weights)):
                weights[i] *= 2.0 if i in front_set else 0.5

        return weights

    @staticmethod
    def _pareto_front(scores: list[tuple[float, ...]], window: int = 100) -> set[int]:
        n = len(scores)
        start = max(0, n - window)
        indices = list(range(start, n))
        if not indices:
            return set()

        dims = len(scores[indices[0]]) if scores else 3

        # 3D and below: O(N) rolling-max sweep (backward compatible path)
        if dims <= 3:
            indices.sort(key=lambda i: (-scores[i][0], -scores[i][1], -scores[i][2]))
            result = []
            max_b = max_c = float("-inf")
            for i in indices:
                _a, b, c = scores[i][0], scores[i][1], scores[i][2]
                if b > max_b or c > max_c:
                    result.append(i)
                    max_b = max(max_b, b)
                    max_c = max(max_c, c)
            return set(result)

        # 4D+: simple O(N²) dominance — fine for window ≤ 100
        indices.sort(key=lambda i: (-scores[i][0],))
        pareto: list[int] = []
        for i in indices:
            dominated = False
            for j in pareto:
                if all(scores[j][d] >= scores[i][d] for d in range(dims)):
                    dominated = True
                    break
            if not dominated:
                pareto = [
                    j for j in pareto if not all(scores[i][d] >= scores[j][d] for d in range(dims))
                ]
                pareto.append(i)
        return set(pareto)

    def _pick_from_pareto_front(self, weights: list[float], now: float) -> bytes:
        f = self.f
        # _cached_weights is lazy-initialized by weighted_pick_seed; the Elo
        # 'pareto' strategy reaches this directly, so ensure it exists.
        if not hasattr(f, "_cached_weights"):
            f._cached_weights = {}
        if len(f.corpus) < 3 or not f.seed_meta:
            return random.choices(f.corpus, weights=weights, k=1)[0]

        # Cache Pareto scores - recompute every 100 execs or when corpus changes
        cache_key = len(f.corpus)
        use_pareto4d = (
            getattr(f, "_use_overlap_density", False)
            and getattr(f, "_overlap_mode", "") == "pareto4d"
        )
        if (
            not hasattr(f, "_pareto_cache")
            or f._pareto_cache_key != cache_key
            or f.exec_count % 100 == 0
        ):
            pareto_scores: list[tuple[float, ...]] = []
            for seed in f.corpus:
                meta = f.seed_meta.get(seed)
                if meta is None:
                    pareto_scores.append((1.0, 1.0, 1.0))
                    continue
                seed_key = f._seed_key(seed)
                sub, div, spa, _cov = f._cached_weights.get(seed_key, (1.0, 1.0, 1.0, 0.5))
                age = now - meta["added_at"]
                burst = max(1.0, 1.0 + f._temperature * (5.0 - 1.0) - (age / 60.0) * f._temperature)
                if use_pareto4d:
                    od = getattr(f, "_overlap_density_cache", {}).get(seed_key, 0.5)
                    pareto_scores.append((sub, burst, spa, od))
                else:
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

    def _log_pick_signals(self, selected: bytes, now: float) -> None:
        """Log ablation pick signals for debugging."""
        f = self.f
        if not f._ablation_file:
            return
        meta = f.seed_meta.get(selected)
        if not meta:
            return
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
        if getattr(f, "_use_overlap_density", False):
            od_val = getattr(f, "_overlap_density_cache", {}).get(seed_key, 0.0)
            f._last_pick_signals["overlap_density"] = f"{od_val:.4f}"

    def weighted_pick_seed(self) -> bytes:
        f = self.f
        now = time.time()

        if not hasattr(f, "_recent_seed_edges"):
            f._recent_seed_edges: list[set[int]] = []
            f._recent_seed_max = 20

        corpus_version = len(f.corpus)
        if not hasattr(f, "_weight_cache"):
            f._weight_cache = None
            f._weight_cache_key = (-1, -1)
            f._cached_weights = {}
        if len(f._cached_weights) > max(corpus_version * 2, 4000):
            keys = list(f._cached_weights)[: len(f._cached_weights) // 2]
            for k in keys:
                del f._cached_weights[k]
        cache_key = (corpus_version, f.exec_count // 50)
        if cache_key != f._weight_cache_key:
            f._weight_cache_key = cache_key
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

        self._log_pick_signals(selected, now)
        return selected
