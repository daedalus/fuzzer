"""Power schedules for seed-level energy allocation.

Ports AFL++ power schedules that control how much "energy" (mutation
budget) each queue entry receives. These operate at the seed level,
complementing operator-level schedules like Thompson sampling and MOpt.

Schedules:
- BASE: AFL's default speed/size/depth scoring (no frequency adjustment)
- FAST: AFLFast's frequency-based energy (rare seeds get more energy)
- COE: Cut-Off Exponential (skip over-fuzzed seeds entirely)
- RARE: tc_ref-based scoring (seeds owning rare edges get boosted)
- MMOPT: Depth-based boost for recent entries
- LIN/QUAD: Linear/quadratic falloff with fuzz count
- GO: AFLGo-style distance-annealed boost (exp(β·(1−norm_dist)))
- AFLGO: the exact AFLGo power schedule — symmetric distance factor
  around 1.0 combined with a time-based cooling temperature T.
  p = (1−norm_dist)(1−T) + 0.5T; factor = 2^(2·log2(32)·(p−0.5));
  T follows an exp/log/lin/quad cooling over t_x minutes to
  exploitation. Near-target seeds get up to 32× energy, far seeds as
  little as 1/32×.
- ENTROPIC: libFuzzer's `-entropic` schedule — energy scales with
  log2(1 + rare-feature count), using the rare-edge-ownership counts
  (tc_ref, rare_edge_count) already collected for RARE/honggfuzz
  scoring. Seeds touching more rare/undersampled features get
  proportionally more mutation budget; seeds with no rare features get
  the schedule-neutral 1.0x factor.

Honggfuzz factors (applied multiplicatively on top of schedule scoring):
- Novelty decay: new-edge bonus that decays over 10 minutes
- Density: coverage-per-byte ratio boost
- Fertility: logarithmic boost for inputs that produced children
- Freshness: time-based boost (<60s: 4x, <5min: 2x, >60min stale: 0.5x)
- CMP progress: boost for inputs making comparison progress
- Entropy penalty: penalize high/very-low entropy inputs
- Timeout penalty: 1/32 energy for timeout-causing inputs
"""

import math

# Byte-entropy cut points for the honggfuzz energy factor, on the 0-100
# scale produced by core.byte_entropy.byte_entropy_pct. These correspond to
# 7.44, 4.96 and 2.00 bits/byte, which is where random/compressed, plain
# text and near-zero inputs respectively fall. Named so the stats counter in
# services.fuzzer classifies against the same numbers the factor applies;
# the two drifting apart would make the reported "ent:" count describe a
# penalty the scorer never charged.
ENTROPY_RANDOM_PCT = 93.0
ENTROPY_STRUCTURED_PCT = 62.0
ENTROPY_SPARSE_PCT = 25.0


class SeedScorer:
    """Compute energy scores for queue entries using various power schedules.

    Each schedule modifies a base score (from speed/size/depth) by a
    frequency-based factor. The factor depends on how often a seed has
    been fuzzed relative to others.

    Args:
        schedule: One of 'base', 'fast', 'coe', 'rare', 'mopt', 'lin', 'quad',
            'go', 'aflgo', 'entropic'.
        max_mult: Maximum havoc multiplier (default 16).
        aflgo_cooling: Cooling schedule for the 'aflgo' power factor:
            'exp', 'log', 'lin' or 'quad'.
        t_x_minutes: Time to exploitation in minutes (AFLGo's -c).
    """

    SCHEDULES = (
        "base",
        "fast",
        "coe",
        "rare",
        "mopt",
        "lin",
        "quad",
        "go",
        "aflgo",
        "entropic",
        "katz",
    )
    COOLING = ("exp", "log", "lin", "quad")

    def __init__(
        self,
        schedule: str = "base",
        max_mult: int = 16,
        aflgo_cooling: str = "exp",
        t_x_minutes: float = 60.0,
    ):
        if schedule not in self.SCHEDULES:
            raise ValueError(f"Unknown schedule: {schedule!r}. Use one of {self.SCHEDULES}")
        if aflgo_cooling not in self.COOLING:
            raise ValueError(f"Unknown cooling schedule: {aflgo_cooling!r}")
        self.schedule = schedule
        self.max_mult = max_mult
        self.max_factor = 32.0
        self.power_beta = 1.0
        self.aflgo_cooling = aflgo_cooling
        self.t_x_minutes = t_x_minutes
        # Running averages for hw perf normalization (EMA)
        self._avg_hw_instructions: float = 0.0
        self._avg_hw_branches: float = 0.0
        self._hw_avg_alpha: float = 0.1  # EMA smoothing factor

    def score(
        self,
        exec_us: int,
        avg_exec_us: int,
        bitmap_size: int,
        avg_bitmap_size: int,
        handicap: int,
        depth: int,
        fuzz_level: int,
        n_fuzz: int,
        total_execs: int,
        tc_ref: int = 0,
        favored: bool = False,
        max_depth: int = 0,
        mean_log_n_fuzz: float = 0.0,
        # Honggfuzz power factors (all optional)
        new_edges: int = 0,
        time_added: float = 0.0,
        now: float = 0.0,
        input_size: int = 0,
        select_count: int = 0,
        child_count: int = 0,
        cmp_progress: int = 0,
        rare_edge_count: int = 0,
        timed_out: bool = False,
        input_entropy: float = -1.0,
        max_cov: int = 0,
        hw_instructions: int = 0,
        hw_branches: int = 0,
        # AFLGo directed-distance annealing
        avg_distance: float = 0.0,
        max_distance: float = 0.0,
        anneal_progress: float = 0.0,
        min_distance: float = 0.0,
        elapsed_sec: float = 0.0,
        t_x_minutes: float = 60.0,
        # K-Scheduler centrality (normalized 0-1 from the katz arm)
        katz_energy: float = 0.0,
    ) -> float:
        """Compute the energy score for a queue entry.

        Args:
            exec_us: Execution time of this seed (microseconds).
            avg_exec_us: Average execution time across all seeds.
            bitmap_size: Number of bitmap bytes this seed covers.
            avg_bitmap_size: Average bitmap size across all seeds.
            handicap: Late-discovery bonus (decays by 1 each cycle).
            depth: Mutation depth from original seed.
            fuzz_level: How many times this seed has been fuzzed.
            n_fuzz: Number of times this seed's path has been hit.
            total_execs: Total executions across all seeds.
            tc_ref: Number of bitmap bytes where this seed is the top contender.
            favored: Whether this seed is in the favored set.
            max_depth: Maximum depth in the queue.
            new_edges: Number of new edges this seed discovered.
            time_added: Timestamp when seed was added (epoch seconds).
            now: Current timestamp (epoch seconds).
            input_size: Size of the seed input in bytes.
            select_count: How many times this seed has been selected.
            child_count: Number of children produced by this seed.
            cmp_progress: CMP solving progress score.
            rare_edge_count: Number of rare edges hit by this seed.
            timed_out: Whether this seed caused a timeout.
            input_entropy: Shannon entropy of input (0-100, -1 = unknown).
            max_cov: Maximum coverage across all seeds.
            hw_instructions: Hardware instruction count delta (from perf counters).
            hw_branches: Hardware branch count delta (from perf counters).

        Returns:
            Energy score (1 to max_mult * 100).
        """
        perf_score = 100.0

        # Speed adjustment (skip for rare schedule)
        if self.schedule != "rare" and avg_exec_us > 0:
            perf_score *= self._speed_factor(exec_us, avg_exec_us)

        # Bitmap size adjustment
        if avg_bitmap_size > 0:
            perf_score *= self._bitmap_factor(bitmap_size, avg_bitmap_size)

        # Handicap adjustment
        if handicap >= 4:
            perf_score *= 4.0
        elif handicap > 0:
            perf_score *= 2.0

        # Depth adjustment
        perf_score *= self._depth_factor(depth)

        # ── Hardware perf factors (apply to ALL schedules) ──────────────
        if hw_instructions > 0:
            # Update running average (EMA)
            if self._avg_hw_instructions == 0:
                self._avg_hw_instructions = float(hw_instructions)
            else:
                self._avg_hw_instructions = (
                    self._avg_hw_instructions * (1 - self._hw_avg_alpha)
                    + hw_instructions * self._hw_avg_alpha
                )
            # Boost inputs that execute more instructions than average
            if self._avg_hw_instructions > 0:
                ratio = hw_instructions / max(1, self._avg_hw_instructions)
                if ratio > 2.0:
                    perf_score *= 2.0
                elif ratio > 1.5:
                    perf_score *= 1.5
                elif ratio > 1.0:
                    perf_score *= 1.2
                elif ratio < 0.3:
                    perf_score *= 0.5

        if hw_branches > 0:
            if self._avg_hw_branches == 0:
                self._avg_hw_branches = float(hw_branches)
            else:
                self._avg_hw_branches = (
                    self._avg_hw_branches * (1 - self._hw_avg_alpha)
                    + hw_branches * self._hw_avg_alpha
                )
            if self._avg_hw_branches > 0:
                ratio = hw_branches / max(1, self._avg_hw_branches)
                if ratio > 2.0:
                    perf_score *= 1.8
                elif ratio > 1.5:
                    perf_score *= 1.4
                elif ratio > 1.0:
                    perf_score *= 1.15
                elif ratio < 0.3:
                    perf_score *= 0.6

        # Schedule-specific frequency adjustment
        if self.schedule == "rare":
            # RARE: additive tc_ref bonus + multiplicative penalty
            perf_score += self.rare_bonus(tc_ref)
            penalty = self._rare_factor(n_fuzz, total_execs, tc_ref)
            perf_score *= penalty
        elif self.schedule in ("fast", "lin", "quad"):
            factor = self._schedule_factor(
                fuzz_level=fuzz_level,
                n_fuzz=n_fuzz,
                total_execs=total_execs,
                tc_ref=tc_ref,
                favored=favored,
                max_depth=max_depth,
                depth=depth,
            )
            if factor > self.max_factor:
                factor = self.max_factor
            perf_score *= factor / self.power_beta
        elif self.schedule == "mopt":
            factor = self._mopt_factor(max_depth, depth)
            perf_score *= factor
        elif self.schedule == "coe":
            if mean_log_n_fuzz > 0 and self.coe_skip(n_fuzz, mean_log_n_fuzz, favored):
                # COE skip: floor energy (documented domain is 1..max_mult*100).
                # Returning the max here would give skipped seeds the most
                # mutations of any seed — the exact inverse of the schedule.
                return 1.0
            factor = self._fast_factor(fuzz_level, n_fuzz, favored)
            if factor > self.max_factor:
                factor = self.max_factor
            perf_score *= factor / self.power_beta
        elif self.schedule == "go":
            perf_score *= self._go_factor(avg_distance, max_distance, anneal_progress)
        elif self.schedule == "aflgo":
            perf_score *= self._aflgo_factor(
                avg_distance,
                max_distance,
                min_distance,
                elapsed_sec,
                t_x_minutes,
                self.aflgo_cooling,
            )
        elif self.schedule == "entropic":
            perf_score *= self._entropic_factor(rare_edge_count, tc_ref)
        elif self.schedule == "katz":
            # Energy = seed's normalized Katz centrality, clamped to
            # [1, max_mult]: one high-centrality seed cannot starve the
            # queue (same clamp rationale as the paper's AFL integration).
            norm = min(max(katz_energy, 0.0), 1.0)
            perf_score *= 1.0 + norm * (self.max_mult - 1)

        # ── Honggfuzz power factors (applied on top of schedule) ────────
        perf_score *= self._honggfuzz_factors(
            new_edges=new_edges,
            time_added=time_added,
            now=now,
            bitmap_size=bitmap_size,
            input_size=input_size,
            child_count=child_count,
            cmp_progress=cmp_progress,
            rare_edge_count=rare_edge_count,
            select_count=select_count,
            depth=depth,
            timed_out=timed_out,
            input_entropy=input_entropy,
            bitmap_size_avg=avg_bitmap_size,
            max_cov=max_cov,
        )

        # Clamp
        perf_score = max(1.0, min(perf_score, self.max_mult * 100.0))

        return perf_score

    def _honggfuzz_factors(
        self,
        new_edges: int = 0,
        time_added: float = 0.0,
        now: float = 0.0,
        bitmap_size: int = 0,
        input_size: int = 0,
        child_count: int = 0,
        cmp_progress: int = 0,
        rare_edge_count: int = 0,
        select_count: int = 0,
        depth: int = 0,
        timed_out: bool = False,
        input_entropy: float = -1.0,
        bitmap_size_avg: int = 0,
        max_cov: int = 0,
    ) -> float:
        """Compute honggfuzz-style multiplicative energy factors.

        Ported from honggfuzz power.c:power_calculateEnergy.
        All factors are multiplicative (1.0 = no change).

        Returns:
            Combined multiplicative factor.
        """
        factor = 1.0

        # Novelty decay: new-edge bonus that decays over 10 minutes
        if new_edges > 0 and now > 0 and time_added > 0:
            age_mins = (now - time_added) / 60.0
            decay = 0 if age_mins < 10 else min(int(age_mins / 10), 6)
            boost = min(new_edges, 8)
            if boost > decay:
                factor *= 2.0 ** (boost - decay)

        # Density: coverage-per-byte ratio boost
        if input_size > 0 and bitmap_size > 0:
            density = (bitmap_size * 100) / input_size
            if density > 200:
                factor *= 2.0
            elif density > 50:
                factor *= 1.5

        # Fertility: logarithmic boost for inputs that produced children
        if child_count > 0:
            factor *= (8 + min(int(math.log2(child_count + 1)), 8)) / 8.0

        # Freshness: time-based boost
        if now > 0 and time_added > 0:
            age_secs = now - time_added
            if age_secs < 60:
                factor *= 4.0  # <60s: 4x
            elif age_secs < 300:
                factor *= 2.0  # <5min: 2x
            elif age_secs > 3600 and child_count == 0:
                factor *= 0.5  # >60min with no children: 0.5x

        # Size penalty: larger inputs get logarithmic penalty
        if input_size > 1024:
            log_size = int(math.log2(input_size))
            if log_size > 10:
                factor /= 2.0 ** min(log_size - 10, 4)

        # CMP progress boost
        if cmp_progress > 0:
            cmp_boost = min(cmp_progress // 8, 4)
            if cmp_boost > 0:
                factor *= (4 + cmp_boost) / 4.0

        # Rare edge bonus
        if rare_edge_count > 0:
            rare_boost = min(rare_edge_count, 8)
            factor *= (8 + rare_boost) / 8.0

        # Diminishing returns: inputs selected many times yield less
        if select_count > 100:
            penalty = min(int(math.log2(select_count / 100)), 3)
            factor /= 2.0**penalty

        # Entropy: penalize random blobs and very sparse data
        if 0 <= input_entropy <= 100:
            if input_entropy > ENTROPY_RANDOM_PCT:
                factor *= 0.5  # High entropy (compressed/random)
            elif input_entropy < ENTROPY_SPARSE_PCT:
                factor *= 0.5  # Very low entropy (zeros)
            elif input_entropy < ENTROPY_STRUCTURED_PCT:
                factor *= 1.5  # Text/structured data boost

        # Timeout: heavy penalty
        if timed_out:
            factor /= 32.0

        return max(0.01, factor)

    def _speed_factor(self, exec_us: int, avg_exec_us: int) -> float:
        """Speed-based multiplier: fast seeds get more energy."""
        if exec_us * 0.1 > avg_exec_us:
            return 0.10
        if exec_us * 0.25 > avg_exec_us:
            return 0.25
        if exec_us * 0.5 > avg_exec_us:
            return 0.50
        if exec_us * 0.75 > avg_exec_us:
            return 0.75
        if exec_us * 4 < avg_exec_us:
            return 3.00
        if exec_us * 3 < avg_exec_us:
            return 2.00
        if exec_us * 2 < avg_exec_us:
            return 1.50
        return 1.00

    def _bitmap_factor(self, bitmap_size: int, avg_bitmap_size: int) -> float:
        """Coverage-based multiplier: better coverage gets more energy."""
        if bitmap_size * 0.3 > avg_bitmap_size:
            return 3.0
        if bitmap_size * 0.5 > avg_bitmap_size:
            return 2.0
        if bitmap_size * 0.75 > avg_bitmap_size:
            return 1.5
        if bitmap_size * 3 < avg_bitmap_size:
            return 0.25
        if bitmap_size * 2 < avg_bitmap_size:
            return 0.50
        if bitmap_size * 1.5 < avg_bitmap_size:
            return 0.75
        return 1.0

    def _depth_factor(self, depth: int) -> float:
        """Depth-based multiplier: deeper mutations get more energy."""
        if depth <= 3:
            return 1.0
        if depth <= 7:
            return 2.0
        if depth <= 13:
            return 3.0
        if depth <= 25:
            return 4.0
        return 5.0

    def _schedule_factor(
        self,
        fuzz_level: int,
        n_fuzz: int,
        total_execs: int,
        tc_ref: int,
        favored: bool,
        max_depth: int,
        depth: int,
    ) -> float:
        """Schedule-specific frequency factor."""
        if self.schedule == "base":
            return 1.0

        if self.schedule == "fast":
            return self._fast_factor(fuzz_level, n_fuzz, favored)

        if self.schedule == "coe":
            return self._coe_factor(fuzz_level, n_fuzz, favored)

        if self.schedule == "rare":
            return self._rare_factor(n_fuzz, total_execs, tc_ref)

        if self.schedule == "mopt":
            return self._mopt_factor(max_depth, depth)

        if self.schedule == "lin":
            if not fuzz_level:
                return 1.0
            return fuzz_level / (n_fuzz + 1)

        if self.schedule == "quad":
            if not fuzz_level:
                return 1.0
            return (fuzz_level * fuzz_level) / (n_fuzz + 1)

        return 1.0

    def _fast_factor(self, fuzz_level: int, n_fuzz: int, favored: bool) -> float:
        """AFLFast's frequency-based energy.

        Rare seeds (low n_fuzz) get 4x; heavily-fuzzed get 0.4x.
        Favored seeds get a 1.15x bonus.
        """
        if not fuzz_level:
            return 1.0

        log_n = math.log2(max(n_fuzz, 1))

        if log_n <= 1:
            factor = 4.0
        elif log_n <= 3:
            factor = 3.0
        elif log_n <= 4:
            factor = 2.0
        elif log_n <= 5:
            factor = 1.0
        elif log_n <= 6:
            factor = 0.8 if not favored else 1.0
        elif log_n <= 7:
            factor = 0.6 if not favored else 1.0
        else:
            factor = 0.4 if not favored else 1.0

        if favored:
            factor *= 1.15

        return factor

    def _coe_factor(self, fuzz_level: int, n_fuzz: int, favored: bool) -> float:
        """Cut-Off Exponential: skip seeds above the mean fuzz count.

        Seeds with log2(n_fuzz) > mean(log2(n_fuzz)) of all seeds get
        factor=0 (skipped), unless they are favored.
        """
        if not fuzz_level:
            return 1.0

        # Note: full COE requires mean computation across all queue entries.
        # Here we use the individual seed's n_fuzz as a proxy.
        # The caller should compute the mean and pass it via a wrapper.
        # For standalone use, we fall through to FAST behavior.
        return self._fast_factor(fuzz_level, n_fuzz, favored)

    def coe_skip(self, n_fuzz: int, mean_log_n_fuzz: float, favored: bool) -> bool:
        """Check if a seed should be skipped under COE scheduling.

        Args:
            n_fuzz: This seed's fuzz count.
            mean_log_n_fuzz: Mean of log2(n_fuzz) across all queue entries.
            favored: Whether this seed is favored.

        Returns:
            True if the seed should be skipped (factor=0).
        """
        if n_fuzz <= 0:
            return False
        return bool(math.log2(n_fuzz) > mean_log_n_fuzz and not favored)

    def _rare_factor(self, n_fuzz: int, total_execs: int, tc_ref: int) -> float:
        """RARE schedule: boost seeds that own rare edges.

        score += tc_ref * 10  (more rare-edge ownership = more energy)
        score *= (1 - n_fuzz / total_execs)  (penalize over-fuzzed)
        """
        if total_execs <= 0:
            return 1.0

        # tc_ref bonus is additive, handled in the score() method directly.
        # Here we return the multiplicative penalty factor.
        penalty = 1.0 - (n_fuzz / total_execs)
        return max(0.01, penalty)

    def rare_bonus(self, tc_ref: int) -> float:
        """The additive tc_ref bonus for RARE scheduling.

        Returns the value to ADD to the base score (tc_ref * 10).
        """
        return tc_ref * 10.0

    def _entropic_factor(self, rare_edge_count: int, tc_ref: int) -> float:
        """ENTROPIC schedule: energy scales with log(rare-feature count).

        libFuzzer's entropic schedule weighs an input by the Shannon
        entropy of its feature-frequency distribution -- inputs that hit
        rarer features earn more energy. We approximate this with the
        rare-feature-ownership counts already tracked for RARE/honggfuzz
        scoring (``tc_ref``: bitmap bytes this seed is the top contender
        for; ``rare_edge_count``: rare edges hit), taking whichever signal
        is available and largest. log2 rather than linear scaling (as
        RARE's additive tc_ref bonus uses) keeps a single very-rare seed
        from dominating the queue the way linear scaling would.
        """
        rare = max(rare_edge_count, tc_ref)
        if rare <= 0:
            return 1.0
        return 1.0 + math.log2(1 + rare)

    def _mopt_factor(self, max_depth: int, depth: int) -> float:
        """MMOPT: boost recent entries (close to max_depth)."""
        if max_depth - depth < 5:
            return 2.0
        return 1.0

    def _aflgo_factor(
        self,
        avg_distance: float,
        max_distance: float,
        min_distance: float,
        elapsed_sec: float,
        t_x_minutes: float,
        cooling: str,
    ) -> float:
        """AFLGo's exact distance-annealed power factor (afl-fuzz.c
        calculate_score, AFLGO_IMPL block).

        Temperature T follows the chosen cooling schedule over
        t_x_minutes (AFLGo's ``-c``); progress = elapsed/(t_x*60).
        normalized_d = (d − min)/(max − min) over the live queue, and
        p = (1 − normalized_d)(1 − T) + 0.5T.  The factor
        2^(2·log2(32)·(p−0.5)) is symmetric around 1.0: at the start
        (T=1) every seed is treated equally; late in the campaign
        (T≈0) near-target seeds get up to 32× energy and far seeds as
        little as 1/32×.  avg_distance < 0 means no distance data.

        Returns:
            Multiplicative factor (1.0 when no distance data).
        """
        if avg_distance < 0 or max_distance <= 0 or t_x_minutes <= 0:
            return 1.0
        progress = elapsed_sec / (t_x_minutes * 60.0)
        T = self._cooling_temperature(progress, cooling)
        if max_distance == min_distance:
            normalized_d = 0.0
        else:
            normalized_d = min(
                max((avg_distance - min_distance) / (max_distance - min_distance), 0.0), 1.0
            )
        p = (1.0 - normalized_d) * (1.0 - T) + 0.5 * T
        return pow(2.0, 2.0 * math.log2(self.max_factor) * (p - 0.5))

    @staticmethod
    def _cooling_temperature(progress: float, cooling: str) -> float:
        """AFLGo's temperature cooling schedules (afl-fuzz.c:4923-4947).

        All four schedules coincide at progress=1 (T=1/20) by
        construction of the log constant (exp(19/2) − 1).
        """
        progress = max(0.0, progress)
        if cooling == "log":
            # alpha = 2 and exp(19/2) − 1 = 13358.7268297
            return 1.0 / (1.0 + 2.0 * math.log(1.0 + progress * 13358.7268297))
        if cooling == "lin":
            return 1.0 / (1.0 + 19.0 * progress)
        if cooling == "quad":
            return 1.0 / (1.0 + 19.0 * progress * progress)
        return 1.0 / pow(20.0, progress)  # exp

    def _go_factor(self, avg_distance: float, max_distance: float, anneal_progress: float) -> float:
        """AFLGo-style distance-annealed energy multiplier.

        During exploitation phase (anneal_progress > 0), seeds near
        the target get exponentially more energy.  During exploration
        phase (anneal_progress ≈ 0), all seeds get uniform energy.

        Formula: energy *= exp(β · (1 - norm_dist))
        where β = anneal_progress * 5.0  (grows from 0 → 5 as campaign matures)
        and norm_dist = avg_distance / max_distance (0 = at target, 1 = farthest)

        Returns:
            Multiplicative factor (1.0 when no distance info, up to 100x cap).
        """
        if self.schedule != "go" or anneal_progress < 0.01 or max_distance <= 0:
            return 1.0
        # avg_distance >= 0 is valid (0 = at target = best case).
        # avg_distance < 0 means no distance data available.
        if avg_distance < 0:
            return 1.0
        norm_dist = min(avg_distance / max_distance, 1.0)
        beta = anneal_progress * 5.0
        bonus = math.exp(beta * (1.0 - norm_dist))
        return min(bonus, 100.0)


def compute_mean_log_n_fuzz(n_fuzz_values: list[int]) -> float:
    """Compute mean of log2(n_fuzz) across queue entries (for COE).

    Args:
        n_fuzz_values: List of n_fuzz counts for each queue entry.

    Returns:
        Mean of log2(n_fuzz) for entries with n_fuzz > 0.
    """
    log_values = [math.log2(n) for n in n_fuzz_values if n > 0]
    if not log_values:
        return 0.0
    return sum(log_values) / len(log_values)
