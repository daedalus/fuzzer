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
"""

import math


class SeedScorer:
    """Compute energy scores for queue entries using various power schedules.

    Each schedule modifies a base score (from speed/size/depth) by a
    frequency-based factor. The factor depends on how often a seed has
    been fuzzed relative to others.

    Args:
        schedule: One of 'base', 'fast', 'coe', 'rare', 'mopt', 'lin', 'quad'.
        max_mult: Maximum havoc multiplier (default 16).
    """

    SCHEDULES = ("base", "fast", "coe", "rare", "mopt", "lin", "quad")

    def __init__(self, schedule: str = "base", max_mult: int = 16):
        if schedule not in self.SCHEDULES:
            raise ValueError(f"Unknown schedule: {schedule!r}. Use one of {self.SCHEDULES}")
        self.schedule = schedule
        self.max_mult = max_mult
        self.max_factor = 32.0
        self.power_beta = 1.0

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
            # COE: fall through to FAST behavior for individual scoring
            # Use coe_skip() for the cut-off check
            factor = self._fast_factor(fuzz_level, n_fuzz, favored)
            if factor > self.max_factor:
                factor = self.max_factor
            perf_score *= factor / self.power_beta

        # Clamp
        perf_score = max(1.0, min(perf_score, self.max_mult * 100.0))

        return perf_score

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
        log_n = math.log2(n_fuzz)
        if log_n > mean_log_n_fuzz and not favored:
            return True
        return False

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

    def _mopt_factor(self, max_depth: int, depth: int) -> float:
        """MMOPT: boost recent entries (close to max_depth)."""
        if max_depth - depth < 5:
            return 2.0
        return 1.0


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
