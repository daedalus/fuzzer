"""Monte Carlo scheduler: Thompson sampling bandit + CEM byte distribution.

Uses JS divergence to adaptively control CEM refit frequency:
- JS → 0 after refit: distribution stabilized, refit less often
- JS stays high: elite set still shifting, refit more aggressively

Also tracks Brier score (binary CRPS) for bandit calibration diagnostics.
"""

import collections
import logging
import math
import random
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fuzzer_tool.core.allan_variance import DispersionIndex
from fuzzer_tool.core.cycle_detect import cesaro_average, floyd_detect
from fuzzer_tool.core.running_stats import (
    RunningMoments,
    kelly_fraction,
    sharpe_ratio,
)

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

log = logging.getLogger(__name__)

# Lower clamp for Beta(alpha, beta) parameters passed to init_arm(): keeps
# random.betavariate() numerically stable for malformed/degenerate priors
# (e.g. a caller passing 0 or a negative value) without silently overriding
# intentionally weak-but-valid priors above this threshold.
MIN_BETA_PARAM = 1e-6


@dataclass(frozen=True)
class StationaryDiagnostics:
    """How `stationary_distribution`'s power iteration actually ended.

    `diff < tol` and "the chain is periodic" are indistinguishable from
    the raw loop alone — both look like "still running at max_iter". This
    disambiguates them so a caller can tell "give it more iterations"
    apart from "it will never converge, this is the correct answer".

    Attributes:
        converged: True if the L1-tolerance check fired before max_iter.
        periodic: True if non-convergence was diagnosed as an exact limit
            cycle (via Floyd's algorithm) rather than plain slow mixing.
            When True, the returned distribution is the Cesaro average
            over one full period, not a single non-converged iterate.
        period: Cycle length. 1 when not periodic.
        iterations: Power-iteration rounds run before the tol check (or
            the cycle-detection phase, if that's how the loop ended).
        cycle_checked: True if Floyd's algorithm actually ran (i.e.
            `detect_cycles=True` was passed and power iteration failed
            to converge). False when the check was skipped entirely —
            either because iteration converged normally, or because
            cycle detection is opt-in and wasn't requested. Lets a
            caller distinguish "checked, no cycle" (periodic=False,
            cycle_checked=True) from "never checked"
            (periodic=False, cycle_checked=False).
    """

    converged: bool
    periodic: bool
    period: int
    iterations: int
    cycle_checked: bool = False


class MonteCarloScheduler:
    """Thompson sampling bandit for mutation ops + CEM byte distribution.

    Combines two Monte Carlo methods:
    1. Thompson sampling to select which mutation operator to use
    2. Cross-entropy method to learn per-position byte distributions

    Args:
        elite_frac: Fraction of elite set to use when fitting CEM distribution.
        refit_interval: How often (in executions) to refit the CEM distribution.

    Examples:
        >>> mc = MonteCarloScheduler()
        >>> mc.init_arm("bit_flip")
        >>> mc.init_arm("byte_flip")
        >>> op = mc.select_op(["bit_flip", "byte_flip"])
        >>> mc.record(op, success=True)
    """

    ELITE_MAX = 200
    # Declares that init_arm() accepts an informative (prior_alpha, prior_beta)
    # override, unlike MOptScheduler/ReplicatorScheduler/EloTracker which use
    # non-Bayesian internal representations (particle positions, population
    # simplex, Elo ratings).
    supports_priors = True

    def __init__(
        self,
        elite_frac: float = 0.1,
        refit_interval: int = 1000,
        pairwise_blend: float = 0.0,
        arm_decay: float = 0.999,
        decay_interval: int = 100,
        hierarchical_pooling: float = 0.0,
        cem_dirichlet_concentration: float = 0.0,
    ):
        self._hierarchical_pooling = max(0.0, min(1.0, hierarchical_pooling))
        self._cem_dirichlet_concentration = cem_dirichlet_concentration
        self.arm_alpha: dict[str, float] = {}
        self.arm_beta: dict[str, float] = {}
        self._pooled_successes = 0.0
        self._pooled_failures = 0.0
        self.arm_decay = arm_decay
        self.decay_interval = decay_interval
        self.elite_frac = elite_frac
        self.base_refit_interval = refit_interval
        self.refit_interval = refit_interval
        self.execs_since_refit = 0
        self.elite_set: list[tuple[int, bytes]] = []
        self.byte_freq: dict[int, dict[int, int]] = {}
        self._prev_byte_freq: dict[int, dict[int, int]] = {}
        self.cem_fitted = False
        self.last_js_divergence: float = 0.0
        # No-change reference for last_js_divergence, estimated at the current
        # sample size. Published so the adaptation can be read, not just felt.
        self.last_js_null_p95: float = 0.0
        self.last_js_null_p99: float = 0.0
        # Brier score tracking for bandit calibration diagnostics
        self._brier_predictions: collections.deque = collections.deque(maxlen=500)
        # Success history for covariance computation
        self._op_success_history: collections.deque = collections.deque(maxlen=2000)

        # Pairwise transition matrix: P(next_op | prev_op)
        # transition_counts[prev][next] = discoveries from (prev, next) pairs
        # transition_total[prev] = total attempts where next followed prev
        self.transition_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.transition_total: dict[str, int] = defaultdict(int)
        self._prev_op: str | None = None
        # Blend factor: 0.0 = pure Thompson, 1.0 = pure pairwise
        self.pairwise_blend = pairwise_blend

        # Thompson draw cache: op -> (a, b, draw). A cached draw is valid
        # while the arm's effective posterior params are unchanged; record()
        # and periodic decay change them, so stale entries are detected by
        # key mismatch. Draws are additionally force-refreshed every N
        # selects so an arm whose posterior never moves cannot keep a stale
        # lucky draw forever and starve the other arms.
        self._thompson_draw_cache: dict[str, tuple[float, float, float]] = {}
        self._draw_refresh_interval = 16
        self._selects_since_refresh = 0

        # Monotonic record() counter driving periodic arm decay. Distinct
        # from len(_op_success_history), which is capped at maxlen=2000 and
        # would otherwise decay on every call once 2000 % interval == 0.
        self._record_count = 0

        # Per-operator dispersion index for non-stationarity detection.
        # When D > 1.5, the operator's success process is bursty (non-i.i.d.),
        # meaning the Beta posterior is overconfident — older observations
        # should decay faster.
        self._op_dispersion: dict[str, DispersionIndex] = {}

        # Per-operator reward moments for Sharpe/Kelly risk-adjusted selection.
        # Tracks the scalar cost-adjusted reward stream per arm so that
        # variance-normalized quality scores can be computed on demand.
        self._op_reward_moments: dict[str, RunningMoments] = {}
        # Blend weight for Sharpe/Kelly vs Thompson sampling.
        # 0.0 = pure Thompson (default), 1.0 = pure Sharpe/Kelly score.
        self._sharpe_kelly_blend: float = 0.0

        # Floyd cycle-detection stats for stationary_distribution(). Cycle
        # detection itself is opt-in (detect_cycles=False by default) since
        # it costs up to 20 * max_iter extra `step` calls on top of a
        # failed power iteration; these counters only move when a caller
        # actually passes detect_cycles=True, so they double as a record
        # of whether the check has ever run.
        self.cycle_checks: int = 0
        self.cycle_detections: int = 0
        self.last_cycle_period: int = 0
        self.max_cycle_period: int = 0

    def _record_cycle_check(self, cycle) -> None:
        """Update cycle-detection counters after an opt-in Floyd check.

        `cycle` is the `CycleResult | None` returned by `floyd_detect`.
        """
        self.cycle_checks += 1
        if cycle is not None and cycle.period > 1:
            self.cycle_detections += 1
            self.last_cycle_period = cycle.period
            self.max_cycle_period = max(self.max_cycle_period, cycle.period)

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register a mutation operator arm with a Beta prior.

        Defaults to the uninformative Beta(1, 1) prior. Callers with prior
        knowledge about an operator's likely usefulness (e.g. static target
        profiling indicating a specific file format) can pass a stronger
        prior to bias early Thompson sampling before any evidence has been
        observed. A no-op if the arm is already registered — the prior only
        applies at first registration and is never overwritten by later
        calls, matching the existing idempotent behavior of this method.

        Args:
            name: Name of the mutation operator.
            prior_alpha: Prior alpha (successes + 1). Must be > 0.
            prior_beta: Prior beta (failures + 1). Must be > 0.
        """
        if name not in self.arm_alpha:
            self.arm_alpha[name] = max(prior_alpha, MIN_BETA_PARAM)
            self.arm_beta[name] = max(prior_beta, MIN_BETA_PARAM)

    def _get_effective_params(self, op: str) -> tuple[float, float]:
        """Get posterior parameters with optional hierarchical shrinkage."""
        a = self.arm_alpha.get(op, 1.0)
        b = self.arm_beta.get(op, 1.0)
        if self._hierarchical_pooling > 0:
            h = self._hierarchical_pooling
            pooled_total = self._pooled_successes + self._pooled_failures
            if pooled_total > 0:
                pooled_alpha = 1.0 + self._pooled_successes
                pooled_beta = 1.0 + self._pooled_failures
                a = (1 - h) * a + h * pooled_alpha
                b = (1 - h) * b + h * pooled_beta
        return a, b

    def select_op(self, ops: list[str], prev_op: str | None = None) -> str:
        """Select mutation operator via Thompson sampling with pairwise transitions.

        When pairwise_blend > 0 and prev_op has transition data, blends
        the unconditional Thompson sample with a pairwise-conditional sample
        that favors operators that historically followed prev_op.

        When sharpe_kelly_blend > 0, risk-normalized scores are blended with
        Thompson draws so high-variance lottery-ticket arms are down-weighted.

        Args:
            ops: Available mutation operators.
            prev_op: The operator used in the previous mutation step (for
                pairwise transition weighting).

        Returns:
            Name of the selected operator.
        """
        # Unconditional Thompson sample for each op. Draws are cached per
        # arm and reused while the effective posterior params are unchanged,
        # so a selection doesn't re-pay 83 betavariate draws when the
        # posterior is piecewise-constant. Periodically all arms are forced
        # to redraw to prevent a frozen stale draw from dominating forever.
        self._selects_since_refresh += 1
        force_refresh = self._selects_since_refresh >= self._draw_refresh_interval
        if force_refresh:
            self._selects_since_refresh = 0
        thompson_vals: dict[str, float] = {}
        for op in ops:
            a, b = self._get_effective_params(op)
            cached = self._thompson_draw_cache.get(op)
            if not force_refresh and cached is not None and cached[0] == a and cached[1] == b:
                thompson_vals[op] = cached[2]
            else:
                draw = random.betavariate(a, b)
                self._thompson_draw_cache[op] = (a, b, draw)
                thompson_vals[op] = draw

        # Sharpe/Kelly blend: when enabled, risk-normalized scores replace
        # or supplement the Thompson draws so high-variance lottery-ticket
        # arms are down-weighted relative to consistent performers.
        if self._sharpe_kelly_blend > 0:
            raw_sk: dict[str, float] = {}
            for op in ops:
                rm = self._op_reward_moments.get(op)
                if rm is None or rm.count < 3:
                    raw_sk[op] = 0.0
                    continue
                s = sharpe_ratio(rm.mean, rm.stddev)
                k = kelly_fraction(rm.mean, rm.variance)
                raw_sk[op] = s + k
            sk_min = min(raw_sk.values())
            sk_max = max(raw_sk.values())
            if math.isinf(sk_max):
                # One or more arms have infinite SK (zero variance, positive
                # mean). Those arms get norm=1; everything else gets 0.
                sk_norm = {op: (1.0 if math.isinf(v) else 0.0) for op, v in raw_sk.items()}
            else:
                rng = sk_max - sk_min
                if rng > 0.0:
                    sk_norm = {op: (v - sk_min) / rng for op, v in raw_sk.items()}
                else:
                    sk_norm = {op: 0.5 for op in raw_sk}
            blend = self._sharpe_kelly_blend
            for op in ops:
                thompson_vals[op] = blend * sk_norm[op] + (1 - blend) * thompson_vals[op]

        # If no pairwise data or blend is zero, use current scores (Thompson
        # or SK-blended) directly.
        if self.pairwise_blend <= 0 or prev_op is None or prev_op not in self.transition_total:
            return max(ops, key=lambda o: thompson_vals[o])

        # Pairwise score: Dirichlet-Multinomial over transition counts
        # With uniform prior (alpha=1), score = count + 1
        total = self.transition_total[prev_op]
        pair_scores = {}
        for op in ops:
            count = self.transition_counts[prev_op].get(op, 0)
            pair_scores[op] = (count + 1) / (total + len(ops))

        # Blend: w * pair + (1-w) * thompson (or SK-blended Thompson)
        w = self.pairwise_blend
        for op in ops:
            thompson_vals[op] = w * pair_scores[op] + (1 - w) * thompson_vals[op]

        # NOTE: `self._prev_op` is deliberately NOT written here. It is the
        # *previously recorded* operator and is advanced by record(), which is
        # the only place that can pair a predecessor with a known outcome.
        # Writing it here set it to the operator being selected, so by the time
        # record() ran for that same operator the `_prev_op != name` guard
        # rejected the pair — and because this line sits after the pure-Thompson
        # early return, it never executed until transitions already existed.
        return max(ops, key=lambda o: thompson_vals[o])

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome for a mutation operator arm.

        Applies exponential decay to all arms periodically (every 100 calls),
        giving recent evidence more weight (non-stationary bandit).

        Args:
            name: Name of the mutation operator.
            success: Whether the mutation produced an interesting result.
            weight: Reward weight (default 1.0). Surprisal-weighted calls
                pass a value in (0, 1] proportional to discovery rarity.
        """
        self._record_count += 1
        self._op_success_history.append((name, success))

        arm_alpha = self.arm_alpha
        arm_beta = self.arm_beta

        # Decay periodically to avoid zeroing out alpha/beta
        if (
            self.arm_decay < 1.0
            and self.decay_interval > 0
            and self._record_count % self.decay_interval == 0
        ):
            for k in arm_alpha:
                arm_alpha[k] *= self.arm_decay
            for k in arm_beta:
                arm_beta[k] *= self.arm_decay

        if success:
            arm_alpha[name] = arm_alpha.get(name, 1.0) + weight
            self._pooled_successes += weight
        else:
            arm_beta[name] = arm_beta.get(name, 1.0) + 1
            self._pooled_failures += 1.0

        # Per-operator dispersion index tracking (for diagnostics only).
        # Note: for binary success data D = 1-p <= 1 mathematically, so
        # D on raw binary is always <= 1. Tracked for diagnostic display.
        if name not in self._op_dispersion:
            self._op_dispersion[name] = DispersionIndex(window=200)
        self._op_dispersion[name].update(float(success))

        # Per-operator reward moments for Sharpe/Kelly selection.
        # Tracks the scalar cost-adjusted weight so risk-adjusted scores
        # can be computed without another pass over history.
        self._op_reward_moments.setdefault(name, RunningMoments()).update(weight)

        # Update pairwise transition matrix on success. `_prev_op` is the
        # operator recorded immediately before this one; record() is called in
        # selection order, so consecutive recorded operators are the observed
        # chain. Advancing it unconditionally at the end is what lets the
        # matrix bootstrap: it must not depend on a branch that itself
        # requires a populated matrix.
        if success and self._prev_op is not None and self._prev_op != name:
            self.transition_counts[self._prev_op][name] += 1
            self.transition_total[self._prev_op] += 1
        self._prev_op = name

    def record_brier(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record a prediction-outcome pair for Brier score diagnostics.

        The predicted probability is the Beta distribution mean for this arm
        at the time of selection. The outcome is *binary*: the event being
        predicted is "did this operator succeed", and the Beta mean is a
        probability for exactly that event.

        `weight` is accepted for call-site symmetry with record() but is
        deliberately not used as the outcome. It carries the surprisal- and
        cost-adjusted reward, which is unbounded above; feeding it in here
        produced outcomes far outside [0, 1] and a "Brier score" in the tens
        (35.57 on the ffmpeg_read_nosan run) for a statistic bounded by 1.

        Brier score = mean((predicted - actual)²) — lower is better.
        """
        a = self.arm_alpha.get(name, 1.0)
        b = self.arm_beta.get(name, 1.0)
        predicted = a / (a + b)  # Beta mean = expected success probability
        outcome = 1.0 if success else 0.0
        self._brier_predictions.append((predicted, outcome))

    def brier_score(self) -> float:
        """Mean Brier score over recent predictions.

        Returns 0.0 if no data. Lower is better calibrated:
        - 0.0  = perfect calibration
        - 0.25 = uninformative (constant 0.5 predictor)
        - 1.0  = worst possible (confident and always wrong)
        """
        if not self._brier_predictions:
            return 0.0
        return sum((p - o) ** 2 for p, o in self._brier_predictions) / len(self._brier_predictions)

    def calibration_report(self) -> dict[str, tuple[float, float, int]]:
        """Compute per-bin calibration: among predictions in [0,0.1), [0.1,0.2), etc.,
        what fraction actually succeeded?

        Returns {bin_label: (mean_predicted, observed_frequency, n_samples)} for
        bins with enough data. n_samples is part of the tuple because a bin's
        observed frequency is meaningless without knowing how many predictions
        landed in it — the report previously declared a Samples column and then
        had nothing to put in it.
        """
        if not self._brier_predictions:
            return {}
        bins: dict[int, list[tuple[float, float]]] = {}
        for pred, outcome in self._brier_predictions:
            b = min(int(pred * 10), 9)
            bins.setdefault(b, []).append((pred, outcome))
        report = {}
        for b, pairs in sorted(bins.items()):
            if len(pairs) < 5:
                continue
            mean_pred = sum(p for p, _ in pairs) / len(pairs)
            mean_actual = sum(o for _, o in pairs) / len(pairs)
            report[f"{b * 10}-{b * 10 + 10}%"] = (mean_pred, mean_actual, len(pairs))
        return report

    # ------------------------------------------------------------------
    # Sharpe / Kelly risk-adjusted selection helpers
    # ------------------------------------------------------------------

    def set_sharpe_kelly_blend(self, blend: float) -> None:
        """Set the Sharpe/Kelly blend weight for risk-adjusted selection.

        Args:
            blend: 0.0 = pure Thompson sampling (default), 1.0 = pure
                Sharpe/Kelly score.  Values between blend the two signals.
        """
        self._sharpe_kelly_blend = max(0.0, min(blend, 1.0))

    def sharpe(self, op: str) -> float:
        """Sharpe ratio for *op* from the recorded reward stream.

        Returns 0.0 if fewer than 3 observations have been recorded for
        this arm, or if the standard deviation is zero.
        """
        rm = self._op_reward_moments.get(op)
        if rm is None or rm.count < 3:
            return 0.0
        return sharpe_ratio(rm.mean, rm.stddev)

    def kelly_fraction(self, op: str) -> float:
        """Kelly criterion fraction for *op* from the recorded reward stream.

        Returns 0.0 if fewer than 3 observations have been recorded for
        this arm, or if the variance is zero.  Negative Kelly means the
        arm is a net loser; the clamp returns 0.0 rather than a negative
        fraction.
        """
        rm = self._op_reward_moments.get(op)
        if rm is None or rm.count < 3:
            return 0.0
        return kelly_fraction(rm.mean, rm.variance)

    def add_elite(self, data: bytes, score: int, temperature: float = 1.0) -> None:
        """Add an input to the elite set for CEM fitting.

        Uses Metropolis criterion: if the elite set is full and the new
        score is worse than the worst in the set, accept with probability
        exp(-ΔE/T) where ΔE = worst_score - score. This lets the elite
        set escape local optima early (high T) while converging greedily
        late (low T).

        Args:
            data: The input bytes.
            score: Quality score (higher is better).
            temperature: SA temperature (1.0 = fully exploratory, 0.0 = greedy).
        """
        if len(self.elite_set) < self.ELITE_MAX:
            self.elite_set.append((score, data))
            return

        self.elite_set.sort(key=lambda x: x[0])
        worst_score = self.elite_set[0][0]
        if score > worst_score:
            self.elite_set[0] = (score, data)
        elif temperature > 0.01:
            delta_e = worst_score - score
            acceptance = math.exp(-delta_e / temperature)
            if random.random() < acceptance:
                self.elite_set[0] = (score, data)

    def maybe_refit(self) -> None:
        """Refit the CEM byte distribution if enough data exists.

        After refitting, computes JS divergence between the new and previous
        byte_freq distributions to adaptively control refit frequency:
        - JS → 0: distribution stabilized → double the interval (up to 4x base)
        - JS > 0.1: still shifting → halve the interval (down to 0.25x base)
        """
        self.execs_since_refit += 1
        has_enough_elite = len(self.elite_set) >= 10
        if self.execs_since_refit < self.refit_interval and not has_enough_elite:
            return
        self.execs_since_refit = 0
        if not self.elite_set:
            return

        # Snapshot previous distribution for JS comparison.
        # Reference swap is safe: self.byte_freq is reassigned to a new
        # dict on the next line, and _prev_byte_freq is never mutated
        # (only iterated in _compute_js), so the old dict is untouched.
        self._prev_byte_freq = self.byte_freq

        n_elite = max(1, int(len(self.elite_set) * self.elite_frac))
        sorted_elite = sorted(self.elite_set, key=lambda x: x[0], reverse=True)
        elite = [d for _, d in sorted_elite[:n_elite]]
        self.byte_freq = {}
        for pos in range(max(len(d) for d in elite)):
            freq: dict[int, int] = {}
            for data in elite:
                if pos < len(data):
                    b = data[pos]
                    freq[b] = freq.get(b, 0) + 1
            self.byte_freq[pos] = freq
        self.cem_fitted = True

        # Learn Dirichlet concentration from elite entropy if enabled
        if self._cem_dirichlet_concentration > 0:
            self._learn_cem_concentration(elite)

        # Compute JS divergence and adapt refit interval
        self.last_js_divergence = self._compute_js()
        self._adapt_interval()

    def _learn_cem_concentration(self, elite: list[bytes]) -> None:
        """Learn the Dirichlet concentration parameter from the elite set.

        Estimates alpha_0 by matching the expected entropy of the Dirichlet
        distribution to the empirical entropy of the elite set.

        The Dirichlet(alpha_0, ..., alpha_0) has expected entropy:
            H(Dir) = ln(Gamma(256*alpha_0)) - 256*ln(Gamma(alpha_0))
                     - (256*alpha_0 - 1) * (psi(256*alpha_0) - psi(alpha_0))

        We solve for alpha_0 such that the empirical per-position entropy
        matches this expectation. When data is highly structured (low entropy),
        alpha_0 increases (stronger prior). When data is uniform (high entropy),
        alpha_0 stays near 1 (weak prior).

        Order-statistics connection (see order_statistics.py Part 3):
        The gaps (spacings) between sorted Uniform(0,1) draws are jointly
        distributed as Dirichlet(1,...,1) — equivalently, normalized i.i.d.
        Exponential(1) draws. This is the same Dirichlet family that CEM uses
        for per-byte distributions, but with a different categorical structure
        (over 256 byte values rather than over gap positions). The concentration
        alpha_0 in CEM controls how Dirichlet-like the byte distribution is:
        high alpha_0 → peaked prior (expects structured data), low alpha_0 →
        weak prior (expects near-uniform data).

        Stores the learned alpha_0 in self._cem_learned_alpha.
        """
        # Compute average empirical entropy across positions
        total_entropy = 0.0
        n_positions = 0
        for _, freq in self.byte_freq.items():
            total = sum(freq.values())
            if total < 2:
                continue
            n_positions += 1
            pos_entropy = 0.0
            for count in freq.values():
                p = count / total
                if p > 0:
                    pos_entropy -= p * math.log2(p)
            total_entropy += pos_entropy

        if n_positions == 0:
            self._cem_learned_alpha = self._cem_dirichlet_concentration
            return

        avg_entropy = total_entropy / n_positions
        # Max entropy for 256 categories = log2(256) = 8
        # Map entropy to alpha in [0.5, 10]:
        #   High entropy (near 8) → low alpha (near 0.5) → weak prior
        #   Low entropy (near 0) → high alpha (near 10) → strong prior
        uniformity = avg_entropy / 8.0  # 0 = fully structured, 1 = uniform
        alpha = self._cem_dirichlet_concentration * (2.0 - uniformity)
        self._cem_learned_alpha = max(0.1, alpha)

    def _cem_alpha(self) -> float:
        """Get the effective Dirichlet concentration parameter for CEM."""
        if self._cem_dirichlet_concentration <= 0:
            return 1.0  # Laplace (add-1) smoothing
        return getattr(self, "_cem_learned_alpha", self._cem_dirichlet_concentration)

    def _freq_to_dist(self, freq: dict[int, int]) -> dict[int, float]:
        """Convert a raw frequency dict to a normalized distribution."""
        total = sum(freq.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in freq.items()}

    def _compute_js(self) -> float:
        """Compute JS divergence between current and previous byte_freq.

        Averages the per-position JS divergence across all positions
        that exist in either distribution.
        """
        if not self._prev_byte_freq or not self.byte_freq:
            return 0.0

        all_positions = set(self._prev_byte_freq) | set(self.byte_freq)
        js_values = array("d")
        for pos in all_positions:
            p = self._freq_to_dist(self._prev_byte_freq.get(pos, {}))
            q = self._freq_to_dist(self.byte_freq.get(pos, {}))
            if not p and not q:
                continue
            js_values.append(self._js_two(p, q))
        return sum(js_values) / len(js_values) if js_values else 0.0

    @staticmethod
    def _js_two(p: dict[int, float], q: dict[int, float]) -> float:
        """JS divergence between two sparse distributions."""
        m: dict[int, float] = {}
        for k in set(p) | set(q):
            m[k] = 0.5 * (p.get(k, 0.0) + q.get(k, 0.0))

        def kl(a: dict[int, float], b: dict[int, float]) -> float:
            return sum(
                pa * math.log(pa / b[k]) for k, pa in a.items() if pa > 0.0 and b.get(k, 0.0) > 0.0
            )

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    # Number of bootstrap replicates used to estimate the no-change reference
    # for the JS statistic. 40 puts the 0.95 and 0.99 quantiles on a usable
    # footing while keeping the whole estimate inside a few milliseconds; the
    # estimate is only consulted once per refit, never on the execution path.
    _JS_NULL_REPLICATES = 40
    _JS_NULL_REPLICATES_NO_NUMPY = 9
    # Floor on replicates, and a wall-clock budget for the whole estimate. Cost
    # scales with input length times elite count: 40 replicates is ~20 ms at
    # 64-byte inputs but ~780 ms at 1024-byte inputs with 100 elites, which is
    # too much even once per refit_interval executions. The loop stops early
    # once the budget is spent, never below the floor. Fewer replicates make the
    # quantiles coarse and biased toward "still moving", which is the direction
    # that refits more often -- the conservative way to be wrong here.
    _JS_NULL_MIN_REPLICATES = 8
    _JS_NULL_BUDGET_SECONDS = 0.05

    def _null_js_quantiles(self, alphas: tuple[float, ...]) -> dict[float, float]:
        """Bootstrap the JS a *stable* distribution would still show at this n.

        JS divergence between two empirical distributions does not go to zero
        when the underlying distribution is unchanged -- it goes to a floor set
        by the sample size and the support width. Measured on byte histograms
        drawn twice from one fixed distribution: mean JS 0.46 at 2 elite
        samples, 0.33 at 20, 0.29 at 100. So "how small is small" is not a
        constant and cannot be read off a table; it has to be estimated at the
        current n.

        The reference is a permutation null: the two samples are pooled and
        re-split at the observed sizes, which is exact under exchangeability at
        any sparsity. Drawing fresh samples from the pooled *proportions*
        instead was tried and is biased low here -- with a 256-value support and
        a handful of elites the pooled support is roughly twice either sample's,
        so replicates overlap more than two real draws would and the reference
        lands under the truth (measured 0.43 against a true null of ~0.63).
        Re-splitting the observations themselves has no such gap.
        """
        prev, curr = self._prev_byte_freq, self.byte_freq
        if not prev or not curr:
            return dict.fromkeys(alphas, 0.0)

        positions = sorted(set(prev) | set(curr))
        pooled = []
        for pos in positions:
            a, b = prev.get(pos, {}), curr.get(pos, {})
            n1, n2 = sum(a.values()), sum(b.values())
            if n1 == 0 or n2 == 0:
                continue
            counts: dict[int, int] = {}
            for k, v in a.items():
                counts[k] = counts.get(k, 0) + v
            for k, v in b.items():
                counts[k] = counts.get(k, 0) + v
            pooled.append((counts, n1, n2))
        if not pooled:
            return dict.fromkeys(alphas, 0.0)

        deadline = time.monotonic() + self._JS_NULL_BUDGET_SECONDS
        if _HAS_NUMPY:
            samples = self._null_js_samples_numpy(
                pooled,
                self._JS_NULL_REPLICATES,
                deadline=deadline,
                min_replicates=self._JS_NULL_MIN_REPLICATES,
            )
        else:
            samples = self._null_js_samples_python(
                pooled,
                self._JS_NULL_REPLICATES_NO_NUMPY,
                deadline=deadline,
                min_replicates=min(self._JS_NULL_MIN_REPLICATES, 4),
            )
        if not samples:
            return dict.fromkeys(alphas, 0.0)
        samples.sort()
        out = {}
        for alpha in alphas:
            # Upper quantile at 1 - alpha, clamped to the sample we have.
            idx = min(len(samples) - 1, int(math.ceil((1.0 - alpha) * len(samples)) - 1))
            out[alpha] = samples[max(0, idx)]
        return out

    @staticmethod
    def _null_js_samples_numpy(
        pooled, replicates: int, deadline: float | None = None, min_replicates: int = 8
    ) -> list[float]:
        """Permutation null, vectorised across positions.

        Positions are grouped by their (n1, n2) pair so one shuffle-and-split
        covers every position in the group at once. Looping positions in Python
        instead costs 1.85 s per refit at 1024-byte inputs with 100 elites,
        which is not affordable even off the execution path.
        """
        rng = np.random.default_rng()
        groups: dict[tuple[int, int], list[np.ndarray]] = {}
        for counts, n1, n2 in pooled:
            total = sum(counts.values())
            if len(counts) < 2 or total < 2:
                continue
            row = np.zeros(256, dtype=np.int64)
            for value, count in counts.items():
                if 0 <= value < 256:
                    row[value] = count
            groups.setdefault((n1, n2), []).append(row)
        if not groups:
            return []

        n_positions = sum(len(rows) for rows in groups.values())
        prepared = []
        for (n1, n2), rows in groups.items():
            counts_2d = np.stack(rows)  # (g, 256)
            # Expand each row's counts into its observation list. Every row in a
            # group has the same total, so this is rectangular.
            total = int(counts_2d[0].sum())
            obs = np.repeat(
                np.tile(np.arange(256, dtype=np.int16), (counts_2d.shape[0], 1)).ravel(),
                counts_2d.ravel(),
            ).reshape(counts_2d.shape[0], total)
            prepared.append((obs, n1, n2))

        out = []
        offsets = None
        for _ in range(replicates):
            acc = 0.0
            for obs, n1, n2 in prepared:
                g, total = obs.shape
                shuffled = rng.permuted(obs, axis=1)
                offsets = (np.arange(g, dtype=np.int64) * 256)[:, None]
                a = np.bincount(
                    (shuffled[:, :n1].astype(np.int64) + offsets).ravel(),
                    minlength=g * 256,
                ).reshape(g, 256).astype(np.float64)
                b = np.bincount(
                    (shuffled[:, n1 : n1 + n2].astype(np.int64) + offsets).ravel(),
                    minlength=g * 256,
                ).reshape(g, 256).astype(np.float64)
                p = a / n1
                q = b / n2
                m = 0.5 * (p + q)
                ratio_p = np.divide(p, m, out=np.ones_like(p), where=m > 0)
                ratio_q = np.divide(q, m, out=np.ones_like(q), where=m > 0)
                acc += 0.5 * float(
                    np.sum(p * np.log(ratio_p, out=np.zeros_like(p), where=ratio_p > 0))
                    + np.sum(q * np.log(ratio_q, out=np.zeros_like(q), where=ratio_q > 0))
                )
            out.append(acc / n_positions)
            if deadline is not None and len(out) >= min_replicates and time.monotonic() > deadline:
                break
        return out

    @staticmethod
    def _null_js_samples_python(
        pooled, replicates: int, deadline: float | None = None, min_replicates: int = 4
    ) -> list[float]:
        per_pos = []
        for counts, n1, n2 in pooled:
            observations = []
            for value, count in counts.items():
                observations.extend([value] * count)
            if len(set(observations)) < 2 or len(observations) < 2:
                continue
            per_pos.append((observations, n1, n2))
        if not per_pos:
            return []

        out = []
        for _ in range(replicates):
            acc = 0.0
            for observations, n1, n2 in per_pos:
                shuffled = observations[:]
                random.shuffle(shuffled)
                a: dict[int, int] = {}
                for k in shuffled[:n1]:
                    a[k] = a.get(k, 0) + 1
                b: dict[int, int] = {}
                for k in shuffled[n1 : n1 + n2]:
                    b[k] = b.get(k, 0) + 1
                p = {k: v / n1 for k, v in a.items()}
                q = {k: v / n2 for k, v in b.items()}
                acc += MonteCarloScheduler._js_two(p, q)
            out.append(acc / len(per_pos))
            if deadline is not None and len(out) >= min_replicates and time.monotonic() > deadline:
                break
        return out

    def _adapt_interval(self) -> None:
        """Adapt refit interval by comparing JS against what no change looks like.

        - JS below the 0.95 quantile of the no-change reference: indistinguishable
          from a stable distribution → double interval
        - JS above the 0.99 quantile: really moving → halve interval
        - In between: no change

        This used to compare the JS divergence -- nats, bounded by ln 2 -- against
        a KS critical value, which is a sup-norm distance between CDFs, and to
        derive that value's sample size from the bandit's pull counts, a quantity
        unrelated to the elite-set distributions being compared. Two unit errors
        stacked. Measured over 8 refits: JS pinned at ~0.63 (91% of the ln 2
        ceiling) against thresholds of 0.026-0.072, so the "stable" branch was
        unreachable after the first refit and the interval ratcheted
        2000 → 1000 → 500 → 250 and stayed at base//4 for the rest of the run.

        The ceiling is not a sign the distribution was changing. Two independent
        draws from one fixed distribution give JS 0.29-0.46 at realistic elite
        sizes, because a 256-value support sampled a few dozen times shares few
        cells with a second such sample. The reference has to be estimated at the
        current sample size, which is what _null_js_quantiles does.
        """
        min_interval = max(1, self.base_refit_interval // 4)
        max_interval = self.base_refit_interval * 4

        quantiles = self._null_js_quantiles((0.05, 0.01))
        stable_threshold = quantiles[0.05]
        unstable_threshold = quantiles[0.01]
        self.last_js_null_p95 = stable_threshold
        self.last_js_null_p99 = unstable_threshold

        if stable_threshold <= 0.0 and unstable_threshold <= 0.0:
            return

        if self.last_js_divergence < stable_threshold:
            self.refit_interval = min(self.refit_interval * 2, max_interval)
        elif self.last_js_divergence > unstable_threshold:
            self.refit_interval = max(self.refit_interval // 2, min_interval)

    def cem_byte(self, pos: int) -> int:
        """Sample a byte at a given position from the CEM distribution.

        Uses a Dirichlet-Multinomial posterior predictive:
            P(byte | data) = (alpha_0 + count_byte) / (256 * alpha_0 + total)

        When cem_dirichlet_concentration > 0, alpha_0 is learned from the
        elite set's entropy. Otherwise falls back to Laplace (add-1) smoothing
        which is equivalent to Dirichlet(1, ..., 1).

        Args:
            pos: Byte position in the input.

        Returns:
            Sampled byte value (0-255).
        """
        freq = self.byte_freq.get(pos)
        if not freq:
            return random.randint(0, 255)
        alpha_0 = self._cem_alpha()
        total = sum(freq.values())
        denom = total + 256.0 * alpha_0
        r = random.random() * denom
        cumulative = 0.0
        for byte_val, count in freq.items():
            cumulative += count + alpha_0
            if r <= cumulative:
                return byte_val
        return random.randint(0, 255)

    def cem_sample(self, length: int) -> bytes:
        """Generate a full input from the CEM distribution.

        Args:
            length: Number of bytes to generate.

        Returns:
            Generated byte sequence.
        """
        return bytes(self.cem_byte(i) for i in range(length))

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Get success/failure counts for each arm.

        Returns:
            Dict mapping operator name to (successes, failures).
        """
        result = {}
        for name in sorted(self.arm_alpha):
            a = self.arm_alpha[name]
            b = self.arm_beta[name]
            result[name] = (max(0.0, a - 1), max(0.0, b - 1))
        return result

    def bandit_stats_raw(self) -> dict[str, tuple[float, float]]:
        """Get raw alpha/beta values for each arm (no prior subtraction).

        Returns:
            Dict mapping operator name to (alpha, beta).
        """
        result = {}
        for name in sorted(self.arm_alpha):
            result[name] = (self.arm_alpha[name], self.arm_beta[name])
        return result

    def transition_stats(self) -> dict[str, dict[str, int]]:
        """Get pairwise transition counts.

        Returns:
            Dict mapping prev_op -> {next_op: discovery_count}.
        """
        return {k: dict(v) for k, v in self.transition_counts.items() if v}

    def save_transitions(self, path: str) -> None:
        """Save transition matrix to JSON."""
        import json

        data = {
            "transition_counts": {k: dict(v) for k, v in self.transition_counts.items()},
            "transition_total": dict(self.transition_total),
        }
        try:
            Path(path).write_text(json.dumps(data, separators=(",", ":")))
        except OSError as e:
            log.debug("Failed to save transitions: %s", e)

    def load_transitions(self, path: str) -> bool:
        """Load transition matrix from JSON. Returns True if loaded."""
        import json

        try:
            data = json.loads(Path(path).read_text())
            for k, v in data.get("transition_counts", {}).items():
                for k2, v2 in v.items():
                    self.transition_counts[k][k2] = v2
            for k, v in data.get("transition_total", {}).items():
                self.transition_total[k] = v
            return True
        except (OSError, json.JSONDecodeError, KeyError):
            return False

    def _stationary_numpy(
        self,
        operators: list[str],
        op_idx: dict[str, int],
        n: int,
        max_iter: int,
        tol: float,
        detect_cycles: bool = False,
    ) -> tuple[dict[str, float], StationaryDiagnostics]:
        """Numpy path for stationary_distribution."""
        P = np.zeros((n, n), dtype=np.float64)
        for prev_op, total in self.transition_total.items():
            if total <= 0 or prev_op not in op_idx:
                continue
            i = op_idx[prev_op]
            targets = self.transition_counts.get(prev_op, {})
            for next_op, count in targets.items():
                if next_op in op_idx:
                    P[i, op_idx[next_op]] = count / total
        row_sums = P.sum(axis=1)
        # Dangling rows (an operator that only ever appears as a target,
        # never as a source) get a self-loop so P stays row-stochastic.
        # `P[mask, np.arange(n)]` pairs a length-k boolean-derived index
        # array with a length-n one elementwise and only avoided an
        # IndexError by accident when k happened to equal n; np.where +
        # matching row/col indices is what "set the diagonal of these
        # rows to 1" actually means.
        dangling = np.where(row_sums < 1e-12)[0]
        P[dangling, :] = 0.0
        P[dangling, dangling] = 1.0

        def step(v: np.ndarray) -> np.ndarray:
            new_v = v @ P
            total = float(new_v.sum())
            return new_v / total if total > 0 else new_v

        pi = np.full(n, 1.0 / n)
        converged = False
        iterations = 0
        for i in range(max_iter):
            new_pi = step(pi)
            diff = float(np.abs(pi - new_pi).sum())
            pi = new_pi
            iterations = i + 1
            if diff < tol:
                converged = True
                break

        diag = StationaryDiagnostics(
            converged=converged, periodic=False, period=1, iterations=iterations
        )
        if not converged and detect_cycles:
            # Slow convergence and an exact non-convergent oscillation both
            # look like "diff never dropped below tol" from the loop above.
            # Floyd's algorithm tells them apart: a periodic chain's
            # sub-dominant eigenmodes decay away, so by the time max_iter
            # is exhausted `pi` is already (up to float noise) inside the
            # limit cycle if one exists — cheap to confirm from here.
            # Opt-in only (detect_cycles=False by default): this costs up
            # to 20 * max_iter additional `step` calls on top of an
            # already-failed power iteration.
            tol_cycle = max(tol * 100, 1e-6)
            cycle = floyd_detect(
                pi,
                step,
                is_close=lambda a, b: bool(np.abs(a - b).sum() < tol_cycle),
                max_steps=20 * max_iter,
            )
            self._record_cycle_check(cycle)
            if cycle is not None and cycle.period > 1:
                pi = cesaro_average(cycle.state, step, cycle.period)
                diag = StationaryDiagnostics(
                    converged=False,
                    periodic=True,
                    period=cycle.period,
                    iterations=iterations,
                    cycle_checked=True,
                )
            else:
                diag = StationaryDiagnostics(
                    converged=False,
                    periodic=False,
                    period=1,
                    iterations=iterations,
                    cycle_checked=True,
                )
        return {op: float(pi[op_idx[op]]) for op in operators}, diag

    @staticmethod
    def _power_iteration_py(
        v: list[float], p_matrix: list[list[float]], n: int, max_iter: int, tol: float
    ) -> list[float]:
        """Pure-Python power iteration: v_{k+1} = P^T @ v_k with convergence check."""
        for _ in range(max_iter):
            new_v = [0.0] * n
            for j in range(n):
                for i in range(n):
                    new_v[j] += p_matrix[i][j] * v[i]
            norm = math.sqrt(sum(x * x for x in new_v))
            if norm < 1e-12:
                break
            new_v = [x / norm for x in new_v]
            diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, new_v, strict=False)))
            v = new_v
            if diff < tol:
                break
        return v

    @staticmethod
    def _power_iteration_py_transpose(
        v: list[float], p_matrix: list[list[float]], n: int, max_iter: int, tol: float
    ) -> tuple[list[float], bool, int]:
        """Pure-Python power iteration: v_{k+1} = v_k @ P with convergence check.

        Returns (v, converged, iterations) — see `_power_iteration_py_step`
        for the reusable per-round update this shares with the
        periodicity check in `_stationary_py`.
        """
        converged = False
        iterations = 0
        for i in range(max_iter):
            new_v = [0.0] * n
            for j in range(n):
                for i2 in range(n):
                    new_v[j] += v[i2] * p_matrix[i2][j]
            total = sum(new_v)
            if total > 0:
                new_v = [x / total for x in new_v]
            diff = sum(abs(a - b) for a, b in zip(v, new_v, strict=False))
            v = new_v
            iterations = i + 1
            if diff < tol:
                converged = True
                break
        return v, converged, iterations

    @staticmethod
    def _power_iteration_py_transpose_step(
        v: list[float], p_matrix: list[list[float]], n: int
    ) -> list[float]:
        """Single round of `v_{k+1} = v_k @ P`, row-normalized. Shared step
        function for both the main loop above and the Floyd cycle check in
        `_stationary_py`, so both walk the exact same sequence.
        """
        new_v = [0.0] * n
        for j in range(n):
            for i in range(n):
                new_v[j] += v[i] * p_matrix[i][j]
        total = sum(new_v)
        return [x / total for x in new_v] if total > 0 else new_v

    @staticmethod
    def _build_transition_matrix_py(transition_total, transition_counts, op_idx, n):
        """Build row-stochastic transition matrix (pure-Python)."""
        p_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for prev_op, total in transition_total.items():
            if total <= 0 or prev_op not in op_idx:
                continue
            i = op_idx[prev_op]
            targets = transition_counts.get(prev_op, {})
            for next_op, count in targets.items():
                if next_op in op_idx:
                    p_matrix[i][op_idx[next_op]] = count / total
        for i in range(n):
            if sum(p_matrix[i]) < 1e-12:
                p_matrix[i][i] = 1.0
        return p_matrix

    def _stationary_py(
        self,
        operators: list[str],
        op_idx: dict[str, int],
        n: int,
        max_iter: int,
        tol: float,
        detect_cycles: bool = False,
    ) -> tuple[dict[str, float], StationaryDiagnostics]:
        """Pure-Python fallback for stationary_distribution."""
        p_matrix = self._build_transition_matrix_py(
            self.transition_total, self.transition_counts, op_idx, n
        )
        pi, converged, iterations = self._power_iteration_py_transpose(
            [1.0 / n] * n, p_matrix, n, max_iter, tol
        )

        diag = StationaryDiagnostics(
            converged=converged, periodic=False, period=1, iterations=iterations
        )
        if not converged and detect_cycles:
            tol_cycle = max(tol * 100, 1e-6)

            def step(v: list[float]) -> list[float]:
                return self._power_iteration_py_transpose_step(v, p_matrix, n)

            def is_close(a: list[float], b: list[float]) -> bool:
                return sum(abs(x - y) for x, y in zip(a, b, strict=False)) < tol_cycle

            cycle = floyd_detect(pi, step, is_close, max_steps=20 * max_iter)
            self._record_cycle_check(cycle)
            if cycle is not None and cycle.period > 1:
                pi = cesaro_average(
                    cycle.state,
                    step,
                    cycle.period,
                    add=lambda a, b: [x + y for x, y in zip(a, b, strict=False)],
                    scale=lambda a, k: [x * k for x in a],
                )
                diag = StationaryDiagnostics(
                    converged=False,
                    periodic=True,
                    period=cycle.period,
                    iterations=iterations,
                    cycle_checked=True,
                )
            else:
                diag = StationaryDiagnostics(
                    converged=False,
                    periodic=False,
                    period=1,
                    iterations=iterations,
                    cycle_checked=True,
                )
        return {op: pi[op_idx[op]] for op in operators}, diag

    def stationary_distribution(
        self,
        max_iter: int = 200,
        tol: float = 1e-8,
        return_diagnostics: bool = False,
        detect_cycles: bool = False,
    ) -> dict[str, float] | tuple[dict[str, float], StationaryDiagnostics]:
        """Compute the stationary distribution π of the transition Markov chain.

        Uses power iteration: π_{k+1} = π_k · P until convergence.
        The stationary distribution satisfies πP = π — it tells you which
        operator sequences the fuzzer naturally settles into.

        Some operator-transition chains are exactly periodic (e.g. an
        alternation A -> B -> A -> B): power iteration on those never
        satisfies the tolerance check and no π_k is individually "the"
        stationary distribution. When `detect_cycles=True` and that
        happens, Floyd cycle detection runs on the iterate sequence (see
        `core/cycle_detect.py`) and the result returned is instead the
        Cesaro (time) average over one full period, which is the value
        that's actually meaningful for a periodic chain — rather than an
        arbitrary non-converged snapshot from whichever iteration
        max_iter happened to cut off at.

        Args:
            max_iter: Maximum power iteration steps.
            tol: Convergence tolerance (L1 norm of change).
            return_diagnostics: If True, also return a
                `StationaryDiagnostics` describing how the result was
                reached (converged normally / periodic-averaged / neither
                / not checked).
            detect_cycles: Opt-in. If True and power iteration fails to
                converge, run Floyd's algorithm to check for an exact
                limit cycle. Off by default: the check costs up to
                20 * max_iter extra `step` calls on top of an
                already-failed power iteration, and most callers only
                want the fast non-converged snapshot. When False (the
                default), a non-converged result is returned as-is and
                `StationaryDiagnostics.cycle_checked` is False.

        Returns:
            Dict mapping operator name -> stationary probability, or
            `(dict, StationaryDiagnostics)` if `return_diagnostics` is True.
        """
        if not self.transition_total:
            empty_diag = StationaryDiagnostics(
                converged=True, periodic=False, period=1, iterations=0
            )
            return ({}, empty_diag) if return_diagnostics else {}

        operators = sorted(
            set(self.transition_total.keys())
            | {op for targets in self.transition_counts.values() for op in targets}
        )
        n = len(operators)
        if n == 0:
            empty_diag = StationaryDiagnostics(
                converged=True, periodic=False, period=1, iterations=0
            )
            return ({}, empty_diag) if return_diagnostics else {}
        if n == 1:
            single = {operators[0]: 1.0}
            single_diag = StationaryDiagnostics(
                converged=True, periodic=False, period=1, iterations=0
            )
            return (single, single_diag) if return_diagnostics else single

        op_idx = {op: i for i, op in enumerate(operators)}

        if _HAS_NUMPY:
            pi_dict, diag = self._stationary_numpy(
                operators, op_idx, n, max_iter, tol, detect_cycles=detect_cycles
            )
        else:
            pi_dict, diag = self._stationary_py(
                operators, op_idx, n, max_iter, tol, detect_cycles=detect_cycles
            )
        return (pi_dict, diag) if return_diagnostics else pi_dict

    def cycle_stats(self) -> dict[str, int]:
        """Snapshot of Floyd cycle-detection stats accumulated so far.

        Only moves when `stationary_distribution(detect_cycles=True)` has
        actually been called and power iteration failed to converge at
        least once; all zero means the check has never run (either
        because it's never been requested, or every call so far
        converged normally).
        """
        return {
            "checks": self.cycle_checks,
            "detections": self.cycle_detections,
            "last_period": self.last_cycle_period,
            "max_period": self.max_cycle_period,
        }

    def _spectral_gap_numpy(
        self, operators: list[str], op_idx: dict[str, int], n: int, max_iter: int, tol: float
    ) -> float:
        """Numpy path for spectral_gap."""
        P = np.zeros((n, n), dtype=np.float64)
        for prev_op, total in self.transition_total.items():
            if total <= 0 or prev_op not in op_idx:
                continue
            i = op_idx[prev_op]
            targets = self.transition_counts.get(prev_op, {})
            for next_op, count in targets.items():
                if next_op in op_idx:
                    P[i, op_idx[next_op]] = count / total
        row_sums = P.sum(axis=1)
        # Same dangling-row fix as `_stationary_numpy` — see the comment
        # there for why `P[mask, np.arange(n)]` was broken.
        dangling = np.where(row_sums < 1e-12)[0]
        P[dangling, :] = 0.0
        P[dangling, dangling] = 1.0

        # Power iteration for dominant eigenvector (left eigenvector = stationary dist)
        v = np.full(n, 1.0 / math.sqrt(n))
        for _ in range(max_iter):
            new_v = P.T @ v
            norm = np.linalg.norm(new_v)
            if norm < 1e-12:
                break
            new_v /= norm
            diff = np.linalg.norm(v - new_v)
            v = new_v
            if diff < tol:
                break

        # Deflate: P_deflated = P - v·v^T  (outer product)
        deflated = P - np.outer(v, v)

        # Power iteration on deflated matrix for λ₂
        w = np.random.randn(n)
        w /= np.linalg.norm(w)
        eigenvalue2 = 0.0
        for _ in range(max_iter):
            new_w = deflated.T @ w
            eigenvalue2 = abs(float(w @ new_w))
            norm = np.linalg.norm(new_w)
            if norm < 1e-12:
                break
            new_w /= norm
            diff = np.linalg.norm(w - new_w)
            w = new_w
            if diff < tol:
                break

        return max(0.0, min(1.0, 1.0 - eigenvalue2))

    def _spectral_gap_py(
        self, operators: list[str], op_idx: dict[str, int], n: int, max_iter: int, tol: float
    ) -> float:
        """Pure-Python fallback for spectral_gap."""
        p_matrix = self._build_transition_matrix_py(
            self.transition_total, self.transition_counts, op_idx, n
        )
        v = self._power_iteration_py([1.0 / n] * n, p_matrix, n, max_iter, tol)

        # Deflate: P_deflated = P - v * v^T
        deflated: list[list[float]] = [
            [p_matrix[i][j] - v[i] * v[j] for j in range(n)] for i in range(n)
        ]
        w = self._power_iteration_py(
            [random.random() for _ in range(n)], deflated, n, max_iter, tol
        )
        eigenvalue2 = abs(
            sum(
                a * b
                for a, b in zip(
                    w,
                    [sum(deflated[i][j] * w[i] for i in range(n)) for j in range(n)],
                    strict=False,
                )
            )
        )
        return max(0.0, min(1.0, 1.0 - eigenvalue2))

    def spectral_gap(self, max_iter: int = 200, tol: float = 1e-8) -> float:
        """Compute the spectral gap of the transition Markov chain.

        The spectral gap is 1 - λ₂ where λ₂ is the second-largest
        eigenvalue. Measures how quickly the operator sequence converges
        to its stationary distribution.

        - Large gap (→1): fast mixing
        - Small gap (→0): slow mixing, stuck in narrow cycles

        Returns:
            Spectral gap in [0, 1].
        """
        if not self.transition_total:
            return 1.0

        operators = sorted(
            set(self.transition_total.keys())
            | {op for targets in self.transition_counts.values() for op in targets}
        )
        n = len(operators)
        if n <= 1:
            return 1.0

        op_idx = {op: i for i, op in enumerate(operators)}

        if _HAS_NUMPY:
            return self._spectral_gap_numpy(operators, op_idx, n, max_iter, tol)
        return self._spectral_gap_py(operators, op_idx, n, max_iter, tol)

    def should_explore(self, gap_threshold: float = 0.1) -> bool:
        """Check if the fuzzer is stuck in an operator cycle.

        Args:
            gap_threshold: Spectral gap below which exploration is recommended.

        Returns:
            True if spectral gap < gap_threshold (stagnation detected).
        """
        return self.spectral_gap() < gap_threshold

    def correlated_select(self, ops: list[str], segment_size: int = 50) -> str:
        """Select operator via correlated Thompson sampling.

        Adds multivariate normal noise whose covariance is the empirical
        operator covariance. Correlated arms get similar score boosts,
        so they're selected together rather than fighting each other.

        Falls back to standard Thompson sampling when insufficient data.

        Args:
            ops: Available mutation operators.
            segment_size: Segments per covariance estimate.

        Returns:
            Name of the selected operator.
        """
        if len(ops) < 3:
            return self._standard_thompson(ops)

        cov = self.operator_covariance(window=2000, segment_size=segment_size)
        if not cov or not all(op in cov for op in ops):
            return self._standard_thompson(ops)

        n = len(ops)
        cov_matrix = [[cov[ops[i]].get(ops[j], 0.0) for j in range(n)] for i in range(n)]

        chol = self._chol(cov_matrix)
        if chol is None:
            return self._standard_thompson(ops)

        z = (
            np.random.randn(n).astype(np.float64)
            if _HAS_NUMPY
            else [random.gauss(0, 1) for _ in range(n)]
        )
        if _HAS_NUMPY:
            noise = chol @ z
        else:
            noise = [0.0] * n
            for i in range(n):
                for j in range(i + 1):
                    noise[i] += chol[i][j] * z[j]

        scores = {}
        for i, op in enumerate(ops):
            a, b = self._get_effective_params(op)
            scores[op] = a / (a + b) + noise[i]

        return max(ops, key=lambda o: scores[o])

    def _standard_thompson(self, ops: list[str]) -> str:
        """Pure Thompson sampling without correlation structure."""
        best_op = None
        best_val = -1.0
        for op in ops:
            a, b = self._get_effective_params(op)
            val = random.betavariate(a, b)
            if val > best_val:
                best_val = val
                best_op = op
        return best_op if best_op is not None else ops[0]

    @staticmethod
    def _chol(matrix: list[list[float]]) -> np.ndarray | None:
        """Cholesky decomposition with regularization via numpy."""
        if _HAS_NUMPY:
            a = np.asarray(matrix, dtype=np.float64)
        else:
            return MonteCarloScheduler._chol_py(matrix)
        n = a.shape[0]
        if n == 0:
            return None
        diag = np.diag(a)
        diag[diag <= 0] = 1.0
        np.fill_diagonal(a, diag)
        diag_min = float(np.min(diag))
        reg = max(diag_min * 0.01, 1e-6)
        a[np.diag_indices_from(a)] += reg
        try:
            return np.linalg.cholesky(a)
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _chol_py(matrix: list[list[float]]) -> list[list[float]] | None:
        """Pure-Python Cholesky decomposition (fallback when numpy unavailable)."""
        n = len(matrix)
        if n == 0:
            return None
        a = [row[:] for row in matrix]
        for i in range(n):
            if a[i][i] <= 0:
                a[i][i] = 1.0
        diag_min = min(a[i][i] for i in range(n))
        reg = max(diag_min * 0.01, 1e-6)
        for i in range(n):
            a[i][i] += reg
        lower = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(lower[i][k] * lower[j][k] for k in range(j))
                if i == j:
                    val = a[i][i] - s
                    if val <= 0:
                        return None
                    lower[i][j] = math.sqrt(val)
                else:
                    lower[i][j] = (a[i][j] - s) / lower[j][j] if lower[j][j] > 0 else 0.0
        return lower

    @staticmethod
    def _solve_cholesky_vec(chol, vec, n: int):
        """Solve ``L L^T x = vec`` for one right-hand side.

        Forward substitution then back substitution against the factor that
        ``_chol`` already produced: O(n^2), and the factorization is used
        rather than discarded. The routine this replaces re-formed
        ``L @ L.T`` and handed it to a general LU against the whole identity
        matrix, which threw the factorization away and paid O(n^3) twice
        over to build an inverse only one column of which was ever read.

        Returns None when the factor is singular, matching the guard the
        matrix path already had.

        numpy has no triangular solve, so each substitution goes through the
        general LU; that still beats the old route because it carries one
        right-hand side instead of n. scipy.linalg.cho_solve would be another
        ~9x on top (35 us against 317 us at n=155) but this project has no
        scipy dependency on purpose -- see the note in core/allan_variance.py.
        """
        if n == 0:
            return None
        if _HAS_NUMPY and isinstance(chol, np.ndarray):
            b = np.asarray(vec, dtype=np.float64)
            if not np.all(np.diagonal(chol) > 0.0):
                return None
            try:
                y = np.linalg.solve(chol, b)
                return np.linalg.solve(chol.T, y)
            except np.linalg.LinAlgError:
                return None
        y = [0.0] * n
        for i in range(n):
            if chol[i][i] <= 0.0:
                return None
            s = sum(chol[i][k] * y[k] for k in range(i))
            y[i] = (vec[i] - s) / chol[i][i]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(chol[k][i] * x[k] for k in range(i + 1, n))
            x[i] = (y[i] - s) / chol[i][i]
        return x

    @staticmethod
    def _dot(a, b, n: int) -> float:
        if _HAS_NUMPY and isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return float(a @ b)
        return float(sum(a[i] * b[i] for i in range(n)))

    def _matrix_ucb_scores(
        self, ops: list[str], mu, inv_mu, base: float, beta: float, t: int, n: int
    ) -> dict[str, float]:
        """UCB scores with the covariance penalty, given ``x = C^-1 mu``.

        Takes the one vector the formula actually reads. The variant this
        replaces took the full inverse and indexed a single row out of it.
        """
        log_t = math.log(t)
        scores: dict[str, float] = {}
        for i, op in enumerate(ops):
            penalty = base - 2.0 * float(inv_mu[i])
            exploration = beta * math.sqrt(max(0.0, log_t + penalty))
            scores[op] = float(mu[i]) + exploration
        return scores

    def _matrix_ucb_prepare(
        self, ops: list[str], segment_size: int
    ) -> tuple[int, list | np.ndarray, list[list[float]]] | None:
        """Prepare means vector and covariance matrix, or None if fallback needed."""
        if len(ops) < 3:
            return None
        means = {}
        for op in ops:
            a = self.arm_alpha.get(op, 1.0)
            b = self.arm_beta.get(op, 1.0)
            means[op] = a / (a + b)
        cov = self.operator_covariance(window=2000, segment_size=segment_size)
        if not cov or not all(op in cov for op in ops):
            return None
        n = len(ops)
        mu: list | np.ndarray = [means[op] for op in ops]
        if _HAS_NUMPY:
            mu = np.array(mu, dtype=np.float64)
        cov_matrix = [[cov[ops[i]].get(ops[j], 0.0) for j in range(n)] for i in range(n)]
        return (n, mu, cov_matrix)

    def matrix_ucb_select(self, ops: list[str], beta: float = 2.0, segment_size: int = 50) -> str:
        """Select operator via matrix-based Upper Confidence Bound.

        Adjusts UCB exploration bonuses using the covariance structure.
        Arms correlated with high-performing arms get reduced exploration.

        Falls back to standard UCB when insufficient data.

        Args:
            ops: Available mutation operators.
            beta: Exploration parameter.
            segment_size: Segments per covariance estimate.

        Returns:
            Name of the selected operator.
        """
        prepared = self._matrix_ucb_prepare(ops, segment_size)
        if prepared is None:
            return self._standard_ucb(ops, beta)
        n, mu, cov_matrix = prepared

        chol = self._chol(cov_matrix)
        if chol is None:
            return self._standard_ucb(ops, beta)

        # The scores need exactly one vector, x = C^-1 mu: the per-arm
        # penalty is base - 2*x[i], and the shared term is the quadratic
        # form mu^T C^-1 mu, which is just mu . x. Neither needs C^-1.
        #
        # This used to solve C X = I for the whole inverse -- n right-hand
        # sides, O(n^3) -- and then compute mu @ inv @ mu and inv @ mu as
        # two separate products of the same quantity. Solving the single
        # right-hand side mu against the factorization already in hand is
        # O(n^2), and mu . x is then free.
        inv_mu = self._solve_cholesky_vec(chol, mu, n)
        if inv_mu is None:
            return self._standard_ucb(ops, beta)

        base = self._dot(mu, inv_mu, n)
        t = sum(self.arm_alpha.values()) + sum(self.arm_beta.values()) - 2 * len(self.arm_alpha)
        t = max(t, 1)

        scores = self._matrix_ucb_scores(ops, mu, inv_mu, base, beta, t, n)
        return max(ops, key=lambda o: scores[o])

    def _standard_ucb(self, ops: list[str], beta: float = 2.0) -> str:
        """Standard UCB1 without covariance adjustment."""
        total = sum(self.arm_alpha.values()) + sum(self.arm_beta.values()) - 2 * len(self.arm_alpha)
        total = max(total, 1)

        best_op = None
        best_score = -1.0
        for op in ops:
            a = self.arm_alpha.get(op, 1.0)
            b = self.arm_beta.get(op, 1.0)
            n_i = max(a + b - 2, 1)
            mean = a / (a + b)
            exploration = beta * math.sqrt(math.log(total) / n_i)
            score = mean + exploration
            if score > best_score:
                best_score = score
                best_op = op
        return best_op if best_op is not None else ops[0]


    def _build_segment_rates(
        self, recent: list, segment_size: int, operators: list[str], op_idx: dict[str, int]
    ) -> list[list[float]]:
        """Build segment success-rate matrix from recent history."""
        segments: list[list[float]] = []
        n_ops = len(operators)
        for start in range(0, len(recent) - segment_size + 1, segment_size):
            chunk = recent[start : start + segment_size]
            rates = [0.0] * n_ops
            counts = [0] * n_ops
            for op, success in chunk:
                if op in op_idx:
                    i = op_idx[op]
                    counts[i] += 1
                    rates[i] += 1.0 if success else 0.0
            for i in range(n_ops):
                if counts[i] > 0:
                    rates[i] /= counts[i]
            segments.append(rates)
        return segments

    def _operator_covariance_numpy(
        self, recent: list, segment_size: int, operators: list[str], op_idx: dict[str, int]
    ) -> dict[str, dict[str, float]]:
        """Numpy path for operator_covariance."""
        segments_list = self._build_segment_rates(recent, segment_size, operators, op_idx)
        if len(segments_list) < 2:
            return {}
        seg_arr = np.array(segments_list, dtype=np.float64)
        cov_arr = np.cov(seg_arr, rowvar=False)
        cov_matrix: dict[str, dict[str, float]] = {op: {} for op in operators}
        for i, op_i in enumerate(operators):
            for j, op_j in enumerate(operators):
                cov_matrix[op_i][op_j] = float(cov_arr[i, j])
        return cov_matrix

    def _operator_covariance_py(
        self, recent: list, segment_size: int, operators: list[str], op_idx: dict[str, int]
    ) -> dict[str, dict[str, float]]:
        """Pure-Python fallback for operator_covariance."""
        n_ops = len(operators)
        segments = self._build_segment_rates(recent, segment_size, operators, op_idx)
        if len(segments) < 2:
            return {}
        n_seg = len(segments)
        means = [sum(s[i] for s in segments) / n_seg for i in range(n_ops)]
        cov_matrix = {op: {} for op in operators}
        for i, op_i in enumerate(operators):
            for j, op_j in enumerate(operators):
                if i == j:
                    var = sum((s[i] - means[i]) ** 2 for s in segments) / (n_seg - 1)
                    cov_matrix[op_i][op_j] = var
                elif i < j:
                    cov_val = sum((s[i] - means[i]) * (s[j] - means[j]) for s in segments) / (
                        n_seg - 1
                    )
                    cov_matrix[op_i][op_j] = cov_val
                    cov_matrix[op_j][op_i] = cov_val
        return cov_matrix

    def operator_covariance(
        self, window: int = 500, segment_size: int = 50
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise covariance of operator success rates.

        Divides history into segments and computes per-operator success
        rate per segment. High positive covariance = redundant operators.

        Args:
            window: Number of recent observations to consider.
            segment_size: Observations per segment.

        Returns:
            Nested dict: covariance[op_a][op_b] = Cov(success_a, success_b).
        """
        if not self._op_success_history:
            return {}

        recent = list(self._op_success_history)[-window:]
        if len(recent) < 2 * segment_size:
            return {}

        operators = sorted(set(self.arm_alpha.keys()) | {op for op, _ in recent})
        if len(operators) < 1:
            return {}

        op_idx = {op: i for i, op in enumerate(operators)}

        if _HAS_NUMPY:
            return self._operator_covariance_numpy(recent, segment_size, operators, op_idx)
        return self._operator_covariance_py(recent, segment_size, operators, op_idx)

    def select_op_minimax(
        self,
        operators: list[str],
        depth: int = 3,
        beam_width: int = 4,
    ) -> str:
        """Select operator using adversarial minimax sequencing (Phase 4).

        Models operator selection as a two-player game:
        - Fuzzer (maximizer): chooses the next operator to apply
        - Target (minimizer): the target's coverage response resists progress

        Uses alpha-beta pruning over a game tree where each ply alternates
        between fuzzer and target moves. The evaluation function estimates
        the expected edge-discovery rate from a sequence of operators.

        Args:
            operators: Candidate operators to choose from.
            depth: Search depth (number of plies).
            beam_width: Maximum number of candidates to explore per ply.

        Returns:
            Selected operator name.
        """
        if not operators:
            return ""
        if len(operators) == 1:
            return operators[0]

        # Get current operator statistics
        alphas = self.arm_alpha
        betas = self.arm_beta

        def evaluate(op: str) -> float:
            """Estimate expected reward for an operator using Thompson sample."""
            a = alphas.get(op, 1.0)
            b = betas.get(op, 1.0)
            # Use mean of Beta distribution as point estimate
            return a / (a + b) if (a + b) > 0 else 0.5

        def minimax(
            ops: list[str],
            d: int,
            alpha: float,
            beta: float,
            maximizing: bool,
            visited: set[str],
        ) -> float:
            """Alpha-beta minimax over operator sequences."""
            if d == 0 or not ops:
                return 0.0

            # Beam search: only consider top candidates
            scored_ops = sorted(ops, key=evaluate, reverse=True)
            beam = scored_ops[:beam_width]

            if maximizing:
                best = -float("inf")
                for op in beam:
                    # Fuzzer's move: apply operator, get reward
                    reward = evaluate(op)
                    # Simulate: after applying op, target responds
                    remaining = [o for o in ops if o != op]
                    val = reward + 0.5 * minimax(
                        remaining, d - 1, alpha, beta, False, visited | {op}
                    )
                    best = max(best, val)
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break  # Beta cutoff
                return best
            else:
                # Target's move: minimize fuzzer's reward
                worst = float("inf")
                for op in beam:
                    # Target "blocks" this operator (reduces its effectiveness)
                    penalty = evaluate(op) * 0.3
                    remaining = [o for o in ops if o != op]
                    val = -penalty + 0.5 * minimax(
                        remaining, d - 1, alpha, beta, True, visited | {op}
                    )
                    worst = min(worst, val)
                    beta = min(beta, worst)
                    if beta <= alpha:
                        break  # Alpha cutoff
                return worst

        # Find the operator that maximizes the minimax value
        best_op = operators[0]
        best_val = -float("inf")
        for op in operators[:beam_width]:
            reward = evaluate(op)
            remaining = [o for o in operators if o != op]
            val = reward + 0.5 * minimax(
                remaining, depth - 1, -float("inf"), float("inf"), False, {op}
            )
            if val > best_val:
                best_val = val
                best_op = op
        return best_op
