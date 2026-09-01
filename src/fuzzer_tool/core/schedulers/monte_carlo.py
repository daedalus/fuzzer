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
from array import array
from collections import defaultdict
from pathlib import Path

from fuzzer_tool.core.allan_variance import DispersionIndex
from fuzzer_tool.core.edge_tracker import ks_significance_threshold
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

    def _adapt_interval(self) -> None:
        """Adapt refit interval based on JS divergence with sample-size-aware thresholds.

        Uses KS critical values instead of fixed thresholds:
        - JS below KS threshold at alpha=0.05: distribution stable → double interval
        - JS above KS threshold at alpha=0.01: still changing → halve interval
        - In between: no change
        """
        min_interval = max(1, self.base_refit_interval // 4)
        max_interval = self.base_refit_interval * 4

        n = sum(self.arm_alpha.values()) + sum(self.arm_beta.values())
        stable_threshold = ks_significance_threshold(max(1, int(n / 2)), alpha=0.05)
        unstable_threshold = ks_significance_threshold(max(1, int(n / 2)), alpha=0.01)

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
        self, operators: list[str], op_idx: dict[str, int], n: int, max_iter: int, tol: float
    ) -> dict[str, float]:
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
        P[row_sums < 1e-12, :] = 0.0
        P[row_sums < 1e-12, np.arange(n)] = 1.0
        pi = np.full(n, 1.0 / n)
        for _ in range(max_iter):
            new_pi = pi @ P
            total = float(new_pi.sum())
            if total > 0:
                new_pi /= total
            diff = float(np.abs(pi - new_pi).sum())
            pi = new_pi
            if diff < tol:
                break
        return {op: float(pi[op_idx[op]]) for op in operators}

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
    ) -> list[float]:
        """Pure-Python power iteration: v_{k+1} = v_k @ P with convergence check."""
        for _ in range(max_iter):
            new_v = [0.0] * n
            for j in range(n):
                for i in range(n):
                    new_v[j] += v[i] * p_matrix[i][j]
            total = sum(new_v)
            if total > 0:
                new_v = [x / total for x in new_v]
            diff = sum(abs(a - b) for a, b in zip(v, new_v, strict=False))
            v = new_v
            if diff < tol:
                break
        return v

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
        self, operators: list[str], op_idx: dict[str, int], n: int, max_iter: int, tol: float
    ) -> dict[str, float]:
        """Pure-Python fallback for stationary_distribution."""
        p_matrix = self._build_transition_matrix_py(
            self.transition_total, self.transition_counts, op_idx, n
        )
        pi = self._power_iteration_py_transpose([1.0 / n] * n, p_matrix, n, max_iter, tol)
        return {op: pi[op_idx[op]] for op in operators}

    def stationary_distribution(self, max_iter: int = 200, tol: float = 1e-8) -> dict[str, float]:
        """Compute the stationary distribution π of the transition Markov chain.

        Uses power iteration: π_{k+1} = π_k · P until convergence.
        The stationary distribution satisfies πP = π — it tells you which
        operator sequences the fuzzer naturally settles into.

        Args:
            max_iter: Maximum power iteration steps.
            tol: Convergence tolerance (L1 norm of change).

        Returns:
            Dict mapping operator name -> stationary probability.
        """
        if not self.transition_total:
            return {}

        operators = sorted(
            set(self.transition_total.keys())
            | {op for targets in self.transition_counts.values() for op in targets}
        )
        n = len(operators)
        if n == 0:
            return {}
        if n == 1:
            return {operators[0]: 1.0}

        op_idx = {op: i for i, op in enumerate(operators)}

        if _HAS_NUMPY:
            return self._stationary_numpy(operators, op_idx, n, max_iter, tol)
        return self._stationary_py(operators, op_idx, n, max_iter, tol)

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
        P[row_sums < 1e-12, :] = 0.0
        P[row_sums < 1e-12, np.arange(n)] = 1.0

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

    def _matrix_ucb_quadratic_form(self, mu: list[float] | np.ndarray, inv_cov, n: int) -> float:
        """Compute quadratic form mu^T @ inv_cov @ mu."""
        if _HAS_NUMPY:
            return float(mu @ inv_cov @ mu)
        base = 0.0
        for i in range(n):
            for j in range(n):
                base += mu[i] * inv_cov[i][j] * mu[j]
        return base

    def _matrix_ucb_scores(
        self, ops: list[str], mu, inv_cov, base: float, beta: float, t: int, n: int
    ) -> dict[str, float]:
        """Compute UCB scores with covariance penalty."""
        scores: dict[str, float] = {}
        if _HAS_NUMPY:
            inv_mu = inv_cov @ mu
            for i, op in enumerate(ops):
                penalty = base - 2.0 * float(inv_mu[i])
                exploration = beta * math.sqrt(max(0.0, math.log(t) + penalty))
                scores[op] = float(mu[i]) + exploration
        else:
            for i, op in enumerate(ops):
                penalty = 0.0
                for j in range(n):
                    penalty += inv_cov[i][j] * mu[j]
                penalty = base - 2 * penalty
                exploration = beta * math.sqrt(max(0.0, math.log(t) + penalty))
                scores[op] = mu[i] + exploration
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

        identity = (
            np.eye(n, dtype=np.float64)
            if _HAS_NUMPY
            else [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        )
        inv_cov = self._solve_cholesky(chol, identity)
        if inv_cov is None:
            return self._standard_ucb(ops, beta)

        base = self._matrix_ucb_quadratic_form(mu, inv_cov, n)
        t = sum(self.arm_alpha.values()) + sum(self.arm_beta.values()) - 2 * len(self.arm_alpha)
        t = max(t, 1)

        scores = self._matrix_ucb_scores(ops, mu, inv_cov, base, beta, t, n)
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

    @staticmethod
    def _solve_cholesky(
        chol: np.ndarray | list[list[float]], rhs: list[list[float]]
    ) -> np.ndarray | None:
        """Solve L @ L^T @ X = rhs via numpy (forward/back substitution)."""
        if _HAS_NUMPY:
            n = np.asarray(chol).shape[0] if not isinstance(chol, np.ndarray) else chol.shape[0]
            if n == 0:
                return None
            b = np.asarray(rhs, dtype=np.float64)
            L = chol if isinstance(chol, np.ndarray) else np.asarray(chol, dtype=np.float64)
            try:
                return np.linalg.solve(L @ L.T, b)
            except np.linalg.LinAlgError:
                return None
        else:
            return MonteCarloScheduler._solve_cholesky_py(chol, rhs)

    @staticmethod
    def _solve_cholesky_py(
        chol: list[list[float]], rhs: list[list[float]]
    ) -> list[list[float]] | None:
        """Pure-Python forward/back substitution (fallback when numpy unavailable)."""
        n = len(chol)
        if n == 0:
            return None
        m = len(rhs[0]) if rhs else 0
        y = [[0.0] * m for _ in range(n)]
        for col in range(m):
            for i in range(n):
                s = sum(chol[i][k] * y[k][col] for k in range(i))
                y[i][col] = (rhs[i][col] - s) / chol[i][i] if chol[i][i] > 0 else 0.0
        x = [[0.0] * m for _ in range(n)]
        for col in range(m):
            for i in range(n - 1, -1, -1):
                s = sum(chol[k][i] * x[k][col] for k in range(i + 1, n))
                x[i][col] = (y[i][col] - s) / chol[i][i] if chol[i][i] > 0 else 0.0
        return x

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
