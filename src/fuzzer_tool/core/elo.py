"""Elo rating system for fuzzer operator scheduling.

Provides a complementary signal to Thompson bandit, MOpt PSO, and
Replicator dynamics. Unlike those mechanisms, Elo offers:

  - Temporal decay via K-factor and exponential rating decay
  - Pairwise competition: "on seed X, operator A found edges, B didn't → A wins"
  - Difficulty adjustment: beating a high-rated operator is worth more
  - A single interpretable number per operator
  - Crash-specific ranking (separate track where "win" = finding a crash)

Usage:
    elo = EloTracker(k_factor=16)
    elo.init_arm("bit_flip")
    elo.init_arm("byte_insert")
    # After execution: bit_flip found new edges, byte_insert didn't
    elo.record_match("bit_flip", "byte_insert", score_a=1.0)
    op = elo.select_op(["bit_flip", "byte_insert"])
"""

import json
import logging
import math
import random

from fuzzer_tool.core.running_stats import RunningMoments

log = logging.getLogger(__name__)

# Minimum observations before the kurtosis-scaled trust gate allows the
# UCB bonus.  High-kurtosis reward distributions need more samples before
# the stddev estimate is trustworthy.
_UCB_MIN_SAMPLES_BASE = 20


def _softmax_select(
    scored: list[tuple[str, float]], temperature: float
) -> str:
    """Weighted random selection via softmax over scored items.

    Args:
        scored: List of (name, score) tuples.  Scores can be on any scale
            — the softmax handles relative differences.
        temperature: Higher = more uniform, lower = more greedy.

    Returns:
        Selected name.
    """
    if not scored:
        return ""
    if len(scored) == 1:
        return scored[0][0]
    max_s = max(s for _, s in scored)
    # Clamp to prevent underflow when max_s == min_s (all equal)
    weights = [math.exp((s - max_s) / max(temperature, 1e-9)) for _, s in scored]
    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    for i, (name, _) in enumerate(scored):
        cumulative += weights[i]
        if r <= cumulative:
            return name
    return scored[-1][0]


class EloTracker:
    """Elo rating tracker for fuzzer operators.

    Args:
        k_factor: Maximum rating change per match (higher = more reactive).
        default_rating: Starting Elo rating for new operators.
        decay: Exponential decay factor per apply_decay() call (0.99 = 1% decay).
        crash_track: If True, maintain separate crash-focused ratings.
        min_matches: Minimum matches before an operator is considered "rated".
            Unrated operators are excluded from ranking and selection.
    """

    def __init__(
        self,
        k_factor: float = 16.0,
        default_rating: float = 1500.0,
        decay: float = 0.99,
        crash_track: bool = True,
        min_matches: int = 10,
    ):
        self.k_factor = k_factor
        self.default_rating = default_rating
        self.decay = decay
        self.crash_track = crash_track
        self.min_matches = min_matches

        self.ratings: dict[str, float] = {}
        self.crash_ratings: dict[str, float] = {}
        self._match_count: dict[str, int] = {}
        self._decay_ticks: int = 0

        self._strategy_ratings: dict[str, float] = {}
        self._strategy_match_count: dict[str, int] = {}

        # Per-operator reward moments for UCB-style exploration bonus.
        self._reward_moments: dict[str, RunningMoments] = {}

    def init_arm(self, name: str) -> None:
        """Register an operator for tracking."""
        if name not in self.ratings:
            self.ratings[name] = self.default_rating
            self._match_count[name] = 0
        if self.crash_track and name not in self.crash_ratings:
            self.crash_ratings[name] = self.default_rating
        if name not in self._reward_moments:
            self._reward_moments[name] = RunningMoments()

    def record_reward(self, op: str, reward: float) -> None:
        """Record a scalar reward for an operator (edges gained, etc.).

        Used by UCB-style selection to estimate per-operator reward
        distributions with mean + variance.
        """
        if op not in self._reward_moments:
            self._reward_moments[op] = RunningMoments()
        self._reward_moments[op].update(reward)

    def select_op_ucb(
        self,
        operators: list[str],
        exploration_weight: float = 1.0,
        temperature: float = 400.0,
    ) -> str:
        """Select operator using UCB-style score: mean + k * stddev.

        Separates operators into "UCB-ready" (enough reward samples) and
        "Elo-fallback" (too few samples) groups.  Each group is scored on
        its own scale and the two groups are not mixed in the same softmax —
        raw Elo ratings (~1500) swamp UCB scores (~0–1) when naively combined.

        High-kurtosis operators require more observations before trusting
        the stddev-based bonus (kurtosis stability guard).

        Only used when --bandit-variance-bonus is enabled (off by default).

        Args:
            operators: Candidate operators.
            exploration_weight: k in mean + k * stddev.
            temperature: Fallback Elo temperature when no reward data.

        Returns:
            Selected operator name.
        """
        if not operators:
            return ""
        if len(operators) == 1:
            return operators[0]

        # Separate operators into two groups to avoid scale mismatch
        # between raw Elo (~1500) and UCB scores (~0–1).
        ucb_ready: list[tuple[str, float]] = []
        elo_only: list[str] = []
        for op in operators:
            moments = self._reward_moments.get(op)
            if moments is None or moments.count < 3:
                elo_only.append(op)
                continue
            kurt = moments.kurtosis
            min_samples = _UCB_MIN_SAMPLES_BASE * max(1.0, 1.0 + kurt * 0.1)
            if moments.count < min_samples:
                elo_only.append(op)
                continue
            score = moments.mean + exploration_weight * moments.stddev
            ucb_ready.append((op, score))

        # Pure UCB selection
        if ucb_ready and not elo_only:
            return _softmax_select(ucb_ready, temperature)

        # Pure Elo selection
        if elo_only and not ucb_ready:
            scored = [(op, self.ratings.get(op, self.default_rating)) for op in elo_only]
            return _softmax_select(scored, temperature)

        # Both groups exist — probabilistically pick which group to draw from,
        # weighted by total sample count in each group.  This keeps both
        # exploration (via Elo group) and exploitation (via UCB group) active
        # while avoiding the scale-mismatch bug.
        total_ucb = sum(
            self._reward_moments[op].count for op, _ in ucb_ready
        )
        total_elo = sum(
            self._match_count.get(op, 0) for op in elo_only
        )
        if random.random() < total_ucb / (total_ucb + total_elo):
            return _softmax_select(ucb_ready, temperature)
        else:
            scored = [(op, self.ratings.get(op, self.default_rating)) for op in elo_only]
            return _softmax_select(scored, temperature)

    def get_reward_moments(self, op: str) -> RunningMoments | None:
        """Get reward statistics for an operator (for diagnostics)."""
        return self._reward_moments.get(op)

    def _expected_score(self, ra: float, rb: float) -> float:
        """Expected score for player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def record_match(self, op_a: str, op_b: str, score_a: float, crash: bool = False) -> None:
        """Record a match between two operators.

        Args:
            op_a: First operator name.
            op_b: Second operator name.
            score_a: Score for op_a (1.0=win, 0.5=draw, 0.0=loss).
            crash: If True, also update crash-specific ratings.
        """
        ra = self.ratings.get(op_a, self.default_rating)
        rb = self.ratings.get(op_b, self.default_rating)

        ea = self._expected_score(ra, rb)
        eb = self._expected_score(rb, ra)

        self.ratings[op_a] = ra + self.k_factor * (score_a - ea)
        self.ratings[op_b] = rb + self.k_factor * ((1.0 - score_a) - eb)
        self._match_count[op_a] = self._match_count.get(op_a, 0) + 1
        self._match_count[op_b] = self._match_count.get(op_b, 0) + 1

        # Track reward distribution per operator
        self.record_reward(op_a, score_a)
        self.record_reward(op_b, 1.0 - score_a)

        if crash and self.crash_track:
            cra = self.crash_ratings.get(op_a, self.default_rating)
            crb = self.crash_ratings.get(op_b, self.default_rating)
            ca = self._expected_score(cra, crb)
            cb = self._expected_score(crb, cra)
            self.crash_ratings[op_a] = cra + self.k_factor * (score_a - ca)
            self.crash_ratings[op_b] = crb + self.k_factor * ((1.0 - score_a) - cb)

    def record_round(
        self,
        operators: list[str],
        winners: set[str],
        edge_counts: dict[str, int] | None = None,
        crash: bool = False,
    ) -> None:
        """Record outcomes for a group of operators used in one iteration.

        When edge_counts is provided with multiple operators having edges,
        uses proportional scoring (edges[op] / max_edges) instead of binary
        win/loss. This gives finer-grained signal.

        When all operators are winners (or all losers), falls back to
        cross-iteration comparison with blended scoring.

        Args:
            operators: All operators used this iteration.
            winners: Subset that found new edges or crashes.
            edge_counts: Per-operator edge discovery counts (optional).
            crash: If True, also update crash ratings.
        """
        losers = [op for op in operators if op not in winners]

        if losers:
            # Normal case: winners beat losers
            if edge_counts and len(winners) > 1:
                # Multiple winners — use proportional scoring among them
                max_edges = max(edge_counts.get(w, 0) for w in winners) or 1
                for w in winners:
                    w_edges = edge_counts.get(w, 0)
                    score = w_edges / max_edges  # proportional
                    for l in losers:
                        self.record_match(w, l, score_a=score, crash=crash)
            else:
                for w in winners:
                    for l in losers:
                        self.record_match(w, l, score_a=1.0, crash=crash)

        elif len(operators) >= 2:
            # All winners or all losers — cross-iteration comparison
            if hasattr(self, "_prev_operators") and self._prev_operators:
                prev_ops = self._prev_operators
                if winners:
                    # Current round found coverage — blend with previous
                    blend = 0.7
                    for w in operators:
                        for p in prev_ops:
                            self.record_match(w, p, score_a=blend, crash=crash)
                else:
                    # Current round didn't find coverage
                    for w in prev_ops:
                        for p in operators:
                            self.record_match(w, p, score_a=0.7, crash=crash)

        self._prev_operators = operators

    def select_op(self, operators: list[str], temperature: float = 400.0) -> str:
        """Select an operator weighted by Elo rating.

        Only considers operators with >= min_matches matches.

        Args:
            operators: Candidate operators.
            temperature: Higher = more uniform selection, lower = more greedy.

        Returns:
            Selected operator name, or first operator if none rated.
        """
        if not operators:
            return ""
        if len(operators) == 1:
            return operators[0]

        # Filter to rated operators
        rated = [op for op in operators if self._match_count.get(op, 0) >= self.min_matches]
        if not rated:
            return operators[0]  # fallback: return first candidate

        ratings = [self.ratings.get(op, self.default_rating) for op in rated]
        max_r = max(ratings)
        weights = [math.exp((r - max_r) / temperature) for r in ratings]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return rated[i]
        return rated[-1]

    def select_crash_op(self, operators: list[str], temperature: float = 400.0) -> str:
        """Select operator weighted by crash-specific Elo rating."""
        if not operators:
            return ""
        if not self.crash_track:
            return self.select_op(operators, temperature)
        if len(operators) == 1:
            return operators[0]

        rated = [op for op in operators if self._match_count.get(op, 0) >= self.min_matches]
        if not rated:
            return operators[0]

        ratings = [self.crash_ratings.get(op, self.default_rating) for op in rated]
        max_r = max(ratings)
        weights = [math.exp((r - max_r) / temperature) for r in ratings]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return rated[i]
        return rated[-1]

    def get_ranking(self, crash: bool = False) -> list[tuple[str, float]]:
        """Return rated operators sorted by rating (highest first).

        Only includes operators with >= min_matches matches.

        Args:
            crash: If True, rank by crash-specific ratings.

        Returns:
            List of (operator_name, rating) tuples.
        """
        src = self.crash_ratings if crash and self.crash_track else self.ratings
        return [
            (op, r)
            for op, r in sorted(src.items(), key=lambda x: -x[1])
            if self._match_count.get(op, 0) >= self.min_matches
        ]

    def get_unrated(self) -> list[str]:
        """Return operators with fewer than min_matches matches."""
        return [op for op, count in self._match_count.items() if count < self.min_matches]

    def record_strategy_match(self, strategy_a: str, strategy_b: str, score_a: float) -> None:
        """Record a match between two operator-selection strategies.

        Used by the meta-scheduler to arbitrate between bandit and MOpt.
        Each strategy is tracked with its own Elo rating pool, separate
        from the per-operator ratings.

        Args:
            strategy_a: First strategy name (e.g. "bandit", "mopt").
            strategy_b: Second strategy name.
            score_a: Score for strategy_a (1.0=win, 0.5=draw, 0.0=loss).
        """
        ra = self._strategy_ratings.get(strategy_a, self.default_rating)
        rb = self._strategy_ratings.get(strategy_b, self.default_rating)

        ea = self._expected_score(ra, rb)
        eb = self._expected_score(rb, ra)

        self._strategy_ratings[strategy_a] = ra + self.k_factor * (score_a - ea)
        self._strategy_ratings[strategy_b] = rb + self.k_factor * ((1.0 - score_a) - eb)
        self._strategy_match_count[strategy_a] = self._strategy_match_count.get(strategy_a, 0) + 1
        self._strategy_match_count[strategy_b] = self._strategy_match_count.get(strategy_b, 0) + 1

    def select_strategy(self, strategies: list[str], temperature: float = 400.0) -> str:
        """Select a strategy weighted by Elo rating.

        Used by the meta-scheduler to pick bandit vs MOpt probabilistically.

        Args:
            strategies: Candidate strategy names.
            temperature: Higher = more uniform, lower = more greedy.

        Returns:
            Selected strategy name, or first strategy if none rated.
        """
        if not strategies:
            return ""
        if len(strategies) == 1:
            return strategies[0]

        rated = [s for s in strategies if self._strategy_match_count.get(s, 0) >= self.min_matches]
        if not rated:
            return strategies[0]

        ratings = [self._strategy_ratings.get(s, self.default_rating) for s in rated]
        max_r = max(ratings)
        weights = [math.exp((r - max_r) / temperature) for r in ratings]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return rated[i]
        return rated[-1]

    def get_strategy_ranking(self) -> list[tuple[str, float]]:
        """Return strategies sorted by Elo rating (highest first)."""
        return [
            (s, r)
            for s, r in sorted(self._strategy_ratings.items(), key=lambda x: -x[1])
            if self._strategy_match_count.get(s, 0) >= self.min_matches
        ]

    def apply_decay(self) -> None:
        """Apply exponential decay to all ratings (call periodically)."""
        self._decay_ticks += 1
        for name in self.ratings:
            self.ratings[name] = (
                self.default_rating + (self.ratings[name] - self.default_rating) * self.decay
            )
        if self.crash_track:
            for name in self.crash_ratings:
                self.crash_ratings[name] = (
                    self.default_rating
                    + (self.crash_ratings[name] - self.default_rating) * self.decay
                )
        for name in self._strategy_ratings:
            self._strategy_ratings[name] = (
                self.default_rating
                + (self._strategy_ratings[name] - self.default_rating) * self.decay
            )

    def get_rating(self, name: str) -> float:
        """Get current Elo rating for an operator."""
        return self.ratings.get(name, self.default_rating)

    def get_crash_rating(self, name: str) -> float:
        """Get current crash-specific Elo rating."""
        return self.crash_ratings.get(name, self.default_rating)

    def save(self, path: str) -> bool:
        """Save state to JSON."""
        reward_moments_ser = {}
        for op, rm in self._reward_moments.items():
            reward_moments_ser[op] = rm.save()
        data = {
            "k_factor": self.k_factor,
            "default_rating": self.default_rating,
            "decay": self.decay,
            "crash_track": self.crash_track,
            "min_matches": self.min_matches,
            "ratings": self.ratings,
            "crash_ratings": self.crash_ratings,
            "match_count": self._match_count,
            "decay_ticks": self._decay_ticks,
            "strategy_ratings": self._strategy_ratings,
            "strategy_match_count": self._strategy_match_count,
            "reward_moments": reward_moments_ser,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            log.info("Elo tracker saved: %s (%d operators)", path, len(self.ratings))
            return True
        except OSError as e:
            log.warning("Failed to save Elo tracker: %s", e)
            return False

    def load(self, path: str) -> bool:
        """Load state from JSON."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.debug("Failed to load Elo tracker: %s", e)
            return False

        self.k_factor = data.get("k_factor", self.k_factor)
        self.default_rating = data.get("default_rating", self.default_rating)
        self.decay = data.get("decay", self.decay)
        self.crash_track = data.get("crash_track", self.crash_track)
        self.min_matches = data.get("min_matches", self.min_matches)
        self.ratings = data.get("ratings", {})
        self.crash_ratings = data.get("crash_ratings", {})
        self._match_count = data.get("match_count", {})
        self._decay_ticks = data.get("decay_ticks", 0)
        self._strategy_ratings = data.get("strategy_ratings", {})
        self._strategy_match_count = data.get("strategy_match_count", {})
        self._reward_moments = {}
        for op, rm_data in data.get("reward_moments", {}).items():
            rm = RunningMoments()
            rm.load(rm_data)
            self._reward_moments[op] = rm
        log.info("Elo tracker loaded: %s (%d operators)", path, len(self.ratings))
        return True


class BayesianEloTracker:
    """Bayesian Elo rating system with posterior uncertainty.

    Unlike the point-estimate EloTracker, each operator maintains a Gaussian
    posterior N(mu, sigma^2) over its true rating. Key differences:

    1. **Posterior updates**: rating updates scale by the uncertainty ratio
       (sigma^2 / (sigma^2 + beta^2)), so uncertain ratings move more.
    2. **Thompson sampling**: selection draws from each posterior and picks
       the highest draw, naturally exploring operators with high uncertainty.
    3. **Adaptive K-factor**: K scales with recent prediction error — when
       the model is consistently surprised, K grows to react faster.
    4. **Adaptive temperature**: temperature adjusts based on how often the
       top-rated operator actually wins (log-likelihood tracking).

    Args:
        initial_mu: Prior mean for new operators (default 1500).
        initial_sigma: Prior standard deviation (default 350 — high
            uncertainty for new operators).
        beta: Game outcome uncertainty (default 200). Higher = slower
            rating changes.
        tau: System noise added per time step (default 5). Higher =
            more non-stationarity.
        min_matches: Minimum matches before an operator is rated.
    """

    def __init__(
        self,
        initial_mu: float = 1500.0,
        initial_sigma: float = 350.0,
        beta: float = 200.0,
        tau: float = 5.0,
        min_matches: int = 10,
    ):
        self.initial_mu = initial_mu
        self.initial_sigma = initial_sigma
        self.beta = beta
        self.tau = tau
        self.min_matches = min_matches

        # Per-operator Gaussian posteriors: N(mu, sigma_sq)
        self.mu: dict[str, float] = {}
        self.sigma_sq: dict[str, float] = {}
        self._match_count: dict[str, int] = {}

        # Strategy ratings (same mechanism)
        self._strategy_mu: dict[str, float] = {}
        self._strategy_sigma_sq: dict[str, float] = {}
        self._strategy_match_count: dict[str, int] = {}

        # Adaptive K-factor tracking
        self._prediction_errors: list[float] = []
        self._base_k = 16.0  # base learning rate
        self._error_window = 100  # lookback for error tracking

        # Adaptive temperature tracking
        self._best_win_rate: list[float] = []
        self._base_temperature = 400.0

    def _expected_score(self, mu_a: float, mu_b: float) -> float:
        """Expected score for player A given their rating posterior means."""
        return 1.0 / (1.0 + 10.0 ** ((mu_b - mu_a) / 400.0))

    def _effective_k(self) -> float:
        """Adaptive K-factor based on recent prediction accuracy.

        When predictions are consistently wrong (>2 standard deviations of
        random), K increases up to 2x to react faster. When accurate, K
        decreases toward base_k / 2.
        """
        if len(self._prediction_errors) < 10:
            return self._base_k
        recent = self._prediction_errors[-min(len(self._prediction_errors), self._error_window):]
        mse = sum(e * e for e in recent) / len(recent)
        # Expected MSE for random predictions with score in [0,1] = 0.25
        # Good predictions approach 0. Scale K from base/2 to base*2
        ratio = min(mse / 0.25, 1.0)
        return self._base_k * (0.5 + 1.5 * ratio)

    def _effective_temperature(self) -> float:
        """Adaptive temperature based on how often the top-rated wins.

        When the best operator wins consistently, temperature decreases
        (more greedy). When outcomes are noisy, temperature increases.
        """
        if len(self._best_win_rate) < 20:
            return self._base_temperature
        recent = self._best_win_rate[-50:]
        win_rate = sum(recent) / len(recent)
        # win_rate near 1.0 → confident, lower T (more greedy)
        # win_rate near 0.5 → noisy, higher T (more exploratory)
        scale = 2.0 - win_rate  # 1.0 → 1.0, 0.5 → 1.5, 0.0 → 2.0
        return self._base_temperature * scale

    def init_arm(self, name: str) -> None:
        """Register an operator with the prior N(initial_mu, initial_sigma^2)."""
        if name not in self.mu:
            self.mu[name] = self.initial_mu
            self.sigma_sq[name] = self.initial_sigma ** 2
            self._match_count[name] = 0

    def record_match(self, op_a: str, op_b: str, score_a: float, crash: bool = False) -> None:
        """Record a match and update both operators' posteriors.

        The update uses the Bayesian Elo rule:
            mu' = mu + (sigma^2 / (sigma^2 + beta^2)) * K * (score - expected)
            sigma'^2 = sigma^2 * (1 - sigma^2 / (sigma^2 + beta^2)) + tau^2

        Args:
            op_a: First operator name.
            op_b: Second operator name.
            score_a: Score for op_a (1.0=win, 0.5=draw, 0.0=loss).
            crash: Reserved for future use (maintains interface parity).
        """
        for name in (op_a, op_b):
            self.init_arm(name)

        mu_a, mu_b = self.mu[op_a], self.mu[op_b]
        sig_a = self.sigma_sq[op_a]
        sig_b = self.sigma_sq[op_b]

        ea = self._expected_score(mu_a, mu_b)
        eb = self._expected_score(mu_b, mu_a)

        k = self._effective_k()

        # Track prediction error
        self._prediction_errors.append(score_a - ea)
        if len(self._prediction_errors) > self._error_window * 2:
            self._prediction_errors = self._prediction_errors[-self._error_window:]

        # Uncertainty-scaled update
        var_a = sig_a / (sig_a + self.beta ** 2)
        var_b = sig_b / (sig_b + self.beta ** 2)

        self.mu[op_a] = mu_a + var_a * k * (score_a - ea)
        self.mu[op_b] = mu_b + var_b * k * ((1.0 - score_a) - eb)

        # Posterior variance shrinks, then adds system noise (tau^2)
        self.sigma_sq[op_a] = sig_a * (1.0 - var_a) + self.tau ** 2
        self.sigma_sq[op_b] = sig_b * (1.0 - var_b) + self.tau ** 2

        self._match_count[op_a] = self._match_count.get(op_a, 0) + 1
        self._match_count[op_b] = self._match_count.get(op_b, 0) + 1

        # Track whether the higher-rated operator won
        # Use >= for the initial tie case (both at initial_mu)
        if mu_a >= mu_b:
            self._best_win_rate.append(1.0 if score_a > 0.5 else 0.0)
        else:
            self._best_win_rate.append(0.0 if score_a > 0.5 else 1.0)
        if len(self._best_win_rate) > 100:
            self._best_win_rate = self._best_win_rate[-100:]

    def record_round(
        self,
        operators: list[str],
        winners: set[str],
        edge_counts: dict[str, int] | None = None,
        crash: bool = False,
    ) -> None:
        """Record outcomes for a group of operators (same logic as EloTracker)."""
        losers = [op for op in operators if op not in winners]

        if losers:
            if edge_counts and len(winners) > 1:
                max_edges = max(edge_counts.get(w, 0) for w in winners) or 1
                for w in winners:
                    w_edges = edge_counts.get(w, 0)
                    score = w_edges / max_edges
                    for l in losers:
                        self.record_match(w, l, score_a=score, crash=crash)
            else:
                for w in winners:
                    for l in losers:
                        self.record_match(w, l, score_a=1.0, crash=crash)

        elif len(operators) >= 2:
            if hasattr(self, "_prev_operators") and self._prev_operators:
                prev_ops = self._prev_operators
                if winners:
                    blend = 0.7
                    for w in operators:
                        for p in prev_ops:
                            self.record_match(w, p, score_a=blend, crash=crash)
                else:
                    for w in prev_ops:
                        for p in operators:
                            self.record_match(w, p, score_a=0.7, crash=crash)

        self._prev_operators = operators

    def _thompson_sample(self, name: str) -> float:
        """Draw from the operator's posterior N(mu, sigma)."""
        mu = self.mu.get(name, self.initial_mu)
        sigma = math.sqrt(self.sigma_sq.get(name, self.initial_sigma ** 2))
        return random.gauss(mu, sigma)

    def select_op(self, operators: list[str], temperature: float | None = None) -> str:
        """Select an operator via Thompson sampling from posteriors.

        Draws from each operator's Gaussian posterior and returns the one
        with the highest draw. Operators with high uncertainty (wide
        posteriors) get a natural exploration bonus.

        Args:
            operators: Candidate operators.
            temperature: Ignored (Thompson sampling replaces softmax).
                Kept for interface compatibility.

        Returns:
            Selected operator name.
        """
        if not operators:
            return ""
        if len(operators) == 1:
            return operators[0]

        rated = [op for op in operators if self._match_count.get(op, 0) >= self.min_matches]
        if not rated:
            return operators[0]

        samples = [(op, self._thompson_sample(op)) for op in rated]
        return max(samples, key=lambda x: x[1])[0]

    def select_crash_op(self, operators: list[str], temperature: float | None = None) -> str:
        """Select by Thompson sampling from crash-agnostic posteriors.

        Currently uses the same posterior (crash-specific tracking can be
        added as a separate set of posteriors following the same pattern).
        """
        return self.select_op(operators)

    def record_strategy_match(self, strategy_a: str, strategy_b: str, score_a: float) -> None:
        """Record a match between two selection strategies."""
        for s in (strategy_a, strategy_b):
            if s not in self._strategy_mu:
                self._strategy_mu[s] = self.initial_mu
                self._strategy_sigma_sq[s] = self.initial_sigma ** 2
                self._strategy_match_count[s] = 0

        mu_a, mu_b = self._strategy_mu[strategy_a], self._strategy_mu[strategy_b]
        sig_a = self._strategy_sigma_sq[strategy_a]
        sig_b = self._strategy_sigma_sq[strategy_b]

        ea = self._expected_score(mu_a, mu_b)
        eb = self._expected_score(mu_b, mu_a)

        var_a = sig_a / (sig_a + self.beta ** 2)
        var_b = sig_b / (sig_b + self.beta ** 2)
        k = self._effective_k()

        self._strategy_mu[strategy_a] = mu_a + var_a * k * (score_a - ea)
        self._strategy_mu[strategy_b] = mu_b + var_b * k * ((1.0 - score_a) - eb)
        self._strategy_sigma_sq[strategy_a] = sig_a * (1.0 - var_a) + self.tau ** 2
        self._strategy_sigma_sq[strategy_b] = sig_b * (1.0 - var_b) + self.tau ** 2
        self._strategy_match_count[strategy_a] += 1
        self._strategy_match_count[strategy_b] += 1

    def select_strategy(self, strategies: list[str], temperature: float | None = None) -> str:
        """Select a strategy via Thompson sampling from its posterior."""
        if not strategies:
            return ""
        if len(strategies) == 1:
            return strategies[0]

        rated = [s for s in strategies if self._strategy_match_count.get(s, 0) >= self.min_matches]
        if not rated:
            return strategies[0]

        samples = [(s, random.gauss(
            self._strategy_mu.get(s, self.initial_mu),
            math.sqrt(self._strategy_sigma_sq.get(s, self.initial_sigma ** 2)),
        )) for s in rated]
        return max(samples, key=lambda x: x[1])[0]

    def get_ranking(self, crash: bool = False) -> list[tuple[str, float]]:
        """Return operators sorted by posterior mean (highest first).

        Args:
            crash: Ignored (maintains interface parity).

        Returns:
            List of (operator_name, mu) tuples.
        """
        return [
            (op, self.mu[op])
            for op, mu in sorted(self.mu.items(), key=lambda x: -x[1])
            if self._match_count.get(op, 0) >= self.min_matches
        ]

    def get_unrated(self) -> list[str]:
        """Return operators with fewer than min_matches matches."""
        return [op for op, count in self._match_count.items() if count < self.min_matches]

    def get_strategy_ranking(self) -> list[tuple[str, float]]:
        """Return strategies sorted by posterior mean (highest first)."""
        return [
            (s, self._strategy_mu[s])
            for s in sorted(self._strategy_mu, key=lambda s: -self._strategy_mu[s])
            if self._strategy_match_count.get(s, 0) >= self.min_matches
        ]

    def apply_decay(self) -> None:
        """Apply system noise (tau) to all posteriors. Called periodically.

        Increases uncertainty for all operators, modeling non-stationarity
        (old observations become less informative). This is the Bayesian
        equivalent of Elo's exponential rating decay.
        """
        for name in self.sigma_sq:
            self.sigma_sq[name] += self.tau ** 2
        for name in self._strategy_sigma_sq:
            self._strategy_sigma_sq[name] += self.tau ** 2

    def get_rating(self, name: str) -> float:
        """Return the posterior mean for an operator."""
        return self.mu.get(name, self.initial_mu)

    def save(self, path: str) -> bool:
        """Save state to JSON."""
        data = {
            "initial_mu": self.initial_mu,
            "initial_sigma": self.initial_sigma,
            "beta": self.beta,
            "tau": self.tau,
            "min_matches": self.min_matches,
            "mu": self.mu,
            "sigma_sq": self.sigma_sq,
            "match_count": self._match_count,
            "strategy_mu": self._strategy_mu,
            "strategy_sigma_sq": self._strategy_sigma_sq,
            "strategy_match_count": self._strategy_match_count,
            "prediction_errors": self._prediction_errors,
            "best_win_rate": self._best_win_rate,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            log.info("BayesianElo tracker saved: %s (%d operators)", path, len(self.mu))
            return True
        except OSError as e:
            log.warning("Failed to save BayesianElo tracker: %s", e)
            return False

    def load(self, path: str) -> bool:
        """Load state from JSON."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.debug("Failed to load BayesianElo tracker: %s", e)
            return False

        self.initial_mu = data.get("initial_mu", self.initial_mu)
        self.initial_sigma = data.get("initial_sigma", self.initial_sigma)
        self.beta = data.get("beta", self.beta)
        self.tau = data.get("tau", self.tau)
        self.min_matches = data.get("min_matches", self.min_matches)
        self.mu = data.get("mu", {})
        self.sigma_sq = data.get("sigma_sq", {})
        self._match_count = data.get("match_count", {})
        self._strategy_mu = data.get("strategy_mu", {})
        self._strategy_sigma_sq = data.get("strategy_sigma_sq", {})
        self._strategy_match_count = data.get("strategy_match_count", {})
        self._prediction_errors = data.get("prediction_errors", [])
        self._best_win_rate = data.get("best_win_rate", [])
        log.info("BayesianElo tracker loaded: %s (%d operators)", path, len(self.mu))
        return True
