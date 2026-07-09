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

log = logging.getLogger(__name__)


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

    def init_arm(self, name: str) -> None:
        """Register an operator for tracking."""
        if name not in self.ratings:
            self.ratings[name] = self.default_rating
            self._match_count[name] = 0
        if self.crash_track and name not in self.crash_ratings:
            self.crash_ratings[name] = self.default_rating

    def _expected_score(self, ra: float, rb: float) -> float:
        """Expected score for player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def record_match(
        self, op_a: str, op_b: str, score_a: float, crash: bool = False
    ) -> None:
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
            if hasattr(self, '_prev_operators') and self._prev_operators:
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

    def select_op(
        self, operators: list[str], temperature: float = 400.0
    ) -> str:
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
        rated = [
            op for op in operators
            if self._match_count.get(op, 0) >= self.min_matches
        ]
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

        rated = [
            op for op in operators
            if self._match_count.get(op, 0) >= self.min_matches
        ]
        if not rated:
            return operators[0]

        ratings = [
            self.crash_ratings.get(op, self.default_rating) for op in rated
        ]
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
            (op, r) for op, r in sorted(src.items(), key=lambda x: -x[1])
            if self._match_count.get(op, 0) >= self.min_matches
        ]

    def get_unrated(self) -> list[str]:
        """Return operators with fewer than min_matches matches."""
        return [
            op for op, count in self._match_count.items()
            if count < self.min_matches
        ]

    def apply_decay(self) -> None:
        """Apply exponential decay to all ratings (call periodically)."""
        self._decay_ticks += 1
        for name in self.ratings:
            self.ratings[name] = (
                self.default_rating
                + (self.ratings[name] - self.default_rating) * self.decay
            )
        if self.crash_track:
            for name in self.crash_ratings:
                self.crash_ratings[name] = (
                    self.default_rating
                    + (self.crash_ratings[name] - self.default_rating) * self.decay
                )

    def get_rating(self, name: str) -> float:
        """Get current Elo rating for an operator."""
        return self.ratings.get(name, self.default_rating)

    def get_crash_rating(self, name: str) -> float:
        """Get current crash-specific Elo rating."""
        return self.crash_ratings.get(name, self.default_rating)

    def save(self, path: str) -> bool:
        """Save state to JSON."""
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
        log.info("Elo tracker loaded: %s (%d operators)", path, len(self.ratings))
        return True
