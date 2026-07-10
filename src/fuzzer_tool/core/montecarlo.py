"""Monte Carlo scheduler: Thompson sampling bandit + CEM byte distribution.

Uses JS divergence to adaptively control CEM refit frequency:
- JS → 0 after refit: distribution stabilized, refit less often
- JS stays high: elite set still shifting, refit more aggressively

Also tracks Brier score (binary CRPS) for bandit calibration diagnostics.

Includes MOptScheduler: Particle Swarm Optimization over operator probability
distributions, an alternative to Thompson sampling that searches the joint
configuration space rather than each operator's marginal success rate.
"""

import collections
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

from fuzzer_tool.core.edge_tracker import ks_significance_threshold

log = logging.getLogger(__name__)


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

    def __init__(
        self,
        elite_frac: float = 0.1,
        refit_interval: int = 1000,
        pairwise_blend: float = 0.0,
        arm_decay: float = 0.999,
    ):
        self.arm_alpha: dict[str, float] = {}
        self.arm_beta: dict[str, float] = {}
        self.arm_decay = arm_decay
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

    def init_arm(self, name: str) -> None:
        """Register a mutation operator arm with prior (1, 1).

        Args:
            name: Name of the mutation operator.
        """
        if name not in self.arm_alpha:
            self.arm_alpha[name] = 1.0
            self.arm_beta[name] = 1.0

    def select_op(self, ops: list[str], prev_op: str | None = None) -> str:
        """Select mutation operator via Thompson sampling with pairwise transitions.

        When pairwise_blend > 0 and prev_op has transition data, blends
        the unconditional Thompson sample with a pairwise-conditional sample
        that favors operators that historically followed prev_op.

        Args:
            ops: Available mutation operators.
            prev_op: The operator used in the previous mutation step (for
                pairwise transition weighting).

        Returns:
            Name of the selected operator.
        """
        # Unconditional Thompson sample for each op
        thompson_vals = {}
        for op in ops:
            a = self.arm_alpha.get(op, 1.0)
            b = self.arm_beta.get(op, 1.0)
            thompson_vals[op] = random.betavariate(a, b)

        # If no pairwise data or blend is zero, use pure Thompson
        if self.pairwise_blend <= 0 or prev_op is None or prev_op not in self.transition_total:
            best_op = max(ops, key=lambda o: thompson_vals[o])
            return best_op

        # Pairwise score: Dirichlet-Multinomial over transition counts
        # With uniform prior (alpha=1), score = count + 1
        total = self.transition_total[prev_op]
        pair_scores = {}
        for op in ops:
            count = self.transition_counts[prev_op].get(op, 0)
            pair_scores[op] = (count + 1) / (total + len(ops))

        # Blend: w * pair + (1-w) * thompson
        w = self.pairwise_blend
        blended = {}
        for op in ops:
            blended[op] = w * pair_scores[op] + (1 - w) * thompson_vals[op]

        best_op = max(ops, key=lambda o: blended[o])
        self._prev_op = best_op
        return best_op

    def record(self, name: str, success: bool) -> None:
        """Record outcome for a mutation operator arm.

        Applies exponential decay to all arms before incrementing,
        giving recent evidence more weight (non-stationary bandit).

        Args:
            name: Name of the mutation operator.
            success: Whether the mutation produced an interesting result.
        """
        self._op_success_history.append((name, success))

        if self.arm_decay < 1.0:
            for k in self.arm_alpha:
                self.arm_alpha[k] *= self.arm_decay
            for k in self.arm_beta:
                self.arm_beta[k] *= self.arm_decay

        if success:
            self.arm_alpha[name] = self.arm_alpha.get(name, 1.0) + 1
        else:
            self.arm_beta[name] = self.arm_beta.get(name, 1.0) + 1

        # Update pairwise transition matrix on success
        if success and self._prev_op is not None and self._prev_op != name:
            self.transition_counts[self._prev_op][name] += 1
            self.transition_total[self._prev_op] += 1

    def record_brier(self, name: str, success: bool) -> None:
        """Record a prediction-outcome pair for Brier score diagnostics.

        The predicted probability is the Beta distribution mean for this arm
        at the time of selection. The outcome is 1.0 (success) or 0.0.
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
        - 0.0 = perfect calibration
        - 0.25 = random baseline
        - 0.5 = worst possible
        """
        if not self._brier_predictions:
            return 0.0
        return sum((p - o) ** 2 for p, o in self._brier_predictions) / len(self._brier_predictions)

    def calibration_report(self) -> dict[str, float]:
        """Compute per-bin calibration: among predictions in [0,0.1), [0.1,0.2), etc.,
        what fraction actually succeeded? Returns bins where we have enough data."""
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
            report[f"{b * 10}-{b * 10 + 10}%"] = (mean_pred, mean_actual)
        return report

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

        # Snapshot previous distribution for JS comparison
        self._prev_byte_freq = {pos: dict(freq) for pos, freq in self.byte_freq.items()}

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

        # Compute JS divergence and adapt refit interval
        self.last_js_divergence = self._compute_js()
        self._adapt_interval()

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
        js_values = []
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

        Args:
            pos: Byte position in the input.

        Returns:
            Sampled byte value (0-255).
        """
        freq = self.byte_freq.get(pos)
        if not freq:
            return random.randint(0, 255)
        total = sum(freq.values()) + 256
        r = random.random() * total
        cumulative = 0
        for byte_val, count in freq.items():
            cumulative += count + 1
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

    def stationary_distribution(
        self, max_iter: int = 200, tol: float = 1e-8
    ) -> dict[str, float]:
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
            set(self.transition_total.keys()) | {
                op for targets in self.transition_counts.values() for op in targets
            }
        )
        n = len(operators)
        if n == 0:
            return {}
        if n == 1:
            return {operators[0]: 1.0}

        op_idx = {op: i for i, op in enumerate(operators)}

        # Build row-stochastic matrix P
        p_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for prev_op, total in self.transition_total.items():
            if total <= 0 or prev_op not in op_idx:
                continue
            i = op_idx[prev_op]
            targets = self.transition_counts.get(prev_op, {})
            for next_op, count in targets.items():
                if next_op in op_idx:
                    p_matrix[i][op_idx[next_op]] = count / total

        # Handle absorbing states
        for i in range(n):
            if sum(p_matrix[i]) < 1e-12:
                p_matrix[i][i] = 1.0

        # Power iteration
        pi = [1.0 / n] * n
        for _ in range(max_iter):
            new_pi = [0.0] * n
            for j in range(n):
                for i in range(n):
                    new_pi[j] += pi[i] * p_matrix[i][j]
            total = sum(new_pi)
            if total > 0:
                new_pi = [x / total for x in new_pi]
            diff = sum(abs(a - b) for a, b in zip(pi, new_pi, strict=False))
            pi = new_pi
            if diff < tol:
                break

        return {op: pi[op_idx[op]] for op in operators}

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
            set(self.transition_total.keys()) | {
                op for targets in self.transition_counts.values() for op in targets
            }
        )
        n = len(operators)
        if n <= 1:
            return 1.0

        op_idx = {op: i for i, op in enumerate(operators)}

        p_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for prev_op, total in self.transition_total.items():
            if total <= 0 or prev_op not in op_idx:
                continue
            i = op_idx[prev_op]
            targets = self.transition_counts.get(prev_op, {})
            for next_op, count in targets.items():
                if next_op in op_idx:
                    p_matrix[i][op_idx[next_op]] = count / total

        for i in range(n):
            if sum(p_matrix[i]) < 1e-12:
                p_matrix[i][i] = 1.0

        # Power iteration for dominant eigenvector
        v = [1.0 / n] * n
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

        # Deflate: P_deflated = P - v * v^T
        deflated: list[list[float]] = [
            [p_matrix[i][j] - v[i] * v[j] for j in range(n)]
            for i in range(n)
        ]

        # Power iteration on deflated matrix for λ₂
        w = [random.random() for _ in range(n)]
        norm = math.sqrt(sum(x * x for x in w))
        w = [x / norm for x in w]

        eigenvalue2 = 0.0
        for _ in range(max_iter):
            new_w = [0.0] * n
            for j in range(n):
                for i in range(n):
                    new_w[j] += deflated[i][j] * w[i]
            dot = sum(a * b for a, b in zip(w, new_w, strict=False))
            eigenvalue2 = abs(dot)
            norm = math.sqrt(sum(x * x for x in new_w))
            if norm < 1e-12:
                break
            new_w = [x / norm for x in new_w]
            diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(w, new_w, strict=False)))
            w = new_w
            if diff < tol:
                break

        return max(0.0, min(1.0, 1.0 - eigenvalue2))

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
        cov_matrix = [
            [cov[ops[i]].get(ops[j], 0.0) for j in range(n)]
            for i in range(n)
        ]

        chol = self._chol(cov_matrix)
        if chol is None:
            return self._standard_thompson(ops)

        z = [random.gauss(0, 1) for _ in range(n)]
        noise = [0.0] * n
        for i in range(n):
            for j in range(i + 1):
                noise[i] += chol[i][j] * z[j]

        scores = {}
        for i, op in enumerate(ops):
            a = self.arm_alpha.get(op, 1.0)
            b = self.arm_beta.get(op, 1.0)
            scores[op] = a / (a + b) + noise[i]

        return max(ops, key=lambda o: scores[o])

    def _standard_thompson(self, ops: list[str]) -> str:
        """Pure Thompson sampling without correlation structure."""
        best_op = None
        best_val = -1.0
        for op in ops:
            a = self.arm_alpha.get(op, 1.0)
            b = self.arm_beta.get(op, 1.0)
            val = random.betavariate(a, b)
            if val > best_val:
                best_val = val
                best_op = op
        return best_op if best_op is not None else ops[0]

    @staticmethod
    def _chol(matrix: list[list[float]]) -> list[list[float]] | None:
        """Cholesky decomposition with regularization.

        Decomposes A = L @ L^T where L is lower triangular.
        Returns None if decomposition fails after regularization.
        """
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

        l = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(l[i][k] * l[j][k] for k in range(j))
                if i == j:
                    val = a[i][i] - s
                    if val <= 0:
                        return None
                    l[i][j] = math.sqrt(val)
                else:
                    l[i][j] = (a[i][j] - s) / l[j][j] if l[j][j] > 0 else 0.0
        return l

    def matrix_ucb_select(
        self, ops: list[str], beta: float = 2.0, segment_size: int = 50
    ) -> str:
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
        if len(ops) < 3:
            return self._standard_ucb(ops, beta)

        means = {}
        for op in ops:
            a = self.arm_alpha.get(op, 1.0)
            b = self.arm_beta.get(op, 1.0)
            means[op] = a / (a + b)

        cov = self.operator_covariance(window=2000, segment_size=segment_size)
        if not cov or not all(op in cov for op in ops):
            return self._standard_ucb(ops, beta)

        n = len(ops)
        mu = [means[ops[i]] for i in range(n)]

        cov_matrix = [
            [cov[ops[i]].get(ops[j], 0.0) for j in range(n)]
            for i in range(n)
        ]

        chol = self._chol(cov_matrix)
        if chol is None:
            return self._standard_ucb(ops, beta)

        inv_cov = self._solve_cholesky(chol, [[1.0 if i == j else 0.0
                                                for j in range(n)]
                                               for i in range(n)])
        if inv_cov is None:
            return self._standard_ucb(ops, beta)

        base = 0.0
        for i in range(n):
            for j in range(n):
                base += mu[i] * inv_cov[i][j] * mu[j]

        t = sum(self.arm_alpha.values()) + sum(self.arm_beta.values()) - 2 * len(self.arm_alpha)
        t = max(t, 1)

        scores = {}
        for i, op in enumerate(ops):
            penalty = 0.0
            for j in range(n):
                penalty += inv_cov[i][j] * mu[j]
            penalty = base - 2 * penalty
            exploration = beta * math.sqrt(max(0.0, math.log(t) + penalty))
            scores[op] = mu[i] + exploration

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
        chol: list[list[float]], rhs: list[list[float]]
    ) -> list[list[float]] | None:
        """Solve L @ L^T @ X = rhs using forward/back substitution."""
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
        n_ops = len(operators)

        segments: list[list[float]] = []
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

        if len(segments) < 2:
            return {}

        n_seg = len(segments)
        means = [sum(s[i] for s in segments) / n_seg for i in range(n_ops)]

        cov_matrix: dict[str, dict[str, float]] = {op: {} for op in operators}

        for i, op_i in enumerate(operators):
            for j, op_j in enumerate(operators):
                if i == j:
                    var = sum((s[i] - means[i]) ** 2 for s in segments) / (n_seg - 1)
                    cov_matrix[op_i][op_j] = var
                elif i < j:
                    cov_val = (
                        sum((s[i] - means[i]) * (s[j] - means[j]) for s in segments)
                        / (n_seg - 1)
                    )
                    cov_matrix[op_i][op_j] = cov_val
                    cov_matrix[op_j][op_i] = cov_val

        return cov_matrix


class _MOptParticle:
    """A single particle in MOpt's PSO over operator probability space."""

    __slots__ = (
        "pos",
        "vel",
        "pbest_pos",
        "pbest_fitness",
        "fitness",
        "name",
        "discoveries",
        "execs_in_window",
    )

    def __init__(self, name: str, n_ops: int):
        self.name = name
        # Uniform initial distribution
        self.pos = [1.0 / n_ops] * n_ops
        self.vel = [0.0] * n_ops
        self.pbest_pos = list(self.pos)
        self.pbest_fitness = -1.0
        self.fitness = 0.0
        self.discoveries: collections.deque = collections.deque(maxlen=200)
        self.execs_in_window = 0


class MOptScheduler:
    """MOpt-style adaptive operator scheduling via Particle Swarm Optimization.

    Maintains K particles, each representing a probability distribution over
    mutation operators. PSO periodically re-optimizes these distributions based
    on recent discovery rate (new coverage per execution window).

    Key difference from Thompson sampling: PSO searches the joint configuration
    space — it can discover that operator combinations work well together,
    rather than evaluating each operator's marginal success independently.

    Args:
        n_particles: Number of PSO particles (default 5).
        window_size: Executions per fitness evaluation window.
        w: Inertia weight (momentum).
        c1: Cognitive coefficient (pull toward personal best).
        c2: Social coefficient (pull toward global best).
        max_vel: Maximum velocity magnitude.
    """

    def __init__(
        self,
        n_particles: int = 5,
        window_size: int = 200,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        max_vel: float = 0.2,
    ):
        self.n_particles = n_particles
        self.window_size = window_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_vel = max_vel

        self.operators: list[str] = []
        self.op_index: dict[str, int] = {}
        self.particles: list[_MOptParticle] = []
        self.global_best_pos: list[float] = []
        self.global_best_fitness = -1.0

        self._total_execs = 0
        self._total_discoveries = 0

    def init_arm(self, name: str) -> None:
        """Register a mutation operator. Rebuilds particles if operators changed."""
        if name in self.op_index:
            return
        idx = len(self.operators)
        self.operators.append(name)
        self.op_index[name] = idx

        # Build or extend particles for the current operator set
        self._rebuild_particles()

    def _rebuild_particles(self):
        """Rebuild all particles for the current operator set."""
        n = len(self.operators)
        old_particles = {p.name: p for p in self.particles}
        self.particles = []
        for i in range(self.n_particles):
            name = f"p{i}"
            if name in old_particles:
                old = old_particles[name]
                # Extend old distribution with small probability for new ops
                new_pos = list(old.pos) + [0.01] * (n - len(old.pos))
                total = sum(new_pos)
                new_pos = [p / total for p in new_pos]
                p = _MOptParticle(name, n)
                p.pos = new_pos
                p.vel = [0.0] * n
                p.pbest_pos = list(new_pos)
            else:
                p = _MOptParticle(name, n)
            self.particles.append(p)
        if not self.global_best_pos or len(self.global_best_pos) != n:
            self.global_best_pos = [1.0 / n] * n

    def select_op(self, ops: list[str]) -> tuple[str, int]:
        """Select an operator using MOpt's PSO-guided selection.

        1. Evaluate particle fitness from recent discoveries
        2. Pick the best particle (or roulette-wheel select)
        3. Sample an operator from that particle's distribution

        Args:
            ops: Available mutation operators for this iteration.

        Returns:
            (operator_name, particle_index) — the particle index is needed
            by record() to attribute discoveries to the correct particle.
        """
        if not self.particles or not self.operators:
            return (ops[0] if ops else "", 0)

        # Update fitness for all particles
        for p in self.particles:
            self._update_fitness(p)

        # Select particle by fitness-proportional selection
        valid = [p for p in self.particles if any(p.pos)]
        if not valid:
            valid = self.particles
        fitnesses = [max(p.fitness, 0.001) for p in valid]
        total_f = sum(fitnesses)
        r = random.random() * total_f
        cumulative = 0.0
        selected_particle = valid[0]
        selected_idx = 0
        for i, (p, f) in enumerate(zip(valid, fitnesses, strict=False)):
            cumulative += f
            if r <= cumulative:
                selected_particle = p
                selected_idx = self.particles.index(p)
                break

        # Sample operator from selected particle's distribution
        op = self._sample_from_particle(selected_particle, ops)
        return (op, selected_idx)

    def _sample_from_particle(self, particle: _MOptParticle, ops: list[str]) -> str:
        """Sample an operator from a particle's probability distribution."""
        # Build distribution over available ops
        probs = []
        for op in ops:
            idx = self.op_index.get(op, -1)
            if idx >= 0 and idx < len(particle.pos):
                probs.append(particle.pos[idx])
            else:
                probs.append(0.0)

        total = sum(probs)
        if total <= 0:
            return random.choice(ops)

        r = random.random() * total
        cumulative = 0.0
        for op, p in zip(ops, probs, strict=False):
            cumulative += p
            if r <= cumulative:
                return op
        return ops[-1]

    def record(self, name: str, success: bool, particle_id: int | None = None) -> None:
        """Record outcome for fitness tracking.

        Args:
            name: Operator that was used.
            success: Whether it produced new coverage.
            particle_id: Index of the particle that selected this operator.
                When None (backward compat), updates all particles.
        """
        self._total_execs += 1
        if success:
            self._total_discoveries += 1

        # Record discovery only in the particle that selected this operator.
        # This is the core fix: each particle's fitness reflects only the
        # outcomes of operators IT chose, enabling PSO to differentiate.
        if particle_id is not None and 0 <= particle_id < len(self.particles):
            p = self.particles[particle_id]
            p.execs_in_window += 1
            p.discoveries.append(1 if success else 0)
        else:
            # Backward compat: update all particles
            for p in self.particles:
                p.execs_in_window += 1
                p.discoveries.append(1 if success else 0)

        # Trigger PSO update when window fills
        if self._total_execs % self.window_size == 0 and self._total_execs > 0:
            self._pso_update()

    def _update_fitness(self, particle: _MOptParticle):
        """Compute particle fitness from its discovery window."""
        if not particle.discoveries or particle.execs_in_window == 0:
            particle.fitness = 0.0
            return
        # Fitness = discovery rate in the window, smoothed
        disc = sum(particle.discoveries)
        total = max(particle.execs_in_window, 1)
        particle.fitness = disc / total

    def _pso_update(self):
        """Run one PSO iteration: update velocities and positions."""
        # Find global best
        for p in self.particles:
            self._update_fitness(p)
            if p.fitness > self.global_best_fitness:
                self.global_best_fitness = p.fitness
                self.global_best_pos = list(p.pos)

        n = len(self.operators)
        if n == 0:
            return

        for p in self.particles:
            # Update velocity: v = w*v + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)
            for i in range(n):
                r1 = random.random()
                r2 = random.random()
                cognitive = self.c1 * r1 * (p.pbest_pos[i] - p.pos[i])
                social = self.c2 * r2 * (self.global_best_pos[i] - p.pos[i])
                p.vel[i] = self.w * p.vel[i] + cognitive + social
                # Clamp velocity
                p.vel[i] = max(-self.max_vel, min(self.max_vel, p.vel[i]))

            # Update position: pos += vel
            for i in range(n):
                p.pos[i] += p.vel[i]

            # Project back to simplex (softmax normalization)
            self._normalize_to_simplex(p)

            # Update personal best
            if p.fitness > p.pbest_fitness:
                p.pbest_fitness = p.fitness
                p.pbest_pos = list(p.pos)

            # Decay window for next iteration
            p.execs_in_window = 0
            p.discoveries.clear()

    def _normalize_to_simplex(self, particle: _MOptParticle):
        """Project velocity-pushed position onto the probability simplex.

        Uses softmax: p_i = exp(x_i) / sum(exp(x_j)).
        This ensures all probabilities are positive and sum to 1.
        """
        # Subtract max for numerical stability
        max_val = max(particle.pos) if particle.pos else 0.0
        exps = [math.exp(x - max_val) for x in particle.pos]
        total = sum(exps)
        if total > 0:
            particle.pos = [e / total for e in exps]
        else:
            n = len(particle.pos)
            particle.pos = [1.0 / n] * n

        # Ensure minimum probability floor (exploration)
        floor = 0.01
        for i in range(len(particle.pos)):
            particle.pos[i] = max(particle.pos[i], floor)
        total = sum(particle.pos)
        particle.pos = [p / total for p in particle.pos]

    def particle_stats(self) -> list[dict]:
        """Get stats for each particle (for diagnostics/logging)."""
        result = []
        for p in self.particles:
            self._update_fitness(p)
            # Find which operator has highest probability
            if p.pos and self.operators:
                best_idx = max(range(len(p.pos)), key=lambda i: p.pos[i])
                best_op = self.operators[best_idx] if best_idx < len(self.operators) else "?"
            else:
                best_op = "?"
            result.append(
                {
                    "name": p.name,
                    "fitness": round(p.fitness, 4),
                    "pbest": round(p.pbest_fitness, 4),
                    "top_op": best_op,
                    "top_prob": round(max(p.pos), 3) if p.pos else 0.0,
                }
            )
        return result

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Compatibility with MonteCarloScheduler interface.

        Returns discovery/failure counts from the global window.
        """
        return {
            "_mopt_global": (
                self._total_discoveries,
                self._total_execs - self._total_discoveries,
            )
        }


# ---------------------------------------------------------------------------
# Shapley value for fair operator attribution
# ---------------------------------------------------------------------------


class ShapleyAttribution:
    """Compute Shapley values for mutation operator contribution.

    The Shapley value fairly distributes credit among operators when
    multiple ops contribute to discoveries. It considers all possible
    orderings of operators and computes the average marginal contribution.

    For fuzzer scheduling, this answers: "if we removed operator X from
    the pool, how much would total coverage drop?" — accounting for
    synergistic effects between operators.

    Uses sampling-based estimation (not exact enumeration) for efficiency.
    With N operators, exact Shapley requires 2^N evaluations. Sampling
    K random permutations gives estimates within epsilon with high probability.

    Args:
        n_samples: Number of random permutations to sample.
        window_size: Number of recent outcomes to consider.
    """

    def __init__(self, n_samples: int = 100, window_size: int = 500):
        self.n_samples = n_samples
        self.window_size = window_size
        # Recent outcomes: list of (operators_used_set, discovered_edges_count)
        self._outcomes: collections.deque = collections.deque(maxlen=window_size)
        # Per-operator: set of edges this operator contributed to
        self._operator_edges: dict[str, set[int]] = defaultdict(set)
        # Global edge set for marginal computation
        self._all_edges: set[int] = set()

    def record(
        self, operators: set[str], new_edges: int, edge_indices: set[int] | None = None
    ) -> None:
        """Record an execution outcome.

        Args:
            operators: Set of mutation operators used in this execution.
            new_edges: Number of new edges discovered (0 if none).
            edge_indices: Optional set of specific new edge indices.
        """
        self._outcomes.append((operators, new_edges))
        if edge_indices:
            for op in operators:
                self._operator_edges[op].update(edge_indices)
            self._all_edges.update(edge_indices)

    def shapley_values(self, operators: list[str] | None = None) -> dict[str, float]:
        """Compute Shapley values for each operator.

        Uses sampling: for each random permutation, compute the marginal
        contribution of each operator (edges uniquely attributable to it
        given the operators before it in the permutation).

        Returns:
            Dict mapping operator name -> Shapley value (in [0, 1]).
            Values sum to 1.0 (or less if some operators have zero contribution).
        """
        if not self._outcomes:
            return {op: 1.0 / max(1, len(operators or [])) for op in (operators or [])}

        if operators is None:
            operators = sorted({op for ops, _ in self._outcomes for op in ops})
        if not operators:
            return {}

        n_ops = len(operators)
        shapley = {op: 0.0 for op in operators}

        for _ in range(self.n_samples):
            perm = operators[:]
            random.shuffle(perm)

            prefix_edges: set[int] = set()
            for op in perm:
                # Marginal contribution = edges this op adds beyond prefix
                op_edges = self._operator_edges.get(op, set())
                marginal = len(op_edges - prefix_edges)
                shapley[op] += marginal
                prefix_edges.update(op_edges)

        # Normalize to [0, 1]
        total = sum(shapley.values())
        if total > 0:
            shapley = {op: v / total for op, v in shapley.items()}
        else:
            shapley = {op: 1.0 / n_ops for op in operators}

        return shapley

    def operator_synergy(self, op_a: str, op_b: str) -> float:
        """Compute synergy between two operators.

        Synergy = I(X_a, X_b; Y) - I(X_a; Y) - I(X_b; Y)
        where X_a, X_b are operator usage indicators and Y is coverage.

        Positive = operators work better together than alone.
        Negative = operators are redundant.
        """
        edges_a = self._operator_edges.get(op_a, set())
        edges_b = self._operator_edges.get(op_b, set())
        if not edges_a or not edges_b:
            return 0.0

        # Approximate: joint coverage minus individual coverages
        joint = len(edges_a | edges_b)
        individual = len(edges_a) + len(edges_b)
        return (joint - individual) / max(1, individual)

    def operator_kernel(
        self, operators: list[str] | None = None
    ) -> dict[str, dict[str, float]]:
        """Build a kernel matrix measuring operator similarity via Jaccard.

        K(i,j) = |E_i ∩ E_j| / |E_i ∪ E_j|
        High K → redundant operators. Low K → complementary.

        Args:
            operators: Operators to include. If None, uses all.

        Returns:
            Nested dict: kernel[op_a][op_b] = Jaccard similarity in [0, 1].
        """
        if operators is None:
            operators = sorted(self._operator_edges.keys())
        if len(operators) < 2:
            return {op: {op: 1.0} for op in operators}

        kernel: dict[str, dict[str, float]] = {op: {} for op in operators}

        for i, op_i in enumerate(operators):
            edges_i = self._operator_edges.get(op_i, set())
            for j, op_j in enumerate(operators):
                if i == j:
                    kernel[op_i][op_j] = 1.0
                elif i < j:
                    edges_j = self._operator_edges.get(op_j, set())
                    if not edges_i and not edges_j:
                        sim = 0.0
                    else:
                        intersection = len(edges_i & edges_j)
                        union = len(edges_i | edges_j)
                        sim = intersection / union if union > 0 else 0.0
                    kernel[op_i][op_j] = sim
                    kernel[op_j][op_i] = sim

        return kernel

    def operator_similarity(self, op_a: str, op_b: str) -> float:
        """Compute Jaccard similarity between two operators."""
        edges_a = self._operator_edges.get(op_a, set())
        edges_b = self._operator_edges.get(op_b, set())
        if not edges_a and not edges_b:
            return 0.0
        intersection = len(edges_a & edges_b)
        union = len(edges_a | edges_b)
        return intersection / union if union > 0 else 0.0

    def redundant_operators(
        self, threshold: float = 0.9, operators: list[str] | None = None
    ) -> list[tuple[str, str, float]]:
        """Find pairs of operators that are near-duplicates.

        Returns pairs where K(i,j) >= threshold, sorted by similarity.

        Args:
            threshold: Minimum Jaccard similarity to consider redundant.
            operators: Operators to check. If None, uses all.

        Returns:
            List of (op_a, op_b, similarity) tuples.
        """
        kernel = self.operator_kernel(operators)
        pairs = []
        seen = set()
        for op_a in kernel:
            for op_b in kernel[op_a]:
                if op_a == op_b:
                    continue
                key = (min(op_a, op_b), max(op_a, op_b))
                if key in seen:
                    continue
                seen.add(key)
                sim = kernel[op_a][op_b]
                if sim >= threshold:
                    pairs.append((op_a, op_b, sim))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def spectral_embedding(
        self, operators: list[str] | None = None, k: int = 2
    ) -> dict[str, list[float]]:
        """Spectral embedding of operators using Laplacian eigenmap.

        Returns low-dimensional coordinates where similar operators cluster.

        Args:
            operators: Operators to embed. If None, uses all.
            k: Number of embedding dimensions.

        Returns:
            Dict mapping operator name -> [dim_0, dim_1, ...] coordinates.
        """
        if operators is None:
            operators = sorted(self._operator_edges.keys())
        n = len(operators)
        if n < k + 1:
            return {op: [0.0] * k for op in operators}

        kernel = self.operator_kernel(operators)

        degrees = [0.0] * n
        for i in range(n):
            for j in range(n):
                degrees[i] += kernel[operators[i]].get(operators[j], 0.0)

        # Normalized Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}
        laplacian: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                w_ij = kernel[operators[i]].get(operators[j], 0.0)
                d_i = math.sqrt(degrees[i]) if degrees[i] > 0 else 1.0
                d_j = math.sqrt(degrees[j]) if degrees[j] > 0 else 1.0
                laplacian[i][j] = -w_ij / (d_i * d_j)
            laplacian[i][i] = 1.0

        # Inverse iteration with deflation for k smallest eigenvectors
        eigenvectors: list[list[float]] = []

        for _ in range(k):
            w = [random.gauss(0, 1) for _ in range(n)]
            norm = math.sqrt(sum(x * x for x in w))
            w = [x / norm for x in w]

            for _ in range(100):
                lw = [0.0] * n
                for i in range(n):
                    for j in range(n):
                        lw[i] += laplacian[i][j] * w[j]
                new_w = [w[i] - 0.5 * lw[i] for i in range(n)]
                norm = math.sqrt(sum(x * x for x in new_w))
                if norm < 1e-12:
                    break
                new_w = [x / norm for x in new_w]
                diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(w, new_w, strict=False)))
                w = new_w
                if diff < 1e-8:
                    break

            eigenvectors.append(w)

            for i in range(n):
                for j in range(n):
                    laplacian[i][j] -= w[i] * w[j]

        embedding: dict[str, list[float]] = {}
        for idx, op in enumerate(operators):
            embedding[op] = [eigenvectors[d][idx] for d in range(k)]

        return embedding

    def ranking(self, operators: list[str] | None = None) -> list[tuple[str, float]]:
        """Return operators ranked by Shapley value.

        Returns:
            List of (operator, shapley_value) sorted descending.
        """
        sv = self.shapley_values(operators)
        return sorted(sv.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Replicator dynamics for operator scheduling
# ---------------------------------------------------------------------------


class ReplicatorScheduler:
    """Operator scheduling via evolutionary replicator dynamics.

    The replicator equation is the canonical dynamics of evolutionary game
    theory: x_i' = x_i * (f_i - phi) where f_i is operator i's fitness
    and phi is the population-average fitness. Operators above average
    grow; those below shrink.

    Unlike Thompson sampling (which models each arm independently) or PSO
    (which searches joint distributions), replicator dynamics models the
    *population* of operators as a game. The equilibrium is a Nash
    equilibrium of the mutation game.

    Advantages over Thompson sampling:
    - Naturally handles operator interactions (via fitness defined on combinations)
    - Converges to evolutionarily stable strategies (ESS), not just best responses
    - Population dynamics are smooth and interpretable

    Args:
        window_size: Executions per fitness evaluation.
        learning_rate: Replicator step size (eta). Smaller = smoother.
        mutation_rate: Minimum probability floor (exploration guarantee).
    """

    def __init__(
        self,
        window_size: int = 200,
        learning_rate: float = 0.1,
        mutation_rate: float = 0.02,
    ):
        self.window_size = window_size
        self.eta = learning_rate
        self.mutation_rate = mutation_rate

        self.operators: list[str] = []
        self.op_index: dict[str, int] = {}
        # Population distribution over operators (probability simplex)
        self.population: list[float] = []
        # Fitness tracking per operator per window
        self._fitness_sum: dict[str, float] = defaultdict(float)
        self._fitness_count: dict[str, int] = defaultdict(int)
        self._execs_in_window = 0
        self._total_execs = 0
        self._total_discoveries = 0
        # History of distributions for convergence diagnostics
        self._history: collections.deque = collections.deque(maxlen=100)

    def init_arm(self, name: str) -> None:
        """Register a mutation operator. Rebuilds population if operators changed."""
        if name in self.op_index:
            return
        idx = len(self.operators)
        self.operators.append(name)
        self.op_index[name] = idx
        # Extend population with uniform distribution
        n = len(self.operators)
        self.population = [1.0 / n] * n

    def select_op(self, ops: list[str]) -> str:
        """Select an operator from the replicator distribution.

        Args:
            ops: Available operators for this iteration.

        Returns:
            Name of the selected operator.
        """
        if not self.population or not self.operators:
            return ops[0] if ops else ""

        # Build probability vector over available ops
        probs = []
        for op in ops:
            idx = self.op_index.get(op, -1)
            if idx >= 0 and idx < len(self.population):
                probs.append(self.population[idx])
            else:
                probs.append(0.0)

        total = sum(probs)
        if total <= 0:
            return random.choice(ops)

        # Roulette wheel selection
        r = random.random() * total
        cumulative = 0.0
        for op, p in zip(ops, probs, strict=False):
            cumulative += p
            if r <= cumulative:
                return op
        return ops[-1]

    def record(self, name: str, success: bool) -> None:
        """Record outcome and trigger replicator update when window fills.

        Args:
            name: Operator that was used.
            success: Whether it produced new coverage.
        """
        self._total_execs += 1
        if success:
            self._total_discoveries += 1

        self._execs_in_window += 1
        self._fitness_sum[name] += 1.0 if success else 0.0
        self._fitness_count[name] += 1

        if self._execs_in_window >= self.window_size:
            self._replicator_update()

    def _replicator_update(self):
        """Run one replicator dynamics step.

        x_i(t+1) = x_i(t) * (1 + eta * (f_i - phi))

        where:
        - x_i = population share of operator i
        - f_i = fitness (success rate) of operator i in this window
        - phi = average fitness across all operators
        - eta = learning rate
        """
        n = len(self.operators)
        if n == 0:
            return

        # Compute fitness for each operator
        fitness = []
        for op in self.operators:
            count = self._fitness_count.get(op, 0)
            if count > 0:
                fitness.append(self._fitness_sum[op] / count)
            else:
                fitness.append(0.0)

        # Average fitness (weighted by population)
        phi = (
            sum(x * f for x, f in zip(self.population, fitness, strict=False))
            if self.population
            else 0.0
        )

        # Replicator step: x_i' = x_i * (1 + eta * (f_i - phi))
        new_pop = []
        for i in range(n):
            growth = 1.0 + self.eta * (fitness[i] - phi)
            new_pop.append(max(0.0, self.population[i] * growth))

        # Normalize to simplex
        total = sum(new_pop)
        new_pop = [x / total for x in new_pop] if total > 0 else [1.0 / n] * n

        # Enforce mutation floor (exploration guarantee)
        # Apply floor, then renormalize, then re-apply floor iteratively
        # to handle the case where normalization pushes values below floor
        for _ in range(3):
            for i in range(n):
                new_pop[i] = max(new_pop[i], self.mutation_rate)
            total = sum(new_pop)
            if total > 0:
                new_pop = [x / total for x in new_pop]

        self.population = new_pop
        self._history.append(list(new_pop))

        # Reset window counters
        self._execs_in_window = 0
        self._fitness_sum.clear()
        self._fitness_count.clear()

    def is_converged(self, threshold: float = 0.01) -> bool:
        """Check if the population distribution has converged.

        Convergence is detected when the last N distributions have
        low variance (population shares barely change).

        Args:
            threshold: Maximum standard deviation across recent distributions
                       to consider converged.

        Returns:
            True if converged.
        """
        if len(self._history) < 5:
            return False

        recent = list(self._history)[-5:]
        # For each operator position, compute std dev across recent distributions
        n_ops = len(self.operators)
        for i in range(n_ops):
            values = [h[i] for h in recent if i < len(h)]
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            if variance**0.5 > threshold:
                return False
        return True

    def dominant_operator(self) -> str | None:
        """Return the operator with highest population share."""
        if not self.population or not self.operators:
            return None
        best_idx = max(range(len(self.population)), key=lambda i: self.population[i])
        return self.operators[best_idx]

    def population_distribution(self) -> dict[str, float]:
        """Return current population as a dict."""
        return {op: self.population[i] for i, op in enumerate(self.operators)}

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Compatibility with MonteCarloScheduler interface."""
        return {
            "_replicator_global": (
                self._total_discoveries,
                self._total_execs - self._total_discoveries,
            )
        }

    def operator_stats(self) -> list[dict]:
        """Get stats for each operator (for diagnostics/logging)."""
        result = []
        for i, op in enumerate(self.operators):
            pop = self.population[i] if i < len(self.population) else 0.0
            count = self._fitness_count.get(op, 0)
            successes = self._fitness_sum.get(op, 0)
            result.append(
                {
                    "name": op,
                    "population": round(pop, 4),
                    "window_successes": int(successes),
                    "window_execs": count,
                }
            )
        return result
