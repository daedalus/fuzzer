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

    def __init__(self, elite_frac: float = 0.1, refit_interval: int = 1000):
        self.arm_alpha: dict[str, float] = {}
        self.arm_beta: dict[str, float] = {}
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

    def init_arm(self, name: str) -> None:
        """Register a mutation operator arm with prior (1, 1).

        Args:
            name: Name of the mutation operator.
        """
        if name not in self.arm_alpha:
            self.arm_alpha[name] = 1.0
            self.arm_beta[name] = 1.0

    def select_op(self, ops: list[str]) -> str:
        """Select mutation operator via Thompson sampling.

        Args:
            ops: Available mutation operators.

        Returns:
            Name of the selected operator.
        """
        best_op = ops[0]
        best_val = -1.0
        for op in ops:
            a = self.arm_alpha.get(op, 1.0)
            b = self.arm_beta.get(op, 1.0)
            val = random.betavariate(a, b)
            if val > best_val:
                best_val = val
                best_op = op
        return best_op

    def record(self, name: str, success: bool) -> None:
        """Record outcome for a mutation operator arm.

        Args:
            name: Name of the mutation operator.
            success: Whether the mutation produced an interesting result.
        """
        if success:
            self.arm_alpha[name] = self.arm_alpha.get(name, 1.0) + 1
        else:
            self.arm_beta[name] = self.arm_beta.get(name, 1.0) + 1

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
            report[f"{b*10}-{b*10+10}%"] = (mean_pred, mean_actual)
        return report

    def add_elite(self, data: bytes, score: int) -> None:
        """Add an input to the elite set for CEM fitting.

        Args:
            data: The input bytes.
            score: Quality score (higher is better).
        """
        self.elite_set.append((score, data))
        if len(self.elite_set) > self.ELITE_MAX:
            self.elite_set.sort(key=lambda x: x[0])
            self.elite_set.pop(0)

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
        self._prev_byte_freq = {
            pos: dict(freq) for pos, freq in self.byte_freq.items()
        }

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
                pa * math.log(pa / b[k])
                for k, pa in a.items()
                if pa > 0.0 and b.get(k, 0.0) > 0.0
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
            result[name] = (a - 1, b - 1)
        return result


class _MOptParticle:
    """A single particle in MOpt's PSO over operator probability space."""

    __slots__ = ("pos", "vel", "pbest_pos", "pbest_fitness", "fitness",
                 "name", "discoveries", "execs_in_window")

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

    def select_op(self, ops: list[str]) -> str:
        """Select an operator using MOpt's PSO-guided selection.

        1. Evaluate particle fitness from recent discoveries
        2. Pick the best particle (or roulette-wheel select)
        3. Sample an operator from that particle's distribution

        Args:
            ops: Available mutation operators for this iteration.

        Returns:
            Name of the selected operator.
        """
        if not self.particles or not self.operators:
            return ops[0] if ops else ""

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
        for p, f in zip(valid, fitnesses, strict=False):
            cumulative += f
            if r <= cumulative:
                selected_particle = p
                break

        # Sample operator from selected particle's distribution
        return self._sample_from_particle(selected_particle, ops)

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

    def record(self, name: str, success: bool) -> None:
        """Record outcome for fitness tracking.

        Args:
            name: Operator that was used.
            success: Whether it produced new coverage.
        """
        self._total_execs += 1
        if success:
            self._total_discoveries += 1

        # Record discovery in all particles' windows
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
            result.append({
                "name": p.name,
                "fitness": round(p.fitness, 4),
                "pbest": round(p.pbest_fitness, 4),
                "top_op": best_op,
                "top_prob": round(max(p.pos), 3) if p.pos else 0.0,
            })
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
