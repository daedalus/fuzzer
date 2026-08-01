"""MOptScheduler: Particle Swarm Optimization over operator probability distributions.

An alternative to Thompson sampling that searches the joint configuration
space rather than each operator's marginal success rate.
"""

import collections
import math
import random


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
        for _, (p, f) in enumerate(zip(valid, fitnesses, strict=False)):
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

    def record(
        self, name: str, success: bool, particle_id: int | None = None, weight: float = 1.0
    ) -> None:
        """Record outcome for fitness tracking.

        Args:
            name: Operator that was used.
            success: Whether it produced new coverage.
            particle_id: Index of the particle that selected this operator.
                When None (backward compat), updates all particles.
            weight: Reward weight (default 1.0). Surprisal-weighted calls
                pass a value in (0, 1] proportional to discovery rarity.
        """
        self._total_execs += 1
        if success:
            self._total_discoveries += 1

        # Record discovery only in the particle that selected this operator.
        # This is the core fix: each particle's fitness reflects only the
        # outcomes of operators IT chose, enabling PSO to differentiate.
        reward = weight if success else 0.0
        if particle_id is not None and 0 <= particle_id < len(self.particles):
            p = self.particles[particle_id]
            p.execs_in_window += 1
            p.discoveries.append(reward)
        else:
            # Backward compat: update all particles
            for p in self.particles:
                p.execs_in_window += 1
                p.discoveries.append(reward)

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
