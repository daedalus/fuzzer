"""MOptScheduler: Particle Swarm Optimization over operator probability distributions.

An alternative to Thompson sampling that searches the joint configuration
space rather than each operator's marginal success rate.
"""

import collections
import random
from collections import defaultdict

#: Fractional jitter applied to initial particle positions. Enough to give
#: PSO a gradient; small enough that no operator starts strongly favoured.
_INIT_SPREAD = 0.5


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

    def __init__(self, name: str, n_ops: int, spread: float = _INIT_SPREAD):
        self.name = name
        # Randomized initial distribution, jittered around uniform.
        #
        # Every particle used to start at exactly 1/n with zero velocity.
        # That makes PSO a no-op by construction: pbest and gbest both equal
        # the particle's own position, so the cognitive term
        # c1*r1*(pbest - pos) and the social term c2*r2*(gbest - pos) are
        # both identically zero, velocity stays zero forever, and no particle
        # ever moves. Measured after 6000 executions: all five particles bit-
        # identical, every velocity component exactly 0.0. A swarm needs
        # positional diversity to generate a gradient; that is what makes it
        # a swarm.
        if n_ops > 0:
            raw = [1.0 + spread * (random.random() * 2.0 - 1.0) for _ in range(n_ops)]
            total = sum(raw)
            self.pos = [x / total for x in raw]
        else:
            self.pos = []
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
        min_prob_frac: Exploration floor, as a fraction of the uniform
            probability ``1/n``. Total floored mass is ``min_prob_frac``
            regardless of how many operators are registered.
        gbest_decay: Per-window decay of the global-best *fitness* record.
            Without it the swarm's attractor is whatever position ever
            scored highest, so an operator that saturates its region of the
            coverage map keeps pulling the swarm toward itself forever.
    """

    # Declares that init_arm() does NOT accept informative priors (PSO
    # carries arm state in particle positions, not Beta-Bernoulli counts).
    supports_priors = False

    def __init__(
        self,
        n_particles: int = 5,
        window_size: int = 200,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        c3: float = 1.5,
        max_vel: float = 0.2,
        min_prob_frac: float = 0.1,
        gbest_decay: float = 0.95,
    ):
        self.n_particles = n_particles
        self.window_size = window_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.max_vel = max_vel
        self.min_prob_frac = min_prob_frac
        self.gbest_decay = gbest_decay

        self.operators: list[str] = []
        self.op_index: dict[str, int] = {}
        self.particles: list[_MOptParticle] = []
        self.global_best_pos: list[float] = []
        self.global_best_fitness = -1.0

        self._total_execs = 0
        self._total_discoveries = 0
        # Per-operator window counters. The swarm previously had no
        # per-operator signal at all: fitness was measured per *particle*, so
        # with five particles that start near-identical the fitness spread is
        # sampling noise on ~40 draws and PSO is searching a 12-dimensional
        # simplex blind. MOpt (Lyu et al., USENIX Sec '19) steers the swarm by
        # each operator's measured efficiency; that vector is reconstructed
        # here and used as a third attractor alongside pbest and gbest.
        self._op_execs: dict[str, int] = defaultdict(int)
        self._op_disc: dict[str, float] = defaultdict(float)
        self._efficiency_pos: list[float] = []

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
                # Extend with the *uniform* share for newly registered ops,
                # preserving relative weights among the ones already known.
                #
                # This used to hand each new operator a flat 0.01. Because
                # init_arm() registers operators one at a time and rebuilds on
                # every call, the first operator registered kept compounding:
                # after twelve registrations it held ~0.90 of the mass and the
                # twelfth held ~0.008, before a single execution had run. The
                # swarm then "converged" on whatever was registered first
                # regardless of its reward. The old softmax projection masked
                # this by crushing every distribution back toward uniform.
                n_old = len(old.pos)
                n_new = n - n_old
                keep = n_old / n if n > 0 else 1.0
                old_total = sum(old.pos) or 1.0
                # New entries are jittered, not set to a flat 1/n. init_arm()
                # registers operators one at a time and rebuilds every time, so
                # a flat share here reconstructs the exactly-uniform vector on
                # every call and destroys the randomized initialization above —
                # re-freezing the swarm no matter how it was seeded.
                new_pos = [x / old_total * keep for x in old.pos] + [
                    (1.0 / n) * (1.0 + _INIT_SPREAD * (random.random() * 2.0 - 1.0))
                    for _ in range(n_new)
                ]
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
        # Floor relative to the best particle rather than at an absolute
        # 0.001. An absolute floor is not a floor once any particle exceeds
        # ~0.1 fitness: the leader takes >99% of executions and the rest never
        # collect enough samples to challenge it.
        best_f = max((p.fitness for p in valid), default=0.0)
        floor = max(0.1 * best_f, 0.001)
        fitnesses = [max(p.fitness, floor) for p in valid]
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
        self._op_execs[name] += 1
        self._op_disc[name] += reward
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
        """Compute particle fitness from its discovery window.

        A particle that received no executions this window keeps its previous
        fitness rather than being reset to zero. Zeroing it was self-
        reinforcing: fitness-proportional particle selection starves the
        low-fitness particles, starvation empties their windows, empty windows
        zero their fitness, and the swarm collapses to whichever particle first
        got a lucky window. Measured: three of five particles pinned at exactly
        0.0 for the whole campaign.
        """
        if not particle.discoveries or particle.execs_in_window == 0:
            return
        # Fitness = discovery rate in the window, smoothed
        disc = sum(particle.discoveries)
        total = max(particle.execs_in_window, 1)
        particle.fitness = disc / total

    def _efficiency_distribution(self, n: int) -> list[float]:
        """Normalize this window's per-operator discovery rates to the simplex.

        Operators with no executions this window are held at the uniform share
        rather than zero, so a temporarily unsampled operator is treated as
        unknown rather than as known-bad.
        """
        uniform = 1.0 / n
        raw = []
        for op in self.operators:
            execs = self._op_execs.get(op, 0)
            raw.append(self._op_disc.get(op, 0.0) / execs if execs else uniform)
        total = sum(raw)
        if total <= 0.0:
            return [uniform] * n
        self._efficiency_pos = [x / total for x in raw]
        return self._efficiency_pos

    def _pso_update(self):
        """Run one PSO iteration: update velocities and positions."""
        n = len(self.operators)
        if n == 0:
            return

        # Personal and global bests must be recorded *before* the velocity
        # step, against the position that actually earned the fitness. The
        # pbest update used to run at the bottom of the loop below, which
        # paired the old fitness with the already-moved position and taught
        # each particle to steer toward a point it had never evaluated.
        self.global_best_fitness *= self.gbest_decay
        for p in self.particles:
            self._update_fitness(p)
            if p.fitness > p.pbest_fitness:
                p.pbest_fitness = p.fitness
                p.pbest_pos = list(p.pos)
            if p.fitness > self.global_best_fitness:
                self.global_best_fitness = p.fitness
                self.global_best_pos = list(p.pos)

        eff = self._efficiency_distribution(n)

        for p in self.particles:
            # v = w*v + c1*r1*(pbest - pos) + c2*r2*(gbest - pos)
            #         + c3*r3*(efficiency - pos)
            for i in range(n):
                r1 = random.random()
                r2 = random.random()
                r3 = random.random()
                cognitive = self.c1 * r1 * (p.pbest_pos[i] - p.pos[i])
                social = self.c2 * r2 * (self.global_best_pos[i] - p.pos[i])
                measured = self.c3 * r3 * (eff[i] - p.pos[i])
                p.vel[i] = self.w * p.vel[i] + cognitive + social + measured
                # Clamp velocity
                p.vel[i] = max(-self.max_vel, min(self.max_vel, p.vel[i]))

            # Update position: pos += vel
            for i in range(n):
                p.pos[i] += p.vel[i]

            # Project back onto the probability simplex
            self._normalize_to_simplex(p)

            # Decay window for next iteration
            p.execs_in_window = 0
            p.discoveries.clear()

        self._op_execs.clear()
        self._op_disc.clear()

    def _normalize_to_simplex(self, particle: _MOptParticle):
        """Project the velocity-pushed position back onto the probability simplex.

        Clips negatives and divides by the sum, then applies an exploration
        floor.

        This used to softmax instead, which made the scheduler incapable of
        converging: positions are *already* a probability distribution, so
        every entry lies in [0, 1] and softmax compresses the whole vector to
        within ``exp(max - min) <= e`` of uniform. With ``max_vel`` clamping
        how far a step can move an entry, no particle could ever concentrate —
        the scheduler measured as statistically indistinguishable from
        ``random.choice`` (tail share 0.080 against a 0.083 uniform baseline,
        linear regret). Softmax is the right projection for *unconstrained
        logits*, which is what ``CMAESScheduler`` keeps; it is the wrong one
        for a vector that is already normalized.

        The floor is a fraction of uniform rather than an absolute constant.
        A fixed 0.01 floor is harmless at the 12 operators a unit test uses,
        but the live registry has 135: ``0.01 * 135 = 1.35`` exceeds the whole
        simplex, so flooring-then-renormalizing returned exactly the uniform
        distribution and silently disabled PSO in production.
        """
        n = len(particle.pos)
        if n == 0:
            return

        clipped = [x if x > 0.0 else 0.0 for x in particle.pos]
        total = sum(clipped)
        particle.pos = [x / total for x in clipped] if total > 0.0 else [1.0 / n] * n

        floor = self.min_prob_frac / n
        if floor > 0.0:
            particle.pos = [max(x, floor) for x in particle.pos]
            total = sum(particle.pos)
            particle.pos = [x / total for x in particle.pos]

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
