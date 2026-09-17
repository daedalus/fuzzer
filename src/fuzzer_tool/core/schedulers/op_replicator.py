"""ReplicatorScheduler: replicator-dynamics bandit over the operator simplex."""

import collections
from collections import defaultdict

from fuzzer_tool.core.marginal_cost import MarginalCostTracker
from fuzzer_tool.core.rand_pool import RandPool


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
        marginal_cost_stop_multiplier: Optional extra shrink applied to an
            operator whose marginal cost (executions spent per new
            discovery, MC_i = Δexecs/Δdiscoveries between the last two
            windows -- see ``core/marginal_cost.py`` and
            ``docs/handover/handover_decision_game_theory_survey_2026-09-13.md``
            §1) exceeds this multiple of the population-average marginal
            cost. This targets the case the fixed ``window_size`` cutoff
            misses on its own: an operator whose cost-per-discovery is
            actively rising (past the U-shaped MC curve's turning point)
            gets penalized before its *average* fitness necessarily drops
            enough for the ordinary replicator update to notice. ``None``
            (the default) disables this entirely -- existing behavior is
            unchanged either way.
    """

    # Declares that init_arm() does NOT accept informative priors (arm
    # state lives in the population simplex, not Beta-Bernoulli counts).
    supports_priors = False

    def __init__(
        self,
        window_size: int = 200,
        learning_rate: float = 0.1,
        mutation_rate: float = 0.02,
        rng: RandPool | None = None,
        marginal_cost_stop_multiplier: float | None = None,
    ):
        self._rng = rng if rng is not None else RandPool()
        self.window_size = window_size
        self.eta = learning_rate
        self.mutation_rate = mutation_rate
        self.marginal_cost_stop_multiplier = marginal_cost_stop_multiplier

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

        # Cumulative (execs, discoveries) per operator -- the "cost" and
        # "output" counters the marginal-cost snapshots are taken from.
        # Kept separate from _fitness_sum/_fitness_count above because
        # those two reset every window (see _replicator_update) while a
        # marginal-cost *difference* needs the running total on both
        # sides of the window boundary.
        self._cum_execs: dict[str, int] = defaultdict(int)
        self._cum_discoveries: dict[str, float] = defaultdict(float)
        self._mc_tracker = MarginalCostTracker()

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
            return self._rng.choice(ops)

        # Roulette wheel selection
        r = self._rng.random() * total
        cumulative = 0.0
        for op, p in zip(ops, probs, strict=False):
            cumulative += p
            if r <= cumulative:
                return op
        return ops[-1]

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome and trigger replicator update when window fills.

        Args:
            name: Operator that was used.
            success: Whether it produced new coverage.
            weight: Reward weight (default 1.0). Surprisal-weighted calls
                pass a value in (0, 1] proportional to discovery rarity.
        """
        self._total_execs += 1
        if success:
            self._total_discoveries += 1

        self._execs_in_window += 1
        self._fitness_sum[name] += weight if success else 0.0
        self._fitness_count[name] += 1

        self._cum_execs[name] += 1
        self._cum_discoveries[name] += weight if success else 0.0

        if self._execs_in_window >= self.window_size:
            self._replicator_update()

    def _replicator_compute_fitness(self) -> tuple[list[float], list[bool]]:
        """Compute fitness vector and data-available flags."""
        fitness = []
        has_data = []
        for op in self.operators:
            count = self._fitness_count.get(op, 0)
            if count > 0:
                fitness.append(self._fitness_sum[op] / count)
                has_data.append(True)
            else:
                fitness.append(0.0)
                has_data.append(False)
        return fitness, has_data

    def _replicator_normalize_with_floor(self, new_pop: list[float], n: int) -> list[float]:
        """Normalize to simplex, enforce mutation floor iteratively."""
        total = sum(new_pop)
        new_pop = [x / total for x in new_pop] if total > 0 else [1.0 / n] * n
        for _ in range(3):
            for i in range(n):
                new_pop[i] = max(new_pop[i], self.mutation_rate)
            total = sum(new_pop)
            if total > 0:
                new_pop = [x / total for x in new_pop]
        return new_pop

    def _replicator_update(self):
        """Run one replicator dynamics step.

        x_i(t+1) = x_i(t) * (1 + eta * (f_i - phi))

        where:
        - x_i = population share of operator i
        - f_i = fitness (success rate) of operator i in this window
        - phi = average fitness across all operators
        - eta = learning rate

        Operators with zero trials are excluded from phi and receive
        neutral growth (fitness = phi), preventing starvation of
        conditionally-relevant operators.
        """
        n = len(self.operators)
        if n == 0:
            return

        fitness, has_data = self._replicator_compute_fitness()

        # Snapshot cumulative (execs, discoveries) for every operator at
        # this window boundary -- unconditionally, so the tracker has two
        # snapshots to diff by the *next* boundary even if
        # marginal_cost_stop_multiplier is turned on only later.
        for op in self.operators:
            self._mc_tracker.record_snapshot(op, self._cum_execs[op], self._cum_discoveries[op])

        if self.population and any(has_data):
            phi = sum(
                x * f for x, f, hd in zip(self.population, fitness, has_data, strict=False) if hd
            ) / sum(x for x, hd in zip(self.population, has_data, strict=False))
        else:
            phi = 0.0

        new_pop = []
        for i in range(n):
            growth = 1.0 + self.eta * (fitness[i] - phi) if has_data[i] else 1.0
            if self.marginal_cost_stop_multiplier is not None and self._mc_tracker.should_stop(
                self.operators[i], self.marginal_cost_stop_multiplier, keys=self.operators
            ):
                # Marginal cost (execs per new discovery) has run away
                # relative to the population average: apply the same
                # magnitude of shrink the ordinary update would apply to
                # the single worst-average-fitness operator, regardless of
                # what this operator's own average happens to be this
                # window. Bounded by min() rather than replacing growth
                # outright, so a stop signal can only ever shrink an
                # operator further, never override a shrink the ordinary
                # rule already computed.
                growth = min(growth, 1.0 - self.eta)
            new_pop.append(max(0.0, self.population[i] * growth))

        self.population = self._replicator_normalize_with_floor(new_pop, n)
        self._history.append(list(self.population))

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

    def operator_marginal_costs(self) -> dict[str, float | None]:
        """Current MC_i (execs per new discovery, between the last two
        window boundaries) per operator, per ``core/marginal_cost.py``.
        ``None`` for an operator with fewer than two window snapshots yet,
        or whose discovery count hasn't moved since the prior snapshot.
        """
        return {op: self._mc_tracker.marginal_cost(op) for op in self.operators}

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
