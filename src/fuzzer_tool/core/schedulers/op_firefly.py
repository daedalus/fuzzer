"""OpFireflyScheduler: Firefly Algorithm over operator probability distributions.

Implements the scheduling scheme from Zhang et al., "FA-Fuzz: A Novel
Scheduling Scheme Using Firefly Algorithm for Mutation-Based Fuzzing" (IEEE,
2023, ieeexplore.ieee.org/document/10305545): a swarm of candidate operator-
probability distributions ("fireflies") that search the joint configuration
space, like ``op_mopt.py``'s ``MOptScheduler`` -- but via firefly attraction
rather than PSO's velocity/inertia.

Model
-----
Each firefly ``i`` is a point on the operator-probability simplex, exactly
like an MOpt particle. Its brightness is its own recent discovery rate. Two
fireflies compare pairwise: a dimmer firefly moves toward every brighter one,
pulled harder the closer they already are --

    x_i += beta(r_ij) * (x_j - x_i) + alpha * eps    for every j with I_j > I_i

with attractiveness decaying with distance as ``beta(r) = beta0 *
exp(-gamma * r^2)`` (the paper's Eq. for beta; ``r_ij`` is Euclidean distance
between the two positions). A firefly with no brighter neighbor -- the
population's current best -- only gets the random term, i.e. a bounded
random walk. This is FA-Fuzz's actual mechanism: unlike MOpt's PSO, there is
no velocity or inertia term carried between windows, and no single global
best pulls the whole swarm -- every firefly pairs against every other one,
so a locally-bright firefly can pull nearby dimmer ones toward it even while
a brighter firefly exists elsewhere in the simplex.

What this module does NOT implement: FA-Fuzz's headline claim is that the
*optimal* operator distribution differs **per seed**, and the paper's whole
point is searching a distribution per seed rather than one shared by the
whole corpus. This module, like ``MOptScheduler``, searches one
population-level swarm shared across every seed -- the same scope MOpt
already has in this repo. Per-seed conditioning is a real, separate
extension (it would need a firefly population keyed by seed rather than one
global population) and is deliberately left for a follow-up rather than
folded into this same change.

Lessons carried over from MOpt, applied here from the start rather than
re-discovered by a second bug hunt:

- **Jittered initial positions.** ``_MOptParticle.__init__`` documents that
  every particle starting at exactly uniform with zero velocity makes PSO a
  no-op by construction (pbest == gbest == pos, so both attractor terms are
  zero forever). The exact same trap applies here: firefly movement is
  *also* driven purely by pairwise position differences, so identical
  starting positions give every pairwise ``x_j - x_i`` term zero regardless
  of brightness, and the swarm never moves. Fireflies are jittered around
  uniform at construction with the same ``_INIT_SPREAD`` MOpt uses.
- **Held fitness, not reset to zero.** ``MOptScheduler._update_fitness``
  documents that zeroing a particle's fitness when its window collects no
  executions is self-reinforcing (fitness-proportional selection starves
  the low-fitness particle, starvation empties its window, empty window
  zeros its fitness, repeat). The same selection rule is used here
  (fitness-proportional over fireflies), so the same fix applies: a firefly
  with an empty window keeps its previous brightness instead of going dark.
- **Selection floor relative to the best, not an absolute constant.**
  ``MOptScheduler.select_op`` documents that an absolute floor like 0.001
  stops being a floor once any particle's fitness clears roughly 0.1 --
  the leader then takes over 99% of draws and nothing else ever collects
  enough samples to challenge it. The floor here is ``0.1 * best fitness``
  (never below 0.001), same as MOpt.
- **Floor as a fraction of uniform, not an absolute constant, when
  projecting back onto the simplex.** ``MOptScheduler._normalize_to_simplex``
  documents that a fixed constant floor is harmless at a dozen operators in
  a unit test but exceeds the whole simplex at the live registry's ~135
  operators, silently disabling exploration. Same fractional floor here.

Off by default, Elo-only, absent from ``_FALLBACK_PRECEDENCE`` -- same
discipline as ``op_katz``, ``op_kuramoto``, ``op_tang`` and
``op_kruskal_count``: whether firefly attraction says anything useful about
operator scheduling that MOpt's PSO does not is an open empirical question
this arm exists to test, not a settled premise, and it has not been run
against this project's own convergence harness or a real ``bench_paired``
A/B yet.
"""

import collections

import numpy as np

from fuzzer_tool.core.chaos import InertiaMode, make_chaos
from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.simplex import project_rows

#: Fractional jitter applied to initial firefly positions. Same rationale
#: and same magnitude as op_mopt.py's _INIT_SPREAD: enough to give firefly
#: attraction a gradient to work with; small enough that no operator starts
#: strongly favoured.
_INIT_SPREAD = 0.5


class _Firefly:
    """A single firefly: a point on the operator probability simplex."""

    __slots__ = (
        "pos",
        "name",
        "discoveries",
        "execs_in_window",
        "fitness",
    )

    def __init__(self, name: str, n_ops: int, rng: RandPool, spread: float = _INIT_SPREAD):
        self.name = name
        if n_ops > 0:
            raw = [1.0 + spread * (rng.random() * 2.0 - 1.0) for _ in range(n_ops)]
            total = sum(raw)
            self.pos = [x / total for x in raw]
        else:
            self.pos = []
        self.discoveries: collections.deque[float] = collections.deque(maxlen=200)
        self.execs_in_window = 0
        self.fitness = 0.0


class OpFireflyScheduler:
    """FA-Fuzz-style adaptive operator scheduling via the Firefly Algorithm.

    Maintains K fireflies, each a probability distribution over mutation
    operators, and moves dimmer fireflies toward brighter ones each window
    -- see the module docstring for the model and what is deliberately not
    implemented (per-seed distributions).

    Args:
        n_fireflies: Population size (default 5, matching MOptScheduler's
            default particle count -- kept small because each window's
            movement step is O(n_fireflies^2 * n_ops), and the swarm gets
            re-evaluated every ``window_size`` executions, not once per
            generation the way an offline FA run would).
        window_size: Executions per brightness evaluation window.
        beta0: Base attractiveness at zero distance.
        gamma: Light-absorption coefficient: how fast attractiveness decays
            with distance. Positions live on the probability simplex, so
            Euclidean distance between two fireflies is bounded by
            ``sqrt(2)``; ``gamma=1.0`` gives meaningful decay across that
            whole range without needing a per-target rescale.
        alpha: Random-walk step scale applied to every firefly every
            window, including the current brightest (which gets nothing
            else, per the model).
        alpha_decay: Per-window multiplicative cooling of ``alpha`` --
            the same cooling idiom ``core/schedulers`` already uses for
            QEA's ``rotation_gate()`` (see
            ``docs/handover``'s QEA cooling entry): early exploration wide,
            later windows settle down instead of perpetually jittering a
            converged swarm.
        min_prob_frac: Exploration floor, as a fraction of the uniform
            probability ``1/n`` -- identical semantics to
            ``MOptScheduler.min_prob_frac``.
        rng: Shared RandPool. Falls back to the process-wide default pool.
        inertia: ``CHAOTIC`` scales each window's random step by a
            logistic-map factor of mean 1 (``core/chaos.py``).
    """

    # Declares that init_arm() does NOT accept informative priors (the
    # swarm carries arm state in firefly positions, not Beta-Bernoulli
    # counts) -- same contract as MOptScheduler.
    supports_priors = False

    def __init__(
        self,
        n_fireflies: int = 5,
        window_size: int = 200,
        beta0: float = 1.0,
        gamma: float = 1.0,
        alpha: float = 0.2,
        alpha_decay: float = 0.97,
        min_prob_frac: float = 0.1,
        rng: RandPool | None = None,
        inertia: InertiaMode = InertiaMode.CONSTANT,
    ):
        self._rng = rng if rng is not None else get_default_rand_pool()
        self._chaos = make_chaos(inertia, self._rng)
        self.n_fireflies = n_fireflies
        self.window_size = window_size
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.min_prob_frac = min_prob_frac

        self.operators: list[str] = []
        self.op_index: dict[str, int] = {}
        self.fireflies: list[_Firefly] = []

        self._total_execs = 0
        self._total_discoveries = 0

    def init_arm(self, name: str) -> None:
        """Register a mutation operator. Rebuilds fireflies if operators changed."""
        if name in self.op_index:
            return
        idx = len(self.operators)
        self.operators.append(name)
        self.op_index[name] = idx
        self._rebuild_fireflies()

    def _rebuild_fireflies(self) -> None:
        """Rebuild all fireflies for the current operator set."""
        n = len(self.operators)
        old = {f.name: f for f in self.fireflies}
        self.fireflies = []
        for i in range(self.n_fireflies):
            name = f"f{i}"
            if name in old:
                prev = old[name]
                # Same extend-preserving-relative-weights approach as
                # MOptScheduler._rebuild_particles, for the same reason:
                # a flat uniform share for new operators, applied every
                # time init_arm() is called (once per operator, one at a
                # time), would re-flatten the whole distribution back
                # toward uniform on every single registration.
                n_old = len(prev.pos)
                n_new = n - n_old
                keep = n_old / n if n > 0 else 1.0
                old_total = sum(prev.pos) or 1.0
                new_pos = [x / old_total * keep for x in prev.pos] + [
                    (1.0 / n) * (1.0 + _INIT_SPREAD * (self._rng.random() * 2.0 - 1.0))
                    for _ in range(n_new)
                ]
                total = sum(new_pos)
                new_pos = [p / total for p in new_pos]
                fly = _Firefly(name, n, rng=self._rng)
                fly.pos = new_pos
            else:
                fly = _Firefly(name, n, rng=self._rng)
            self.fireflies.append(fly)

    def select_op(self, ops: list[str]) -> tuple[str, int]:
        """Select an operator via fitness-proportional firefly selection.

        Returns:
            (operator_name, firefly_index) -- the index is needed by
            record() to attribute the outcome to the firefly that drew it,
            exactly like MOptScheduler's particle_id.
        """
        if not self.fireflies or not self.operators:
            return (ops[0] if ops else "", 0)

        valid = [f for f in self.fireflies if any(f.pos)]
        if not valid:
            valid = self.fireflies
        # Floor relative to the best firefly rather than an absolute
        # constant -- see module docstring.
        best_f = max((f.fitness for f in valid), default=0.0)
        floor = max(0.1 * best_f, 0.001)
        fitnesses = [max(f.fitness, floor) for f in valid]
        total_f = sum(fitnesses)
        r = self._rng.random() * total_f
        cumulative = 0.0
        selected = valid[0]
        for f, fit in zip(valid, fitnesses, strict=False):
            cumulative += fit
            if r <= cumulative:
                selected = f
                break
        selected_idx = self.fireflies.index(selected)

        op = self._sample_from_firefly(selected, ops)
        return (op, selected_idx)

    def _sample_from_firefly(self, firefly: _Firefly, ops: list[str]) -> str:
        """Sample an operator from a firefly's probability distribution."""
        probs = []
        for op in ops:
            idx = self.op_index.get(op, -1)
            probs.append(firefly.pos[idx] if 0 <= idx < len(firefly.pos) else 0.0)

        total = sum(probs)
        if total <= 0:
            return str(self._rng.choice(ops))

        r = self._rng.random() * total
        cumulative = 0.0
        for op, p in zip(ops, probs, strict=False):
            cumulative += p
            if r <= cumulative:
                return op
        return ops[-1]

    def record(
        self, name: str, success: bool, firefly_id: int | None = None, weight: float = 1.0
    ) -> None:
        """Record outcome for brightness tracking.

        Args:
            name: Operator that was used.
            success: Whether it produced new coverage.
            firefly_id: Index of the firefly that selected this operator.
                When None (backward compat), updates every firefly.
            weight: Reward weight (default 1.0). Surprisal-weighted calls
                pass a value in (0, 1] proportional to discovery rarity.
        """
        self._total_execs += 1
        reward = weight if success else 0.0
        if success:
            self._total_discoveries += 1

        if firefly_id is not None and 0 <= firefly_id < len(self.fireflies):
            targets = [self.fireflies[firefly_id]]
        else:
            targets = self.fireflies
        for fly in targets:
            fly.execs_in_window += 1
            fly.discoveries.append(reward)

        if self._total_execs % self.window_size == 0 and self._total_execs > 0:
            self._firefly_update()

    def _update_fitness(self, firefly: _Firefly) -> None:
        """Brightness = discovery rate in the window.

        A firefly with no executions this window keeps its previous
        brightness rather than being reset to zero -- see module docstring
        ("Held fitness, not reset to zero").
        """
        if not firefly.discoveries or firefly.execs_in_window == 0:
            return
        disc = sum(firefly.discoveries)
        total = max(firefly.execs_in_window, 1)
        firefly.fitness = disc / total

    def _firefly_update(self) -> None:
        """Run one firefly-movement round: pairwise attraction + random walk."""
        n = len(self.operators)
        if n == 0:
            return

        for fly in self.fireflies:
            self._update_fitness(fly)

        # Synchronous update: every firefly's move is computed from the
        # *pre-round* snapshot of positions and brightness, then applied
        # after every firefly's target has been computed. Updating in
        # place instead would make firefly k's move depend on whether
        # firefly k-1 already moved this round -- an ordering artifact
        # with no basis in the model.
        #
        # Vectorized: diff[i, j] = pos_j - pos_i; only brighter j attract i
        # (this also drops j == i). Noise draws keep the scalar order
        # (firefly-major, operator-minor), so a seeded pool is unchanged.
        #
        #   new_i = pos_i + sum_j beta_ij * (pos_j - pos_i) + alpha * U(-1, 1)
        # One step size per window; chaotic mode scales it by a mean-1 factor.
        alpha = self.alpha if self._chaos is None else self.alpha * self._chaos.alpha_factor()

        if self.fireflies:
            pos = np.array([fly.pos for fly in self.fireflies])
            fit = np.array([fly.fitness for fly in self.fireflies])
            diff = pos[None, :, :] - pos[:, None, :]
            brighter = fit[None, :] > fit[:, None]
            r2 = np.einsum("ijk,ijk->ij", diff, diff)
            beta = np.where(brighter, self.beta0 * np.exp(-self.gamma * r2), 0.0)
            noise = np.array(self._rng.random_list(pos.size)).reshape(pos.shape)
            moved = pos + np.einsum("ij,ijk->ik", beta, diff) + alpha * (noise * 2.0 - 1.0)
            moved = project_rows(moved, self.min_prob_frac)
            for fly, row in zip(self.fireflies, moved, strict=True):
                fly.pos = row.tolist()
                fly.execs_in_window = 0
                fly.discoveries.clear()

        self.alpha *= self.alpha_decay

    def _normalize_to_simplex(self, firefly: _Firefly) -> None:
        """Project a post-movement position back onto the probability simplex.

        Same clip-negative / renormalize / fractional-floor sequence as
        ``MOptScheduler._normalize_to_simplex``, for the same reason: an
        absolute floor constant is harmless at a dozen operators and
        disables exploration entirely at the live registry's ~135.
        """
        if not firefly.pos:
            return

        firefly.pos = project_rows(np.array([firefly.pos]), self.min_prob_frac)[0].tolist()

    def firefly_stats(self) -> list[dict[str, str | float]]:
        """Get stats for each firefly (for diagnostics/logging)."""
        result: list[dict[str, str | float]] = []
        for fly in self.fireflies:
            self._update_fitness(fly)
            if fly.pos and self.operators:
                best_idx = max(range(len(fly.pos)), key=lambda i: fly.pos[i])
                best_op = self.operators[best_idx] if best_idx < len(self.operators) else "?"
            else:
                best_op = "?"
            result.append(
                {
                    "name": fly.name,
                    "fitness": round(fly.fitness, 4),
                    "top_op": best_op,
                    "top_prob": round(max(fly.pos), 3) if fly.pos else 0.0,
                }
            )
        return result

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Compatibility with MonteCarloScheduler interface.

        Returns discovery/failure counts from the global window.
        """
        return {
            "_firefly_global": (
                self._total_discoveries,
                self._total_execs - self._total_discoveries,
            )
        }
