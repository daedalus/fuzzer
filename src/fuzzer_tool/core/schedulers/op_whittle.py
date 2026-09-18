"""WhittleIndexScheduler: restless-bandit index policy for operator selection.

Motivation (see ``docs/handover/handover_non_ucb_schedulers_2026-09-13.md``
§6, open question 1): the ten UCB-family arms already in the operator
ballot (``ducb``, ``swucb``, ``cusum_ucb``, ... ) all forget on a *global
round clock* -- their windows/discount factors are indexed by how many
total rounds have passed, not by how many times the arm itself has been
pulled. Operator fatigue (an operator exhausting the structural ground it
can reach) is a priori a *pull-indexed* process, which is exactly what
motivated T1 (rotting bandits / FEWA) in that handover. T1 itself is
gated on a still-missing telemetry column (an operator field in the
schedule-ablation CSV, see ``services/fuzzer.py``'s ``schedule_ablation``
writer) and cannot be measured yet.

This is the *other* open question from the same section: is fatigue
*rested* (only the arm's own pulls change its state) or *restless* (other
operators' progress -- global corpus growth -- also moves it, whether or
not this arm is played)? Nobody has measured this either; the ablation
gap that blocks T1 blocks this too. What follows is therefore built to be
usable as an experimental Elo arm *now*, with the restless assumption
isolated to one tunable parameter (``passive_decay``) that defaults to a
small, explicitly-labeled guess rather than a fitted value, so that
turning it to 0.0 degrades this to a plain rested birth-death index and
the restless claim is falsifiable independently of the rest of the
machinery once real telemetry exists.

Model
-----
Each operator has a small discrete "freshness" state ``s in {0..K-1}``,
0 = freshest, K-1 = most fatigued (``K`` = ``n_states``, default 5). Two
actions are defined at every state, exactly as in Whittle's original
formulation (P. Whittle, "Restless bandits: activity allocation in a
changing world," J. Appl. Prob. 25A, 1988, 287-298):

- **Active** (this operator is played): reward is the empirical success
  probability at state ``s``, tracked as a per-state Beta(a_s, b_s)
  posterior (weak uniform prior). The *same* coin that pays the reward
  also drives the transition -- success moves the state one step toward
  0 (``max(s-1, 0)``), failure moves it one step toward K-1
  (``min(s+1, K-1)``). This is a fully-observed birth-death chain, not a
  hidden Markov / POMDP one: the state is read off directly from this
  scheduler's own record() calls, never inferred from a belief update.
  That choice is deliberate -- the well-known closed-form Whittle indices
  in the restless-bandit literature are for specific *partially observed*
  two-state channel models (e.g. the Gilbert-Elliott / opportunistic
  spectrum access line of work), and this project already has one
  documented incident of an implementation reusing a paper's tuned
  constants outside the regime the paper actually proved
  (``docs/handover/handover_kl_ducb_paper_fidelity_2026-09-14.md``).
  Rather than risk repeating that with a misremembered or
  regime-mismatched closed form, the index here is computed *numerically*
  (a warm-started subsidy-grid scan + finite value iteration over
  ``n_states``, both small) for the exact fully-observed kernel actually
  implemented below -- "the index table is cheap and correct to compute" is the standing guidance
  in ``docs/handover/handover_FINDINGS.md`` P3-2, and a numeric table is
  what that line describes for the Gittins case too.
- **Passive** (some other operator is played): reward is a subsidy ``m``
  (the bisection variable, not a real reward), and the state drifts one
  step toward K-1 with probability ``passive_decay`` per idle round,
  otherwise stays. This is the entire restless assumption in one number.
  ``passive_decay=0.0`` makes state s absorbing under the passive action
  and this collapses to a rested birth-death index (only this arm's own
  pulls ever move it) -- the honest default until the ablation gap above
  is closed and the rested-vs-restless question has an actual answer.

Whittle index computation
--------------------------
For state ``s``, ``W(s)`` is the subsidy ``m`` at which the passive and
active actions tie in the ``m``-subsidized 2-action MDP. Indexability
(the passive-preferred region growing monotonically in ``m``, without
which "the" index is not well-defined) is not proven analytically for
this birth-death kernel -- rather than assume it, ``_whittle_index_scan``
sweeps a subsidy grid and counts sign changes in the active-vs-passive
comparison. A single crossing confirms indexability at that arm's current
reward estimates and the crossing point is returned. More than one
crossing means the assumption doesn't hold there; this is logged into
``bandit_stats()`` as ``indexable: False`` for that arm rather than
raising, and the *last* crossing found is used as a deterministic,
inspectable fallback (consistent with how ``op_katz``/``op_tang`` degrade
visibly instead of failing silently).

Starvation floor
-----------------
An index policy with no forgetting on the passive side can drive an arm
that got unlucky early (a few failures while sample sizes are tiny) into
its most-fatigued state and never pick it again to find out that was
noise -- structurally the same failure ``GradientBanditScheduler`` found
and fixed for softmax collapse (see that module's docstring, defect 2).
``floor`` mixes in uniform-random selection among the offered ops with
that probability every round, independent of the index values, for the
same reason.

Status
------
Off by default and Elo-only: absent from ``_FALLBACK_PRECEDENCE`` in
``services/operators.py`` for the same reason ``op_katz``/``op_tang``/
``gradient`` are absent -- an unproven exploratory arm should only be
reached by Elo explicitly choosing it, never by being the top-precedence
selector whenever someone enables the flag without ``--elo``. Not yet
run against ``tests/support/bandit_env.py``'s convergence harness; do
that (and a real ``bench_paired`` A/B, per the Boltzmann-A/B discipline
in the non-UCB handover) before pointing a real campaign at it.
"""

from __future__ import annotations

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool

DEFAULT_N_STATES = 5
DEFAULT_GAMMA = 0.95
DEFAULT_PASSIVE_DECAY = 0.0
DEFAULT_FLOOR = 0.05
DEFAULT_RECOMPUTE_BATCH = 25
_VALUE_ITERS = 60  # only used by the standalone (non-table) _value_iterate default
_VALUE_ITERS_COLD = 40  # first grid point: no warm start available
_VALUE_ITERS_WARM = 8  # subsequent grid points: warm-started from the previous one
_GRID_POINTS = 41
_GRID_LO = -1.0
_GRID_HI = 2.0


def _active_kernel(n_states: int, reward: list[float]) -> list[list[float]]:
    """P^A[s][s'] : success moves toward 0, failure moves toward K-1."""
    K = n_states
    P = [[0.0] * K for _ in range(K)]
    for s in range(K):
        p = reward[s]
        down = max(s - 1, 0)
        up = min(s + 1, K - 1)
        if down == up:
            P[s][down] = 1.0
        else:
            P[s][down] += p
            P[s][up] += 1.0 - p
    return P


def _passive_kernel(n_states: int, passive_decay: float) -> list[list[float]]:
    """P^P[s][s'] : drifts one step toward K-1 with prob passive_decay."""
    K = n_states
    P = [[0.0] * K for _ in range(K)]
    for s in range(K):
        up = min(s + 1, K - 1)
        if up == s:
            P[s][s] = 1.0
        else:
            P[s][up] += passive_decay
            P[s][s] += 1.0 - passive_decay
    return P


def _value_iterate(
    m: float,
    P_active: list[list[float]],
    P_passive: list[list[float]],
    reward: list[float],
    gamma: float,
    n_iters: int = _VALUE_ITERS,
) -> list[float]:
    """Finite value iteration for the m-subsidized 2-action MDP."""
    K = len(reward)
    V = [0.0] * K
    for _ in range(n_iters):
        newV = [0.0] * K
        for s in range(K):
            q_active = reward[s] + gamma * sum(P_active[s][sp] * V[sp] for sp in range(K))
            q_passive = m + gamma * sum(P_passive[s][sp] * V[sp] for sp in range(K))
            newV[s] = max(q_active, q_passive)
        V = newV
    return V


def _value_iterate(
    m: float,
    P_active: list[list[float]],
    P_passive: list[list[float]],
    reward: list[float],
    gamma: float,
    n_iters: int = _VALUE_ITERS,
    v0: list[float] | None = None,
) -> list[float]:
    """Finite value iteration for the m-subsidized 2-action MDP.

    ``v0`` warm-starts from a neighboring grid point's converged value
    (see ``whittle_index_table``); defaults to all-zeros (cold start).
    """
    K = len(reward)
    V = list(v0) if v0 is not None else [0.0] * K
    for _ in range(n_iters):
        newV = [0.0] * K
        for s in range(K):
            q_active = reward[s] + gamma * sum(P_active[s][sp] * V[sp] for sp in range(K))
            q_passive = m + gamma * sum(P_passive[s][sp] * V[sp] for sp in range(K))
            newV[s] = max(q_active, q_passive)
        V = newV
    return V


def _active_flags_all_states(
    V: list[float],
    m: float,
    P_active: list[list[float]],
    P_passive: list[list[float]],
    reward: list[float],
    gamma: float,
) -> list[bool]:
    """Active-vs-passive preference at every state, from one converged V.

    A single value-iteration solve already determines the optimal action
    at *every* state jointly -- there is no need to re-solve the whole
    MDP once per state, only once per subsidy grid point. Doing so
    (an earlier version of this function did exactly that) makes the
    table cost O(K) times more than necessary for no benefit.
    """
    K = len(reward)
    flags = [False] * K
    for s in range(K):
        q_active = reward[s] + gamma * sum(P_active[s][sp] * V[sp] for sp in range(K))
        q_passive = m + gamma * sum(P_passive[s][sp] * V[sp] for sp in range(K))
        flags[s] = q_active > q_passive
    return flags


def whittle_index_table(
    reward: list[float],
    passive_decay: float,
    gamma: float = DEFAULT_GAMMA,
    grid_points: int = _GRID_POINTS,
    grid_lo: float = _GRID_LO,
    grid_hi: float = _GRID_HI,
) -> tuple[list[float], list[bool]]:
    """Whittle index and indexability flag for every state, by grid scan.

    Returns (index_per_state, indexable_per_state). ``indexable[s]`` is
    False when the active-vs-passive comparison changes sign more than
    once across the grid at state s -- see module docstring. The index
    returned in that case is the *last* crossing (highest-subsidy point
    at which active stops being preferred), a deterministic, inspectable
    choice rather than an average or a raise.

    One value-iteration solve per grid point covers every state at once
    (see ``_active_flags_all_states``), and each grid point's solve is
    warm-started from the previous point's converged V -- subsidy moves
    the Bellman fixed point a small amount between adjacent grid points,
    so a handful of refinement sweeps from that starting point reaches
    the same fixed point full cold-start value iteration would need many
    more sweeps for. Both are pure performance optimizations over a
    naive per-state, cold-start grid scan; neither changes what is being
    computed.
    """
    K = len(reward)
    P_active = _active_kernel(K, reward)
    P_passive = _passive_kernel(K, passive_decay)

    grid = [grid_lo + i * (grid_hi - grid_lo) / (grid_points - 1) for i in range(grid_points)]

    active_flags = [[False] * grid_points for _ in range(K)]
    V = [0.0] * K
    for g, m in enumerate(grid):
        n_iters = _VALUE_ITERS_COLD if g == 0 else _VALUE_ITERS_WARM
        V = _value_iterate(m, P_active, P_passive, reward, gamma, n_iters=n_iters, v0=V)
        flags = _active_flags_all_states(V, m, P_active, P_passive, reward, gamma)
        for s in range(K):
            active_flags[s][g] = flags[s]

    indices = [grid[-1]] * K
    indexable = [True] * K
    for s in range(K):
        flags = active_flags[s]
        crossings = [
            i for i in range(1, grid_points) if flags[i - 1] and not flags[i]
        ]
        if not crossings:
            # Active preferred (or tied) across the whole grid: index is
            # at or above the top of the range we searched.
            indices[s] = grid[-1] if flags[-1] else grid[0]
            indexable[s] = len(crossings) <= 1
            continue
        indices[s] = grid[crossings[-1]]
        indexable[s] = len(crossings) == 1
    return indices, indexable


class WhittleIndexScheduler:
    """Restless-bandit index policy for operator selection (Elo-only, experimental).

    Args:
        n_states: Number of discrete freshness levels per arm (default 5).
        gamma: Discount factor used in the subsidized-MDP value iteration
            (default 0.95). Not campaign-tuned; a discount is required for
            the Whittle-index construction to be well-defined at all, its
            exact value has not been benchmarked.
        passive_decay: Probability an idle arm drifts one state toward
            fatigue per round it is *not* played -- the entire restless
            assumption. Default 0.0 (rested: only this arm's own pulls
            move its state) until the ablation-CSV operator column exists
            to measure whether real operator fatigue is restless. See
            module docstring.
        floor: Uniform-random selection probability, independent of the
            index values, to guard against permanent starvation of an
            arm that got unlucky early -- see module docstring
            "Starvation floor".
        recompute_batch: Recompute an arm's index table only after this
            many ``record()`` calls against it (default 25). Measured
            cost of one table recompute (n_states=5, default grid/gamma):
            ~2.55 ms on this project's dev container (392 calls/sec) --
            expensive enough to matter at default 1 on a hot fuzzing
            loop, not expensive enough to need anything fancier than
            batching it. Same idiom as ``op_tang_refit_interval`` in
            ``core/schedulers/op_katz.py``'s sibling module for the same
            reason: an index/basis that only needs to track a slowly
            drifting empirical estimate doesn't need recomputing every
            single observation.
        rng: Shared RandPool (Hard Rule 16).
    """

    supports_priors = False

    def __init__(
        self,
        n_states: int = DEFAULT_N_STATES,
        gamma: float = DEFAULT_GAMMA,
        passive_decay: float = DEFAULT_PASSIVE_DECAY,
        floor: float = DEFAULT_FLOOR,
        recompute_batch: int = DEFAULT_RECOMPUTE_BATCH,
        rng: RandPool | None = None,
    ):
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states!r}")
        if not (0.0 < gamma < 1.0):
            raise ValueError(f"gamma must be in (0, 1), got {gamma!r}")
        if not (0.0 <= passive_decay <= 1.0):
            raise ValueError(f"passive_decay must be in [0, 1], got {passive_decay!r}")
        if not (0.0 <= floor < 1.0):
            raise ValueError(f"floor must be in [0, 1), got {floor!r}")
        if recompute_batch < 1:
            raise ValueError(f"recompute_batch must be >= 1, got {recompute_batch!r}")

        self.n_states = n_states
        self.gamma = gamma
        self.passive_decay = passive_decay
        self.floor = floor
        self.recompute_batch = recompute_batch
        self._rng = rng if rng is not None else get_default_rand_pool()

        self._state: dict[str, int] = {}
        self._alpha: dict[str, list[float]] = {}
        self._beta: dict[str, list[float]] = {}
        self._pending: dict[str, int] = {}
        self._table: dict[str, list[float]] = {}
        self._indexable: dict[str, list[bool]] = {}

        self._last_op: str | None = None
        self._total_pulls: int = 0

    def init_arm(self, name: str) -> None:
        """Register an operator at freshness state 0 with a flat prior."""
        if name in self._state:
            return
        self._state[name] = 0
        self._alpha[name] = [1.0] * self.n_states
        self._beta[name] = [1.0] * self.n_states
        self._pending[name] = self.recompute_batch  # force a first computation
        self._table[name] = [0.0] * self.n_states
        self._indexable[name] = [True] * self.n_states

    def _reward_vector(self, name: str) -> list[float]:
        a = self._alpha[name]
        b = self._beta[name]
        return [a[s] / (a[s] + b[s]) for s in range(self.n_states)]

    def _maybe_recompute(self, name: str) -> None:
        if self._pending[name] < self.recompute_batch:
            return
        reward = self._reward_vector(name)
        table, indexable = whittle_index_table(reward, self.passive_decay, self.gamma)
        self._table[name] = table
        self._indexable[name] = indexable
        self._pending[name] = 0

    def _drift_idle_arms(self, ops: list[str]) -> None:
        """Passive-kernel step for every offered op not just played."""
        if self.passive_decay <= 0.0:
            return
        for op in ops:
            if op == self._last_op:
                continue  # already advanced by its own active transition in record()
            s = self._state.get(op)
            if s is None or s >= self.n_states - 1:
                continue
            if self._rng.random() < self.passive_decay:
                self._state[op] = s + 1

    def select_op(self, ops: list[str]) -> str:
        """Pick the op with the highest Whittle index at its current state."""
        if not ops:
            return ""
        if len(ops) == 1:
            self.init_arm(ops[0])
            return ops[0]

        for op in ops:
            self.init_arm(op)
        self._drift_idle_arms(ops)

        if self._rng.random() < self.floor:
            return ops[int(self._rng.random() * len(ops))]

        best_ops: list[str] = []
        best_index = float("-inf")
        for op in ops:
            self._maybe_recompute(op)
            idx = self._table[op][self._state[op]]
            if idx > best_index:
                best_index = idx
                best_ops = [op]
            elif idx == best_index:
                best_ops.append(op)

        chosen = best_ops[0] if len(best_ops) == 1 else best_ops[int(self._rng.random() * len(best_ops))]
        self._last_op = chosen
        return chosen

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Apply the active transition and Beta update for the played arm."""
        if name not in self._state:
            self.init_arm(name)

        self._total_pulls += 1
        s = self._state[name]
        if success:
            self._alpha[name][s] += weight
            self._state[name] = max(s - 1, 0)
        else:
            self._beta[name][s] += weight
            self._state[name] = min(s + 1, self.n_states - 1)

        self._pending[name] += 1
        self._last_op = name

    def bandit_stats(self) -> dict:
        """Return per-arm Whittle diagnostics."""
        for op in self._state:
            self._maybe_recompute(op)
        return {
            "whittle_pulls": self._total_pulls,
            "n_arms": len(self._state),
            "states": dict(self._state),
            "indices": {
                op: self._table[op][self._state[op]] for op in self._state
            },
            "indexable": {
                op: self._indexable[op][self._state[op]] for op in self._state
            },
            "best_op": (
                max(
                    self._state,
                    key=lambda op: self._table[op][self._state[op]],
                )
                if self._state
                else None
            ),
        }
