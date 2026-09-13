"""OpKatzScheduler: Katz centrality over the operator *transition* graph.

``core/schedulers/katz.py`` (the seed-picker's ``katz`` Elo arm) is
deliberately DAG-only: it solves the fixed point by successor-summing for
exactly ``depth`` rounds, which is only exact -- and only terminates without
an explicit cap -- because the ICFG horizon graph it operates over is
acyclic by construction (``build_horizon_graph`` guarantees it, ``katz_scores``
raises if it isn't).

An operator transition graph is not acyclic: op_A can follow op_B can follow
op_A, trivially (most havoc chains do exactly this). So this ports the
classical Katz (1953) formulation instead of the DAG-restricted one:

    c = sum_{k=0}^inf alpha^k A^k beta = (I - alpha*A)^-1 beta

which converges for any alpha < 1/spectral_radius(A), cycles included.
alpha is chosen automatically as a fixed fraction of that bound so callers
never have to reason about the graph's spectral radius by hand.

A[i][j] is the row-normalized rate at which a *discovery-linked* transition
from op_i to op_j has been observed -- the same transition semantics as
``MonteCarloScheduler.transition_counts`` (successor recorded only on
``success=True``, so the graph encodes "op_j tends to land right after op_i
finds something," not just co-occurrence). beta_i is op_i's own raw success
rate (0 for an operator with no track record yet), so score = "how good is
this op on its own" amplified by "does it feed into other productive ops."

An earlier version of this beta used ``1 - success_rate`` (mirroring the
seed-side katz.py's frontier-seeking convention, which favors *unexplored*
graph regions). That is actively wrong here: it assigns zero injection to
exactly the operators an exploitation-oriented scheduler should be
amplifying, so a perfectly-reinforcing cycle between two 100%-successful
operators scored *below* an isolated 50%-successful one (0.045 vs 0.5) in
testing. Seeds and operators want opposite biases from the same graph
algorithm -- seed selection wants to explore the frontier, operator
selection wants to exploit a working chain -- so the sign flip here is
deliberate, not a simplification of the seed-side arm.

Empirical note: the synthetic paired benchmark referenced in commit
history predates this beta-sign fix and used the inverted (wrong)
convention throughout, so its p-values/win-rates do not carry over as-is
and should not be cited as evidence either way for this corrected version
-- rerun before trusting any specific number. It is landed off by default
regardless: enable it only if there's a concrete reason to suspect
chain-like dependencies among this target's operators (e.g. a structural
op reliably setting up ground that a later byte-level op then exploits),
and measure on a real campaign before trusting it.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool

DEFAULT_ALPHA_FRACTION = 0.85  # fraction of 1/spectral_radius(A) to use


def build_transition_matrix(
    transition_counts: dict[str, dict[str, int]],
    ops: list[str],
) -> np.ndarray:
    """Row-normalized [op x op] transition-rate matrix, ordered by ``ops``.

    Rows with no recorded transitions are all-zero: a pure sink that
    receives centrality from its own beta but propagates nothing onward.
    """
    n = len(ops)
    idx = {op: i for i, op in enumerate(ops)}
    a = np.zeros((n, n), dtype=np.float64)
    for src, row in transition_counts.items():
        if src not in idx:
            continue
        i = idx[src]
        total = sum(row.get(op, 0) for op in ops)
        if total <= 0:
            continue
        for dst, count in row.items():
            if dst in idx and count > 0:
                a[i, idx[dst]] = count / total
    return a


def classical_katz_scores(
    a: np.ndarray,
    beta: np.ndarray | None = None,
    alpha_fraction: float = DEFAULT_ALPHA_FRACTION,
) -> np.ndarray:
    """Solve c = (I - alpha*a)^-1 @ beta -- i.e. c = beta + alpha*a@c.

    That is successor-summing: c[i] accumulates alpha*c[j] for each
    observed i->j transition (row i, column j of ``a``), the same
    convention as the DAG ``katz.py``'s ``contrib = bincount(src,
    weights=c[dst])`` / ``nxt[u] += alpha*contrib[u]``.

    alpha is auto-picked below 1/spectral_radius(a) when a has a nonzero
    spectral radius (i.e. contains a cycle). A purely feed-forward graph
    (no cycles at all -- rho=0, ``a`` nilpotent) still has a valid,
    *finite* fixed point for any alpha via the terminating Neumann series
    (I + alpha*a + alpha^2*a^2 + ...), so that case uses alpha_fraction
    directly rather than skipping propagation -- returning beta unchanged
    there was the bug an earlier version of this function had: a single
    a->b edge with no cycle has rho=0, and short-circuiting on that
    silently zeroed out the entire propagation term for exactly the kind
    of one-shot "op_a sets up op_b" chain this scheduler exists to catch.
    """
    n = a.shape[0]
    if beta is None:
        beta = np.ones(n, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if n == 0:
        return beta
    if not a.any():
        return beta.copy()
    eigvals = np.linalg.eigvals(a)
    rho = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    alpha = alpha_fraction / rho if rho > 1e-12 else alpha_fraction
    identity = np.eye(n, dtype=np.float64)
    try:
        c = np.linalg.solve(identity - alpha * a, beta)
    except np.linalg.LinAlgError:
        return beta.copy()
    return c


class OpKatzScheduler:
    """Elo-arm operator scheduler: Katz centrality over discovery transitions.

    Lifecycle mirrors ``FPLScheduler``/``ReplicatorScheduler``: the fuzzer
    holds one instance, calls :meth:`record` on every outcome (mirroring the
    ``record(op, success, weight=...)`` signature shared by most schedulers
    in this package), and :meth:`select_op` picks from the offered arm list.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16).
        alpha_fraction: Passed through to :func:`classical_katz_scores`.
    """

    supports_priors = False

    def __init__(self, rng: RandPool | None = None, alpha_fraction: float = DEFAULT_ALPHA_FRACTION):
        if rng is None:
            raise ValueError("OpKatzScheduler requires a RandPool (Hard Rule 16)")
        self._rng = rng
        self.alpha_fraction = alpha_fraction
        self.transition_counts: dict[str, dict[str, int]] = {}
        self.successes: dict[str, float] = {}
        self.attempts: dict[str, float] = {}
        self._prev_op: str | None = None

    def record(self, op: str, success: bool, weight: float = 1.0) -> None:
        """Record one outcome. Weight scales the success credit only.

        Mirrors the shared ``record(op, success, weight=...)`` contract so
        this can sit in the same reward fan-out loop as replicator/exp3/etc
        (``Fuzzer._record_operator_strategy_matches``). The transition edge
        (prev_op -> op) is recorded on success regardless of weight, same as
        ``MonteCarloScheduler.transition_counts`` -- weight modulates how
        much the op's own rate moves, not whether a discovery "counts" as a
        transition at all.
        """
        self.attempts[op] = self.attempts.get(op, 0.0) + 1.0
        if success:
            self.successes[op] = self.successes.get(op, 0.0) + max(weight, 0.0)
            if self._prev_op is not None and self._prev_op != op:
                row = self.transition_counts.setdefault(self._prev_op, {})
                row[op] = row.get(op, 0) + 1
        self._prev_op = op

    def scores(self, ops: list[str]) -> dict[str, float]:
        """Katz centrality per op, restricted to the offered ``ops`` list.

        beta_i is op_i's raw success rate (0.0 if never attempted) --
        exploitation-favoring, deliberately the opposite sign of the
        seed-side katz.py's frontier-seeking beta. See module docstring.
        """
        a = build_transition_matrix(self.transition_counts, ops)
        rates = []
        for op in ops:
            attempts = self.attempts.get(op, 0.0)
            rate = self.successes.get(op, 0.0) / attempts if attempts > 0 else 0.0
            rates.append(rate)
        beta = np.clip(np.asarray(rates, dtype=np.float64), 0.0, 1.0)
        c = classical_katz_scores(a, beta, self.alpha_fraction)
        return dict(zip(ops, c.tolist(), strict=True))

    def select_op(self, ops: list[str]) -> str:
        """Softmax-free weighted pick: scores shifted non-negative, sampled by mass.

        Uses a direct weighted draw (not a temperature softmax) since Katz
        scores are already a bounded positive-ish quantity once shifted,
        and reserving softmax/temperature for the Elo meta-layer above this
        arm keeps the two exploration mechanisms from compounding.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        s = self.scores(ops)
        vals = np.array([s[op] for op in ops], dtype=np.float64)
        shifted = vals - vals.min() + 1e-9
        total = float(shifted.sum())
        probs = shifted / total if total > 0 else np.full(len(ops), 1.0 / len(ops))
        r = self._rng.random()
        cumulative = 0.0
        for op, p in zip(ops, probs.tolist(), strict=True):
            cumulative += p
            if r <= cumulative:
                return op
        return ops[-1]
