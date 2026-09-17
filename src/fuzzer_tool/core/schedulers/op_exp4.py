"""Exp4Scheduler: expert-advice bandit (EXP4) over operator categories.

EXP4 (Auer et al., 2002) maintains a distribution over *experts*. Each
expert recommends a distribution over arms; the algorithm mixes those
recommendations and importance-weights the observed reward back onto the
experts.

Here the expert set is one policy per operator category (bit, byte, block,
dict, structural, …) plus a uniform-explore expert. Category expert *c*
puts mass only on operators in category *c* that appear in the offered
list (uniform within the category). The uniform expert spreads mass over
every offered operator.

This is non-UCB and complementary to EXP3 (which weights individual
operators) and Hierarchical Thompson (which samples category then op from
Beta posteriors): EXP4 is adversarial over the *category* layer, so a
sudden regime shift that invalidates a whole category is handled by the
outer exponential weights rather than by slowly decaying Beta counts.

Selection:

    q_e = (1 − γ) w_e / Σw  +  γ / E
    ξ_e(a) = expert e's distribution over offered arms
    p(a)   = Σ_e q_e ξ_e(a)
    sample a ∼ p

Update (importance-weighted):

    r̂_e = r · ξ_e(a) / p(a)
    w_e ← w_e · exp(γ · r̂_e / E)

``record`` is only meaningful for draws this scheduler itself produced
(same contract as EXP3); the fuzzer gates the fan-out on
``selector == "exp4"``.

References:
- Auer, Cesa-Bianchi, Freund, Schapire, "The Nonstochastic Multiarmed
  Bandit Problem" (SIAM J. Comput. 2002) — EXP3 / EXP4
"""

from __future__ import annotations

import math

from fuzzer_tool.core.operator_categories import UNCATEGORIZED, category_of
from fuzzer_tool.core.rand_pool import RandPool

_UNIFORM_EXPERT = "uniform"


class Exp4Scheduler:
    """EXP4 expert-advice bandit; experts = operator categories + uniform.

    Args:
        gamma: Exploration mix in [0, 1]. Higher → closer to uniform over
            experts (and thus over arms).
        rng: Shared RandPool (Hard Rule 16).
    """

    supports_priors = False

    def __init__(self, gamma: float = 0.1, rng: RandPool | None = None):
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma!r}")
        self.gamma = gamma
        self._rng = rng if rng is not None else RandPool()

        # Expert weights (relative). Keys: category names + "uniform".
        self.weights: dict[str, float] = {_UNIFORM_EXPERT: 1.0}
        self._op_to_cat: dict[str, str] = {}
        self._ops: set[str] = set()
        self._total_pulls: int = 0

        # Last draw: needed for the importance-weighted expert update.
        self._last_op: str | None = None
        self._last_p_arm: float = 0.0
        self._last_expert_xi: dict[str, float] = {}  # expert -> ξ_e(a)
        self._last_q: dict[str, float] = {}

    def _cat_for(self, name: str) -> str:
        cat = self._op_to_cat.get(name)
        if cat is None:
            cat = category_of(name)
            self._op_to_cat[name] = cat
        return cat

    def init_arm(self, name: str) -> None:
        """Register an operator; ensures its category expert exists."""
        if name in self._ops:
            return
        self._ops.add(name)
        cat = self._cat_for(name)
        self.weights.setdefault(cat, 1.0)

    def _experts_for(self, ops: list[str]) -> list[str]:
        """Experts that have support on at least one offered op, plus uniform."""
        cats = {_UNIFORM_EXPERT}
        for op in ops:
            cats.add(self._cat_for(op))
        # Stable order for reproducibility
        rest = sorted(c for c in cats if c != _UNIFORM_EXPERT)
        return [_UNIFORM_EXPERT, *rest]

    def _expert_xi(self, expert: str, ops: list[str]) -> dict[str, float]:
        """Distribution ξ_e over *ops* for *expert*."""
        if expert == _UNIFORM_EXPERT:
            u = 1.0 / len(ops)
            return {op: u for op in ops}
        members = [op for op in ops if self._cat_for(op) == expert]
        if not members:
            return {op: 0.0 for op in ops}
        u = 1.0 / len(members)
        return {op: (u if op in members else 0.0) for op in ops}

    def select_op(self, ops: list[str]) -> str:
        """Sample an operator from the EXP4 mixture over category experts."""
        if not ops:
            return ""
        if len(ops) == 1:
            self.init_arm(ops[0])
            self._last_op = ops[0]
            self._last_p_arm = 1.0
            self._last_expert_xi = {_UNIFORM_EXPERT: 1.0}
            self._last_q = {_UNIFORM_EXPERT: 1.0}
            return ops[0]

        for op in ops:
            self.init_arm(op)

        experts = self._experts_for(ops)
        E = len(experts)
        total_w = sum(self.weights.get(e, 1.0) for e in experts)
        if total_w <= 0:
            total_w = float(E)

        # q over experts
        q: dict[str, float] = {}
        for e in experts:
            w = self.weights.get(e, 1.0)
            q[e] = (1.0 - self.gamma) * (w / total_w) + self.gamma / E
        self._last_q = dict(q)

        # p(a) = Σ_e q_e ξ_e(a)
        p_arm: dict[str, float] = {op: 0.0 for op in ops}
        xi_by_expert: dict[str, dict[str, float]] = {}
        for e in experts:
            xi = self._expert_xi(e, ops)
            xi_by_expert[e] = xi
            qe = q[e]
            for op in ops:
                p_arm[op] += qe * xi[op]

        # Numerical floor so roulette never starves an arm that should have mass
        s = sum(p_arm.values())
        if s <= 0:
            u = 1.0 / len(ops)
            p_arm = {op: u for op in ops}
            s = 1.0
        else:
            p_arm = {op: v / s for op, v in p_arm.items()}

        r = self._rng.random()
        cumulative = 0.0
        chosen = ops[-1]
        for op in ops:
            cumulative += p_arm[op]
            if r <= cumulative:
                chosen = op
                break

        self._last_op = chosen
        self._last_p_arm = max(p_arm[chosen], 1e-12)
        # ξ_e(chosen) per expert — used in the IW update
        self._last_expert_xi = {
            e: xi_by_expert[e].get(chosen, 0.0) for e in experts
        }
        return chosen

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Importance-weighted expert update for the last EXP4 draw.

        Only applies when *name* matches the arm this scheduler selected;
        shadow records from other strategies are ignored (unbiased IW).
        """
        if name != self._last_op or not self._last_expert_xi:
            return

        self._total_pulls += 1
        reward = min(1.0, max(0.0, weight if success else 0.0))
        E = max(len(self._last_expert_xi), 1)
        p_a = self._last_p_arm

        for expert, xi_a in self._last_expert_xi.items():
            if xi_a <= 0.0:
                continue
            # r̂_e = r * ξ_e(a) / p(a)
            rhat = reward * xi_a / p_a
            w = self.weights.get(expert, 1.0)
            # Clamp exponent to avoid overflow on rare high-IW updates
            scaled = self.gamma * rhat / E
            if scaled > 50.0:
                scaled = 50.0
            self.weights[expert] = w * math.exp(scaled)

        # Renormalize if weights blow up
        max_w = max(self.weights.values()) if self.weights else 1.0
        if max_w > 1e9:
            scale = 1.0 / max_w
            for k in self.weights:
                self.weights[k] *= scale

    def last_selection_probs(self) -> dict[str, float]:
        """Not the arm mixture; returns last expert distribution *q*."""
        return dict(self._last_q)

    def bandit_stats(self) -> dict:
        """EXP4 diagnostics."""
        return {
            "exp4_pulls": self._total_pulls,
            "exp4_experts": len(self.weights),
            "exp4_top_expert": (
                max(self.weights, key=self.weights.get) if self.weights else None
            ),
            "exp4_weights": {k: round(v, 6) for k, v in sorted(self.weights.items())},
        }
