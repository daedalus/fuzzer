"""``op_credit``: operator selection on class-deduplicated credit (P3-3).

Design: ``docs/handover/handover_edge_id_axis_2026-09-18.md``, "Operator side: reward
shaping, not another bandit". Nothing in F1-F13 touched operator *selection*, and the
bandit family is already 34 modules deep, so the contribution here is the reward, and the
selector is deliberately the thinnest one that can consume it: Thompson sampling over a
Beta posterior. Because the reward changes and the selector is a stock one, an Elo A/B
against the existing arms isolates the reward.

The reward is ``substrate.class_credit(edges the operator was on the round it found)``:

* **A duplicate class pays once, not once per member** (F10: 126 of 445 edges are
  copies, largest class 45). An operator that walks into a 45-edge straight-line chain
  earns one unit.
* **A derived edge pays nothing** once P1-2 has confirmed the relations against
  ``core/icfg.py``; until then ``substrate.derived`` is empty and this is a no-op.
* **The credit is a function of the current partition, not an accumulator.** This is
  the answer to P3-3's paper question, what happens to credit when a class splits
  mid-campaign: nothing stored can go stale, because the arm stores only the edges an
  operator found and re-derives the credit at read time against the classes as they are
  now. A split maps the same edges onto more classes, so credit rises -- correctly, since
  the input that split the class showed those edges were not one branch after all.
* **Fatigue is indexed by the arm's own pulls.** Credit is a set cardinality, so an
  operator that keeps re-finding the same classes stops earning while its pulls keep
  counting. Nothing depends on a global clock.

Posterior: ``Beta(1 + credit, 1 + max(pulls - credit, 0) * (1 - DECAY * saturation))``.
The ``2^H`` re-tempering follows the handover -- effective edges collapsing while the raw
edge count is flat is a drift signal, and the response is to raise exploration and decay
stale failure evidence, "not to reset". ``DECAY`` is uncalibrated (the handover gives the
direction, not the size).

Abstains from the ballot while the preflight gate is closed (``operator_strategy_pool``
consults :meth:`available`): under per-process ids every phantom id is a discovery owned
by one seed, and credit for those is noise. Off by default and unproven; deliberately not
in ``_FALLBACK_PRECEDENCE``, like ``op_tang``.
"""

from __future__ import annotations

from fuzzer_tool.core.edge_matrix import MatrixSubstrate
from fuzzer_tool.core.rand_pool import RandPool

#: Share of stale failure evidence dropped at full saturation. Uncalibrated.
DECAY = 0.5
EXPLORE_BASE = 0.05


class OpCreditScheduler:
    """Elo-arm operator scheduler over the shared :class:`MatrixSubstrate`.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16).
        substrate: The canonical edge space, shared with ``ResidualSeedScheduler``.
    """

    supports_priors = False

    def __init__(self, rng: RandPool | None = None, substrate: MatrixSubstrate | None = None):
        if rng is None:
            raise ValueError("OpCreditScheduler requires a RandPool (Hard Rule 16)")
        if substrate is None:
            raise ValueError("OpCreditScheduler requires a MatrixSubstrate")
        self._rng = rng
        self.substrate = substrate
        self._found: dict[str, set[int]] = {}
        self._pulls: dict[str, float] = {}
        self._cache_key: tuple[int, int] | None = None
        self._credit: dict[str, int] = {}
        self._observed = 0

    def init_arm(self, name: str) -> None:
        self._pulls.setdefault(name, 0.0)
        self._found.setdefault(name, set())

    def observe_new_edges(self, op: str, new_edge_ids) -> None:
        """Remember the edge ids *op* was on when they were first discovered."""
        if not new_edge_ids:
            return
        row = self._found.setdefault(op, set())
        before = len(row)
        row.update(new_edge_ids)
        self._observed += len(row) - before

    def record(self, op: str, success: bool, weight: float = 1.0) -> None:
        """Count one pull. Credit comes from :meth:`observe_new_edges`, not *success*.

        The shared boolean means "this round produced a new edge", which scores a
        phantom id and a duplicate-chain edge the same as a real class.
        """
        self._pulls[op] = self._pulls.get(op, 0.0) + 1.0

    def available(self) -> bool:
        return self.substrate.trusted

    def credit(self, op: str) -> int:
        """Distinct canonical classes *op* has found, against the classes as of now."""
        key = (self.substrate.version, self._observed)
        if key != self._cache_key:
            self._cache_key = key
            self._credit = {
                o: self.substrate.class_credit(edges) for o, edges in self._found.items()
            }
        return self._credit.get(op, 0)

    def shaped_weight(self, new_edge_ids) -> float:
        """Fraction of a round's new edges that are independent, in [0, 1].

        For a caller that wants to scale the shared reward weight rather than run this
        selector: 45 new edges that are one class weigh 1/45. Not wired anywhere; it is
        the form the handover calls "the cheapest A/B available".
        """
        edges = list(new_edge_ids)
        return self.substrate.class_credit(edges) / len(edges) if edges else 0.0

    def select_op(self, ops: list[str]) -> str:
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        if not self.available():
            return str(self._rng.choice(ops))
        sat = self.substrate.saturation_signal()
        if self._rng.random() < min(1.0, EXPLORE_BASE * (1.0 + sat)):
            return ops[int(self._rng.randrange(len(ops)))]
        best, best_v = ops[0], -1.0
        for op in ops:
            cr = float(self.credit(op))
            stale = max(self._pulls.get(op, 0.0) - cr, 0.0) * (1.0 - DECAY * sat)
            theta = float(self._rng.betavariate(1.0 + cr, 1.0 + stale))
            if theta > best_v:
                best, best_v = op, theta
        return best

    def bandit_stats(self) -> dict:
        return {
            "op_credit_gate": self.substrate.gate_state(),
            "op_credit_credit": {o: self.credit(o) for o in self._found},
            "op_credit_pulls": dict(self._pulls),
            "op_credit_fitted": self.substrate.fold is not None,
        }
