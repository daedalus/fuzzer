"""SeedRoundRobinScheduler: deterministic seed cycling.

The seed-arena counterpart of ``core/schedulers/op_round_robin.py``. The
Elo meta-scheduler runs two separate tournaments -- one over operator
schedulers under plain strategy names, one over seed-selection strategies
under ``seed_``-prefixed keys (``services/seed_picker.py::_pick_seed_elo``,
no cross competition; see ``core/schedulers/seed_canary.py``'s module
docstring for the fuller version of this split). ``RoundRobinScheduler``
serves the operator arena as a deterministic baseline: proven, signal-free,
and -- unlike ``op_katz``/``op_tang``/``op_canary`` -- trusted enough to sit
in ``services/operators.py``'s no-``--elo`` top-precedence list rather than
being Elo-only. This module gives the seed arena the same baseline.

Unlike ``SeedCanaryScheduler``, round-robin is not a deliberately-bad floor:
it is a real, if simple, scheduling policy -- cycle through the corpus in
registration order, ignore the outcome signal entirely. That is also why,
mirroring ``op_round_robin``'s standalone reach, it is wired into
``SeedPicker.pick_seed()``'s no-``--elo`` fallback chain (alongside
kruskal-count and the entropy/residual arms) as well as into the Elo pool:
it needs no arbiter to be worth running.

``record()`` is kept for Elo-arena compatibility only (a rated strategy
needs an outcome signal for its Elo match to resolve); the selection logic
in ``select_seed`` never reads it, same as ``RoundRobinScheduler.select_op``
on the operator side.
"""

from __future__ import annotations


class SeedRoundRobinScheduler:
    """Simple round-robin scheduler for seed selection.

    Cycles through registered corpus seed keys in fixed registration
    order. Provides a deterministic baseline for the seed-arena Elo
    tournament, and needs no arbiter to run standalone.
    """

    #: No meaningful priors for round-robin, mirrors RoundRobinScheduler.
    supports_priors = False

    def __init__(self) -> None:
        self._index = 0
        self._seed_counts: dict[str, list[float]] = {}  # seed_id -> [successes, failures]
        self._seed_order: list[str] = []  # maintained registration order

    def init_arm(self, seed_id: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register a seed key with zero initial counts.

        ``prior_alpha``/``prior_beta`` are accepted for interface parity
        with the other seed schedulers' ``init_arm`` but ignored (see
        ``supports_priors``). Re-registering a seed never resets it.
        """
        if seed_id not in self._seed_counts:
            self._seed_counts[seed_id] = [0.0, 0.0]  # [successes, failures]
            self._seed_order.append(seed_id)

    def select_seed(self, seed_ids: list[str]) -> str:
        """Select the next seed key via round-robin cycling.

        Candidates not yet registered are registered on the fly, the
        same just-in-time behavior every other seed scheduler in this
        package uses, so a caller never has to pre-register the corpus
        before its first pick.
        """
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]
        for seed_id in seed_ids:
            self.init_arm(seed_id)

        # Filter to only registered seeds in preferred (registration)
        # order, same fallback chain as RoundRobinScheduler.select_op.
        available = [s for s in self._seed_order if s in seed_ids]
        if not available:
            available = [s for s in seed_ids if s in self._seed_counts]
        if not available:
            return seed_ids[0]

        seed_id = available[self._index % len(available)]
        self._index += 1
        return seed_id

    def record(self, seed_id: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome for Elo compatibility; ignored by selection."""
        self.init_arm(seed_id)
        r = min(1.0, max(0.0, float(weight))) if success else 0.0
        self._seed_counts[seed_id][0] += r
        self._seed_counts[seed_id][1] += 1.0 - r

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Return success/failure counts for each registered arm."""
        return {
            seed_id: (successes, failures)
            for seed_id, (successes, failures) in sorted(self._seed_counts.items())
        }
