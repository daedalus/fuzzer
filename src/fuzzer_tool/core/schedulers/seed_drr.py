"""SeedDRRScheduler: deficit round robin over the corpus.

The cost-aware sibling of ``seed_round_robin.py``. Plain round robin gives
every seed the same number of visits, so a seed that takes 10x longer to
execute takes 10x the wall clock. DRR (``core/fair_queue.py``) gives every
seed the same *time*: each visit credits ``quantum * weight`` and a seed is
served while its credit covers its cost, so a slow seed is visited less
often and a favored seed (weight > 1) more often.

    picks per round at quantum=4:  fast seed (cost 1) -> 4
                                   slow seed (cost 4) -> 1     equal time

With flat cost and weight it is exactly ``SeedRoundRobinScheduler``, which
is the falsification condition: the arm can only differ from round robin
where per-seed exec cost varies. Cost and weight are supplied per pick by
the caller (``SeedPicker._pick_seed_drr_seed``) from the cost ledger, so
this module holds no reference to the fuzzer.

Like every arm here it needs the bandit interface (``init_arm`` /
``record`` / ``bandit_stats``) so the seed-arena Elo match can resolve;
``select_seed`` never reads what ``record`` stores.
"""

from __future__ import annotations

from collections.abc import Callable

from fuzzer_tool.core.fair_queue import NEUTRAL_COST, DeficitRR

#: Weight of an AFL-favored seed against 1.0 for the rest. A posture, not a
#: measurement: favored seeds are the minimal set covering every edge, so
#: they earn a larger share of time, but the constant is not fitted to data.
FAVORED_WEIGHT = 2.0


def _unit(_key: str) -> float:
    return 1.0


class SeedDRRScheduler:
    """Deficit round robin for seed selection; cost and weight per pick."""

    #: No meaningful priors, mirrors SeedRoundRobinScheduler.
    supports_priors = False

    def __init__(self, quantum: float = NEUTRAL_COST) -> None:
        self._drr = DeficitRR(quantum)
        self._seed_counts: dict[str, list[float]] = {}  # seed_id -> [successes, failures]

    def init_arm(self, seed_id: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register a seed key; priors are ignored and re-registering never resets."""
        if seed_id not in self._seed_counts:
            self._seed_counts[seed_id] = [0.0, 0.0]

    def select_seed(
        self,
        seed_ids: list[str],
        cost_fn: Callable[[str], float] = _unit,
        weight_fn: Callable[[str], float] = _unit,
    ) -> str:
        """Next seed key by deficit round robin; unregistered keys join on the fly."""
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]

        for seed_id in seed_ids:
            self.init_arm(seed_id)
        return self._drr.pick(seed_ids, cost_fn, weight_fn)

    def record(self, seed_id: str, success: bool, weight: float = 1.0) -> None:
        """Elo-compatibility signal only; selection ignores it."""
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
