"""SeedEEVDFScheduler: earliest eligible virtual deadline first (Linux >= 6.6).

The seed-arena port of the Linux CFS successor. Like ``seed_drr`` every seed
gets an equal share of target *time* (cost from the cost ledger, favored
seeds 2x); unlike DRR, a seed that ran ahead of the virtual clock is not
eligible until the clock catches up, and a new seed joins at the clock with
zero lag -- served soon, but never allowed to catch up on history::

    cost a=1 b=8:   a b a a a a a a a b ...

Mechanics live in ``core/fair_queue.py::EEVDF``. Flat cost and weight are
exactly ``seed_round_robin``. ``record`` only feeds the Elo match.
"""

from __future__ import annotations

from collections.abc import Callable

from fuzzer_tool.core.fair_queue import EEVDF, NEUTRAL_COST
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts


def _unit(_key: str) -> float:
    return 1.0


class SeedEEVDFScheduler(ArmCounts):
    """EEVDF for seed selection; cost and weight per pick."""

    #: No informative priors: selection reads cost and weight only.
    supports_priors = False

    def __init__(self, slice_: float = NEUTRAL_COST) -> None:
        super().__init__()
        self._eevdf = EEVDF(slice_)

    def select_seed(
        self,
        seed_ids: list[str],
        cost_fn: Callable[[str], float] = _unit,
        weight_fn: Callable[[str], float] = _unit,
    ) -> str:
        self._trim(seed_ids)
        return self._eevdf.pick(seed_ids, cost_fn, weight_fn)
