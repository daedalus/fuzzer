"""SeedStrideScheduler: stride scheduling over the corpus (Waldspurger 1995).

Deterministic proportional share: each seed holds tickets (its weight) and
is served in proportion, interleaved, with no RNG variance::

    tickets fav=2 std=1:   fav std fav fav std fav ...

Weight comes from the caller per pick (``SeedPicker``: AFL-favored 2x), so
this module holds no reference to the fuzzer. Flat weight is exactly
``seed_round_robin`` -- the falsification condition. ``record`` only feeds
the Elo match; selection ignores it.
"""

from __future__ import annotations

from collections.abc import Callable

from fuzzer_tool.core.fair_queue import Stride
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts


def _unit(_key: str) -> float:
    return 1.0


class SeedStrideScheduler(ArmCounts):
    """Stride scheduling for seed selection; tickets per pick."""

    #: No informative priors: selection reads tickets, never the posterior.
    supports_priors = False

    def __init__(self) -> None:
        super().__init__()
        self._stride = Stride()

    def select_seed(self, seed_ids: list[str], weight_fn: Callable[[str], float] = _unit) -> str:
        self._trim(seed_ids)
        return self._stride.pick(seed_ids, weight_fn)
