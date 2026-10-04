"""SeedNewestScheduler: the Growing Tree 'newest' policy as a seed arm.

Growing Tree (jamisbuck.org/mazes) is one loop with one knob: keep a
frontier, pick a cell by policy. ``newest`` is the recursive backtracker
(DFS: long corridors), ``random`` is Prim (bushy). Corpus = frontier, seed
picker = policy. The corpus already had ``round_robin`` (oldest) and
``weighted`` (random); no arm picked the newest seed
(docs/handover/handover_maze_algorithms_2026-09-24.md).

The weighted mixture ``newest:p, random:1-p`` from the Growing Tree string
syntax::

    coin < p_newest   ->  most recently registered live seed
    otherwise         ->  uniform live seed

Both knob endpoints are named policies and serve as falsification:
``p_newest=0`` is uniform random, ``p_newest=1`` is pure newest. Pure newest
locks onto the latest seed until a find registers a newer one (same failure
class as ``handover_op_katz_lockin_fix``); a seed is never retired on
exhaustion here, which is the open part of the maze analogy. Off by default
and out of ``--hail-mary`` until it wins its own A/B.

``record()`` feeds the Elo match only; selection never reads it.
"""

from __future__ import annotations

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

DEFAULT_P_NEWEST = 0.5


class SeedNewestScheduler(ArmCounts):
    """Newest live seed with probability ``p_newest``, else uniform."""

    #: No meaningful priors; recency is the whole signal.
    supports_priors = False

    def __init__(self, rng, p_newest: float = DEFAULT_P_NEWEST) -> None:
        if rng is None:
            raise ValueError("SeedNewestScheduler requires a RandPool (Hard Rule 16)")
        # Written to reject NaN too: every comparison with NaN is False.
        if not 0.0 <= p_newest <= 1.0:
            raise ValueError(f"p_newest must be in [0, 1], got {p_newest!r}")
        super().__init__()
        self._rng = rng
        self._p_newest = p_newest
        self._clock = 0
        self._born: dict[str, int] = {}  # key -> registration stamp

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register *name* with the next stamp; re-registering never renews it."""
        if name in self._counts:
            return
        super().init_arm(name)
        self._clock += 1
        self._born[name] = self._clock

    def select_seed(self, seed_ids: list[str]) -> str:
        if not seed_ids:
            return ""
        for key in seed_ids:
            self.init_arm(key)
        self._trim(seed_ids)
        if len(self._born) > len(self._counts):
            self._born = {k: v for k, v in self._born.items() if k in self._counts}
        if len(seed_ids) == 1:
            return seed_ids[0]

        if self._rng.random() < self._p_newest:
            return max(seed_ids, key=self._born.__getitem__)
        return self._rng.choice(seed_ids)
