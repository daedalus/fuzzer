"""PositionChangedScheduler: skip bytes the target never reads.

Every learning arm here trains on ``GAIN``/``MISS``, which cannot tell an
*inert* byte (the target ignores it) from a *live but unproductive* one (the
target reads it, nothing new yet): both are misses. The execution trace can.
AFL's effector map asks "did the trace move?" per byte at one exec per byte;
this arm asks it of every round's mutated offsets, for free, and pools the
answer the way group testing does (``core/skipdet.py`` names this gap)::

    round mutates {3, 17, 40}, trace unchanged  ->  all three look inert
    round mutates {3, 17, 40}, trace moved      ->  each gets 1/3 of a hit

    w(bin) = (moved + PRIOR_A) / (trials + PRIOR_A + PRIOR_B)   (_bin_rates)

One unchanged round clears every offset it touched at once; a moved round
spreads credit, since any one of them may have done it. Inert regions sink to
a low weight, never zero, and untried bins keep the prior's 0.5.

``moved(data)`` is injected (``Fuzzer._path_moved``): the current trace's
rolling path hash against the parent seed's, or None when either is
unavailable (no SHM, shim without the hash, no baseline) -- the round is then
not credited. The signal is independent of the round's outcome and weight.

Off-policy extra: credited every settled round whoever served. Declines
(uniform, charged to the arm) on a seed with no evidence yet. Not persisted
(like ``kl_ducb``: the table rebuilds from a few rounds).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bin_rates import BinRates
from fuzzer_tool.core.schedulers.pos_base import Outcome

PRIOR_A = 1.0  # Laplace prior: an untried bin weighs 0.5
PRIOR_B = 1.0


class PositionChangedScheduler:
    """Rate-weighted bins, rate = how often mutating the bin moved the trace."""

    name = "changed"

    def __init__(self, rng: RandPool, moved: Callable[[bytes], bool | None]) -> None:
        self._moved = moved
        self._rates = BinRates(rng, PRIOR_A, PRIOR_B)

    def propose(self, data: bytes, buf_len: int) -> int | None:
        return self._rates.propose(data, buf_len)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Pooled test: unchanged -> 0 for all; moved -> 1/k each."""
        live = [o for o in offsets if o >= 0]
        if not live:
            return

        moved = self._moved(data)
        if moved is None:
            return

        self._rates.credit(data, live, 1.0 / len(live) if moved else 0.0)

    def weights(self, data: bytes, buf_len: int) -> np.ndarray | None:
        """Per-bin weights (tests, stats); None before any evidence."""
        return self._rates.weights(data, buf_len)
