"""PositionKLDUCBScheduler: discounted KL-UCB over a seed's offset bins.

Position-arena counterpart of ``core/schedulers/op_kl_ducb.py``. Every
other *learning* position proposer in the pool (``pos_burn_front.py``) is
an ad-hoc heat/fuel/cooling heuristic; this one reuses the discounted
KL-UCB machinery already swept and fixed at ``xi=0.10, exploration=1.0``
against Garivier-Cappe / Garivier-Moulines for the operator axis (see
``handover_kl_ducb_paper_fidelity_2026-09-14.md``), so the arena gets a
theoretically-grounded rival to burn-front's heuristic on the same axis.

Bins are the arms: same binning scheme as ``pos_burn_front`` /
``pos_canary`` / ``pos_round_robin`` (``width = ceil(len(data) /
MAX_BINS)``, fixed from the parent seed's length; LRU-bounded over
``MAX_SEEDS`` seeds), each with its own ``KL_DUCBScheduler`` instance,
bin index as the arm name. ``UCBBase._arm_count``/``_arm_mean`` default
missing arms to zero evidence, so a bin needs no upfront registration --
it is simply named in the ``ops`` list ``select_op`` is handed each call.

``record()`` maps ``Outcome.GAIN``/``MISS`` to the boolean ``success``
``KL_DUCBScheduler.record`` expects, splitting ``weight`` evenly across
the round's offsets, the same convention ``pos_burn_front.record`` and
``pos_canary.record`` use. Like those two, and like ``round_robin``/
``fibonacci``, it is credited off-policy: ``services/position_arena.py``
feeds every settled round's outcome to this scheduler's per-bin bandit
regardless of which arm actually proposed the offset, so its statistics
reflect the true outcome distribution at each bin, not just the rounds
it happened to win.

No state persistence (mirrors ``op_kl_ducb.py``, which the Fuzzer never
saves either): unlike ``burn_front``'s heat maps, a KL-UCB posterior
rebuilds quickly from a handful of observations and is not worth the
``--resume`` bookkeeping.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import xxhash

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_kl_ducb import KL_DUCBScheduler
from fuzzer_tool.core.schedulers.pos_base import Outcome

MAX_BINS = 4096  # offsets per seed are binned down to this, as pos_burn_front
MAX_SEEDS = 256  # LRU bound on per-seed bandits


@lru_cache(maxsize=64)
def _bin_names(num_bins: int) -> list[str]:
    """Arm names "0".."num_bins-1", built once per size (select_op only reads)."""
    return [str(b) for b in range(num_bins)]


@dataclass
class _Bandit:
    width: int
    ucb: KL_DUCBScheduler


class PositionKLDUCBScheduler:
    """Discounted KL-UCB position proposer: a seed's offset bins as arms.

    A principled bandit alternative to ``BurnFrontPositionScheduler``'s
    heuristic for the same role in the position arena's Elo pool.
    """

    name = "kl_ducb"

    def __init__(
        self,
        rng: RandPool,
        gamma: float = 0.9999,
        xi: float = 0.10,
        exploration: float = 1.0,
    ) -> None:
        self._rng = rng
        self._gamma = gamma
        self._xi = xi
        self._exploration = exploration
        self._bandits: OrderedDict[int, _Bandit] = OrderedDict()

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """First byte of the KL-UCB-selected bin; None on an empty buffer."""
        if buf_len <= 0:
            return None
        bandit = self._bandit_for(data)
        num_bins = max(1, -(-len(data) // bandit.width)) if data else 1

        arm = bandit.ucb.select_op(_bin_names(num_bins))
        if not arm:
            return self._rng.randint(0, buf_len - 1)

        last = buf_len - 1
        return min(int(arm) * bandit.width, last)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """One bandit pull per offset's bin, ``weight`` split across the round.

        An offset past the parent seed's length (the child grew) still
        credits its own extrapolated bin, clamped to the last known bin --
        mirrors ``pos_burn_front.record``'s handling of the same case.
        """
        offsets = [o for o in offsets if o >= 0]
        if not offsets:
            return
        bandit = self._bandit_for(data)
        num_bins = max(1, -(-len(data) // bandit.width)) if data else 1

        success = outcome is Outcome.GAIN
        share = weight / len(offsets)
        for off in offsets:
            b = min(off // bandit.width, num_bins - 1)
            bandit.ucb.record(str(b), success, weight=share)

    def seed_count(self) -> int:
        return len(self._bandits)

    def bandit_stats(self, data: bytes) -> dict:
        """KL-D-UCB diagnostics for one seed's bandit; empty when unknown."""
        bandit = self._bandits.get(self._key(data))
        return bandit.ucb.bandit_stats() if bandit else {}

    @staticmethod
    def _key(data: bytes) -> int:
        return xxhash.xxh3_64_intdigest(data)

    def _bandit_for(self, data: bytes) -> _Bandit:
        key = self._key(data)
        bandit = self._bandits.get(key)
        if bandit is None:
            width = max(1, -(-len(data) // MAX_BINS)) if data else 1
            ucb = KL_DUCBScheduler(
                gamma=self._gamma,
                xi=self._xi,
                exploration=self._exploration,
                rng=self._rng,
            )
            bandit = self._bandits[key] = _Bandit(width=width, ucb=ucb)
            while len(self._bandits) > MAX_SEEDS:
                self._bandits.popitem(last=False)
        self._bandits.move_to_end(key)
        return bandit
