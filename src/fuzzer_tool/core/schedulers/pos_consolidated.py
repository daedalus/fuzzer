"""PositionConsolidatedScheduler: the learning position arms in one proposer.

Each position arm owns one idea; this one keeps the idea each contributes:
experts *propose* candidates, two rate tables *score* them, and a bounded
weighted draw picks one::

    candidates                         scored by (product, each clamped)
    K_UNIFORM uniform offsets  (spark)     context tilt   cross-seed byte-context rate
    boundary  content edges    (cold)      bin tilt       per-seed offset-bin rate
    levy      jump at last gain (local)
    bins      tilt-weighted bin (per-seed)

    feature                     from
    exploration floor           uniform / burn_front spark
    warm start on a new seed    pos_context (rates pooled across seeds)
    structure without feedback  pos_boundary (class/delimiter/entropy edges)
    locality of gains           pos_levy (heavy-tailed walk, stale anchor dropped)
    per-seed exploitation       pos_kl_ducb / burn_front / rare_mask (bin rates)
    bounded tilt, no collapse   pos_context (each tilt in [W_MIN, W_MAX])

Cold (no evidence) every weight is 1.0: a uniform pick among uniform and
boundary candidates -- the falsification condition. Never declines on a
non-empty buffer.

``record`` is credited off-policy on every settled round, whoever served
the positions: context and levy learn as their own arms do; a bin gets one
trial per offset, a success per offset on GAIN. Context and levy state is
persisted (``to_dict``/``from_dict``); bin rates rebuild quickly and are not.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bin_rates import MAX_BINS, BinRates
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.schedulers.pos_boundary import PositionBoundaryScheduler
from fuzzer_tool.core.schedulers.pos_context import (
    PRIOR_A,
    PRIOR_B,
    W_MAX,
    W_MIN,
    PositionContextScheduler,
)
from fuzzer_tool.core.schedulers.pos_levy import PositionLevyScheduler

log = logging.getLogger(__name__)

K_UNIFORM = 6  # uniform candidates per proposal: the exploration floor
STATE_VERSION = 1


def _width(data: bytes) -> int:
    """Bin width, as ``_bin_rates``: fixed from the parent seed's length."""
    return max(1, -(-len(data) // MAX_BINS))


class PositionConsolidatedScheduler:
    """Experts propose, cross-seed and per-seed rates score, bounded draw picks."""

    name = "consolidated"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng
        self._context = PositionContextScheduler(rng)
        self._levy = PositionLevyScheduler(rng)
        self._boundary = PositionBoundaryScheduler(rng)
        self._bins = BinRates(rng, PRIOR_A, PRIOR_B)
        # (parent, credits, tilt, prefix sum): a round's proposals share one
        # parent, so the O(bins) tilt is rebuilt only after a credit.
        self._credits = 0
        self._memo: tuple[bytes, int, list[float], np.ndarray] | None = None

    @property
    def context_obs(self) -> int:
        return self._context.obs

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """Best-weighted of the experts' candidates; None on an empty buffer."""
        if buf_len < 1 or not data:
            return None

        span = min(buf_len, len(data))
        cands = self._rng.randint_list(0, span - 1, K_UNIFORM)
        last = buf_len - 1
        for expert in (self._boundary, self._levy):
            pos = expert.propose(data, buf_len)
            if pos is not None:
                cands.append(min(max(pos, 0), last))

        # One O(bins) pass serves both the bin expert and the scoring.
        tilt, cum = self._tilt_for(data)
        if tilt is not None:
            cands.append(self._bin_draw(cum, _width(data), buf_len))

        weights = [self._score(data, o, tilt) for o in cands]
        return int(self._rng.weighted_choice(cands, weights))

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Off-policy credit to every learner; invalid weights skip the bins."""
        self._context.record(data, offsets, outcome, weight)
        self._levy.record(data, offsets, outcome, weight)
        if not math.isfinite(weight) or weight <= 0:
            return

        share = 1.0 if outcome is Outcome.GAIN else 0.0
        self._bins.credit(data, offsets, share)
        self._credits += 1

    def weight(self, data: bytes, offset: int) -> float:
        """Score of ``offset`` in ``data``: context tilt times bin tilt."""
        return self._score(data, offset, self._tilt_for(data)[0])

    def anchor(self, data: bytes) -> int | None:
        return self._levy.anchor(data)

    def seed_count(self) -> int:
        return self._bins.seed_count()

    def to_dict(self) -> dict:
        return {
            "version": STATE_VERSION,
            "context": self._context.to_dict(),
            "levy": self._levy.to_dict(),
        }

    def from_dict(self, data) -> None:
        """Replace persisted state with *data*'s; a malformed payload starts fresh."""
        self._context.from_dict({})
        self._levy.from_dict({})
        if not data:
            return

        try:
            if data.get("version") != STATE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            context, levy = data["context"], data["levy"]
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            log.warning("consolidated position state unreadable, starting fresh: %s", e)
            return

        self._context.from_dict(context)
        self._levy.from_dict(levy)

    def _score(self, data: bytes, offset: int, tilt: list[float] | None) -> float:
        w = self._context.tilt(data, offset)
        if tilt is None:
            return w

        return w * tilt[min(offset // _width(data), len(tilt) - 1)]

    def _bin_draw(self, cum: np.ndarray, width: int, buf_len: int) -> int:
        """A byte in a tilt-weighted bin starting inside *buf_len*.

        *cum* is the tilt's prefix sum; its first ``live`` entries are the
        prefix sum over the live bins, so a shrunk buffer needs no rebuild.
        """
        live = min(len(cum), -(-buf_len // width))
        top = cum[live - 1]
        b = min(int(np.searchsorted(cum[:live], self._rng.random() * top, side="right")), live - 1)

        start = b * width
        span = min(width, buf_len - start)
        return start if span <= 1 else start + self._rng.randint(0, span - 1)

    def _tilt_for(self, data: bytes) -> tuple[list[float] | None, np.ndarray | None]:
        """Memoized ``(tilt, prefix sum)`` of *data*; ``(None, None)`` when unseen."""
        memo = self._memo
        if memo is not None and memo[0] is data and memo[1] == self._credits:
            return memo[2], memo[3]

        tilt = self._bin_tilt(data)
        if tilt is None:
            return None, None

        self._memo = (data, self._credits, tilt.tolist(), np.cumsum(tilt))
        return self._memo[2], self._memo[3]

    def _bin_tilt(self, data: bytes) -> np.ndarray | None:
        """Per-bin rate over the seed's pooled rate, clamped; None when unseen.

        e.g. bin 3 gains / 3 trials, seed 3 / 6: (4 / 24) / (4 / 27) = 1.125
        """
        counts = self._bins.counts(data)
        if counts is None:
            return None

        n = counts[0].astype(np.float64)
        s = counts[1].astype(np.float64)
        ab = PRIOR_A + PRIOR_B
        pooled = (s.sum() + PRIOR_A) / (n.sum() + ab)
        return np.clip((s + PRIOR_A) / (n + ab) / pooled, W_MIN, W_MAX)
