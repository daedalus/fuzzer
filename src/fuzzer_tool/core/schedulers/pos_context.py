"""PositionContextScheduler: which offsets pay off, learned from byte context.

Every other learning position arm (``pos_burn_front``, ``pos_kl_ducb``,
``pos_fractal``) keys its evidence on *offsets within one seed*: each new
seed starts cold, and byte content is invisible to it. This arm keys on the
*local byte context* of an offset instead, and pools the evidence across the
whole corpus, so a brand-new seed of a familiar format starts warm.

Context of offset ``o`` in parent seed ``data``::

    ctx = (cls(data[o]), cls(data[o-1]) or BOF, decile(o / len(data)))

    e.g.  "PNG\\r\\n"  o=4 -> (CONTROL, CONTROL, 0)   7 * 8 * 10 = 560 cells

Classes come from ``_bytecls.py``. ``BOF`` is an eighth "previous" value for
``o == 0``; the decile is the offset's relative position, so header, body and
trailer contexts do not blur together.

Per cell, two discounted counts (``succ``, ``fail``) form a Beta-style rate
``(succ + PRIOR_A) / (succ + fail + PRIOR_A + PRIOR_B)``. ``propose`` draws
``K_CANDIDATES`` uniform offsets, weights each by its cell's rate over the
global rate (clamped to ``[W_MIN, W_MAX]``) and picks one: a bounded tilt
over uniform, never a collapse onto a few offsets. Below ``MIN_OBS`` credited
offsets it declines (uniform, charged to the arm by the arena).

``record`` is credited off-policy on every settled round, whoever served the
positions, splitting ``weight`` evenly across the round's offsets like the
other arms. Both tables are multiplied by ``DISCOUNT`` every
``DISCOUNT_EVERY`` credited offsets so stale evidence fades.

State is global and fixed-size (two ``NUM_CTX`` float tables): no per-seed
LRU. Persisted via ``state_store`` for ``--resume``.
"""

from __future__ import annotations

import logging
import math
from array import array
from collections.abc import Sequence
from typing import NamedTuple

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bytecls import BYTE_CLASS, NUM_CLASSES
from fuzzer_tool.core.schedulers.pos_base import Outcome

log = logging.getLogger(__name__)

BOF_PREV = NUM_CLASSES  # "previous class" of offset 0
NUM_PREV = NUM_CLASSES + 1
DECILES = 10
NUM_CTX = NUM_CLASSES * NUM_PREV * DECILES  # 560

MIN_OBS = 200  # credited offsets before the arm stops declining
K_CANDIDATES = 16  # uniform draws reweighted per proposal
PRIOR_A = 1.0  # miss-dominated base rate: 1 / 21
PRIOR_B = 20.0
W_MIN = 0.25  # weight clamp: a mild tilt, never a collapse
W_MAX = 4.0
DISCOUNT_EVERY = 256  # credited offsets between discounts
DISCOUNT = 0.95
STATE_VERSION = 1


class ContextStat(NamedTuple):
    """One context cell as reported by ``top_contexts``."""

    cls: int
    prev_cls: int
    decile: int
    rate: float
    count: float


def ctx_index(data: bytes, o: int) -> int:
    """Flat table index of offset ``o``'s context in ``data``."""
    cls = BYTE_CLASS[data[o]]
    prev = BYTE_CLASS[data[o - 1]] if o else BOF_PREV
    decile = min(DECILES - 1, o * DECILES // len(data))
    return (cls * NUM_PREV + prev) * DECILES + decile


def _rate(succ: float, fail: float) -> float:
    return (succ + PRIOR_A) / (succ + fail + PRIOR_A + PRIOR_B)


def _parse_table(raw: object) -> array:
    """Validated float table from persisted state; raises on any defect."""
    if not isinstance(raw, list | tuple):
        raise TypeError("table is not a list")
    if len(raw) != NUM_CTX:
        raise ValueError(f"table length {len(raw)} != {NUM_CTX}")
    for v in raw:
        if not isinstance(v, int | float):
            raise TypeError(f"non-numeric cell {v!r}")
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"cell {v!r} not finite and >= 0")
    return array("d", raw)


class PositionContextScheduler:
    """Byte-context position proposer with evidence pooled across seeds."""

    name = "context"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng
        self._reset()

    @property
    def obs(self) -> int:
        """Credited offsets so far (never discounted)."""
        return self._obs

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """Best-weighted of K uniform candidates; None while cold or empty."""
        if self._obs < MIN_OBS or not data or buf_len < 1:
            return None

        span = min(buf_len, len(data))
        cands = self._rng.randint_list(0, span - 1, K_CANDIDATES)
        base = _rate(self._succ_total, self._fail_total)
        weights = [self._weight(ctx_index(data, o), base) for o in cands]
        return int(self._rng.weighted_choice(cands, weights))

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Credit each in-range offset's context; ``weight`` split evenly."""
        if not data or not math.isfinite(weight) or weight <= 0:
            return
        valid = [o for o in offsets if 0 <= o < len(data)]
        if not valid:
            return

        share = weight / len(valid)
        gain = outcome is Outcome.GAIN
        for o in valid:
            # Re-read each pass: _discount() below swaps the arrays.
            table = self._succ if gain else self._fail
            table[ctx_index(data, o)] += share
            if gain:
                self._succ_total += share
            else:
                self._fail_total += share
            self._obs += 1
            if self._obs % DISCOUNT_EVERY == 0:
                self._discount()

    def context_counts(self, data: bytes, offset: int) -> tuple[float, float]:
        """``(succ, fail)`` of the cell holding ``offset``; zeros if out of range."""
        if not 0 <= offset < len(data):
            return (0.0, 0.0)
        idx = ctx_index(data, offset)
        return (self._succ[idx], self._fail[idx])

    def tilt(self, data: bytes, offset: int) -> float:
        """Clamped context-rate tilt of ``offset``; 1.0 while cold or out of range."""
        if self._obs < MIN_OBS or not 0 <= offset < len(data):
            return 1.0
        return self._weight(ctx_index(data, offset), _rate(self._succ_total, self._fail_total))

    def top_contexts(self, n: int) -> list[ContextStat]:
        """The ``n`` best-rated seen contexts, best first."""
        return self._ranked()[: max(0, n)]

    def worst_contexts(self, n: int) -> list[ContextStat]:
        """The ``n`` worst-rated seen contexts, worst first."""
        return self._ranked()[::-1][: max(0, n)]

    def to_dict(self) -> dict:
        return {
            "version": STATE_VERSION,
            "succ": list(self._succ),
            "fail": list(self._fail),
            "obs": self._obs,
        }

    def from_dict(self, data) -> None:
        """Replace all state with *data*'s; a malformed payload starts fresh."""
        self._reset()
        if not data:
            return
        try:
            if data.get("version") != STATE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            succ = _parse_table(data["succ"])
            fail = _parse_table(data["fail"])
            obs = data["obs"]
            if not isinstance(obs, int) or obs < 0:
                raise ValueError(f"obs {obs!r}")
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            log.warning("context position state unreadable, starting fresh: %s", e)
            return

        self._succ, self._fail, self._obs = succ, fail, obs
        self._succ_total, self._fail_total = sum(succ), sum(fail)

    def _reset(self) -> None:
        self._succ = array("d", bytes(8 * NUM_CTX))
        self._fail = array("d", bytes(8 * NUM_CTX))
        self._succ_total = 0.0
        self._fail_total = 0.0
        self._obs = 0

    def _weight(self, idx: int, base: float) -> float:
        w = _rate(self._succ[idx], self._fail[idx]) / base
        return min(W_MAX, max(W_MIN, w))

    def _discount(self) -> None:
        self._succ = array("d", (v * DISCOUNT for v in self._succ))
        self._fail = array("d", (v * DISCOUNT for v in self._fail))
        self._succ_total = sum(self._succ)
        self._fail_total = sum(self._fail)

    def _ranked(self) -> list[ContextStat]:
        """Seen contexts, best rate first (ties: larger count first)."""
        stats = []
        for idx in range(NUM_CTX):
            s, f = self._succ[idx], self._fail[idx]
            if s + f <= 0:
                continue
            rest, decile = divmod(idx, DECILES)
            cls, prev = divmod(rest, NUM_PREV)
            stats.append(ContextStat(cls, prev, decile, _rate(s, f), s + f))
        return sorted(stats, key=lambda c: (c.rate, c.count), reverse=True)
