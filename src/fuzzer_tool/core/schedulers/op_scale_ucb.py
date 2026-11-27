"""ScaleUCBScheduler: choose the mutation scale first, then the operator.

An exec budget cannot resolve *where* (a bit flip pins one position, says
nothing about structure) and *what shape* (a splice moves structure, smears
position) at once. So learn the scale separately: few bands, each pooling the
evidence of all its operators, resolve fast; the operator inside the band
refines on what is left.

    candidates --group by band--> {bit, byte, chunk, field, structure, learned}
        band  = MultiScaleUCB over the bands present
        op    = MultiScaleUCB over that band's candidates
    record(op) -> band arm and op arm both credited

Bands come from the registry category (``category_of``); an unknown name goes
to ``OTHER_BAND`` so it stays selectable. Both levels read per-arm at their
own resolution (Lepski, see ``op_ms_ucb.py``), so a band whose yield steps
is re-read at a short memory.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from fuzzer_tool.core.operator_categories import category_of
from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool
from fuzzer_tool.core.schedulers.op_ms_ucb import MultiScaleUCBScheduler

#: Band for operators the registry does not know.
OTHER_BAND = "other"

#: Registry category -> mutation scale band, finest to coarsest.
BAND_OF_CATEGORY = {
    "bit": "bit",
    "byte": "byte",
    "block": "chunk",
    "regularity": "chunk",
    "dict": "field",
    "format": "field",
    "structural": "structure",
    "radamsa": "structure",
    "adaptive": "learned",
}


#: Every band, fixed order, for stable diagnostics keys.
_ALL_BANDS = (*dict.fromkeys(BAND_OF_CATEGORY.values()), OTHER_BAND)


class ScaleUCBScheduler:
    """Two-level multi-scale UCB: mutation scale band, then operator.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16), given to both levels.
    """

    supports_priors = False  # Both levels keep discounted sums, no Beta prior.

    def __init__(self, rng: RandPool | None = None) -> None:
        rng = rng if rng is not None else get_default_rand_pool()
        self._bands = MultiScaleUCBScheduler(rng=rng)
        self._ops = MultiScaleUCBScheduler(rng=rng)
        self._band_cache: dict[str, str] = {}
        self._band_pulls: Counter[str] = Counter()
        self._pulls = 0

    def band_of(self, op: str) -> str:
        """Scale band of *op*; memoised (the registry is fixed per run)."""
        band = self._band_cache.get(op)
        if band is None:
            band = BAND_OF_CATEGORY.get(category_of(op), OTHER_BAND)
            self._band_cache[op] = band
        return band

    def init_arm(self, name: str) -> None:
        """Arms are created lazily at both levels."""
        self.band_of(name)

    def select_op(self, ops: list[str]) -> str:
        """Pick a band among the candidates' bands, then an op inside it."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        groups: dict[str, list[str]] = {}
        band_of = self.band_of
        for op in ops:
            groups.setdefault(band_of(op), []).append(op)

        band = self._bands.select_op(list(groups))
        return self._ops.select_op(groups[band])

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Credit both the operator and its band."""
        band = self.band_of(name)
        self._pulls += 1
        self._band_pulls[band] += 1
        self._bands.record(band, success, weight)
        self._ops.record(name, success, weight)

    def bandit_stats(self) -> dict[str, Any]:
        """Return total and per-band pull counts (numeric, JSON-safe)."""
        stats = {"scale_ucb_pulls": self._pulls}
        for band in _ALL_BANDS:
            stats[f"scale_ucb_band_{band}"] = self._band_pulls[band]
        return stats
