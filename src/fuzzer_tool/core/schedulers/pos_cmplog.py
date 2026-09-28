"""PositionCmplogScheduler: land mutations on comparison operands.

Cmplog already tells the fuzzer where a seed's bytes are compared against
something: ``seed_meta[seed]["redqueen_offsets"]`` (up to 50 offsets, set
where redqueen matching runs) and, with ``--weizz-tags``, a Weizz
``StructureMap`` whose spans are flagged ``IS_LEN`` / ``IS_MAGIC`` /
``IS_CHECKSUM`` / ``IS_INPUT_TO_STATE``. Today only the redqueen operator
reads them. This arm hands the same offsets to *every* operator, so a bit
flip, an arithmetic op or a havoc step can land on a magic number, a length
field or a checksum::

    targets = flagged spans (weights: len 1.5, others 1.0)
            + redqueen offsets (weight 1.0, +/-1 jitter)

    pick    = weighted over targets; uniform inside a span

Passive in v1: ``record()`` is a no-op, so the arm learns nothing and keeps
no persisted state. The per-seed target lists are a derived cache (an LRU of
``MAX_SEEDS`` seeds), rebuilt when the seed's metadata changes shape: tags go
dirty, the tag map is replaced, or redqueen finds more offsets.

Declines (``None``, which the arena turns into a uniform offset charged to
this arm) when the seed has no cmplog data at all, and with probability
``EPSILON`` otherwise, so a stale or wrong target set cannot lock the arm
onto a fixed subset. A pure decliner rates exactly as uniform in the arena.

Wiring differs from the learners: it is a tracker-style arm (see
``PositionArena._add_trackers``) gated on cmplog being live, not an
off-policy "extra".
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import xxhash

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.weizz_tags import TagFlags

log = logging.getLogger(__name__)

MAX_SEEDS = 256  # LRU bound on the per-seed target cache
MAX_TARGETS = 512  # cap on spans + points kept per seed
EPSILON = 0.1  # uniform escape (decline) probability
POINT_JITTER = 1  # redqueen offsets get +/- this many bytes

W_LEN = 1.5
W_CHECKSUM = 1.0
W_MAGIC = 1.0
W_INPUT_TO_STATE = 1.0
W_POINT = 1.0

# (flag, weight); a span carrying several flags takes the largest weight.
SPAN_FLAGS: tuple[tuple[TagFlags, float], ...] = (
    (TagFlags.IS_LEN, W_LEN),
    (TagFlags.IS_CHECKSUM, W_CHECKSUM),
    (TagFlags.IS_MAGIC, W_MAGIC),
    (TagFlags.IS_INPUT_TO_STATE, W_INPUT_TO_STATE),
)


@dataclass(slots=True)
class _Target:
    """One place to land: ``[start, end)``; a point has ``end == start + 1``."""

    start: int
    end: int
    weight: float
    point: bool


@dataclass(slots=True)
class _Cached:
    sig: tuple
    targets: list[_Target]
    extent: int  # max start over targets: cheap "does the buffer cover all of them"


class PositionCmplogScheduler:
    """Propose offsets from redqueen matches and Weizz-flagged spans."""

    name = "cmplog"

    def __init__(
        self,
        rng: RandPool,
        meta_of: Callable[[bytes], dict | None],
        smap_of: Callable[[bytes], object | None],
    ) -> None:
        self._rng = rng
        self._meta_of = meta_of
        self._smap_of = smap_of
        self._cache: OrderedDict[int, _Cached] = OrderedDict()

    # -- target construction ------------------------------------------------

    @staticmethod
    def _signature(meta: dict) -> tuple:
        """Cheap fingerprint of the metadata the targets are derived from."""
        rq = meta.get("redqueen_offsets")
        return (
            id(meta),
            bool(meta.get("weizz_tags_dirty")),
            len(rq) if isinstance(rq, (list, tuple)) else 0,
            meta.get("weizz_tags_len"),
        )

    def _build(self, data: bytes, meta: dict) -> list[_Target]:
        spans: dict[tuple[int, int], float] = {}
        try:
            smap = self._smap_of(data)  # None when absent or dirty
        except Exception:  # never raise on the hot path
            log.debug("cmplog position: structure map unavailable", exc_info=True)
            smap = None
        if smap is not None:
            for flag, weight in SPAN_FLAGS:
                try:
                    found = smap.flagged_spans(flag)
                except Exception:
                    log.debug("cmplog position: flagged_spans failed", exc_info=True)
                    continue
                for start, end, _cid in found:
                    if start < 0 or end <= start:
                        continue
                    key = (int(start), int(end))
                    if weight > spans.get(key, 0.0):
                        spans[key] = weight

        targets = [_Target(s, e, w, False) for (s, e), w in spans.items()]

        points: set[int] = set()
        rq = meta.get("redqueen_offsets")
        if isinstance(rq, (list, tuple)):
            for o in rq:
                if isinstance(o, int) and not isinstance(o, bool) and o >= 0:
                    points.add(o)
        targets.extend(_Target(o, o + 1, W_POINT, True) for o in sorted(points))

        if len(targets) > MAX_TARGETS:
            # Stable sort: keeps the earliest of equal weight.
            targets.sort(key=lambda t: -t.weight)
            del targets[MAX_TARGETS:]
        return targets

    def _targets(self, data: bytes) -> _Cached | None:
        try:
            meta = self._meta_of(data)
        except Exception:
            log.debug("cmplog position: meta lookup failed", exc_info=True)
            return None
        if not isinstance(meta, dict):
            return None

        key = xxhash.xxh3_64_intdigest(data)
        sig = self._signature(meta)
        hit = self._cache.get(key)
        if hit is not None and hit.sig == sig:
            self._cache.move_to_end(key)
            return hit

        targets = self._build(data, meta)
        entry = _Cached(sig, targets, max((t.start for t in targets), default=0))
        self._cache[key] = entry
        self._cache.move_to_end(key)
        while len(self._cache) > MAX_SEEDS:
            self._cache.popitem(last=False)
        return entry

    # -- protocol -------------------------------------------------------------

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None
        entry = self._targets(data)
        if entry is None or not entry.targets:
            return None
        if self._rng.random() < EPSILON:
            return None

        targets = entry.targets
        if entry.extent >= buf_len:  # buffer shrank: drop targets past its end
            targets = [t for t in targets if t.start < buf_len]
            if not targets:
                return None

        t = self._rng.weighted_choice(targets, [x.weight for x in targets])
        if t.point:
            pos = t.start + self._rng.randint(-POINT_JITTER, POINT_JITTER)
        else:
            pos = self._rng.randint(t.start, min(t.end, buf_len) - 1)
        return min(max(pos, 0), buf_len - 1)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """No-op in v1: the arm is passive (targets come from seed metadata)."""
