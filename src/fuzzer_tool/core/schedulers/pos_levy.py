"""PositionLevyScheduler: a heavy-tailed jump around a seed's last gain offset.

Gains tend to cluster near earlier gains, but at an unknown scale.
``core/schedulers/pos_burn_front.py`` conducts heat with a Gaussian kernel of
fixed width (``KERNEL_SIGMA`` bins), and ``core/schedulers/pos_fractal.py``
adapts *resolution* rather than jump length. This arm keeps one integer per
seed -- the offset of its most recent gain, the *anchor* -- and proposes the
anchor plus a Levy-flight step: most proposals land within a few bytes of the
anchor, a few land far away, and no scale is baked in::

    u    ~ U(0, 1]                      (clamped away from 0)
    step = floor(X_MIN * (u ** (-1 / (ALPHA - 1)) - 1))     Lomax / Pareto II
    pos  = reflect(anchor +/- step)

With ``ALPHA = 2`` and ``X_MIN = 1``, ``step = floor(1/u) - 1``: half the
proposals hit the anchor byte itself (``u > 1/2``), one sixth land one byte
away, and ``P(step >= k) = 1 / (k + 1)``, so the tail is unbounded (capped at
the live buffer length). The ``- 1`` matters: a plain Pareto ``floor(1/u)``
has a minimum step of 1 and could never re-propose the byte that gained.

The step is drawn from the anchor, not from the previous proposal, so this is
a heavy-tailed *kernel* around the last gain, not a random walk that can
drift away from it. Positions past either end of the buffer are *reflected*
(triangle-wave fold), not clamped: clamping would pile every overshoot onto
byte 0 or the last byte, the same failure ``pos_fibonacci.py`` documents for
seed-sized bins on a shrunk buffer.

Staleness. ``record()`` is credited every settled round for the parent seed,
whoever served the position (off-policy, like ``burn_front``/``kl_ducb``/
``fractal``). A gain re-anchors; ``STALE`` consecutive misses drop the anchor
so the arm declines (uniform, charged to itself) rather than orbit a site that
stopped paying. ``SPARK_RATE`` of proposals are uniform escapes regardless.

Per-seed state (LRU-bounded, ``MAX_SEEDS``): the anchor, the current miss
streak, and a ring of the last ``GAP_RING`` distances between consecutive
anchors. The gaps are not read by ``propose``; they are kept so a later
version can fit the tail exponent (``core/zipf.py``) instead of fixing
``ALPHA``. Persisted through ``state_store`` (``Fuzzer._save_learned``) so
``--resume`` keeps the anchors.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import xxhash

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

log = logging.getLogger(__name__)

STATE_VERSION = 1
MAX_SEEDS = 256  # LRU bound on per-seed walks
X_MIN = 1.0  # tail scale, in bytes
ALPHA = 2.0  # tail exponent; P(step >= k) ~ k ** -(ALPHA - 1). Must be > 1.
SPARK_RATE = 0.05  # uniform escapes from the anchor
STALE = 32  # consecutive misses before the anchor is dropped
GAP_RING = 64  # anchor-to-anchor distances kept per seed
_U_FLOOR = 1e-12  # keeps u ** -(1/(ALPHA-1)) finite


@dataclass
class _Walk:
    anchor: int | None = None
    misses: int = 0
    gaps: list[int] = field(default_factory=list)


def _reflect(pos: int, last: int) -> int:
    """Fold *pos* into ``[0, last]`` as a triangle wave (reflecting walls)."""
    if last <= 0:
        return 0
    period = 2 * last
    p = pos % period  # Python modulo: non-negative for a positive period
    return p if p <= last else period - p


class PositionLevyScheduler:
    name = "levy"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng
        self._walks: OrderedDict[int, _Walk] = OrderedDict()

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """Anchor plus a heavy-tailed step; None when the seed has no anchor."""
        walk = self._walks.get(self._key(data))
        if buf_len <= 0 or walk is None or walk.anchor is None:
            return None

        last = buf_len - 1
        if self._rng.random() < SPARK_RATE:
            return self._rng.randint(0, last)

        u = max(self._rng.random(), _U_FLOOR)
        # Cap in float space first: int() of a huge float is exact but pointless.
        step = int(min(X_MIN * (u ** (-1.0 / (ALPHA - 1.0)) - 1.0), buf_len))
        sign = 1 if self._rng.random() < 0.5 else -1
        return _reflect(min(walk.anchor, last) + sign * step, last)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """GAIN re-anchors on one of the round's offsets; a miss streak drops it.

        ``weight`` is accepted for protocol parity and not used: unlike the
        heat maps, the anchor is a single site, so there is nothing to split
        the round's weight across. Every offset of a gain round is an equally
        good candidate, so one is chosen uniformly.
        """
        offsets = [o for o in offsets if o >= 0]
        if not data:
            return

        if outcome is Outcome.GAIN:
            if not offsets:
                return
            walk = self._walk_for(data)
            new = offsets[self._rng.randint(0, len(offsets) - 1)]
            if walk.anchor is not None:
                walk.gaps.append(abs(new - walk.anchor))
                del walk.gaps[:-GAP_RING]
            walk.anchor = new
            walk.misses = 0
            return

        # A miss never creates a walk: there is nothing to go stale.
        walk = self._walks.get(self._key(data))
        if walk is None or walk.anchor is None:
            return
        walk.misses += 1
        if walk.misses >= STALE:
            walk.anchor = None
            walk.misses = 0

    def anchor(self, data: bytes) -> int | None:
        walk = self._walks.get(self._key(data))
        return walk.anchor if walk else None

    def walk_state(self, data: bytes) -> tuple[int | None, int, list[int]] | None:
        """``(anchor, misses, gaps)`` for one seed (gaps copied); None when unknown."""
        walk = self._walks.get(self._key(data))
        return (walk.anchor, walk.misses, list(walk.gaps)) if walk else None

    def seed_count(self) -> int:
        return len(self._walks)

    def to_dict(self) -> dict:
        """Walks oldest-first, so a restore keeps the LRU order."""
        return {
            "version": STATE_VERSION,
            "walks": {k: (w.anchor, w.misses, list(w.gaps)) for k, w in self._walks.items()},
        }

    def from_dict(self, data) -> None:
        """Replace every walk with *data*'s; a malformed payload clears them."""
        self._walks = OrderedDict()
        if not data:
            return
        try:
            if data.get("version") != STATE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            walks: OrderedDict[int, _Walk] = OrderedDict()
            for k, (anchor, misses, gaps) in data["walks"].items():
                anchor = None if anchor is None else int(anchor)
                misses = int(misses)
                gaps = [int(g) for g in gaps][-GAP_RING:]
                if (anchor is not None and anchor < 0) or misses < 0 or any(g < 0 for g in gaps):
                    raise ValueError("negative anchor, miss count or gap")
                walks[int(k)] = _Walk(anchor, misses, gaps)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            log.warning("levy position state unreadable, starting fresh: %s", e)
            return

        # Oldest entries are dropped first, as _walk_for would have.
        while len(walks) > MAX_SEEDS:
            walks.popitem(last=False)
        self._walks = walks

    @staticmethod
    def _key(data: bytes) -> int:
        return xxhash.xxh3_64_intdigest(data)

    def _walk_for(self, data: bytes) -> _Walk:
        key = self._key(data)
        walk = self._walks.get(key)
        if walk is None:
            walk = self._walks[key] = _Walk()
            while len(self._walks) > MAX_SEEDS:
                self._walks.popitem(last=False)
        self._walks.move_to_end(key)
        return walk
