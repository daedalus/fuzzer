"""PositionLineageScheduler: land mutations near the sites that made the seed.

With ``--lineage`` the corpus records, for every admitted child,
``seed_meta[child]["parent_sites"]``: the offsets its producing round
mutated, next to ``parent_ops`` (the operators, aligned with the sites) and
``lineage_depth``. A child exists because a mutation at one of those sites
found coverage, which makes them the best cold-start prior for the child's
own mutations: the learning arms (``burn_front``, ``kl_ducb``, ``fractal``)
start from zero on every new seed, this one starts from where its parent
paid off::

    sites   = parent_sites, minus delocalised operators' sites
              (only when parent_ops is aligned with parent_sites)
    pick    = uniform over sites
    landing = site + two-sided geometric jitter (mean JITTER bytes),
              reflected at the buffer edges

The jitter makes a run of mutations cover the neighbourhood instead of
hammering one byte; adjacent bytes are usually one field.

Passive in v1: ``record()`` is a no-op and nothing is persisted (the corpus
already persists the metadata this reads). There is no cache either: the
lookup is one dict get and a short list scan.

Sites are offsets in the *parent's* coordinates. For length-preserving
operators they are also valid in the child; for length-changing ones they
drift, which v1 accepts (the arena's live-buffer coordinates have the same
imprecision). Walking ``parent_key`` several generations up with offset
correction is the deferred v2.

Declines (``None``, which the arena turns into a uniform offset charged to
this arm) when the seed has no usable sites (initial corpus, lineage off,
only delocalised sites) and with probability ``EPSILON`` otherwise, so a
stale lineage cannot lock the arm onto a fixed subset. A pure decliner rates
exactly as uniform in the arena.

Wiring differs from the off-policy learners: it is a tracker-style arm (see
``PositionArena._add_trackers``) gated on ``--lineage`` being on, because
without it no seed carries ``parent_sites`` and the arm would be a pure
decliner that the uniform-floor inspection flags for no reason.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Collection, Sequence

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

log = logging.getLogger(__name__)

EPSILON = 0.2  # uniform escape (decline) probability
JITTER = 8  # mean of the geometric jitter magnitude, in bytes
MAX_SITES = 64  # cap on sites considered per seed (a round lands a handful)
_U_MIN = 1e-12  # guard: log(0) when random() returns exactly 0.0


def reflect(pos: int, n: int) -> int:
    """Fold *pos* into ``[0, n - 1]`` by reflecting at both edges.

    ``-x -> x`` and ``n - 1 + x -> n - 1 - x``; a triangle wave, so any
    distance folds back in range (a site far past a shrunken buffer's end
    included). Reflecting rather than clamping avoids piling proposals on
    the first and last byte.
    """
    if n <= 1:
        return 0
    period = 2 * (n - 1)
    p = pos % period  # Python modulo: non-negative for a positive period
    return p if p <= n - 1 else period - p


class PositionLineageScheduler:
    """Propose offsets from the mutation sites that produced the seed."""

    name = "lineage"

    def __init__(
        self,
        rng: RandPool,
        meta_of: Callable[[bytes], dict | None],
        delocalised: Collection[str] = (),
    ) -> None:
        """``meta_of(seed)`` returns the seed's metadata dict or ``None``.

        ``delocalised`` names the operators whose recorded site is not a
        true offset (``services.operators._DELOCALISED_OPS``); it is
        injected because ``core`` must not import ``services``.
        """
        self._rng = rng
        self._meta_of = meta_of
        self._delocalised = frozenset(delocalised)

    # -- site extraction ------------------------------------------------------

    def _sites(self, data: bytes) -> list[int]:
        """Usable parent sites for *data* (``[]`` when there are none)."""
        try:
            meta = self._meta_of(data)
        except Exception:  # never raise on the hot path
            log.debug("lineage position: meta lookup failed", exc_info=True)
            return []
        if not isinstance(meta, dict):
            return []
        sites = meta.get("parent_sites")
        if not isinstance(sites, (list, tuple)) or not sites:
            return []

        ops = meta.get("parent_ops")
        aligned = (
            bool(self._delocalised) and isinstance(ops, (list, tuple)) and len(ops) == len(sites)
        )
        out: list[int] = []
        for i, s in enumerate(sites):
            if not isinstance(s, int) or isinstance(s, bool) or s < 0:
                continue
            if aligned and ops[i] in self._delocalised:
                continue
            out.append(s)
            if len(out) >= MAX_SITES:
                break
        return out

    # -- protocol -------------------------------------------------------------

    def _jitter(self) -> int:
        """Signed two-sided geometric jitter with mean magnitude ``JITTER``."""
        # Magnitude k >= 0, P(k) = (1-q)^k q with q = 1/(1+JITTER): mean JITTER.
        u = max(self._rng.random(), _U_MIN)
        q = 1.0 / (1.0 + JITTER)
        k = int(math.log(u) / math.log1p(-q))
        return -k if self._rng.random() < 0.5 else k

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None
        sites = self._sites(data)
        if not sites:
            return None
        if self._rng.random() < EPSILON:
            return None

        site = sites[self._rng.randint(0, len(sites) - 1)]
        return reflect(site + self._jitter(), buf_len)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """No-op in v1: the arm is passive (sites come from seed metadata)."""
