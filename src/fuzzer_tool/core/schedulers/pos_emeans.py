"""PositionEMeansScheduler: offsets from an e-means codebook of the seed's gains.

Every other arm turns gain evidence into a per-bin *rate* and draws a bin from
it. This one treats the offsets that gained as a 1-D point cloud and keeps a
small codebook (``K`` centroids) of it, trained the way the e-means paper
(Bessa et al., arXiv:2611.09501; code in ``faiss/contrib/e_means``) trains
k-means (equation, section and table numbers below are the paper's):

* **EMA update** (Eq. 2-4). Each settled gain round is a mini-batch. Every slot's mass
  decays by ``gamma = 1 - alpha`` and a slot that received points moves::

      N_new = gamma * N + alpha * n_j
      c_new = c + (alpha * n_j / N_new) * (mean_j - c)

  so a centroid is a recency-weighted mean and an idle slot only loses mass.
  Stale gain regions fade without a sliding-window buffer.
* **Cosine step** (Eq. 5, floor 0.1). ``alpha`` anneals from ``ALPHA0`` to ``0.1 * ALPHA0`` over
  ``HORIZON`` gain rounds of that seed, then holds (``_scheduled_alpha`` in
  the reference).
* **Initial mass 2**-16.** A freshly seeded slot snaps to its first batch mean.
* **Revival by proportional split** (App. H; opt-in here, ``UNUSED_THRESHOLD > 0``,
  CLI ``--pos-emeans-tau``): every
  ``REVIVE_EVERY`` updates, a slot below ``tau * mean(N)`` is replaced by a split
  of a donor drawn in proportion to mass (Gumbel top-k), mass halved. The dead
  slot's residual mass (below the threshold) is dropped, as in the reference.

Proposals draw a slot in proportion to its mass, then ``c + sigma * N(0, 1)``,
``sigma`` from the slot's EMA second moment (floored at ``len / 256``, one
default bin). ``EXPLORE`` of draws are uniform escapes.

Where this departs from the reference, on purpose:

* **No scaling laws** (Eq. 6-7, Table 2). ``alpha0`` and the batch size come
  from laws fitted at ``n >= 1e5``, ``k >= 256``; a seed has tens of gain
  rounds and ``K = 8``, and the paper lists low dimension as a limitation.
  ``ALPHA0`` is a plain constant and is untuned.
* **Revival is off by default, on a code comment, not on the paper.** The paper
  uses ``tau = 0.1`` and shows it matters (Table 1: +3.0% train / +2.2% test
  error without it; Table 14: 47 dead clusters with it, 4,313 without), but
  only at d >= 96. faiss's ``_NO_REVIVAL_MAX_D`` comment says revival cost
  7-10% held-out error on 2-D geo data and turns it off for ``d <= 2``;
  offsets are 1-D, so that default is followed here. Unverified for this use:
  bench arms ``pos-arena-emeans`` and ``pos-arena-emeans-revive``
  (``--pos-emeans-tau 0.1``) settle it.
* **1-D splits are +-sigma apart, not 1e-4 relative.** In 1-D an
  epsilon split ties on ``argmin`` and the child starves.
* **Empty slots are seeded leader-style** (a point farther than ``SEED_SEP``
  floors from every live slot takes the next empty one) in place of sampling
  ``K`` data points up front; the stream has no up-front data.
* **No finalization pass** (last-epoch unweighted means, Sec. 3.2): the stream
  has no last epoch.
* **Different problem.** The paper minimizes MSE on a fixed dataset over a known
  epoch budget (the cosine decay assumes it). Here the gain stream drifts and
  never ends, so none of its speedup results carry over, only the update rule.

Known bias: evidence is the offsets of gaining rounds only (misses carry no
point), so a region that is proposed more also gains more. ``EXPLORE`` and the
arena's Elo pool are the only counterweights; ``pos_changed`` keeps exposure
in its rate and does not have this.

Weight: a round with ``m`` offsets credits each with ``1/m`` of a point, as
``BinRates`` does. Off-policy: credited every settled round whoever served.
Declines (``None``) until the seed has one gain. Per-seed state is LRU-bounded
(``MAX_SEEDS``), not persisted.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from fuzzer_tool.core.hashing import hash64_len
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers._bin_rates import MAX_SEEDS
from fuzzer_tool.core.schedulers.pos_base import Outcome

K = 8  # codebook slots per seed
ALPHA0 = 0.25  # initial EMA step (untuned; see module doc)
ALPHA_MIN_FACTOR = 0.1  # cosine floor, as faiss base._ALPHA_MIN_FACTOR
HORIZON = 128  # gain rounds over which alpha anneals
INIT_USAGE = 2.0**-16  # faiss base._INIT_USAGE
UNUSED_THRESHOLD = 0.0  # tau; 0 = no revival (reference default for d <= 2)
REVIVE_EVERY = 5  # updates between revival passes
EXPLORE = 0.10  # uniform escapes
MIN_SIGMA_DIV = 256  # sigma floor = max(1, len / MIN_SIGMA_DIV)
SEED_SEP = 2.0  # leader-seeding radius in sigma floors


def scheduled_alpha(alpha0: float, step: int, total: int) -> float:
    """Cosine-anneal *alpha0* toward ``ALPHA_MIN_FACTOR * alpha0`` (faiss form)."""
    frac = min(step, total) / max(1, total)
    cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
    return alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)


class _Codebook:
    """One seed's slots: centroid, EMA mass, EMA second moment, update count."""

    __slots__ = ("c", "m2", "n", "t")

    def __init__(self) -> None:
        self.c = np.zeros(K)
        self.n = np.zeros(K)  # 0 = empty slot
        self.m2 = np.zeros(K)
        self.t = 0

    def sigma(self, floor: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.m2 - self.c * self.c, 0.0)).clip(min=floor)


class PositionEMeansScheduler:
    """Offsets drawn from an EMA-trained 1-D codebook of the seed's gain offsets."""

    name = "emeans"

    def __init__(
        self,
        rng: RandPool,
        *,
        alpha0: float = ALPHA0,
        horizon: int = HORIZON,
        unused_threshold: float = UNUSED_THRESHOLD,
    ) -> None:
        if not 0.0 < alpha0 <= 1.0:
            raise ValueError("alpha0 must be in (0, 1]")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if unused_threshold < 0.0:
            raise ValueError("unused_threshold must be >= 0")
        self._rng = rng
        self._alpha0 = alpha0
        self._horizon = horizon
        self._tau = unused_threshold
        self._books: OrderedDict[int, _Codebook] = OrderedDict()

    # -- protocol ---------------------------------------------------------------

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None
        book = self._books.get(hash64_len(data))
        if book is None or book.t == 0:
            return None

        if self._rng.random() < EXPLORE:
            return self._rng.randint(0, buf_len - 1)

        live = np.flatnonzero(book.n > 0)
        mass = book.n[live]
        pick = self._rng.random() * float(mass.sum())
        j = live[min(int(np.searchsorted(np.cumsum(mass), pick, side="right")), live.size - 1)]
        sigma = float(book.sigma(self._floor(buf_len))[j])
        off = int(round(float(book.c[j]) + self._rng.gauss(0.0, sigma)))
        return min(max(off, 0), buf_len - 1)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Gain rounds are one mini-batch; misses carry no point and are ignored.

        ``weight`` is accepted for protocol parity; each round counts once so
        the EMA mass stays comparable across rounds.
        """
        if outcome is not Outcome.GAIN or not data:
            return
        live = [o for o in offsets if o >= 0]
        if not live:
            return

        key = hash64_len(data)
        book = self._books.get(key)
        if book is None:
            book = self._books[key] = _Codebook()
        self._books.move_to_end(key)
        while len(self._books) > MAX_SEEDS:
            self._books.popitem(last=False)

        top = len(data) - 1
        self._update(book, [min(o, top) for o in live], self._floor(len(data)), top)

    # -- training ---------------------------------------------------------------

    def _floor(self, length: int) -> float:
        return max(1.0, length / MIN_SIGMA_DIV)

    def _update(self, book: _Codebook, xs: list[int], floor: float, top: int) -> None:
        pts = np.asarray(xs, dtype=np.float64)

        for x in pts:  # leader seeding of empty slots
            empty = np.flatnonzero(book.n == 0.0)
            if empty.size == 0:
                break
            live = book.n > 0.0
            if live.any() and float(np.abs(book.c[live] - x).min()) <= SEED_SEP * floor:
                continue
            j = empty[0]
            book.c[j], book.n[j], book.m2[j] = x, INIT_USAGE, x * x

        live = np.flatnonzero(book.n > 0.0)
        label = live[np.abs(pts[:, None] - book.c[live][None, :]).argmin(axis=1)]
        w = 1.0 / pts.size
        cnt = np.bincount(label, minlength=K) * w
        sums = np.bincount(label, weights=pts, minlength=K) * w
        sqs = np.bincount(label, weights=pts * pts, minlength=K) * w

        alpha = scheduled_alpha(self._alpha0, book.t, self._horizon)
        book.n *= 1.0 - alpha
        hit = cnt > 0.0
        mass = book.n[hit] + alpha * cnt[hit]
        step = alpha * cnt[hit] / mass
        book.c[hit] += step * (sums[hit] / cnt[hit] - book.c[hit])
        book.m2[hit] += step * (sqs[hit] / cnt[hit] - book.m2[hit])
        book.n[hit] = mass
        book.t += 1

        if self._tau > 0.0 and book.t % REVIVE_EVERY == 0:
            self._revive(book, floor, top)

    def _revive(self, book: _Codebook, floor: float, top: int) -> None:
        """Split mass-proportional donors into slots below ``tau * mean(N)``."""
        threshold = self._tau * float(book.n.sum()) / K
        unused = book.n < threshold
        donors = ~unused & (book.n > 0.0)
        count = min(int(unused.sum()), int(donors.sum()))
        if count == 0:
            return

        keys = np.full(K, -np.inf)
        for j in np.flatnonzero(donors):  # Gumbel top-k: sampling without replacement
            u = max(self._rng.random(), 1e-300)
            keys[j] = math.log(book.n[j]) - math.log(-math.log(u))
        source = np.argsort(keys)[-count:][::-1]
        target = np.flatnonzero(unused)[:count]

        sigma = book.sigma(floor)
        for s, t in zip(source, target, strict=True):
            centre, spread = float(book.c[s]), float(sigma[s])
            half = book.n[s] / 2.0
            book.c[s] = min(max(centre - spread, 0.0), top)
            book.c[t] = min(max(centre + spread, 0.0), top)
            book.n[s] = book.n[t] = half
            for j in (s, t):
                book.m2[j] = book.c[j] ** 2 + (spread / 2.0) ** 2

    # -- introspection ----------------------------------------------------------

    def centroids(self, data: bytes, buf_len: int | None = None) -> np.ndarray | None:
        """Live ``(centroid, mass, sigma)`` rows sorted by centroid; None before any gain."""
        book = self._books.get(hash64_len(data))
        if book is None or book.t == 0:
            return None
        live = np.flatnonzero(book.n > 0.0)
        sigma = book.sigma(self._floor(buf_len if buf_len else len(data)))
        rows = np.stack([book.c[live], book.n[live], sigma[live]], axis=1)
        return rows[np.argsort(rows[:, 0])]
