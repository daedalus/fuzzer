"""Gross corpus flux: telling a quiet corpus from a balanced one.

The corpus size is a *net* quantity, and a net quantity cannot distinguish
the two ways a plateau happens:

- **Quiet.** Additions and evictions have both gone to roughly zero. Nothing
  new is being admitted and nothing is being displaced. That is a stall.
- **Balanced.** Additions and evictions are both running, at the same rate.
  The fuzzer is still admitting seeds that cover something and still evicting
  ones that have been subsumed. That is a healthy dynamic equilibrium.

Both hold ``len(corpus)`` flat, so the number the report already shows is
identical in the two cases. The gross flux ``additions + evictions``
separates them: it is ~0 in the first and large in the second. This is the
distinction the ``dynamic equilibrium`` sense of the word carries and the net
figure throws away -- a system at equilibrium is not a system at rest, it is
one whose forward and reverse rates match.

Rejections are counted as a third channel rather than folded into either.
A rejected candidate is not an addition that was later undone: it never
entered, it displaced nothing, and its rate answers a different question
(how hard admission is working) from the rate at which admitted seeds turn
over. Summing them would make a corpus that rejects everything look busy.

Windowing follows ``DispersionIndex``: a fixed-length deque of per-tick
counts, so the reported rates are recent rather than campaign-lifetime
averages. Cumulative totals are kept alongside, because the lifetime figure
is the one that survives a resume usefully while a window of ticks does not
mean much across a restart.
"""

from __future__ import annotations

import collections

DEFAULT_WINDOW = 200
"""Ticks retained. Matches ``DispersionIndex``'s default, and a tick is
``_stats_effective_interval()`` -- on the order of 10 s of work -- so this is
roughly the last half hour of a campaign."""


class CorpusFlux:
    """Per-tick additions, evictions and rejections, with gross and net rates.

    Call :meth:`record_addition`, :meth:`record_eviction` and
    :meth:`record_rejection` as they happen; call :meth:`tick` once per stats
    interval to close the bucket. Counting into a pending bucket rather than
    straight into the window is deliberate: evictions arrive in batches from
    ``auto_minimize_corpus`` and ``deprioritize_near_duplicates`` while
    additions trickle in one at a time, so bucketing by tick is what makes
    the two rates comparable at all.
    """

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        self.window = max(int(window), 1)
        self._buckets: collections.deque[tuple[int, int, int]] = collections.deque(
            maxlen=self.window
        )
        self._pending_add = 0
        self._pending_evict = 0
        self._pending_reject = 0
        self.total_additions = 0
        self.total_evictions = 0
        self.total_rejections = 0

    # ── recording ────────────────────────────────────────────────────

    def record_addition(self, count: int = 1) -> None:
        self._pending_add += int(count)
        self.total_additions += int(count)

    def record_eviction(self, count: int = 1) -> None:
        self._pending_evict += int(count)
        self.total_evictions += int(count)

    def record_rejection(self, count: int = 1) -> None:
        self._pending_reject += int(count)
        self.total_rejections += int(count)

    def tick(self) -> None:
        """Close the current bucket and start a new one."""
        self._buckets.append((self._pending_add, self._pending_evict, self._pending_reject))
        self._pending_add = 0
        self._pending_evict = 0
        self._pending_reject = 0

    # ── reading ──────────────────────────────────────────────────────

    @property
    def ticks(self) -> int:
        return len(self._buckets)

    def windowed(self) -> tuple[int, int, int]:
        """Summed ``(additions, evictions, rejections)`` over the window."""
        adds = sum(b[0] for b in self._buckets)
        evicts = sum(b[1] for b in self._buckets)
        rejects = sum(b[2] for b in self._buckets)
        return adds, evicts, rejects

    def gross(self) -> int:
        """``additions + evictions`` over the window. The quantity net loses."""
        adds, evicts, _ = self.windowed()
        return adds + evicts

    def net(self) -> int:
        """``additions - evictions`` over the window: the change in size."""
        adds, evicts, _ = self.windowed()
        return adds - evicts

    def turnover(self) -> float | None:
        """Fraction of gross flux that cancels out, in ``[0, 1]``.

        ``1 - |net| / gross``. 1.0 is perfect balance -- every admission
        matched by an eviction -- and 0.0 is one-directional growth or decay.
        None when there has been no flux at all, which is the *quiet* case
        and must not be reported as balance: a corpus where nothing happens
        has ``net == 0`` and would otherwise score a misleading 1.0.
        """
        adds, evicts, _ = self.windowed()
        total = adds + evicts
        if total <= 0:
            return None
        return 1.0 - abs(adds - evicts) / total

    def z_score(self) -> float | None:
        """Standardized net drift: is it a trend, or noise?

        ``turnover`` alone can't say whether a nonzero net is a real
        directional pull or just the imbalance you'd expect from a handful
        of admissions and evictions landing unevenly. Model each event in
        the window as an i.i.d. +-1 step (addition or eviction); under the
        null hypothesis of undirected churn, mu=0 and sigma=1 per step, so
        by the CLT the standardized sum is::

            Z_n = sum(X_i - mu) / (sigma * sqrt(n)) = net / sqrt(gross)

        None when ``gross`` is 0 -- the same 0/0 case ``turnover`` guards,
        and for the same reason: reporting 0.0 would claim a confident
        "no drift" verdict where there is no evidence either way. The
        normal approximation is only good for a reasonably large ``gross``
        (rule of thumb: several dozen events); at small gross this is a
        rough estimate, which is exactly why :meth:`is_significant_drift`
        exists rather than eyeballing this number directly -- a single
        event always has |z|=1 no matter which direction it went.
        """
        gross = self.gross()
        if gross <= 0:
            return None
        return self.net() / (gross**0.5)

    def is_significant_drift(self, threshold: float = 1.96) -> bool | None:
        """Whether ``z_score`` clears a two-sided significance threshold.

        Default 1.96 is the ~95% two-sided normal critical value. None when
        there has been no flux at all (see :meth:`z_score`).
        """
        z = self.z_score()
        if z is None:
            return None
        return abs(z) >= threshold

    def rates(self) -> dict[str, float]:
        """Per-tick rates over the window. Empty dict before the first tick."""
        if not self._buckets:
            return {}
        n = float(len(self._buckets))
        adds, evicts, rejects = self.windowed()
        return {
            "additions_per_tick": adds / n,
            "evictions_per_tick": evicts / n,
            "rejections_per_tick": rejects / n,
            "gross_per_tick": (adds + evicts) / n,
            "net_per_tick": (adds - evicts) / n,
        }

    def summary(self) -> dict:
        adds, evicts, rejects = self.windowed()
        return {
            "ticks": len(self._buckets),
            "window_additions": adds,
            "window_evictions": evicts,
            "window_rejections": rejects,
            "gross": adds + evicts,
            "net": adds - evicts,
            "turnover": self.turnover(),
            "z_score": self.z_score(),
            "significant_drift": self.is_significant_drift(),
            "total_additions": self.total_additions,
            "total_evictions": self.total_evictions,
            "total_rejections": self.total_rejections,
            **self.rates(),
        }

    # ── persistence ──────────────────────────────────────────────────

    def save(self) -> dict:
        return {
            "window": self.window,
            "buckets": [list(b) for b in self._buckets],
            "total_additions": self.total_additions,
            "total_evictions": self.total_evictions,
            "total_rejections": self.total_rejections,
        }

    def load(self, data: dict) -> None:
        self.window = max(int(data.get("window", self.window)), 1)
        self._buckets = collections.deque(
            (tuple(int(v) for v in b[:3]) for b in data.get("buckets", [])),
            maxlen=self.window,
        )
        self.total_additions = int(data.get("total_additions", 0))
        self.total_evictions = int(data.get("total_evictions", 0))
        self.total_rejections = int(data.get("total_rejections", 0))
        self._pending_add = 0
        self._pending_evict = 0
        self._pending_reject = 0
