"""StrataSeedScheduler: Thompson over frontier families, rarity within (§3.3).

Design: ``docs/handover/handover_strata_schedulers_2026-09-19.md``.

    1. phi  = argmax Beta(a_f, b_f) draw over ledger.frontier()
    2. seed ~ w(s) = mean over s's edges e in phi of log1p(N / owner(e))
              (uniform when phi is read at family resolution)
    3. hit  : the pick's round confirmed a new edge in phi -> a_phi += 1
       miss : finalised at the next select                  -> b_phi += 1

Only rounds mutating the picked seed are credited: another arm's pick
finding an edge in phi says nothing about this arm.

Mean, not sum: IDF sums are size proxies (rho 0.91 with live edges, §1.3).
Example: N=5, seed owns e1 (owner 2) and e2 (owner 1) in phi ->
w = (log1p(2.5) + log1p(5)) / 2.

Abstains (``None``) when the sancov guard is not "present" (F11), the
frontier is empty, or phi has no live seed. Off by default; Elo arm plus a
non-Elo fallback, like ``kruskal_count``. Not A/B validated.
"""

from __future__ import annotations

import math
from collections.abc import Container
from enum import Enum

from fuzzer_tool.core.edge_ledger import EdgeLedger, Res
from fuzzer_tool.core.rand_pool import RandPool

_PRIOR = (1.0, 1.0)


class Guard(Enum):
    """``elf.sancov_guard_status`` tri-state."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class StrataSeedScheduler:
    """Seed arm over :class:`EdgeLedger` families.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16).
        ledger: The fuzzer's ledger (shared with ``op_strata``).
        guard: Build instrumentation verdict; anything but PRESENT abstains.
    """

    def __init__(self, rng: RandPool | None, ledger: EdgeLedger, guard: Guard):
        if rng is None:
            raise ValueError("StrataSeedScheduler requires a RandPool (Hard Rule 16)")
        self._rng = rng
        self.ledger = ledger
        self._guard = guard
        self._post: dict[int, list[float]] = {}
        self.last_phi: int | None = None
        self.last_key: str | None = None
        self._picks = 0
        self._hits = 0

    def available(self) -> bool:
        return self._guard is Guard.PRESENT and bool(self.ledger.frontier())

    def posterior(self, fam: int) -> tuple[float, float]:
        a, b = self._post.get(fam, _PRIOR)
        return (a, b)

    def select_key(self, live: Container[str]) -> str | None:
        """Pick a seed key; None to abstain."""
        self._finalise_miss()
        front = self.ledger.frontier()
        if not front:
            return None

        # 1. Thompson over frontier families (one vectorised draw).
        post = [self._post.get(f, _PRIOR) for f in front]
        draws = self._rng.betavariate_array([p[0] for p in post], [p[1] for p in post])
        phi = front[int(draws.argmax())]

        # 2. Rarity-weighted seed within phi.
        seeds = [k for k in self.ledger.seeds_in(phi) if k in live]
        if not seeds:
            return None
        pick = self._rng.weighted_choice(seeds, self._weights(phi, seeds))
        self.last_phi = phi
        self.last_key = pick
        self._picks += 1
        return pick

    def _weights(self, phi: int, seeds: list[str]) -> list[float]:
        led = self.ledger
        if led.res(phi) is Res.FAMILY:
            return [1.0] * len(seeds)

        n = led.n_seeds
        out = []
        for k in seeds:
            edges = led.edges_in(k, phi)
            out.append(sum(math.log1p(n / led.owner(e)) for e in edges) / len(edges))
        return out

    def credit(self, families: frozenset[int], seed_key: str) -> None:
        """Round outcome for *seed_key*: families with a confirmed new edge."""
        phi = self.last_phi
        if phi is None or seed_key != self.last_key or phi not in families:
            return
        self._post.setdefault(phi, list(_PRIOR))[0] += 1.0
        self._hits += 1
        self.last_phi = None

    def _finalise_miss(self) -> None:
        phi = self.last_phi
        if phi is None:
            return
        self._post.setdefault(phi, list(_PRIOR))[1] += 1.0
        self.last_phi = None

    def stats(self) -> dict:
        return {
            "picks": self._picks,
            "hits": self._hits,
            "frontier": len(self.ledger.frontier()),
            "families": len(self._post),
            "eff_edges": self.ledger.eff_edges(),
        }

    # ── Persistence ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "post": {f: list(p) for f, p in self._post.items()},
            "picks": self._picks,
            "hits": self._hits,
        }

    @classmethod
    def from_dict(
        cls, data: dict, rng: RandPool, ledger: EdgeLedger, guard: Guard
    ) -> StrataSeedScheduler:
        s = cls(rng, ledger, guard)
        s._post = {int(f): [float(p[0]), float(p[1])] for f, p in data["post"].items()}
        s._picks = int(data["picks"])
        s._hits = int(data["hits"])
        return s
