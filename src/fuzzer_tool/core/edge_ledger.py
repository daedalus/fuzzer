"""EdgeLedger: confirmed edge ids folded into families (strata).

Design: ``docs/handover/handover_strata_schedulers_2026-09-19.md`` §3.2.
A family is ``id >> ctx_bits`` -- the base edge with its context tag
stripped. Families are immune to F1 (ASLR tag drift); tags are not.

    confirmed ids ─> observe(seed) ─┬─> owner(edge), family_owner(fam)
                                    ├─> frontier(): owner/n < PROLOGUE_FRAC
                                    └─> Novelty(edges, families, level)

Resolution per family: ``FAMILY`` when trust is ``UNSTABLE`` (ids move
between processes) or the family's tag occupancy reaches ``OCC_CAP``
(tag space saturated); else ``TAG``. At ``FAMILY`` a new tag in a known
family is not novel.

Ids are only masked, shifted and compared. No arithmetic on their order:
every family-level output is invariant under a tag bijection within a
family (pinned by test).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fuzzer_tool.core.scheduler_substrate import effective_edges

#: Midpoint of the measured 0.14..0.86 prologue gap (fuzzgoat, handover §1.2).
PROLOGUE_FRAC = 0.5
#: Tag occupancy past which a family is read at family resolution:
#: 0 clean families vs 6 under ASLR reached 0.95 (§1.2).
OCC_CAP = 0.95
#: Family shift for ctx-free builds; families still form at 8 (§1.2).
FAMILY_SHIFT_DEFAULT = 8


class Res(Enum):
    TAG = 1
    FAMILY = 2


class Trust(Enum):
    UNKNOWN = 0
    STABLE = 1
    UNSTABLE = 2


@dataclass(frozen=True)
class Novelty:
    edges: frozenset[int]  # ids never seen before
    families: frozenset[int]  # families novel at their own resolution
    level: Res  # TAG if any novel family is read at tag resolution


_NO_NOVELTY = Novelty(frozenset(), frozenset(), Res.TAG)


class EdgeLedger:
    """Seed x family incidence over confirmed edge ids.

    Args:
        ctx_bits: The target's ``__AFL_CTX_BITS``; 0/None -> ``FAMILY_SHIFT_DEFAULT``.
    """

    def __init__(self, ctx_bits: int | None):
        self.shift = ctx_bits or FAMILY_SHIFT_DEFAULT
        self.trust = Trust.UNKNOWN
        self._seen: set[int] = set()
        self._edge_owner: dict[int, int] = {}
        self._fam_tags: dict[int, set[int]] = {}
        # seed -> fam -> edges; fam -> seeds (dict for insertion order).
        self._seed_fams: dict[str, dict[int, set[int]]] = {}
        self._fam_seeds: dict[int, dict[str, None]] = {}
        # frontier() memo, invalidated by any ownership change.
        self._version = 0
        self._front: tuple[int, list[int]] = (-1, [])

    # ── Updates ─────────────────────────────────────────────────────────
    def observe(self, seed_key: str, confirmed: frozenset[int] | set[int]) -> Novelty:
        """Fold one seed's confirmed ids; return what was novel."""
        if not confirmed:
            return _NO_NOVELTY

        shift = self.shift
        mask = (1 << shift) - 1
        self._version += 1
        fams = self._seed_fams.setdefault(seed_key, {})
        new_edges = []
        new_fams = set()
        for e in confirmed:
            fam = e >> shift
            if fam not in self._fam_seeds:
                new_fams.add(fam)
                self._fam_seeds[fam] = {}
                self._fam_tags[fam] = set()
            if e not in self._seen:
                self._seen.add(e)
                new_edges.append(e)
            self._fam_tags[fam].add(e & mask)

            owned = fams.get(fam)
            if owned is None:
                owned = fams[fam] = set()
                self._fam_seeds[fam][seed_key] = None
            if e not in owned:
                owned.add(e)
                self._edge_owner[e] = self._edge_owner.get(e, 0) + 1

        # A known family counts only while it is read at tag resolution.
        novel = set(new_fams)
        for e in new_edges:
            fam = e >> shift
            if fam not in novel and self.res(fam) is Res.TAG:
                novel.add(fam)
        if not novel:
            return Novelty(frozenset(new_edges), frozenset(), Res.TAG)

        level = Res.TAG if any(self.res(f) is Res.TAG for f in novel) else Res.FAMILY
        return Novelty(frozenset(new_edges), frozenset(novel), level)

    def forget(self, seed_key: str) -> None:
        """Drop a seed's ownership (corpus eviction). Seen-ness is kept."""
        fams = self._seed_fams.pop(seed_key, None)
        if fams is None:
            return
        self._version += 1
        for fam, edges in fams.items():
            self._fam_seeds[fam].pop(seed_key, None)
            for e in edges:
                self._edge_owner[e] -= 1

    def set_trust(self, t: Trust) -> None:
        self.trust = t

    # ── Queries ─────────────────────────────────────────────────────────
    @property
    def n_seeds(self) -> int:
        return len(self._seed_fams)

    def owner(self, edge: int) -> int:
        return self._edge_owner.get(edge, 0)

    def family_owner(self, fam: int) -> int:
        return len(self._fam_seeds.get(fam, ()))

    def occupancy(self, fam: int) -> float:
        return len(self._fam_tags.get(fam, ())) / (1 << self.shift)

    def res(self, fam: int) -> Res:
        if self.trust is Trust.UNSTABLE or self.occupancy(fam) >= OCC_CAP:
            return Res.FAMILY
        return Res.TAG

    def frontier(self) -> list[int]:
        """Families owned by fewer than ``PROLOGUE_FRAC`` of seeds, sorted."""
        if self._front[0] == self._version:
            return self._front[1]

        n = self.n_seeds
        cut = PROLOGUE_FRAC * n
        front = sorted(f for f, s in self._fam_seeds.items() if 0 < len(s) < cut)
        self._front = (self._version, front)
        return front

    def seeds_in(self, fam: int) -> list[str]:
        return list(self._fam_seeds.get(fam, ()))

    def edges_in(self, seed_key: str, fam: int) -> list[int]:
        return list(self._seed_fams.get(seed_key, {}).get(fam, ()))

    def rarest_family(self, seed_key: str) -> int | None:
        """The seed's least-owned family (lowest id on ties); None if unknown."""
        fams = self._seed_fams.get(seed_key)
        if not fams:
            return None
        return min(fams, key=lambda f: (len(self._fam_seeds[f]), f))

    def eff_edges(self) -> float:
        """``2 ** H`` of the edge owner marginal. Logged only (F5)."""
        return effective_edges(self._edge_owner)

    # ── Persistence (state_store pickles the dict) ──────────────────────
    def to_dict(self) -> dict:
        return {
            "shift": self.shift,
            "trust": self.trust.value,
            "seen": sorted(self._seen),
            "seeds": {
                k: {f: sorted(es) for f, es in fams.items()} for k, fams in self._seed_fams.items()
            },
            "fam_tags": {f: sorted(t) for f, t in self._fam_tags.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> EdgeLedger:
        shift = data["shift"]
        if not isinstance(shift, int) or shift <= 0:
            raise ValueError(f"bad shift {shift!r}")
        led = cls(shift)
        led.trust = Trust(data["trust"])
        led._seen = set(data["seen"])
        led._fam_tags = {int(f): set(t) for f, t in data["fam_tags"].items()}
        for f in led._fam_tags:
            led._fam_seeds[f] = {}
        for key, fams in data["seeds"].items():
            led._seed_fams[key] = {}
            for f, es in fams.items():
                led._seed_fams[key][int(f)] = set(es)
                led._fam_seeds.setdefault(int(f), {})[key] = None
                for e in es:
                    led._edge_owner[e] = led._edge_owner.get(e, 0) + 1
        return led
