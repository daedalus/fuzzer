"""Joint (block) liveness probing for coverage-dead regions.

``LiveBitMaskEstimator`` (``core/live_bit_mask.py``) judges a region dead
when many consecutive mutations *inside that region alone* never moved
coverage. That is a single-region test. A region can pass it and still
matter in combination with another one: two guard fields that must both
leave their default before the parser takes a new branch each look inert
when mutated by themselves. Block sensitivity of a Boolean function can
exceed its single-bit sensitivity by a polynomial factor (family 132 of
openai/math, see ``docs/handover/handover_openai_math_survey_2026-10-06.md``),
so "dead under single-region probes" is genuinely weaker than "dead".

This module is the bookkeeping for the cheapest repair: occasionally mutate
one byte in *each* of two already-dead regions at once and watch whether
coverage moves. The pair is the unit of evidence:

- a probe that moves coverage is a *hit* for that pair;
- ``confirm_hits`` hits on one pair (not one: unstable edges would otherwise
  buy a false revocation) mark BOTH regions as jointly live, which makes the
  caller restore their full mutation weight (``_region_liveness_factor``);
- a pair that takes ``probes_per_pair`` probes without confirming is
  resolved and never drawn again, so the quadratic pair space is bounded by
  budget, not enumerated.

What this cannot find is an *arithmetic* relation between regions (a field
plus the checksum over it): two independent random byte changes essentially
never satisfy it. That is the checksum patcher's job, not this module's.

State is per seed content hash, the same key as ``OperatorEngine``'s region
cache, and is dropped with it. Like the per-region estimators it is not
persisted across ``--resume``.
"""

from __future__ import annotations

# Fraction of mutation rounds that attempt a joint probe, when the seed has
# at least two dead regions. Untuned: a probe round is a whole round spent
# on a mutant that, by hypothesis, is usually inert.
JOINT_PROBE_RATE = 1.0 / 64.0
# Probes one pair may consume before it is resolved as independent.
JOINT_PROBES_PER_PAIR = 32
# Coverage-moving probes on one pair needed to declare it jointly live.
JOINT_CONFIRM_HITS = 2
# Distinct pairs tracked per seed. Bounds memory and total probe spend on a
# seed with many dead regions (d dead regions have d*(d-1)/2 pairs).
JOINT_MAX_PAIRS_PER_SEED = 256
# Random pair draws per pick before giving up for this round.
_PICK_TRIES = 8


class _SeedPairs:
    """Pair ledger for one seed: ``pairs[(i, j)] = [trials, hits]``."""

    __slots__ = ("pairs", "revoked")

    def __init__(self) -> None:
        self.pairs: dict[tuple[int, int], list[int]] = {}
        self.revoked: set[int] = set()


class JointLiveness:
    """Pair-probe ledger deciding which dead regions are jointly live.

    Args:
        rng: A ``RandPool`` (Hard Rule 16); only ``random()`` and ``randint``
            are used.
        rate: Probability that ``want_probe()`` says yes. Must be in (0, 1].
        probes_per_pair: Probe budget per pair before it is resolved.
        confirm_hits: Hits required to revoke the pair's regions.
        max_pairs: Distinct pairs tracked per seed.
    """

    def __init__(
        self,
        rng,
        rate: float = JOINT_PROBE_RATE,
        probes_per_pair: int = JOINT_PROBES_PER_PAIR,
        confirm_hits: int = JOINT_CONFIRM_HITS,
        max_pairs: int = JOINT_MAX_PAIRS_PER_SEED,
    ) -> None:
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"rate must be in (0, 1], got {rate}")
        if probes_per_pair < 1:
            raise ValueError(f"probes_per_pair must be >= 1, got {probes_per_pair}")
        if not 1 <= confirm_hits <= probes_per_pair:
            raise ValueError(f"confirm_hits must be in [1, probes_per_pair], got {confirm_hits}")
        if max_pairs < 1:
            raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
        self._rng = rng
        self.rate = rate
        self.probes_per_pair = probes_per_pair
        self.confirm_hits = confirm_hits
        self.max_pairs = max_pairs
        self._seeds: dict[int, _SeedPairs] = {}
        # Lifetime counters, for the banner / reports.
        self.probes = 0
        self.hits = 0
        self.revoked_regions = 0

    def want_probe(self) -> bool:
        """True for the ``rate`` fraction of rounds that should try a probe."""
        return self._rng.random() < self.rate

    def pick_pair(self, key: int, dead: list[int]) -> tuple[int, int] | None:
        """Draw an unresolved pair of distinct dead regions, or None.

        Regions already revoked for *key* are not candidates. A new pair is
        refused once the seed is at ``max_pairs``; an existing pair is
        returned only while it has probe budget left.
        """
        state = self._seeds.get(key)
        revoked = state.revoked if state is not None else ()
        live = [r for r in dead if r not in revoked]
        n = len(live)
        if n < 2:
            return None
        pairs = state.pairs if state is not None else {}
        for _ in range(_PICK_TRIES):
            a = live[self._rng.randint(0, n - 1)]
            b = live[self._rng.randint(0, n - 1)]
            if a == b:
                continue
            pair = (a, b) if a < b else (b, a)
            entry = pairs.get(pair)
            if entry is None:
                if len(pairs) >= self.max_pairs:
                    continue
                return pair
            if entry[0] < self.probes_per_pair:
                return pair
        return None

    def record(self, key: int, pair: tuple[int, int], moved: bool) -> bool:
        """Fold one probe's outcome in; True iff it revoked a region.

        *moved* is whether the probe's coverage differed from the parent's.
        """
        state = self._seeds.get(key)
        if state is None:
            state = self._seeds[key] = _SeedPairs()
        entry = state.pairs.setdefault(pair, [0, 0])
        entry[0] += 1
        self.probes += 1
        if not moved:
            return False
        entry[1] += 1
        self.hits += 1
        if entry[1] < self.confirm_hits:
            return False
        fresh = [r for r in pair if r not in state.revoked]
        state.revoked.update(pair)
        self.revoked_regions += len(fresh)
        return bool(fresh)

    def is_revoked(self, key: int, region_idx: int) -> bool:
        """True iff *region_idx* of seed *key* was found jointly live."""
        state = self._seeds.get(key)
        return state is not None and region_idx in state.revoked

    def drop(self, key: int) -> None:
        """Forget *key* (called when its region layout is evicted)."""
        self._seeds.pop(key, None)

    def summary(self) -> str:
        """One line for the startup banner / end-of-run report."""
        return (
            f"probes={self.probes} hits={self.hits} "
            f"revoked_regions={self.revoked_regions} seeds={len(self._seeds)}"
        )
