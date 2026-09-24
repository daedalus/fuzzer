"""Entropy-gradient seed strategy: chase seeds behind recent entropy growth.

Plan: ``docs/handover/handover_entropy_seed_schedulers_2026-09-19.md`` §4.

Distinct from ``seed_entropy_deviation.py``/``seed_entropy_kl.py``/
``seed_entropy_zscore.py`` (§1-3), which all score a seed by a *static*
property of its own bytes: this scores a seed by its *track record* -- how
much the pooled corpus byte-entropy has moved, in bits, immediately after
that seed was picked and mutated into an accepted child. A seed whose
recent children keep contributing new byte-pattern diversity is picked
more; one whose children haven't moved the pooled distribution decays, the
byte-content analogue of what ``analyzer_critical_slowing.py`` already does
for discovery-rate variance.

Credit assignment, done the way the handover doc corrects itself into
doing it
-------------------------------------------------------------------------
The naive version of this idea reads the pooled corpus entropy on a timer
and credits whichever seed was picked in that window -- exactly the
shared-effect failure mode ``handover_non_ucb_schedulers_2026-09-13.md``
§0 already documents in this codebase for Elo match credit fanned out to
arms that weren't played: between one pick of seed S and the next tick,
other seeds get picked and mutated too, and the change in pooled entropy
is a joint effect of everything admitted in that window, not attributable
to S alone.

This instead credits a seed only for the entropy its own *direct
children* contribute, at the moment each child is admitted to the corpus
(``Fuzzer.save_to_corpus`` call sites, via :meth:`record_child`) rather
than on a shared timer -- there is no cross-seed attribution question left
to answer because each admission names its own parent. The marginal is
measured against ``CumulativeByteEntropy``'s own running totals: fold the
child in, diff ``bits()`` before/after, an O(1) operation against a
256-bin table rather than an O(corpus) rescan. No scratch copy is needed
for this (the diff is taken around the one fold this pool wants to make
anyway, so folding *is* discarding the copy).

The pool this measures against is owned here and kept in sync with the
live corpus incrementally -- appended-to on each call rather than
diffed with a fresh ``set(corpus)`` every time, since this hook fires on
every admission (not just when this arm is picked, unlike
``seed_entropy_kl.py``'s pool, which only has to be fresh when its own
``select()`` runs). A full corpus shrink (auto-minimize pruning seeds) is
rebuilt from scratch, but that happens once per prune, not once per
admission.

EWMA mechanics reuse ``DiscountedUCBBase``'s lazy-discount trick
(``core/schedulers/ucb_common.py``) rather than decaying every stored
credit on every admission: a single global ``_discount`` scales every
credit on read, and an update divides by the current discount before
adding -- O(1) per admission regardless of how many seeds have ever been
credited, instead of the O(seeds credited) an eager per-seed decay would
cost on every single admission.

Open limitation, stated rather than hidden: unlike §1-3's warm-up gates
(which retire once the corpus mean/variance is trustworthy), there is no
way to detect "no entropy-productive seed exists yet" other than the
credited-count floor below -- a target whose accepted children never move
the pooled distribution (every child looks like more of the same) will
sit at ``MIN_OBSERVATIONS`` credited admissions with every credit at 0.0
and never warm past a coin flip among the observed seeds. That is a
correct read of the target, not a bug in the gate.
"""

from __future__ import annotations

from typing import Any

from fuzzer_tool.core.byte_entropy import ENTROPY_SAMPLE_CAP, CumulativeByteEntropy

#: Credited admissions (a resolvable parent + a real marginal reading)
#: before scores are trusted enough to select on. Same gate value as
#: EntropyDeviationSeedStrategy/EntropyZScoreSeedStrategy.MIN_OBSERVATIONS.
MIN_OBSERVATIONS = 20

#: Floor added to every weight so a seed with no credited children yet (or
#: whose children never moved the pool) is rare rather than unreachable.
#: Matches seed_entropy_kl/seed_kruskal_count.
MIN_WEIGHT = 1e-6

#: Per-admission multiplicative decay applied to every seed's stored
#: credit. Not yet A/B validated (same disclaimer as entropy_kl/
#: entropy_deviation/entropy_zscore's defaults) -- 0.98 means a seed's
#: credit from ~35 admissions ago has decayed to about a third of its
#: original weight, which is the same order of "recent" this project's
#: other rotting/discounted schedulers (FEWA, D-UCB) target by default.
DEFAULT_DECAY = 0.98

#: Below this, the global discount is folded back into the stored credits
#: and reset to 1.0 -- same renormalisation floor DiscountedUCBBase uses,
#: for the same reason: letting `_discount` keep shrinking underflows it
#: to 0.0 long before a real campaign's admission count would otherwise
#: require a rescale.
_RENORMALISE_FLOOR = 1e-12


class EntropyGradientSeedStrategy:
    """Elo-arbitrated ``entropy_gradient`` seed arm (``--entropy-gradient``).

    Unlike its §1-3 siblings, this strategy has to observe *every* corpus
    admission to assign credit correctly, not just the ones that happen
    while it is the arm Elo elected to pick with -- :meth:`record_child`
    is called unconditionally from the fuzzer's admission sites, off-policy,
    the same way ``FEWAScheduler.record()`` observes every operator outcome
    regardless of which scheduler picked the operator.
    """

    def __init__(
        self,
        rng: Any,
        cap: int = ENTROPY_SAMPLE_CAP,
        decay: float = DEFAULT_DECAY,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay!r}")
        self._rng = rng
        self._cap = cap
        self._decay = decay
        self._min_observations = min_observations

        # Live-corpus pool, synced incrementally in _sync().
        self._pool = CumulativeByteEntropy()
        self._live: set[bytes] = set()
        self._synced_len = 0

        # Lazy-discounted per-seed credit (DiscountedUCBBase's trick):
        # stored value is credit / _discount; read scales by _discount.
        self._credit_rel: dict[bytes, float] = {}
        self._discount = 1.0

        self._credited = 0
        self._selected = 0

    @property
    def warmed(self) -> bool:
        """True once enough children have been credited to trust a pick."""
        return self._credited >= self._min_observations

    # ── credit assignment ────────────────────────────────────────────

    def record_child(self, parent: bytes | None, child: bytes, corpus: list[bytes]) -> None:
        """Credit *parent* for the pooled-entropy delta its new *child* adds.

        Called once per corpus admission, right after *child* joined
        *corpus* (mirrors ``Fuzzer._record_lineage_insert``'s call
        contract). A no-op credit-wise when *parent* is ``None`` (a root
        seed, e.g. from the initial corpus load) -- the pool is still kept
        in sync so later admissions measure their marginal against a
        complete baseline.
        """
        for seed in self._sync(corpus):
            if seed == child and parent is not None:
                before = self._pool.bits()
                self._pool.add(seed, self._cap)
                self._live.add(seed)
                delta = self._pool.bits() - before
                self._credit_update(parent, delta)
                self._credited += 1
            else:
                self._pool.add(seed, self._cap)
                self._live.add(seed)

    def _sync(self, corpus: list[bytes]) -> list[bytes]:
        """Seeds in *corpus* the pool hasn't folded in yet.

        Diffs against the tail grown since the last call rather than a
        fresh ``set(corpus)`` -- cheap because this runs on every
        admission, not just when this arm is picked. A shrink (corpus
        pruned since the last call) forces a full rebuild, which costs
        O(corpus) but happens once per prune rather than once per
        admission.
        """
        n = len(corpus)
        if n < self._synced_len:
            self._pool = CumulativeByteEntropy()
            self._live = set()
            self._synced_len = 0

        pending = [s for s in corpus[self._synced_len :] if s not in self._live]
        self._synced_len = n
        return pending

    def _credit_update(self, parent: bytes, delta: float) -> None:
        """Decay every stored credit by one round, then add *delta* to *parent*.

        Lazy global-discount trick (``DiscountedUCBBase._arm_update``):
        decaying every entry on every admission would be O(seeds credited)
        per admission; scaling on read instead makes this O(1).
        """
        if self._decay < 1.0:
            self._discount *= self._decay
            if self._discount < _RENORMALISE_FLOOR:
                self._renormalise()
        if delta:
            inv = 1.0 / self._discount
            self._credit_rel[parent] = self._credit_rel.get(parent, 0.0) + delta * inv

    def _renormalise(self) -> None:
        d = self._discount
        for key in self._credit_rel:
            self._credit_rel[key] *= d
        self._discount = 1.0

    def _credit(self, seed: bytes) -> float:
        """Current (discounted) credit for *seed*, 0.0 if never credited."""
        rel = self._credit_rel.get(seed)
        return 0.0 if rel is None else rel * self._discount

    # ── picker interface ─────────────────────────────────────────────

    def scores(self, seeds: list[bytes]) -> list[float]:
        """Current credit per seed, aligned 1:1 with ``seeds``."""
        return [self._credit(s) for s in seeds]

    def select(self, seeds: list[bytes]) -> bytes | None:
        """Draw proportional to ``max(credit, 0) + MIN_WEIGHT``; None while cold.

        Declining while ``not warmed`` (rather than picking uniformly
        among all-zero credits) matches every sibling entropy arm's
        convention: a scheduler with nothing to go on yet hands back to
        the picker instead of pretending its score means something.
        """
        if not seeds or not self.warmed:
            return None

        # Credit is signed (a flat child lowers pooled entropy); a negative
        # weight breaks weighted_choice, so floor it -- "never helped".
        weights = [max(self._credit(s), 0.0) + MIN_WEIGHT for s in seeds]
        self._selected += 1
        chosen: bytes = self._rng.weighted_choice(seeds, weights)
        return chosen

    def stats(self) -> dict[str, Any]:
        live = [self._credit(s) for s in self._live] if self._live else []
        return {
            "credited": self._credited,
            "selected": self._selected,
            "pooled": len(self._live),
            "warmed": self.warmed,
            "mean_credit": (sum(live) / len(live)) if live else 0.0,
        }
