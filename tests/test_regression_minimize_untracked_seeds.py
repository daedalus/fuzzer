"""Minimization must not drop coverage held by seeds the tracker evicted.

``EdgeTracker.seed_edges`` is capped at ``max_tracked_seeds`` (1,000 by
default; 592 KiB per tracked seed on an ffmpeg-shaped target).  Past the cap
``_maybe_prune`` evicts seeds, and their edges stay in ``cumulative_edges``
but leave the only per-seed map the corpus minimizer reads.  A cover computed
over that partial map is a cover of less than the corpus: ``_select_budget``
then drops every seed holding an edge it cannot see, and the coverage goes
with them -- silently, because ``_recover_uncovered`` can only re-add seeds
whose edges it can still attribute.

The oracle here is the edge set the test recorded, never the tracker: what a
seed covers is a fact about the seed, not about how the tracker stores it.
"""

import tempfile
from pathlib import Path

from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.services.corpus_manager import CorpusManager, seed_key
from tests.test_corpus_minimization import MockFuzzer

# Ceiling low enough that a 300-seed corpus evicts most of them, matching the
# 1,000-seed default on a 20k-seed campaign.
CEILING = 20
SHARED_EDGES = 60
UNIQUE_SEEDS = 20
SHARED_SPAN = 4
TOTAL_SEEDS = 300
BUDGET = 50


def _build(max_tracked_seeds: int = CEILING, budget: int | None = BUDGET):
    """300 seeds over 80 edges; 20 hold an edge no other seed holds.

    Every seed records through ``record_edges`` so the tracker prunes exactly
    as it does on a campaign.  The 20 unique-edge seeds score lowest
    (``coverage_edges=1`` against 4), so a budget that trims by score drops
    them first -- which is only safe if their edges are still attributable.
    The shared pool needs ~15 four-edge seeds to cover, so the minimum cover
    (~35) fits the budget and the trim happens.
    """
    f = MockFuzzer(Path(tempfile.mkdtemp()))
    et = EdgeTracker(max_tracked_seeds=max_tracked_seeds)
    f._edge_tracker = et
    if budget is not None:
        f.max_corpus = budget

    truth: dict[str, set[int]] = {}
    seeds: list[bytes] = []
    for i in range(TOTAL_SEEDS):
        seed = f"seed_{i:04d}_".encode() + bytes([i % 256]) * 48
        if i < UNIQUE_SEEDS:
            edges = {i}
        else:
            base = UNIQUE_SEEDS + (i % SHARED_EDGES)
            edges = {(base + k) % SHARED_EDGES + UNIQUE_SEEDS for k in range(SHARED_SPAN)}
        seeds.append(seed)
        truth[seed_key(seed)] = edges
        et.record_edges(seed_key(seed), edges)

    f.corpus = seeds
    f.seed_meta = {
        s: {"fuzz_count": 1, "coverage_edges": truth[seed_key(s)].__len__(), "added_at": 100.0 + i}
        for i, s in enumerate(seeds)
    }
    return f, et, truth


def _covered(seeds, truth: dict[str, set[int]]) -> set[int]:
    """Edges the corpus covers, from the test's own record."""
    covered: set[int] = set()
    for seed in seeds:
        covered |= truth[seed_key(seed)]
    return covered


class TestEvictedSeedCoverage:
    def test_control_full_tracking_keeps_coverage(self):
        """Control: with nothing evicted the same scenario is lossless.

        A tracker that fits the whole corpus is the reference the capped one
        is compared against.  If this fails the oracle is wrong, not the fix.
        """
        f, et, truth = _build(max_tracked_seeds=TOTAL_SEEDS + 1)
        assert len(et.seed_edges) == TOTAL_SEEDS

        CorpusManager(f).auto_minimize_corpus()

        assert _covered(f.corpus, truth) == et.cumulative_edges
        assert len(f.corpus) == BUDGET

    def test_coverage_survives_eviction(self):
        """The regression: evicted seeds still answer for the edges they alone held."""
        f, et, truth = _build()
        assert len(et.seed_edges) < TOTAL_SEEDS, "precondition: the cap must have evicted"

        CorpusManager(f).auto_minimize_corpus()

        assert _covered(f.corpus, truth) == et.cumulative_edges
        assert len(f.corpus) <= BUDGET

    def test_ledger_does_not_outlive_the_corpus(self):
        """Adversarial: entries for seeds that left the corpus must be dropped.

        The ledger only answers for seeds a cover could still pick; keeping
        the rest grows it without bound over a long campaign.
        """
        f, et, _truth = _build()

        CorpusManager(f).auto_minimize_corpus()

        live = {seed_key(s) for s in f.corpus}
        assert set(et.evicted_sole) <= live

    def test_retracked_seed_drops_its_stale_entry(self):
        """Adversarial: a seed that runs again is tracked, so its ledger copy is stale."""
        f, et, truth = _build()
        evicted = next(iter(et.evicted_sole))

        et.record_edges(evicted, truth[evicted])

        assert evicted in et.seed_edges
        assert evicted not in et.evicted_sole

    def test_ledger_answers_for_every_edge_the_tracker_knows(self):
        """Falsification: eviction may not make an edge unattributable.

        This is the invariant the minimizer depends on: every edge in
        ``cumulative_edges`` has at least one seed that still answers for it.
        """
        _f, et, _truth = _build()

        attributed: set[int] = set()
        for key in {*et.seed_edges, *et.evicted_sole}:
            attributed |= et.attributed_edges(key)

        assert attributed == et.cumulative_edges

    def test_ledger_survives_a_state_round_trip(self):
        """Falsification: a resumed campaign must keep the attribution it had.

        ``--resume`` restores the tracker from ``state.json``; losing the
        ledger there would make the first post-resume minimize drop the edges
        the previous session had already evicted.
        """
        _f, et, _truth = _build()

        restored = EdgeTracker()
        restored.from_dict(et.to_dict())

        assert set(restored.evicted_sole) == set(et.evicted_sole)
        for key in et.evicted_sole:
            assert restored.attributed_edges(key) == et.attributed_edges(key)
        assert restored.cumulative_edges == et.cumulative_edges
