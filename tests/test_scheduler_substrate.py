"""The substrate a scheduler reads before it scores anything.

Three things, each of them a consequence of
``docs/handover/handover_edge_id_axis_2026-09-18.md``: whether per-edge
statistics can be trusted at all (F1, F11), how many edges the execution
volume is actually spread over (F5), and which edges no input has yet told
apart (F10).
"""

import pytest

from fuzzer_tool.core import scheduler_substrate
from fuzzer_tool.core.edge_tracker import EdgeTracker
from fuzzer_tool.core.scheduler_substrate import (
    EdgeCanonicalizer,
    coverage_trust,
    effective_edges,
)


class TestEffectiveEdges:
    def test_uniform_hits_give_back_the_edge_count(self):
        assert effective_edges({i: 10 for i in range(8)}) == pytest.approx(8.0)

    def test_one_edge_taking_everything_gives_one(self):
        assert effective_edges({1: 500, 2: 0, 3: 0}) == pytest.approx(1.0)

    def test_concentration_is_what_moves_it_not_edge_count(self):
        """Same 100 edges, all the volume on one: effectively a single edge."""
        spread = {i: 1 for i in range(100)}
        concentrated = {i: (1000 if i == 0 else 1) for i in range(100)}
        assert effective_edges(spread) == pytest.approx(100.0)
        assert effective_edges(concentrated) < 10.0

    def test_accepts_a_bare_iterable(self):
        assert effective_edges([4, 4, 4, 4]) == pytest.approx(4.0)

    def test_empty_and_all_zero_are_zero(self):
        assert effective_edges({}) == 0.0
        assert effective_edges({1: 0, 2: 0}) == 0.0

    def test_tracker_exposes_it(self):
        tracker = EdgeTracker()
        tracker._global_edge_hits.update({1: 5, 2: 5, 3: 5, 4: 5})
        assert tracker.effective_edges() == pytest.approx(4.0)

    def test_tracker_with_nothing_recorded(self):
        assert EdgeTracker().effective_edges() == 0.0


class TestCoverageTrust:
    def test_no_target_is_trusted(self):
        assert coverage_trust(None) == (True, None)

    def test_coverage_off_short_circuits(self):
        """The premise does not hold, so there is nothing to warn about."""
        assert coverage_trust("/nonexistent", use_coverage=False) == (True, None)

    def test_ptrace_needs_no_build_flags(self):
        assert coverage_trust("/nonexistent", ptrace=True) == (True, None)

    def test_unstable_ids_are_not_trusted(self):
        """Per-process ids make every edge a singleton owned by one seed."""
        trusted, reason = coverage_trust("/bin/sh", id_stability=0.007)
        assert not trusted
        assert "reproducible" in reason

    def test_perfect_stability_passes_that_check(self):
        trusted, _ = coverage_trust("/bin/sh", id_stability=1.0)
        assert trusted is True  # /bin/sh is stripped -> "unknown", not "absent"

    def test_stability_is_checked_before_the_binary(self):
        """A moving id space invalidates the statistics whatever the build."""
        trusted, reason = coverage_trust("/nonexistent/binary", id_stability=0.5)
        assert not trusted
        assert "reproducible" in reason


class TestEdgeCanonicalizer:
    def test_identical_profiles_collapse(self):
        seeds = {
            "a": {1: 3, 2: 3, 3: 1},
            "b": {1: 7, 2: 7, 3: 2},
        }
        canon = EdgeCanonicalizer()
        canon.refit(seeds)
        assert canon.class_of(1) == canon.class_of(2)
        assert canon.class_of(3) != canon.class_of(1)
        assert canon.multiplicity(1) == 2
        assert canon.multiplicity(3) == 1
        stats = canon.stats()
        assert (stats["edges"], stats["classes"], stats["duplicate_edges"]) == (3, 2, 1)
        assert stats["largest_class"] == 2

    def test_counts_matter_not_just_presence(self):
        """Two edges hit by the same seeds at different rates are distinct."""
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 1, 2: 2}})
        assert canon.class_of(1) != canon.class_of(2)

    def test_a_split_is_picked_up_on_refit(self):
        """Classes must not be sticky: one input can tell members apart."""
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 4, 2: 4}})
        assert canon.multiplicity(1) == 2
        canon.refit({"a": {1: 4, 2: 4}, "b": {1: 1}})
        assert canon.multiplicity(1) == 1
        assert canon.stats()["duplicate_edges"] == 0

    def test_unseen_edges_are_their_own_class(self):
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 1}})
        assert canon.class_of(99) == 99
        assert canon.multiplicity(99) == 1

    def test_empty_corpus(self):
        canon = EdgeCanonicalizer()
        canon.refit({})
        assert canon.stats() == {
            "edges": 0,
            "classes": 0,
            "duplicate_edges": 0,
            "duplicate_fraction": 0.0,
            "largest_class": 0,
        }

    def test_an_edge_missing_from_a_seed_reads_as_zero(self):
        """Absence is part of the profile, so it separates classes."""
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 2, 2: 2}, "b": {1: 2}})
        assert canon.class_of(1) != canon.class_of(2)


def _tuple_classes(seeds):
    """The pre-fingerprint refit, kept as the oracle: group edges by dense profile."""
    keys = list(seeds)
    profiles = {}
    for i, k in enumerate(keys):
        for e, c in seeds[k].items():
            profiles.setdefault(e, [0] * len(keys))[i] = int(c)
    groups = {}
    for e, prof in profiles.items():
        groups.setdefault(tuple(prof), []).append(e)
    return {e: min(m) for m in groups.values() for e in m}


def _planted_corpus(n_seeds=60, n_edges=400):
    """Deterministic corpus with duplicate chains, near-miss chains and large counts.

    Edge e belongs to family e // 4, so each family of four is a chain with an
    identical profile -- except that edge 4k+3 differs from its family in every
    seventh seed (a near miss the hash must not merge).
    """
    seeds = {}
    for s in range(n_seeds):
        hc = {}
        for e in range(n_edges):
            fam = e // 4
            if (fam * 7 + s * 13) % 5 == 0:
                continue  # absent in this seed
            c = 1 + (fam * 31 + s * 17) % 0xFFFFFF
            if e % 4 == 3 and s % 7 == 0:
                c += 1
            hc[e] = c
        seeds[f"s{s}"] = hc
    return seeds


class TestFingerprintRefit:
    """refit groups by a linear fingerprint of the profile, not the dense profile.

    h(e) = sum_s count(s, e) * r_s mod 2^64 is linear in the column, so equal
    columns hash equal and unequal ones collide only if their difference is
    orthogonal to r (Schwartz-Zippel). Two independent weight families make a
    collision need both at once.
    """

    def test_oracle_agrees_with_itself_first(self):
        seeds = _planted_corpus()
        shuffled = dict(reversed(list(seeds.items())))
        assert _tuple_classes(seeds) == _tuple_classes(shuffled)  # Hard Rule 46

    def test_matches_the_dense_profile_grouping(self):
        seeds = _planted_corpus()
        canon = EdgeCanonicalizer()
        canon.refit(seeds)
        want = _tuple_classes(seeds)
        assert {e: canon.class_of(e) for e in want} == want
        assert canon.stats()["classes"] == len(set(want.values()))
        assert canon.stats()["classes"] > 100  # near misses stayed split

    def test_seed_order_does_not_change_classes(self):
        seeds = _planted_corpus()
        a, b = EdgeCanonicalizer(), EdgeCanonicalizer()
        a.refit(seeds)
        b.refit(dict(reversed(list(seeds.items()))))
        assert all(a.class_of(e) == b.class_of(e) for e in range(400))

    def test_explicit_zero_equals_absence(self):
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 0, 2: 5}, "b": {2: 5, 3: 0}})
        assert canon.class_of(1) == canon.class_of(3)
        assert canon.class_of(2) != canon.class_of(1)

    def test_planted_collision_in_one_family_is_split_by_the_other(self, monkeypatch):
        # With unit weights in family 0, h0 is the column sum: (1, 2) and (2, 1) collide.
        real = scheduler_substrate._seed_weights
        monkeypatch.setattr(
            scheduler_substrate,
            "_seed_weights",
            lambda n, family: scheduler_substrate.np.ones(n, dtype=scheduler_substrate.np.uint64)
            if family == 0
            else real(n, family),
        )
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 1, 2: 2}, "b": {1: 2, 2: 1}})
        assert canon.class_of(1) != canon.class_of(2)

    def test_collision_in_both_families_merges(self, monkeypatch):
        # Adversarial: the guarantee is probabilistic. If both families collide,
        # nothing else checks -- this pins that the second family is the only guard.
        monkeypatch.setattr(
            scheduler_substrate,
            "_seed_weights",
            lambda n, family: scheduler_substrate.np.ones(n, dtype=scheduler_substrate.np.uint64),
        )
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: 1, 2: 2}, "b": {1: 2, 2: 1}})
        assert canon.class_of(1) == canon.class_of(2)

    def test_weights_are_odd_distinct_and_deterministic(self):
        w0 = scheduler_substrate._seed_weights(1000, 0)
        w1 = scheduler_substrate._seed_weights(1000, 1)
        assert (w0 & 1).all() and (w1 & 1).all()  # odd: invertible mod 2^64
        assert len(set(w0.tolist())) == 1000
        assert not (w0 == w1).any()
        assert (w0 == scheduler_substrate._seed_weights(1000, 0)).all()

    def test_class_array_is_class_of_vectorised(self):
        seeds = _planted_corpus()
        canon = EdgeCanonicalizer()
        canon.refit(seeds)
        probe = scheduler_substrate.np.array([0, 3, 7, 399, 5000, -1], dtype="int64")
        got = canon.class_array(probe).tolist()
        assert got == [canon.class_of(int(e)) for e in probe]
        assert got[-2:] == [5000, -1]  # unseen edges are their own class
        empty = EdgeCanonicalizer()
        assert empty.class_array(probe).tolist() == probe.tolist()

    def test_maximal_counts_do_not_overflow_into_a_merge(self):
        top = 0xFFFFFF  # the SHM count field's width
        canon = EdgeCanonicalizer()
        canon.refit({"a": {1: top, 2: top - 1}, "b": {1: top, 2: top}})
        assert canon.class_of(1) != canon.class_of(2)
