"""Tests for core/edge_matrix.py: rank helpers, the fold, the gate, class credit.

Oracles are hand-derived on matrices small enough to read (Hard Rule 39/46).
"""

import numpy as np
import pytest

from fuzzer_tool.core.edge_matrix import (
    MatrixSubstrate,
    average_ranks,
    build_fold,
    partial_rank_corr,
    profiles_from_tracker,
    rank_corr,
    residualize_ranks,
)
from fuzzer_tool.core.scheduler_substrate import EdgeCanonicalizer


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    """coverage_trust returns early for a target-less run, so the gate tests need one."""
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")


def _p(*rows):
    return {f"s{i}": dict(r) for i, r in enumerate(rows)}


class _Tracker:
    def __init__(self, profiles):
        self.seed_hit_counts = profiles
        self.seed_edges = {k: set(v) for k, v in profiles.items()}
        self.cumulative_edges = set().union(*self.seed_edges.values())

    def grow(self, key, hc):
        self.seed_hit_counts[key] = hc
        self.seed_edges[key] = set(hc)
        self.cumulative_edges |= set(hc)


class TestRankHelpers:
    def test_average_ranks_ties(self):
        assert average_ranks(np.array([5.0, 1.0, 5.0, 3.0])).tolist() == [2.5, 0.0, 2.5, 1.0]

    def test_rank_corr_control_against_itself(self):
        # Hard Rule 46: the reference against itself before anything else.
        x = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6])
        assert rank_corr(x, x) == pytest.approx(1.0)
        assert rank_corr(x, -x) == pytest.approx(-1.0)

    def test_rank_corr_hand_value_with_ties(self):
        # ranks x = [0,1,2,3], y = [0.5,0.5,2,3]: cov 4.5, var 5 and 4.5 -> 4.5/sqrt(22.5)
        got = rank_corr(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 5.0, 6.0]))
        assert got == pytest.approx(0.9486832980505138)

    def test_rank_corr_undefined_is_zero(self):
        assert rank_corr(np.ones(5), np.arange(5.0)) == 0.0
        assert rank_corr(np.arange(2.0), np.arange(2.0)) == 0.0

    def test_partial_equals_plain_when_the_control_is_uncorrelated(self):
        # z = [2,4,1,3] has rank correlation exactly 0 with x = y = [1,2,3,4].
        x = np.array([1.0, 2.0, 3.0, 4.0])
        z = np.array([2.0, 4.0, 1.0, 3.0])
        assert rank_corr(x, z) == 0.0
        assert partial_rank_corr(x, x, z) == pytest.approx(1.0)

    def test_partial_is_zero_when_the_control_explains_everything(self):
        # The Tang shape: the score is volume, so controlling for volume leaves nothing.
        z = np.arange(6.0)
        assert partial_rank_corr(z * 2, z + 1, z) == 0.0

    def test_residual_hand_value(self):
        # y ranks [1,0,3,2] on x ranks [0,1,2,3]: cov 3, var 5, slope 0.6.
        got = residualize_ranks(np.array([2.0, 1.0, 4.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0]))
        assert got == pytest.approx([0.4, -1.2, 1.2, -0.4])

    def test_residual_of_a_monotone_function_of_the_control_is_zero(self):
        x = np.array([3.0, 1.0, 2.0, 9.0])
        assert residualize_ranks(np.exp(x), x) == pytest.approx(np.zeros(4))

    def test_residual_with_constant_control_is_centred_ranks(self):
        got = residualize_ranks(np.array([1.0, 2.0, 3.0]), np.ones(3))
        assert got == pytest.approx([-1.0, 0.0, 1.0])


class TestFold:
    def test_duplicate_columns_fold_and_owners_are_incidence(self):
        # Edges 10-12 are one chain: identical columns. Edge 1000 has volume 1000 in one
        # seed, edge 20 has volume 1 in three: owners follow incidence, not volume (F6).
        profiles = _p(
            {10: 1, 11: 1, 12: 1, 1000: 1000}, {10: 1, 11: 1, 12: 1, 20: 1}, {20: 1}, {20: 1}
        )
        fold, _ = build_fold(profiles, EdgeCanonicalizer())
        assert fold.n_edges == 5 and fold.n_classes == 3
        owners = sorted(fold.class_owners.values())
        assert owners == [1, 2, 3]  # {1000}:1, chain:2, {20}:3

    def test_mass_counts_a_chain_once(self):
        # s0 covers the 3-edge chain (owners 2) and nothing else; mass = 1/2, not 3/2.
        profiles = _p({10: 1, 11: 1, 12: 1}, {10: 1, 11: 1, 12: 1, 9: 1}, {8: 1})
        fold, _ = build_fold(profiles, EdgeCanonicalizer())
        assert fold.mass[0] == pytest.approx(0.5)
        assert fold.mass[2] == pytest.approx(1.0)
        assert fold.degree[0] == 1 and fold.total[0] == 3

    def test_derived_edges_carry_no_mass(self):
        fold, _ = build_fold(_p({1: 1, 2: 5}, {1: 1}, {3: 1}), EdgeCanonicalizer(), derived={2})
        assert fold.degree[0] == 1

    def test_skips_report_a_reason(self):
        assert build_fold(_p({1: 1}, {1: 1}), EdgeCanonicalizer())[0] is None
        fold, why = build_fold(_p({1: 1}, {2: 1}, {3: 1}), EdgeCanonicalizer(), cell_budget=2)
        assert fold is None and "budget" in why

    def test_budget_counts_nonzeros_not_the_dense_product(self):
        # 300 seeds x 30000 edges = 9e6 cells, far past the old 2e6 cell bound,
        # but one edge each besides a shared prologue: 600 nonzeros.
        profiles = _p(*({0: 1, 1 + 100 * s: 1} for s in range(300)))
        profiles["pad"] = {29999: 1}
        fold, why = build_fold(profiles, EdgeCanonicalizer(), cell_budget=1000)
        assert fold is not None, why
        assert fold.n_edges == 302
        fold, why = build_fold(profiles, EdgeCanonicalizer(), cell_budget=600)
        assert fold is None and "601 nonzeros" in why

    def test_matches_the_per_cell_formulation(self):
        """The vectorised fold equals the loop it replaced, derived edges included."""
        profiles = {
            f"s{s}": {e: 1 + (e // 3 * 7 + s) % 5 for e in range(60) if (e // 3 + s) % 4}
            for s in range(40)
        }
        derived = {7, 8, 30}
        canon = EdgeCanonicalizer()
        fold, _ = build_fold(profiles, canon, derived=derived)

        # Reference: the per-cell Python loop, classes from the same refit.
        per_seed = [
            sorted({canon.class_of(e) for e in hc if e not in derived}) for hc in profiles.values()
        ]
        owners: dict[int, int] = {}
        for classes in per_seed:
            for c in classes:
                owners[c] = owners.get(c, 0) + 1
        assert [c.tolist() for c in fold.seed_classes] == per_seed
        assert fold.class_owners == owners
        assert fold.mass.tolist() == pytest.approx(
            [sum(1.0 / owners[c] for c in cls) for cls in per_seed]
        )
        assert fold.total.tolist() == [float(sum(hc.values())) for hc in profiles.values()]
        assert fold.degree.tolist() == [float(len(c)) for c in per_seed]
        assert (fold.n_edges, fold.n_classes) == (60, len(owners))

    def test_every_edge_derived_folds_to_nothing(self):
        fold, _ = build_fold(_p({1: 1}, {1: 2}, {2: 1}), EdgeCanonicalizer(), derived={1, 2})
        assert fold.n_classes == 0 and fold.class_owners == {}
        assert fold.mass.tolist() == [0.0, 0.0, 0.0]
        assert [c.tolist() for c in fold.seed_classes] == [[], [], []]

    def test_a_seed_with_only_derived_edges_is_an_empty_row(self):
        fold, _ = build_fold(_p({1: 1}, {1: 1, 2: 2}, {3: 1}), EdgeCanonicalizer(), derived={1})
        assert fold.seed_classes[0].tolist() == []
        assert fold.degree[0] == 0 and fold.mass[0] == 0.0 and fold.total[0] == 1.0

    def test_seed_without_counts_reads_as_one_per_edge(self):
        class T:
            seed_edges = {"a": {1, 2}}
            seed_hit_counts: dict = {}

        assert profiles_from_tracker(T()) == {"a": {1: 1, 2: 1}}


class TestGate:
    def test_states(self):
        s = MatrixSubstrate(target="/x")
        assert s.trusted and s.gate_state() == "unverified"
        s.set_stability(1.0)
        assert s.trusted and s.gate_state() == "open"
        s.set_stability(0.007)
        assert not s.trusted and s.gate_state() == "closed"
        assert "not reproducible" in s.distrust_reason
        s.set_stability(None)
        assert s.trusted

    def test_no_target_has_nothing_to_distrust(self):
        # coverage_trust's own carve-out (in-process callable): the stability check is
        # not reached, and this module does not second-guess that decision.
        s = MatrixSubstrate()
        s.set_stability(0.0)
        assert s.trusted

    def test_no_coverage_carve_out_matches_coverage_trust(self):
        s = MatrixSubstrate(target="/x", use_coverage=False)
        s.set_stability(0.1)
        assert s.trusted  # the premise does not hold, so there is nothing to distrust

    def test_absent_instrumentation_closes_the_gate(self, monkeypatch):
        monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "absent")
        s = MatrixSubstrate(target="/x")
        assert not s.trusted and "no compiler-inserted" in s.distrust_reason


class TestRefit:
    def test_cadence_and_stamp(self):
        t = _Tracker(_p({1: 1}, {2: 1}, {3: 1}))
        s = MatrixSubstrate(refit_interval=100)
        assert s.maybe_refit(t, 0) and s.version == 1
        assert not s.maybe_refit(t, 50)
        assert not s.maybe_refit(t, 500)  # tracker unchanged
        t.grow("s3", {4: 1})
        assert s.maybe_refit(t, 600) and s.version == 2

    def test_too_small_records_the_reason(self):
        s = MatrixSubstrate()
        assert not s.maybe_refit(_Tracker(_p({1: 1}, {2: 1})), 0)
        assert s.fold is None and "seeds" in s.skip_reason

    def test_saturation_needs_flat_edges_and_falling_2h(self):
        s = MatrixSubstrate()
        assert s.saturation_signal() == 0.0
        s._history.extend([(100, 40.0), (100, 35.0), (100, 30.0), (100, 20.0)])
        assert s.saturation_signal() == 1.0  # (40-20)/40 = 0.5 of the start, scale 0.25, clipped
        s._history.clear()
        s._history.extend([(100, 40.0), (100, 38.0), (100, 37.0), (100, 36.0)])
        assert s.saturation_signal() == pytest.approx(0.4)
        s._history.clear()
        s._history.extend([(100, 40.0), (110, 30.0), (120, 20.0), (130, 10.0)])
        assert s.saturation_signal() == 0.0  # still finding edges: not saturation


class TestClassCredit:
    def test_chain_pays_once_and_unknown_edges_stand_alone(self):
        t = _Tracker(_p({10: 1, 11: 1, 12: 1}, {10: 1, 11: 1, 12: 1, 20: 4}, {20: 4}))
        s = MatrixSubstrate()
        s.maybe_refit(t, 0)
        assert s.class_credit({10, 11, 12}) == 1
        assert s.class_credit({10, 20}) == 2
        assert s.class_credit({900, 901}) == 2

    def test_a_split_raises_credit_because_nothing_is_stored(self):
        # P3-3's paper question. Edges 10 and 11 are one class until an input tells them
        # apart; credit for the same edge set is re-derived and rises from 1 to 2.
        t = _Tracker(_p({10: 1, 11: 1}, {10: 1, 11: 1}, {30: 1}))
        s = MatrixSubstrate()
        s.maybe_refit(t, 0)
        assert s.class_credit({10, 11}) == 1
        t.grow("s3", {10: 1})  # the first input that separates them
        s.maybe_refit(t, 10_000)
        assert s.class_credit({10, 11}) == 2

    def test_derived_edges_earn_nothing(self):
        s = MatrixSubstrate()
        s.derived = frozenset({11})
        assert s.class_credit({10, 11}) == 1
