"""Continuum wiring: flux ranking in invasion, diagnostics in the regime detector.

The regime detector deliberately does *not* classify on continuum signals
yet.  Handover §6 orders the work instrument-first and only promotes the
Reynolds diagnostic once it has been shown to correlate with the existing
CRITICAL label; :meth:`continuum_correlation` is what makes that step
measurable.  These tests pin the ordering so a later change has to be
deliberate.
"""

from fuzzer_tool.core.coverage_regime import CoverageRegimeDetector
from fuzzer_tool.core.critical_slowing import CriticalSlowingDown
from fuzzer_tool.core.navier_stokes import ContinuumField
from fuzzer_tool.core.percolation import CoverageRegime
from fuzzer_tool.services.seed_picker import INVASION_STUCK_THRESHOLD, invasion_select


def _field(velocity=2.0, failure_rate=0.25):
    f = ContinuumField()
    f.observe(
        occupancy={"a": 1.0, "b": 20.0, "c": 3.0},
        adjacency={"a": {"b"}, "b": {"a", "c"}, "c": {"b"}},
        velocity=velocity,
        failure_rate=failure_rate,
    )
    return f


class TestInvasionFluxRanking:
    def test_without_flux_the_resistance_rule_is_unchanged(self):
        """FALSIFICATION: default path must be byte-identical."""
        stats = {"havoc": (8.0, 2.0), "splice": (3.0, 7.0)}
        assert invasion_select(stats) == "havoc"

    def test_flux_map_can_change_the_winner(self):
        stats = {"lo_rate": (4.0, 6.0), "hi_rate": (8.0, 2.0)}
        # A flux map that favours the lower-resistance-losing arm.
        chosen = invasion_select(stats, flux_map={"lo_rate": 1.9, "hi_rate": 0.2})
        assert chosen == "lo_rate"

    def test_stuck_contract_is_the_same_predicate(self):
        """Flux picks WHICH arm, never WHETHER the cluster is stuck."""
        # Every arm below the 1/threshold success rate -> stuck either way.
        rate = 1.0 / INVASION_STUCK_THRESHOLD
        stats = {"a": (rate / 2, 1.0 - rate / 2), "b": (rate / 4, 1.0 - rate / 4)}
        assert invasion_select(stats) is None
        assert invasion_select(stats, flux_map={"a": 2.0, "b": 1.0}) is None

    def test_flux_entries_outside_operator_stats_are_ignored(self):
        """ADVERSARIAL: a stale map must not return an unavailable operator."""
        stats = {"havoc": (8.0, 2.0)}
        assert invasion_select(stats, flux_map={"gone": 2.0, "havoc": 0.1}) == "havoc"

    def test_empty_flux_map_falls_back_to_resistance(self):
        stats = {"havoc": (8.0, 2.0), "splice": (3.0, 7.0)}
        assert invasion_select(stats, flux_map={}) == "havoc"

    def test_missing_flux_entry_does_not_crash(self):
        stats = {"havoc": (8.0, 2.0), "splice": (7.0, 3.0)}
        assert invasion_select(stats, flux_map={"havoc": 1.5}) == "havoc"

    def test_empty_frontier_still_short_circuits(self):
        stats = {"havoc": (8.0, 2.0)}
        assert invasion_select(stats, frontier_edges=set(), flux_map={"havoc": 2.0}) is None

    def test_ties_break_deterministically(self):
        stats = {"b": (8.0, 2.0), "a": (8.0, 2.0)}
        chosen = invasion_select(stats, flux_map={"a": 1.0, "b": 1.0})
        assert chosen == "a"


class TestRegimeContinuumInstrumentation:
    def _detector(self, continuum=None):
        return CoverageRegimeDetector(
            csd=CriticalSlowingDown(),
            homogeneity=None,
            stall_threshold=10_000,
            continuum=continuum,
        )

    def test_continuum_does_not_change_classification(self):
        """FALSIFICATION: instrument-only means the label is untouched."""
        plain = self._detector()
        instr = self._detector(continuum=_field(velocity=1e6, failure_rate=0.0))
        args = (2.0, 0, None)
        a = plain.observe(*args, execs_since_edge=5, exec_count=1000)
        b = instr.observe(*args, execs_since_edge=5, exec_count=1000)
        assert a is b is CoverageRegime.SUPERCRITICAL

    def test_reason_carries_the_diagnostic(self):
        d = self._detector(continuum=_field())
        d.observe(2.0, 0, None, execs_since_edge=5, exec_count=1000)
        assert "Re=" in d.reason

    def test_reason_is_untouched_without_a_field(self):
        d = self._detector()
        d.observe(2.0, 0, None, execs_since_edge=5, exec_count=1000)
        assert "Re=" not in d.reason

    def test_correlation_groups_reynolds_by_regime(self):
        f = _field(velocity=4.0)
        d = self._detector(continuum=f)
        d.observe(2.0, 0, None, execs_since_edge=5, exec_count=1000)
        d.observe(0.0, 0, None, execs_since_edge=50_000, exec_count=2000)

        corr = d.continuum_correlation()
        assert CoverageRegime.SUPERCRITICAL in corr
        assert CoverageRegime.SUBCRITICAL in corr
        assert corr[CoverageRegime.SUPERCRITICAL] > 0.0

    def test_correlation_empty_without_a_field(self):
        d = self._detector()
        d.observe(2.0, 0, None, execs_since_edge=5, exec_count=1000)
        assert d.continuum_correlation() == {}

    def test_correlation_ignores_unobserved_field(self):
        d = self._detector(continuum=ContinuumField())
        d.observe(2.0, 0, None, execs_since_edge=5, exec_count=1000)
        assert d.continuum_correlation() == {}
