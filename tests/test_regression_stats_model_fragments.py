"""The GARCH / continuum stats fragments must never take down print_stats.

Both helpers read attributes off the fuzzer with ``getattr``.  A stand-in
that answers *every* attribute -- a test double, or an object restored from
a partial state -- returns a non-``None`` non-number, and formatting that
with ``:.2f`` raises ``TypeError`` and kills the whole stats line, not just
the fragment.  Guard on the value's type, not on ``is None``.
"""

from unittest.mock import MagicMock

from fuzzer_tool.core.garch import OnlineGarch11
from fuzzer_tool.core.navier_stokes import ContinuumField
from fuzzer_tool.services.stats import StatsReporter


def _reporter():
    return StatsReporter.__new__(StatsReporter)


class TestGarchFragment:
    def test_answers_everything_stand_in_yields_empty(self):
        f = MagicMock()
        assert _reporter()._print_stats_garch_str(f) == ""

    def test_absent_model_yields_empty(self):
        f = object()
        assert _reporter()._print_stats_garch_str(f) == ""

    def test_unfitted_model_yields_empty(self):
        f = MagicMock()
        f._garch = OnlineGarch11(min_obs=64)
        assert _reporter()._print_stats_garch_str(f) == ""

    def test_fitted_model_renders(self):
        g = OnlineGarch11(min_obs=8, refit_interval=0)
        for v in (1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0):
            g.update(v)
        f = MagicMock()
        f._garch = g
        out = _reporter()._print_stats_garch_str(f)
        assert out.startswith(" | vol: ")


class TestContinuumFragment:
    def test_answers_everything_stand_in_yields_empty(self):
        f = MagicMock()
        assert _reporter()._print_stats_continuum_str(f) == ""

    def test_absent_field_yields_empty(self):
        f = object()
        assert _reporter()._print_stats_continuum_str(f) == ""

    def test_unobserved_field_yields_empty(self):
        f = MagicMock()
        f._continuum = ContinuumField()
        assert _reporter()._print_stats_continuum_str(f) == ""

    def test_observed_field_renders(self):
        field = ContinuumField()
        field.observe(
            occupancy={"a": 1.0, "b": 8.0},
            adjacency={"a": {"b"}, "b": {"a"}},
            velocity=2.0,
            failure_rate=0.3,
        )
        f = MagicMock()
        f._continuum = field
        out = _reporter()._print_stats_continuum_str(f)
        assert out.startswith(" | Re: ")
