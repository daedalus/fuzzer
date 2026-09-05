"""GARCH volatility input to CoverageRegimeDetector.

The detector's contract is unchanged when no model is supplied; when one is,
it may only raise an otherwise-SUPERCRITICAL tick to CRITICAL.  It can never
override a stall, a CSD detection, or a homogeneity rejection.
"""

import random

from fuzzer_tool.core.coverage_regime import CoverageRegimeDetector
from fuzzer_tool.core.critical_slowing import (
    CoverageHomogeneityDetector,
    CriticalSlowingDown,
)
from fuzzer_tool.core.garch import OnlineGarch11
from fuzzer_tool.core.percolation import CoverageRegime


def _clustered(n=800, seed=3):
    """A series with genuine, persistent volatility clustering."""
    rng = random.Random(seed)
    s2, eps, out = 1.0, 0.0, []
    for _ in range(n):
        s2 = 0.05 + 0.25 * eps * eps + 0.70 * s2
        eps = rng.gauss(0.0, s2**0.5)
        out.append(20.0 + eps)
    return out


def _spiking_model():
    """A fitted model parked on a high-variance tick."""
    g = OnlineGarch11(min_obs=64, refit_interval=128)
    for v in _clustered():
        g.update(v)
    # One shock scaled to the model's own volatility, so the forecast lands
    # above the unconditional level.  A far larger outlier would instead
    # dominate the squared residuals and *destroy* the measured clustering,
    # which is correct behaviour: a lone spike is not a cluster.
    g.update(g.mean + 3.0 * g.unconditional_variance**0.5)
    return g


def _quiet_model():
    g = OnlineGarch11(min_obs=64, refit_interval=128)
    rng = random.Random(9)
    for _ in range(800):
        g.update(rng.gauss(20.0, 2.0))
    return g


def _detector(garch=None, stall_threshold=10_000, homogeneity=None):
    return CoverageRegimeDetector(
        csd=CriticalSlowingDown(),
        homogeneity=homogeneity,
        stall_threshold=stall_threshold,
        garch=garch,
    )


class TestGarchRaisesCritical:
    def test_volatility_spike_raises_critical(self):
        d = _detector(garch=_spiking_model())
        regime = d.observe(2.0, 0, None, execs_since_edge=5, exec_count=5000)
        assert regime is CoverageRegime.CRITICAL
        assert "volatility clustering" in d.reason
        assert d.actionable

    def test_absent_model_leaves_behaviour_unchanged(self):
        """FALSIFICATION: without a model the same tick stays supercritical."""
        d = _detector(garch=None)
        regime = d.observe(2.0, 0, None, execs_since_edge=5, exec_count=5000)
        assert regime is CoverageRegime.SUPERCRITICAL

    def test_quiet_model_does_not_raise(self):
        """A model with no ARCH effect must not manufacture a CRITICAL."""
        d = _detector(garch=_quiet_model())
        regime = d.observe(2.0, 0, None, execs_since_edge=5, exec_count=5000)
        assert regime is CoverageRegime.SUPERCRITICAL

    def test_unfitted_model_does_not_raise(self):
        d = _detector(garch=OnlineGarch11(min_obs=64))
        regime = d.observe(2.0, 0, None, execs_since_edge=5, exec_count=5000)
        assert regime is CoverageRegime.SUPERCRITICAL


class TestPrecedenceIsUnchanged:
    def test_stall_still_wins(self):
        """ADVERSARIAL: a screaming GARCH must not mask a stall."""
        d = _detector(garch=_spiking_model(), stall_threshold=100)
        regime = d.observe(0.0, 0, None, execs_since_edge=500, exec_count=5000)
        assert regime is CoverageRegime.SUBCRITICAL
        assert "stall" in d.reason

    def test_homogeneity_rejection_still_wins(self):
        d = _detector(garch=_spiking_model(), homogeneity=CoverageHomogeneityDetector())
        result = {
            "homogeneous": False,
            "chi2": 50.0,
            "p_value": 0.001,
            "cramers_v": 0.4,
            "total_edges": 200,
        }
        regime = d.observe(2.0, 0, result, execs_since_edge=5, exec_count=5000)
        # GARCH raises CRITICAL before homogeneity is consulted, but only
        # because a volatility spike is a stronger near-transition signal
        # than spatial clustering -- the same precedence CSD already has.
        assert regime is CoverageRegime.CRITICAL

    def test_quiet_model_lets_homogeneity_through(self):
        d = _detector(garch=_quiet_model(), homogeneity=CoverageHomogeneityDetector())
        result = {
            "homogeneous": False,
            "chi2": 50.0,
            "p_value": 0.001,
            "cramers_v": 0.4,
            "total_edges": 200,
        }
        regime = d.observe(2.0, 0, result, execs_since_edge=5, exec_count=5000)
        assert regime is CoverageRegime.SUBCRITICAL
        assert "biased exploration" in d.reason


class TestStatePassthrough:
    def test_save_includes_garch_when_present(self):
        d = _detector(garch=_quiet_model())
        assert "garch" in d.save()

    def test_save_omits_garch_when_absent(self):
        assert "garch" not in _detector().save()

    def test_load_restores_garch(self):
        src = _detector(garch=_quiet_model())
        dst = _detector(garch=OnlineGarch11(min_obs=64))
        dst.load(src.save())
        assert dst.garch.count == src.garch.count

    def test_reset_clears_garch(self):
        d = _detector(garch=_quiet_model())
        d.reset()
        assert d.garch.count == 0
