"""Gated Φ multi-target schedule (--target-schedule phi)."""

from fuzzer_tool.core.percolation import CoverageRegime, estimate_time_to_next_discovery
from fuzzer_tool.core.target_schedule import TargetSchedule


def test_phi_is_a_schedule_choice():
    assert TargetSchedule.PHI.value == "phi"
    assert "phi" in {m.value for m in TargetSchedule}


def test_phi_weights_prefer_higher_estimate():
    """Harder-looking target (subcritical, no progress) should outweigh an
    easy one when weight ∝ time-to-next-discovery."""

    class Soft:
        cumulative_edges = set(range(100))

    class Hard:
        cumulative_edges = set()

    soft = estimate_time_to_next_discovery(
        Soft(), coverage_regime=CoverageRegime.SUPERCRITICAL, target_delta=1
    )
    hard = estimate_time_to_next_discovery(
        Hard(), coverage_regime=CoverageRegime.SUBCRITICAL, target_delta=1
    )
    assert hard > soft > 0


def test_phi_profile_changes_estimate():
    class T:
        cumulative_edges = {0}

    low = estimate_time_to_next_discovery(
        T(),
        coverage_regime=CoverageRegime.CRITICAL,
        phi_profile={1: 1, 2: 1, 5: 1},
        target_delta=1,
    )
    high = estimate_time_to_next_discovery(
        T(),
        coverage_regime=CoverageRegime.CRITICAL,
        phi_profile={1: 20, 2: 20, 5: 20},
        target_delta=1,
    )
    assert low > high > 0
