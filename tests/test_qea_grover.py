import math

import numpy as np

from fuzzer_tool.core.qea import _uniform_amplitudes, collapse
from fuzzer_tool.core.qea_grover import (
    LiveFractionTracker,
    grover_angle,
    grover_k_opt,
    grover_success_probability,
    grover_theta,
)


def test_theta_and_kopt_match_worked_example():
    # N=16, M=2 (x*x % 11 == 3): k*=1.67, k=2 gives 0.9453, k=3 overshoots to 0.3301
    assert math.isclose(grover_theta(2, 16), 0.36136712, rel_tol=1e-6)
    assert math.isclose(grover_k_opt(2, 16), 1.6734079, rel_tol=1e-6)
    assert math.isclose(grover_success_probability(2, 2, 16), 0.9453, abs_tol=1e-4)
    assert math.isclose(grover_success_probability(3, 2, 16), 0.3301, abs_tol=1e-4)


def test_kopt_infinite_without_marked_items():
    assert grover_k_opt(0, 16) == math.inf


def test_angle_is_clamped_and_monotone_in_m():
    lo = grover_angle(1, 4096)
    hi = grover_angle(2048, 4096)
    assert 0.005 <= lo < hi <= 0.2


def test_tracker_falls_back_without_evidence():
    assert LiveFractionTracker(64).angle(0.05) == 0.05


def test_or_mask_saturates_on_random_collapses():
    """Regression for the measured negative result: dead bits differ too."""
    np.random.seed(0)
    n = 256
    amps = _uniform_amplitudes(n)
    tr = LiveFractionTracker(n)
    tr.observe_improvement(
        int.from_bytes(collapse(amps), "big"), int.from_bytes(collapse(amps), "big")
    )
    assert tr.m > n // 4  # one sample already marks ~half the bits
