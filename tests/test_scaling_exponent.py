"""Tests for core/scaling_exponent.py — anomalous-diffusion classification.

Three falsifiers, matching the module docstring: a pure symmetric random
walk must recover alpha near 1 (diffusive); a linear drift with small
additive noise must recover alpha near 2 (ballistic); an i.i.d.-noise-
about-a-constant series must recover alpha near 0 (trapped).
"""

from __future__ import annotations

import random

from fuzzer_tool.core.scaling_exponent import (
    ScalingExponentDetector,
    classify_exponent,
    estimate_scaling_exponent,
    mean_squared_displacement,
)


def _random_walk(n: int, seed: int = 1) -> list[float]:
    rng = random.Random(seed)
    x = 0.0
    out = []
    for _ in range(n):
        x += rng.choice((-1.0, 1.0))
        out.append(x)
    return out


def _ballistic(n: int, seed: int = 2, velocity: float = 1.0, noise: float = 0.05) -> list[float]:
    rng = random.Random(seed)
    return [velocity * i + rng.gauss(0, noise) for i in range(n)]


def _confined(n: int, seed: int = 3, level: float = 50.0, noise: float = 1.0) -> list[float]:
    rng = random.Random(seed)
    return [level + rng.gauss(0, noise) for _ in range(n)]


def test_mean_squared_displacement_basic():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert mean_squared_displacement(values, 1) == 1.0
    assert mean_squared_displacement(values, 5) is None
    assert mean_squared_displacement(values, 0) is None


def test_random_walk_is_diffusive():
    values = _random_walk(1000, seed=1)
    alpha = estimate_scaling_exponent(values)
    assert alpha is not None
    assert 0.6 <= alpha <= 1.4, f"expected diffusive alpha near 1, got {alpha}"
    assert classify_exponent(alpha) == "diffusive"


def test_random_walk_diffusive_on_average_across_seeds():
    alphas = [estimate_scaling_exponent(_random_walk(1000, seed=seed)) for seed in range(20)]
    assert all(a is not None for a in alphas)
    mean_alpha = sum(alphas) / len(alphas)
    assert 0.85 <= mean_alpha <= 1.15, f"mean alpha across seeds: {mean_alpha}"


def test_ballistic_drift_is_superdiffusive():
    values = _ballistic(200, seed=2)
    alpha = estimate_scaling_exponent(values)
    assert alpha is not None
    assert alpha > 1.5, f"expected ballistic alpha near 2, got {alpha}"
    assert classify_exponent(alpha) == "ballistic"


def test_confined_noise_is_subdiffusive():
    values = _confined(200, seed=3)
    alpha = estimate_scaling_exponent(values)
    assert alpha is not None
    assert alpha < 0.5, f"expected trapped alpha near 0, got {alpha}"
    assert classify_exponent(alpha) == "trapped"


def test_constant_series_is_undefined_not_a_crash():
    values = [5.0] * 100
    alpha = estimate_scaling_exponent(values)
    assert alpha is None


def test_too_short_series_returns_none():
    assert estimate_scaling_exponent([1.0, 2.0]) is None
    assert estimate_scaling_exponent([]) is None


def test_detector_insufficient_data_before_window_fills():
    d = ScalingExponentDetector(window=128)
    for v in _random_walk(10, seed=1):
        d.update(v)
    verdict = d.verdict()
    assert verdict["state"] == "insufficient_data"
    assert verdict["alpha"] is None


def test_detector_classifies_random_walk():
    d = ScalingExponentDetector(window=1000)
    for v in _random_walk(1000, seed=1):
        d.update(v)
    verdict = d.verdict()
    assert verdict["state"] == "diffusive"
    assert verdict["alpha"] is not None


def test_detector_classifies_ballistic():
    d = ScalingExponentDetector(window=200)
    for v in _ballistic(200, seed=2):
        d.update(v)
    verdict = d.verdict()
    assert verdict["state"] == "ballistic"


def test_detector_classifies_confined():
    d = ScalingExponentDetector(window=200)
    for v in _confined(200, seed=3):
        d.update(v)
    verdict = d.verdict()
    assert verdict["state"] == "trapped"


def test_detector_rolling_window_evicts_old_values():
    d = ScalingExponentDetector(window=1000)
    for v in _confined(1000, seed=3):
        d.update(v)
    for v in _random_walk(1000, seed=1):
        d.update(v)
    verdict = d.verdict()
    assert verdict["state"] == "diffusive"
