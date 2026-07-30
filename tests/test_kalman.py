"""Tests for the Kalman filter module."""

import math
import random
from pathlib import Path

import pytest

from fuzzer_tool.core.kalman import KalmanFilter, RobustKF


# ── KalmanFilter: 1D ────────────────────────────────────────────────────


class TestKalmanFilter1D:
    def test_initial_state(self):
        kf = KalmanFilter(dim=1, initial_state=(42.0,))
        assert kf.estimate == 42.0
        assert kf.derivative == 0.0  # no derivative in 1D
        assert kf.uncertainty == 1.0  # default init cov
        assert kf.is_initialized
        assert kf.dim == 1

    def test_lazy_initialization(self):
        kf = KalmanFilter(dim=1)
        assert not kf.is_initialized
        assert kf.estimate == 0.0

        innovation = kf.update(10.0)
        assert kf.is_initialized
        assert kf.estimate == 10.0  # snap to first obs
        assert innovation == 0.0  # no innovation on snap

    def test_predict_then_update(self):
        kf = KalmanFilter(dim=1, process_noise=0.01, measurement_noise=0.5)
        kf.update(0.0)  # snap
        innovation = kf.update(5.0)
        # Innovation should be positive (measured above prediction)
        assert innovation > 0
        # Estimate should move toward measurement
        assert 0.0 < kf.estimate < 5.0

    def test_convergence_constant_signal(self):
        """KF should converge to the true value of a constant signal."""
        kf = KalmanFilter(dim=1, process_noise=0.001, measurement_noise=1.0)
        for _ in range(50):
            kf.update(42.0)
        assert abs(kf.estimate - 42.0) < 0.5

    def test_uncertainty_decreases_with_observations(self):
        kf = KalmanFilter(dim=1, process_noise=0.001, measurement_noise=1.0)
        uncertainties = []
        for _ in range(20):
            kf.update(0.0)
            uncertainties.append(kf.uncertainty)
        # Uncertainty should monotonically decrease
        for i in range(1, len(uncertainties)):
            assert uncertainties[i] <= uncertainties[i - 1] + 1e-10

    def test_noisy_signal_tracks_1d(self):
        """1D KF tracks a slowly drifting signal."""
        random.seed(42)
        kf = KalmanFilter(dim=1, process_noise=0.1, measurement_noise=0.5)
        true_value = 100.0
        for i in range(50):
            true_value += 0.5  # slow drift
            measurement = true_value + random.gauss(0, 0.5)
            kf.update(measurement)
        # 1D has no derivative so it lags the ramp, but should be
        # within ~10 of the final true value.
        assert abs(kf.estimate - true_value) < 15.0

    def test_noisy_signal_tracks_2d(self):
        """2D KF with predict() tracks a ramping signal."""
        random.seed(42)
        kf = KalmanFilter(dim=2, process_noise=0.05, measurement_noise=0.5)
        true_value = 100.0
        for i in range(50):
            true_value += 0.5  # slow drift
            measurement = true_value + random.gauss(0, 0.5)
            kf.predict(1.0)
            kf.update(measurement)

        # 2D with derivative should track the ramp more closely.
        assert abs(kf.estimate - true_value) < 5.0
        # Derivative should be positive (signal is drifting up).
        assert kf.derivative > 0.0

    def test_derivative_tracking(self):
        kf = KalmanFilter(dim=2, process_noise=0.05, measurement_noise=1.0)
        for t in range(1, 21):
            kf.predict(1.0)
            kf.update(float(t * 3))  # slope = 3
        assert abs(kf.derivative - 3.0) < 1.5  # approximate

    def test_predict_called_before_update(self):
        """Calling predict() before update() is safe."""
        kf = KalmanFilter(dim=1, initial_state=(0.0,))
        kf.predict(10.0)  # should not crash
        kf.update(5.0)
        assert kf.is_initialized

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be 1 or 2"):
            KalmanFilter(dim=3)

    def test_negative_dt(self):
        kf = KalmanFilter(dim=1, initial_state=(0.0,))
        with pytest.raises(ValueError, match="dt must be positive"):
            kf.predict(-1.0)

    def test_save_load_round_trip(self):
        kf = KalmanFilter(dim=2, process_noise=0.01, measurement_noise=0.5)
        for i in range(20):
            kf.predict(1.0)
            kf.update(float(i))
        saved = kf.save()
        kf2 = KalmanFilter(dim=2)
        kf2.load(saved)

        assert kf2.dim == kf.dim
        assert kf2.estimate == pytest.approx(kf.estimate)
        assert kf2.derivative == pytest.approx(kf.derivative)
        assert kf2.uncertainty == pytest.approx(kf.uncertainty)

    def test_reset(self):
        kf = KalmanFilter(dim=1, initial_state=(10.0,))
        kf.update(5.0)
        kf.reset()
        assert not kf.is_initialized
        assert kf.estimate == 0.0

    def test_reset_with_state(self):
        kf = KalmanFilter(dim=1)
        kf.reset(state=(100.0,))
        assert kf.is_initialized
        assert kf.estimate == 100.0

    def test_innovation_property(self):
        kf = KalmanFilter(dim=1, initial_state=(0.0,))
        innovation = kf.update(5.0)
        assert innovation == pytest.approx(kf.innovation)
        assert kf.innovation_variance > 0

    def test_state_covariance_properties(self):
        kf = KalmanFilter(dim=2, initial_state=(1.0, 0.5))
        assert kf.state == [1.0, 0.5]
        cov = kf.covariance
        assert len(cov) == 2
        assert len(cov[0]) == 2
        # Cov should be symmetric
        assert cov[0][1] == pytest.approx(cov[1][0])


# ── RobustKF ─────────────────────────────────────────────────────────────


class TestRobustKF:
    def test_initial_state(self):
        rkf = RobustKF(dim=1, initial_state=(10.0,))
        assert rkf.estimate == 10.0
        assert rkf.huber_threshold == 3.0

    def test_huber_gating_activates_on_outlier(self):
        """Huber gating inflates the step-R on large innovations."""
        rkf = RobustKF(dim=1, process_noise=0.001, measurement_noise=0.5)
        rkf.update(0.0)  # snap
        for _ in range(10):
            rkf.update(0.0)  # settle

        # The filter is now settled near 0 with small P.
        r_base_before = rkf.effective_measurement_noise
        rkf.update(1000.0)  # huge outlier

        # After the outlier, the effective R should be base (gating
        # is one-shot), but the step-wise R was inflated.
        # The filter should not have been pulled far.
        assert abs(rkf.estimate) < 100.0  # not pulled all the way to 1000

    def test_huber_gating_one_shot(self):
        """Gating does NOT persist the inflated R to next step."""
        rkf = RobustKF(dim=1, process_noise=0.001, measurement_noise=0.5)
        rkf.update(0.0)
        for _ in range(10):
            rkf.update(0.0)

        # Inject an outlier.
        rkf.update(1000.0)

        # After one normal observation, effective R must be back to base.
        rkf.update(0.0)
        assert rkf.effective_measurement_noise == pytest.approx(0.5, rel=0.1)

    def test_adaptive_r_grows_for_noisy_signal(self):
        """Adaptive R (not gating) should increase effective R over time."""
        rkf = RobustKF(
            dim=1, process_noise=0.01, measurement_noise=0.1,
            adaptive_r_gain=0.05, huber_threshold=100.0,  # disable gating
        )
        r_base = rkf.effective_measurement_noise
        for _ in range(60):
            rkf.update(random.gauss(0, 2.0))  # very noisy relative to initial R
        assert rkf.effective_measurement_noise > r_base * 1.5

    def test_effective_r_stable_without_adaptive_gain(self):
        """Without adaptive gain, effective R stays at base."""
        rkf = RobustKF(
            dim=1, process_noise=0.01, measurement_noise=0.1,
            adaptive_r_gain=0.0, huber_threshold=100.0,  # disable gating too
        )
        for _ in range(20):
            rkf.update(random.gauss(0, 2.0))
        assert rkf.effective_measurement_noise == pytest.approx(0.1)

    def test_constant_signal_convergence(self):
        """Robust KF should still converge to constant signal."""
        rkf = RobustKF(dim=1, process_noise=0.001, measurement_noise=0.5)
        for _ in range(50):
            rkf.update(42.0)
        assert abs(rkf.estimate - 42.0) < 0.5

    def test_overridden_params(self):
        rkf = RobustKF(dim=2, huber_threshold=2.0, max_r_inflation=20.0)
        assert rkf.huber_threshold == 2.0
        assert rkf.effective_measurement_noise == pytest.approx(0.1)
        # Parameters should persist through save/load
        saved = rkf.save()
        rkf2 = RobustKF()
        rkf2.load(saved)
        assert rkf2.huber_threshold == 2.0

    def test_save_load_robust(self):
        rkf = RobustKF(dim=2, process_noise=0.01, measurement_noise=0.5)
        for i in range(20):
            rkf.predict(1.0)
            rkf.update(float(i))
        saved = rkf.save()
        assert "huber_threshold" in saved
        assert "adaptive_r_gain" in saved
        assert "r_eff" in saved

        rkf2 = RobustKF()
        rkf2.load(saved)
        assert rkf2.estimate == pytest.approx(rkf.estimate)
        assert rkf2.effective_measurement_noise == pytest.approx(rkf.effective_measurement_noise)

    def test_reset(self):
        rkf = RobustKF(dim=1, initial_state=(10.0,))
        rkf.update(5.0)
        rkf.reset()
        assert not rkf.is_initialized
        assert rkf.estimate == 0.0


# ── Integration-oriented smoke tests ────────────────────────────────────


class TestKFApplications:
    def test_denoise_constant_signal_with_outliers(self):
        """Smoke test for Application 2 (denoising upstream of CSD/Allan)."""
        rkf = RobustKF(dim=2, process_noise=0.01, measurement_noise=0.5)
        true_rate = 50.0
        random.seed(0)
        for _ in range(40):
            rkf.predict(1.0)
            obs = true_rate + random.gauss(0, 0.5)
            rkf.update(obs)

        stable_estimate = rkf.estimate

        # Inject 5 large spikes (simulating bursty discovery)
        for _ in range(5):
            rkf.predict(1.0)
            rkf.update(true_rate + random.gauss(0, 10.0))
        for _ in range(10):
            rkf.predict(1.0)
            rkf.update(true_rate + random.gauss(0, 0.5))

        # Robust estimate should not be pulled far by the spikes.
        assert abs(rkf.estimate - stable_estimate) < 10.0

    def test_latency_tracking_smoke(self):
        """Smoke test for Application 1 (adaptive settle time)."""
        # Latency ≈ 20ms with occasional GC pauses to 100ms
        kf = KalmanFilter(dim=1, process_noise=0.005, measurement_noise=0.01, initial_state=(0.02,))
        true_latency = 0.02
        for _ in range(30):
            obs = true_latency + random.gauss(0, 0.005)
            kf.update(max(0.0, obs))
        # Should track around 20ms
        assert 0.01 < kf.estimate < 0.04

    def test_eps_tracking_smoke(self):
        """Smoke test for Application 3 (EPS estimation)."""
        kf = KalmanFilter(dim=2, process_noise=0.1, measurement_noise=10.0)
        true_eps = 1000.0
        for _ in range(20):
            kf.predict(1.0)
            kf.update(true_eps + random.gauss(0, 50))
        assert abs(kf.estimate - true_eps) < 50
