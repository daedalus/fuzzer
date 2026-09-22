"""Tests for KuramotoSyncMonitor (core/analyzers/analyzer_kuramoto_sync.py)."""

from fuzzer_tool.core.analyzers.analyzer_kuramoto_sync import KuramotoSyncMonitor


def _diag(r, n_arms=4, critical_coupling=2.0, spectral_radius=0.5, psi=0.0):
    return {
        "r": r,
        "psi": psi,
        "n_arms": n_arms,
        "critical_coupling": critical_coupling,
        "spectral_radius": spectral_radius,
    }


class TestInit:
    def test_defaults(self):
        m = KuramotoSyncMonitor()
        assert m.window_size == 50
        assert m.rise_threshold == 1.5
        assert m.min_observations == 20
        assert m.last_r == 0.0
        assert m.last_critical_coupling == float("inf")
        assert m.last_spectral_radius == 0.0
        assert m.n_observations == 0

    def test_custom_params(self):
        m = KuramotoSyncMonitor(window_size=10, rise_threshold=2.0, min_observations=5)
        assert m.window_size == 10
        assert m.rise_threshold == 2.0
        assert m.min_observations == 5


class TestObserve:
    def test_updates_last_values(self):
        m = KuramotoSyncMonitor()
        m.observe(_diag(r=0.42, critical_coupling=1.23, spectral_radius=0.87, n_arms=6))
        assert m.last_r == 0.42
        assert m.last_critical_coupling == 1.23
        assert m.last_spectral_radius == 0.87
        assert m._last_n_arms == 6
        assert m.n_observations == 1

    def test_missing_keys_degrade_gracefully(self):
        m = KuramotoSyncMonitor()
        m.observe({})
        assert m.last_r == 0.0
        assert m.last_critical_coupling == float("inf")
        assert m.last_spectral_radius == 0.0
        assert m._last_n_arms == 0
        assert m.n_observations == 1

    def test_infinite_critical_coupling_preserved(self):
        m = KuramotoSyncMonitor()
        m.observe(_diag(r=0.1, critical_coupling=float("inf")))
        assert m.last_critical_coupling == float("inf")

    def test_multiple_observations_increment_counter(self):
        m = KuramotoSyncMonitor()
        for i in range(5):
            m.observe(_diag(r=0.1 * i))
        assert m.n_observations == 5


class TestStatus:
    def test_too_few_arms_never_detects(self):
        m = KuramotoSyncMonitor(min_observations=3)
        for _ in range(30):
            m.observe(_diag(r=0.9, n_arms=1))
        detected, reason = m.status()
        assert not detected
        assert "arms" in reason

    def test_no_detection_with_stable_r(self):
        m = KuramotoSyncMonitor(min_observations=5, rise_threshold=1.5)
        for _ in range(10):
            m.observe(_diag(r=0.3))
        m.status()
        for _ in range(20):
            m.observe(_diag(r=0.3))
        detected, _ = m.status()
        assert not detected

    def test_detects_rising_order_parameter(self):
        # Mirrors test_critical_slowing.py's
        # test_detects_rising_variance_and_autocorrelation: establish a
        # flat baseline, then feed a steadily rising, noisy r(t) series
        # approaching synchronization (r -> 1).
        m = KuramotoSyncMonitor(min_observations=5, rise_threshold=1.5)
        for _ in range(10):
            m.observe(_diag(r=0.2))
        m.status()

        for i in range(20):
            r = min(0.2 + i * 0.03, 0.99)
            m.observe(_diag(r=r))
        detected, reason = m.status()
        assert detected
        assert "variance" in reason


class TestResetSaveLoad:
    def test_reset_clears_state(self):
        m = KuramotoSyncMonitor(min_observations=3)
        for i in range(10):
            m.observe(_diag(r=0.1 * i))
        m.reset()
        assert m.last_r == 0.0
        assert m.last_critical_coupling == float("inf")
        assert m.n_observations == 0
        assert len(m._csd._history) == 0

    def test_save_load_roundtrip(self):
        m = KuramotoSyncMonitor(window_size=10, rise_threshold=2.0, min_observations=3)
        for i in range(8):
            m.observe(_diag(r=0.05 * i, critical_coupling=1.5, spectral_radius=0.4, n_arms=5))

        data = m.save()

        m2 = KuramotoSyncMonitor()
        m2.load(data)

        assert m2.last_r == m.last_r
        assert m2.last_critical_coupling == m.last_critical_coupling
        assert m2.last_spectral_radius == m.last_spectral_radius
        assert m2.n_observations == m.n_observations
        assert m2.window_size == 10
        assert m2.rise_threshold == 2.0
        assert m2.min_observations == 3
        assert list(m2._csd._history) == list(m._csd._history)

    def test_load_missing_csd_key_leaves_fresh_detector(self):
        m = KuramotoSyncMonitor()
        m.load({"last_r": 0.5})
        assert m.last_r == 0.5
        assert len(m._csd._history) == 0
