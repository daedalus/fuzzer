"""Tests for secretary-problem optimal stopping."""

import math

from fuzzer_tool.core.secretary import SecretaryStopping


class TestSecretaryStopping:
    """Unit tests for SecretaryStopping class."""

    def test_initial_state(self):
        sec = SecretaryStopping()
        assert len(sec._observations) == 0
        assert sec.exploration_fraction() == 1.0
        stop, reason = sec.should_stop()
        assert not stop
        assert "need" in reason

    def test_exploration_phase(self):
        sec = SecretaryStopping(min_observations=5)
        for i in range(3):
            sec.observe(0.1 * i)
        stop, reason = sec.should_stop()
        assert not stop
        assert "need" in reason  # not enough observations yet

    def test_stops_on_good_enough(self):
        sec = SecretaryStopping(min_observations=5, exploration_frac=0.3)
        # Add 10 observations with improving quality
        for i in range(10):
            sec.observe(0.1 * i)
        # After 10 observations, best is at index 9 (rank 1)
        # exploration_frac=0.3 means threshold = 3 steps since improvement
        # Since best is last, steps_since_improvement = 0
        stop, _ = sec.should_stop()
        # Should not stop yet (best is improving)
        assert not stop

    def test_stops_after_plateau(self):
        sec = SecretaryStopping(min_observations=5, exploration_frac=0.3)
        # Add observations that plateau
        for i in range(5):
            sec.observe(0.5)  # all same quality
        # After 5 observations, best is 0.5 at index 0 (rank 1)
        # steps_since_improvement = 4 (no improvement since first)
        # threshold = floor(5 * 0.3) = 1
        # Since steps_since_improvement (4) >= threshold (1), check rank
        # rank (1) <= threshold (1), so should stop
        stop, reason = sec.should_stop()
        assert stop
        assert "rank" in reason

    def test_exploration_fraction(self):
        sec = SecretaryStopping(min_observations=5, exploration_frac=0.5)
        # During exploration phase (before enough observations)
        assert sec.exploration_fraction() == 1.0
        # Add some observations
        for i in range(10):
            sec.observe(0.1 * i)
        # Best is improving, so still in exploration
        assert sec.exploration_fraction() == 1.0

    def test_rank_computation(self):
        sec = SecretaryStopping()
        sec.observe(0.3)
        sec.observe(0.5)
        sec.observe(0.1)
        sec.observe(0.4)
        # Best is 0.5 at index 1, rank should be 1
        assert sec.rank_of_best() == 1

    def test_reset(self):
        sec = SecretaryStopping()
        sec.observe(0.5)
        sec.observe(0.3)
        sec.reset()
        assert len(sec._observations) == 0
        assert sec._best_value == -math.inf

    def test_save_load(self):
        sec = SecretaryStopping(window_size=100, exploration_frac=0.4)
        sec.observe(0.5)
        sec.observe(0.3)
        state = sec.save()

        sec2 = SecretaryStopping()
        sec2.load(state)
        assert len(sec2._observations) == 2
        assert sec2.window_size == 100
        assert sec2.exploration_frac == 0.4

    def test_sliding_window(self):
        sec = SecretaryStopping(window_size=5)
        for i in range(10):
            sec.observe(0.1 * i)
        # Window should only keep last 5
        assert len(sec._observations) == 5

    def test_repr(self):
        sec = SecretaryStopping()
        sec.observe(0.5)
        r = repr(sec)
        assert "SecretaryStopping" in r
        assert "n=1" in r
