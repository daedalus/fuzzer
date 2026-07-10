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
        assert "need" in reason

    def test_stops_on_good_enough(self):
        sec = SecretaryStopping(min_observations=5, exploration_frac=0.3)
        for i in range(10):
            sec.observe(0.1 * i)
        stop, _ = sec.should_stop()
        assert not stop

    def test_rank_varies_not_constant(self):
        sec = SecretaryStopping(min_observations=3, decay=1.0)
        sec.observe(0.5)
        assert sec.rank_of_best() == 1.0

        sec.observe(0.3)
        assert sec.rank_of_best() == 1.0

        sec.observe(0.7)
        assert sec.rank_of_best() == 2.0

        sec.observe(0.1)
        assert sec.rank_of_best() == 2.0

        sec.observe(0.9)
        assert sec.rank_of_best() == 3.0

    def test_rank_decreases_when_records_slide_out(self):
        sec = SecretaryStopping(window_size=5, min_observations=3, decay=1.0)
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            sec.observe(v)
        assert sec.rank_of_best() == 5.0

        sec.observe(0.1)
        assert sec.rank_of_best() == 4.0

        sec.observe(0.1)
        assert sec.rank_of_best() == 3.0

    def test_decay_weights_recent_records_more(self):
        sec_no = SecretaryStopping(decay=1.0, min_observations=3)
        sec_d = SecretaryStopping(decay=0.5, min_observations=3)

        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            sec_no.observe(v)
            sec_d.observe(v)

        assert sec_no.rank_of_best() == 5.0

        expected = 0.5**4 + 0.5**3 + 0.5**2 + 0.5**1 + 0.5**0
        assert abs(sec_d.rank_of_best() - expected) < 0.001

    def test_stops_on_plateau(self):
        sec = SecretaryStopping(min_observations=5, exploration_frac=0.3, decay=1.0)
        sec.observe(0.9)
        for _ in range(9):
            sec.observe(0.1)
        stop, reason = sec.should_stop()
        assert stop
        assert "rank" in reason

    def test_does_not_stop_during_improvement(self):
        sec = SecretaryStopping(min_observations=5, exploration_frac=0.3, decay=1.0)
        for i in range(10):
            sec.observe(0.1 * i)
        stop, _ = sec.should_stop()
        assert not stop

    def test_exploration_fraction(self):
        sec = SecretaryStopping(min_observations=5, exploration_frac=0.5)
        assert sec.exploration_fraction() == 1.0
        for i in range(10):
            sec.observe(0.1 * i)
        assert sec.exploration_fraction() == 1.0

    def test_reset(self):
        sec = SecretaryStopping()
        sec.observe(0.5)
        sec.observe(0.3)
        sec.reset()
        assert len(sec._observations) == 0
        assert len(sec._record_flags) == 0
        assert sec._record_count == 0
        assert sec._best_value == -math.inf

    def test_save_load(self):
        sec = SecretaryStopping(window_size=100, exploration_frac=0.4)
        sec.observe(0.5)
        sec.observe(0.3)
        state = sec.save()

        sec2 = SecretaryStopping()
        sec2.load(state)
        assert len(sec2._observations) == 2
        assert len(sec2._record_flags) == 2
        assert sec2._record_count == 1
        assert sec2.window_size == 100
        assert sec2.exploration_frac == 0.4
        assert sec2.decay == 0.95

    def test_save_load_roundtrip_preserves_rank(self):
        sec = SecretaryStopping(window_size=10, decay=0.8)
        for v in [0.1, 0.3, 0.2, 0.5, 0.4]:
            sec.observe(v)
        rank_before = sec.rank_of_best()

        state = sec.save()
        sec2 = SecretaryStopping()
        sec2.load(state)
        assert abs(sec2.rank_of_best() - rank_before) < 0.001

    def test_sliding_window(self):
        sec = SecretaryStopping(window_size=5)
        for i in range(10):
            sec.observe(0.1 * i)
        assert len(sec._observations) == 5
        assert len(sec._record_flags) == 5

    def test_repr(self):
        sec = SecretaryStopping()
        sec.observe(0.5)
        r = repr(sec)
        assert "SecretaryStopping" in r
        assert "n=1" in r

    def test_equal_observations_not_records(self):
        sec = SecretaryStopping(min_observations=3, decay=1.0)
        for _ in range(5):
            sec.observe(0.5)
        assert sec.rank_of_best() == 1.0

    def test_strictly_increasing_all_records(self):
        sec = SecretaryStopping(min_observations=3, decay=1.0)
        for i in range(10):
            sec.observe(float(i))
        assert sec.rank_of_best() == 10.0

    def test_stops_after_exploration_with_old_best(self):
        sec = SecretaryStopping(min_observations=5, exploration_frac=0.3, decay=1.0)
        sec.observe(1.0)
        for _ in range(19):
            sec.observe(0.01)
        stop, reason = sec.should_stop()
        assert stop
        assert "rank" in reason
