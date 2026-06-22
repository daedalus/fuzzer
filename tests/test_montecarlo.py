"""Tests for MonteCarloScheduler."""

from fuzzer_tool.core.montecarlo import MonteCarloScheduler


class TestMonteCarloScheduler:
    def test_init_defaults(self):
        mc = MonteCarloScheduler()
        assert mc.elite_frac == 0.1
        assert mc.refit_interval == 1000
        assert not mc.cem_fitted

    def test_init_arm(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        assert "bit_flip" in mc.arm_alpha
        assert mc.arm_alpha["bit_flip"] == 1.0
        assert mc.arm_beta["bit_flip"] == 1.0

    def test_init_arm_idempotent(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.init_arm("bit_flip")
        assert mc.arm_alpha["bit_flip"] == 1.0

    def test_select_op(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.init_arm("byte_flip")
        op = mc.select_op(["bit_flip", "byte_flip"])
        assert op in ("bit_flip", "byte_flip")

    def test_record_success(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.record("bit_flip", success=True)
        assert mc.arm_alpha["bit_flip"] == 2.0

    def test_record_failure(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.record("bit_flip", success=False)
        assert mc.arm_beta["bit_flip"] == 2.0

    def test_add_elite(self):
        mc = MonteCarloScheduler()
        mc.add_elite(b"AAAA", score=10)
        assert len(mc.elite_set) == 1

    def test_add_elite_caps_at_max(self):
        mc = MonteCarloScheduler()
        for i in range(250):
            mc.add_elite(b"AAAA", score=i)
        assert len(mc.elite_set) == MonteCarloScheduler.ELITE_MAX

    def test_maybe_refit_no_op(self):
        mc = MonteCarloScheduler(refit_interval=100)
        mc.maybe_refit()
        assert not mc.cem_fitted

    def test_maybe_refit_triggers(self):
        mc = MonteCarloScheduler(refit_interval=3)
        for i in range(50):
            mc.add_elite(bytes([i % 256] * 4), score=i)
        # Simulate fuzz_one incrementing execs_since_refit
        mc.execs_since_refit = 3
        mc.maybe_refit()
        assert mc.cem_fitted

    def test_cem_byte_range(self):
        mc = MonteCarloScheduler()
        mc.byte_freq[0] = {0: 10, 128: 5, 255: 1}
        for _ in range(100):
            b = mc.cem_byte(0)
            assert 0 <= b <= 255

    def test_cem_byte_unfitted(self):
        mc = MonteCarloScheduler()
        b = mc.cem_byte(0)
        assert 0 <= b <= 255

    def test_cem_sample_length(self):
        mc = MonteCarloScheduler()
        result = mc.cem_sample(8)
        assert len(result) == 8

    def test_bandit_stats(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.record("bit_flip", success=True)
        mc.record("bit_flip", success=False)
        stats = mc.bandit_stats()
        assert stats["bit_flip"] == (1.0, 1.0)
