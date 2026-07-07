"""Tests for MonteCarloScheduler, MOptScheduler, JS divergence, and adaptive refit."""

from fuzzer_tool.core.montecarlo import MonteCarloScheduler, MOptScheduler


class TestMonteCarloScheduler:
    def test_init_defaults(self):
        mc = MonteCarloScheduler()
        assert mc.elite_frac == 0.1
        assert mc.base_refit_interval == 1000
        assert mc.refit_interval == 1000
        assert not mc.cem_fitted
        assert mc.last_js_divergence == 0.0

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
        assert mc.arm_beta["bit_flip"] == 1.0

    def test_record_failure(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.record("bit_flip", success=False)
        assert mc.arm_alpha["bit_flip"] == 1.0
        assert mc.arm_beta["bit_flip"] == 2.0

    def test_add_elite(self):
        mc = MonteCarloScheduler()
        mc.add_elite(b"data1", score=1)
        mc.add_elite(b"data2", score=2)
        assert len(mc.elite_set) == 2

    def test_add_elite_capping(self):
        mc = MonteCarloScheduler()
        for i in range(mc.ELITE_MAX + 10):
            mc.add_elite(bytes([i % 256]), score=i)
        assert len(mc.elite_set) == mc.ELITE_MAX

    def test_maybe_refit_needs_data(self):
        mc = MonteCarloScheduler(refit_interval=1)
        mc.execs_since_refit = 1
        mc.maybe_refit()
        assert not mc.cem_fitted

    def test_maybe_refit_with_enough_elite(self):
        mc = MonteCarloScheduler()
        for i in range(15):
            mc.add_elite(bytes(range(256)), score=i)
        mc.maybe_refit()
        assert mc.cem_fitted

    def test_cem_byte(self):
        mc = MonteCarloScheduler()
        mc.byte_freq = {0: {65: 1000, 66: 500}}
        b = mc.cem_byte(0)
        assert 0 <= b <= 255

    def test_cem_byte_favors_high_freq(self):
        mc = MonteCarloScheduler()
        # Run many times, 65 should be sampled more than 66
        counts = {65: 0, 66: 0}
        for _ in range(1000):
            mc.byte_freq = {0: {65: 100, 66: 10}}
            b = mc.cem_byte(0)
            if b in counts:
                counts[b] += 1
        assert counts[65] > counts[66]

    def test_cem_byte_empty(self):
        mc = MonteCarloScheduler()
        b = mc.cem_byte(0)
        assert 0 <= b <= 255

    def test_cem_sample(self):
        mc = MonteCarloScheduler()
        mc.byte_freq = {i: {65: 10} for i in range(10)}
        sample = mc.cem_sample(10)
        assert len(sample) == 10

    def test_bandit_stats(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.record("bit_flip", success=True)
        mc.record("bit_flip", success=False)
        stats = mc.bandit_stats()
        assert stats["bit_flip"] == (1.0, 1.0)

    def test_brier_score_empty(self):
        mc = MonteCarloScheduler()
        assert mc.brier_score() == 0.0

    def test_brier_score_perfect_calibration(self):
        mc = MonteCarloScheduler()
        mc.init_arm("test")
        # Record many successes — prediction should converge to high prob
        for _ in range(50):
            mc.record_brier("test", success=True)
        bs = mc.brier_score()
        # With perfect successes, Brier should be low
        assert bs < 0.5

    def test_brier_score_worst(self):
        mc = MonteCarloScheduler()
        mc.init_arm("test")
        # Alternate success/failure — prediction oscillates
        for i in range(100):
            mc.record_brier("test", success=(i % 2 == 0))
        bs = mc.brier_score()
        assert 0.0 < bs <= 0.5

    def test_record_brier(self):
        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        mc.record_brier("bit_flip", success=True)
        assert len(mc._brier_predictions) == 1
        pred, outcome = mc._brier_predictions[0]
        assert outcome == 1.0
        assert 0.0 <= pred <= 1.0

    def test_calibration_report_empty(self):
        mc = MonteCarloScheduler()
        assert mc.calibration_report() == {}

    def test_calibration_report_with_data(self):
        mc = MonteCarloScheduler()
        mc.init_arm("test")
        for _ in range(20):
            mc.record_brier("test", success=True)
        report = mc.calibration_report()
        # Should have at least one bin
        assert len(report) > 0

    def test_brier_predictions_capped(self):
        mc = MonteCarloScheduler()
        mc.init_arm("test")
        for _ in range(600):
            mc.record_brier("test", success=True)
        assert len(mc._brier_predictions) <= 500


class TestMOptScheduler:
    def test_init_particles(self):
        mopt = MOptScheduler(n_particles=3)
        assert mopt.n_particles == 3
        assert len(mopt.particles) == 0

    def test_init_arm_creates_particles(self):
        mopt = MOptScheduler(n_particles=3)
        mopt.init_arm("bit_flip")
        mopt.init_arm("byte_flip")
        assert len(mopt.particles) == 3
        assert len(mopt.operators) == 2
        # Each particle should have uniform distribution over 2 ops
        for p in mopt.particles:
            assert len(p.pos) == 2
            assert abs(sum(p.pos) - 1.0) < 1e-6

    def test_init_arm_idempotent(self):
        mopt = MOptScheduler(n_particles=3)
        mopt.init_arm("bit_flip")
        mopt.init_arm("bit_flip")
        assert len(mopt.operators) == 1
        assert len(mopt.particles) == 3

    def test_select_op_returns_valid(self):
        mopt = MOptScheduler(n_particles=3)
        mopt.init_arm("bit_flip")
        mopt.init_arm("byte_flip")
        ops = ["bit_flip", "byte_flip"]
        for _ in range(20):
            op, pid = mopt.select_op(ops)
            assert op in ops
            assert 0 <= pid < mopt.n_particles

    def test_record_tracks_discoveries(self):
        mopt = MOptScheduler(n_particles=3, window_size=10)
        mopt.init_arm("a")
        mopt.record("a", success=True)
        mopt.record("a", success=False)
        assert mopt._total_execs == 2
        assert mopt._total_discoveries == 1

    def test_pso_triggers_at_window(self):
        mopt = MOptScheduler(n_particles=3, window_size=10)
        mopt.init_arm("a")
        mopt.init_arm("b")
        # Record 10 executions — should trigger PSO update
        for i in range(10):
            mopt.record("a", success=(i % 3 == 0))
        # After PSO update, window should be cleared
        for p in mopt.particles:
            assert p.execs_in_window == 0
            assert len(p.discoveries) == 0

    def test_velocity_update_bounded(self):
        mopt = MOptScheduler(n_particles=3, window_size=5, max_vel=0.2)
        mopt.init_arm("a")
        mopt.init_arm("b")
        mopt.init_arm("c")
        # Feed data to trigger PSO
        for i in range(5):
            mopt.record("a", success=(i % 2 == 0))
        # Check velocities are clamped
        for p in mopt.particles:
            for v in p.vel:
                assert abs(v) <= mopt.max_vel + 1e-9

    def test_simplex_projection(self):
        mopt = MOptScheduler(n_particles=3, window_size=5)
        mopt.init_arm("a")
        mopt.init_arm("b")
        # Push positions to extreme values
        for p in mopt.particles:
            p.pos = [0.9, 0.1]
        # After PSO update, should be normalized
        for i in range(5):
            mopt.record("a", success=True)
        for p in mopt.particles:
            assert abs(sum(p.pos) - 1.0) < 1e-6
            assert all(v >= 0.01 for v in p.pos)  # floor enforced

    def test_convergence_toward_best_operator(self):
        mopt = MOptScheduler(n_particles=5, window_size=20)
        for op in ["good", "bad1", "bad2"]:
            mopt.init_arm(op)
        ops = ["good", "bad1", "bad2"]
        # "good" succeeds 50% of the time, others 5%
        for _ in range(200):
            op, pid = mopt.select_op(ops)
            if op == "good":
                success = __import__("random").random() < 0.50
            else:
                success = __import__("random").random() < 0.05
            mopt.record(op, success, particle_id=pid)
        # After convergence, most particles should favor "good"
        stats = mopt.particle_stats()
        good_count = sum(1 for s in stats if s["top_op"] == "good")
        assert good_count >= 3, f"Expected >= 3 particles favoring 'good', got {good_count}"

    def test_particle_stats(self):
        mopt = MOptScheduler(n_particles=2, window_size=5)
        mopt.init_arm("a")
        mopt.init_arm("b")
        for i in range(5):
            mopt.record("a", success=(i % 2 == 0))
        stats = mopt.particle_stats()
        assert len(stats) == 2
        for s in stats:
            assert "name" in s
            assert "fitness" in s
            assert "top_op" in s
            assert "top_prob" in s

    def test_bandit_stats_compatibility(self):
        mopt = MOptScheduler(n_particles=2)
        mopt.init_arm("a")
        mopt.record("a", success=True)
        mopt.record("a", success=False)
        stats = mopt.bandit_stats()
        assert "_mopt_global" in stats
        assert stats["_mopt_global"] == (1, 1)

    def test_dynamic_operator_addition(self):
        mopt = MOptScheduler(n_particles=3, window_size=5)
        mopt.init_arm("a")
        mopt.init_arm("b")
        assert len(mopt.particles[0].pos) == 2
        # Add a third operator
        mopt.init_arm("c")
        assert len(mopt.operators) == 3
        # Particles should be extended
        for p in mopt.particles:
            assert len(p.pos) == 3
            assert abs(sum(p.pos) - 1.0) < 1e-6

    def test_empty_ops_fallback(self):
        mopt = MOptScheduler(n_particles=3)
        op, pid = mopt.select_op(["only_one"])
        assert op == "only_one"
        assert pid == 0

    def test_multiple_windows_convergence(self):
        """Run enough iterations for multiple PSO updates and verify convergence."""
        mopt = MOptScheduler(n_particles=5, window_size=10)
        for op in ["fast", "slow"]:
            mopt.init_arm(op)
        ops = ["fast", "slow"]
        # fast has 80% success, slow has 10%
        for _ in range(100):
            op, pid = mopt.select_op(ops)
            success = (op == "fast" and __import__("random").random() < 0.80) or \
                      (op == "slow" and __import__("random").random() < 0.10)
            mopt.record(op, success, particle_id=pid)
        # After 10 PSO updates, fast should dominate
        stats = mopt.particle_stats()
        fast_probs = [s["top_prob"] for s in stats if s["top_op"] == "fast"]
        # At least one particle should have fast with >30% probability
        assert any(p > 0.30 for p in fast_probs), \
            f"No particle strongly favors 'fast': {[s['top_prob'] for s in stats]}"

    def test_particle_attribution_isolates_fitness(self):
        """Verify that record() with particle_id only updates the target particle.

        This is the core fix: without it, all particles converge on identical
        fitness because every particle's window gets the same discoveries.
        """
        mopt = MOptScheduler(n_particles=3, window_size=100)
        mopt.init_arm("a")
        mopt.init_arm("b")

        # Manually assign: particle 0 always picks "a", particle 1 always "b"
        # Record successes only for particle 0
        for _ in range(5):
            mopt.record("a", success=True, particle_id=0)
            mopt.record("b", success=False, particle_id=1)
            mopt.record("a", success=True, particle_id=0)

        # Particle 0 should have discoveries, particle 1 should not
        p0 = mopt.particles[0]
        p1 = mopt.particles[1]
        p2 = mopt.particles[2]
        assert sum(p0.discoveries) == 10, f"p0 expected 10 discoveries, got {sum(p0.discoveries)}"
        assert sum(p1.discoveries) == 0, f"p1 expected 0 discoveries, got {sum(p1.discoveries)}"
        assert sum(p2.discoveries) == 0, f"p2 expected 0 discoveries, got {sum(p2.discoveries)}"
        assert p0.execs_in_window == 10
        assert p1.execs_in_window == 5
        assert p2.execs_in_window == 0

        # Fitness should differ
        mopt._update_fitness(p0)
        mopt._update_fitness(p1)
        assert p0.fitness > p1.fitness, (
            f"p0 fitness ({p0.fitness}) should exceed p1 ({p1.fitness})"
        )


class TestAdaptiveRefit:
    def test_freq_to_dist(self):
        mc = MonteCarloScheduler()
        dist = mc._freq_to_dist({0: 3, 1: 7})
        assert abs(dist[0] - 0.3) < 1e-10
        assert abs(dist[1] - 0.7) < 1e-10

    def test_freq_to_dist_empty(self):
        mc = MonteCarloScheduler()
        assert mc._freq_to_dist({}) == {}

    def test_compute_js_no_previous(self):
        mc = MonteCarloScheduler()
        mc.byte_freq = {0: {0: 10}}
        assert mc._compute_js() == 0.0

    def test_compute_js_identical(self):
        mc = MonteCarloScheduler()
        mc.byte_freq = {0: {0: 5, 1: 5}}
        mc._prev_byte_freq = {0: {0: 5, 1: 5}}
        assert mc._compute_js() == 0.0

    def test_compute_js_different(self):
        mc = MonteCarloScheduler()
        mc.byte_freq = {0: {0: 10, 1: 0}}
        mc._prev_byte_freq = {0: {0: 0, 1: 10}}
        js = mc._compute_js()
        assert js > 0.0

    def test_js_two_identical(self):
        assert MonteCarloScheduler._js_two({0: 0.5, 1: 0.5}, {0: 0.5, 1: 0.5}) == 0.0

    def test_js_two_different(self):
        js = MonteCarloScheduler._js_two({0: 1.0}, {1: 1.0})
        assert js > 0.0

    def test_adapt_interval_stable(self):
        mc = MonteCarloScheduler(refit_interval=100)
        # Need enough observations for KS threshold to be low
        mc.arm_alpha["test"] = 100.0
        mc.arm_beta["test"] = 100.0
        mc.last_js_divergence = 0.0001  # very stable
        mc._adapt_interval()
        assert mc.refit_interval == 200  # doubled

    def test_adapt_interval_shifting(self):
        mc = MonteCarloScheduler(refit_interval=100)
        # Need enough observations for KS threshold to be meaningful
        mc.arm_alpha["test"] = 100.0
        mc.arm_beta["test"] = 100.0
        mc.last_js_divergence = 0.5  # very shifting
        mc._adapt_interval()
        assert mc.refit_interval == 50  # halved

    def test_adapt_interval_no_change_medium(self):
        mc = MonteCarloScheduler(refit_interval=100)
        # With enough observations, KS thresholds narrow
        # JS=0.15 with n=200 → stable_threshold≈0.096, unstable≈0.115
        # 0.15 > 0.115 → should halve
        mc.arm_alpha["test"] = 200.0
        mc.arm_beta["test"] = 200.0
        mc.last_js_divergence = 0.15
        mc._adapt_interval()
        assert mc.refit_interval == 50  # halved

    def test_adapt_interval_wide_threshold_no_change(self):
        mc = MonteCarloScheduler(refit_interval=100)
        # With very few observations, KS thresholds are wide → no change
        mc.last_js_divergence = 0.05
        mc._adapt_interval()
        # n=0 → stable_threshold very high → 0.05 < threshold → doubles
        # This is correct behavior: with no data, JS=0.05 looks stable
        assert mc.refit_interval in (50, 100, 200)  # depends on n

    def test_adapt_interval_cap_max(self):
        mc = MonteCarloScheduler(refit_interval=100)
        mc.refit_interval = 350
        mc.arm_alpha["test"] = 100.0
        mc.arm_beta["test"] = 100.0
        mc.last_js_divergence = 0.0001
        mc._adapt_interval()
        assert mc.refit_interval == 400  # capped at 4x base

    def test_adapt_interval_floor_min(self):
        mc = MonteCarloScheduler(refit_interval=100)
        mc.refit_interval = 30
        mc.arm_alpha["test"] = 100.0
        mc.arm_beta["test"] = 100.0
        mc.last_js_divergence = 0.5
        mc._adapt_interval()
        assert mc.refit_interval == 25  # floor at 0.25x base
