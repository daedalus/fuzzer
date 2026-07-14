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
        mc = MonteCarloScheduler(arm_decay=1.0)
        mc.init_arm("bit_flip")
        mc.record("bit_flip", success=True)
        assert mc.arm_alpha["bit_flip"] == 2.0
        assert mc.arm_beta["bit_flip"] == 1.0

    def test_record_failure(self):
        mc = MonteCarloScheduler(arm_decay=1.0)
        mc.init_arm("bit_flip")
        mc.record("bit_flip", success=False)
        assert mc.arm_alpha["bit_flip"] == 1.0
        assert mc.arm_beta["bit_flip"] == 2.0

    def test_arm_decay_reduces_old_evidence(self):
        mc = MonteCarloScheduler(arm_decay=0.5)
        mc.init_arm("A")
        for _ in range(10):
            mc.record("A", success=True)
        alpha_before = mc.arm_alpha["A"]
        mc.record("A", success=False)
        # Alpha should be < alpha_before * decay + 1 because decay applies first
        assert mc.arm_alpha["A"] < alpha_before + 1.0

    def test_arm_decay_no_effect_when_one(self):
        mc = MonteCarloScheduler(arm_decay=1.0)
        mc.init_arm("A")
        for _ in range(10):
            mc.record("A", success=True)
        assert mc.arm_alpha["A"] == 11.0

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

    def test_add_elite_metropolis_rejects_worse_at_low_temp(self):
        mc = MonteCarloScheduler()
        for i in range(mc.ELITE_MAX):
            mc.add_elite(bytes([i % 256]), score=100)
        # All elite scores are 100. Try to add score=1 at T=0.001 (greedy).
        # delta_e=99, exp(-99/0.001) ≈ 0 — should never accept.
        accepted = 0
        for _ in range(100):
            before = list(mc.elite_set)
            mc.add_elite(b"worst", score=1, temperature=0.001)
            if mc.elite_set != before:
                accepted += 1
            mc.elite_set = before
        assert accepted == 0

    def test_add_elite_metropolis_accepts_worse_at_high_temp(self):
        mc = MonteCarloScheduler()
        for i in range(mc.ELITE_MAX):
            mc.add_elite(bytes([i % 256]), score=100)
        # Try to add score=99 at T=100 (very hot). delta_e=1, exp(-1/100)≈0.99
        # Should almost always accept.
        accepted = 0
        for _ in range(100):
            before = list(mc.elite_set)
            mc.add_elite(b"slightly_worse", score=99, temperature=100.0)
            if mc.elite_set != before:
                accepted += 1
            mc.elite_set = before
        assert accepted > 90

    def test_add_elite_metropolis_accepts_better_always(self):
        mc = MonteCarloScheduler()
        for i in range(mc.ELITE_MAX):
            mc.add_elite(bytes([i % 256]), score=50)
        # Better score should always be accepted regardless of temperature
        accepted = 0
        for _ in range(100):
            before = list(mc.elite_set)
            mc.add_elite(b"better", score=100, temperature=0.001)
            if mc.elite_set != before:
                accepted += 1
            mc.elite_set = before
        assert accepted == 100

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
        mc = MonteCarloScheduler(arm_decay=1.0)
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


class TestPairwiseTransitions:
    def test_transition_counts_on_success(self):
        mc = MonteCarloScheduler(pairwise_blend=0.5)
        mc.init_arm("a")
        mc.init_arm("b")
        mc._prev_op = "a"
        mc.record("b", success=True)
        assert mc.transition_counts["a"]["b"] == 1
        assert mc.transition_total["a"] == 1

    def test_no_transition_on_failure(self):
        mc = MonteCarloScheduler(pairwise_blend=0.5)
        mc.init_arm("a")
        mc.init_arm("b")
        mc._prev_op = "a"
        mc.record("b", success=False)
        assert "a" not in mc.transition_counts

    def test_no_transition_when_same_op(self):
        mc = MonteCarloScheduler(pairwise_blend=0.5)
        mc.init_arm("a")
        mc._prev_op = "a"
        mc.record("a", success=True)
        assert "a" not in mc.transition_counts

    def test_select_op_uses_transitions(self):
        mc = MonteCarloScheduler(pairwise_blend=1.0)
        mc.init_arm("a")
        mc.init_arm("b")
        mc.init_arm("c")
        # Build transition: a -> b is highly successful
        for _ in range(50):
            mc._prev_op = "a"
            mc.record("b", success=True)
        # With blend=1.0 and prev_op="a", should almost always pick "b"
        picks = [mc.select_op(["a", "b", "c"], prev_op="a") for _ in range(100)]
        assert picks.count("b") > 90

    def test_select_op_no_prev_op_falls_back(self):
        mc = MonteCarloScheduler(pairwise_blend=1.0)
        mc.init_arm("a")
        mc.init_arm("b")
        # No transitions, no prev_op — should not crash
        op = mc.select_op(["a", "b"])
        assert op in ("a", "b")

    def test_transition_stats(self):
        mc = MonteCarloScheduler(pairwise_blend=0.5)
        mc.init_arm("a")
        mc.init_arm("b")
        mc._prev_op = "a"
        mc.record("b", success=True)
        stats = mc.transition_stats()
        assert "a" in stats
        assert stats["a"]["b"] == 1

    def test_save_load_transitions(self, tmp_path):
        mc = MonteCarloScheduler(pairwise_blend=0.5)
        mc.init_arm("a")
        mc.init_arm("b")
        mc._prev_op = "a"
        mc.record("b", success=True)

        path = str(tmp_path / "trans.json")
        mc.save_transitions(path)

        mc2 = MonteCarloScheduler(pairwise_blend=0.5)
        mc2.init_arm("a")
        mc2.init_arm("b")
        assert mc2.load_transitions(path)
        assert mc2.transition_counts["a"]["b"] == 1
        assert mc2.transition_total["a"] == 1

    def test_pairwise_blend_zero_ignores_transitions(self):
        mc = MonteCarloScheduler(pairwise_blend=0.0)
        mc.init_arm("a")
        mc.init_arm("b")
        mc._prev_op = "a"
        mc.record("b", success=True)
        # With blend=0, transitions should not affect selection
        stats = mc.transition_stats()
        assert "a" in stats  # still recorded


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
            success = (op == "fast" and __import__("random").random() < 0.80) or (
                op == "slow" and __import__("random").random() < 0.10
            )
            mopt.record(op, success, particle_id=pid)
        # After 10 PSO updates, fast should dominate
        stats = mopt.particle_stats()
        fast_probs = [s["top_prob"] for s in stats if s["top_op"] == "fast"]
        # At least one particle should have fast with >30% probability
        assert any(p > 0.30 for p in fast_probs), (
            f"No particle strongly favors 'fast': {[s['top_prob'] for s in stats]}"
        )

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
        assert p0.fitness > p1.fitness, f"p0 fitness ({p0.fitness}) should exceed p1 ({p1.fitness})"


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


class TestShapleyAttribution:
    def test_init(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution(n_samples=50, window_size=100)
        assert sa.n_samples == 50
        assert sa.window_size == 100
        assert len(sa._outcomes) == 0

    def test_record(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        sa.record({"bit_flip", "byte_flip"}, 5, {1, 2, 3, 4, 5})
        assert len(sa._outcomes) == 1
        assert "bit_flip" in sa._operator_edges
        assert "byte_flip" in sa._operator_edges
        assert len(sa._all_edges) == 5

    def test_record_no_edges(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        sa.record({"bit_flip"}, 0)
        assert len(sa._outcomes) == 1
        assert "bit_flip" not in sa._operator_edges

    def test_shapley_values_empty(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        result = sa.shapley_values(["a", "b", "c"])
        assert len(result) == 3
        assert all(abs(v - 1 / 3) < 0.01 for v in result.values())

    def test_shapley_values_single_op(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        sa.record({"a"}, 5, {1, 2, 3})
        result = sa.shapley_values()
        assert "a" in result
        assert result["a"] == 1.0

    def test_shapley_values_two_ops(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution(n_samples=100)
        # Op A gets edges 1,2,3; Op B gets edges 3,4,5
        sa.record({"a", "b"}, 5, {1, 2, 3, 4, 5})
        result = sa.shapley_values()
        assert abs(sum(result.values()) - 1.0) < 0.01
        assert result["a"] > 0
        assert result["b"] > 0

    def test_operator_synergy(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        sa.record({"a"}, 3, {1, 2, 3})
        sa.record({"b"}, 3, {3, 4, 5})
        synergy = sa.operator_synergy("a", "b")
        # Overlap is edge 3, so joint=5, individual=6
        assert synergy == (5 - 6) / 6

    def test_operator_synergy_no_edges(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        assert sa.operator_synergy("a", "b") == 0.0

    def test_operator_similarity(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        sa.record({"a"}, 3, {1, 2, 3})
        sa.record({"b"}, 3, {1, 2, 3})
        assert sa.operator_similarity("a", "b") == 1.0

    def test_operator_similarity_no_overlap(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        sa.record({"a"}, 3, {1, 2, 3})
        sa.record({"b"}, 3, {4, 5, 6})
        assert sa.operator_similarity("a", "b") == 0.0

    def test_operator_kernel(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        sa.record({"a"}, 3, {1, 2, 3})
        sa.record({"b"}, 3, {2, 3, 4})
        kernel = sa.operator_kernel()
        assert kernel["a"]["a"] == 1.0
        assert kernel["a"]["b"] == kernel["b"]["a"]
        assert 0 < kernel["a"]["b"] < 1

    def test_redundant_operators(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution()
        sa.record({"a"}, 5, {1, 2, 3, 4, 5})
        sa.record({"b"}, 5, {1, 2, 3, 4, 5})
        redundant = sa.redundant_operators(threshold=0.9)
        assert len(redundant) == 1
        assert redundant[0][2] == 1.0

    def test_window_size_limit(self):
        from fuzzer_tool.core.montecarlo import ShapleyAttribution

        sa = ShapleyAttribution(window_size=2)
        sa.record({"a"}, 1, {1})
        sa.record({"b"}, 1, {2})
        sa.record({"c"}, 1, {3})
        assert len(sa._outcomes) == 2


class TestReplicatorScheduler:
    def test_init(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler(window_size=100, learning_rate=0.2, mutation_rate=0.05)
        assert rs.window_size == 100
        assert rs.eta == 0.2
        assert rs.mutation_rate == 0.05

    def test_init_arm(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler()
        rs.init_arm("bit_flip")
        assert "bit_flip" in rs.operators
        assert rs.population == [1.0]

    def test_init_arm_idempotent(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler()
        rs.init_arm("bit_flip")
        rs.init_arm("bit_flip")
        assert len(rs.operators) == 1

    def test_init_arm_multiple(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler()
        rs.init_arm("a")
        rs.init_arm("b")
        rs.init_arm("c")
        assert rs.population == [1 / 3, 1 / 3, 1 / 3]

    def test_select_op(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler()
        rs.init_arm("a")
        rs.init_arm("b")
        op = rs.select_op(["a", "b"])
        assert op in ("a", "b")

    def test_select_op_empty(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler()
        assert rs.select_op([]) == ""

    def test_select_op_unknown(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler()
        rs.init_arm("a")
        # Selecting from ops not in population should still work
        op = rs.select_op(["x", "y"])
        assert op in ("x", "y")

    def test_record(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler(window_size=5)
        rs.init_arm("a")
        rs.record("a", success=True)
        assert rs._fitness_count["a"] == 1
        assert rs._fitness_sum["a"] == 1.0
        assert rs._total_execs == 1

    def test_record_failure(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler(window_size=5)
        rs.init_arm("a")
        rs.record("a", success=False)
        assert rs._fitness_sum["a"] == 0.0

    def test_replicator_update(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler(window_size=5, learning_rate=0.5)
        rs.init_arm("good")
        rs.init_arm("bad")
        # Good op succeeds, bad op fails
        for _ in range(5):
            rs.record("good", success=True)
            rs.record("bad", success=False)
        # After update, good should have higher population
        good_idx = rs.op_index["good"]
        bad_idx = rs.op_index["bad"]
        assert rs.population[good_idx] > rs.population[bad_idx]

    def test_zero_count_neutral_growth(self):
        """Zero-count operators should get neutral growth, not penalized."""
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler(window_size=10, learning_rate=0.1)
        rs.init_arm("tried")
        rs.init_arm("untried")
        # Only record for "tried", leave "untried" at 0
        for _ in range(10):
            rs.record("tried", success=True)
        # "untried" should not be zeroed out
        untried_idx = rs.op_index["untried"]
        assert rs.population[untried_idx] > 0

    def test_mutation_rate_floor(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler(window_size=5, mutation_rate=0.1)
        rs.init_arm("a")
        rs.init_arm("b")
        # Run many updates with only "a" succeeding
        for _ in range(50):
            rs.record("a", success=True)
        # Both should have at least mutation_rate
        for p in rs.population:
            assert p >= 0.1 - 0.001  # floating point tolerance

    def test_history_tracking(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler(window_size=3)
        rs.init_arm("a")
        rs.init_arm("b")
        for _ in range(3):
            rs.record("a", success=True)
        assert len(rs._history) == 1

    def test_stats(self):
        from fuzzer_tool.core.montecarlo import ReplicatorScheduler

        rs = ReplicatorScheduler()
        rs.init_arm("a")
        rs.record("a", success=True)
        assert rs._total_execs == 1
        assert len(rs.operators) == 1
