"""Tests for core/schedules.py — power schedules for seed-level energy."""

import math

import pytest

from fuzzer_tool.core.schedules import SeedScorer, compute_mean_log_n_fuzz


class TestSeedScorerInit:
    def test_valid_schedules(self):
        for s in SeedScorer.SCHEDULES:
            sc = SeedScorer(s)
            assert sc.schedule == s

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            SeedScorer("bogus")

    def test_default_is_base(self):
        sc = SeedScorer()
        assert sc.schedule == "base"


class TestSpeedFactor:
    def test_very_slow(self):
        sc = SeedScorer()
        # exec_us * 0.1 > avg_exec_us → 0.10
        assert sc._speed_factor(1000, 50) == 0.10

    def test_slow(self):
        sc = SeedScorer()
        assert sc._speed_factor(1000, 200) == 0.25

    def test_moderately_slow(self):
        sc = SeedScorer()
        # exec_us * 0.75 > avg_exec_us → 0.75
        assert sc._speed_factor(1000, 600) == 0.75

    def test_slightly_slow(self):
        sc = SeedScorer()
        # exec_us * 0.75 = 750 < 800 → no slow bracket → 1.0
        assert sc._speed_factor(1000, 800) == 1.0

    def test_normal(self):
        sc = SeedScorer()
        assert sc._speed_factor(1000, 1000) == 1.00

    def test_fast(self):
        sc = SeedScorer()
        # exec_us * 4 = 400 < 500 → 3.0
        assert sc._speed_factor(100, 500) == 3.00

    def test_faster(self):
        sc = SeedScorer()
        # exec_us * 3 = 300 < 400 → 2.0
        assert sc._speed_factor(100, 400) == 2.00

    def test_very_fast(self):
        sc = SeedScorer()
        # exec_us * 2 = 200 < 500 → 1.5
        assert sc._speed_factor(100, 250) == 1.50


class TestBitmapFactor:
    def test_much_better_coverage(self):
        sc = SeedScorer()
        assert sc._bitmap_factor(100, 20) == 3.0

    def test_better_coverage(self):
        sc = SeedScorer()
        assert sc._bitmap_factor(100, 40) == 2.0

    def test_slightly_better(self):
        sc = SeedScorer()
        assert sc._bitmap_factor(100, 60) == 1.5

    def test_average(self):
        sc = SeedScorer()
        assert sc._bitmap_factor(100, 100) == 1.0

    def test_slightly_worse(self):
        sc = SeedScorer()
        assert sc._bitmap_factor(100, 160) == 0.75

    def test_worse(self):
        sc = SeedScorer()
        assert sc._bitmap_factor(100, 250) == 0.50

    def test_much_worse(self):
        sc = SeedScorer()
        assert sc._bitmap_factor(100, 350) == 0.25


class TestDepthFactor:
    def test_shallow(self):
        sc = SeedScorer()
        assert sc._depth_factor(0) == 1.0
        assert sc._depth_factor(3) == 1.0

    def test_medium(self):
        sc = SeedScorer()
        assert sc._depth_factor(4) == 2.0
        assert sc._depth_factor(7) == 2.0

    def test_deep(self):
        sc = SeedScorer()
        assert sc._depth_factor(8) == 3.0
        assert sc._depth_factor(13) == 3.0

    def test_very_deep(self):
        sc = SeedScorer()
        assert sc._depth_factor(14) == 4.0
        assert sc._depth_factor(25) == 4.0

    def test_extremely_deep(self):
        sc = SeedScorer()
        assert sc._depth_factor(26) == 5.0
        assert sc._depth_factor(100) == 5.0


class TestBaseSchedule:
    def test_base_uses_speed_bitmap_depth(self):
        sc = SeedScorer("base")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        assert score == 100.0  # all factors = 1.0


class TestFastSchedule:
    def test_rare_seed_gets_high_energy(self):
        sc = SeedScorer("fast")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        assert score > 100.0  # fast factor > 1

    def test_heavily_fuzzed_gets_low_energy(self):
        sc = SeedScorer("fast")
        score_heavy = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=200,
            n_fuzz=200,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        score_light = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        assert score_heavy < score_light

    def test_favored_bonus(self):
        sc = SeedScorer("fast")
        score_favored = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=100,
            n_fuzz=100,
            total_execs=1000,
            tc_ref=0,
            favored=True,
        )
        score_unfavored = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=100,
            n_fuzz=100,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        assert score_favored > score_unfavored


class TestCoeSchedule:
    def test_coe_skip_above_mean(self):
        sc = SeedScorer("coe")
        assert sc.coe_skip(n_fuzz=64, mean_log_n_fuzz=4.0, favored=False) is True

    def test_coe_skip_below_mean(self):
        sc = SeedScorer("coe")
        assert sc.coe_skip(n_fuzz=8, mean_log_n_fuzz=4.0, favored=False) is False

    def test_coe_skip_favored_not_skipped(self):
        sc = SeedScorer("coe")
        assert sc.coe_skip(n_fuzz=64, mean_log_n_fuzz=4.0, favored=True) is False

    def test_coe_skip_zero_n_fuzz(self):
        sc = SeedScorer("coe")
        assert sc.coe_skip(n_fuzz=0, mean_log_n_fuzz=4.0, favored=False) is False


class TestRareSchedule:
    def test_rare_penalty_for_overfuzzed(self):
        sc = SeedScorer("rare")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=0,
            n_fuzz=900,
            total_execs=1000,
            tc_ref=0,
            favored=False,
            max_depth=0,
        )
        # n_fuzz/total_execs = 0.9, penalty = 0.1
        assert score < 100.0

    def test_rare_bonus_adds_tc_ref(self):
        sc = SeedScorer("rare")
        assert sc.rare_bonus(10) == 100.0
        assert sc.rare_bonus(0) == 0.0

    def test_rare_zero_total_execs(self):
        sc = SeedScorer("rare")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=0,
            tc_ref=0,
            favored=False,
            max_depth=0,
        )
        # Should not crash, returns at least 1.0
        assert score >= 1.0


class TestMoptSchedule:
    def test_mopt_recent_entry_boost(self):
        sc = SeedScorer("mopt")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=25,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=1000,
            tc_ref=0,
            favored=False,
            max_depth=26,
        )
        assert score > 100.0  # depth boost

    def test_mopt_old_entry_no_boost(self):
        sc = SeedScorer("mopt")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=1000,
            tc_ref=0,
            favored=False,
            max_depth=26,
        )
        assert score == 100.0  # no depth boost


class TestLinSchedule:
    def test_lin_factor(self):
        sc = SeedScorer("lin")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=10,
            n_fuzz=5,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        # factor = 10/6 ≈ 1.67
        assert score > 100.0


class TestQuadSchedule:
    def test_quad_factor(self):
        sc = SeedScorer("quad")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=10,
            n_fuzz=5,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        # factor = 100/6 ≈ 16.67
        assert score > 100.0


class TestGoSchedule:
    """AFLGo-style distance-annealed schedule tests."""

    def test_go_factor_near_target(self):
        """Seed near the target gets large bonus at full anneal."""
        sc = SeedScorer("go")
        bonus = sc._go_factor(avg_distance=0.1, max_distance=10.0, anneal_progress=1.0)
        # beta=5, norm_dist=0.01 → exp(5*0.99) ≈ 145, capped at 100
        assert bonus == 100.0

    def test_go_factor_far_target(self):
        """Seed at max distance gets no bonus regardless of anneal."""
        sc = SeedScorer("go")
        bonus = sc._go_factor(avg_distance=10.0, max_distance=10.0, anneal_progress=1.0)
        # beta=5, norm_dist=1.0 → exp(5*0) = 1.0
        assert bonus == pytest.approx(1.0)

    def test_go_factor_exploration_phase(self):
        """No distance bonus during exploration phase (anneal_progress ≈ 0)."""
        sc = SeedScorer("go")
        bonus = sc._go_factor(avg_distance=0.1, max_distance=10.0, anneal_progress=0.0)
        assert bonus == 1.0

    def test_go_factor_no_distance_data(self):
        """Without distance data, factor is 1.0 regardless."""
        sc = SeedScorer("go")
        bonus = sc._go_factor(avg_distance=0.0, max_distance=0.0, anneal_progress=1.0)
        assert bonus == 1.0

    def test_go_factor_not_go_schedule(self):
        """_go_factor returns 1.0 when schedule is not 'go'."""
        sc = SeedScorer("base")
        bonus = sc._go_factor(avg_distance=0.1, max_distance=10.0, anneal_progress=1.0)
        assert bonus == 1.0

    def test_go_score_integration(self):
        """Full score() call with go schedule and distance info."""
        sc = SeedScorer("go")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
            tc_ref=0,
            avg_distance=0.5,
            max_distance=10.0,
            anneal_progress=1.0,
        )
        # norm_dist=0.05, beta=5 → exp(5*0.95) ≈ 117, capped 100
        # base=100 * speed(1.0) * bitmap(1.0) * handicap(1.0) * depth(1.0) * go(100) = 10000, clamped to 1600
        assert 100.0 <= score <= 1600.0


class TestAflgoSchedule:
    """Exact AFLGo power-schedule tests (afl-fuzz.c calculate_score).

    Reference formulas (hand-evaluated literals below):
      T = 1/20^progress (exp), 1/(1+19·progress) (lin),
          1/(1+19·progress²) (quad),
          1/(1+2·ln(1+progress·13358.7268297)) (log)
      p = (1−nd)(1−T) + 0.5T,  factor = 2^(2·log2(32)·(p−0.5))
    """

    def test_uniform_energy_at_start(self):
        """T=1 at progress=0 → p=0.5 → factor=1.0 for any distance."""
        sc = SeedScorer("aflgo")
        for d, mn, mx in [(0.1, 0.0, 10.0), (9.9, 0.0, 10.0)]:
            f = sc._aflgo_factor(d, mx, mn, elapsed_sec=0.0, t_x_minutes=60.0, cooling="exp")
            assert f == pytest.approx(1.0)

    def test_near_target_late_campaign(self):
        """T≈0, nd=0 → p=1 → factor=2^5=32; nd=1 → factor=2^-5=1/32."""
        sc = SeedScorer("aflgo")
        late = 60.0 * 60.0 * 60.0  # 60× the t_x window → T≈0
        near = sc._aflgo_factor(0.0, 10.0, 0.0, late, 60.0, "exp")
        far = sc._aflgo_factor(10.0, 10.0, 0.0, late, 60.0, "exp")
        assert near == pytest.approx(32.0)
        assert far == pytest.approx(1.0 / 32.0)

    def test_mid_campaign_exact_value(self):
        """exp cooling at progress=1: T=0.05, nd=0 → factor=2^4.75."""
        sc = SeedScorer("aflgo")
        f = sc._aflgo_factor(0.0, 10.0, 0.0, elapsed_sec=3600.0, t_x_minutes=60.0, cooling="exp")
        assert f == pytest.approx(26.908685288118864)

    def test_symmetric_factor_at_nd_half(self):
        """nd=0.5 at T=0.05 → p=0.5 → factor=1.0."""
        sc = SeedScorer("aflgo")
        f = sc._aflgo_factor(5.0, 10.0, 0.0, elapsed_sec=3600.0, t_x_minutes=60.0, cooling="exp")
        assert f == pytest.approx(1.0)

    def test_cooling_temperature_variants(self):
        """All four cooling schedules at progress=0.5 (hand-evaluated)."""
        sc = SeedScorer("aflgo")
        assert sc._cooling_temperature(0.5, "exp") == pytest.approx(0.22360679774997896)
        assert sc._cooling_temperature(0.5, "lin") == pytest.approx(0.09523809523809523)
        assert sc._cooling_temperature(0.5, "quad") == pytest.approx(0.17391304347826086)
        assert sc._cooling_temperature(0.5, "log") == pytest.approx(0.05372342171453818)
        # negative progress is clamped to 0
        assert sc._cooling_temperature(-1.0, "exp") == pytest.approx(1.0)

    def test_no_distance_data(self):
        sc = SeedScorer("aflgo")
        assert sc._aflgo_factor(-1.0, 10.0, 0.0, 3600.0, 60.0, "exp") == 1.0
        assert sc._aflgo_factor(5.0, 0.0, 0.0, 3600.0, 60.0, "exp") == 1.0

    def test_min_distance_normalization(self):
        """normalized_d uses (d−min)/(max−min); min==max → nd=0."""
        sc = SeedScorer("aflgo")
        late = 60.0 * 60.0 * 60.0
        # d == min → nd=0 → max boost
        assert sc._aflgo_factor(5.0, 10.0, 5.0, late, 60.0, "exp") == pytest.approx(32.0)
        # d == max → nd=1 → 1/32
        assert sc._aflgo_factor(10.0, 10.0, 5.0, late, 60.0, "exp") == pytest.approx(1.0 / 32.0)
        # min == max → nd=0 → max boost
        assert sc._aflgo_factor(7.0, 7.0, 7.0, late, 60.0, "exp") == pytest.approx(32.0)

    def test_aflgo_score_integration(self):
        """Full score() path with the aflgo schedule applies the factor."""
        sc = SeedScorer("aflgo")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=1,
            n_fuzz=1,
            total_execs=100,
            tc_ref=0,
            avg_distance=0.0,
            max_distance=10.0,
            min_distance=0.0,
            elapsed_sec=3600.0 * 60,
            t_x_minutes=60.0,
        )
        # base 100 · 32 (aflgo) = 3200, clamped to max_mult*100 = 1600
        assert score == pytest.approx(1600.0)

    def test_unknown_cooling_rejected(self):
        with pytest.raises(ValueError):
            SeedScorer("aflgo", aflgo_cooling="cubic")


class TestHandicap:
    def test_handicap_0(self):
        sc = SeedScorer("base")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=0,
            depth=0,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        assert score == 100.0

    def test_handicap_1(self):
        sc = SeedScorer("base")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=1,
            depth=0,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        assert score == 200.0  # 2x

    def test_handicap_4(self):
        sc = SeedScorer("base")
        score = sc.score(
            exec_us=100,
            avg_exec_us=100,
            bitmap_size=50,
            avg_bitmap_size=50,
            handicap=4,
            depth=0,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        assert score == 400.0  # 4x


class TestClamping:
    def test_minimum_is_1(self):
        sc = SeedScorer("base", max_mult=16)
        score = sc.score(
            exec_us=10000,
            avg_exec_us=1,
            bitmap_size=1,
            avg_bitmap_size=1000,
            handicap=0,
            depth=0,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        assert score >= 1.0

    def test_maximum_is_max_mult_times_100(self):
        sc = SeedScorer("base", max_mult=16)
        score = sc.score(
            exec_us=1,
            avg_exec_us=10000,
            bitmap_size=1000,
            avg_bitmap_size=1,
            handicap=10,
            depth=30,
            fuzz_level=0,
            n_fuzz=0,
            total_execs=1000,
            tc_ref=0,
            favored=False,
        )
        assert score <= 1600.0


class TestComputeMeanLogNFuzz:
    def test_empty_list(self):
        assert compute_mean_log_n_fuzz([]) == 0.0

    def test_all_zeros(self):
        assert compute_mean_log_n_fuzz([0, 0, 0]) == 0.0

    def test_single_value(self):
        assert compute_mean_log_n_fuzz([8]) == pytest.approx(math.log2(8))

    def test_multiple_values(self):
        vals = [2, 4, 8]
        expected = (math.log2(2) + math.log2(4) + math.log2(8)) / 3
        assert compute_mean_log_n_fuzz(vals) == pytest.approx(expected)

    def test_mixed_zeros_and_positive(self):
        vals = [0, 0, 4, 16]
        expected = (math.log2(4) + math.log2(16)) / 2
        assert compute_mean_log_n_fuzz(vals) == pytest.approx(expected)
