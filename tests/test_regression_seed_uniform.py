"""Regression tests for the seed_uniform scheduler unit behavior."""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_uniform import SeedUniformScheduler


class TestSeedUniformScheduler:
    def test_seed_uniform_scheduler_exists_and_init(self):
        scheduler = SeedUniformScheduler()
        assert hasattr(scheduler, "select_seed")
        assert hasattr(scheduler, "record")
        assert hasattr(scheduler, "bandit_stats")
        assert hasattr(scheduler, "init_arm")
        assert hasattr(scheduler, "supports_priors")
        assert scheduler.supports_priors is False

    def test_seed_uniform_select_seed_uniform_random(self):
        scheduler = SeedUniformScheduler()
        scheduler.init_arm("seed_a")
        scheduler.init_arm("seed_b")
        scheduler.init_arm("seed_c")

        scheduler._rng = RandPool(seed=42)
        seed_ids = ["seed_a", "seed_b", "seed_c"]

        results = [scheduler.select_seed(seed_ids) for _ in range(10)]
        unique_seeds = set(results)
        assert "seed_a" in unique_seeds
        assert "seed_b" in unique_seeds
        assert "seed_c" in unique_seeds

    def test_seed_uniform_record_and_stats(self):
        scheduler = SeedUniformScheduler()
        scheduler.init_arm("seed_a")
        scheduler.init_arm("seed_b")

        scheduler.record("seed_a", True, 1.0)
        scheduler.record("seed_a", False, 1.0)
        scheduler.record("seed_b", True, 1.0)

        stats = scheduler.bandit_stats()
        assert stats["seed_a"] == (1.0, 1.0)
        assert stats["seed_b"] == (1.0, 0)

    def test_seed_uniform_empty_and_single(self):
        scheduler = SeedUniformScheduler()

        assert scheduler.select_seed([]) == ""
        assert scheduler.select_seed(["only"]) == "only"
        assert scheduler.select_seed(["only"]) == "only"

    def test_seed_uniform_register_on_fly(self):
        scheduler = SeedUniformScheduler()
        scheduler._rng = RandPool(seed=42)
        seed_ids = ["seed_x", "seed_y"]
        result = scheduler.select_seed(seed_ids)
        assert result in seed_ids
        assert "seed_x" in scheduler._seed_counts
        assert "seed_y" in scheduler._seed_counts
