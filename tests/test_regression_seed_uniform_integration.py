"""Integration tests for seed_uniform scheduler wiring."""

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_uniform import SeedUniformScheduler


def test_seed_uniform_selects_different_seeds():
    """seed_uniform should pick different seeds over multiple calls."""
    scheduler = SeedUniformScheduler(rng=RandPool(seed=42))
    seed_ids = ["seed_a", "seed_b", "seed_c"]

    results = [scheduler.select_seed(seed_ids) for _ in range(20)]
    unique_seeds = set(results)
    assert len(unique_seeds) >= 2, f"Expected at least 2 unique seeds, got {unique_seeds}"


def test_seed_uniform_single_and_empty():
    """seed_uniform handles single element and empty lists."""
    scheduler = SeedUniformScheduler()
    assert scheduler.select_seed([]) == ""
    assert scheduler.select_seed(["only"]) == "only"
    assert scheduler.select_seed(["only"]) == "only"


def test_seed_uniform_flag_wired_in_cli():
    """seed_uniform_scheduler flag should be in hail mary flags."""
    from fuzzer_tool.cli import commands

    assert "seed_uniform_scheduler" in commands._HAIL_MARY_FLAGS


def test_seed_uniform_scheduler_wired_in_init():
    """seed_uniform_scheduler init should be wired in fuzzer __init__."""
    import inspect

    from fuzzer_tool.services.fuzzer import Fuzzer

    sig = inspect.signature(Fuzzer.__init__)
    assert "seed_uniform_scheduler" in sig.parameters
