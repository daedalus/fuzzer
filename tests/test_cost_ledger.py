"""Tests for the per-seed cost ledger (core/cost_ledger.py).

The ledger exists because ``meta["total_time"]`` was accumulated and only ever
divided by ``fuzz_count``, which is the wrong denominator twice over: the
initial seed replay in ``Fuzzer.run`` bumps the count with no time credited,
and ``total_time`` was absent from the persisted corpus state while
``fuzz_count`` was not.  Both make a seed read as cheaper than it is, and the
readers floor at 1 microsecond, so "unmeasured" and "free" were the same value.
"""

import math

import pytest

from fuzzer_tool.core.cost_ledger import (
    cost_samples,
    effective_fuzz_count,
    seed_exec_time,
    seed_exec_us,
)


class TestCostSamples:
    def test_absent_key_is_zero(self):
        assert cost_samples({}) == 0
        assert cost_samples({"fuzz_count": 400}) == 0

    def test_negative_and_garbage_clamp_to_zero(self):
        assert cost_samples({"cost_samples": -3}) == 0
        assert cost_samples({"cost_samples": None}) == 0
        assert cost_samples({"cost_samples": "many"}) == 0

    def test_counts_are_returned(self):
        assert cost_samples({"cost_samples": 7}) == 7


class TestSeedExecTime:
    def test_mean_over_samples_not_fuzz_count(self):
        # 10 timed executions totalling 1 s, but 50 fuzzes on the counter.
        # Dividing by fuzz_count would give 20 ms; the truth is 100 ms.
        meta = {"total_time": 1.0, "cost_samples": 10, "fuzz_count": 50}
        assert seed_exec_time(meta, 0.5) == pytest.approx(0.1)

    def test_no_samples_yields_the_fallback(self):
        # This is the resume case: a restored seed carries a count and no time.
        meta = {"total_time": 0.0, "cost_samples": 0, "fuzz_count": 400}
        assert seed_exec_time(meta, 0.002) == pytest.approx(0.002)

    def test_legacy_meta_without_the_key_yields_the_fallback(self):
        assert seed_exec_time({"total_time": 0.0, "fuzz_count": 99}, 0.003) == pytest.approx(0.003)

    def test_samples_with_zero_time_yields_the_fallback(self):
        # A clock that never advanced is not evidence of a free seed.
        meta = {"total_time": 0.0, "cost_samples": 5}
        assert seed_exec_time(meta, 0.004) == pytest.approx(0.004)


class TestSeedExecUs:
    def test_converts_to_microseconds(self):
        meta = {"total_time": 0.5, "cost_samples": 5}
        assert seed_exec_us(meta, 1.0) == pytest.approx(100_000.0)

    def test_unmeasured_seed_gets_the_corpus_mean_not_the_floor(self):
        # The regression this module exists for: before, an unmeasured seed
        # produced 0/large -> 0 -> floored to 1.0us, making it the cheapest
        # seed in the corpus and pushing it into the favored set on every
        # resume. It must now read as typical, not as free.
        meta = {"total_time": 0.0, "cost_samples": 0, "fuzz_count": 400}
        assert seed_exec_us(meta, 160.0) == pytest.approx(160.0)
        assert seed_exec_us(meta, 160.0) > 1.0

    def test_floor_still_applies(self):
        meta = {"total_time": 1e-12, "cost_samples": 1000}
        assert seed_exec_us(meta, 50.0) == 1.0


class TestEffectiveFuzzCount:
    def test_equals_sample_count_under_uniform_cost(self):
        # The falsification condition, as a test: where per-execution cost
        # does not vary, the cost form and the count form are the same number,
        # so every consumer built on it is a no-op there.
        mean = 0.002
        meta = {"total_time": 20 * mean, "cost_samples": 20}
        assert effective_fuzz_count(meta, mean) == pytest.approx(20.0)

    def test_expensive_seed_reads_higher_than_its_count(self):
        mean = 0.002
        meta = {"total_time": 10 * 0.020, "cost_samples": 10}
        assert effective_fuzz_count(meta, mean) == pytest.approx(100.0)

    def test_cheap_seed_reads_lower_than_its_count(self):
        mean = 0.002
        meta = {"total_time": 100 * 0.0002, "cost_samples": 100}
        assert effective_fuzz_count(meta, mean) == pytest.approx(10.0)

    def test_falls_back_to_fuzz_count_without_samples(self):
        meta = {"total_time": 0.0, "cost_samples": 0, "fuzz_count": 33}
        assert effective_fuzz_count(meta, 0.002) == pytest.approx(33.0)

    def test_falls_back_to_fuzz_count_before_any_mean_exists(self):
        meta = {"total_time": 1.0, "cost_samples": 5, "fuzz_count": 5}
        assert effective_fuzz_count(meta, 0.0) == pytest.approx(5.0)

    def test_empty_meta_is_zero(self):
        assert effective_fuzz_count({}, 0.002) == pytest.approx(0.0)


class TestStaleCriterion:
    """The §1a consumer: futility measured in budget, not in picks."""

    def _stale(self, metas, mean):
        from fuzzer_tool.services.stats import STALE_SEED_EXEC_EQUIVALENTS

        return [
            i
            for i, m in enumerate(metas)
            if effective_fuzz_count(m, mean) >= STALE_SEED_EXEC_EQUIVALENTS
            and m.get("coverage_edges", 0) == 0
        ]

    def test_agrees_with_the_count_criterion_under_uniform_cost(self):
        mean = 0.001
        metas = [
            {"total_time": n * mean, "cost_samples": n, "fuzz_count": n, "coverage_edges": 0}
            for n in (10, 49, 50, 200)
        ]
        assert self._stale(metas, mean) == [2, 3]

    def test_expensive_seed_is_stale_before_fifty_picks(self):
        mean = 0.001
        # 10 picks, but each cost 20x the corpus mean: 200 execs of budget.
        meta = {"total_time": 10 * 0.020, "cost_samples": 10, "fuzz_count": 10, "coverage_edges": 0}
        assert self._stale([meta], mean) == [0]

    def test_cheap_seed_is_not_stale_at_fifty_picks(self):
        mean = 0.001
        meta = {
            "total_time": 50 * 0.00001,
            "cost_samples": 50,
            "fuzz_count": 50,
            "coverage_edges": 0,
        }
        assert self._stale([meta], mean) == []

    def test_productive_seed_is_never_stale(self):
        mean = 0.001
        meta = {
            "total_time": 500 * mean,
            "cost_samples": 500,
            "fuzz_count": 500,
            "coverage_edges": 3,
        }
        assert self._stale([meta], mean) == []


class TestBoltzmannEnergy:
    """The §1b consumer: energy in execution units, never in raw seconds."""

    def _weight(self, meta, mean, T):
        n = max(effective_fuzz_count(meta, mean), 1.0)
        return math.exp(-math.log(n + 1) / T)

    def test_identical_to_the_count_form_under_uniform_cost(self):
        mean = 0.002
        T = 1.5
        for n in (1, 5, 50, 400):
            meta = {"total_time": n * mean, "cost_samples": n, "fuzz_count": n}
            assert self._weight(meta, mean, T) == pytest.approx(math.exp(-math.log(n + 1) / T))

    def test_expensive_seed_is_down_weighted_relative_to_a_cheap_one(self):
        mean = 0.002
        T = 1.5
        cheap = {"total_time": 20 * 0.0002, "cost_samples": 20, "fuzz_count": 20}
        pricey = {"total_time": 20 * 0.020, "cost_samples": 20, "fuzz_count": 20}
        # Same fuzz_count, so the old energy term gave these equal weight.
        assert self._weight(cheap, mean, T) > self._weight(pricey, mean, T)

    def test_energy_does_not_collapse_on_a_short_campaign(self):
        # Raw seconds through log(x+1) would put every seed at E ~ 0 and turn
        # the arm uniform without T changing. Execution units do not.
        mean = 0.0002
        T = 1.0
        light = {"total_time": 2 * mean, "cost_samples": 2, "fuzz_count": 2}
        heavy = {"total_time": 300 * mean, "cost_samples": 300, "fuzz_count": 300}
        assert self._weight(light, mean, T) / self._weight(heavy, mean, T) > 50
