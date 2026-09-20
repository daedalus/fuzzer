"""Tests for ResidualSeedScheduler (core/schedulers/seed_residual.py)."""

import numpy as np
import pytest

from fuzzer_tool.core.edge_matrix import MatrixSubstrate
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers import seed_residual as sr
from fuzzer_tool.core.schedulers.seed_residual import ResidualSeedScheduler


@pytest.fixture(autouse=True)
def _instrumented_target(monkeypatch):
    """coverage_trust returns early for a target-less run, so the gate tests need one."""
    monkeypatch.setattr("fuzzer_tool.core.elf.sancov_guard_status", lambda _t: "present")


class _Tracker:
    def __init__(self, profiles):
        self.seed_hit_counts = profiles
        self.seed_edges = {k: set(v) for k, v in profiles.items()}
        self.cumulative_edges = set().union(*self.seed_edges.values())


class _Rng:
    """Scripted RNG: exhausting an iterator is the tripwire (Hard Rule 39)."""

    def __init__(self, randoms=(), randranges=()):
        self._r, self._rr = iter(randoms), iter(randranges)

    def random(self):
        return next(self._r)

    def randrange(self, n):
        return next(self._rr)


def _fit(profiles, **kw):
    sub = MatrixSubstrate(target="/x", **kw)
    sub.maybe_refit(_Tracker(profiles), 0)
    return sub


# s0..s4 share one class (edges 1-3 carry the same count in a seed) at growing volume;
# s5 is the low-volume seed with a private class.
VOLUME = {f"s{i}": {1: 100 + i, 2: 100 + i, 3: 100 + i} for i in range(5)}
VOLUME["s5"] = {9: 1}

# Strictly nested: mass == degree == rank(total) for every seed, i.e. a row sum.
ROW_SUM = {f"s{i}": {10 * i + j: j + 1 for j in range(i + 1)} for i in range(5)}


class TestConstruction:
    def test_requires_rng_and_substrate(self):
        with pytest.raises(ValueError):
            ResidualSeedScheduler(rng=None, substrate=MatrixSubstrate())
        with pytest.raises(ValueError):
            ResidualSeedScheduler(rng=RandPool(seed=1), substrate=None)


class TestScore:
    def test_low_volume_seed_with_private_class_outranks_the_high_volume_crowd(self):
        # Hand-derived residuals: s5 +1.43, s4 +0.57, ..., s0 -1.14. Raw volume alone
        # would have ranked s5 last.
        s = ResidualSeedScheduler(RandPool(seed=1), _fit(VOLUME))
        energies = {k: s.seed_energy(k) for k in VOLUME}
        assert max(energies, key=energies.get) == "s5"
        assert energies["s5"] > energies["s4"] > energies["s0"]

    def test_a_row_sum_scores_flat(self):
        # The Tang failure mode: score == volume in disguise. The residual removes all
        # of it, and the falsification monitor reports it.
        s = ResidualSeedScheduler(RandPool(seed=1), _fit(ROW_SUM))
        e = [s.seed_energy(k) for k in ROW_SUM]
        assert max(e) == pytest.approx(min(e))
        f = s.falsification()
        assert f["rho_total"] == 0.0 and f["rho_degree"] == 0.0

    def test_energy_never_reaches_zero(self):
        s = ResidualSeedScheduler(RandPool(seed=1), _fit(VOLUME))
        assert min(s.seed_energy(k) for k in VOLUME) >= sr.EPS

    def test_unknown_seed_gets_the_median(self):
        s = ResidualSeedScheduler(RandPool(seed=1), _fit(VOLUME))
        assert s.seed_energy("new") == pytest.approx(
            float(np.median([s.seed_energy(k) for k in VOLUME]))
        )

    def test_a_chain_counts_once_in_the_mass(self):
        # The fold, not the arm, is doing this; assert it reaches the score. Two seeds
        # cover the same 3-edge chain; a third has one private edge.
        profiles = {"a": {1: 1, 2: 1, 3: 1}, "b": {1: 1, 2: 1, 3: 1}, "c": {7: 1}}
        sub = _fit(profiles)
        assert sub.fold.mass[0] == pytest.approx(0.5)


class TestFalsification:
    def _corpus(self):
        rng = np.random.default_rng(0)
        profiles = {}
        for i in range(12):
            edges = rng.choice(40, size=int(rng.integers(3, 15)), replace=False)
            profiles[f"s{i}"] = {int(e): int(rng.integers(1, 9)) for e in edges}
        return profiles

    def test_reports_partials_only_with_enough_outcomes(self):
        s = ResidualSeedScheduler(RandPool(seed=1), _fit(self._corpus()))
        assert "partial_total" not in s.falsification({"s0": 1.0})
        f = s.falsification({f"s{i}": float(i % 4) for i in range(12)})
        assert f["outcomes"] == 12.0
        assert -1.0 <= f["partial_total"] <= 1.0 and -1.0 <= f["partial_degree"] <= 1.0

    def test_outcome_fn_is_logged_at_refit_and_never_breaks_scoring(self, caplog):
        sub = _fit(self._corpus())
        s = ResidualSeedScheduler(
            RandPool(seed=1), sub, outcome_fn=lambda: {f"s{i}": float(i) for i in range(12)}
        )
        with caplog.at_level("INFO"):
            s._rescore()
        assert s.last_falsification and "falsification" in caplog.text

        def boom():
            raise RuntimeError("diagnostic bug")

        s2 = ResidualSeedScheduler(RandPool(seed=1), sub, outcome_fn=boom)
        assert s2.seed_energy("s0") > 0  # the exception was swallowed

    def test_unfitted_is_empty(self):
        assert ResidualSeedScheduler(RandPool(seed=1), MatrixSubstrate()).falsification() == {}


class TestSelection:
    def test_abstains_when_unfitted(self):
        s = ResidualSeedScheduler(RandPool(seed=1), MatrixSubstrate())
        assert s.select_index(["a"]) is None

    def test_abstains_when_the_gate_is_closed(self):
        sub = _fit(VOLUME)
        sub.set_stability(0.5)
        s = ResidualSeedScheduler(RandPool(seed=1), sub)
        assert not s.available() and s.select_index(list(VOLUME)) is None

    def test_exploit_draw_lands_by_cumulative_energy(self):
        keys = list(VOLUME)
        sub = _fit(VOLUME)
        assert ResidualSeedScheduler(_Rng(randoms=[0.99, 0.0]), sub).select_index(keys) == 0
        assert (
            ResidualSeedScheduler(_Rng(randoms=[0.99, 0.999999]), sub).select_index(keys)
            == len(keys) - 1
        )

    def test_explore_draw_is_a_uniform_index(self):
        s = ResidualSeedScheduler(_Rng(randoms=[0.0], randranges=[3]), _fit(VOLUME))
        assert s.select_index(list(VOLUME)) == 3

    @pytest.mark.parametrize("seed", [7, 8])
    def test_empirical_frequency_follows_energy(self, seed):
        # Hard Rule 46: two independent seeded runs of the sampler against the exact
        # probabilities, so a broken oracle shows up as the two disagreeing.
        keys = list(VOLUME)
        s = ResidualSeedScheduler(RandPool(seed=seed), _fit(VOLUME))
        n, counts = 20000, [0] * len(keys)
        for _ in range(n):
            counts[s.select_index(keys)] += 1
        w = s.weights(keys)
        for i in range(len(keys)):
            p = sr.EXPLORE_BASE / len(keys) + (1 - sr.EXPLORE_BASE) * w[i] / w.sum()
            assert counts[i] / n == pytest.approx(p, abs=0.02)


class TestStats:
    def test_fitted(self):
        st = ResidualSeedScheduler(RandPool(seed=1), _fit(VOLUME)).stats()
        assert st["gate"] == "unverified" and st["fitted"] and st["seeds"] == 6
        assert st["duplicate_edges"] == 2  # edges 1-3 fold to one class

    def test_unfitted_and_distrusted(self):
        st = ResidualSeedScheduler(RandPool(seed=1), MatrixSubstrate()).stats()
        assert st["fitted"] is False and st["skip_reason"]
        sub = MatrixSubstrate(target="/x")
        sub.set_stability(0.1)
        assert "distrust_reason" in ResidualSeedScheduler(RandPool(seed=1), sub).stats()
