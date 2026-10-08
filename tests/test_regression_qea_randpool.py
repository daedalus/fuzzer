"""Regression: QEA drew from global ``np.random`` instead of its RandPool.

collapse(), mutate_amplitudes() and collapse_correlated() used the legacy
global numpy stream (Hard Rule 16). That stream is in no RandPool snapshot,
so a resumed ``--qea`` run diverged from an uninterrupted one, and any
unrelated ``np.random`` consumer shifted QEA's draws.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from fuzzer_tool.core.qea import (
    QEALifecycle,
    _uniform_amplitudes,
    _zero_coupling,
    collapse,
    collapse_correlated,
    mutate_amplitudes,
)
from fuzzer_tool.core.rand_pool import RandPool

POOL_SEED = 1234
OTHER_POOL_SEED = 4321
GLOBAL_SEEDS = (1, 99999)
CORPUS = [b"aaaaaaaa", b"bbbbbbbb", b"cccccccc", b"dddddddd"]
STEPS = 40
N_BITS = 256


def _edge_tracker():
    """Minimal EdgeTracker stub with the attributes QEA touches."""
    et = MagicMock()
    et.cumulative_edges = set(range(8))
    et.seed_edges = {}
    et.compute_wasserstein_weight.return_value = 1.0
    return et


def _run(pool_seed: int, global_seed: int, correlation: bool) -> tuple:
    """Drive a full QEA lifecycle (collapse + evolve + mutate); return its trace."""
    np.random.seed(global_seed)
    qea = QEALifecycle(
        pop_size=4,
        generation_size=3,
        mutation_prob=0.5,
        use_correlation=correlation,
        rng=RandPool(seed=pool_seed),
    )
    et = _edge_tracker()
    qea.initialize(list(CORPUS), et)

    # Interleave draws: global perturbation between steps must not leak in.
    seeds = []
    for step in range(STEPS):
        seeds.append(qea.pick_seed())
        np.random.random(step + 1)
        qea.on_fuzz_result(seeds[-1], step % 7 == 0, step % 5, et)

    amps = tuple(ind.amplitudes.tobytes() for ind in qea.population)
    return tuple(seeds), amps


@pytest.mark.parametrize("correlation", [False, True])
class TestLifecycle:
    def test_control_same_seed_identical(self, correlation):
        # Hard Rule 46: the oracle must pass on identical inputs first.
        a = _run(POOL_SEED, GLOBAL_SEEDS[0], correlation)
        b = _run(POOL_SEED, GLOBAL_SEEDS[0], correlation)
        assert a == b

    def test_regression_global_numpy_does_not_leak(self, correlation):
        a = _run(POOL_SEED, GLOBAL_SEEDS[0], correlation)
        b = _run(POOL_SEED, GLOBAL_SEEDS[1], correlation)
        assert a == b

    def test_falsification_pool_seed_moves_output(self, correlation):
        a = _run(POOL_SEED, GLOBAL_SEEDS[0], correlation)
        b = _run(OTHER_POOL_SEED, GLOBAL_SEEDS[0], correlation)
        assert a != b


class TestPrimitives:
    def _amps(self):
        return _uniform_amplitudes(N_BITS)

    def test_collapse_ignores_global(self):
        np.random.seed(GLOBAL_SEEDS[0])
        a = collapse(self._amps(), rng=RandPool(seed=POOL_SEED))
        np.random.seed(GLOBAL_SEEDS[1])
        b = collapse(self._amps(), rng=RandPool(seed=POOL_SEED))
        assert a == b

    def test_collapse_matches_pool_draws(self):
        # Expected side derived from the pool's raw uniforms, not collapse().
        amps = self._amps()
        u = RandPool(seed=POOL_SEED).random_array(N_BITS)
        expected = np.packbits((u >= amps * amps).astype(np.uint8)).tobytes()
        assert collapse(amps, rng=RandPool(seed=POOL_SEED)) == expected

    def test_mutate_ignores_global(self):
        np.random.seed(GLOBAL_SEEDS[0])
        a = mutate_amplitudes(self._amps(), prob=0.5, rng=RandPool(seed=POOL_SEED))
        np.random.seed(GLOBAL_SEEDS[1])
        b = mutate_amplitudes(self._amps(), prob=0.5, rng=RandPool(seed=POOL_SEED))
        np.testing.assert_array_equal(a, b)

    def test_correlated_ignores_global(self):
        coupling = _zero_coupling(N_BITS // 8) + 0.3
        np.random.seed(GLOBAL_SEEDS[0])
        a = collapse_correlated(self._amps(), coupling, rng=RandPool(seed=POOL_SEED))
        np.random.seed(GLOBAL_SEEDS[1])
        b = collapse_correlated(self._amps(), coupling, rng=RandPool(seed=POOL_SEED))
        assert a == b

    def test_global_state_untouched(self):
        # Adversarial: no QEA primitive may advance the global stream.
        np.random.seed(GLOBAL_SEEDS[0])
        _, before_key, before_pos, *_ = np.random.get_state()
        rng = RandPool(seed=POOL_SEED)
        collapse(self._amps(), rng=rng)
        mutate_amplitudes(self._amps(), prob=1.0, rng=rng)
        collapse_correlated(self._amps(), _zero_coupling(N_BITS // 8), rng=rng)
        _, after_key, after_pos, *_ = np.random.get_state()
        np.testing.assert_array_equal(before_key, after_key)
        assert after_pos == before_pos


class TestAdversarial:
    def test_zero_length(self):
        rng = RandPool(seed=POOL_SEED)
        empty = np.zeros(0, dtype=np.float64)
        assert collapse(empty, rng=rng) == b""
        assert collapse_correlated(empty, _zero_coupling(0), rng=rng) == b""
        assert len(mutate_amplitudes(empty, prob=1.0, rng=rng)) == 0

    @pytest.mark.parametrize(("alpha", "byte"), [(1.0, 0x00), (0.0, 0xFF)])
    def test_certain_bits(self, alpha, byte):
        # α=1 -> P(bit=0)=1; α=0 -> P(bit=1)=1, for every pool seed.
        amps = np.full(N_BITS, alpha, dtype=np.float64)
        for seed in (POOL_SEED, OTHER_POOL_SEED):
            out = collapse(amps, rng=RandPool(seed=seed))
            assert out == bytes([byte]) * (N_BITS // 8)

    def test_mutate_prob_zero_is_identity(self):
        amps = _uniform_amplitudes(N_BITS)
        ref = amps.copy()
        mutate_amplitudes(amps, prob=0.0, rng=RandPool(seed=POOL_SEED))
        np.testing.assert_array_equal(amps, ref)

    def test_mutate_prob_one_stays_in_bounds(self):
        lo, hi = 0.2, 0.3
        amps = mutate_amplitudes(
            _uniform_amplitudes(N_BITS),
            prob=1.0,
            alpha_min=lo,
            alpha_max=hi,
            rng=RandPool(seed=POOL_SEED),
        )
        assert ((amps >= lo) & (amps < hi)).all()
