"""Regression: CEM Dirichlet concentration was learned in the wrong direction.

`_learn_cem_concentration` mapped low elite entropy (structured data) to a
*larger* α. In the posterior predictive ``(n + α) / (N + 256α)`` a larger α
pulls toward uniform, so structured positions were smoothed toward random
bytes. α is now the Dirichlet-Multinomial MLE (`core.dirichlet.dm_alpha`).
"""

import numpy as np

from fuzzer_tool.core.dirichlet import dm_alpha
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.op_monte_carlo import MonteCarloScheduler

BYTE_VALUES = 256
N_ELITE = 20
INPUT_LEN = 16
CONCENTRATION = 1.0


def _fitted(elite: list[bytes]) -> MonteCarloScheduler:
    mc = MonteCarloScheduler(
        elite_frac=1.0, cem_dirichlet_concentration=CONCENTRATION, rng=RandPool(seed=0)
    )
    for i, data in enumerate(elite):
        mc.add_elite(data, score=i)
    mc.maybe_refit()
    return mc


def _predictive(mc: MonteCarloScheduler, pos: int, byte: int) -> float:
    freq = mc.byte_freq[pos]
    alpha = mc._cem_alpha()
    return (freq.get(byte, 0) + alpha) / (sum(freq.values()) + BYTE_VALUES * alpha)


def _random_elite(seed: int) -> list[bytes]:
    g = np.random.default_rng(seed)
    return [bytes(g.integers(0, BYTE_VALUES, INPUT_LEN, dtype=np.uint8)) for _ in range(N_ELITE)]


def test_regression_cem_concentration_direction():
    """Constant elite bytes must keep nearly all predictive mass."""
    mc = _fitted([b"MAGIC\x00\x01\x02" * 2] * N_ELITE)
    assert _predictive(mc, 0, ord("M")) > 0.9


def test_structured_alpha_below_random_alpha():
    """Falsification: structured elite ⇒ strictly smaller α than random elite."""
    structured = _fitted([b"MAGIC\x00\x01\x02" * 2] * N_ELITE)._cem_alpha()
    random_ = _fitted(_random_elite(seed=1))._cem_alpha()
    assert structured < random_


def test_alpha_equals_dm_mle_of_byte_freq():
    """α is the MLE computed independently from the fitted per-position counts."""
    mc = _fitted(_random_elite(seed=2)[:10] + [b"\x7fELF" * 4] * 10)
    rows = [list(freq.values()) for freq in mc.byte_freq.values()]
    assert mc._cem_alpha() == dm_alpha(rows, BYTE_VALUES, CONCENTRATION)


def test_adversarial_single_elite_keeps_finite_alpha():
    """One elite input: every position is a single observation (no info on α)."""
    mc = _fitted([b"A" * INPUT_LEN])
    assert 0.0 < mc._cem_alpha() < float("inf")
    assert 0 <= mc.cem_byte(0) <= 255


def test_disabled_keeps_laplace():
    """concentration 0 (default) keeps the old add-1 behaviour."""
    mc = MonteCarloScheduler(elite_frac=1.0, rng=RandPool(seed=0))
    for i in range(N_ELITE):
        mc.add_elite(b"A" * INPUT_LEN, score=i)
    mc.maybe_refit()
    assert mc._cem_alpha() == 1.0
