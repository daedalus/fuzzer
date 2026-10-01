"""Regression: Markov plateau compared JS divergence against a KS critical value.

``ks_significance_threshold(n) = 1.358 / sqrt(n)`` bounds a sup-CDF distance,
not a JS divergence. Under a stationary source the JS change between
snapshots shrinks like ``1 / n``, so at n = 20k bytes the KS threshold sat
~1000x above the noise floor and a real distribution shift was reported as a
plateau. The threshold is now the null of the measured quantity: per context,
adding m draws to n gives E[JS] = (K - 1) m / (8 n N), N = n + m, and the
contexts' mean JS is compared against null mean + z95 * null sd.
"""

from __future__ import annotations

import math
import random

import pytest

from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble

Z95 = 1.6448536269514722
BATCH = 200
ALPHABET = b"ABCD"


def _chain(order: int = 0) -> MarkovChain:
    chain = MarkovChain(order=order)
    chain._snapshot_interval = 1
    return chain


def _uniform(rng: random.Random) -> bytes:
    return bytes(rng.choice(ALPHABET) for _ in range(BATCH))


def _warm(chain: MarkovChain, rng: random.Random, batches: int) -> list[bool]:
    flags = []
    for _ in range(batches):
        chain.train(_uniform(rng))
        flags.append(chain.snapshot_and_check_plateau())
    return flags


def test_regression_shift_is_not_a_plateau():
    """Falsification: a stationary run, then one all-'A' batch. The KS
    threshold (~0.0096 at n=20k) called this a plateau."""
    rng = random.Random(7)
    chain = _chain()
    _warm(chain, rng, 100)
    chain.train(b"A" * BATCH)
    assert not chain.snapshot_and_check_plateau()
    assert chain.last_js_divergence > chain.last_plateau_threshold


def test_stationary_source_is_a_plateau():
    """Control: the same source against itself must read as plateau at
    roughly the 95% the null threshold promises."""
    rng = random.Random(11)
    chain = _chain()
    _warm(chain, rng, 20)
    flags = _warm(chain, rng, 200)
    assert sum(flags) / len(flags) >= 0.85


def test_threshold_matches_analytic_null():
    """Order 0, K=2: n=4 then m=2 more -> scale = m / (8 n N) = 1/96."""
    chain = _chain()
    chain.train(b"AABB")
    chain.snapshot_and_check_plateau()
    chain.train(b"AB")
    chain.snapshot_and_check_plateau()
    n, m, k = 4, 2, 2
    scale = m / (8 * n * (n + m))
    expected = (k - 1) * scale + Z95 * math.sqrt(2 * (k - 1)) * scale
    assert chain.last_plateau_threshold == pytest.approx(expected)


def test_constant_input_plateaus():
    """Adversarial: one symbol per context -> null and JS are both 0. A
    model that cannot change has plateaued; strict '<' would never say so."""
    chain = _chain()
    flags = [chain.train(b"\x00" * BATCH) or chain.snapshot_and_check_plateau() for _ in range(5)]
    assert flags[-1]
    assert chain.last_plateau_threshold == 0.0


def test_new_context_is_learning_not_noise():
    """Adversarial: a context absent from the previous snapshot contributes
    JS but no null mass, so its appearance is never mistaken for noise."""
    rng = random.Random(3)
    chain = _chain(order=1)
    _warm(chain, rng, 50)
    chain.train(b"Z" * 4 + _uniform(rng))
    assert not chain.snapshot_and_check_plateau()


def test_ensemble_constant_input_plateaus():
    """Adversarial: the ensemble aggregates thresholds; with all of them 0 a
    strict '<' made a deterministic ensemble unable to plateau."""
    ens = MarkovEnsemble(orders=[0, 1])
    for chain in ens.chains.values():
        chain._snapshot_interval = 1
    flags = [ens.train(b"\x00" * BATCH) or ens.snapshot_and_check_plateau() for _ in range(5)]
    assert flags[-1]
