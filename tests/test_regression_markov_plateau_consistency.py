"""Regression: the ensemble's plateau flag and its JS number disagreed.

``MarkovEnsemble.snapshot_and_check_plateau`` returned ``any(results)`` while
setting ``last_js_divergence`` to the ``max`` across chains. The flag therefore
came from the *most* converged chain and the number from the *least* converged
one -- two answers to the same question, and both were consumed:

- ``corpus_manager`` logs "Markov plateau detected (JS=%.4f)" off the flag,
  printing the number.
- ``seed_picker._pick_markov_seed`` gates ``gen_rate`` on the number.

Measured with orders 0/1/2 on high-entropy input: the ensemble returned True
(order-0 had converged) while reporting JS=0.0886, against order-2's own
threshold of 0.0124. So corpus_manager announced a plateau for a chain that by
this code's own test had not plateaued, and seed_picker, reading that same
field, evaluated 0.0886 < 0.0124 as False and left the generation rate at 0.15.
The log and the behaviour contradicted each other in the same iteration.

seed_picker also recomputed its own threshold from the *ensemble's*
``_contexts_seen`` rather than the one the decision was made with, so even a
consistent aggregate could have been judged against the wrong number.

Both are now derived from the weights ``_select_chain`` samples with, and the
threshold is published alongside the value.
"""

import random

import pytest

from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble

HIGH_ENTROPY = "high_entropy"
CONVERGING = "converging"


def _ensemble(orders=(0, 1, 2), interval=10):
    ens = MarkovEnsemble(orders=list(orders))
    ens._snapshot_interval = interval
    for chain in ens.chains.values():
        chain._snapshot_interval = interval
    return ens


def _train(ens, kind, n, size=200, seed=9):
    rng = random.Random(seed)
    flags = []
    for _ in range(n):
        if kind == HIGH_ENTROPY:
            data = bytes(rng.randrange(256) for _ in range(size))
        else:
            data = bytes(rng.choice(b"ABCD") for _ in range(size))
        ens.train(data)
        flags.append(ens.snapshot_and_check_plateau())
    return flags


class TestFlagAgreesWithValue:
    @pytest.mark.parametrize("kind", [HIGH_ENTROPY, CONVERGING])
    def test_flag_never_contradicts_the_published_number(self, kind):
        """The invariant the old code broke: a True flag must be consistent
        with the value and threshold the object reports."""
        ens = _ensemble()
        flags = _train(ens, kind, 240, seed=9 if kind == HIGH_ENTROPY else 2)
        for flag in flags:
            if flag:
                assert ens.last_js_divergence < ens.last_plateau_threshold

    def test_still_learning_is_not_reported_as_plateau(self):
        """The measured case: order-0 converges on random bytes while the
        higher orders are still moving, and the ensemble claimed plateau."""
        ens = _ensemble()
        flags = _train(ens, HIGH_ENTROPY, 240)
        assert not any(flags)
        # order-0 really has converged -- the old any() is why this misfired.
        assert ens.chains[0].last_js_divergence < ens.chains[2].last_js_divergence

    def test_converged_model_still_detects_plateau(self):
        """The fix must not simply suppress the signal."""
        ens = _ensemble()
        flags = _train(ens, CONVERGING, 400, seed=2)
        assert any(flags)
        assert ens.last_js_divergence < ens.last_plateau_threshold


class TestSelectionWeighting:
    def test_aggregate_uses_select_chain_weights(self):
        ens = _ensemble()
        _train(ens, HIGH_ENTROPY, 120)
        weights = ens._selection_weights()
        assert pytest.approx(sum(weights.values())) == 1.0
        expected = sum(weights[o] * ens.chains[o].last_js_divergence for o in ens.orders)
        assert ens.last_js_divergence == pytest.approx(expected)

    def test_aggregate_is_not_the_max(self):
        """Guards the specific regression: max() ignored the weighting."""
        ens = _ensemble()
        _train(ens, HIGH_ENTROPY, 120)
        per_chain = [c.last_js_divergence for c in ens.chains.values()]
        assert ens.last_js_divergence < max(per_chain)
        assert ens.last_js_divergence > min(per_chain)

    def test_weights_survive_an_untrained_ensemble(self):
        ens = _ensemble()
        weights = ens._selection_weights()
        assert pytest.approx(sum(weights.values())) == 1.0
        assert all(w > 0 for w in weights.values())


class TestPublishedThreshold:
    def test_chain_publishes_the_threshold_it_used(self):
        chain = MarkovChain(order=1)
        chain._snapshot_interval = 10
        assert chain.last_plateau_threshold == 0.0
        rng = random.Random(3)
        for _ in range(60):
            chain.train(bytes(rng.choice(b"ABCD") for _ in range(200)))
            chain.snapshot_and_check_plateau()
        assert chain.last_plateau_threshold > 0.0

    def test_ensemble_threshold_is_weighted_like_the_value(self):
        ens = _ensemble()
        _train(ens, HIGH_ENTROPY, 120)
        weights = ens._selection_weights()
        expected = sum(weights[o] * ens.chains[o].last_plateau_threshold for o in ens.orders)
        assert ens.last_plateau_threshold == pytest.approx(expected)

    def test_seed_picker_gate_matches_the_flag(self):
        """seed_picker's branch and the plateau flag must not disagree."""
        ens = _ensemble()
        for kind, seed in ((HIGH_ENTROPY, 9), (CONVERGING, 2)):
            ens = _ensemble()
            flags = _train(ens, kind, 240, seed=seed)
            gate = ens.last_js_divergence < ens.last_plateau_threshold
            if flags[-1]:
                assert gate, "flag says plateau, seed_picker's gate says no"
