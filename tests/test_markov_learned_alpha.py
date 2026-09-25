"""MarkovChain / MarkovEnsemble learned Dirichlet smoothing (AlphaMode.LEARNED)."""

import numpy as np

from fuzzer_tool.core.dirichlet import AlphaMode, dm_alpha
from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble
from fuzzer_tool.core.rand_pool import RandPool

BYTE_VALUES = 256
DEFAULT_SMOOTHING = 0.01
STRUCTURED = [b"GET /index.html HTTP/1.1\r\nHost: a\r\n\r\n"] * 30


def _random_corpus(seed: int) -> list[bytes]:
    g = np.random.default_rng(seed)
    return [bytes(g.integers(0, BYTE_VALUES, 64, dtype=np.uint8)) for _ in range(30)]


def _chain(mode: AlphaMode) -> MarkovChain:
    return MarkovChain(order=1, alpha_mode=mode, rng=RandPool(seed=0))


def test_learned_equals_mle_of_transitions():
    """Smoothing is the MLE computed independently from the transition counts."""
    mc = _chain(AlphaMode.LEARNED)
    mc.train_corpus(STRUCTURED + _random_corpus(seed=1))
    rows = [list(c.values()) for c in mc.transitions.values()]
    assert mc.smoothing == dm_alpha(rows, BYTE_VALUES, DEFAULT_SMOOTHING)


def test_falsification_structured_below_random():
    structured = _chain(AlphaMode.LEARNED)
    structured.train_corpus(STRUCTURED)
    random_ = _chain(AlphaMode.LEARNED)
    random_.train_corpus(_random_corpus(seed=2))
    assert structured.smoothing < DEFAULT_SMOOTHING < random_.smoothing


def test_fixed_mode_untouched():
    mc = _chain(AlphaMode.FIXED)
    mc.train_corpus(STRUCTURED)
    assert mc.smoothing == DEFAULT_SMOOTHING


def test_snapshot_refits_incremental_training():
    """corpus_manager trains one input at a time; the snapshot tick refits."""
    mc = _chain(AlphaMode.LEARNED)
    mc._snapshot_interval = 2
    mc.train(STRUCTURED[0])
    mc.snapshot_and_check_plateau()
    assert mc.smoothing == DEFAULT_SMOOTHING
    mc.train(STRUCTURED[0])
    mc.snapshot_and_check_plateau()
    assert mc.smoothing < DEFAULT_SMOOTHING


def test_learned_lowers_codelength_on_structured():
    """The point of the MLE: held-in structured data costs fewer bits."""
    fixed = _chain(AlphaMode.FIXED)
    learned = _chain(AlphaMode.LEARNED)
    for mc in (fixed, learned):
        mc.train_corpus(STRUCTURED)
    assert learned.codelength(STRUCTURED[0]) < fixed.codelength(STRUCTURED[0])


def test_adversarial_empty_and_single_byte():
    mc = _chain(AlphaMode.LEARNED)
    mc.train_corpus([])
    assert mc.smoothing == DEFAULT_SMOOTHING
    mc.train_corpus([b"A"])
    assert mc.smoothing == DEFAULT_SMOOTHING
    assert len(mc.generate(8)) == 8


def test_ensemble_propagates_mode_and_survives_roundtrip():
    ens = MarkovEnsemble(orders=[0, 1], alpha_mode=AlphaMode.LEARNED, rng=RandPool(seed=0))
    ens.train_corpus(STRUCTURED)
    learned = {o: c.smoothing for o, c in ens.chains.items()}
    for chain in ens.chains.values():
        rows = [list(c.values()) for c in chain.transitions.values()]
        assert chain.smoothing == dm_alpha(rows, BYTE_VALUES, DEFAULT_SMOOTHING)

    restored = MarkovEnsemble(orders=[0, 1], alpha_mode=AlphaMode.LEARNED, rng=RandPool(seed=0))
    restored.from_dict(ens.to_dict())
    assert {o: c.smoothing for o, c in restored.chains.items()} == learned
    assert all(c.alpha_mode is AlphaMode.LEARNED for c in restored.chains.values())
