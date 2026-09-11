"""Regression: ``--seed`` reaches the Markov model.

Fuzzer built ``MarkovChain(order=...)`` / ``MarkovEnsemble(...)`` without
``rng``, so both fell back to ``RandPool()`` -- OS entropy, which neither
``--seed`` nor ``random.seed`` reaches. Every ``markov_bytes`` mutation
(and Markov generation) was irreproducible, so a crash found through one
could not be replayed from its seed. ``MarkovEnsemble.from_dict`` then did
the same to every chain it rebuilt on resume.

It surfaced in ``test_fuzzer.py::test_seed_reproducibility`` once aa6425d
added two operators: the selection order shifted and ``markov_bytes``
landed inside the test's ten mutations.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fuzzer_tool.core.markov import MarkovChain, MarkovEnsemble
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.fuzzer import Fuzzer


def _fuzzer(tmp_path, **kwargs):
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=256,
            timeout=1,
            **kwargs,
        )


@pytest.mark.parametrize("order", [1, "1,2,3"])
def test_markov_draws_from_the_fuzzers_pool(tmp_path, order):
    f = _fuzzer(tmp_path, seed=7, markov_order=order)
    assert f.markov._rng is f._rng
    chains = getattr(f.markov, "chains", {})
    assert all(c._rng is f._rng for c in chains.values())


def test_same_seed_samples_the_same_bytes(tmp_path):
    """Untrained model: every sample_byte is a pool draw."""

    def sample(sub, seed):
        f = _fuzzer(tmp_path / sub, seed=seed)
        return [f.markov.sample_byte(b"") for _ in range(64)]

    assert sample("a", 11) == sample("b", 11)
    assert sample("c", 11) != sample("d", 12)


@pytest.mark.parametrize("fmt", ["ensemble", "legacy"])
def test_from_dict_keeps_the_ensembles_pool(fmt):
    pool = RandPool(3)
    src = MarkovEnsemble(orders=[1, 2], rng=RandPool(1))
    src.train(b"abcabcabd")
    data = src.to_dict()
    if fmt == "legacy":
        chain = MarkovChain(order=1, rng=RandPool(1))
        chain.train(b"abcabcabd")
        data = chain.to_dict()

    dst = MarkovEnsemble(orders=[1], rng=pool)
    dst.from_dict(data)
    assert dst.chains
    assert all(c._rng is pool for c in dst.chains.values())
