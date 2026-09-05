"""_pick_bayesian_seed hashed the corpus twice per pick.

It built ``seed_ids`` from ``f.corpus`` in order, handed the ids to
``select_seed()``, got an id back -- losing the position -- and then scanned
``f.corpus`` recomputing ``_seed_key`` on every entry to find it again. The
index was available all along.
"""

from types import SimpleNamespace

import pytest

from fuzzer_tool.core.seed_quality import BayesianSeedQuality
from fuzzer_tool.services.seed_picker import SeedPicker


class _CountingKey:
    """Wraps the real key function and counts calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, data: bytes) -> str:
        self.calls += 1
        return data.hex()


def _picker(corpus, quality, key):
    fuzzer = SimpleNamespace(
        corpus=list(corpus),
        _seed_quality=quality,
        _seed_key=key,
        _rand_pool=SimpleNamespace(choice=lambda seq: seq[0]),
    )
    return SeedPicker(fuzzer), fuzzer


def _quality(ids):
    q = BayesianSeedQuality()
    for sid in ids:
        q.init_seed(sid)
    return q


CORPUS = [bytes([i]) * 8 for i in range(20)]


def test_corpus_is_hashed_once_per_pick():
    key = _CountingKey()
    picker, _ = _picker(CORPUS, _quality([c.hex() for c in CORPUS]), key)

    picker._pick_bayesian_seed()

    assert key.calls == len(CORPUS)


def test_returns_the_seed_the_selector_chose(monkeypatch):
    key = _CountingKey()
    quality = _quality([c.hex() for c in CORPUS])
    picker, _ = _picker(CORPUS, quality, key)
    monkeypatch.setattr(quality, "select_index", lambda _ids: 7)

    assert picker._pick_bayesian_seed() == CORPUS[7]


def test_duplicate_seeds_do_not_collapse(monkeypatch):
    """Adversarial: identical seeds share a key, so the id could not
    disambiguate them; the index can."""
    corpus = [b"same", b"same", b"other"]
    key = _CountingKey()
    quality = _quality([c.hex() for c in corpus])
    picker, _ = _picker(corpus, quality, key)
    monkeypatch.setattr(quality, "select_index", lambda _ids: 1)

    assert picker._pick_bayesian_seed() is corpus[1]


def test_every_corpus_position_is_reachable(monkeypatch):
    key = _CountingKey()
    quality = _quality([c.hex() for c in CORPUS])
    picker, _ = _picker(CORPUS, quality, key)
    for i in range(len(CORPUS)):
        monkeypatch.setattr(quality, "select_index", lambda _ids, i=i: i)

        assert picker._pick_bayesian_seed() == CORPUS[i]


def test_unregistered_seeds_get_registered():
    key = _CountingKey()
    quality = BayesianSeedQuality()
    picker, _ = _picker(CORPUS, quality, key)

    picker._pick_bayesian_seed()

    assert len(quality._alpha) == len(CORPUS)


@pytest.mark.parametrize("size", [1, 2, 20])
def test_result_is_always_a_corpus_member(size):
    corpus = CORPUS[:size]
    picker, _ = _picker(corpus, _quality([c.hex() for c in corpus]), _CountingKey())

    assert picker._pick_bayesian_seed() in corpus
