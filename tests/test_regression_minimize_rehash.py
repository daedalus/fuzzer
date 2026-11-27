"""Regression: auto-minimize re-hashed every corpus seed on every pass.

``auto_minimize_corpus``, ``_set_aside`` and ``_commit_minimize`` called
``hash_data`` (pure-Python XXH64) per seed, bypassing ``CorpusManager``'s
memoized ``seed_key`` -- the same function. GStreamer, 3.6k seeds up to
64 KiB: 13.7% of wall in ``key16``. Corpus seeds now hit the memo.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import fuzzer_tool.adapters.filesystem as fs
from fuzzer_tool.adapters.filesystem import hash_data
from fuzzer_tool.services.corpus_manager import SEED_KEY_MEMO_SLACK, CorpusManager

SEEDS = [b"alpha" * 40, b"bravo" * 40, b"charlie" * 40]


def _cm(irreplaceable=()):
    corpus = list(SEEDS)
    f = SimpleNamespace(
        corpus=corpus,
        seed_meta={s: {"fuzz_count": 1} for s in corpus},
        irreplaceable_hashes={hash_data(s) for s in irreplaceable},
    )
    return CorpusManager.__new__(CorpusManager), f


def _wire(cm, f):
    cm.f = f
    cm._key_memo = {}
    for s in f.corpus:
        cm.seed_key(s)  # warm: admission hashes each seed once
    return cm


def _count(monkeypatch) -> list[int]:
    calls = [0]

    def counting(data):
        calls[0] += 1
        return fs.key16(data)

    monkeypatch.setattr(fs, "hash_data", counting)
    return calls


def test_regression_minimize_rehash(monkeypatch):
    """_set_aside reads warm keys: no XXH64 per seed."""
    cm = _wire(*_cm(irreplaceable=[SEEDS[1]]))
    calls = _count(monkeypatch)
    irreplaceable, _fresh = cm._set_aside(list(SEEDS))
    assert irreplaceable == [SEEDS[1]]
    assert calls[0] == 0


@pytest.mark.parametrize("data", [*SEEDS, b"not-in-corpus", bytearray(b"alpha" * 40)])
def test_memo_key_equals_hash_data(data):
    """Falsification: the memoized key is the persisted key, in and out of the corpus."""
    cm = _wire(*_cm())
    assert cm.seed_key(data) == hash_data(bytes(data))


def test_cold_memo_still_sets_aside(monkeypatch):
    """Adversarial: a memo cleared at its bound recomputes, never misses a seed."""
    cm = _wire(*_cm(irreplaceable=[SEEDS[0], SEEDS[2]]))
    cm._key_memo.clear()
    assert SEED_KEY_MEMO_SLACK > 0
    irreplaceable, _ = cm._set_aside(list(SEEDS))
    assert irreplaceable == [SEEDS[0], SEEDS[2]]
