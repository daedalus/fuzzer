"""Regression: autotoken document frequencies lived in an unbounded Counter.

Every distinct token in the corpus got an entry: 502k entries on a 5k-seed
synthetic corpus (100 junk tokens per seed) to return 200. Counts now go
through a Misra–Gries summary of ``AUTOTOKEN_COUNTERS_PER_TOKEN * max_tokens``
counters; at 64x that kept 198-200 of the exact top 200 on the same corpus.
"""

from __future__ import annotations

from collections import Counter

import pytest

from fuzzer_tool.core.misra_gries import MisraGries
from fuzzer_tool.services import import_corpus
from fuzzer_tool.services.import_corpus import (
    AUTOTOKEN_COUNTERS_PER_TOKEN,
    build_autotoken_dictionary,
    extract_tokens,
)

MAGIC = b"magic"


class _PeakSpy(MisraGries):
    """Records the largest table size reached."""

    peak = 0

    def add(self, key):
        count = super().add(key)
        _PeakSpy.peak = max(_PeakSpy.peak, len(self))
        return count


@pytest.fixture
def spy(monkeypatch):
    _PeakSpy.peak = 0
    monkeypatch.setattr(import_corpus, "MisraGries", _PeakSpy)
    return _PeakSpy


def _write_corpus(root, seeds: list[bytes]):
    d = root / "corpus" / "seeds" / "00"
    d.mkdir(parents=True)
    for i, data in enumerate(seeds):
        (d / f"id_{i:04x}").write_bytes(data)
    return str(root / "corpus")


def _junk_seed(i: int, n: int, with_magic: bool) -> bytes:
    words = [f"junk{i}x{j}" for j in range(n)]
    if with_magic:
        words.append(MAGIC.decode())
    return " ".join(words).encode()


def test_regression_counter_is_bounded(tmp_path, spy):
    max_tokens = 1
    corpus = _write_corpus(tmp_path, [_junk_seed(i, 20, True) for i in range(50)])

    tokens = build_autotoken_dictionary(corpus, max_tokens=max_tokens)

    assert tokens == [MAGIC]
    assert 0 < spy.peak <= AUTOTOKEN_COUNTERS_PER_TOKEN * max_tokens


def test_matches_exact_counter_below_capacity(tmp_path):
    seeds = [b"alpha beta gamma", b"alpha beta", b"alpha delta", b"epsilon"]
    corpus = _write_corpus(tmp_path, seeds)

    def exact() -> list[bytes]:
        c: Counter[bytes] = Counter()
        for s in seeds:
            c.update(extract_tokens(s))
        return [t for t, _ in c.most_common(3)]

    # Control: the oracle agrees with a second run of itself.
    assert exact() == exact()
    assert build_autotoken_dictionary(corpus, max_tokens=3) == exact()


def test_falsify_single_seed_token_never_beats_shared(tmp_path):
    seeds = [_junk_seed(i, 5, True) for i in range(10)]
    corpus = _write_corpus(tmp_path, seeds)

    assert build_autotoken_dictionary(corpus, max_tokens=1) == [MAGIC]


def test_adversarial_all_distinct_tokens(tmp_path, spy):
    max_tokens = 2
    corpus = _write_corpus(tmp_path, [_junk_seed(i, 50, False) for i in range(40)])

    tokens = build_autotoken_dictionary(corpus, max_tokens=max_tokens)

    assert len(tokens) <= max_tokens
    assert spy.peak <= AUTOTOKEN_COUNTERS_PER_TOKEN * max_tokens
