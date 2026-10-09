"""Regression: CorpusManager.seed_key memoizes corpus seeds.

ffmpeg campaign, 6,000 rounds on ~2,900 seeds: 5.1M seed_key calls, each an
xxh64 over the whole seed -- the Pareto rebuild and the weight refresh walk
the corpus every round. Corpus seeds are long-lived bytes objects whose
hash is cached, so a dict hit is O(1).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import xxhash

from fuzzer_tool.services.corpus_manager import CorpusManager


def _expected(data: bytes) -> str:
    return xxhash.xxh64(data).hexdigest()[:16]


def _manager(seeds: list[bytes]) -> CorpusManager:
    return CorpusManager(SimpleNamespace(seed_meta={s: {} for s in seeds}))


def test_regression_seed_key_memo_matches_hash():
    seeds = [bytes([i]) * (i + 1) for i in range(50)]
    cm = _manager(seeds)
    for _ in range(3):
        assert [cm.seed_key(s) for s in seeds] == [_expected(s) for s in seeds]


def test_regression_seed_key_memo_skips_non_corpus():
    """Falsification: mutants never enter the memo, so it cannot leak them."""
    cm = _manager([b"seed"])
    for i in range(200):
        mutant = b"m" + i.to_bytes(4, "little")
        assert cm.seed_key(mutant) == _expected(mutant)
    cm.seed_key(b"seed")
    assert len(cm._key_memo) == 1


def test_regression_seed_key_memo_adversarial_inputs():
    """bytearray (unhashable), empty, and equal-content distinct objects."""
    cm = _manager([b"", b"abc"])
    assert cm.seed_key(bytearray(b"abc")) == _expected(b"abc")
    assert cm.seed_key(b"") == _expected(b"")
    twin = bytes(bytearray(b"abc"))
    assert cm.seed_key(twin) == _expected(b"abc")


def test_regression_seed_key_memo_bounded_after_churn():
    """Seeds leaving seed_meta must not pile up in the memo forever."""
    meta: dict[bytes, dict] = {}
    cm = CorpusManager(SimpleNamespace(seed_meta=meta))
    for i in range(5000):
        seed = i.to_bytes(4, "little") * 8
        meta.clear()
        meta[seed] = {}
        cm.seed_key(seed)
    assert len(cm._key_memo) <= 2 * len(meta) + 1024


def test_regression_seed_key_memo_speed():
    seeds = [bytes([i % 256]) * 4096 + i.to_bytes(4, "little") for i in range(3000)]
    cm = _manager(seeds)
    for s in seeds:
        cm.seed_key(s)

    t0 = time.perf_counter()
    for _ in range(20):
        for s in seeds:
            _expected(s)
    reference = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        for s in seeds:
            cm.seed_key(s)
    elapsed = time.perf_counter() - t0

    assert elapsed * 2 < reference, f"memo {elapsed:.3f}s vs hash {reference:.3f}s"
