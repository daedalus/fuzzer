"""Regression: corpus prunes forced a full literal rescan and could skip new seeds.

``_op_corpus_literal_insert`` reset its ``LiteralAccumulator`` whenever the
corpus shrank, rescanning every seed: GStreamer prunes ~1/s on a 5,000-seed,
~8 KB corpus, 5.7% of wall. And when a prune and appends landed between two
calls with ``len`` still >= ``scanned``, the index tail skipped new seeds.
``LiteralAccumulator.sync`` tracks scanned seeds by identity.
"""

from __future__ import annotations

import pytest

from fuzzer_tool.core.mutations import generic
from fuzzer_tool.core.mutations.generic import LiteralAccumulator, extract_corpus_literals


def _seed(tag: str) -> bytearray:
    return bytearray(f"\x00{tag}word\x00{len(tag)}42\x00".encode())


@pytest.fixture
def scans(monkeypatch):
    """Count seeds handed to the scanner."""
    seen: list[bytes] = []
    real = generic._scan_literals

    def counting(raw, acc):
        seen.append(bytes(raw))
        real(raw, acc)

    monkeypatch.setattr(generic, "_scan_literals", counting)
    return seen


def _lits(corpus) -> tuple[set[bytes], set[bytes]]:
    """Oracle: literals of a fresh full scan."""
    ints, strs = extract_corpus_literals(list(corpus))
    return set(ints), set(strs)


def test_regression_prune_no_rescan(scans):
    corpus = [_seed(f"s{i}") for i in range(50)]
    acc = LiteralAccumulator()
    acc.sync(corpus)
    scans.clear()
    del corpus[10]
    acc.sync(corpus)
    assert scans == []


def test_regression_prune_then_append(scans):
    """Prune 1 + append 2 between calls: both new seeds are scanned."""
    corpus = [_seed(t) for t in ("aaa", "bbb", "ccc")]
    acc = LiteralAccumulator()
    acc.sync(corpus)
    del corpus[0]
    corpus += [_seed("ddd"), _seed("eee")]
    scans.clear()
    acc.sync(corpus)
    assert sorted(scans) == sorted(bytes(s) for s in corpus[-2:])
    ints, strs = _lits(corpus)
    assert ints <= set(acc.int_lits) and strs <= set(acc.str_lits)


def test_growth_scans_tail_only(scans):
    """Falsification: plain growth still scans just the new seeds."""
    corpus = [_seed(f"g{i}") for i in range(5)]
    acc = LiteralAccumulator()
    acc.sync(corpus)
    corpus.append(_seed("new"))
    scans.clear()
    acc.sync(corpus)
    assert scans == [bytes(corpus[-1])]


def test_churn_stays_bounded():
    """Adversarial: full corpus turnover keeps stale literals bounded."""
    acc = LiteralAccumulator()
    for gen in range(40):
        corpus = [_seed(f"t{gen}x{i}") for i in range(20)]
        acc.sync(corpus)
        ints, strs = _lits(corpus)
        assert strs <= set(acc.str_lits)
        live = len(ints) + len(strs)
        assert len(acc.int_lits) + len(acc.str_lits) <= 2 * live + live


def test_wholesale_replace_same_objects(scans):
    """Minimization rebinds the list to survivors: same objects, nothing rescanned."""
    corpus = [_seed(f"m{i}") for i in range(30)]
    acc = LiteralAccumulator()
    acc.sync(corpus)
    scans.clear()
    acc.sync(corpus[::3])
    assert scans == []


def test_operator_uses_sync(scans):
    """Wiring: the operator keeps its accumulator across a prune."""
    from types import SimpleNamespace

    from fuzzer_tool.core.rand_pool import RandPool
    from fuzzer_tool.services.operators import OperatorEngine

    eng = OperatorEngine.__new__(OperatorEngine)
    corpus = [_seed(f"o{i}") for i in range(10)]
    eng._ctx_cache = SimpleNamespace(corpus=corpus, _rng=RandPool(seed=1))
    eng._op_corpus_literal_insert(bytearray(64), 0, b"")
    scans.clear()
    del corpus[3]
    eng._op_corpus_literal_insert(bytearray(64), 0, b"")
    assert scans == []
