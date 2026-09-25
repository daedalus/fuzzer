"""Regression: the byte-entropy readout froze at its corpus-load value.

``Fuzzer._corpus_entropy`` was only fed by ``load_corpus()``: seeds admitted,
trimmed or pruned mid-run never reached it, so ``byte-ent:`` showed the same
number all campaign. Oracle: ``report._corpus_byte_entropy`` rescans the live
corpus independently.
"""

import types

import pytest

from fuzzer_tool.core.byte_entropy import CumulativeByteEntropy
from fuzzer_tool.services.corpus_manager import CorpusManager
from fuzzer_tool.services.report import _corpus_byte_entropy
from tests.test_corpus_minimization import MockFuzzer


def _mgr(tmp_path, corpus):
    f = MockFuzzer(tmp_path)
    f.corpus = list(corpus)
    f._corpus_entropy = CumulativeByteEntropy()
    for s in corpus:
        f._corpus_entropy.add(s)
    return f, CorpusManager(f)


def _assert_tracks(f):
    assert f._corpus_entropy.bits() == pytest.approx(_corpus_byte_entropy(f.corpus))


def test_control_rescan_matches_fresh_tracker(tmp_path):
    """Control: oracle and tracker agree before any mid-run change."""
    f, _ = _mgr(tmp_path, [b"\x00" * 64, bytes(range(64))])
    _assert_tracks(f)


def test_regression_admitted_seed_reaches_readout(tmp_path):
    f, mgr = _mgr(tmp_path, [b"\x00" * 64])
    before = f._corpus_entropy.bits()

    mgr.save_to_corpus(bytes(range(256)))

    assert f._corpus_entropy.bits() != before
    _assert_tracks(f)


def test_regression_pruned_near_dup_leaves_readout(tmp_path):
    seeds = [bytes([i]) * 64 for i in range(12)]
    f, mgr = _mgr(tmp_path, seeds)
    a, b = mgr.seed_key(seeds[0]), mgr.seed_key(seeds[1])
    f._edge_tracker.find_near_duplicate_seeds = lambda max_hamming: [(a, b, 0.0)]

    mgr.deprioritize_near_duplicates()

    assert len(f.corpus) == len(seeds) - 1
    _assert_tracks(f)


def test_regression_trimmed_seed_swaps_in_readout(tmp_path):
    parent = b"P" * 32
    data = b"\x00" * 32 + bytes(range(32))
    f, mgr = _mgr(tmp_path, [parent, data])
    f.ptrace_cov = types.SimpleNamespace(edge_map=bytearray([1, 1]))
    f._runner = types.SimpleNamespace(run_target=lambda d: (0, None))

    mgr.trim_new_coverage(data, parent)

    assert data not in f.corpus
    _assert_tracks(f)


def test_no_tracker_is_safe(tmp_path):
    """Adversarial: resumed state from before the tracker existed has none;
    admission must neither raise nor invent one."""
    f = MockFuzzer(tmp_path)
    mgr = CorpusManager(f)

    mgr.save_to_corpus(b"abc" * 20)

    assert f.corpus
    assert getattr(f, "_corpus_entropy", None) is None


def test_regression_rebuild_after_wholesale_swap(tmp_path):
    """Seed transforms / minimize replace ``f.corpus`` outright."""
    f, mgr = _mgr(tmp_path, [b"\x00" * 64])
    f.corpus = [bytes(range(64)), b"\xff" * 8]

    mgr.rebuild_entropy()

    _assert_tracks(f)
