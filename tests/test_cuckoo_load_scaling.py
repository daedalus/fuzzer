"""Regression tests for CuckooFilter load-scaling findings.

Covers: sizing vs MAX_LOAD, transactional (rollback) failed inserts,
private RNG isolation, realised false-positive rate, fingerprint
uniformity, and update_bytes generational reset on kick failure.
"""

from __future__ import annotations

import os
import random
from collections import Counter

import pytest

from fuzzer_tool.core.cuckoo import CuckooFilter


def _keys(n: int, salt: int = 0) -> list[bytes]:
    rng = random.Random(1234 + salt)
    return [rng.randbytes(12) + i.to_bytes(4, "big") for i in range(n)]


@pytest.mark.parametrize("cap", [1000, 4096, 5000, 100_000, 262_144, 500_000, 1_048_576, 1_048_577])
def test_load_at_capacity_never_exceeds_max_load(cap):
    cf = CuckooFilter(capacity=cap)
    assert cap / (cf.size * cf.bucket_size) <= CuckooFilter.MAX_LOAD + 1e-12


@pytest.mark.parametrize("cap", [4096, 50_000, 262_144])
def test_fill_to_capacity_has_no_failures_or_false_negatives(cap):
    cf = CuckooFilter(capacity=cap)
    ks = _keys(cap)
    assert all(cf.add(k) for k in ks)
    assert cf.n_failed == 0
    assert all(cf.contains(k) for k in ks)


def test_failed_add_is_rolled_back_no_false_negatives():
    # Force overload: shrink the table so kick chains must fail.
    cf = CuckooFilter(capacity=10**6, bucket_size=4, fingerprint_size=12)
    cf.size, cf._mask = 256, 255
    cf.buckets = [[] for _ in range(256)]
    stored = []
    failed = 0
    for k in _keys(1200):
        snapshot = [list(b) for b in cf.buckets] if failed < 3 else None
        if cf.add(k):
            stored.append(k)
        else:
            failed += 1
            if snapshot is not None:
                assert cf.buckets == snapshot  # exact state restored
    assert failed > 0 and cf.n_failed == failed
    assert cf.count == len(stored)
    assert all(cf.contains(k) for k in stored)


def test_private_rng_does_not_touch_global_stream():
    random.seed(5)
    expected = random.random()
    random.seed(5)
    cf = CuckooFilter(capacity=64, bucket_size=1)
    for k in _keys(64):
        cf.add(k)  # bucket_size=1 guarantees kicking
    assert random.random() == expected


def test_kicking_is_deterministic_per_rng_seed():
    def run(seed):
        cf = CuckooFilter(capacity=64, bucket_size=1, rng_seed=seed)
        for k in _keys(64):
            cf.add(k)
        return cf.buckets

    assert run(7) == run(7)


def test_realised_fpr_is_below_bloom_1e_3():
    cf = CuckooFilter(capacity=200_000)
    for k in _keys(200_000):
        cf.add(k)
    probes = _keys(200_000, salt=99)
    fpr = sum(cf.contains(k) for k in probes) / len(probes)
    assert fpr < 1e-3
    assert cf.expected_fpr < 1e-3


def test_fingerprint_is_uniform_and_nonzero():
    cf = CuckooFilter(capacity=1000, fingerprint_size=4)
    counts = Counter(cf._get_fingerprint(os.urandom(8)) for _ in range(150_000))
    assert 0 not in counts and set(counts) <= set(range(1, 16))
    expect = 150_000 / 15
    assert all(abs(c - expect) < 0.08 * expect for c in counts.values())


def test_fingerprint_size_one_is_valid():
    cf = CuckooFilter(capacity=100, fingerprint_size=1)
    assert {cf._get_fingerprint(bytes([i])) for i in range(50)} == {1}


def test_alt_index_is_involution():
    cf = CuckooFilter(capacity=5000)
    for k in _keys(200):
        fp, i1, i2 = cf._locate(k)
        assert cf._get_alt_index(fp, i2) == i1


def test_update_bytes_resets_generation_when_add_fails():
    cf = CuckooFilter(capacity=10**6, bucket_size=1, fingerprint_size=12, max_kicks=10)
    cf.size, cf._mask = 8, 7
    cf.buckets = [[] for _ in range(8)]
    for i in range(200):
        cf.update_bytes(_k := i.to_bytes(4, "big"), reset_on_full=True)
        assert cf.contains(_k)  # always tracked, never silently dropped
    assert cf.count <= 8


def test_update_bytes_without_reset_reports_unseen_on_failure():
    cf = CuckooFilter(capacity=10**6, bucket_size=1, fingerprint_size=12, max_kicks=10)
    cf.size, cf._mask = 8, 7
    cf.buckets = [[] for _ in range(8)]
    results = [cf.update_bytes(i.to_bytes(4, "big")) for i in range(100)]
    assert cf.n_failed > 0 and not any(results)
