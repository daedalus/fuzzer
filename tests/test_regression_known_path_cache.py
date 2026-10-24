"""Regression: ``_scan_new`` re-scanned every path it had already folded.

The fast path only skipped the scan when an exec repeated the *previous*
exec. A new seed is picked every exec, so it rarely fired and ~20 numpy calls
ran per exec. A path whose ``(path_hash, edge_count)`` was already folded
cannot carry a new id, bucket or maximum, so the scan is skipped and the
cached ids are returned (AFL's known trace checksum).
"""

from __future__ import annotations

import ctypes

import pytest

from fuzzer_tool.adapters import shm as shm_mod
from fuzzer_tool.adapters.shm import ShmCoverage, home_slot

COUNT_MASK = 0xFFFFFF
U64 = 0xFFFFFFFFFFFFFFFF
MAP = 4096

PATH_A = {1: 1, 2: 1, 3: 1}
PATH_B = {1: 1, 2: 1}


def _fire(shm: ShmCoverage, eid: int, count: int) -> None:
    """Write one edge the way the shim does.

    Claims bump edge_count; the hash advances once per hit.
    """
    gen = shm.read_generation() & 0xFF
    pos = home_slot(eid, shm.num_entries)
    for i in range(shm.PROBE_MAX):
        e = shm._entries[(pos + i) % shm.num_entries]
        if e.edge_id == 0:
            ctypes.c_uint64.from_address(shm._ptr + 16).value += 1
        if e.edge_id in (0, eid):
            e.edge_id = eid
            e.count = (gen << 24) | (count & COUNT_MASK)
            break
    ph = ctypes.c_uint64.from_address(shm._ptr + 8)
    for _ in range(count):
        ph.value = (ph.value * 31 ^ eid) & U64


def _exec(shm: ShmCoverage, hits: dict[int, int]) -> tuple[bool, set[int]]:
    """One execution: reset, shim zeroes the hash, edges fire, scan."""
    shm.reset_edge_map()
    ctypes.c_uint64.from_address(shm._ptr + 8).value = 0
    for eid, count in hits.items():
        _fire(shm, eid, count)
    has_new, ids = shm.is_new_coverage_with_edges()
    return has_new, set(ids)


@pytest.fixture
def cov():
    shm = ShmCoverage(size=MAP)
    yield shm
    shm.cleanup()


@pytest.fixture
def scans(cov, monkeypatch):
    """Count slow-path table scans."""
    calls = []
    real = cov._active_columns

    def spy():
        calls.append(1)
        return real()

    monkeypatch.setattr(cov, "_active_columns", spy)
    return calls


def test_regression_known_path_cache(cov, scans):
    """A, B, A: the second A is answered without scanning the table."""
    assert _exec(cov, PATH_A) == (True, set(PATH_A))
    _exec(cov, PATH_B)
    before = len(scans)
    assert _exec(cov, PATH_A) == (False, set(PATH_A))
    assert len(scans) == before
    assert cov.last_new_ids == frozenset()
    assert cov.last_old_bucket_novel is False
    assert cov.new_max_edges == 0


def test_new_edge_still_scanned(cov, scans):
    """Falsification: a new edge on a later exec is still found."""
    _exec(cov, PATH_A)
    _exec(cov, PATH_B)
    has_new, ids = _exec(cov, {**PATH_A, 9: 1})
    assert has_new and 9 in ids
    assert cov.last_new_ids == frozenset({9})


def test_new_bucket_still_scanned(cov):
    """Adversarial: same edges, higher count is a new hash and a new bucket."""
    _exec(cov, PATH_A)
    _exec(cov, PATH_B)
    has_new, _ = _exec(cov, {**PATH_A, 3: 8})
    assert has_new
    assert cov.last_old_bucket_novel


def test_mask_clears_cache(cov, scans):
    """Adversarial: masking edits what a scan reported; cached ids must go."""
    _exec(cov, PATH_A)
    _exec(cov, PATH_B)
    cov.mask_edges({3})
    before = len(scans)
    _exec(cov, PATH_A)
    assert len(scans) == before + 1


def test_cache_bounded(cov, monkeypatch):
    """Memory stays bounded across many distinct paths."""
    cap = 8
    monkeypatch.setattr(shm_mod, "_KNOWN_PATH_IDS_MAX", cap)
    for eid in range(1, 40):
        _exec(cov, {eid: 1, eid + 100: 1})
        assert cov._known_path_ids <= cap


def test_zero_hash_not_cached(cov):
    """A hand-built table (hash 0) never enters the cache."""
    _fire(cov, 5, 1)
    ctypes.c_uint64.from_address(cov._ptr + 8).value = 0
    assert cov.is_new_coverage()
    assert not cov._known_paths
