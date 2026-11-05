"""Probe-displacement histogram: how far each stored edge sits from its home slot.

Displacement = (slot - home_slot(edge_id, n)) mod n, in [0, PROBE_MAX). The
histogram folds every claimed entry (live or stale) when the table is lost
(generation wrap, resize) and adds the current table on read.
"""

from unittest.mock import MagicMock

import numpy as np

from fuzzer_tool.adapters.shm import ShmCoverage, displacements, home_slot
from fuzzer_tool.services.report import _coverage_analysis

N = 1024
WRAP_RESETS = 256


def _ids_homed_at(home: int, k: int, n: int = N) -> list[int]:
    """First *k* positive ids whose home slot is *home* (deterministic search)."""
    out = []
    i = 1
    while len(out) < k:
        if home_slot(i, n) == home:
            out.append(i)
        i += 1
    return out


def _shm(n: int = N) -> ShmCoverage:
    shm = ShmCoverage(size=n)
    shm.reset_diag()
    return shm


def test_displacements_match_scalar_home_slot():
    ids = np.array([1, 2, 233, 0xFFFFFF, 0x9E3779B1, 0xFFFFFFFF], dtype=np.uint32)
    slots = np.array([0, 5, 1023, 7, 512, 1], dtype=np.int64)
    want = [(int(s) - home_slot(int(e), N)) % N for s, e in zip(slots, ids, strict=True)]
    assert displacements(slots, ids, N).tolist() == want


def test_displacements_non_power_of_two():
    n = 1000
    ids = np.arange(1, 500, dtype=np.uint32)
    slots = np.array([home_slot(int(e), n) for e in ids], dtype=np.int64)
    assert not displacements(slots, ids, n).any()


def test_collision_chain_fills_consecutive_bins():
    k = 5
    shm = _shm()
    try:
        for eid in _ids_homed_at(100, k):
            assert shm.record_edge(eid)
        hist, peak = shm.displacement_histogram()
        assert hist[:k].tolist() == [1] * k
        assert hist[k:].sum() == 0
        assert peak == k / N
    finally:
        shm.cleanup()


def test_probe_wrapping_past_last_slot_is_short():
    shm = _shm()
    try:
        for eid in _ids_homed_at(N - 1, 2):
            shm.record_edge(eid)
        hist, _ = shm.displacement_histogram()
        assert hist[0] == 1 and hist[1] == 1
    finally:
        shm.cleanup()


def test_generation_wrap_keeps_history():
    ids = _ids_homed_at(7, 3)
    shm = _shm()
    try:
        for eid in ids:
            shm.record_edge(eid)
        for _ in range(WRAP_RESETS):
            shm.reset_edge_map()
        assert not shm.edge_id_array().size  # table wiped

        hist, _ = shm.displacement_histogram()
        assert hist[:3].tolist() == [1, 1, 1]

        for eid in ids:
            shm.record_edge(eid)
        hist, _ = shm.displacement_histogram()
        assert hist[:3].tolist() == [2, 2, 2]
    finally:
        shm.cleanup()


def test_resize_folds_old_table():
    shm = _shm()
    try:
        for eid in _ids_homed_at(3, 2):
            shm.record_edge(eid)
        shm.resize(2 * N)
        hist, peak = shm.displacement_histogram()
        assert hist[:2].tolist() == [1, 1]
        assert peak == 2 / N
    finally:
        shm.cleanup()


def test_falsification_empty_table_is_all_zero():
    shm = _shm()
    try:
        hist, peak = shm.displacement_histogram()
        assert hist.size == ShmCoverage.PROBE_MAX + 1
        assert not hist.any()
        assert peak == 0.0
    finally:
        shm.cleanup()


def test_adversarial_foreign_entry_lands_in_overflow_bin():
    """An entry outside the probe window (foreign writer) must not be lost or crash."""
    shm = _shm()
    try:
        eid = _ids_homed_at(0, 1)[0]
        shm._entries[N // 2].edge_id = eid
        shm._entries[N // 2].count = 1
        hist, _ = shm.displacement_histogram()
        assert hist[ShmCoverage.PROBE_MAX] == 1
        assert hist.sum() == 1
    finally:
        shm.cleanup()


def test_report_prints_displacement_line():
    shm = _shm()
    try:
        for eid in _ids_homed_at(9, 2):
            shm.record_edge(eid)
        f = MagicMock()
        f.shm_cov = shm
        f.corpus_dir = "/nonexistent-corpus-for-report"
        text = _coverage_analysis(f)
        assert "Probe displacement:" in text
        assert "max 1" in text
    finally:
        shm.cleanup()


def test_report_skips_line_without_histogram():
    f = MagicMock()
    f.shm_cov.size = N
    f.shm_cov.cumulative_edges = 0
    f.shm_cov.read_dropped_edges.return_value = 0
    f.shm_cov._seen = None
    f.corpus_dir = "/nonexistent-corpus-for-report"
    assert "Probe displacement" not in _coverage_analysis(f)
