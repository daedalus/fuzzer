"""New-id detection prefilters on the virgin map, same ids as the set diff.

``_check_new_coverage`` diffed every live id against ``_seen_edge_ids``
(~5k ids per ffmpeg exec). An id with virgin bits is already seen, so only
virgin-zero ids need the set lookup. Entries are written the way the shim
writes them, not via ``record_edge`` (which updates the seen set itself).
"""

import ctypes

from fuzzer_tool.adapters.shm import VIRGIN_DENSE_MAX, ShmCoverage, home_slot

COUNT_MASK = 0xFFFFFF


def _shim_write(shm: ShmCoverage, eid: int, count: int) -> None:
    # Claim the home slot (or the next free one) and stamp the generation.
    gen = shm.read_generation() & 0xFF
    pos = home_slot(eid, shm.num_entries)
    for i in range(shm.PROBE_MAX):
        idx = (pos + i) % shm.num_entries
        if shm._entries[idx].edge_id in (0, eid):
            shm._entries[idx].edge_id = eid
            shm._entries[idx].count = (gen << 24) | (count & COUNT_MASK)
            break
    ec = ctypes.c_uint64.from_address(shm._ptr + 16)
    ec.value += 1
    ph = ctypes.c_uint64.from_address(shm._ptr + 8)
    ph.value = (ph.value * 31 ^ eid) & 0xFFFFFFFFFFFFFFFF


def _exec(shm: ShmCoverage, hits: dict[int, int]) -> tuple[bool, frozenset[int]]:
    shm.reset_edge_map()
    for eid, count in hits.items():
        _shim_write(shm, eid, count)
    has_new, _ids = shm.is_new_coverage_with_edges()
    return has_new, shm.last_new_ids


def test_regression_new_ids_first_and_repeat():
    shm = ShmCoverage(size=4096)
    assert _exec(shm, dict.fromkeys(range(1, 51), 1)) == (True, frozenset(range(1, 51)))
    has_new, new = _exec(shm, dict.fromkeys(range(25, 81), 1))
    assert has_new and new == frozenset(range(51, 81))
    assert _exec(shm, dict.fromkeys(range(1, 81), 1)) == (False, frozenset())


def test_regression_count_zero_id_seen_once():
    # Adversarial: a count-0 entry lands in the seen set with no virgin bit.
    shm = ShmCoverage(size=4096)
    assert _exec(shm, {7: 0, 8: 1})[1] == frozenset({7, 8})
    assert _exec(shm, {7: 0, 8: 1, 9: 1})[1] == frozenset({9})


def test_regression_masked_ids_never_new():
    shm = ShmCoverage(size=4096)
    shm.mask_edges({3, 4})
    assert _exec(shm, {3: 1, 4: 2, 5: 1})[1] == frozenset({5})


def test_regression_wide_ids_new_once():
    # Adversarial: ids past the dense virgin range use the overflow map.
    shm = ShmCoverage(size=4096)
    wide = VIRGIN_DENSE_MAX + 5
    assert _exec(shm, {wide: 1, 2: 1})[1] == frozenset({wide, 2})
    assert _exec(shm, {wide: 3, 2: 1})[1] == frozenset()


def test_regression_ids_past_virgin_size_are_candidates():
    # Falsification: an id beyond the grown virgin array is unseen, not old.
    shm = ShmCoverage(size=4096)
    _exec(shm, {1: 1})
    far = shm._virgin.size + 1000
    assert _exec(shm, {1: 1, far: 1})[1] == frozenset({far})
