"""Regression: the 8-bit generation tag wraps every 256 resets.

An entry keeps the tag of the last execution in which its edge fired, and the
read path filters on ``(count >> 24) == read_generation()``.  With no wipe,
an edge that fired once and went quiet is therefore read as live again at
exactly reset 256, 512, 768 ... -- coverage credited to an execution that
never reached that code.

``afl_shim.c:786`` has carried the correct wipe since the generation scheme
landed, but ``__afl_map_reset()`` has never had a caller; the live reset is
``ShmCoverage.reset_edge_map()``, which is where the wipe belongs.

These assert the aliasing period exactly rather than "no ghosts eventually",
because the period is the diagnostic: a wipe on the wrong branch (say, every
reset, or at gen 1) still passes a vaguer test while destroying either the
performance win or the current generation's own entries.
"""

import ctypes

import pytest

from fuzzer_tool.adapters.shm import (
    SHM_METADATA_SIZE,
    ShmCoverage,
)

GEN_PERIOD = 256


@pytest.fixture
def cov():
    c = ShmCoverage(size=1024)
    yield c
    c.cleanup()


def _resets_where_live(c, edge_id, n):
    """Reset ``n`` times, returning the reset numbers at which *edge_id* reads live."""
    live_at = []
    for i in range(1, n + 1):
        c.reset_edge_map()
        if edge_id in c.get_edge_ids():
            live_at.append(i)
    return live_at


def test_quiet_edge_never_reappears_across_three_wraps(cov):
    cov.record_edge(4242)
    assert cov.get_edge_ids() == {4242}
    assert _resets_where_live(cov, 4242, 3 * GEN_PERIOD) == []


def test_wipe_lands_on_the_wrap_and_only_on_the_wrap(cov):
    """The table is zeroed when the tag returns to 0, and at no other reset."""
    cov.record_edge(4242)
    table = ctypes.string_at(cov._ptr + SHM_METADATA_SIZE, cov.table_bytes)
    assert table.count(b"\x00") != len(table), "planted edge should be in the table"

    zeroed_at = []
    for i in range(1, 2 * GEN_PERIOD + 1):
        cov.reset_edge_map()
        table = ctypes.string_at(cov._ptr + SHM_METADATA_SIZE, cov.table_bytes)
        if table.count(b"\x00") == len(table):
            zeroed_at.append(i)
        # Re-plant unconditionally: an empty table at the next check then
        # means a wipe happened at that reset, not that nothing was ever
        # there to wipe.
        cov.record_edge(4242)

    # 256 resets from gen 0 returns the tag to 0; 512 is the second wrap.
    assert zeroed_at == [GEN_PERIOD, 2 * GEN_PERIOD]


def test_wrap_preserves_the_header(cov):
    """The wipe covers the table only — the front header has one writer."""
    cov.record_edge(4242)
    ctypes.c_uint64.from_address(cov._ptr).value = 0xDEADBEEF  # path_hash
    diag_before = cov.read_diag() & 0x00FFFFFF  # ctx bits + drop count

    for _ in range(GEN_PERIOD):
        cov.reset_edge_map()

    assert cov.read_generation() == 0
    assert ctypes.c_uint64.from_address(cov._ptr).value == 0xDEADBEEF
    assert cov.read_diag() & 0x00FFFFFF == diag_before


def test_edge_refired_after_the_wipe_is_visible_again(cov):
    """The wipe bounds staleness; it does not blind the reader."""
    cov.record_edge(4242)
    for _ in range(GEN_PERIOD):
        cov.reset_edge_map()
    assert cov.get_edge_ids() == set()

    cov.record_edge(4242)
    assert cov.get_edge_ids() == {4242}
    assert cov.get_edge_counts()[4242] == 1


def test_wrap_does_not_strand_the_read_fast_path(cov):
    """Re-insertion after a wipe advances edge_count, so the next scan rescans.

    The fast path skips the table when edge_count and path_hash are both
    unchanged.  A wipe that left the header's cumulative edge_count untouched
    while emptying the table would make a re-fired edge invisible until some
    unrelated edge moved the counter.
    """
    cov.record_edge(4242)
    cov.is_new_coverage_with_edges()  # prime _last_edge_count / _last_path_hash
    count_before = cov.read_edge_count()

    for _ in range(GEN_PERIOD):
        cov.reset_edge_map()
    cov.record_edge(4242)

    assert cov.read_edge_count() > count_before
    _, ids = cov.is_new_coverage_with_edges()
    assert ids == {4242}
