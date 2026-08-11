"""Regression: PtraceCoverage.record_edge shifts prev like the C shim.

record_edge hashed (rel ^ prev_location) % map_size with
prev_location = rel % map_size, so a block that re-entered itself
consecutively produced prev == rel and bucket = rel ^ rel == 0 —
self-looping blocks all aliased to bucket 0. The C shim's
__afl_prev_loc = cur_loc >> 1 avoids this, so ptrace now mirrors it
(prev = rel >> 1). Pinned here with oracles derived independently from
the hash formula.
"""

import pytest

from fuzzer_tool.services.ptrace_coverage import PtraceCoverage


@pytest.fixture
def cov():
    c = PtraceCoverage("/bin/true", map_size=256)
    c.reset_edge_map()
    return c


def _hash_rel(rel, prev, map_size):
    return (rel ^ prev) % map_size


def test_self_loop_no_longer_aliases_to_bucket_zero(cov):
    """A consecutively re-entered block hashes to rel ^ (rel >> 1), not 0."""
    rel = 5
    assert cov.record_edge(rel)  # prev=0 -> bucket = rel % map_size
    assert cov.edge_map[rel % cov.map_size] == 1
    assert cov.record_edge(rel)  # same block again -> prev = rel >> 1
    expected = _hash_rel(rel, rel >> 1, cov.map_size)
    assert expected != 0  # shim-parity shift keeps the self-loop non-degenerate
    assert cov.edge_map[expected] == 1
    assert cov.edge_map[0] == 0  # old behavior collapsed it to bucket 0


def test_prev_shift_matches_shim_for_consecutive_blocks(cov):
    """prev for the next edge is the previous rel shifted, like __afl_prev_loc."""
    rel1, rel2 = 0x1234, 0x2345
    cov.record_edge(rel1)
    assert cov.record_edge(rel2)  # new bucket per the shifted prev
    assert cov.edge_map[_hash_rel(rel2, rel1 >> 1, cov.map_size)] == 1
    # The just-hit block becomes the shifted prev for the next hash.
    assert cov.record_edge(rel2)
    assert cov.edge_map[_hash_rel(rel2, rel2 >> 1, cov.map_size)] == 1


def test_relative_addresses_from_base_use_shifted_prev(cov):
    """With _base_address set, rel = addr - base feeds the shifted prev."""
    cov._base_address = 0x1000
    addr1, addr2 = 0x1000 + 7, 0x1000 + 9
    cov.record_edge(addr1)
    assert cov.record_edge(addr2)
    assert cov.edge_map[_hash_rel(9, 7 >> 1, cov.map_size)] == 1
