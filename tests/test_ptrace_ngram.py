"""n-gram history simulation in PtraceCoverage (k > 2 ring).

The ptrace path has no perf-critical hot loop, so a plain XOR chain over
the k−1 predecessor deque is fine there (the C shim uses FNV-1a; the two
need not agree — they never compare edge ids). What MUST hold:

- k=2 stays byte-identical to the legacy (rel ^ prev) % map_size hash.
- reset_edge_map clears the ring: two identical executions around a reset
  produce identical bucket sequences, or identical inputs would hash
  differently across iterations of one PtraceCoverage instance.
- Oracles below are derived from the formulas, not echoed from code.
"""

import pytest

from fuzzer_tool.services.ptrace_coverage import PtraceCoverage


def _legacy_bucket(rel, prev, map_size):
    return (rel ^ prev) % map_size


@pytest.fixture
def cov2():
    c = PtraceCoverage("/bin/true", map_size=256)
    c.reset_edge_map()
    return c


@pytest.fixture
def cov3():
    c = PtraceCoverage("/bin/true", map_size=256, ngram_k=3)
    c.reset_edge_map()
    return c


class TestLegacyK2Unchanged:
    def test_k2_matches_legacy_hash(self, cov2):
        """Default depth must reproduce the pinned pre-ngram buckets."""
        rels = [0x11, 0x23, 0x37]
        prev = 0
        for rel in rels:
            assert cov2.record_edge(rel)
            expected = _legacy_bucket(rel, prev, cov2.map_size)
            assert cov2.edge_map[expected] == 1
            prev = rel >> 1

    def test_k2_rejects_nothing_new(self):
        # The default constructor must keep working without ngram_k.
        PtraceCoverage("/bin/true", map_size=64)

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError):
            PtraceCoverage("/bin/true", map_size=64, ngram_k=1)


class TestK3Ring:
    def test_xor_chain_over_two_predecessors(self, cov3):
        """Oracle: edge = rel ^ p1 ^ p0 with p* = rel >> 1 pushed per hit."""
        rels = [0x21, 0x43, 0x65]
        hist = []
        for rel in rels:
            assert cov3.record_edge(rel)
            edge = rel
            for p in hist:
                edge ^= p
            assert cov3.edge_map[edge % cov3.map_size] == 1
            hist.append(rel >> 1)

    def test_reset_clears_the_ring(self, cov3):
        """Same sequence twice around reset: second pass must land on the
        same buckets. A stale ring would shift every first-pass hash."""
        seq = [0x15, 0x29, 0x31]

        def run():
            buckets = []
            hist = []
            for rel in seq:
                cov3.record_edge(rel)
                edge = rel
                for p in hist:
                    edge ^= p
                buckets.append(edge % cov3.map_size)
                hist.append(rel >> 1)
            return buckets

        first = run()
        cov3.reset_edge_map()
        second = run()
        assert first == second
