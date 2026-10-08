"""KatzChannel.observe: one execution's node bitmap read sparsely from SHM.

The bitmap holds one bit per ICFG node (3.16M on ffmpeg) and an execution
sets ~900 of them. ``observe`` must give exactly what the dense path
(unpack every bit, then ``record``) gave, without its per-exec full-width
temporaries. Expected values below are derived from the planted node
indices by plain arithmetic, never from the channel itself.
"""

import ctypes

import numpy as np
import pytest

from fuzzer_tool.adapters.shm import NodeBitmapShm
from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.services.katz_channel import KatzChannel

# Not a multiple of 8 or 64: the last byte and last word are partial.
N_NODES = 203


def _channel(n=N_NODES):
    src = np.arange(n - 1, dtype=np.int64)
    dst = src + 1
    icfg = InterproceduralCFG(
        [0x10 * i for i in range(n)], [f"f{i}" for i in range(n)], src, dst, {}
    )
    ch = KatzChannel(icfg, {k: k for k in range(n)})
    ch.bmp = NodeBitmapShm(num_nodes=n)
    return ch


@pytest.fixture
def channels():
    made = []

    def make(n=N_NODES):
        ch = _channel(n)
        made.append(ch)
        return ch

    yield make
    for ch in made:
        ch.cleanup()


def _plant(ch, nodes):
    """Set node bits the way the shim does: bits[i >> 3] |= 1 << (i & 7)."""
    for i in nodes:
        cell = ctypes.c_uint8.from_address(ch.bmp._ptr + 4 + (i >> 3))
        cell.value |= 1 << (i & 7)


def _packed(nodes, n=N_NODES):
    out = bytearray((n + 7) // 8)
    for i in nodes:
        out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def _hits(nodes, n=N_NODES):
    counts = [0.0] * n
    for i in nodes:
        counts[i] += 1.0
    return counts


def _payload(ch):
    return ctypes.string_at(ch.bmp._ptr + 4, ch.bmp.size_bytes)


class TestObserve:
    def test_counts_and_mask_match_planted_nodes(self, channels):
        ch = channels()
        nodes = [0, 7, 8, 63, 64, 65, 127, 200, 202]
        _plant(ch, nodes)
        assert ch.observe(seed_key="s") is True
        assert ch.hit_counts.tolist() == _hits(nodes)
        assert ch._masks["s"] == _packed(nodes)
        assert ch.exec_count == 1

    def test_clears_bitmap_after_read(self, channels):
        ch = channels()
        _plant(ch, [5, 150])
        ch.observe()
        assert _payload(ch) == bytes(ch.bmp.size_bytes)

    def test_mask_or_merges_across_executions(self, channels):
        ch = channels()
        _plant(ch, [1, 100])
        ch.observe(seed_key="s")
        _plant(ch, [100, 201])
        ch.observe(seed_key="s")
        assert ch._masks["s"] == _packed([1, 100, 201])
        assert ch.hit_counts.tolist() == _hits([1, 100, 100, 201])

    def test_no_seed_key_counts_without_mask(self, channels):
        ch = channels()
        _plant(ch, [42])
        ch.observe()
        assert ch._masks == {}
        assert ch.hit_counts[42] == 1.0

    def test_matches_dense_record(self, channels):
        """Same stream through observe and through the dense record path."""
        rng = np.random.default_rng(7)
        sparse, dense = channels(), channels()
        for step in range(20):
            nodes = sorted(set(rng.integers(0, N_NODES, size=int(rng.integers(1, 40))).tolist()))
            key = f"k{step % 3}" if step % 2 else None
            _plant(sparse, nodes)
            sparse.observe(seed_key=key)
            bits = np.zeros(N_NODES, dtype=bool)
            bits[nodes] = True
            dense.record(bits, seed_key=key)
        assert sparse.hit_counts.tolist() == dense.hit_counts.tolist()
        assert sparse._masks == dense._masks
        assert sparse.exec_count == dense.exec_count

    def test_control_dense_against_itself(self, channels):
        """Hard Rule 46: the dense oracle must agree with a second run of itself."""
        a, b = channels(), channels()
        for nodes in ([3, 9], [9, 202], [0]):
            bits = np.zeros(N_NODES, dtype=bool)
            bits[nodes] = True
            a.record(bits, seed_key="s")
            b.record(bits.copy(), seed_key="s")
        assert a.hit_counts.tolist() == b.hit_counts.tolist()
        assert a._masks == b._masks


class TestObserveFalsification:
    def test_empty_bitmap_records_nothing(self, channels):
        ch = channels()
        assert ch.observe(seed_key="s") is False
        assert ch.exec_count == 0
        assert ch._masks == {}
        assert not ch.hit_counts.any()

    def test_without_shm_is_false(self):
        ch = _channel()
        ch.cleanup()
        ch.bmp = None
        assert ch.observe(seed_key="s") is False
        assert ch.exec_count == 0


class TestObserveAdversarial:
    def test_padding_bits_past_last_node_are_ignored(self, channels):
        """The shim bounds-checks against size_bytes * 8, not the node count,
        so bits 203..207 of the last byte are reachable; they name no node."""
        ch = channels()
        last = ctypes.c_uint8.from_address(ch.bmp._ptr + 4 + ch.bmp.size_bytes - 1)
        last.value = 0xFF  # nodes 200..202 plus five padding bits
        assert ch.observe() is True
        assert ch.hit_counts.tolist() == _hits([200, 201, 202])

    def test_only_padding_bits_records_nothing(self, channels):
        ch = channels()
        last = ctypes.c_uint8.from_address(ch.bmp._ptr + 4 + ch.bmp.size_bytes - 1)
        last.value = 0xF8  # padding only (bits 203..207)
        assert ch.observe(seed_key="s") is False
        assert ch.exec_count == 0
        assert ch._masks == {}

    def test_every_node_set(self, channels):
        ch = channels()
        _plant(ch, range(N_NODES))
        ch.observe(seed_key="s")
        assert ch.hit_counts.tolist() == [1.0] * N_NODES
        assert ch._masks["s"] == _packed(range(N_NODES))

    def test_single_node_graph(self, channels):
        ch = channels(1)
        _plant(ch, [0])
        assert ch.observe(seed_key="s") is True
        assert ch.hit_counts.tolist() == [1.0]
        assert ch._masks["s"] == b"\x01"
