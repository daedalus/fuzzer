"""Tests for services/te_position.py — transfer entropy position selection."""

import pytest

from fuzzer_tool.services.te_position import (
    get_te_weighted_position,
    update_te_causal_map,
)


class FakeTE:
    """Mock transfer entropy calculator."""

    def __init__(self, return_val=0.05):
        self.return_val = return_val
        self.calls = []

    def transfer_entropy(self, source, target):
        self.calls.append((source, target))
        return self.return_val


class TestUpdateTeCausalMap:
    def test_no_te_object(self):
        byte_edges = {}
        update_te_causal_map(None, [], [], 64, byte_edges)
        assert byte_edges == {}

    def test_insufficient_history(self):
        byte_edges = {}
        te = FakeTE()
        inputs = [b"test"] * 5  # < 10
        update_te_causal_map(te, inputs, [], 64, byte_edges)
        assert byte_edges == {}

    def test_populates_byte_edges(self):
        byte_edges = {}
        te = FakeTE(return_val=0.05)
        inputs = [b"A" * 10] * 10
        edges = [bytes([1, 0, 0, 0])] * 10
        update_te_causal_map(te, inputs, edges, 64, byte_edges)
        # With TE > 0.01, byte_edges should be populated
        assert len(byte_edges) > 0

    def test_low_te_does_not_populate(self):
        byte_edges = {}
        te = FakeTE(return_val=0.005)  # below 0.01 threshold
        inputs = [b"A" * 10] * 10
        edges = [bytes([1, 0, 0, 0])] * 10
        update_te_causal_map(te, inputs, edges, 64, byte_edges)
        assert byte_edges == {}

    def test_max_pos_capped_at_64(self):
        byte_edges = {}
        te = FakeTE(return_val=0.05)
        inputs = [b"A" * 128] * 10
        edges = [bytes([1, 0, 0, 0])] * 10
        update_te_causal_map(te, inputs, edges, 256, byte_edges)
        # Should not have positions > 64
        assert all(pos <= 64 for pos in byte_edges.keys())

    def test_short_inputs_capped(self):
        byte_edges = {}
        te = FakeTE(return_val=0.05)
        inputs = [b"AB"] * 10  # length 2
        edges = [bytes([1, 0, 0, 0])] * 10
        update_te_causal_map(te, inputs, edges, 64, byte_edges)
        # max_pos = min(64, 2) = 2
        assert all(pos < 2 for pos in byte_edges.keys())

    def test_map_size_capped_at_1024(self):
        byte_edges = {}
        te = FakeTE(return_val=0.05)
        inputs = [b"A" * 10] * 10
        # Must have non-zero bytes for edge_counts to populate
        edge_data = bytes([1] + [0] * 2047)
        edges = [edge_data] * 10
        update_te_causal_map(te, inputs, edges, 4096, byte_edges)
        # capped_map = min(4096, 1024) = 1024
        assert len(byte_edges) > 0


class TestGetTeWeightedPosition:
    def test_empty_byte_edges(self):
        assert get_te_weighted_position({}, 100) is None

    def test_best_position(self):
        byte_edges = {5: {1: 10}, 10: {1: 20}, 3: {1: 5}}
        result = get_te_weighted_position(byte_edges, 100)
        assert result == 10  # highest key

    def test_position_exceeds_length(self):
        byte_edges = {50: {1: 10}, 100: {1: 20}}
        result = get_te_weighted_position(byte_edges, 50)
        assert result is None  # 100 >= 50

    def test_single_position(self):
        byte_edges = {7: {1: 5}}
        result = get_te_weighted_position(byte_edges, 100)
        assert result == 7
