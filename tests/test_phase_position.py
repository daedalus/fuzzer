"""Tests for record-phase extrapolation of the TE causal byte map."""

from __future__ import annotations

import pytest

from fuzzer_tool.services.te_position import get_phase_weighted_position
from tests.support.scripted_rng import ScriptedRng

STRIDE = 16
HOT = 5


def _locked_map(records: int = 12, hits: int = 40) -> dict[int, dict[int, int]]:
    """Causal map with every observation at offset HOT of a STRIDE record."""
    return {HOT + STRIDE * k: {100 + k: hits} for k in range(records)}


def _flat_map(positions: int = 32, hits: int = 40) -> dict[int, dict[int, int]]:
    """Causal map whose offsets sweep every residue of the record."""
    return {3 * k: {200 + k: hits} for k in range(positions)}


def test_locked_map_extrapolates_past_the_te_window():
    # The TE map only ever observes offsets below its own 64-byte cap. A
    # confirmed lock must be able to name a position far beyond it.
    rng = ScriptedRng(randints=[60])
    pos = get_phase_weighted_position(_locked_map(), 4096, STRIDE, rng)
    assert pos == 60 * STRIDE + HOT
    assert pos > 64


def test_first_record_is_reachable():
    rng = ScriptedRng(randints=[0])
    assert get_phase_weighted_position(_locked_map(), 4096, STRIDE, rng) == HOT


def test_scattered_map_returns_none():
    # FALSIFICATION: without the significance gate this would happily return
    # a position derived from a mean phase that means nothing.
    rng = ScriptedRng(randints=[3])
    assert get_phase_weighted_position(_flat_map(), 4096, STRIDE, rng) is None


def test_no_stride_returns_none():
    rng = ScriptedRng(randints=[3])
    assert get_phase_weighted_position(_locked_map(), 4096, None, rng) is None


def test_stride_one_returns_none():
    rng = ScriptedRng(randints=[3])
    assert get_phase_weighted_position(_locked_map(), 4096, 1, rng) is None


def test_empty_map_returns_none():
    rng = ScriptedRng(randints=[3])
    assert get_phase_weighted_position({}, 4096, STRIDE, rng) is None


def test_buffer_shorter_than_the_hot_offset_returns_none():
    # ADVERSARIAL: the lock is real but no slot fits, and an ungated
    # `record * stride + offset` would index past the buffer.
    rng = ScriptedRng(randints=[0])
    assert get_phase_weighted_position(_locked_map(), HOT, STRIDE, rng) is None


def test_returned_position_is_always_inside_the_buffer():
    # ADVERSARIAL: the RNG is asked for the last slot; it must still fit.
    for buf_len in (6, 7, 21, 22, 37, 100, 1000):
        slots = (buf_len - HOT + STRIDE - 1) // STRIDE
        if slots <= 0:
            continue
        rng = ScriptedRng(randints=[slots - 1])
        pos = get_phase_weighted_position(_locked_map(), buf_len, STRIDE, rng)
        assert pos is not None
        assert 0 <= pos < buf_len, f"buf_len={buf_len} pos={pos}"


def test_rng_is_asked_for_a_slot_index_not_a_byte():
    # Pinning the contract: the draw is over records, so the scripted value
    # is multiplied by the stride rather than used as the answer.
    rng = ScriptedRng(randints=[2])
    assert get_phase_weighted_position(_locked_map(), 4096, STRIDE, rng) == 2 * STRIDE + HOT


def test_dominant_single_position_returns_none():
    # ADVERSARIAL: one position holding almost all the edge weight is
    # phase-locked with itself. Kish's effective n must reject it.
    causal = _locked_map()
    causal[HOT] = {999: 10**6}
    rng = ScriptedRng(randints=[1])
    assert get_phase_weighted_position(causal, 4096, STRIDE, rng) is None


def test_alpha_gates_the_decision():
    causal = _locked_map(records=3)
    rng_strict = ScriptedRng(randints=[1])
    rng_loose = ScriptedRng(randints=[1])
    assert get_phase_weighted_position(causal, 4096, STRIDE, rng_strict, alpha=0.001) is None
    assert get_phase_weighted_position(causal, 4096, STRIDE, rng_loose, alpha=0.5) is not None


def test_does_not_consume_rng_when_it_declines():
    # An empty script makes any draw a StopIteration, so this asserts the
    # declining paths leave the shared RNG stream untouched.
    for causal, stride in ((_flat_map(), STRIDE), (_locked_map(), None), ({}, STRIDE)):
        assert get_phase_weighted_position(causal, 4096, stride, ScriptedRng()) is None


def test_weights_are_total_edge_hits_per_position():
    # Two positions, antipodal on the record, so the mean phase lands on
    # whichever carries more total hits -- but the split is even enough that
    # effective n stays usable.
    causal = {HOT: {1: 30, 2: 30, 3: 30}, HOT + STRIDE // 2: {4: 25, 5: 25}}
    rng = ScriptedRng(randints=[0])
    pos = get_phase_weighted_position(causal, 4096, STRIDE, rng, alpha=0.9)
    assert pos == HOT


def test_stride_larger_than_the_buffer_returns_none():
    rng = ScriptedRng(randints=[0])
    assert get_phase_weighted_position(_locked_map(), 4, 4096, rng) is None


@pytest.mark.parametrize("bad", [0, -1, -16])
def test_non_positive_stride_returns_none(bad):
    assert get_phase_weighted_position(_locked_map(), 4096, bad, ScriptedRng()) is None


class _MemoFuzzer:
    """Minimal stand-in exposing only what the phase-lock memo reads."""

    def __init__(self, byte_edges):
        self._te_byte_edges = byte_edges
        self._te_causal_version = 0
        self._rng = ScriptedRng(randints=[0] * 64)


def _reporter(byte_edges):
    from fuzzer_tool.core.rand_pool import RandPool
    from fuzzer_tool.services.stats import StatsReporter

    return StatsReporter(_MemoFuzzer(byte_edges), rng=RandPool())


class TestPhaseLockMemo:
    def test_lock_is_computed_once_per_causal_version(self):
        rep = _reporter(_locked_map())
        rep.get_phase_weighted_position(4096, STRIDE)
        first = rep._phase_lock
        rep.get_phase_weighted_position(4096, STRIDE)
        assert rep._phase_lock is first

    def test_causal_map_update_invalidates_the_memo(self):
        # FALSIFICATION: a memo keyed on the stride alone would keep serving
        # a lock computed from causal evidence that has since been replaced.
        rep = _reporter(_locked_map())
        rep.get_phase_weighted_position(4096, STRIDE)
        assert rep._phase_lock is not None

        rep.f._te_byte_edges.clear()
        rep.f._te_byte_edges.update(_flat_map())
        rep.f._te_causal_version += 1
        assert rep.get_phase_weighted_position(4096, STRIDE) is None

    def test_stride_change_invalidates_the_memo(self):
        # ADVERSARIAL: two seeds with different inferred strides interleave.
        rep = _reporter(_locked_map())
        assert rep.get_phase_weighted_position(4096, STRIDE) is not None
        assert rep._phase_lock is not None
        assert rep.get_phase_weighted_position(4096, 1) is None
        assert rep._phase_lock is None

    def test_update_te_causal_map_bumps_the_version(self):
        # The memo is only correct if the producer actually bumps the key.
        rep = _reporter({})
        rep.f._te = None
        rep.f._te_input_history = []
        rep.f._te_edge_history = []
        rep.f.map_size = 1 << 16
        before = rep.f._te_causal_version
        rep.update_te_causal_map()
        assert rep.f._te_causal_version == before + 1


class _SelectFuzzer:
    """Mock exposing the position sources `select_position` consults."""

    def __init__(self, stride, te_pos=3):
        from fuzzer_tool.core.rand_pool import RandPool

        self.max_len = 1 << 20
        self._rng = RandPool(seed=1234)
        self._use_transfer_entropy = True
        self._te = object()
        self._use_mi = False
        self._mi = None
        self._use_sensitivity = False
        self._sensitivity = None
        self._crash_mi = None
        self._use_region_profile = False
        self.seed_meta = {}
        self._stride = stride
        self._te_pos = te_pos
        self.phase_calls = []

    def _get_te_weighted_position(self, _n):
        return self._te_pos

    def _get_phase_weighted_position(self, input_length, stride):
        self.phase_calls.append((input_length, stride))
        if stride is None:
            return None
        return HOT + STRIDE  # a second-record slot, past the TE window


class TestSelectPositionWiring:
    def _engine(self, fuzzer):
        from fuzzer_tool.services.operators import OperatorEngine

        return OperatorEngine(fuzzer)

    def test_stride_reaches_the_position_source(self):
        f = _SelectFuzzer(STRIDE)
        seed = bytes(4096)
        f.seed_meta[seed] = {"record_stride": STRIDE}
        self._engine(f).select_position(bytearray(seed), seed)
        assert f.phase_calls == [(4096, STRIDE)]

    def test_seed_without_stride_passes_none(self):
        f = _SelectFuzzer(None)
        seed = bytes(4096)
        f.seed_meta[seed] = {"record_stride": None}
        self._engine(f).select_position(bytearray(seed), seed)
        assert f.phase_calls == [(4096, None)]

    def test_unknown_seed_passes_none(self):
        # ADVERSARIAL: the buffer is mid-mutation, so `data` may not be a
        # corpus member with metadata at all.
        f = _SelectFuzzer(STRIDE)
        seed = bytes(4096)
        self._engine(f).select_position(bytearray(seed), seed)
        assert f.phase_calls == [(4096, None)]

    def test_phase_candidate_is_actually_reachable(self):
        # FALSIFICATION: if phase_pos were computed but left out of the
        # candidate list, the returned position could never be HOT+STRIDE.
        f = _SelectFuzzer(STRIDE)
        seed = bytes(4096)
        f.seed_meta[seed] = {"record_stride": STRIDE}
        engine = self._engine(f)
        drawn = {engine.select_position(bytearray(seed), seed) for _ in range(200)}
        assert HOT + STRIDE in drawn
        assert f._te_pos in drawn

    def test_not_consulted_when_transfer_entropy_is_off(self):
        f = _SelectFuzzer(STRIDE)
        f._use_transfer_entropy = False
        seed = bytes(4096)
        f.seed_meta[seed] = {"record_stride": STRIDE}
        self._engine(f).select_position(bytearray(seed), seed)
        assert f.phase_calls == []
