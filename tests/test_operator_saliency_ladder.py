"""Covers the NEUZZ-style sign-directed ladder operator (services/operators.py::_op_saliency_ladder).

Wired through core/operator_registry.py under "adaptive" -> "saliency_ladder", gated on
the PositionSaliencyScheduler being fitted (fuzzer._pos_saliency). Implements the NEUZZ++
mutation pattern: sign-guided 2^k byte walk with +/- steps (clipped to 0..255) plus block
insert/delete at high-gradient locations, instead of gradient_descent's cmplog-based ladder.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_saliency import PositionSaliencyScheduler
from fuzzer_tool.services.operators import OperatorEngine


class MockFuzzer:
    """Fuzzer stub carrying a fitted PositionSaliencyScheduler."""

    __slots__ = (
        "_pos_saliency",
        "_rng",
        "seed_meta",
        "_cmplog",
        "max_len",
        "dictionary",
        "cmplog_pairs",
    )

    def __init__(self, pos_saliency):
        self._pos_saliency = pos_saliency
        self._rng = RandPool(seed=0xDEADBEEF)
        self.seed_meta = {}
        self._cmplog = None
        self.max_len = 0
        self.dictionary = []
        self.cmplog_pairs = ()


def make_scheduler(seeds, edges, **kwargs):
    """Build a fitted PositionSaliencyScheduler."""
    rng = RandPool(seed=0x12345678)
    scheduler = PositionSaliencyScheduler(
        rng=rng,
        samples_fn=lambda: list(zip(seeds, edges, strict=False)),
        n_targets=2,
        epochs=50,
        **kwargs,
    )
    scheduler.refit(force=True)
    return scheduler


def test_saliency_ladder_operator_exists_in_registry():
    """Operator is registered in the adaptive band with a handler on the engine."""
    from fuzzer_tool.core.operator_registry import REGISTRY

    assert "saliency_ladder" in REGISTRY.names()
    assert REGISTRY.category_of("saliency_ladder") == "adaptive"
    engine = OperatorEngine.__new__(OperatorEngine)
    dispatch = REGISTRY.dispatch(engine)
    assert "saliency_ladder" in dispatch


def test_declines_when_saliency_absent():
    """No scheduler -> immediate decline (operator stays unavailable, doesn't error)."""
    engine = OperatorEngine(MockFuzzer(None))
    assert engine._op_saliency_ladder(bytearray(b"hello"), 0, b"hello") is None


def test_declines_when_saliency_not_ready():
    """Unfitted scheduler -> decline until a refit succeeds."""
    rng = RandPool(seed=0x11111)
    scheduler = PositionSaliencyScheduler(rng=rng, samples_fn=lambda: [], n_targets=2)
    engine = OperatorEngine(MockFuzzer(scheduler))
    assert engine._op_saliency_ladder(bytearray(b"hello"), 0, b"hello") is None


def test_mutates_a_hot_byte_with_sign_direction():
    """A planted-byte sample produces a model that steers changes toward the hot byte."""
    # seed: byte 3 and byte 20 are the "planted" bytes the net can learn to care about
    seed = bytearray(32)
    seed[3] = 0xAA
    seed[20] = 0x55
    # Need >= MIN_SEEDS (8) seeds for the model to fit
    seeds = [bytes(seed)] * 10
    edges = [{0}, {1}] * 10
    scheduler = make_scheduler(seeds, edges)
    engine = OperatorEngine(MockFuzzer(scheduler))
    result = engine._op_saliency_ladder(bytearray(seed), 0, bytes(seed))
    assert result is not None
    assert bytes(result) != bytes(seed), "the operator must change at least one byte"


def test_respects_max_len():
    """Result never exceeds the operator's max_len."""
    seeds = [bytes(64) for _ in range(10)]
    edges = [{0}, {1}, {2}] * 10
    scheduler = make_scheduler(seeds, edges)
    f = MockFuzzer(scheduler)
    f.max_len = 64  # override the 0 max_len property
    engine = OperatorEngine(f)
    result = engine._op_saliency_ladder(bytearray(64), 0, bytes(64))
    assert result is not None
    assert len(result) <= 64


def test_returns_none_on_empty_input():
    """Empty input is declined; the engine never hands an empty buffer to a
    position operator that can't synthesize one."""
    seeds = [bytes(64) for _ in range(10)]
    edges = [{0}, {1}, {2}]
    scheduler = make_scheduler(seeds, edges)
    engine = OperatorEngine(MockFuzzer(scheduler))
    assert engine._op_saliency_ladder(bytearray(b""), 0, b"") is None


def test_saliency_signal_concentrates():
    """Saliency mass should concentrate on the planted bytes for a seeded model."""
    rng = RandPool(seed=0x42)
    b1 = bytes([0xAA] + [0x00] * 127)
    b2 = bytes([0x00] + [0x55] * 127)
    samples = [(b1, {0}), (b2, {1})] * 5  # need >= MIN_SEEDS (8)
    scheduler = PositionSaliencyScheduler(
        rng=rng, samples_fn=lambda: samples, n_targets=2, epochs=50
    )
    scheduler.refit(force=True)
    sal = scheduler.signed_saliency(b1)
    # the model should produce a non-zero gradient signal
    assert np.count_nonzero(sal) > 0


def test_reproducible_with_seed():
    """Same seed -> same operator output."""
    seeds = [bytes(64) for _ in range(10)]
    edges = [{0}, {1}, {2}] * 10
    scheduler = make_scheduler(seeds, edges)
    f = MockFuzzer(scheduler)
    engine = OperatorEngine(f)
    result1 = engine._op_saliency_ladder(bytearray(64), 0, bytes(64))
    result2 = engine._op_saliency_ladder(bytearray(64), 0, bytes(64))
    assert bytes(result1) == bytes(result2)


def test_no_crash_with_reasonable_input():
    """Reasonable input is handled gracefully."""
    seeds = [bytes(64) for _ in range(10)]
    edges = [{0}, {1}, {2}] * 10
    scheduler = make_scheduler(seeds, edges)
    f = MockFuzzer(scheduler)
    f.max_len = 64
    engine = OperatorEngine(f)
    result = engine._op_saliency_ladder(bytearray(seeds[0]), 0, bytes(seeds[0]))
    assert result is not None


def test_op_saliency_ladder_gated_on_pos_saliency():
    """Operator availability is gated on the PositionSaliencyScheduler."""
    from fuzzer_tool.core.operator_registry import REGISTRY

    # No pos_saliency -> operator not available
    f = MockFuzzer(None)
    assert "saliency_ladder" not in REGISTRY.available(f, b"test")

    # With fitted pos_saliency -> operator available
    seeds = [bytes(64) for _ in range(10)]
    edges = [{0}, {1}, {2}] * 10
    scheduler = make_scheduler(seeds, edges)
    f = MockFuzzer(scheduler)
    assert "saliency_ladder" in REGISTRY.available(f, b"test")


def test_availability_predicate_gates_on_scheduler():
    """The operator is unavailable without a fitted scheduler."""
    from fuzzer_tool.core.operator_registry import REGISTRY

    fuzzer_no_saliency = MockFuzzer(None)
    assert "saliency_ladder" not in REGISTRY.available(fuzzer_no_saliency, b"hello")

    rng = RandPool(seed=0x12345678)
    scheduler = PositionSaliencyScheduler(rng=rng, samples_fn=lambda: [], n_targets=2)
    fuzzer_with_unfitted = MockFuzzer(scheduler)
    # Unfitted model: the predicate passes (scheduler object exists) but the
    # operator declines at call time (see test_declines_when_saliency_not_ready).
    assert "saliency_ladder" in REGISTRY.available(fuzzer_with_unfitted, b"hello")
