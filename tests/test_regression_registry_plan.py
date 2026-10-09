"""REGISTRY.available() runs a precomputed plan with one batched trickle draw.

Per exec it re-classified every spec and drew ``rng.random()`` once per
never-seen format (~49 draws, ~87 us per build_ops on a default fuzzer).
The plan is built once per registration change and the trickle is one
``random_sequential(k)`` call: the same stream, so seeded runs match.
"""

import pytest

from fuzzer_tool.core import operator_registry as reg
from fuzzer_tool.core.operator_registry import REGISTRY, OperatorSpec
from fuzzer_tool.core.rand_pool import RandPool
from tests.support.operator_env import make_minimal_fuzzer as make_fuzzer
from tests.test_regression_registry_one_context import _per_spec

_PLAIN = b"plain text"
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40


class _SpyRng:
    """Delegates to a RandPool; counts scalar and batched float draws."""

    def __init__(self, seed):
        self._pool = RandPool(seed)
        self.scalar = 0
        self.batches: list[int] = []

    def random(self):
        self.scalar += 1
        return self._pool.random()

    def random_sequential(self, count):
        self.batches.append(count)
        return self._pool.random_sequential(count)


class _ScalarOnlyRng:
    """Old-style mock: random() only."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


def _unseen_formats(fuzzer, data) -> int:
    live = getattr(fuzzer, "_live_formats", None) or set()
    return sum(
        1 for n, s in reg._FORMAT_SNIFFERS.items() if n not in live and not (data and s(data))
    )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("data", [b"", _PLAIN, _PNG, b"PK\x03\x04" + b"a" * 30])
def test_matches_per_spec_oracle(seed, data):
    """Falsification: same ops, same order, same rng stream as per-spec."""
    a, b = make_fuzzer(seed=seed), make_fuzzer(seed=seed)
    assert REGISTRY.available(a, data) == _per_spec(b, data)
    assert a._rng.random() == b._rng.random()  # streams still aligned


def test_control_oracle_matches_itself():
    """Rule 46: the per-spec oracle agrees with a second run of itself."""
    a, b = make_fuzzer(seed=3), make_fuzzer(seed=3)
    assert _per_spec(a, _PLAIN) == _per_spec(b, _PLAIN)


def test_one_trickle_batch_per_call():
    """Falsification: k unseen formats -> one random_sequential(k), no scalars."""
    f = make_fuzzer(seed=1)
    f._rng = _SpyRng(1)
    k = _unseen_formats(f, _PLAIN)
    REGISTRY.available(f, _PLAIN)
    assert (f._rng.scalar, f._rng.batches) == (0, [k])


def test_plan_built_once(monkeypatch):
    """Falsification: repeated calls reuse the plan."""
    calls = {"n": 0}
    real = REGISTRY._build_plan

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(REGISTRY, "_build_plan", counting)
    REGISTRY._plan_key = None
    f = make_fuzzer(seed=1)
    for _ in range(5):
        REGISTRY.available(f, _PLAIN)
    assert calls["n"] == 1


def test_register_and_pop_invalidate_plan():
    """Adversarial: tests pop REGISTRY._ops directly; a stale plan would
    keep offering the removed op."""
    name = "plan_probe_op"
    REGISTRY.register(OperatorSpec(name=name, category="bit", handler_name="_op_x"))
    try:
        assert name in REGISTRY.available(make_fuzzer(seed=1), _PLAIN)
    finally:
        REGISTRY._ops.pop(name, None)
    assert name not in REGISTRY.available(make_fuzzer(seed=1), _PLAIN)


@pytest.mark.parametrize("value, offered", [(0.0, True), (0.99, False)])
def test_scalar_only_rng_falls_back(value, offered):
    """Adversarial: a mock without random_sequential still gets the trickle."""
    f = make_fuzzer(seed=1)
    f._rng = _ScalarOnlyRng(value)
    k = _unseen_formats(f, _PLAIN)
    ops = REGISTRY.available(f, _PLAIN)
    assert f._rng.calls == k
    assert ("png_chunk_mutate" in ops) is offered


def test_no_rng_offers_every_format():
    """Adversarial: rng None keeps the permissive old behaviour."""
    f = make_fuzzer(seed=1)
    f._rng = None
    ops = set(REGISTRY.available(f, _PLAIN))
    assert set(reg._FORMAT_SNIFFERS) <= ops
