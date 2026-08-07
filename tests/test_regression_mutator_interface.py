"""Regression tests for the class-based mutator interface (Port 3, wtf Mutator_t)."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.mutator_interface import MutatorBase
from fuzzer_tool.core.operator_registry import OperatorRegistry, OperatorSpec


class _Rng:
    def __init__(self, seed: int = 7):
        self._r = random.Random(seed)

    def randint(self, a, b):
        return self._r.randint(a, b)

    def choice(self, seq):
        return self._r.choice(seq)


class _UpperMutator(MutatorBase):
    name = "upper_test"
    category = "byte"

    def mutate(self, data, rng, max_len=0, **ctx):
        return data.upper()


class _DecliningMutator(MutatorBase):
    name = "declining_test"
    category = "byte"

    def mutate(self, data, rng, max_len=0, **ctx):
        return None


class _GrowingMutator(MutatorBase):
    """Deliberately ignores max_len — the adapter must clamp it."""

    name = "growing_test"
    category = "byte"

    def mutate(self, data, rng, max_len=0, **ctx):
        return data * 100


class _RaisingMutator(MutatorBase):
    name = "raising_test"
    category = "byte"

    def mutate(self, data, rng, max_len=0, **ctx):
        raise RuntimeError("boom")


class _CountingMutator(MutatorBase):
    name = "counting_test"
    category = "adaptive"

    def __init__(self):
        self.coverage_events = []

    def mutate(self, data, rng, max_len=0, **ctx):
        return data + b"!"

    def on_new_coverage(self, seed, new_edges):
        self.coverage_events.append((seed, new_edges))


class _GatedMutator(MutatorBase):
    name = "gated_test"
    category = "byte"

    def mutate(self, data, rng, max_len=0, **ctx):
        return data[::-1]

    def is_available(self, fuzzer, data):
        return getattr(fuzzer, "gate_open", False)


class _FakeFuzzer:
    max_len = 64
    gate_open = False

    def __init__(self):
        self._rand_pool = _Rng()


class _FakeEngine:
    def __init__(self):
        self.f = _FakeFuzzer()


class TestMutatorBaseContract:
    def test_cannot_instantiate_without_mutate(self):
        class Incomplete(MutatorBase):
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_on_new_coverage_defaults_to_noop(self):
        _UpperMutator().on_new_coverage(b"seed", 3)  # must not raise

    def test_is_available_defaults_true(self):
        assert _UpperMutator().is_available(_FakeFuzzer(), b"data") is True


class TestRegistration:
    def test_registers_and_appears_in_names(self):
        reg = OperatorRegistry()
        reg.register_mutator(_UpperMutator())
        assert "upper_test" in reg.names()

    def test_category_is_honored(self):
        reg = OperatorRegistry()
        reg.register_mutator(_UpperMutator())
        assert reg.category_of("upper_test") == "byte"
        assert "upper_test" in reg.categories()["byte"]

    def test_rejects_nameless_mutator(self):
        class Nameless(MutatorBase):
            def mutate(self, data, rng, max_len=0, **ctx):
                return data

        reg = OperatorRegistry()
        with pytest.raises(ValueError, match="no name"):
            reg.register_mutator(Nameless())

    def test_rejects_non_mutator_object(self):
        class Bogus:
            name = "bogus"

        reg = OperatorRegistry()
        with pytest.raises(TypeError, match="mutate"):
            reg.register_mutator(Bogus())

    def test_duplicate_registration_rejected(self):
        reg = OperatorRegistry()
        reg.register_mutator(_UpperMutator())
        with pytest.raises(ValueError, match="duplicate"):
            reg.register_mutator(_UpperMutator())

    def test_mutators_listed_separately_from_function_ops(self):
        reg = OperatorRegistry()
        reg.register(OperatorSpec(name="fn_op", category="byte", handler_name="_op_fn_op"))
        m = _UpperMutator()
        reg.register_mutator(m)
        assert reg.mutators() == [m]


class TestDispatchAdapter:
    def _dispatch(self, mutator):
        reg = OperatorRegistry()
        reg.register_mutator(mutator)
        engine = _FakeEngine()
        return reg.dispatch(engine)[mutator.name], engine

    def test_adapter_presents_op_handler_signature(self):
        handler, _ = self._dispatch(_UpperMutator())
        out = handler(bytearray(b"abc"), 0, b"abc")
        assert bytes(out) == b"ABC"

    def test_returns_none_when_mutator_declines(self):
        handler, _ = self._dispatch(_DecliningMutator())
        assert handler(bytearray(b"abc"), 0, b"abc") is None

    def test_returns_none_when_output_unchanged(self):
        """An operator that returns its input unchanged is a no-op; the
        adapter normalizes that to None like the _op_* convention."""
        handler, _ = self._dispatch(_UpperMutator())
        assert handler(bytearray(b"ABC"), 0, b"ABC") is None

    def test_clamps_to_max_len(self):
        """The adapter must enforce max_len centrally — a mutator that
        ignores it must not be able to grow the buffer without bound."""
        handler, engine = self._dispatch(_GrowingMutator())
        out = handler(bytearray(b"abcd"), 0, b"abcd")
        assert len(out) <= engine.f.max_len

    def test_mutator_exception_is_contained(self):
        """A third-party mutator raising must not break the fuzz loop."""
        handler, _ = self._dispatch(_RaisingMutator())
        assert handler(bytearray(b"abc"), 0, b"abc") is None

    def test_handler_name_is_readable(self):
        handler, _ = self._dispatch(_UpperMutator())
        assert handler.__name__ == "_op_upper_test"

    def test_function_ops_still_dispatch_normally(self):
        """Mixed table: the class-based path must not disturb the
        existing getattr-based resolution."""
        reg = OperatorRegistry()
        reg.register(OperatorSpec(name="fn_op", category="byte", handler_name="_op_fn_op"))
        reg.register_mutator(_UpperMutator())

        class Engine(_FakeEngine):
            def _op_fn_op(self, buf, byte_idx, data):
                return bytearray(b"from_function")

        table = reg.dispatch(Engine())
        assert bytes(table["fn_op"](bytearray(b"x"), 0, b"x")) == b"from_function"
        assert bytes(table["upper_test"](bytearray(b"y"), 0, b"y")) == b"Y"


class TestAvailabilityGating:
    def test_is_available_drives_registry_availability(self):
        reg = OperatorRegistry()
        reg.register_mutator(_GatedMutator())
        f = _FakeFuzzer()

        assert "gated_test" not in reg.available(f, b"data")
        f.gate_open = True
        assert "gated_test" in reg.available(f, b"data")


class TestCoverageFeedback:
    def test_notify_reaches_registered_mutators(self):
        reg = OperatorRegistry()
        m = _CountingMutator()
        reg.register_mutator(m)
        reg.notify_new_coverage(b"seed", 5)
        assert m.coverage_events == [(b"seed", 5)]

    def test_notify_survives_a_raising_hook(self):
        """One bad hook must not stop the others from being notified."""

        class Exploding(MutatorBase):
            name = "exploding_test"

            def mutate(self, data, rng, max_len=0, **ctx):
                return data

            def on_new_coverage(self, seed, new_edges):
                raise RuntimeError("boom")

        reg = OperatorRegistry()
        reg.register_mutator(Exploding())
        good = _CountingMutator()
        reg.register_mutator(good)

        reg.notify_new_coverage(b"seed", 1)  # must not raise
        assert good.coverage_events == [(b"seed", 1)]


class TestGlobalRegistryUnaffected:
    def test_builtin_registry_has_no_mutators_by_default(self):
        """Port 3 is opt-in scaffolding: registering nothing must leave
        the shipped operator table exactly as it was."""
        from fuzzer_tool.core.operator_registry import REGISTRY

        assert REGISTRY.mutators() == []
