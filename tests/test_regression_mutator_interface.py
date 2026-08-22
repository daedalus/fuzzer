"""Regression tests for the class-based mutator interface (Port 3, wtf Mutator_t)."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase
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
    """Gates on a *declared* context field, not an arbitrary fuzzer attribute.

    It used to read ``fuzzer.gate_open``, which is the coupling this
    interface exists to prevent: a mutator that gates on any attribute it
    likes has the whole Fuzzer as its contract, and there is then nothing
    to keep stable. Dictionary presence is the shape the built-in
    availability predicates already use (see ``_AVAILABILITY`` in
    ``operator_registry``).
    """

    name = "gated_test"
    category = "byte"

    def mutate(self, data, rng, max_len=0, *, context=None, **ctx):
        return data[::-1]

    def is_available(self, context, data):
        return bool(context.dictionary)


class _ContextCapturingMutator(MutatorBase):
    """Records what the adapter handed it, so the projection is assertable."""

    name = "capture_test"
    category = "byte"

    def __init__(self):
        self.seen = []

    def mutate(self, data, rng, max_len=0, *, context=None, **ctx):
        self.seen.append((max_len, context, dict(ctx)))
        return data + b"!"


class _FakeCmplog:
    def __init__(self, pairs=()):
        self.pairs = list(pairs)


class _FakeFuzzer:
    max_len = 64

    def __init__(self, dictionary=(), cmplog_pairs=None, corpus=()):
        self._rand_pool = _Rng()
        self.dictionary = list(dictionary)
        self.corpus = list(corpus)
        self._cmplog = _FakeCmplog(cmplog_pairs) if cmplog_pairs is not None else None


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
        assert _UpperMutator().is_available(MutationContext(), b"data") is True


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
        f.dictionary.append(b"GET")
        assert "gated_test" in reg.available(f, b"data")

    def test_availability_predicate_receives_a_context_not_a_fuzzer(self):
        """The registry must not leak the fuzzer through the second door.

        ``mutate()`` and ``is_available()`` are two entry points onto the
        same state; narrowing only one of them would leave the coupling
        intact and harder to see.
        """
        seen = []

        class Recording(MutatorBase):
            name = "recording_avail"

            def mutate(self, data, rng, max_len=0, *, context=None, **ctx):
                return data

            def is_available(self, context, data):
                seen.append(context)
                return True

        reg = OperatorRegistry()
        reg.register_mutator(Recording())
        reg.available(_FakeFuzzer(dictionary=[b"A"]), b"data")

        assert seen and all(isinstance(c, MutationContext) for c in seen)
        assert seen[0].dictionary == [b"A"]


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


class TestMutationContext:
    """The narrow interface itself (port item P1-4).

    ``mutate()`` and ``is_available()`` used to be handed the whole
    ``Fuzzer``. These pin the replacement while it still has no
    implementors -- ``src/`` contains no ``MutatorBase`` subclass, which is
    the only reason the signature could be changed at all.
    """

    def test_constructible_with_no_fuzzer_at_all(self):
        """The point of the extraction: a mutator is testable without one.

        ``test_operator_smoke.py`` has to compile a target binary to reach
        ``_op_bit_flip``, because the operator's contract is the whole
        Fuzzer. Nothing here needs a process, a corpus directory or a
        compiler.
        """
        ctx = MutationContext(max_len=64, dictionary=[b"GET", b"POST"])
        assert ctx.max_len == 64
        assert ctx.dictionary == [b"GET", b"POST"]
        assert ctx.cmplog_pairs == ()
        assert ctx.corpus == ()

    def test_from_fuzzer_projects_the_four_fields(self):
        f = _FakeFuzzer(dictionary=[b"A"], cmplog_pairs=[(b"x", b"y")], corpus=[b"seed"])
        ctx = MutationContext.from_fuzzer(f)
        assert ctx.max_len == 64
        assert ctx.dictionary == [b"A"]
        assert ctx.cmplog_pairs == [(b"x", b"y")]
        assert ctx.corpus == [b"seed"]

    def test_from_fuzzer_tolerates_missing_state(self):
        """cmplog is absent whenever --cmplog is off, which is the default.

        A missing attribute must read as "no such state" rather than
        raising inside a third-party mutator's call path, where the
        adapter's except-clause would log it as the mutator's fault.
        """

        class Bare:
            pass

        ctx = MutationContext.from_fuzzer(Bare())
        assert ctx.max_len == 0
        assert ctx.dictionary == ()
        assert ctx.cmplog_pairs == ()
        assert ctx.corpus == ()

    def test_from_fuzzer_maps_none_cmplog_to_empty_pairs(self):
        ctx = MutationContext.from_fuzzer(_FakeFuzzer())
        assert ctx.cmplog_pairs == ()

    def test_context_carries_no_reference_to_the_fuzzer(self):
        """Not a style point: it is what makes the four fields the contract.

        If the instance kept the fuzzer anywhere reachable, a mutator
        could go around the interface and the declared surface would stop
        meaning anything.
        """
        ctx = MutationContext.from_fuzzer(_FakeFuzzer())
        assert set(MutationContext.__slots__) == {
            "max_len",
            "dictionary",
            "cmplog_pairs",
            "corpus",
        }
        assert not hasattr(ctx, "__dict__")
        with pytest.raises(AttributeError):
            ctx.smuggled = object()

    def test_adapter_passes_a_context_and_not_the_fuzzer(self):
        m = _ContextCapturingMutator()
        reg = OperatorRegistry()
        reg.register_mutator(m)
        engine = _FakeEngine()
        engine.f = _FakeFuzzer(dictionary=[b"tok"], corpus=[b"a", b"b"])

        reg.dispatch(engine)["capture_test"](bytearray(b"in"), 0, b"in")

        (max_len, context, extras) = m.seen[0]
        assert isinstance(context, MutationContext)
        assert "fuzzer" not in extras, "the Fuzzer is still being smuggled through **ctx"
        assert extras == {}
        assert max_len == context.max_len == 64
        assert context.dictionary == [b"tok"]
        assert context.corpus == [b"a", b"b"]

    def test_context_is_rebuilt_per_call_so_max_len_stays_current(self):
        """Adaptive length capping moves max_len mid-campaign.

        Caching the context on the adapter closure would freeze the cap at
        whatever it was when the operator table was built, and the clamp in
        the adapter would then enforce a stale bound -- silently, since a
        too-generous cap produces valid-looking oversized inputs.
        """
        m = _ContextCapturingMutator()
        reg = OperatorRegistry()
        reg.register_mutator(m)
        engine = _FakeEngine()
        engine.f = _FakeFuzzer()
        handler = reg.dispatch(engine)["capture_test"]

        handler(bytearray(b"in"), 0, b"in")
        engine.f.max_len = 8
        handler(bytearray(b"in"), 0, b"in")

        assert [seen[0] for seen in m.seen] == [64, 8]

    def test_adapter_clamps_to_the_refreshed_max_len(self):
        reg = OperatorRegistry()
        reg.register_mutator(_GrowingMutator())
        engine = _FakeEngine()
        engine.f = _FakeFuzzer()
        handler = reg.dispatch(engine)["growing_test"]

        assert len(handler(bytearray(b"ab"), 0, b"ab")) == 64
        engine.f.max_len = 8
        assert len(handler(bytearray(b"ab"), 0, b"ab")) == 8
