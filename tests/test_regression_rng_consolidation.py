"""Regression tests for the ``self._rng`` consolidation.

Five defects landed together when the Hard Rule 16 migration renamed the
fuzzer's pool handle from ``_rand_pool`` to ``_rng`` without following
every reader, and a blind ``random.`` -> ``self._rng.`` rewrite hit
call sites that have no ``self``:

  1. ``MutationContext.from_fuzzer`` still read ``_rand_pool``, so every
     format operator saw ``ctx._rng is None``.
  2. ``operator_registry`` read ``f._rand_pool`` bare -- an AttributeError,
     not a soft None.
  3. ``BayesianEloTracker`` read ``self._rng`` that no ``__init__`` set.
  4. ``SeedPicker`` likewise, in the katz/tang/aflgo arms.
  5. ``_cdf_pick`` is a module-level function, so ``self._rng`` there was a
     NameError, and it called ``.choices`` which RandPool does not have.

Each test asserts the reachable symptom, not the spelling of the fix.
"""

import inspect
import math
import random

import pytest

from fuzzer_tool.core.elo import BayesianEloTracker
from fuzzer_tool.core.mutator_interface import MutationContext
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.seed_picker import SeedPicker, _cdf_pick


class _StubFuzzer:
    """Just the two attributes ``from_fuzzer`` needs for this assertion."""

    max_len = 64

    def __init__(self, pool):
        self._rng = pool


class TestContextCarriesThePool:
    def test_from_fuzzer_finds_the_pool(self):
        pool = RandPool(seed=1)
        assert MutationContext.from_fuzzer(_StubFuzzer(pool))._rng is pool

    def test_absent_pool_still_reads_as_none_not_crash(self):
        class Bare:
            max_len = 0

        assert MutationContext.from_fuzzer(Bare())._rng is None

    def test_no_stale_alias_survives_on_the_context(self):
        # The whole point of the consolidation: one name, not five.
        ctx = MutationContext.from_fuzzer(_StubFuzzer(RandPool(seed=1)))
        assert "_rng" in MutationContext.__slots__
        for dead in ("rand_pool", "_rand", "_randpool", "rng"):
            assert dead not in MutationContext.__slots__
            assert not hasattr(ctx, dead)


class TestBayesianEloHasAPool:
    def _trained(self, **kw):
        t = BayesianEloTracker(min_matches=0, **kw)
        for _ in range(20):
            t.record_match("a", "b", 1.0)
        return t

    def test_thompson_sampling_reaches_a_pool(self):
        # Adversarial: min_matches=0 forces the rated branch, which is the
        # only one that draws. The pre-fix tracker raised AttributeError here.
        assert self._trained().select_op(["a", "b"]) in {"a", "b"}

    def test_strategy_selection_reaches_a_pool(self):
        t = self._trained()
        for s in ("s1", "s2"):
            for _ in range(20):
                t.record_strategy_match(s, "s2", 1.0)
        assert t.select_strategy(["s1", "s2"]) in {"s1", "s2"}

    def test_injected_pool_is_the_one_used(self):
        pool = RandPool(seed=7)
        assert self._trained(rng=pool)._rng is pool

    def test_default_pool_is_a_randpool(self):
        # Falsification: a tracker that quietly kept the stdlib module would
        # still "work" above, so pin the type.
        assert isinstance(BayesianEloTracker()._rng, RandPool)


class TestSeedPickerHasAPool:
    def test_picker_binds_a_pool_without_a_fuzzer_pool(self):
        class Bare:
            pass

        assert isinstance(SeedPicker(Bare())._rng, RandPool)

    def test_picker_prefers_the_fuzzers_pool(self):
        pool = RandPool(seed=3)
        assert SeedPicker(_StubFuzzer(pool))._rng is pool


class TestCdfPickTakesItsPool:
    def test_signature_has_no_self(self):
        # The bug was a module-level function referring to ``self``.
        params = list(inspect.signature(_cdf_pick).parameters)
        assert "self" not in params
        assert params[-1] == "rng"

    def test_fast_path_matches_an_independent_cdf_walk(self):
        # Equivalence derived arithmetically, not echoed from the code:
        # one random() per pick, scaled by the total, bisected on the
        # prefix sums.
        population = ["a", "b", "c", "d"]
        weights = [1.0, 3.0, 2.0, 4.0]
        total = sum(weights)
        cum, acc = [], 0.0
        for w in weights:
            acc += w
            cum.append(acc)

        for draw in (0.0, 0.05, 0.3, 0.55, 0.9, 0.999):
            expected = next(p for p, c in zip(population, cum, strict=True) if draw * total < c)

            class OneDraw:
                def __init__(self, value):
                    self._value = value

                def random(self):
                    return self._value

            got = _cdf_pick(population, weights, {}, "slot", OneDraw(draw))
            assert got == expected, f"draw={draw}"

    def test_degenerate_inputs_raise_instead_of_picking_a_wrong_seed(self):
        # RandPool.weighted_choice does NOT reproduce random.choices' errors:
        # a length mismatch returns a value, and a zero/non-finite total
        # walks bisect off the prefix sum (IndexError). Deferring would turn
        # a loud caller-visible error into a silently wrong pick, so the
        # checks live in _cdf_pick with CPython's messages.
        class Exploding:
            def weighted_choice(self, seq, weights):  # pragma: no cover
                raise AssertionError("degenerate input was handed to the pool")

            def random(self):  # pragma: no cover
                raise AssertionError("fast path taken on a degenerate input")

        with pytest.raises(ValueError, match="number of weights"):
            _cdf_pick(["x", "y"], [1.0], {}, "slot", Exploding())

        with pytest.raises(ValueError, match="greater than zero"):
            _cdf_pick(["x", "y"], [0.0, 0.0], {}, "slot", Exploding())

        with pytest.raises(ValueError, match="must be finite"):
            _cdf_pick(["x", "y"], [math.inf, 1.0], {}, "slot", Exploding())

    def test_error_shape_matches_stdlib_choices(self):
        # Equivalence derived from CPython, not echoed from _cdf_pick: the
        # old code deferred to random.choices, so the messages must match.
        for weights in ([1.0], [0.0, 0.0], [math.inf, 1.0]):
            with pytest.raises(ValueError) as stdlib:
                random.choices(["x", "y"], weights=weights, k=1)
            with pytest.raises(ValueError) as ours:
                _cdf_pick(["x", "y"], list(weights), {}, "slot", RandPool(seed=1))
            assert str(ours.value) == str(stdlib.value)

    def test_negative_infinity_reports_as_non_positive(self):
        # Order matters: CPython checks non-positive before non-finite.
        with pytest.raises(ValueError, match="greater than zero"):
            _cdf_pick(["x", "y"], [-math.inf, 1.0], {}, "slot", RandPool(seed=1))

    def test_empty_population_still_defers_to_the_pool(self):
        with pytest.raises(IndexError):
            _cdf_pick([], [], {}, "slot", RandPool(seed=1))

    def test_cdf_cache_is_keyed_by_weight_identity(self):
        population = ["a", "b"]
        weights = [1.0, 1.0]
        store = {}
        _cdf_pick(population, weights, store, "slot", RandPool(seed=1))
        assert store["slot"][0] is weights

        fresh = [1.0, 1.0]
        _cdf_pick(population, fresh, store, "slot", RandPool(seed=1))
        assert store["slot"][0] is fresh
