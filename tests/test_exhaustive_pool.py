"""Exhaustive enumeration through the PRNG interface (port item P1-5).

See ``docs/tigerbeetle_four_fuzzers_port.md``. Two halves:

1. ``ExhaustivePool`` itself, checked against spaces whose cardinality is
   known in closed form -- products, ``n!``, ``n!/(n-k)!`` -- because a
   generator that claims to be exhaustive and is not would otherwise be
   invisible: every test using it would still pass, having quietly
   explored a subset.

2. The real operator table driven through it. ``OperatorEngine`` reads its
   randomness from ``self.f._rand_pool``, so substituting the pool turns
   "run this operator once" into "run it once per reachable combination of
   draws" for every operator that draws only bounded values.

The second half is what found the two ``max_len`` escapes fixed alongside
this file. Both were reachable on a small fraction of paths and neither
had ever failed a random test.
"""

from __future__ import annotations

import itertools
import math

import pytest

from fuzzer_tool.core.exhaustive_pool import (
    BulkDrawError,
    ContinuousDrawError,
    DepthExceededError,
    ExhaustivePool,
    ExhaustivePoolError,
    NondeterministicDrawError,
)
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.services.operators import OperatorEngine

from .support.operator_env import make_minimal_fuzzer

# ── The pool's own contract ──────────────────────────────────────────


class TestCardinality:
    """Spaces with a known closed-form size, walked and counted.

    Counting distinct *outcomes* rather than runs on purpose: a pool that
    visited every combination but returned a constant would pass a
    run-count check.
    """

    def test_product_of_two_independent_draws(self):
        pool = ExhaustivePool()
        seen = [(pool.randint(0, 2), pool.randint(0, 1)) for _ in pool.runs()]
        assert pool.exhausted
        assert sorted(seen) == sorted(itertools.product(range(3), range(2)))

    def test_randrange_covers_its_bound(self):
        pool = ExhaustivePool()
        seen = {pool.randrange(5) for _ in pool.runs()}
        assert pool.exhausted
        assert seen == set(range(5))

    def test_choice_covers_every_element(self):
        pool = ExhaustivePool()
        seen = {pool.choice("abcd") for _ in pool.runs()}
        assert pool.exhausted
        assert seen == set("abcd")

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_shuffle_enumerates_n_factorial_permutations(self, n):
        pool = ExhaustivePool()
        perms = set()
        for _ in pool.runs():
            seq = list(range(n))
            pool.shuffle(seq)
            perms.add(tuple(seq))
        assert pool.exhausted
        assert len(perms) == math.factorial(n)

    def test_sample_enumerates_ordered_selections_without_replacement(self):
        pool = ExhaustivePool()
        seen = {tuple(pool.sample([10, 20, 30, 40], 2)) for _ in pool.runs()}
        assert pool.exhausted
        # n!/(n-k)! = 4*3, and no element may repeat within a selection.
        assert len(seen) == 12
        assert all(len(set(t)) == 2 for t in seen)

    def test_branching_depth_varies_with_earlier_draws(self):
        """The odometer must handle a tree, not just a rectangle.

        A prefix is replayed value-for-value, so an operator whose control
        flow depends on an earlier draw takes the identical path and the
        positions stay aligned. This is the property that lets real
        operators -- which branch on buffer length, on which byte was
        picked, on whether a candidate list came back empty -- be walked
        at all.
        """
        pool = ExhaustivePool()
        seen = []
        for _ in pool.runs():
            first = pool.randint(0, 1)
            seen.append((first, pool.randint(0, 3)) if first == 0 else (first,))
        assert pool.exhausted
        assert sorted(seen) == [(0, 0), (0, 1), (0, 2), (0, 3), (1,)]

    def test_no_draws_at_all_is_one_run(self):
        pool = ExhaustivePool()
        assert sum(1 for _ in pool.runs()) == 1
        assert pool.exhausted


class TestDegenerateBounds:
    def test_single_choice_draws_are_not_recorded(self):
        """A bound of 1 is not a choice, and must not double the tree.

        Operators call ``randint(0, 0)`` constantly -- any time a
        computed span leaves exactly one legal position. Recording those
        would multiply the run count by 1 while doubling the depth, and
        depth is the budget that runs out.
        """
        pool = ExhaustivePool()
        runs = 0
        for _ in pool.runs():
            runs += 1
            assert pool.randint(4, 4) == 4
            assert pool.randrange(1) == 0
            assert pool.choice([99]) == 99
        assert runs == 1
        assert pool.max_depth_seen == 0

    def test_mirrors_randpool_on_degenerate_inputs(self):
        """Same answers as RandPool where RandPool declines to raise.

        RandPool returns 0 for a non-positive randrange and ``a`` for an
        inverted randint rather than raising. An enumeration that raised
        instead would report operators as broken that are not.
        """
        pool = ExhaustivePool()
        real = RandPool(1)
        for _ in pool.runs():
            assert pool.randrange(0) == real.randrange(0) == 0
            assert pool.randrange(-3) == real.randrange(-3) == 0
            assert pool.randint(5, 2) == real.randint(5, 2) == 5

    def test_empty_sequence_raises_like_randpool(self):
        pool = ExhaustivePool()
        for _ in pool.runs():
            with pytest.raises(IndexError):
                pool.choice([])
            with pytest.raises(IndexError):
                pool.weighted_choice([], [])
            break

    def test_weighted_choice_ignores_magnitudes_but_not_zeros(self):
        """Enumeration asks what is reachable, not what is likely.

        A 1-in-1000 branch is visited exactly as often as the other 999,
        which is the whole reason to enumerate. A weight of zero is
        different in kind -- it is unreachable -- so it is excluded.
        """
        pool = ExhaustivePool()
        seen = {pool.weighted_choice("abcd", [1, 1000, 0, 1]) for _ in pool.runs()}
        assert pool.exhausted
        assert seen == {"a", "b", "d"}

    def test_weighted_choice_with_all_zero_weights_is_unreachable(self):
        pool = ExhaustivePool()
        for _ in pool.runs():
            with pytest.raises(IndexError, match="every weight is zero"):
                pool.weighted_choice("ab", [0, 0])
            break


class TestRefusals:
    """The pool refuses rather than guessing, because a skipped draw still
    reports ``exhausted``."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda p: p.random(),
            lambda p: p.gauss(),
            lambda p: p.expovariate(1.0),
            lambda p: p.betavariate(2.0, 2.0),
            lambda p: p.gammavariate(2.0),
            lambda p: p.lognormvariate(),
            lambda p: p.random_list(3),
            lambda p: p.gauss_list(0.0, 1.0, 3),
        ],
    )
    def test_continuous_draws_refuse(self, call):
        pool = ExhaustivePool()
        with pytest.raises(ContinuousDrawError):
            call(pool)

    def test_continuous_error_names_the_cheap_fix(self):
        """Most of these are coin flips, not real continuous draws.

        21 operators are unenumerable purely because a branch is written
        ``rng.random() < 0.5`` instead of ``rng.randint(0, 1)``. The
        message says so, since the person who hits it is the person who
        can change it.
        """
        pool = ExhaustivePool()
        with pytest.raises(ContinuousDrawError, match="coin flip"):
            pool.random()

    @pytest.mark.parametrize(
        "call",
        [
            lambda p: p.randbytes(4),
            lambda p: p.randint_list(0, 255, 4),
            lambda p: p.randrange_list(10, 4),
            lambda p: p.choice_list("abc", 4),
            lambda p: p.weighted_choice_list("abc", [1, 1, 1], 4),
        ],
    )
    def test_bulk_draws_refuse_by_default(self, call):
        pool = ExhaustivePool()
        with pytest.raises(BulkDrawError):
            call(pool)

    def test_bulk_draws_enumerate_when_opted_into(self):
        pool = ExhaustivePool(allow_bulk=True)
        seen = {pool.randbytes(2) for _ in pool.runs()}
        assert pool.exhausted
        assert len(seen) == 256 * 256

    def test_empty_bulk_draws_need_no_opt_in(self):
        """Zero-width bulk draws branch one way and are not a budget risk."""
        pool = ExhaustivePool()
        assert pool.randbytes(0) == b""
        assert pool.randint_list(0, 9, 0) == []
        assert pool.choice_list("abc", 0) == []

    def test_depth_cap_raises_rather_than_truncating(self):
        """A truncated path is an unexplored path, and must not read as covered."""
        pool = ExhaustivePool(max_depth=3)
        with pytest.raises(DepthExceededError, match="max_depth=3"):
            for _ in pool.runs():
                for _ in range(4):
                    pool.randint(0, 1)

    def test_nondeterminism_is_reported_not_absorbed(self):
        """A bound that changes on a replayed prefix cannot happen by chance.

        The prefix is replayed value-for-value, so a function of the draws
        alone must request identical bounds. A mismatch means entropy is
        entering from somewhere this pool does not intermediate -- a
        module-level ``random``, a clock, a set iteration order.
        """
        pool = ExhaustivePool()
        bounds = itertools.cycle([4, 7])
        with pytest.raises(NondeterministicDrawError):
            for _ in pool.runs():
                pool.randint(0, 1)
                pool.randrange(next(bounds))


class TestBudget:
    def test_budget_exhaustion_is_distinguishable_from_completion(self):
        """``exhausted`` must be false on a partial walk, or it means nothing."""
        pool = ExhaustivePool(max_runs=10)
        count = sum(1 for _ in pool.runs() for _ in [pool.randint(0, 99)])
        assert count == 10
        assert pool.budget_exhausted
        assert not pool.exhausted

    def test_completion_clears_the_budget_flag(self):
        pool = ExhaustivePool(max_runs=1000)
        for _ in pool.runs():
            pool.randint(0, 4)
        assert pool.exhausted
        assert not pool.budget_exhausted
        assert pool.runs_completed == 5


class TestRandPoolParity:
    def test_implements_every_public_randpool_method(self):
        """A method RandPool gains and this pool does not is a silent hole.

        The operator would raise AttributeError under enumeration and be
        filed as "not enumerable" rather than as "the harness is stale".
        """
        missing = [
            name
            for name in dir(RandPool)
            if not name.startswith("_") and callable(getattr(RandPool, name))
            if not hasattr(ExhaustivePool, name)
        ]
        assert not missing, f"ExhaustivePool does not intercept: {missing}"

    def test_reseed_is_a_harmless_no_op(self):
        """An operator that reseeds mid-run must not crash the walk.

        Safe to ignore only because this pool has no entropy to reset;
        the enumeration order is state, not a seed.
        """
        pool = ExhaustivePool()
        seen = set()
        for _ in pool.runs():
            pool.reseed(12345)
            seen.add(pool.randint(0, 2))
        assert pool.exhausted
        assert seen == {0, 1, 2}


# ── Applied to the real operator table ───────────────────────────────


def _operator_names() -> list[str]:
    engine = OperatorEngine(make_minimal_fuzzer(pool=ExhaustivePool()))
    return sorted(REGISTRY.dispatch(engine))


def _enumerate_operator(name: str, seed: bytes, max_len: int, max_runs: int = 4000):
    """Walk every reachable output of one operator, or report why not.

    Returns ``(status, outputs)``. ``status`` is ``"enumerated"`` only when
    the space was covered; every other value means the caller must not
    treat the outputs as complete.
    """
    pool = ExhaustivePool(max_depth=16, max_runs=max_runs)
    fuzzer = make_minimal_fuzzer(pool=pool)
    fuzzer.max_len = max_len
    handler = REGISTRY.dispatch(OperatorEngine(fuzzer))[name]
    outputs = []
    try:
        for _ in pool.runs():
            buf = bytearray(seed)
            result = handler(buf, 0, bytes(seed))
            outputs.append((result, bytes(buf)))
    except ContinuousDrawError:
        return "continuous", outputs
    except BulkDrawError:
        return "bulk", outputs
    except DepthExceededError:
        return "too_deep", outputs
    return ("over_budget" if pool.budget_exhausted else "enumerated"), outputs


class TestOperatorEnumeration:
    SEED = bytes(range(8))
    MAX_LEN = 8

    def test_a_substantial_share_of_operators_is_enumerable(self):
        """A floor, not an exact count, so adding an operator is not a failure.

        Measured at the time of writing: 70 of 134 fully enumerable on an
        8-byte buffer, 21 blocked by continuous draws, 14 by bulk draws,
        4 too deep, 25 over budget. The floor exists to catch the
        regression where the pool stops intercepting something and
        everything silently reclassifies.
        """
        names = _operator_names()
        assert len(names) > 100, "operator table unexpectedly small"
        enumerated = [
            n for n in names if _enumerate_operator(n, self.SEED, self.MAX_LEN)[0] == "enumerated"
        ]
        assert len(enumerated) >= 55, (
            f"only {len(enumerated)} of {len(names)} operators enumerable; "
            f"the pool may have stopped intercepting a draw method"
        )

    def test_no_operator_exceeds_max_len_on_any_reachable_path(self):
        """The invariant this whole file exists to state.

        Every other operator respects max_len, so the two that did not
        were not following a different convention -- they had simply not
        been run down the right path. Random testing had not reached
        either in the life of the project.

        Checked at several caps because the two escapes had different
        shapes: one grew by a fixed +1 regardless of the cap, the other
        jumped to a fixed 13-byte pattern whenever the cap was smaller
        than that.
        """
        for max_len, seed in ((8, bytes(range(8))), (4, b"abcd"), (2, b"ab"), (1, b"a")):
            offenders = []
            for name in _operator_names():
                status, outputs = _enumerate_operator(name, seed, max_len)
                if status != "enumerated":
                    continue
                worst = max((len(r if r is not None else b) for r, b in outputs), default=0)
                if worst > max_len:
                    offenders.append((name, worst))
            assert not offenders, f"max_len={max_len}: {offenders}"

    def test_every_reachable_output_is_bytes_or_none(self):
        for name in _operator_names():
            status, outputs = _enumerate_operator(name, self.SEED, self.MAX_LEN)
            if status != "enumerated":
                continue
            bad = [
                type(r).__name__
                for r, _ in outputs
                if r is not None and not isinstance(r, bytes | bytearray)
            ]
            assert not bad, f"{name} returned {set(bad)}"

    def test_no_operator_draws_entropy_the_pool_does_not_own(self):
        """A clean sweep, recorded because the negative result is the finding.

        ``NondeterministicDrawError`` fires when a replayed prefix asks for
        a different bound, which is how a module-level ``random`` or a set
        iteration order would show up. Across the whole table, none does --
        so ``--fuzz-seed`` genuinely controls operator behaviour, which had
        been assumed rather than checked.
        """
        for name in _operator_names():
            try:
                _enumerate_operator(name, self.SEED, self.MAX_LEN)
            except NondeterministicDrawError as exc:  # pragma: no cover
                pytest.fail(f"{name}: {exc}")
            except ExhaustivePoolError:
                pass


class TestMaxLenEscapes:
    """Regressions for the two operators the enumeration caught.

    Both mutate in place and return None, so neither reaches mutate()'s
    post-operator ``f.max_len`` clamp -- that clamp only runs in the
    ``result is not None`` branch, as ``_op_fuse_this``'s docstring already
    warned after the same class of bug cost an unbounded-growth incident.
    """

    def test_utf8_widen_declines_when_the_extra_byte_would_not_fit(self):
        """Grows by exactly +1, which is why it went unnoticed.

        A one-byte overrun per application looks like rounding. It is not:
        the operator can be selected again on its own output, and nothing
        downstream caps this path.
        """
        status, outputs = _enumerate_operator("utf8_widen", b"abcd", max_len=4)
        assert status == "enumerated"
        assert all(len(b) <= 4 for _, b in outputs)
        assert any(b == b"abcd" for _, b in outputs), "should decline, not truncate"

    def test_utf8_widen_still_widens_when_there_is_room(self):
        """The guard must not turn the operator off entirely."""
        status, outputs = _enumerate_operator("utf8_widen", b"abcd", max_len=16)
        assert status == "enumerated"
        widened = [bytes(b) for _, b in outputs if len(b) == 5]
        assert widened, "operator no longer produces the overlong encoding"
        # An overlong 2-byte encoding of a 7-bit value has a 0xC0/0xC1 lead
        # byte -- the shortest form would have used one byte. That is the
        # property the operator exists to produce.
        assert all(any(x in (0xC0, 0xC1) for x in b) for b in widened)

    def test_regex_bomb_only_uses_patterns_that_fit(self):
        """Patterns are filtered, not truncated: a truncated bomb is not a bomb."""
        from fuzzer_tool.core.mutations import REGEX_BOMBS

        shortest = min(len(p.encode()) for p in REGEX_BOMBS)
        for max_len in (shortest, shortest + 2, 13):
            status, outputs = _enumerate_operator("regex_bomb", b"x" * max_len, max_len)
            assert status == "enumerated"
            assert all(len(b) <= max_len for _, b in outputs), max_len
            assert any(any(p.encode() in bytes(b) for p in REGEX_BOMBS) for _, b in outputs), (
                f"no bomb placed at max_len={max_len}"
            )

    def test_regex_bomb_declines_when_no_pattern_fits(self):
        from fuzzer_tool.core.mutations import REGEX_BOMBS

        too_small = min(len(p.encode()) for p in REGEX_BOMBS) - 1
        status, outputs = _enumerate_operator("regex_bomb", b"x" * too_small, too_small)
        assert status == "enumerated"
        assert all(bytes(b) == b"x" * too_small for _, b in outputs)
