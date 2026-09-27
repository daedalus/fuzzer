"""Exhaustive enumeration through the PRNG interface.

See ``docs/learnings/2026-08-22-exhaustive-pool-p1-5.md``. Two halves:

1. ``ExhaustivePool`` itself, checked against spaces whose cardinality is
   known in closed form -- products, ``n!``, ``n!/(n-k)!`` -- because a
   generator that claims to be exhaustive and is not would otherwise be
   invisible: every test using it would still pass, having quietly
   explored a subset.

2. The real operator table driven through it. ``OperatorEngine`` reads its
   randomness from ``self.f._rng``, so substituting the pool turns
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
    ORDER_LEXICOGRAPHIC,
    ORDER_SPREAD,
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
            lambda p: p.dirichlet([1.0, 2.0]),
        ],
    )
    def test_continuous_draws_refuse(self, call):
        pool = ExhaustivePool()
        with pytest.raises(ContinuousDrawError):
            call(pool)

    def test_continuous_error_names_the_cheap_fix(self):
        """Most of these were coin flips, not real continuous draws.

        As of 2026-09-02, 21 operators were unenumerable purely because a
        branch was written ``rng.random() < 0.5`` instead of
        ``rng.randint(0, 1)``. Fixed 2026-09-26 for every fixed-probability
        site (``TestCoinFlipRewrite``); the message stays as a guide for
        the ~13 remaining ``rng.random() < <runtime value>`` sites and any
        new ones, since the person who hits it is the person who can
        change it.
        """
        pool = ExhaustivePool()
        with pytest.raises(ContinuousDrawError, match="coin flip"):
            pool.random()

    @pytest.mark.parametrize(
        "call",
        [
            lambda p: p.randbytes(4),
            lambda p: p.randint_list(0, 255, 4),
            lambda p: p.randrange_list(10, 5),
            lambda p: p.choice_list("abc", 11),
            lambda p: p.weighted_choice_list("abc", [1, 1, 1], 11),
            lambda p: p.categorical([1.0, 1.0, 1.0], 11),
        ],
    )
    def test_bulk_draws_refuse_when_over_cap(self, call):
        """Each call here exceeds the 65,536-path auto-enumerate cap alone."""
        pool = ExhaustivePool()
        with pytest.raises(BulkDrawError, match="auto-enumerate cap"):
            call(pool)

    @pytest.mark.parametrize(
        "call,expected_paths",
        [
            (lambda p: p.randbytes(2), 256**2),
            (lambda p: p.randint_list(0, 255, 2), 256**2),
            (lambda p: p.randrange_list(10, 4), 10**4),
            (lambda p: p.choice_list("abc", 4), 3**4),
            (lambda p: p.weighted_choice_list("abc", [1, 1, 1], 4), 3**4),
            (lambda p: p.categorical([1.0, 1.0, 1.0], 4), 3**4),
        ],
    )
    def test_bulk_draws_auto_enumerate_under_cap(self, call, expected_paths):
        """A call within the per-call cap needs no ``allow_bulk`` opt-in."""
        pool = ExhaustivePool()
        seen = set()
        for _ in pool.runs():
            result = call(pool)
            seen.add(tuple(result) if isinstance(result, list) else result)
        assert pool.exhausted
        assert len(seen) == expected_paths

    def test_bulk_draws_enumerate_when_opted_into(self):
        pool = ExhaustivePool(allow_bulk=True)
        seen = {pool.randbytes(2) for _ in pool.runs()}
        assert pool.exhausted
        assert len(seen) == 256 * 256

    def test_bulk_calls_combine_past_per_run_cap(self):
        """Two calls that each fit alone can still exceed the cap combined.

        ``randbytes(2)`` alone is exactly the 65,536-path cap; a second one
        in the same run multiplies that to 65,536**2, which must still
        require ``allow_bulk=True`` even though neither call is refused in
        isolation -- the risk the handover flagged for a per-call-only cap.
        """
        pool = ExhaustivePool()
        with pytest.raises(BulkDrawError, match="combined bulk-draw space"):
            for _ in pool.runs():
                pool.randbytes(2)
                pool.randbytes(2)

    def test_bulk_calls_combine_within_cap(self):
        """Two small calls whose product still fits the cap both succeed."""
        pool = ExhaustivePool()
        seen = set()
        for _ in pool.runs():
            seen.add((pool.randbytes(1), pool.randbytes(1)))
        assert pool.exhausted
        assert len(seen) == 256 * 256

    def test_bulk_path_budget_resets_each_run(self):
        """The per-run product must not accumulate across separate runs."""
        pool = ExhaustivePool()
        seen = set()
        for _ in pool.runs():
            seen.add(pool.randbytes(2))
        assert pool.exhausted
        assert len(seen) == 256**2

    def test_max_bulk_paths_per_call_is_configurable(self):
        pool = ExhaustivePool(max_bulk_paths_per_call=100)
        with pytest.raises(BulkDrawError, match="auto-enumerate cap"):
            pool.randbytes(1)  # 256 paths, over the lowered 100-path cap

    def test_bulk_draws_still_enumerate_when_opted_into_over_the_cap(self):
        """``allow_bulk=True`` bypasses both the per-call and per-run cap."""
        pool = ExhaustivePool(allow_bulk=True, max_bulk_paths_per_call=1)
        seen = set()
        for _ in pool.runs():
            seen.add((pool.randbytes(1), pool.randbytes(1)))
        assert pool.exhausted
        assert len(seen) == 256 * 256

    def test_categorical_skips_zero_weight(self):
        pool = ExhaustivePool(allow_bulk=True)
        seen = {tuple(pool.categorical([1.0, 0.0, 2.0], 2)) for _ in pool.runs()}
        assert seen == set(itertools.product((0, 2), repeat=2))

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

    def test_a_truncated_lexicographic_walk_pins_the_leading_draws(self):
        """The defect spread order exists for, stated as a test.

        ``exhausted`` already reports that the space was not covered. What
        it does not report is that the omission is entirely at one end: the
        odometer carries left, so the first draw never leaves zero and the
        walk is a prefix of one position rather than a sample of four.
        """
        pool = ExhaustivePool(max_runs=20_000)
        reached = [set() for _ in range(4)]
        for _ in pool.runs():
            for i in range(4):
                reached[i].add(pool.randrange(256))
        assert pool.budget_exhausted
        assert [len(s) for s in reached] == [1, 1, 79, 256]


class TestSpreadOrder:
    def test_spread_reaches_every_value_at_every_position(self):
        """20,000 runs of a 4.29-billion-wide space, all 256 values, all four draws.

        The lexicographic control for the identical space and budget is
        ``test_a_truncated_lexicographic_walk_pins_the_leading_draws``
        above; the contrast between the two is the whole claim.
        """
        pool = ExhaustivePool(max_runs=20_000, order=ORDER_SPREAD)
        reached = [set() for _ in range(4)]
        for _ in pool.runs():
            for i in range(4):
                reached[i].add(pool.randrange(256))
        assert pool.budget_exhausted
        assert not pool.exhausted, "a sample must never claim coverage"
        assert pool.strided_runs > 0
        assert pool.space_size == 256**4
        assert [len(s) for s in reached] == [256, 256, 256, 256]

    def test_a_space_that_fits_is_still_walked_exhaustively(self):
        """Spread order must not cost exhaustion where exhaustion was available.

        Reordering only helps a walk that cannot finish. Applying it to one
        that can would replace a proof with a sample, so the constructor
        argument is a ceiling on the reordering, not a request for it.
        """
        pool = ExhaustivePool(max_runs=10_000, order=ORDER_SPREAD)
        seen = set()
        for _ in pool.runs():
            seen.add((pool.randrange(5), pool.randrange(7)))
        assert pool.exhausted
        assert not pool.budget_exhausted
        assert pool.order == ORDER_SPREAD, "the request is not rewritten"
        assert pool.strided_runs == 0, "a space this size must not be strided"
        assert len(seen) == 35
        assert pool.runs_completed == 35

    def test_the_stride_is_a_full_period_permutation(self):
        """Coprimality is load-bearing, so it is asserted and not assumed.

        A stride sharing a factor ``d`` with the space cycles through
        ``space/d`` indices and never visits the rest -- a walk that looks
        spread and is blind to a fixed fraction of the space. Checked as a
        permutation on sizes small enough to enumerate, which is the only
        way to check it at all: inside the pool the space is by definition
        larger than the budget.
        """
        from math import gcd

        from fuzzer_tool.core.exhaustive_pool import _coprime_stride

        for space in (3, 17, 64, 100, 256, 1001, 4096, 426_888):
            stride = _coprime_stride(space)
            assert gcd(stride, space) == 1, f"stride {stride} not coprime to {space}"
            visited = {(k * stride) % space for k in range(space)}
            assert len(visited) == space

    def test_the_stride_sits_near_the_golden_ratio_point(self):
        """Full period gives coverage; the phi placement gives low discrepancy.

        Both are required and they are separate properties -- a stride of 1
        is coprime to everything and reproduces the odometer exactly.
        """
        from fuzzer_tool.core.exhaustive_pool import _coprime_stride

        for space in (1000, 65_536, 426_888, 2**32):
            stride = _coprime_stride(space)
            # Within a handful of steps of space/phi: the search walks
            # outward from the target and consecutive integers are coprime,
            # so it can never need to go far.
            assert abs(stride - space * 0.6180339887498949) < 16
        assert _coprime_stride(2**32) == 2654435769, "Knuth's constant falls out of this"

    def test_a_rectangular_tree_diverges_nowhere(self):
        pool = ExhaustivePool(max_runs=500, order=ORDER_SPREAD)
        for _ in pool.runs():
            pool.randrange(40)
            pool.randrange(40)
            pool.randrange(40)
        assert pool.space_size == 64_000
        assert pool.shape_divergences == 0

    def test_a_path_dependent_tree_is_clamped_and_counted(self):
        """The cost of jumping instead of replaying, made observable.

        Spread order cannot replay a prefix value-for-value, so a bound
        that depends on an earlier draw will not match the one the stride
        assumed. Clamping keeps every draw legal; the counter is what stops
        that from being silent, since it also means the run is no longer
        the index the stride picked.
        """
        pool = ExhaustivePool(max_runs=300, order=ORDER_SPREAD)
        for _ in pool.runs():
            first = pool.randrange(50)
            # Width depends on the first draw, so the shape is not rectangular.
            pool.randrange(first + 1)
        assert pool.shape_divergences > 0

    def test_nondeterminism_still_raises_in_lexicographic_order(self):
        """The clamp is scoped to spread order and must not weaken the default.

        ``NondeterministicDrawError`` is how outside entropy shows up, and
        the sweep asserting no operator has any is the reason it matters.
        """
        pool = ExhaustivePool(max_runs=50)
        bounds = itertools.cycle([4, 7])
        with pytest.raises(NondeterministicDrawError):
            for _ in pool.runs():
                pool.randint(0, 1)
                pool.randrange(next(bounds))

    def test_an_unknown_order_is_refused(self):
        with pytest.raises(ValueError, match="lexicographic/spread"):
            ExhaustivePool(order="halton")


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


def _enumerate_operator(
    name: str,
    seed: bytes,
    max_len: int,
    max_runs: int = 4000,
    order: str = ORDER_LEXICOGRAPHIC,
):
    """Walk every reachable output of one operator, or report why not.

    Returns ``(status, outputs)``. ``status`` is ``"enumerated"`` only when
    the space was covered; every other value means the caller must not
    treat the outputs as complete.
    """
    pool = ExhaustivePool(max_depth=16, max_runs=max_runs, order=order)
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


def _walk_operator(name: str, seed: bytes, max_len: int, max_runs: int = 4000):
    """``_enumerate_operator``, falling back to a spread sample over budget.

    Returns ``(status, outputs)`` where ``"sampled"`` means the outputs are
    a low-discrepancy sample of the operator's space rather than all of it.
    A sample is enough for any *universally quantified* claim -- max_len is
    respected, the return type is bytes-or-None -- because one
    counterexample falsifies those, and it is not enough for an
    existentially quantified one ("the operator can produce X"), which is
    why this is a separate function rather than a change to the one above.

    The fallback matters because the lexicographic walk pins the leading
    draws to zero once the budget bites: the 27 over-budget operators were
    not being checked against the max_len invariant at all, and that
    invariant is the one that caught ``utf8_widen`` and ``regex_bomb``.
    """
    status, outputs = _enumerate_operator(name, seed, max_len, max_runs)
    if status != "over_budget":
        return status, outputs
    status, outputs = _enumerate_operator(name, seed, max_len, max_runs, order=ORDER_SPREAD)
    return ("sampled" if status == "over_budget" else status), outputs


class TestOperatorEnumeration:
    SEED = bytes(range(8))
    MAX_LEN = 8

    def test_a_substantial_share_of_operators_is_enumerable(self):
        """A floor, not an exact count, so adding an operator is not a failure.

        Measured at the time of writing: 70 of 134 fully enumerable on an
        8-byte buffer, 21 blocked by continuous draws, 14 by bulk draws,
        4 too deep, 25 over budget. Re-measured 2026-09-12 as the table
        grew to 208: 129 enumerable, 34 continuous, 8 bulk, 10 too deep,
        27 over budget -- the last group is the one ``_walk_operator``
        now samples rather than skips. Re-measured 2026-09-26 after the
        coin-flip rewrite (``TestCoinFlipRewrite``): 157 enumerable, only
        2 continuous, 7 bulk, 15 too deep, 46 over budget -- the floor
        below is raised to match, and still exists to catch the
        regression where the pool stops intercepting something and
        everything silently reclassifies.
        """
        names = _operator_names()
        assert len(names) > 100, "operator table unexpectedly small"
        enumerated = [
            n for n in names if _enumerate_operator(n, self.SEED, self.MAX_LEN)[0] == "enumerated"
        ]
        assert len(enumerated) >= 150, (
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
                status, outputs = _walk_operator(name, seed, max_len)
                if status not in ("enumerated", "sampled"):
                    continue
                worst = max((len(r if r is not None else b) for r, b in outputs), default=0)
                if worst > max_len:
                    offenders.append((name, worst, status))
            assert not offenders, f"max_len={max_len}: {offenders}"

    def test_the_over_budget_operators_are_checked_and_not_skipped(self):
        """A floor on how many operators the spread fallback brings in.

        Without it these were not merely unproven, they were unexamined:
        the lexicographic walk pins the leading draws to zero, so every
        claim above quietly excluded them. This asserts the fallback keeps
        working, because the failure mode is silent -- the invariant tests
        pass either way, just over fewer operators.
        """
        sampled = [
            n
            for n in _operator_names()
            if _walk_operator(n, self.SEED, self.MAX_LEN)[0] == "sampled"
        ]
        assert len(sampled) >= 15, (
            f"only {len(sampled)} operators reached by the spread fallback; "
            f"the over-budget set is no longer being checked"
        )

    def test_every_reachable_output_is_bytes_or_none(self):
        for name in _operator_names():
            status, outputs = _walk_operator(name, self.SEED, self.MAX_LEN)
            if status not in ("enumerated", "sampled"):
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


class TestCoinFlipRewrite:
    """Operators freed from ``ContinuousDrawError`` by the coin-flip rewrite.

    Each of these called ``rng.random() < <literal>`` somewhere on its path
    to a fixed probability (0.5, 0.3, 0.75, ...); rewritten as
    ``rng.randint(0, N-1) < K`` for the minimal ``N`` exact for that
    literal, the branch is a bounded draw and the operator is fully
    enumerable rather than refused. See
    ``docs/handover/handover_coin_flip_bulk_budget_2026-09-26.md`` for the
    full site census, including the ~13 sites left as ``rng.random() < p``
    because ``p`` is a runtime value (a scheduler's ``epsilon``, a solver's
    acceptance ratio) rather than a fixed literal, and the two genuine
    continuous draws (``expovariate``, a raw float packed into WEBM bytes)
    that are not coin flips and were correctly left alone.
    """

    SEED = TestOperatorEnumeration.SEED
    MAX_LEN = TestOperatorEnumeration.MAX_LEN

    NEWLY_ENUMERABLE = [
        "arithmetic",
        "bitcast_float",
        "bitcast_int32",
        "corpus_literal_insert",
        "count_overflow",
        "cycle_lock",
        "float_squeeze",
        "interesting_16",
        "interesting_32",
        "interesting_8",
        "perm_lock",
        "radamsa_num",
        "size_field_overflow",
        "spectral_peak",
        "type_promote",
        "varsize",
        "webp_chunk_mutate",
    ]

    @pytest.mark.parametrize("name", NEWLY_ENUMERABLE)
    def test_operator_is_now_fully_enumerable(self, name):
        status, _ = _enumerate_operator(name, self.SEED, self.MAX_LEN)
        assert status == "enumerated", (
            f"{name} was expected to be enumerable after the coin-flip "
            f"rewrite but reported {status!r}"
        )

    NEWLY_REACHABLE = [
        # Moved from "continuous" (silently unreachable) to a status the
        # existing invariant tests (TestOperatorEnumeration) do check --
        # over_budget is sampled via the spread fallback, too_deep and bulk
        # are honest classifications rather than a masked refusal.
        "birthday_collide",
        "degenerate_geometry",
        "elias_delta",
        "elias_gamma",
        "gcd_worst_case",
        "length_miscalculate",
        "monotone_fill",
        "protobuf_chunk_mutate",
        "swap_bytes",
        "der_len_mutate",
        "der_tag_mutate",
        "der_tlv_insert",
        "der_tlv_reorder",
        "rle",
        "kmer_starve",
    ]

    @pytest.mark.parametrize("name", NEWLY_REACHABLE)
    def test_operator_is_no_longer_masked_as_continuous(self, name):
        status, _ = _enumerate_operator(name, self.SEED, self.MAX_LEN)
        assert status != "continuous", (
            f"{name} still reports 'continuous' after the coin-flip "
            f"rewrite -- a bounded draw further down its path may have "
            f"been missed"
        )

    def test_only_two_operators_remain_genuinely_continuous(self):
        """Regression floor: a coin flip reappearing here should fail loudly.

        ``block_shuffle_variable`` draws real ``expovariate`` gap lengths
        and ``webm_chunk_mutate`` packs ``random() * 10`` into an IEEE
        double for a WEBM float field -- both are genuine continuous
        values, not probability thresholds, so neither is rewritten.
        """
        names = _operator_names()
        continuous = [
            n for n in names if _enumerate_operator(n, self.SEED, self.MAX_LEN)[0] == "continuous"
        ]
        assert set(continuous) == {"block_shuffle_variable", "webm_chunk_mutate"}


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
