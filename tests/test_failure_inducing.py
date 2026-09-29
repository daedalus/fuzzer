"""Tests for core/failure_inducing.py -- FIC-style failing-combination isolation."""

from __future__ import annotations

import itertools
import random

import pytest

from fuzzer_tool.core import covering_array as ca
from fuzzer_tool.core import failure_inducing as fi

VS = [(0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2)]


def _fail_when(schema: dict[int, int]):
    """Oracle failing iff the row contains every (param, value) of *schema*."""
    return lambda row: all(row[i] == v for i, v in schema.items())


class TestIsolate:
    @pytest.mark.parametrize(
        "schema", [{2: 1}, {0: 2, 3: 1}, {1: 0, 2: 2, 4: 1}, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}]
    )
    def test_recovers_exact_schema(self, schema):
        row = tuple(schema.get(i, 0) for i in range(5))
        # make sure the seed row really has the schema and fails
        res = fi.isolate(row, VS, _fail_when(schema), rng=random.Random(3))
        assert res.status == fi.ISOLATED
        assert res.params == schema
        assert not _fail_when(schema)(res.passing_row)

    def test_probe_count_is_linear_not_exhaustive(self):
        row = (1, 1, 1, 1, 1)
        res = fi.isolate(row, VS, _fail_when({0: 1, 4: 1}), rng=random.Random(0))
        assert res.probes <= 2 * len(VS) + 4  # << 3**5 = 243

    def test_every_single_removal_passes_when_not_monotone_free(self):
        """1-minimality: dropping any schema param from the hybrid must pass."""
        schema = {0: 2, 3: 1}
        oracle = _fail_when(schema)
        row = (2, 0, 0, 1, 0)
        res = fi.isolate(row, VS, oracle, rng=random.Random(5))
        for drop in res.params:
            keep = [i for i in res.params if i != drop]
            hyb = tuple(row[i] if i in keep else res.passing_row[i] for i in range(5))
            assert not oracle(hyb)

    def test_two_disjoint_schemas_returns_one_minimal(self):
        """Either {0:1} or {1:1} alone fails; result must be one of them, minimal."""
        oracle = lambda r: r[0] == 1 or r[1] == 1  # noqa: E731
        res = fi.isolate((1, 1, 0, 0, 0), VS, oracle, rng=random.Random(1))
        assert res.status == fi.ISOLATED
        assert res.params in ({0: 1}, {1: 1})

    def test_not_failing_row(self):
        res = fi.isolate((0, 0, 0, 0, 0), VS, lambda r: False)
        assert res.status == fi.NOT_FAILING
        assert res.params == {}

    def test_unconditional_failure_has_no_schema(self):
        res = fi.isolate((0, 1, 2, 0, 1), VS, lambda r: True, companion_tries=5)
        assert res.status == fi.UNCONDITIONAL
        assert res.params == {}

    def test_single_value_domains_are_unresolved(self):
        vs = [(0,), (0, 1, 2), (0, 1, 2)]
        res = fi.isolate((0, 1, 1), vs, lambda r: r[1] == 1, rng=random.Random(2))
        assert res.status == fi.ISOLATED
        assert res.unresolved == (0,)
        assert res.params == {1: 1}

    def test_seed_value_outside_domain_is_allowed(self):
        vs = [(0, 1), (0, 1)]
        res = fi.isolate((640, 1), vs, lambda r: r[0] == 640, rng=random.Random(0))
        assert res.params == {0: 640}

    def test_verify_confirms_sufficient_schema(self):
        res = fi.isolate(
            (1, 0, 2, 0, 0), VS, _fail_when({0: 1, 2: 2}), rng=random.Random(0), verify_samples=20
        )
        assert res.verified is True and res.counterexample is None

    def test_verify_flags_insufficient_schema(self):
        # Non-monotone: fails iff r0==1 and r1 != 2. When the companion's r1
        # happens to be 1 the reduction sees {0} as sufficient, but random
        # rows with r0==1 and r1==2 pass -- verification must catch that.
        oracle = lambda r: r[0] == 1 and r[1] != 2  # noqa: E731
        outs = [
            fi.isolate((1, 0, 0, 0, 0), VS, oracle, rng=random.Random(s), verify_samples=50)
            for s in range(12)
        ]
        assert any(o.verified is False for o in outs)
        for o in outs:
            assert o.status == fi.ISOLATED
            if o.verified is False:
                assert o.params == {0: 1}
                assert o.counterexample is not None and not oracle(o.counterexample)

    def test_budget_truncates_but_schema_still_fails(self):
        oracle = _fail_when({0: 1, 1: 1, 2: 1})
        row = (1, 1, 1, 1, 1)
        res = fi.isolate(row, VS, oracle, rng=random.Random(0), max_probes=4)
        assert res.status == fi.TRUNCATED
        hyb = tuple(row[i] if i in res.params else res.passing_row[i] for i in range(5))
        assert oracle(hyb)
        assert res.probes <= 4

    def test_memoizes_repeated_rows(self):
        calls = []

        def oracle(r):
            calls.append(r)
            return r[0] == 1

        fi.isolate((1, 0, 0, 0, 0), VS, oracle, rng=random.Random(0))
        assert len(calls) == len(set(calls))

    def test_deterministic_default_rng(self):
        a = fi.isolate((1, 0, 0, 0, 0), VS, _fail_when({0: 1}))
        b = fi.isolate((1, 0, 0, 0, 0), VS, _fail_when({0: 1}))
        assert a == b

    def test_length_mismatch_and_empty_domain_raise(self):
        with pytest.raises(ValueError):
            fi.isolate((1, 2), VS, lambda r: True)
        with pytest.raises(ValueError):
            fi.isolate((1,), [()], lambda r: True)

    def test_works_with_stdlib_random_and_randint_only_rng(self):
        class R:
            def __init__(self):
                self._r = random.Random(9)

            def randint(self, a, b):
                return self._r.randint(a, b)

        res = fi.isolate((1, 0, 0, 0, 0), VS, _fail_when({0: 1}), rng=R())
        assert res.params == {0: 1}


class TestWithCoveringArray:
    def test_covering_array_row_that_fails_is_reduced_to_its_pair(self):
        """End to end with the covering array: a planted pairwise bug is
        found in some row, and isolation returns exactly that pair."""
        vs = [(0, 1, 2, 3)] * 6
        bug = {1: 3, 4: 2}
        oracle = _fail_when(bug)
        rows = ca.generate(vs, t=2, rng=random.Random(11))
        failing = [r for r in rows if oracle(r)]
        assert failing, "t=2 coverage must hit every planted pairwise bug"
        for r in failing:
            assert fi.isolate(r, vs, oracle, rng=random.Random(0)).params == bug

    def test_exhaustive_small_space_all_single_and_pair_bugs(self):
        vs = [(0, 1), (0, 1), (0, 1)]
        for size in (1, 2):
            for idxs in itertools.combinations(range(3), size):
                schema = dict.fromkeys(idxs, 1)
                row = tuple(1 if i in schema else 0 for i in range(3))
                res = fi.isolate(row, vs, _fail_when(schema), rng=random.Random(0))
                assert res.params == schema


class TestFormat:
    def test_named_and_unnamed(self):
        s = fi.FailureSchema(status=fi.ISOLATED, params={3: 16, 1: 2}, probes=9, verified=True)
        assert (
            fi.format_schema(s, ["a", "b", "c", "d"]) == "b=2 & d=16 (isolated, verified, 9 probes)"
        )
        assert fi.format_schema(s).startswith("p1=2 & p3=16")

    def test_empty(self):
        s = fi.FailureSchema(status=fi.UNCONDITIONAL, probes=3)
        assert "no parameter isolated" in fi.format_schema(s)
