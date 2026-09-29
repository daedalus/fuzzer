"""Tests for core/op_chain2.py -- sparse second-order operator transition table."""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.op_chain2 import SecondOrderChain


class TestRecord:
    def test_counts_observed_triples_only(self):
        c = SecondOrderChain()
        c.record("a", "b", "c", success=True)
        c.record("a", "b", "c", success=True)
        c.record("a", "b", "d", success=True)
        assert c.total("a", "b") == 3
        assert c.count("a", "b", "c") == 2
        assert c.count("a", "b", "d") == 1
        assert c.count("x", "y", "z") == 0

    def test_failures_are_not_counted(self):
        c = SecondOrderChain()
        c.record("a", "b", "c", success=False)
        assert c.total("a", "b") == 0

    def test_self_transition_ignored_like_first_order(self):
        c = SecondOrderChain()
        c.record("a", "b", "b", success=True)
        assert c.total("a", "b") == 0

    def test_none_history_ignored(self):
        c = SecondOrderChain()
        c.record(None, "b", "c", success=True)
        c.record("a", None, "c", success=True)
        assert len(c) == 0


class TestScores:
    def test_unseen_context_returns_none(self):
        assert SecondOrderChain().scores(["a", "b"], "x", "y") is None

    def test_scores_are_dirichlet_smoothed_and_sum_to_one(self):
        c = SecondOrderChain()
        for _ in range(3):
            c.record("a", "b", "c", success=True)
        s = c.scores(["c", "d", "e"], "a", "b")
        assert s is not None
        assert s["c"] == pytest.approx((3 + 1) / (3 + 3))
        assert s["d"] == pytest.approx(1 / 6)
        assert sum(s.values()) == pytest.approx(1.0)

    def test_distinguishes_contexts_first_order_would_merge(self):
        # After b: c follows when preceded by a, d follows when preceded by x.
        c = SecondOrderChain()
        for _ in range(20):
            c.record("a", "b", "c", success=True)
            c.record("x", "b", "d", success=True)
        sa = c.scores(["c", "d"], "a", "b")
        sx = c.scores(["c", "d"], "x", "b")
        assert sa["c"] > sa["d"]
        assert sx["d"] > sx["c"]


class TestBounded:
    def test_never_exceeds_cap(self):
        c = SecondOrderChain(max_contexts=50)
        rng = random.Random(1)
        for _ in range(5000):
            p2, p1, n = (f"op{rng.randrange(30)}" for _ in range(3))
            c.record(p2, p1, n, success=True)
            assert len(c) <= 50

    def test_eviction_keeps_heavily_observed_context(self):
        c = SecondOrderChain(max_contexts=10)
        for _ in range(100):
            c.record("hot", "ctx", "n", success=True)
        for i in range(200):
            c.record(f"a{i}", f"b{i}", "n", success=True)
        assert c.total("hot", "ctx") == 100

    def test_cap_must_be_positive(self):
        with pytest.raises(ValueError):
            SecondOrderChain(max_contexts=0)


class TestState:
    def test_roundtrip(self):
        c = SecondOrderChain()
        c.record("a", "b", "c", success=True)
        c.record("a", "b", "d", success=True)
        d = SecondOrderChain.from_state(c.to_state())
        assert d.count("a", "b", "c") == 1
        assert d.total("a", "b") == 2

    def test_from_malformed_state_is_empty(self):
        assert len(SecondOrderChain.from_state({"junk": 1})) == 0
        assert len(SecondOrderChain.from_state(None)) == 0
