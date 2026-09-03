"""The unreachable-function penalty must come from every target, not the last one.

``_compute_distances`` documents the penalty as "max reachable distance
+ 5". It read ``visited`` after the per-target loop had finished, so the
value came from whichever target ``target_names`` — a set — happened to
yield last. With two targets at different depths the penalty therefore
flipped between two values depending on set iteration order, and could
land *below* a distance it was supposed to sit above.

The graph here is built to make that visible: one target sits at the end
of a long chain, the other has no callers at all, so the two candidate
penalties are 26 and 5 while the deepest assigned distance is 21. Reading
the wrong frontier does not merely shift the penalty, it drops it below
distances it is supposed to dominate.

Falsification note: because the ordering comes from a set of strings,
the pre-fix code is *order-dependent*, not uniformly wrong. Run against
it, this module fails under PYTHONHASHSEED 0,1,2,3,5,6,7 and passes
under 4. That flakiness is the bug; it is why the defect survived a
suite that only ever saw one ordering per run.
"""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core.distance import TargetDistance
from fuzzer_tool.core.randomness import CorpusInvariants


def build_calculator(call_graph, functions, target_names):
    """A calculator with just enough state for _compute_distances."""
    calc = TargetDistance.__new__(TargetDistance)
    calc.call_graph = call_graph
    calc.functions = functions
    calc._distances = {}
    calc.target_addrs = list(range(len(target_names)))
    addr_of = dict(enumerate(target_names))
    calc._addr_to_function = lambda a: addr_of.get(a)
    return calc


class TestUnreachablePenalty:
    def test_penalty_uses_the_deepest_target_not_the_last(self):
        # c0 -> c1 -> ... -> c20 -> deep_target, and an isolated
        # shallow_target with no callers at all. `orphan` reaches neither.
        # The two candidate penalties are 25 and 5, and the deepest
        # assigned distance is 21 — so reading the wrong frontier does not
        # merely shift the penalty, it drops it below distances it is
        # supposed to dominate.
        call_graph = {f"c{i}": {f"c{i + 1}"} for i in range(20)}
        call_graph["c20"] = {"deep_target"}
        functions = {f"c{i}": None for i in range(21)}
        functions.update({"deep_target": None, "shallow_target": None, "orphan": None})
        calc = build_calculator(call_graph, functions, ["deep_target", "shallow_target"])
        calc._compute_distances()

        assert calc._distances["orphan"] == pytest.approx(26.0)

    def test_penalty_sits_above_every_assigned_distance(self):
        """The invariant the penalty exists to hold."""
        call_graph = {f"f{i}": {f"f{i + 1}"} for i in range(20)}
        functions = {f"f{i}": None for i in range(21)}
        functions["orphan"] = None
        calc = build_calculator(call_graph, functions, ["f20", "f1"])
        calc._compute_distances()

        penalty = calc._distances["orphan"]
        assigned = [v for k, v in calc._distances.items() if k != "orphan"]
        assert assigned
        assert penalty > max(assigned)

    def test_order_independent(self):
        """Same graph, both target orderings, same penalty."""
        call_graph = {"a": {"b"}, "b": {"c"}, "c": {"t_deep"}, "z": {"t_near"}}
        functions = {k: None for k in ("a", "b", "c", "z", "t_deep", "t_near", "orphan")}
        first = build_calculator(call_graph, functions, ["t_deep", "t_near"])
        first._compute_distances()
        second = build_calculator(call_graph, functions, ["t_near", "t_deep"])
        second._compute_distances()
        assert first._distances["orphan"] == second._distances["orphan"]

    def test_single_target_unchanged(self):
        """The one-target case had no bug and must keep its old value."""
        call_graph = {"a": {"b"}, "b": {"t"}}
        functions = {k: None for k in ("a", "b", "t", "orphan")}
        calc = build_calculator(call_graph, functions, ["t"])
        calc._compute_distances()
        assert calc._distances["orphan"] == pytest.approx(7.0)


class TestLockedBitRatioPopcount:
    """popcount(concat(bytes)) == sum(popcount(b)) — same number, one call."""

    def test_matches_per_byte_sum(self):
        rng = random.Random(20260902)
        for _ in range(300):
            mask = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 200)))
            want = sum(bin(m).count("1") for m in mask) / (8.0 * len(mask))
            assert CorpusInvariants(
                mask=mask, n_samples=16, common_length=len(mask)
            ).locked_bit_ratio == pytest.approx(want)

    @pytest.mark.parametrize(
        ("mask", "expected"),
        [
            (b"", 0.0),
            (b"\x00" * 8, 0.0),
            (b"\xff" * 8, 1.0),
            (b"\x0f", 0.5),
            # Leading zero bytes must not be swallowed by the integer
            # conversion: the divisor is len(mask), not the integer's width.
            (b"\x00\x00\xff", pytest.approx(1 / 3)),
        ],
    )
    def test_edge_cases(self, mask, expected):
        assert (
            CorpusInvariants(mask=mask, n_samples=16, common_length=len(mask)).locked_bit_ratio
            == expected
        )
