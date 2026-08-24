"""Regression tests for the MonteCarloScheduler pairwise transition matrix.

Finding 18: pairwise transition tracking could never bootstrap. `record()`
populates `transition_counts` only when `self._prev_op` is set, and the only
assignment to `_prev_op` sat in `select_op()` AFTER the early return taken
whenever `prev_op not in self.transition_total` — i.e. after the branch that
requires a non-empty matrix. Empty matrix -> early return -> `_prev_op` never
set -> `record()` never counts a transition -> matrix stays empty.

The existing tests in `tests/test_montecarlo.py` all missed this because they
assign `mc._prev_op` by hand before calling `record()`, pre-loading exactly
the internal state the production path can never reach. That is the same
shape as `test_hex_escape` and `test_different_stderr`: an assertion written
against the code rather than against the system. The tests here drive the
scheduler only through its public interface, in the order
`services/operators.py` and `services/fuzzer.py` actually call it.

The second assignment defect was subtler: `_prev_op = best_op` in `select_op`
set the predecessor to the operator being selected, so when `record()` ran for
that same operator the `_prev_op != name` guard rejected the pair. `_prev_op`
is now advanced by `record()` alone.
"""

import random

from fuzzer_tool.core.schedulers.monte_carlo import MonteCarloScheduler

OPS = ["a", "b", "c", "d"]


def drive(mc, n, outcome, seed=0):
    """Drive the scheduler exactly as the fuzzer does.

    `services/operators.py` does `op = mc.select_op(ops, prev_op=prev)` then
    `prev = op`; `services/fuzzer.py` later calls `mc.record(op, ok)`.
    """
    random.seed(seed)
    prev = None
    hits = 0
    for _ in range(n):
        op = mc.select_op(OPS, prev_op=prev)
        ok = outcome(prev, op)
        hits += ok
        mc.record(op, success=ok)
        prev = op
    return hits


class TestBootstrap:
    def test_transitions_populate_from_public_interface(self):
        """The matrix must fill without anyone poking `_prev_op` first."""
        mc = MonteCarloScheduler(pairwise_blend=0.5, arm_decay=1.0)
        for o in OPS:
            mc.init_arm(o)

        drive(mc, 500, lambda prev, op: True)

        assert mc.transition_counts, "pairwise matrix never bootstrapped"
        assert sum(mc.transition_total.values()) > 0

    def test_prev_op_is_the_previously_recorded_op(self):
        """`_prev_op` is the predecessor, not the operator being recorded."""
        mc = MonteCarloScheduler(pairwise_blend=0.5)
        for o in ("a", "b"):
            mc.init_arm(o)

        mc.record("a", success=True)
        assert mc._prev_op == "a"

        mc.record("b", success=True)
        assert mc.transition_counts["a"]["b"] == 1
        assert mc.transition_total["a"] == 1
        assert mc._prev_op == "b"

    def test_select_op_does_not_advance_prev_op(self):
        """Selecting must not clobber the predecessor record() needs."""
        mc = MonteCarloScheduler(pairwise_blend=1.0, arm_decay=1.0)
        for o in OPS:
            mc.init_arm(o)

        mc.record("a", success=True)
        assert mc._prev_op == "a"

        # Both branches of select_op: no transition data yet (early return),
        # then again once the matrix is populated.
        mc.select_op(OPS, prev_op="a")
        assert mc._prev_op == "a"

        mc.record("b", success=True)
        mc.select_op(OPS, prev_op="a")
        assert mc._prev_op == "b"

    def test_failure_does_not_count_but_still_advances(self):
        mc = MonteCarloScheduler(pairwise_blend=0.5)
        for o in ("a", "b", "c"):
            mc.init_arm(o)

        mc.record("a", success=True)
        mc.record("b", success=False)
        assert "a" not in mc.transition_counts
        # A failed step is still a step: the chain must not stall on it.
        assert mc._prev_op == "b"

        mc.record("c", success=True)
        assert mc.transition_counts["b"]["c"] == 1


class TestSignalIsUsed:
    def test_conditional_reward_is_learned(self):
        """With a reward that exists only on a specific transition, a blended
        scheduler must beat the pure-Thompson one it degenerated into.

        Ground truth: "b" pays off only when it follows "a". No unconditional
        arm posterior can represent that, so any separation between the two
        configurations is attributable to the pairwise term.
        """

        def outcome(prev, op):
            return op == "b" and prev == "a"

        def run(blend, seed):
            mc = MonteCarloScheduler(pairwise_blend=blend, arm_decay=1.0)
            for o in OPS:
                mc.init_arm(o)
            return drive(mc, 3000, outcome, seed=seed)

        seeds = range(8)
        blended = [run(0.6, s) for s in seeds]
        thompson = [run(0.0, s) for s in seeds]

        # Averaged over seeds to keep this from being a coin flip; the
        # measured gap is roughly 30% on this harness.
        assert sum(blended) > sum(thompson)

    def test_learned_transition_steers_selection(self):
        mc = MonteCarloScheduler(pairwise_blend=1.0, arm_decay=1.0)
        for o in OPS:
            mc.init_arm(o)

        # Teach a -> b through the public interface only.
        for _ in range(50):
            mc.record("a", success=False)
            mc.record("b", success=True)

        picks = [mc.select_op(OPS, prev_op="a") for _ in range(100)]
        assert picks.count("b") > 90
        # Assert the matrix is what carries this, not "b"'s arm posterior:
        # without the transition entry the same picks would follow from
        # Thompson alone, and this test would pass on the broken scheduler.
        assert mc.transition_counts["a"]["b"] == 50
        assert mc.transition_total["a"] == 50
