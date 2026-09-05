"""A Beta posterior is a *pair*; record() must never create half of one.

``record()`` writes ``op_alpha`` on success and ``op_beta`` on failure, each
via ``.get(name, 1.0)``. An arm that has only ever succeeded therefore exists
in ``op_alpha`` and not in ``op_beta``. The decay block in ``select_op()``
iterates ``op_alpha`` and subscripts ``op_beta[k]`` bare, so the next decay
checkpoint raises KeyError.

``init_arm()`` seeds both dicts, which is why startup-registered operators are
safe; operators that self-register at runtime through
``REGISTRY.register_mutator()`` (weizz_structural, fractal_voronoi) reach
``record()`` without it. ``_cap()`` does not heal the gap — it returns early
while ``alpha + beta`` is under ``max_pseudocount`` (200), which is exactly
the state a fresh arm is in.
"""

import pytest

from fuzzer_tool.core.schedulers.hierarchical import HierarchicalBanditScheduler

BASE_OPS = ["bit_flip", "byte_flip", "arith8"]
LATE_OP = "voronoi_cell_swap"


def _armed():
    sched = HierarchicalBanditScheduler()
    for op in BASE_OPS:
        sched.init_arm(op)
    return sched


@pytest.mark.parametrize("success", [True, False])
def test_record_keeps_both_op_params(success):
    """One-sided outcomes still leave a complete posterior pair."""
    sched = _armed()
    sched.record(LATE_OP, success=success)

    assert LATE_OP in sched.op_alpha
    assert LATE_OP in sched.op_beta


@pytest.mark.parametrize("success", [True, False])
def test_record_keeps_both_cat_params(success):
    """Same invariant at the category level."""
    sched = _armed()
    sched.record(LATE_OP, success=success)
    cat = sched._cat_for(LATE_OP)

    assert cat in sched.cat_alpha
    assert cat in sched.cat_beta


def test_decay_survives_success_only_runtime_arm():
    """The reported crash: decay checkpoint after a success-only late arm."""
    sched = _armed()
    sched.record(LATE_OP, success=True)
    sched._total_pulls = sched.decay_interval  # land exactly on a decay tick

    chosen = sched.select_op([*BASE_OPS, LATE_OP])

    assert chosen in {*BASE_OPS, LATE_OP}


def test_decay_survives_failure_only_runtime_arm():
    """Mirror case — beta-only arms must not break the decay walk either."""
    sched = _armed()
    sched.record(LATE_OP, success=False)
    sched._total_pulls = sched.decay_interval

    assert sched.select_op([*BASE_OPS, LATE_OP]) in {*BASE_OPS, LATE_OP}


def test_decay_scales_both_sides_of_a_late_arm():
    """Adversarial: the repaired key must decay, not merely exist."""
    sched = _armed()
    sched.record(LATE_OP, success=True)
    before = (sched.op_alpha[LATE_OP], sched.op_beta[LATE_OP])
    sched._total_pulls = sched.decay_interval
    sched.select_op([*BASE_OPS, LATE_OP])

    assert sched.op_alpha[LATE_OP] == pytest.approx(before[0] * sched.arm_decay)
    assert sched.op_beta[LATE_OP] == pytest.approx(before[1] * sched.arm_decay)


def test_unregistered_arm_defaults_match_init_arm():
    """A repaired arm must land on the same prior init_arm() would have set."""
    seeded = HierarchicalBanditScheduler()
    seeded.init_arm(LATE_OP)

    lazy = HierarchicalBanditScheduler()
    lazy.record(LATE_OP, success=False)

    assert lazy.op_alpha[LATE_OP] == seeded.op_alpha[LATE_OP]
    assert lazy.op_beta[LATE_OP] == seeded.op_beta[LATE_OP] + 1
