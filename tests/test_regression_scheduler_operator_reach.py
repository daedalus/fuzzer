"""Regression tests: every scheduler must be able to select every operator.

The failure that motivates this file was not visible to the
existing scheduler tests (import-graph independence, fallback precedence,
convergence-to-best-arm) because all three assume the arm is *reachable* and
only ask how often it gets pulled.

``HierarchicalBanditScheduler`` resolved an operator's category by looking it
up in ``OPERATOR_CATEGORIES``, a snapshot of ``REGISTRY.categories()`` taken
at import. Operators registered afterwards through
``REGISTRY.register_mutator()`` -- the documented extension path in
``core/mutator_interface`` -- are absent from that snapshot, and a miss meant
"skip this operator" in both ``init_arm`` and ``select_op``. The arm was
never initialized and never a candidate: 0 pulls out of 60,000 selections,
permanently, for an operator every other scheduler could reach.

``GPUCBScheduler`` had the milder form of the same bug -- an all-zero feature
vector, which under the RBF kernel is not "similar to nothing" but ~0.61
similar to *every* category at the default length scale, above
``kernel_floor``.

These are reachability floors, not convergence tests: each asserts only that
an operator is selected *at least once*, which is the property a bandit
cannot trade away for exploitation no matter how it weighs its arms.
"""

from __future__ import annotations

import random

import pytest

from fuzzer_tool.core import schedulers as S
from fuzzer_tool.core.mutator_interface import MutatorBase
from fuzzer_tool.core.operator_categories import UNCATEGORIZED, category_of
from fuzzer_tool.core.operator_registry import REGISTRY
from fuzzer_tool.services.operators import CONTEXT_DIM

# Rounds per reachability run. The slowest built-in scheduler to cover the
# whole table is MOpt, whose last operator first fires at ~2.5k selections
# with 147 arms; 20k leaves a wide margin without making the suite slow.
_ROUNDS = 20_000

#: Bernoulli success rate fed to record(). Deliberately low and uniform: a
#: uniform reward gives no scheduler a reason to prefer any arm, so anything
#: that still fails to reach an operator is failing structurally rather than
#: because exploitation crowded the arm out.
_P_SUCCESS = 0.05


def _drive(select, record, ops, rounds=_ROUNDS, seed=1234):
    """Run *rounds* select/record cycles; return the set of ops ever chosen."""
    random.seed(seed)
    seen: set[str] = set()
    for _ in range(rounds):
        op = select(ops)
        seen.add(op)
        record(op, random.random() < _P_SUCCESS)
    return seen


def _all_operator_schedulers():
    """(label, select_fn, record_fn) for every operator-selection scheduler.

    Built as factories rather than instances so each test gets a fresh
    scheduler, and kept as an explicit list rather than derived from
    ``schedulers.__all__`` so that adding a scheduler without adding it here
    is a visible omission -- see ``test_every_exported_scheduler_is_covered``.
    """
    cmaes = S.CMAESScheduler()
    mc = S.MonteCarloScheduler()
    mopt = S.MOptScheduler()
    replicator = S.ReplicatorScheduler()
    exp3 = S.Exp3Scheduler()
    eps = S.EpsilonGreedyScheduler()
    hier = S.HierarchicalBanditScheduler()
    gp = S.GPUCBScheduler()
    lin = S.ContextualLinUCBScheduler(dim=CONTEXT_DIM)
    ducb = S.DUCBScheduler()
    swucb = S.SWUCBScheduler()
    cucb = S.CUCBScheduler()

    def _ctx(_op):
        return [random.random() for _ in range(CONTEXT_DIM)]

    return [
        ("CMAESScheduler", cmaes, cmaes.select_op, cmaes.record),
        ("MonteCarloScheduler", mc, mc.select_op, mc.record),
        # MOpt returns (op, particle_id) -- the services layer unpacks it.
        ("MOptScheduler", mopt, lambda o: mopt.select_op(o)[0], mopt.record),
        ("ReplicatorScheduler", replicator, replicator.select_op, replicator.record),
        ("Exp3Scheduler", exp3, exp3.select_op, exp3.record),
        ("EpsilonGreedyScheduler", eps, eps.select_op, eps.record),
        ("HierarchicalBanditScheduler", hier, hier.select_op, hier.record),
        ("GPUCBScheduler", gp, gp.select_op, gp.record),
        # LinUCB takes a per-arm context callable and a numeric reward.
        (
            "ContextualLinUCBScheduler",
            lin,
            lambda o: lin.select_op(o, _ctx),
            lambda n, ok: lin.record(n, _ctx(n), 1.0 if ok else 0.0),
        ),
        ("DUCBScheduler", ducb, ducb.select_op, ducb.record),
        ("SWUCBScheduler", swucb, swucb.select_op, swucb.record),
        # CUCB batches a round; select_op() closes any round left open.
        ("CUCBScheduler", cucb, cucb.select_op, cucb.record),
    ]


@pytest.fixture
def plugin_mutator():
    """Register a mutator through the runtime extension path, then remove it.

    This is the case the import-time category snapshot cannot see. The
    teardown reaches into ``REGISTRY._ops`` because the registry has no
    public unregister -- deliberately, since removing an operator mid-run
    would desynchronise every scheduler's arm table. A test that adds one to
    the process-wide singleton still has to put the table back.
    """

    class _PluginMutator(MutatorBase):
        name = "reach_probe_plugin"
        # A category the built-in taxonomy does not have, so this also
        # exercises GP-UCB widening its one-hot basis for a new axis.
        category = "plugin_probe"

        def mutate(self, data, rng, max_len=0, **ctx):
            return data.upper()

    REGISTRY.register_mutator(_PluginMutator())
    try:
        yield "reach_probe_plugin"
    finally:
        REGISTRY._ops.pop("reach_probe_plugin", None)
        REGISTRY._categories_cache = None


class TestAllSchedulersReachAllOperators:
    @pytest.mark.parametrize(
        "label",
        [entry[0] for entry in _all_operator_schedulers()],
    )
    def test_every_registered_operator_is_selected(self, label):
        """Registering the whole operator table must leave none unselectable."""
        names = REGISTRY.names()
        entry = next(e for e in _all_operator_schedulers() if e[0] == label)
        _, scheduler, select, record = entry
        for name in names:
            scheduler.init_arm(name)

        seen = _drive(select, record, names)
        missing = [n for n in names if n not in seen]
        assert not missing, f"{label} never selected {len(missing)} operator(s): {missing[:10]}"

    def test_every_exported_scheduler_is_covered(self):
        """schedulers.__all__ minus the seed scheduler == the list above.

        MCTSSeedScheduler picks *seeds*, not operators, so it has no
        init_arm/select_op(ops) surface and is excluded by design. Anything
        else appearing in __all__ without appearing here is a scheduler
        nothing checks for reachability.
        """
        covered = {entry[0] for entry in _all_operator_schedulers()}
        exported = set(S.__all__) - {"MCTSSeedScheduler", "AlphaBetaMCTSSeedScheduler"}
        assert exported == covered, f"uncovered schedulers: {sorted(exported - covered)}"


class TestRuntimeRegisteredMutatorIsReachable:
    def test_registry_knows_the_plugin(self, plugin_mutator):
        assert plugin_mutator in REGISTRY.names()
        assert category_of(plugin_mutator) == "plugin_probe"

    def test_hierarchical_initializes_and_selects_it(self, plugin_mutator):
        """The exact bug: 0 pulls out of 60,000 before the fix."""
        names = REGISTRY.names()
        hier = S.HierarchicalBanditScheduler()
        for name in names:
            hier.init_arm(name)

        assert plugin_mutator in hier.op_alpha, (
            "init_arm() dropped a registered operator missing from the import-time snapshot"
        )
        seen = _drive(hier.select_op, hier.record, names)
        assert plugin_mutator in seen

    def test_gp_ucb_gives_it_a_real_one_hot(self, plugin_mutator):
        """Not the all-zero vector, which is ~0.61 similar to everything."""
        gp = S.GPUCBScheduler()
        names = REGISTRY.names()
        for name in names:
            gp.init_arm(name)

        feat = gp._features[plugin_mutator]
        assert sum(feat) == 1.0, "plugin operator got a degenerate feature vector"
        # Widening the basis must not desynchronise the older vectors --
        # _rbf() zips them with strict=True.
        assert {len(f) for f in gp._features.values()} == {len(gp._cat_names)}
        assert plugin_mutator in _drive(gp.select_op, gp.record, names)

    def test_widening_preserves_existing_kernel_values(self, plugin_mutator):
        """Zero-padding an existing one-hot must leave every kernel entry equal."""
        builtins = [n for n in REGISTRY.names() if n != plugin_mutator]
        probe = builtins[:6]

        before = S.GPUCBScheduler()
        for name in builtins:
            before.init_arm(name)
        baseline = before.kernel_matrix(probe)

        after = S.GPUCBScheduler()
        for name in builtins:
            after.init_arm(name)
        after.init_arm(plugin_mutator)  # widens the basis
        assert after.kernel_matrix(probe) == baseline


class TestUnknownOperatorNamesStaySelectable:
    """A name the registry has never heard of must not be silently dropped.

    Reachable through a resumed state file written by a build with a
    different operator table, or a harness passing synthetic names. The old
    behaviour made such an operator unselectable while still accepting it
    into the candidate list, so the fuzzer would appear to be offering it.
    """

    def test_category_of_falls_back_rather_than_raising(self):
        assert category_of("no_such_operator_xyz") == UNCATEGORIZED

    def test_hierarchical_selects_a_mix_of_known_and_unknown(self):
        ops = ["bit_flip", "byte_flip", "havoc", "no_such_operator_xyz"]
        hier = S.HierarchicalBanditScheduler()
        for op in ops:
            hier.init_arm(op)
        seen = _drive(hier.select_op, hier.record, ops, rounds=4_000)
        assert set(ops) == seen

    def test_hierarchical_records_outcomes_for_unknown_names(self):
        hier = S.HierarchicalBanditScheduler()
        hier.init_arm("no_such_operator_xyz")
        hier.record("no_such_operator_xyz", True)
        assert hier.op_alpha["no_such_operator_xyz"] > 1.0
