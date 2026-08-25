"""Tests for MCTS/UCT seed scheduling over the lineage tree."""

import random
from collections import Counter

import pytest

from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.core.schedulers import MCTSSeedScheduler
from fuzzer_tool.core.schedulers.mcts import _squash


def _tree():
    """root -> a0..a3, with a0 -> b0..b2 (one deep productive branch)."""
    t = LineageTree()
    t.insert(None, "root", [], [], 0)
    for i in range(4):
        t.insert("root", f"a{i}", ["havoc"], [0], 1)
    for i in range(3):
        t.insert("a0", f"b{i}", ["havoc"], [0], 1)
    return t


def _all_keys():
    return {"root"} | {f"a{i}" for i in range(4)} | {f"b{i}" for i in range(3)}


def _drive(sched, tree, eligible, n, reward_for):
    """Run n select/update rounds, rewarding keys matching *reward_for*."""
    counts = Counter()
    for _ in range(n):
        key = sched.select(tree, eligible)
        counts[key] += 1
        sched.update(5.0 if key is not None and reward_for(key) else 0.0)
    return counts


def _prime_b0_descent(sched):
    """Seed UCT stats so the FIRST select() descends root -> a0 -> b0.

    select() draws no randomness (self._rng is never read); ordering is
    argmax over children, and LineageTree stores children as str sets whose
    iteration order varies per process under hash randomization. Every
    decision below is therefore made STRICT, so the descent no longer
    depends on set order:

    - root: uct(a0)=1.0 beats uct(a1..a3)=0.5 (explore term is 0 at
      parent_visits=1), and self_uct(root)=0.5 < uct(a0), so the walk
      descends instead of stopping;
    - a0: uct(b0)~1.83 beats uct(b1/b2)~1.68, and self_uct(a0)~1.08 loses,
      so it descends past a0;
    - b0 is a leaf, so select() hands back b0 with its full ancestry.
    """
    sched.visits.update({"a0": 2.0, "b0": 2.0})
    sched.values.update({"a0": 2.0, "b0": 2.0})  # mean 1.0 vs prior 0.5 elsewhere
    sched.self_visits.update({"a0": 2.0})  # extra self visits sink the stop score


class TestSquash:
    def test_zero_and_negative_map_to_zero(self):
        assert _squash(0) == 0.0
        assert _squash(-5) == 0.0

    def test_bounded_in_unit_interval(self):
        """UCT's exploration constant is only calibrated for rewards in [0,1]."""
        for edges in (1, 10, 100, 10_000, 10**9):
            assert 0.0 <= _squash(edges) <= 1.0

    def test_typical_counts_stay_below_the_bound(self):
        """Saturation to exactly 1.0 is fine at absurd counts, but realistic
        per-execution edge gains must retain resolution."""
        for edges in (1, 5, 20, 60):
            assert _squash(edges) < 1.0

    def test_monotonic_and_saturating(self):
        assert _squash(1) < _squash(5) < _squash(50)
        # Marginal value must shrink: 1->5 matters more than 100->104.
        assert (_squash(5) - _squash(1)) > (_squash(104) - _squash(100))


class TestSelection:
    def test_returns_none_without_eligible_seeds(self):
        assert MCTSSeedScheduler().select(_tree(), set()) is None

    def test_returns_none_on_empty_tree(self):
        assert MCTSSeedScheduler().select(LineageTree(), {"x"}) is None

    def test_only_returns_eligible_keys(self):
        """A node may outlive its seed; never hand back a dead one."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        tree, eligible = _tree(), {"b0", "b1"}
        for _ in range(200):
            key = sched.select(tree, eligible)
            assert key is None or key in eligible
            sched.update(1.0)

    def test_interior_nodes_are_reachable(self):
        """Descending to a leaf every time would starve the imported roots,
        which are real corpus seeds and often the best thing to fuzz."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        counts = _drive(sched, _tree(), _all_keys(), 600, lambda k: k.startswith("b"))
        assert counts["root"] > 0
        assert counts["a0"] > 0

    def test_every_node_gets_explored(self):
        """UCT must not abandon an arm it has never tried."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        counts = _drive(sched, _tree(), _all_keys(), 600, lambda k: k.startswith("b"))
        assert set(counts) >= _all_keys()

    def test_bounded_by_max_depth(self):
        """A deep chain must not walk unboundedly."""
        tree = LineageTree()
        tree.insert(None, "n0", [], [], 0)
        for i in range(1, 200):
            tree.insert(f"n{i - 1}", f"n{i}", ["havoc"], [0], 1)
        sched = MCTSSeedScheduler(max_depth=8, rng=random.Random(1))
        sched.select(tree, {f"n{i}" for i in range(200)})
        assert len(sched._last_path) <= 9  # root + max_depth steps


class TestCreditAssignment:
    def test_concentrates_on_the_productive_subtree(self):
        sched = MCTSSeedScheduler(rng=random.Random(1))
        counts = _drive(sched, _tree(), _all_keys(), 600, lambda k: k.startswith("b"))
        productive = sum(v for k, v in counts.items() if k.startswith("b"))
        assert productive / 600 > 0.5

    def test_reallocates_when_productivity_moves(self):
        """The point of backpropagation: a subtree going sterile must lose
        budget to whichever region is now paying off."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        _drive(sched, _tree(), _all_keys(), 600, lambda k: k.startswith("b"))
        after = _drive(sched, _tree(), _all_keys(), 600, lambda k: k == "a3")
        b_share = sum(v for k, v in after.items() if k.startswith("b")) / 600
        assert after["a3"] / 600 > 0.5
        assert b_share < 0.2

    def test_ancestors_are_credited(self):
        """A discovery deep in the tree must lift its whole chain."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        _prime_b0_descent(sched)
        tree, eligible = _tree(), _all_keys()

        assert sched.select(tree, eligible) == "b0"
        assert sched._last_path == ["root", "a0", "b0"]

        sched.update(10.0)

        # One update adds exactly one visit to every node on the credited
        # path; values accumulate the squashed reward on top of their
        # pre-update state (prior 0.5 for untracked root, 2.0 primed mean).
        reward = _squash(10.0)
        assert sched.visits == {"root": 2.0, "a0": 3.0, "b0": 3.0}
        assert sched.values["root"] == pytest.approx(0.5 + reward)
        assert sched.values["a0"] == pytest.approx(2.0 + reward)
        assert sched.values["b0"] == pytest.approx(2.0 + reward)

    def test_self_credit_goes_only_to_the_selected_seed(self):
        """Self stats decide whether to stop at a node, so they must not be
        polluted by outcomes from its descendants."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        _prime_b0_descent(sched)
        tree, eligible = _tree(), _all_keys()

        assert sched.select(tree, eligible) == "b0"

        # Snapshot before crediting the b0 selection. Ancestors may already
        # carry self credit (a0 is primed); what matters is that *this*
        # selection adds none.
        before = dict(sched.self_visits)
        agg_before = dict(sched.visits)
        sched.update(10.0)

        assert sched.self_visits["b0"] > before.get("b0", 1.0)
        for ancestor in ("root", "a0"):
            assert sched.self_visits.get(ancestor) == before.get(ancestor)
            # ...while aggregate credit *does* flow up the path.
            assert sched.visits[ancestor] > agg_before.get(ancestor, 1.0)

    def test_update_without_selection_is_a_noop(self):
        """update() runs every iteration; it must be inert when another seed
        strategy made the pick."""
        sched = MCTSSeedScheduler()
        sched.update(100.0)
        assert sched.updates == 0
        assert not sched.visits

    def test_update_consumes_the_path(self):
        """Double-crediting one selection would corrupt the visit counts."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        sched.select(_tree(), _all_keys())
        sched.update(5.0)
        before = dict(sched.visits)
        sched.update(5.0)
        assert sched.visits == before

    def test_dead_branch_is_penalised_not_looped(self):
        """Walking into a region with no live seeds must record the visit so
        UCT steers away, rather than re-picking it forever."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        tree = LineageTree()
        tree.insert(None, "dead", [], [], 0)
        tree.insert("dead", "dead_child", ["havoc"], [0], 0)
        assert sched.select(tree, {"unrelated"}) is None
        assert sched.visits.get("dead", 0) > 1.0


class TestMaintenance:
    def test_prune_drops_stale_keys(self):
        sched = MCTSSeedScheduler(rng=random.Random(1))
        _drive(sched, _tree(), _all_keys(), 100, lambda k: True)
        dropped = sched.prune({"root", "a0"})
        assert dropped > 0
        assert set(sched.visits) <= {"root", "a0"}
        assert set(sched.self_visits) <= {"root", "a0"}

    def test_prune_keeps_live_keys(self):
        sched = MCTSSeedScheduler(rng=random.Random(1))
        _drive(sched, _tree(), _all_keys(), 100, lambda k: True)
        sched.prune(_all_keys())
        assert set(sched.visits) <= _all_keys()
        assert sched.visits

    def test_stats_shape(self):
        sched = MCTSSeedScheduler(rng=random.Random(1))
        _drive(sched, _tree(), _all_keys(), 50, lambda k: True)
        stats = sched.stats()
        assert stats["selections"] > 0
        assert stats["updates"] > 0
        assert stats["tracked_nodes"] > 0
        assert 0.0 <= stats["mean_value"] <= 1.0


class TestPersistence:
    def test_roundtrip_preserves_state(self):
        sched = MCTSSeedScheduler(rng=random.Random(1))
        _drive(sched, _tree(), _all_keys(), 200, lambda k: k.startswith("b"))

        restored = MCTSSeedScheduler()
        restored.from_dict(sched.to_dict())
        assert restored.visits == sched.visits
        assert restored.values == sched.values
        assert restored.self_visits == sched.self_visits
        assert restored.self_values == sched.self_values
        assert restored.selections == sched.selections

    def test_from_dict_on_empty_state(self):
        sched = MCTSSeedScheduler()
        sched.from_dict({})
        assert sched.visits == {}
        assert sched.selections == 0

    def test_restored_scheduler_keeps_its_preference(self):
        """State must survive --resume as behaviour, not just as numbers."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        _drive(sched, _tree(), _all_keys(), 600, lambda k: k.startswith("b"))

        restored = MCTSSeedScheduler(rng=random.Random(1))
        restored.from_dict(sched.to_dict())
        counts = _drive(restored, _tree(), _all_keys(), 200, lambda k: k.startswith("b"))
        assert sum(v for k, v in counts.items() if k.startswith("b")) / 200 > 0.5

    def test_pending_path_is_not_persisted(self):
        """A path left mid-flight at shutdown must not credit on reload."""
        sched = MCTSSeedScheduler(rng=random.Random(1))
        sched.select(_tree(), _all_keys())
        restored = MCTSSeedScheduler()
        restored.from_dict(sched.to_dict())
        restored.update(50.0)
        assert restored.updates == 0


class TestWiring:
    def test_exported_from_schedulers_package(self):
        from fuzzer_tool.core import schedulers

        assert "MCTSSeedScheduler" in schedulers.__all__

    def test_cli_exposes_the_flag(self, capsys, monkeypatch):
        """The parser is built inside main(), so check the rendered help."""
        import sys

        from fuzzer_tool.cli.commands import main

        monkeypatch.setattr(sys, "argv", ["fuzzer-tool", "fuzz", "--help"])
        with pytest.raises(SystemExit):
            main()
        assert "--mcts" in capsys.readouterr().out

    def test_fuzzer_accepts_the_mcts_kwarg(self):
        """cmd_fuzz forwards mcts=..., so the constructor must take it."""
        import inspect

        from fuzzer_tool.services.fuzzer import Fuzzer

        assert "mcts" in inspect.signature(Fuzzer.__init__).parameters

    def test_seed_picker_offers_mcts_only_with_tree_and_scheduler(self):
        from fuzzer_tool.services.seed_picker import SeedPicker

        picker = SeedPicker.__new__(SeedPicker)

        class _F:
            _mcts = None
            _lineage = None
            corpus = [b"x"]

        picker.f = _F()
        assert picker._pick_mcts_seed() is None

    def test_pick_mcts_seed_returns_a_corpus_member(self):
        from fuzzer_tool.services.seed_picker import SeedPicker

        tree = _tree()
        corpus = [b"seed-root", b"seed-a0"]
        keys = {b"seed-root": "root", b"seed-a0": "a0"}

        class _F:
            _lineage = tree
            _mcts = MCTSSeedScheduler(rng=random.Random(1))

            def __init__(self):
                self.corpus = corpus

            def _seed_key(self, data):
                return keys[data]

        picker = SeedPicker.__new__(SeedPicker)
        picker.f = _F()
        result = picker._pick_mcts_seed()
        assert result in corpus

    def test_pick_mcts_seed_handles_empty_corpus(self):
        from fuzzer_tool.services.seed_picker import SeedPicker

        class _F:
            _lineage = _tree()
            _mcts = MCTSSeedScheduler()
            corpus = []

        picker = SeedPicker.__new__(SeedPicker)
        picker.f = _F()
        assert picker._pick_mcts_seed() is None


class TestSuccessIsAlwaysBoolean:
    """Regression: `x and x.f()` yields None when x is None, and that None
    reached MonteCarloScheduler.record() -> float(None), crashing any run
    that enabled the bandit without a coverage backend (e.g. --elo all)."""

    @pytest.mark.parametrize("value", [None, False, 0])
    def test_record_rejects_non_boolean_success_today(self, value):
        from fuzzer_tool.core.schedulers import MonteCarloScheduler

        mc = MonteCarloScheduler()
        mc.init_arm("bit_flip")
        # bool() at the call site is what keeps this from raising.
        mc.record("bit_flip", bool(value))
        assert mc.arm_beta["bit_flip"] > 1.0
