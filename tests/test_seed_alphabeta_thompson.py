"""Tests for the Thompson-descent 'alphabeta' seed arm (P0-1 of the generators handover).

The previous minimax implementation returned only a root: 1 distinct seed in 300
rounds on a 363-node forest, 0 non-root picks, 5-41 ms per select. These tests
pin the behaviours that were broken.
"""

from __future__ import annotations

import random
import time

from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_mcts import (
    AlphaBetaMCTSSeedScheduler,
    MCTSSeedScheduler,
)


def _forest(n_roots=3, fanout=3, depth=4, seed=1):
    r = random.Random(seed)
    tree = LineageTree()
    keys: list[str] = []
    counter = 0

    def mk() -> str:
        nonlocal counter
        counter += 1
        return f"{counter:016x}"

    frontier = []
    for _ in range(n_roots):
        k = mk()
        tree.insert(None, k, ["x"], [0], r.randint(0, 5))
        keys.append(k)
        frontier.append((k, 0))
    while frontier:
        parent, d = frontier.pop()
        if d >= depth:
            continue
        for _ in range(fanout):
            k = mk()
            tree.insert(parent, k, ["x"], [0], r.randint(0, 5))
            keys.append(k)
            frontier.append((k, d + 1))
    return tree, keys


def _sched(seed=0):
    return AlphaBetaMCTSSeedScheduler(rng=RandPool(seed=seed))


class TestSelection:
    def test_returns_non_root_seeds(self):
        tree, keys = _forest()
        roots = set(tree.roots())
        s = _sched()
        r = random.Random(7)
        picked = []
        for _ in range(300):
            k = s.select(tree, set(keys))
            picked.append(k)
            s.update(r.random() * 4)
        assert any(k not in roots for k in picked)

    def test_explores_many_distinct_seeds(self):
        tree, keys = _forest()
        s = _sched()
        r = random.Random(7)
        picked = set()
        for _ in range(300):
            k = s.select(tree, set(keys))
            picked.add(k)
            s.update(r.random() * 4)
        # The minimax version picked exactly one.
        assert len(picked) > 30

    def test_only_returns_eligible_keys(self):
        tree, keys = _forest()
        eligible = set(keys[::2])
        s = _sched()
        for _ in range(200):
            k = s.select(tree, eligible)
            assert k is None or k in eligible
            s.update(1.0)

    def test_empty_eligible_returns_none(self):
        tree, _ = _forest()
        assert _sched().select(tree, set()) is None

    def test_empty_tree_returns_none(self):
        assert _sched().select(LineageTree(), {"deadbeefdeadbeef"}) is None

    def test_can_stop_at_interior_node(self):
        # A productive interior node with barren children must be chosen itself.
        tree = LineageTree()
        tree.insert(None, "root", ["x"], [0], 1)
        tree.insert("root", "a", ["x"], [0], 1)
        tree.insert("root", "b", ["x"], [0], 1)
        s = _sched()
        elig = {"root", "a", "b"}
        for _ in range(400):
            k = s.select(tree, elig)
            s.update(5.0 if k == "root" else 0.0)
        counts = {"root": 0, "a": 0, "b": 0}
        for _ in range(300):
            k = s.select(tree, elig)
            counts[k] += 1
            s.update(5.0 if k == "root" else 0.0)
        assert counts["root"] > counts["a"] + counts["b"]

    def test_passes_through_ineligible_ancestor(self):
        tree = LineageTree()
        tree.insert(None, "gone", ["x"], [0], 1)
        tree.insert("gone", "leaf", ["x"], [0], 1)
        s = _sched()
        seen = {s.select(tree, {"leaf"}) for _ in range(20)}
        assert seen == {"leaf"}


class TestLearning:
    def test_productive_subtree_gains_share(self):
        tree = LineageTree()
        tree.insert(None, "r", ["x"], [0], 1)
        tree.insert("r", "good", ["x"], [0], 1)
        tree.insert("r", "bad", ["x"], [0], 1)
        elig = {"good", "bad"}
        s = _sched(3)
        for _ in range(300):
            k = s.select(tree, elig)
            s.update(6.0 if k == "good" else 0.0)
        tail = []
        for _ in range(200):
            k = s.select(tree, elig)
            tail.append(k)
            s.update(6.0 if k == "good" else 0.0)
        assert tail.count("good") > 150

    def test_barren_subtree_still_gets_some_visits(self):
        # Thompson sampling keeps exploring: the barren arm must not hit zero.
        tree = LineageTree()
        tree.insert(None, "r", ["x"], [0], 1)
        tree.insert("r", "good", ["x"], [0], 1)
        tree.insert("r", "bad", ["x"], [0], 1)
        elig = {"good", "bad"}
        s = _sched(4)
        picks = []
        for _ in range(600):
            k = s.select(tree, elig)
            picks.append(k)
            s.update(6.0 if k == "good" else 0.0)
        assert picks.count("bad") >= 1

    def test_dead_region_records_zero_reward_visit(self):
        tree = LineageTree()
        tree.insert(None, "gone", ["x"], [0], 1)
        s = _sched()
        assert s.select(tree, {"elsewhere"}) is None
        assert s.visits.get("gone", 0.0) == 1.0
        assert s.values.get("gone", 0.0) == 0.0


class TestDeterminism:
    def test_same_seed_same_sequence(self):
        tree, keys = _forest()
        runs = []
        for _ in range(2):
            s = _sched(11)
            r = random.Random(5)
            seq = []
            for _ in range(80):
                k = s.select(tree, set(keys))
                seq.append(k)
                s.update(r.random() * 3)
            runs.append(seq)
        assert runs[0] == runs[1]


class TestCost:
    def test_select_is_cheap_on_a_large_forest(self):
        tree, keys = _forest(n_roots=5, fanout=4, depth=6)  # 27k nodes
        s = _sched()
        elig = set(keys)
        t0 = time.perf_counter()
        for _ in range(200):
            s.select(tree, elig)
            s.update(1.0)
        per = (time.perf_counter() - t0) / 200
        # The minimax version cost ~41 ms here; a descent is a handful of draws.
        assert per < 0.005, f"{per * 1000:.2f} ms/select"

    def test_select_not_much_slower_than_uct(self):
        tree, keys = _forest()
        elig = set(keys)

        def run(s):
            t0 = time.perf_counter()
            for _ in range(200):
                s.select(tree, elig)
                s.update(1.0)
            return time.perf_counter() - t0

        uct = run(MCTSSeedScheduler(rng=RandPool(seed=0)))
        ts = run(_sched())
        assert ts < uct * 30 + 0.05


class TestPersistence:
    def test_roundtrip(self):
        tree, keys = _forest()
        s = _sched()
        for _ in range(50):
            s.select(tree, set(keys))
            s.update(2.0)
        blob = s.to_dict()
        s2 = _sched()
        s2.from_dict(blob)
        assert s2.visits == s.visits and s2.values == s.values
        assert s2.self_visits == s.self_visits
        assert s2.selections == s.selections

    def test_legacy_minimax_blob_is_discarded(self):
        s = _sched()
        s.from_dict({"visits": {"a": 9000.0}, "values": {"a": 8000.0}, "selections": 5, "updates": 5})
        assert s.visits == {} and s.values == {}
        assert s.selections == 5

    def test_prune_drops_dead_keys(self):
        tree, keys = _forest()
        s = _sched()
        for _ in range(50):
            s.select(tree, set(keys))
            s.update(2.0)
        live = set(keys[:5])
        s.prune(live)
        assert set(s.visits) <= live and set(s.self_visits) <= live

    def test_stats_shape(self):
        s = _sched()
        assert set(s.stats()) == {"selections", "updates", "tracked_nodes", "mean_value"}
