"""Tests for the weighted mutation lineage tree (Stage 1: data structure + wiring)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fuzzer_tool.core.lineage import GAMMA, LineageTree
from fuzzer_tool.services.fuzzer import Fuzzer


def _mk_fuzzer(**kwargs):
    tmpdir = tempfile.mkdtemp(prefix="fuzz_lineage_")
    defaults = dict(
        target="/bin/true",
        corpus_dir=f"{tmpdir}/corpus",
        crashes_dir=f"{tmpdir}/crashes",
        max_len=256,
        timeout=1,
        mutations_per_input=2,
    )
    defaults.update(kwargs)
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(**defaults)


class TestLineageTreeInsert:
    def test_insert_root(self):
        tree = LineageTree()
        node = tree.insert(None, "a", [], [], 5)
        assert node.depth == 0
        assert node.parent_key is None
        assert node.node_weight == 5
        assert node.active
        assert len(tree) == 1

    def test_insert_child_records_parent_ops_sites(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        tree.insert("a", "b", ["flip_bits", "havoc"], [3, 7], 2)
        node = tree.get("b")
        assert node.parent_key == "a"
        assert node.depth == 1
        assert tree.op_name(node.child_ops[0]) == "flip_bits"
        assert tree.op_name(node.child_ops[1]) == "havoc"
        assert node.child_sites == [3, 7]

    def test_insert_idempotent(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        first = tree.insert("a", "b", ["bitflip"], [1], 2)
        second = tree.insert("a", "b", ["other"], [9], 99)
        assert first is second
        assert tree.get("b").node_weight == 2
        assert len(tree) == 2

    def test_insert_unknown_parent_creates_orphan_root(self):
        tree = LineageTree()
        node = tree.insert("missing", "b", ["bitflip"], [1], 2)
        assert node.depth == 0
        assert node.parent_key == "missing"
        assert len(tree) == 1

    def test_subtree_weight_propagates_with_gamma(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 10)
        tree.insert("a", "b", ["bitflip"], [1], 5)
        # root gets w(b) * gamma^1
        assert tree.subtree_weight("a") == pytest.approx(10 + 5 * GAMMA)
        assert tree.subtree_weight("b") == pytest.approx(5)

    def test_subtree_weight_discounts_by_depth(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 0)
        tree.insert("a", "b", [], [], 4)
        tree.insert("b", "c", [], [], 8)
        assert tree.subtree_weight("a") == pytest.approx(4 * GAMMA + 8 * GAMMA**2)
        assert tree.subtree_weight("b") == pytest.approx(4 + 8 * GAMMA)

    def test_subtree_weight_missing_key(self):
        assert LineageTree().subtree_weight("nope") == 0.0


class TestLineageTreeAncestry:
    def _chain(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        tree.insert("a", "b", ["bitflip"], [1], 1)
        tree.insert("b", "c", ["havoc"], [2], 1)
        return tree

    def test_ancestors_bottom_up(self):
        tree = self._chain()
        keys = [n.key for n in tree.ancestors("c")]
        assert keys == ["b", "a"]

    def test_ancestors_of_root_empty(self):
        tree = self._chain()
        assert tree.ancestors("a") == []

    def test_ancestors_missing_key_empty(self):
        tree = self._chain()
        assert tree.ancestors("zzz") == []

    def test_lca(self):
        tree = self._chain()
        tree.insert("b", "d", [], [], 1)
        assert tree.lca("c", "d") == "b"
        assert tree.lca("c", "a") == "a"
        assert tree.lca("a", "a") == "a"

    def test_lca_disconnected_returns_none(self):
        tree = self._chain()
        tree.insert(None, "x", [], [], 1)
        assert tree.lca("c", "x") is None

    def test_lca_missing_key_returns_none(self):
        tree = self._chain()
        assert tree.lca("c", "zzz") is None

    def test_lca_distance(self):
        tree = self._chain()
        tree.insert("b", "d", [], [], 1)
        assert tree.lca_distance("c", "d") == 2
        assert tree.lca_distance("c", "a") == 2
        assert tree.lca_distance("c", "c") == 0

    def test_lca_distance_disconnected(self):
        tree = self._chain()
        tree.insert(None, "x", [], [], 1)
        assert tree.lca_distance("c", "x") == -1


class TestLineageTreeCredit:
    def _three_node(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 2)
        tree.insert("a", "b", ["bitflip"], [1], 3)
        tree.insert("a", "c", ["havoc"], [2], 4)
        return tree

    def test_recent_credit_sums_subtree_positive_deltas(self):
        tree = self._three_node()
        cov = {"a": (10, 6), "b": (20, 18), "c": (30, 25)}
        # a: 10-6=4, b: 20-18=2, c: 30-25=5 → total 11
        assert tree.recent_credit("a", lambda k: cov[k]) == pytest.approx(11)

    def test_recent_credit_clamps_negative_delta_to_zero(self):
        tree = self._three_node()
        cov = {"a": (5, 10), "b": (8, 20), "c": (9, 9)}
        # a: max(5-10,0)=0, b: 0, c: 0 → 0
        assert tree.recent_credit("a", lambda k: cov[k]) == 0.0

    def test_recent_credit_missing_node_zero(self):
        tree = LineageTree()
        assert tree.recent_credit("zzz", lambda k: (1, 0)) == 0.0

    def test_recent_credit_leaf_only(self):
        tree = self._three_node()
        cov = {"a": (10, 6), "b": (20, 18), "c": (30, 25)}
        assert tree.recent_credit("b", lambda k: cov[k]) == pytest.approx(2)

    def test_operator_credit_discounted_by_depth(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 0)
        tree.insert("a", "b", ["bitflip"], [1], 4)
        tree.insert("b", "c", ["bitflip"], [2], 8)
        # 4*gamma^1 + 8*gamma^2
        assert tree.operator_credit("bitflip") == pytest.approx(4 * GAMMA + 8 * GAMMA**2)
        assert tree.operator_credit("havoc") == 0.0


class TestLineageTreePrune:
    def test_prune_subtree_soft_deletes(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        tree.insert("a", "b", [], [], 1)
        tree.insert("b", "c", [], [], 1)
        tree.insert("a", "d", [], [], 1)
        assert tree.prune_subtree("b") == 2  # b and c
        assert tree.get("b").active is False
        assert tree.get("c").active is False
        assert tree.get("a").active is True
        assert tree.get("d").active is True

    def test_prune_missing_key_zero(self):
        assert LineageTree().prune_subtree("zzz") == 0

    def test_prune_then_hard_drop_past_max_inactive(self):
        tree = LineageTree(max_inactive=1)
        tree.insert(None, "a", [], [], 1)
        tree.insert("a", "b", [], [], 1)
        tree.insert("b", "c", [], [], 1)
        tree.prune_subtree("b")  # marks b, c inactive
        assert "b" not in tree.nodes  # oldest inactive dropped
        assert "c" in tree.nodes

    def test_prune_keeps_children_pointers(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        tree.insert("a", "b", [], [], 1)
        tree.insert("b", "c", [], [], 1)
        tree.prune_subtree("b")
        assert tree.get("c").parent_key == "b"
        assert tree.lca("c", "c") == "c"

    def test_chain_from_after_prune(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        tree.insert("a", "b", ["bitflip"], [1], 1)
        tree.insert("b", "c", ["havoc"], [2], 1)
        tree.prune_subtree("b")
        chain = tree.chain_from("c")
        assert [k for k, _, _, _, _ in chain] == ["c", "b", "a"]
        assert chain[0][4] is False  # inactive
        assert chain[1][4] is False
        assert chain[2][4] is True


class TestLineageTreeChain:
    def test_chain_from_root_ward(self):
        tree = LineageTree()
        tree.insert(None, "a", [], [], 1)
        tree.insert("a", "b", ["bitflip"], [3], 1)
        tree.insert("b", "c", ["havoc"], [7], 1)
        chain = tree.chain_from("c")
        assert [(k, ops, sites, d) for k, ops, sites, d, _ in chain] == [
            ("c", [tree._op_id("havoc")], [7], 2),
            ("b", [tree._op_id("bitflip")], [3], 1),
            ("a", [], [], 0),
        ]

    def test_chain_stops_at_orphan_parent(self):
        tree = LineageTree()
        tree.insert("missing", "b", ["bitflip"], [1], 1)
        chain = tree.chain_from("b")
        assert len(chain) == 1
        assert chain[0][0] == "b"


class TestLineageTreeRebuild:
    def _meta(self):
        return {
            b"seed_a": {
                "parent_key": None,
                "new_edge_count": 5,
                "lineage_depth": 0,
            },
            b"seed_b": {
                "parent_key": "aaaa1111",
                "parent_ops": ["bitflip"],
                "parent_sites": [3],
                "new_edge_count": 2,
                "lineage_depth": 1,
            },
        }

    def test_rebuild_builds_nodes(self):
        tree = LineageTree()
        n = tree.rebuild_from_meta(self._meta(), lambda b: b.decode())
        assert n == 2
        assert tree.get("seed_a").depth == 0
        b = tree.get("seed_b")
        assert b.depth == 1
        assert b.parent_key == "aaaa1111"
        assert tree.op_name(b.child_ops[0]) == "bitflip"
        assert b.child_sites == [3]

    def test_rebuild_missing_parent_orphan(self):
        meta = {b"seed_b": {"parent_key": "ghost", "new_edge_count": 1}}
        tree = LineageTree()
        tree.rebuild_from_meta(meta, lambda b: b.decode())
        node = tree.get("seed_b")
        assert node.parent_key == "ghost"
        assert node.depth == 0
        assert len(tree.ancestors("seed_b")) == 0

    def test_rebuild_idempotent(self):
        tree = LineageTree()
        tree.rebuild_from_meta(self._meta(), lambda b: b.decode())
        first = {k: (n.depth, n.node_weight, n.parent_key) for k, n in tree.nodes.items()}
        tree.rebuild_from_meta(self._meta(), lambda b: b.decode())
        second = {k: (n.depth, n.node_weight, n.parent_key) for k, n in tree.nodes.items()}
        assert first == second
        assert len(tree) == 2

    def test_rebuild_accumulates_subtree_weights(self):
        meta = {
            b"a": {"parent_key": None, "new_edge_count": 10},
            b"b": {"parent_key": "k-a", "parent_ops": ["bitflip"], "new_edge_count": 4},
        }
        tree = LineageTree()
        tree.rebuild_from_meta(meta, lambda b: "k-" + b.decode())
        assert tree.subtree_weight("k-a") == pytest.approx(10 + 4 * GAMMA)

    def test_rebuild_non_dict_meta_skipped(self):
        tree = LineageTree()
        tree.rebuild_from_meta({b"a": "junk"}, lambda b: b.decode())
        assert len(tree) == 0


class TestLineageOpIntern:
    def test_op_names_roundtrip(self):
        tree = LineageTree()
        tree.insert(None, "a", ["bitflip", "havoc", "bitflip"], [1, 2, 3], 1)
        node = tree.get("a")
        assert [tree.op_name(o) for o in node.child_ops] == ["bitflip", "havoc", "bitflip"]

    def test_op_name_out_of_range(self):
        tree = LineageTree()
        assert tree.op_name(999) == "op:999"


class TestLineageFuzzerWiring:
    def test_lineage_off_disables_tree(self):
        f = _mk_fuzzer()
        assert f._use_lineage is False
        assert f._lineage is None

    def test_lineage_on_creates_tree(self):
        f = _mk_fuzzer(lineage=True)
        assert f._use_lineage is True
        assert isinstance(f._lineage, LineageTree)
        # Rebuilt from seed_meta, which includes the default seed added by
        # _load_corpus — so at least the root exists.
        assert len(f._lineage) >= 1

    def test_lineage_off_mutate_untouched(self):
        f = _mk_fuzzer()
        result = f.mutate(b"AAAA")
        assert isinstance(result, bytes)

    def test_lineage_rebuild_uses_seed_meta(self):
        f = _mk_fuzzer(lineage=True)
        key_a = f._seed_key(b"seed_a")
        f.seed_meta = {
            b"seed_a": {"parent_key": None, "new_edge_count": 3},
            b"seed_b": {
                "parent_key": key_a,
                "parent_ops": ["bitflip"],
                "new_edge_count": 1,
            },
        }
        f._lineage.rebuild_from_meta(f.seed_meta, f._seed_key)
        assert len(f._lineage) == 2
        # f._seed_key produces a 16-hex key for seed_a
        assert f._lineage.get(key_a).depth == 0
        b_node = f._lineage.get(f._seed_key(b"seed_b"))
        assert b_node.depth == 1
        assert b_node.parent_key == key_a


class _FakeShm:
    """Minimal shm_cov stand-in for trim_new_coverage."""

    def __init__(self, edges):
        self.edges = set(edges)

    def get_edge_ids(self):
        return set(self.edges)


class _FakeRunner:
    def run_target(self, data):
        return (0, "")


class TestLineageRecording:
    """Stage 2: ops/sites/new-edge recording into seed_meta."""

    def _mk_recording_fuzzer(self, **kwargs):
        f = _mk_fuzzer(**kwargs)
        # Simulate what mutate() + the `if new:` block leave behind.
        f._last_ops_used = ["bitflip", "havoc"]
        f._last_ops_with_sites = [("bitflip", 3), ("havoc", 7)]
        f._last_new_edge_count = 5
        return f

    def test_save_to_corpus_records_lineage_fields(self):
        f = self._mk_recording_fuzzer(lineage=True)
        parent = b"PARENT" * 8
        child = b"CHILD" * 8
        f.save_to_corpus(child, parent=parent)
        meta = f.seed_meta[child]
        assert meta["parent_key"] == f._seed_key(parent)
        assert meta["parent_ops"] == ["bitflip", "havoc"]
        assert meta["parent_sites"] == [3, 7]
        assert meta["new_edge_count"] == 5
        assert meta["coverage_edges_baseline"] == 0

    def test_save_to_corpus_lineage_off_no_fields(self):
        f = self._mk_recording_fuzzer()
        parent = b"PARENT" * 8
        child = b"CHILD" * 8
        f.save_to_corpus(child, parent=parent)
        meta = f.seed_meta[child]
        assert "parent_key" not in meta
        assert "parent_ops" not in meta
        assert "parent_sites" not in meta
        assert "new_edge_count" not in meta

    def test_save_to_corpus_no_parent_is_root(self):
        f = self._mk_recording_fuzzer(lineage=True)
        seed = b"SEED" * 8
        f.save_to_corpus(seed)
        assert "parent_key" not in f.seed_meta[seed]
        assert "parent_ops" not in f.seed_meta[seed]

    def test_state_roundtrip_preserves_lineage_fields(self):
        import os

        f = self._mk_recording_fuzzer(lineage=True)
        parent = b"PARENT" * 8
        child = b"CHILD" * 8
        f.save_to_corpus(child, parent=parent)
        f._corpus_manager.save_state()

        # Fresh fuzzer on the same corpus dir with resume → load_state path.
        f2 = _mk_fuzzer(
            lineage=True,
            resume=True,
            corpus_dir=f.corpus_dir,
            crashes_dir=str(Path(f.crashes_dir)),
        )
        meta = f2.seed_meta.get(child)
        assert meta is not None
        assert meta["parent_key"] == f._seed_key(parent)
        assert meta["parent_ops"] == ["bitflip", "havoc"]
        assert meta["parent_sites"] == [3, 7]
        assert meta["new_edge_count"] == 5
        assert os.path.exists(f.corpus_dir / "state.pkl.gz")

    def test_state_roundtrip_missing_fields_have_fallbacks(self):
        f = _mk_fuzzer(lineage=True)
        seed = b"PLAIN" * 8
        f.save_to_corpus(seed)
        f._corpus_manager.save_state()
        f2 = _mk_fuzzer(lineage=True, resume=True, corpus_dir=f.corpus_dir)
        meta = f2.seed_meta.get(seed)
        assert meta is not None
        assert meta.get("parent_key") is None
        assert meta.get("parent_ops") == []
        assert meta.get("parent_sites") == []
        assert meta.get("new_edge_count") == 0
        assert meta.get("coverage_edges_baseline") == 0

    def test_trim_new_coverage_carries_lineage(self):
        f = _mk_fuzzer(lineage=True)
        f.shm_cov = _FakeShm({1, 2, 3, 4})
        f._runner = _FakeRunner()
        seed = b"A" * 40
        f.corpus = [seed]
        key = f._seed_key(seed)
        f.seed_meta[seed] = {
            "fuzz_count": 0,
            "coverage_edges": 4,
            "momentum": 0.0,
            "edge_bitmap": bytearray(0),
            "redqueen_offsets": [],
            "added_at": 0,
            "lineage_depth": 2,
            "parent_key": key + "x",  # some upstream parent
            "parent_ops": ["bitflip"],
            "parent_sites": [3],
            "new_edge_count": 4,
            "coverage_edges_baseline": 1,
        }
        f._corpus_manager.trim_new_coverage(seed, seed)
        assert len(f.corpus) == 1
        trimmed = f.corpus[0]
        assert trimmed == seed[:20]
        tm = f.seed_meta[trimmed]
        assert tm["parent_key"] == key + "x"  # original's lineage preserved
        assert tm["parent_ops"] == ["bitflip", "trim"]
        assert tm["parent_sites"] == [3, 20]  # cut at len(seed) // 2
        assert tm["new_edge_count"] == 4
        assert tm["coverage_edges_baseline"] == 1
        assert tm["lineage_depth"] == 3

    def test_trim_new_coverage_lineage_off_no_trim_op(self):
        f = _mk_fuzzer()
        f.shm_cov = _FakeShm({1, 2, 3, 4})
        f._runner = _FakeRunner()
        seed = b"A" * 40
        f.corpus = [seed]
        f.seed_meta[seed] = {
            "fuzz_count": 0,
            "coverage_edges": 4,
            "momentum": 0.0,
            "edge_bitmap": bytearray(0),
            "redqueen_offsets": [],
            "added_at": 0,
            "lineage_depth": 2,
        }
        f._corpus_manager.trim_new_coverage(seed, seed)
        tm = f.seed_meta[f.corpus[0]]
        assert "parent_ops" not in tm
        assert "parent_sites" not in tm


class TestLineageCrashSidecar:
    def test_sidecar_renders_mutation_sites(self):
        from fuzzer_tool.core.crash_metadata import CrashMetadata

        meta = CrashMetadata()
        meta.parent_sites = [3, 7, 9]
        text = meta.format_sidecar()
        assert "mutation_sites: 3, 7, 9" in text

    def test_sidecar_omits_sites_when_empty(self):
        from fuzzer_tool.core.crash_metadata import CrashMetadata

        text = CrashMetadata().format_sidecar()
        assert "mutation_sites" not in text


class TestLineageInsertWiring:
    """Stage 3: fuzz_one inserts lineage nodes when seeds join the corpus."""

    def _mk_fz(self, **kwargs):
        f = _mk_fuzzer(lineage=kwargs.pop("lineage", True), **kwargs)
        f._runner = _FakeRunner()  # not used; _run_target patched
        return f

    def test_fuzz_one_inserts_lineage_node(self):
        f = self._mk_fz()
        seed = f.corpus[0]
        before = len(f._lineage)
        with (
            patch.object(f, "_run_target", return_value=(0, "")),
            patch.object(f, "_is_interesting", return_value=True),
            patch.object(f, "_is_crash", return_value=False),
        ):
            assert f.fuzz_one(seed) is True
        assert len(f._lineage) == before + 1
        # The mutated child descends from the seed with this iteration's ops.
        child_key = f._seed_key(f.corpus[-1])
        node = f._lineage.get(child_key)
        assert node is not None
        assert node.parent_key == f._seed_key(seed)
        assert node.depth == 1
        # Ops recorded on the node match what mutate() applied this iteration.
        applied_ops = [op for op, _ in f._last_ops_with_sites]
        assert [f._lineage.op_name(o) for o in node.child_ops] == applied_ops
        assert node.child_sites == [s for _, s in f._last_ops_with_sites]
        assert node.node_weight == 0  # no coverage in this mocked run

    def test_fuzz_one_lineage_off_no_insert(self):
        f = self._mk_fz(lineage=False)
        seed = f.corpus[0]
        assert f._lineage is None
        with (
            patch.object(f, "_run_target", return_value=(0, "")),
            patch.object(f, "_is_interesting", return_value=True),
            patch.object(f, "_is_crash", return_value=False),
        ):
            assert f.fuzz_one(seed) is True
        # Corpus grew but no lineage tree exists — no crash, no node.
        assert f._use_lineage is False

    def test_record_lineage_insert_duplicate_skipped(self):
        f = self._mk_fz()
        parent = b"PARENT" * 8
        f.save_to_corpus(parent)
        child = b"CHILD" * 8
        f._last_ops_with_sites = [("bitflip", 2)]
        f._last_new_edge_count = 4
        n_before = len(f.corpus)
        # Simulate a duplicate save: corpus length unchanged after save.
        f._record_lineage_insert(child, parent, n_before)
        assert len(f._lineage) == 0 or f._seed_key(child) not in f._lineage

    def test_record_lineage_insert_success_adds_node(self):
        f = self._mk_fz()
        parent = b"PARENT" * 8
        child = b"CHILD" * 8
        f.save_to_corpus(parent)
        f.corpus.append(child)  # simulate successful add
        f._last_ops_with_sites = [("bitflip", 2), ("havoc", 5)]
        f._last_new_edge_count = 7
        n_before = len(f.corpus) - 1
        f._record_lineage_insert(child, parent, n_before)
        node = f._lineage.get(f._seed_key(child))
        assert node is not None
        assert node.parent_key == f._seed_key(parent)
        assert node.node_weight == 7
        assert node.child_sites == [2, 5]

    def test_record_lineage_insert_off_noop(self):
        f = self._mk_fz(lineage=False)
        parent = b"PARENT" * 8
        child = b"CHILD" * 8
        f.corpus.append(child)
        f._record_lineage_insert(child, parent, len(f.corpus) - 1)
        assert f._lineage is None


class TestLineageMinimizePruning:
    """Stage 4: auto-minimize drops unproductive lineage branches."""

    def _setup(self, with_lineage=True):
        from fuzzer_tool.services.corpus_manager import CorpusManager
        from tests.test_corpus_minimization import MockFuzzer

        f = MockFuzzer(Path(tempfile.mkdtemp()))
        f.max_corpus_bytes = 35
        f._use_lineage = with_lineage
        if with_lineage:
            f._lineage = LineageTree()
        f._mgr = CorpusManager(f)
        return f

    def _seed_meta(self, fuzz_count=1, coverage=0, baseline=0):
        return {
            "fuzz_count": fuzz_count,
            "coverage_edges": coverage,
            "coverage_edges_baseline": baseline,
            "momentum": 0.0,
            "edge_bitmap": bytearray(0),
            "redqueen_offsets": [],
            "added_at": 0,
        }

    def test_dropped_parent_removes_kept_child(self):
        f = self._setup()
        parent = b"P" * 30  # low density → dropped by knapsack
        child = b"C" * 10  # small + mild score → kept by knapsack
        f.corpus = [parent, child]
        # Child delta is 0 (coverage == baseline) so the branch credit is 0,
        # and its structural weight 1*gamma < 1.0 → branch is unproductive.
        f.seed_meta = {
            parent: self._seed_meta(coverage=0),
            child: self._seed_meta(coverage=1, baseline=1),
        }
        pk = f._mgr.seed_key(parent)
        f._lineage.insert(None, pk, [], [], 0)
        f._lineage.insert(pk, f._mgr.seed_key(child), ["bitflip"], [1], 1)
        f._mgr.auto_minimize_corpus()
        # Both pruned: parent by score, child by the branch rule.
        assert parent not in f.corpus
        assert child not in f.corpus

    def test_fresh_child_protected_from_branch_drop(self):
        f = self._setup()
        parent = b"P" * 31
        child = b"C" * 10  # fresh → set aside before scoring
        other = b"O" * 5
        f.corpus = [parent, child, other]
        f.seed_meta = {
            parent: self._seed_meta(coverage=0),
            child: self._seed_meta(fuzz_count=0, coverage=0),
            other: self._seed_meta(coverage=1, baseline=1),
        }
        pk = f._mgr.seed_key(parent)
        f._lineage.insert(None, pk, [], [], 0)
        f._lineage.insert(pk, f._mgr.seed_key(child), ["bitflip"], [1], 0)
        f._mgr.auto_minimize_corpus()
        # Knapsack drops parent (31B + 5B > 35); branch rule would drop the
        # child but fresh seeds are protected.
        assert parent not in f.corpus
        assert child in f.corpus

    def test_productive_parent_keeps_subtree(self):
        f = self._setup()
        parent = b"P" * 40  # too big for the 35-byte budget → dropped by size
        child = b"C" * 10
        f.corpus = [parent, child]
        # Parent gained coverage since baseline → credit > 0 → no branch drop.
        f.seed_meta = {
            parent: self._seed_meta(coverage=5, baseline=0),
            child: self._seed_meta(coverage=1, baseline=1),
        }
        pk = f._mgr.seed_key(parent)
        f._lineage.insert(None, pk, [], [], 0)
        f._lineage.insert(pk, f._mgr.seed_key(child), ["bitflip"], [1], 1)
        f._mgr.auto_minimize_corpus()
        # Parent dropped by score; child kept (branch had credit).
        assert parent not in f.corpus
        assert child in f.corpus

    def test_lineage_off_unchanged(self):
        f = self._setup(with_lineage=False)
        parent = b"P" * 30
        child = b"C" * 10
        f.corpus = [parent, child]
        f.seed_meta = {
            parent: self._seed_meta(coverage=0),
            child: self._seed_meta(coverage=1, baseline=1),
        }
        f._mgr.auto_minimize_corpus()
        # No lineage → normal knapsack behavior only: child kept, parent dropped.
        assert parent not in f.corpus
        assert child in f.corpus

    def test_baseline_reset_after_minimize(self):
        f = self._setup()
        a = b"A" * 5
        b = b"B" * 5
        c = b"C" * 5
        f.corpus = [a, b, c]
        f.seed_meta = {
            a: self._seed_meta(coverage=3, baseline=1),
            b: self._seed_meta(coverage=7, baseline=2),
            c: self._seed_meta(coverage=0, baseline=0),
        }
        f.max_corpus_bytes = 0  # count-budget path; nothing drops
        f._mgr.auto_minimize_corpus()
        for seed in (a, b, c):
            m = f.seed_meta[seed]
            assert m["coverage_edges_baseline"] == m["coverage_edges"]


class TestLineageDiversityWeights:
    """Stage 6: LCA-based diversity multiplier in _compute_weights."""

    def _mk(self):
        f = _mk_fuzzer(lineage=True)
        f.corpus = [b"A", b"B", b"C", b"D"]
        now = 1000.0
        f.seed_meta = {
            s: {
                "fuzz_count": 1,
                "coverage_edges": 0,
                "momentum": 0.0,
                "added_at": now,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
            }
            for s in f.corpus
        }
        ka, kb, kc, kd = (f._seed_key(s) for s in f.corpus)
        # Chain A -> B -> C (depth 2) and an isolated root D.
        f._lineage.insert(None, ka, [], [], 0)
        f._lineage.insert(ka, kb, ["bitflip"], [0], 0)
        f._lineage.insert(kb, kc, ["havoc"], [0], 0)
        f._lineage.insert(None, kd, [], [], 0)
        f._cached_weights = {}
        return f

    def _mults(self, f):
        from fuzzer_tool.services.seed_picker import SeedPicker

        sp = SeedPicker(f)
        w_on = sp._compute_weights(1000.0)
        f._use_lineage = False
        w_off = sp._compute_weights(1000.0)
        return {s: on / off for s, on, off in zip(f.corpus, w_on, w_off, strict=True)}

    def test_chain_seeds_boosted_isolated_root_flat(self):
        f = self._mk()
        mults = self._mults(f)
        # Hand-computed multipliers from the tree distances:
        #   A: avg dist 1.5 -> 1 + 0.5 * (1.5/4) = 1.1875
        #   B: avg dist 1.0 -> 1 + 0.5 * (1.0/4) = 1.125
        #   D: no connected peers -> 1.0
        assert mults[b"A"] == pytest.approx(1.1875, rel=1e-3)
        assert mults[b"B"] == pytest.approx(1.125, rel=1e-3)
        assert mults[b"C"] == pytest.approx(1.1875, rel=1e-3)
        assert mults[b"D"] == pytest.approx(1.0, rel=1e-3)

    def test_lineage_off_no_diversity_term(self):
        f = self._mk()
        f._use_lineage = False
        from fuzzer_tool.services.seed_picker import SeedPicker

        sp = SeedPicker(f)
        w1 = sp._compute_weights(1000.0)
        w2 = sp._compute_weights(1000.0)
        assert w1 == w2  # no tree -> deterministic, no multiplier involved

    def test_single_seed_no_crash(self):
        f = _mk_fuzzer(lineage=True)
        f.corpus = [b"ONLY"]
        now = 1000.0
        f.seed_meta = {
            f.corpus[0]: {
                "fuzz_count": 1,
                "coverage_edges": 0,
                "momentum": 0.0,
                "added_at": now,
                "edge_bitmap": bytearray(0),
                "redqueen_offsets": [],
            }
        }
        f._lineage.insert(None, f._seed_key(f.corpus[0]), [], [], 0)
        f._cached_weights = {}
        from fuzzer_tool.services.seed_picker import SeedPicker

        sp = SeedPicker(f)
        weights = sp._compute_weights(now)
        assert len(weights) == 1
        assert weights[0] > 0
