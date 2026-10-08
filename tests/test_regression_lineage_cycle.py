"""Regression: seed-lineage parent cycles hung every tree walker.

``rebuild_from_meta`` accepted ``parent_key == own key`` and mutual parent
links; ``_subtree_keys``/``chain_from`` then grew lists forever and
``recent_credit`` recursed without bound. ``trim_new_coverage`` produced the
self-loop: trimming a child down to its parent's bytes overwrote the parent's
meta with the child's ``parent_key`` -- the parent's own key.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fuzzer_tool.core.lineage import LineageTree
from fuzzer_tool.services.fuzzer import Fuzzer

WALK_TIMEOUT_S = 2
LONG_CYCLE = 500


def _key(seed: bytes) -> str:
    return seed.decode()


def _meta(parent: str | None) -> dict:
    return {"parent_key": parent, "new_edge_count": 1}


def _zero_credit(_key: str) -> tuple[int, int]:
    return (0, 0)


def _unit_credit(_key: str) -> tuple[int, int]:
    return (1, 0)


def _ring(n: int) -> dict:
    """n-node parent ring: k0 <- k1 <- ... <- k{n-1} <- k0."""
    return {f"k{i}".encode(): _meta(f"k{(i - 1) % n}") for i in range(n)}


def _assert_walks_bounded(tree: LineageTree, keys: list[str]) -> None:
    total = len(tree)
    for k in keys:
        assert len(tree.subtree_keys(k)) <= total
        assert len(tree.chain_from(k)) <= total
        assert tree.recent_credit(k, _unit_credit) <= total


@pytest.mark.timeout(WALK_TIMEOUT_S)
def test_regression_lineage_self_parent():
    tree = LineageTree()
    tree.rebuild_from_meta({b"a": _meta("a")}, _key)

    assert tree.subtree_keys("a") == ["a"]
    assert [c[0] for c in tree.chain_from("a")] == ["a"]
    assert tree.recent_credit("a", _zero_credit) == 0.0
    assert tree.roots() == ["a"]


@pytest.mark.timeout(WALK_TIMEOUT_S)
def test_regression_lineage_two_cycle():
    tree = LineageTree()
    tree.rebuild_from_meta({b"a": _meta("b"), b"b": _meta("a")}, _key)

    # Exactly one link survives: one root, one child.
    assert len(tree.roots()) == 1
    _assert_walks_bounded(tree, ["a", "b"])
    root = tree.roots()[0]
    assert sorted(tree.subtree_keys(root)) == ["a", "b"]


@pytest.mark.timeout(WALK_TIMEOUT_S)
def test_regression_walkers_survive_corrupt_children():
    """Walkers stay bounded even if _children itself holds a cycle."""
    tree = LineageTree()
    tree.insert(None, "a", [], [], 1)
    tree.insert("a", "b", [], [], 1)
    tree._children.setdefault("b", set()).add("a")

    assert sorted(tree.subtree_keys("a")) == ["a", "b"]
    assert tree.recent_credit("a", _unit_credit) == 2.0


def test_acyclic_tree_exact():
    """Falsification: the cycle guard must not cut a legitimate tree.

    r -> {x, y}, x -> z. Expected sets written by hand.
    """
    meta = {
        b"r": _meta(None),
        b"x": _meta("r"),
        b"y": _meta("r"),
        b"z": _meta("x"),
    }
    tree = LineageTree()
    tree.rebuild_from_meta(meta, _key)

    assert sorted(tree.subtree_keys("r")) == ["r", "x", "y", "z"]
    assert sorted(tree.subtree_keys("x")) == ["x", "z"]
    assert [c[0] for c in tree.chain_from("z")] == ["z", "x", "r"]
    assert tree.recent_credit("r", _unit_credit) == 4.0
    assert tree.roots() == ["r"]
    assert tree.get("z").depth == 2


@pytest.mark.timeout(WALK_TIMEOUT_S)
def test_adversarial_long_cycle_and_orphans():
    """Long ring + tail hanging off it + orphan with a missing parent."""
    meta = _ring(LONG_CYCLE)
    meta[b"tail"] = _meta("k0")
    meta[b"orph"] = _meta("ghost")
    tree = LineageTree()
    tree.rebuild_from_meta(meta, _key)

    ring_roots = [r for r in tree.roots() if r.startswith("k")]
    assert len(ring_roots) == 1
    assert len(tree.subtree_keys(ring_roots[0])) == LONG_CYCLE + 1
    assert "orph" in tree.roots()
    assert tree.get("orph").parent_key == "ghost"
    _assert_walks_bounded(tree, ["k0", f"k{LONG_CYCLE - 1}", "tail", "orph"])


@pytest.fixture
def fuzzer(tmp_path):
    with (
        patch("os.path.isfile", return_value=True),
        patch("os.access", return_value=True),
    ):
        return Fuzzer(
            target="/bin/true",
            corpus_dir=str(tmp_path / "corpus"),
            crashes_dir=str(tmp_path / "crashes"),
            max_len=256,
            timeout=1,
            mutations_per_input=2,
            lineage=True,
        )


class _FakeShm:
    def get_edge_ids(self):
        return {1, 2, 3}


class _FakeRunner:
    def run_target(self, _data):
        return (0, "")


def _seed(f: Fuzzer, data: bytes, parent_key: str | None) -> None:
    f.corpus.append(data)
    f.seed_meta[data] = {
        "fuzz_count": 0,
        "coverage_edges": 3,
        "momentum": 0.0,
        "edge_bitmap": bytearray(0),
        "redqueen_offsets": [],
        "added_at": 0,
        "lineage_depth": 0 if parent_key is None else 1,
        "parent_key": parent_key,
        "parent_ops": [],
        "parent_sites": [],
        "new_edge_count": 1,
    }


def test_regression_trim_to_parent_self_loop(fuzzer):
    """Child trims to exactly its parent's bytes: parent meta must survive."""
    fuzzer.shm_cov = _FakeShm()
    fuzzer._runner = _FakeRunner()
    parent = b"P" * 20
    child = parent + b"C" * 20
    pkey = fuzzer._seed_key(parent)
    _seed(fuzzer, parent, None)
    _seed(fuzzer, child, pkey)

    fuzzer._corpus_manager.trim_new_coverage(child, parent)

    assert fuzzer.seed_meta[parent]["parent_key"] is None
    for seed, meta in fuzzer.seed_meta.items():
        assert meta.get("parent_key") != fuzzer._seed_key(seed)
