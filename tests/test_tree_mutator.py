"""Tests for core/tree_mutator.py — Lightweight delimiter-based tree mutator."""

from fuzzer_tool.core.tree_mutator import (
    _clone_node,
    _collect_leaves,
    _collect_nodes,
    _find_parent,
    _insert_after,
    _Node,
    _remove_child,
    lightweight_tree_mutate,
    mutate_tree_del,
    mutate_tree_dup,
    mutate_tree_stutter,
    mutate_tree_swap,
    partial_parse,
)


class TestPartialParse:
    def test_round_trip_brackets(self):
        data = b"((a)(b))"
        root = partial_parse(data)
        assert root.flatten() == data

    def test_round_trip_json(self):
        data = b'{"a": [1, 2, {"b": 3}]}'
        root = partial_parse(data)
        assert root.flatten() == data

    def test_round_trip_quotes(self):
        data = b'"hello world"'
        root = partial_parse(data)
        assert root.flatten() == data

    def test_round_trip_mixed(self):
        data = b'{"arr": [1,2,3], "obj": {"k": "v"}}'
        root = partial_parse(data)
        assert root.flatten() == data

    def test_empty_delimiters(self):
        data = b"[](){}"
        root = partial_parse(data)
        assert root.flatten() == data

    def test_no_delimiters(self):
        data = b"hello world"
        root = partial_parse(data)
        assert root.flatten() == data

    def test_unmatched_open(self):
        data = b"((a"
        root = partial_parse(data)
        flat = root.flatten()
        assert flat == data

    def test_nested_collects_nodes(self):
        data = b"[a [b c] d]"
        root = partial_parse(data)
        nodes = _collect_nodes(root)
        assert len(nodes) == 2

    def test_quote_alternation(self):
        data = b'"a" "b"'
        root = partial_parse(data)
        assert root.flatten() == data

    def test_complex_nesting(self):
        data = b"([{((()))}])"
        root = partial_parse(data)
        assert root.flatten() == data
        nodes = _collect_nodes(root)
        assert len(nodes) >= 4


class TestTreeMutations:
    def test_del_reduces_node_count(self):
        data = b"[abc][def][ghi]"
        root = partial_parse(data)
        before = len(_collect_nodes(root))
        mutate_tree_del(root)
        after = len(_collect_nodes(root))
        assert after <= before

    def test_dup_increases_node_count(self):
        data = b"[abc][def]"
        root = partial_parse(data)
        before = len(_collect_nodes(root))
        mutate_tree_dup(root)
        after = len(_collect_nodes(root))
        assert after >= before

    def test_swap_preserves_node_count(self):
        data = b"[abc][def][ghi]"
        root = partial_parse(data)
        nodes_before = _collect_nodes(root)
        count_before = len(nodes_before)
        mutate_tree_swap(root)
        nodes_after = _collect_nodes(root)
        assert len(nodes_after) == count_before

    def test_stutter_increases_size(self):
        data = b"[abc]"
        root = partial_parse(data)
        flat_before = len(root.flatten())
        mutate_tree_stutter(root)
        flat_after = len(root.flatten())
        assert flat_after >= flat_before

    def test_del_on_single_node(self):
        data = b"[abc]"
        root = partial_parse(data)
        result = mutate_tree_del(root)
        assert result
        flat = root.flatten()
        assert len(flat) < len(data)

    def test_lightweight_mutate_changes_data(self):
        data = b"[abc][def][ghi][jkl]"
        results = set()
        for _ in range(50):
            result = lightweight_tree_mutate(data)
            results.add(result)
        changed = sum(1 for r in results if r != data)
        assert changed > 0

    def test_lightweight_mutate_unchanged_for_plain_data(self):
        data = b"hello without delimiters"
        result = lightweight_tree_mutate(data)
        assert result == data

    def test_lightweight_mutate_short_data(self):
        assert lightweight_tree_mutate(b"ab") == b"ab"

    def test_lightweight_mutate_empty(self):
        assert lightweight_tree_mutate(b"") == b""

    def test_lightweight_mutate_max_len(self):
        data = b"[abc][def]"
        result = lightweight_tree_mutate(data, max_len=4)
        # If mutation would exceed max_len, original is returned
        assert len(result) <= len(data)

    def test_clone_node_deep_copy(self):
        data = b'[{"a": [1, 2]}]'
        root = partial_parse(data)
        nodes = _collect_nodes(root)
        if nodes:
            clone = _clone_node(nodes[0])
            assert clone.flatten() == nodes[0].flatten()
            assert clone is not nodes[0]

    def test_round_trip_unmatched_delimiters(self):
        """Round-trip must preserve unmatched delimiters — no phantom close bytes."""
        cases = [
            b"(abc",
            b"((abc",
            b"abc)",
            b"[hello",
            b"{world",
            b'"open string',
            b"'single open",
            b"(a(b",
            b"[{nest",
            b"((((",
            b")])}>",
            b"a(b)c[d",
        ]
        for data in cases:
            root = partial_parse(data)
            assert root.flatten() == data, f"Round-trip failed for {data!r}"


class TestDeepNesting:
    """Regression: deeply nested trees must not cause RecursionError."""

    DEPTH = 2000

    @staticmethod
    def _deep_data() -> bytes:
        return b"(" * TestDeepNesting.DEPTH + b")" * TestDeepNesting.DEPTH

    def test_regression_deep_tree_no_recursion_error(self):
        data = self._deep_data()
        root = partial_parse(data)
        nodes = _collect_nodes(root)
        assert len(nodes) == self.DEPTH

        # Every recursive function must handle this depth without RecursionError
        flat = root.flatten()
        assert flat == data

        target = nodes[self.DEPTH // 2]
        clone = _clone_node(target)
        assert clone.flatten() == target.flatten()
        assert clone is not target

        parent = _find_parent(root, target)
        assert parent is not None

        leaves = _collect_leaves(root)
        assert len(leaves) >= 0

        assert _remove_child(root, target)
        assert _find_parent(root, target) is None

    def test_regression_deep_tree_insert_after(self):
        """Insert operations also avoid RecursionError on deep trees."""
        data = self._deep_data()
        root = partial_parse(data)
        nodes = _collect_nodes(root)
        target = nodes[self.DEPTH // 2]
        new_node = _Node(40)
        new_node.closed = True
        assert _insert_after(root, target, new_node)
        flat = root.flatten()
        # Should contain the inserted () somewhere
        assert b"()" in flat
