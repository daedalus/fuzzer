"""Lightweight delimiter-based tree mutator.

Port of Radamsa's ``sed-tree-*`` operators (radamsa/rad/mutations.scm).
Performs a partial parse using common delimiter pairs ``() {} [] \"\" '' <>``,
builds a tree of nested nodes, mutates the tree, and flattens back to bytes.

Unlike ``grammar.py``, this requires no grammar definition — it heuristically
detects structure from delimiter usage alone.
"""

import random

# ── Delimiter pairs ───────────────────────────────────────────────────

# Maps opening byte -> closing byte
_DELIMITERS: dict[int, int] = {
    40: 41,  # ()
    91: 93,  # []
    123: 125,  # {}
    34: 34,  # ""
    39: 39,  # ''
}
# Note: <> is deliberately excluded — it's too aggressive in ordinary text
# and XML/HTML is handled separately by the grammar-based mutations.


def _find_delim(byte: int) -> int | None:
    """Return the matching close delimiter for *byte*, or None."""
    return _DELIMITERS.get(byte)


def _is_delim(byte: int) -> bool:
    return byte in _DELIMITERS


# ── Parse tree types ──────────────────────────────────────────────────


class _Node:
    """A parsed node: either raw bytes or a delimited tree node."""

    def __init__(self, open_byte: int | None = None):
        self.open: int | None = open_byte  # opening delimiter byte (None for root/raw)
        self.closed: bool = False  # True only when parser matched a close byte
        self.children: list[_Node | bytes] = []  # child nodes or raw byte chunks

    def is_leaf(self) -> bool:
        return not self.children

    def flatten(self) -> bytes:
        """Flatten the tree back to raw bytes."""
        parts: list[bytes] = []
        if self.open is not None:
            parts.append(bytes([self.open]))
        for child in self.children:
            if isinstance(child, _Node):
                parts.append(child.flatten())
            else:
                parts.append(child)
        if self.open is not None and self.closed:
            close = _find_delim(self.open)
            if close is not None:
                parts.append(bytes([close]))
        return b"".join(parts)


# ── Parser ────────────────────────────────────────────────────────────


def partial_parse(data: bytes) -> _Node:
    """Parse *data* into a tree using delimiter matching.

    This is a best-effort parse: if delimiters are unmatched, the
    remaining bytes are appended as a raw tail.  The result is always
    a valid tree that flattens back to the original bytes.
    """
    root = _Node()
    stack = [root]  # current node stack
    i = 0
    buf: list[bytes] = []  # raw byte accumulator

    def flush():
        if buf:
            chunk = b"".join(buf)
            if stack:
                stack[-1].children.append(chunk)
            buf.clear()

    while i < len(data):
        byte = data[i]
        close = _find_delim(byte)
        if close is not None:
            if byte == close:
                # Self-matching delimiter ("", ''): alternate open/close
                if stack and stack[-1].open == byte:
                    # Closing
                    flush()
                    if len(stack) > 1:
                        stack[-1].closed = True
                        stack.pop()
                else:
                    # Opening
                    flush()
                    node = _Node(byte)
                    stack[-1].children.append(node)
                    stack.append(node)
            else:
                # Non-self-matching delimiter: always open
                flush()
                node = _Node(byte)
                stack[-1].children.append(node)
                stack.append(node)
        elif stack and stack[-1].open is not None and byte == _find_delim(stack[-1].open):
            # Closing delimiter for non-self-matching (open != close)
            flush()
            if len(stack) > 1:
                stack[-1].closed = True
                stack.pop()
        else:
            buf.append(bytes([byte]))
        i += 1

    flush()
    return root


# ── Mutations ─────────────────────────────────────────────────────────


def _collect_nodes(node: _Node) -> list[_Node]:
    """Return all delimited nodes in the tree (depth-first)."""
    nodes = []
    for child in node.children:
        if isinstance(child, _Node):
            nodes.append(child)
            nodes.extend(_collect_nodes(child))
    return nodes


def _collect_leaves(node: _Node) -> list[_Node | bytes]:
    """Return all leaf children (either _Node or raw bytes)."""
    leaves = []
    for child in node.children:
        if isinstance(child, _Node):
            if child.is_leaf():
                leaves.append(child)
            else:
                leaves.extend(_collect_leaves(child))
        else:
            leaves.append(child)
    return leaves


def mutate_tree_del(root: _Node) -> bool:
    """Delete a random node from the tree."""
    nodes = _collect_nodes(root)
    if len(nodes) < 1:
        return False
    target = random.choice(nodes)
    # Find parent and remove
    _remove_child(root, target)
    return True


def mutate_tree_dup(root: _Node) -> bool:
    """Duplicate a random node in-place."""
    nodes = _collect_nodes(root)
    if len(nodes) < 1:
        return False
    target = random.choice(nodes)
    # Find parent and insert copy after target
    dup = _clone_node(target)
    _insert_after(root, target, dup)
    return True


def mutate_tree_swap(root: _Node) -> bool:
    """Swap two random nodes in the tree."""
    nodes = _collect_nodes(root)
    if len(nodes) < 2:
        return False
    a, b = random.sample(nodes, 2)
    _swap_nodes(root, a, b)
    return True


def mutate_tree_stutter(root: _Node) -> bool:
    """Repeat a random subtree path multiple times."""
    nodes = _collect_nodes(root)
    if len(nodes) < 1:
        return False
    target = random.choice(nodes)
    n_reps = random.randint(2, 64)
    clone = _clone_node(target)
    # Insert multiple copies
    for _ in range(n_reps):
        _insert_after(root, target, _clone_node(clone))
    return True


# ── Tree editing helpers ──────────────────────────────────────────────


def _remove_child(root: _Node, target: _Node) -> bool:
    """Remove *target* from its parent's children."""
    for child in root.children:
        if child is target:
            root.children.remove(target)
            return True
        if isinstance(child, _Node):
            if _remove_child(child, target):
                return True
    return False


def _insert_after(root: _Node, target: _Node, new_node: _Node) -> bool:
    """Insert *new_node* after *target* in the tree."""
    for i, child in enumerate(root.children):
        if child is target:
            root.children.insert(i + 1, new_node)
            return True
        if isinstance(child, _Node):
            if _insert_after(child, target, new_node):
                return True
    return False


def _swap_nodes(root: _Node, a: _Node, b: _Node) -> bool:
    """Swap positions of nodes *a* and *b* in the tree."""
    parent_a = _find_parent(root, a)
    parent_b = _find_parent(root, b)
    if parent_a is None or parent_b is None:
        return False
    ia = parent_a.children.index(a)
    ib = parent_b.children.index(b)
    parent_a.children[ia] = b
    parent_b.children[ib] = a
    return True


def _find_parent(root: _Node, target: _Node) -> _Node | None:
    """Find the parent of *target* in the tree."""
    for child in root.children:
        if child is target:
            return root
        if isinstance(child, _Node):
            result = _find_parent(child, target)
            if result is not None:
                return result
    return None


def _clone_node(node: _Node) -> _Node:
    """Deep-copy a node."""
    new = _Node(node.open)
    new.closed = node.closed
    for child in node.children:
        if isinstance(child, _Node):
            new.children.append(_clone_node(child))
        else:
            new.children.append(child)
    return new


# ── Public API ────────────────────────────────────────────────────────


def lightweight_tree_mutate(data: bytes, max_len: int = 65536) -> bytes:
    """Apply a random tree mutation to *data* using Radamsa's heuristic.

    Args:
        data: Input bytes.
        max_len: Maximum output length.

    Returns:
        Mutated bytes, or original input if too short or mutation failed.
    """
    if len(data) < 4:
        return data

    root = partial_parse(data)
    nodes = _collect_nodes(root)

    # If no delimited nodes found, return data unchanged
    if len(nodes) < 1:
        return data

    # Choose a random mutation
    op = random.choice(["del", "dup", "swap", "stutter"])

    mutated = False
    if op == "del":
        mutated = mutate_tree_del(root)
    elif op == "dup":
        mutated = mutate_tree_dup(root)
    elif op == "swap":
        mutated = mutate_tree_swap(root)
    elif op == "stutter":
        mutated = mutate_tree_stutter(root)

    if not mutated:
        return data

    result = root.flatten()
    if len(result) > max_len:
        return data
    return result


__all__ = [
    "partial_parse",
    "lightweight_tree_mutate",
    "mutate_tree_del",
    "mutate_tree_dup",
    "mutate_tree_swap",
    "mutate_tree_stutter",
]
