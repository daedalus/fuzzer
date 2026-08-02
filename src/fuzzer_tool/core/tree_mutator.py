"""Lightweight delimiter-based tree mutator.

Port of Radamsa's ``sed-tree-*`` operators (radamsa/rad/mutations.scm).
Performs a partial parse using common delimiter pairs ``() {} [] \"\" '' <>``,
builds a tree of nested nodes, mutates the tree, and flattens back to bytes.

Unlike ``grammar.py``, this requires no grammar definition — it heuristically
detects structure from delimiter usage alone.
"""

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

# Fast lookup table: byte -> closing byte, or 0xFF if not a delimiter.
_DELIM_CLOSE = bytes(_DELIMITERS.get(b, 0xFF) for b in range(256))

# Pre-computed single-byte objects: _BYTE_BYTES[byte] = bytes([byte]).
_BYTE_BYTES = [bytes([b]) for b in range(256)]

# Pre-computed close-byte lookup: _CLOSE_TABLE[open_byte] = close_byte, or 0xFF.
_CLOSE_TABLE = bytes(_DELIMITERS.get(b, 0xFF) for b in range(256))

# Pre-allocated op tuple — avoids allocation on every call
_TREE_OPS = ("del", "dup", "swap", "stutter")


def _find_delim(byte: int) -> int | None:
    """Return the matching close delimiter for *byte*, or None."""
    c = _DELIM_CLOSE[byte]
    return c if c != 0xFF else None


def _is_delim(byte: int) -> bool:
    return _DELIM_CLOSE[byte] != 0xFF


# ── Parse tree types ──────────────────────────────────────────────────


class _Node:
    """A parsed node: either raw bytes or a delimited tree node."""

    __slots__ = ("open", "closed", "children")

    def __init__(self, open_byte: int | None = None):
        self.open: int | None = open_byte
        self.closed: bool = False
        self.children: list[_Node | bytes] = []

    def is_leaf(self) -> bool:
        return not self.children

    def flatten(self) -> bytes:
        """Flatten the tree back to raw bytes (iterative stack).

        Stack entries are either:
          - ``(node, 0)``: enter node (append open delim, push children + close)
          - ``(node, 1)``: exit node (append close delim)
          - ``bytes``: raw literal to append verbatim
        """
        parts: list[bytes] = []
        stack: list[tuple[_Node, int] | bytes] = [(self, 0)]
        while stack:
            item = stack.pop()
            if isinstance(item, bytes):
                parts.append(item)
                continue
            node, state = item
            if state == 1:
                if node.open is not None and node.closed:
                    close = _CLOSE_TABLE[node.open]
                    if close != 0xFF:
                        parts.append(_BYTE_BYTES[close])
                continue
            # Enter — push close marker, children (reversed), then open delim
            stack.append((node, 1))
            for child in reversed(node.children):
                if isinstance(child, _Node):
                    stack.append((child, 0))
                else:
                    stack.append(child)
            if node.open is not None:
                parts.append(_BYTE_BYTES[node.open])
        return b"".join(parts)


# ── Parser ────────────────────────────────────────────────────────────


def partial_parse(data: bytes) -> _Node:
    """Parse *data* into a tree using delimiter matching.

    This is a best-effort parse: if delimiters are unmatched, the
    remaining bytes are appended as a raw tail.  The result is always
    a valid tree that flattens back to the original bytes.
    """
    root = _Node()
    stack = [root]
    i = 0
    buf: list[bytes] = []

    def flush():
        if buf:
            chunk = b"".join(buf)
            if stack:
                stack[-1].children.append(chunk)
            buf.clear()

    delim_close = _DELIM_CLOSE
    close_table = _CLOSE_TABLE
    _append = buf.append
    _NodeCls = _Node
    n = len(data)

    while i < n:
        byte = data[i]
        close = delim_close[byte]
        if close != 0xFF:
            if byte == close:
                if stack and stack[-1].open == byte:
                    flush()
                    if len(stack) > 1:
                        stack[-1].closed = True
                        stack.pop()
                else:
                    flush()
                    node = _NodeCls(byte)
                    stack[-1].children.append(node)
                    stack.append(node)
            else:
                flush()
                node = _NodeCls(byte)
                stack[-1].children.append(node)
                stack.append(node)
        elif stack:
            top = stack[-1]
            if top.open is not None and byte == close_table[top.open]:
                flush()
                if len(stack) > 1:
                    top.closed = True
                    stack.pop()
            else:
                _append(_BYTE_BYTES[byte])
        else:
            _append(_BYTE_BYTES[byte])
        i += 1

    flush()
    return root


# ── Mutations ─────────────────────────────────────────────────────────


def _collect_nodes(node: _Node) -> list[_Node]:
    """Return all delimited nodes in the tree (depth-first, iterative)."""
    nodes = []
    stack = [node]
    while stack:
        current = stack.pop()
        for child in current.children:
            if isinstance(child, _Node):
                nodes.append(child)
                stack.append(child)
    return nodes


def _collect_leaves(node: _Node) -> list[_Node | bytes]:
    """Return all leaf children (iterative depth-first)."""
    leaves: list[_Node | bytes] = []
    stack = [node]
    while stack:
        current = stack.pop()
        for child in reversed(current.children):
            if isinstance(child, _Node):
                if child.is_leaf():
                    leaves.append(child)
                else:
                    stack.append(child)
            else:
                leaves.append(child)
    return leaves


def mutate_tree_del(root: _Node, rng=None) -> bool:
    """Delete a random node from the tree."""
    nodes = _collect_nodes(root)
    n = len(nodes)
    if n < 1:
        return False
    idx = rng.randrange(n) if rng is not None else __import__("random").randrange(n)
    target = nodes[idx]
    _remove_child(root, target)
    return True


def mutate_tree_dup(root: _Node, rng=None) -> bool:
    """Duplicate a random node in-place."""
    nodes = _collect_nodes(root)
    n = len(nodes)
    if n < 1:
        return False
    idx = rng.randrange(n) if rng is not None else __import__("random").randrange(n)
    target = nodes[idx]
    dup = _clone_node(target)
    _insert_after(root, target, dup)
    return True


def mutate_tree_swap(root: _Node, rng=None) -> bool:
    """Swap two random nodes in the tree."""
    nodes = _collect_nodes(root)
    n = len(nodes)
    if n < 2:
        return False
    if rng is not None:
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
    else:
        import random as _rand

        i = _rand.randrange(n)
        j = _rand.randrange(n - 1)
        if j >= i:
            j += 1
    _swap_nodes(root, nodes[i], nodes[j])
    return True


def mutate_tree_stutter(root: _Node, rng=None) -> bool:
    """Repeat a random subtree path multiple times."""
    nodes = _collect_nodes(root)
    n = len(nodes)
    if n < 1:
        return False
    if rng is not None:
        target = nodes[rng.randrange(n)]
        n_reps = rng.randint(2, 64)
    else:
        import random as _rand

        target = nodes[_rand.randrange(n)]
        n_reps = _rand.randint(2, 64)
    clone = _clone_node(target)
    for _ in range(n_reps):
        _insert_after(root, target, _clone_node(clone))
    return True


# ── Tree editing helpers ──────────────────────────────────────────────


def _remove_child(root: _Node, target: _Node) -> bool:
    """Remove *target* from its parent's children (iterative)."""
    stack = [root]
    while stack:
        current = stack.pop()
        for child in current.children:
            if child is target:
                current.children.remove(target)
                return True
            if isinstance(child, _Node):
                stack.append(child)
    return False


def _insert_after(root: _Node, target: _Node, new_node: _Node) -> bool:
    """Insert *new_node* after *target* in the tree (iterative)."""
    stack = [root]
    while stack:
        current = stack.pop()
        for i, child in enumerate(current.children):
            if child is target:
                current.children.insert(i + 1, new_node)
                return True
            if isinstance(child, _Node):
                stack.append(child)
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
    """Find the parent of *target* in the tree (iterative)."""
    stack = [root]
    while stack:
        current = stack.pop()
        for child in current.children:
            if child is target:
                return current
            if isinstance(child, _Node):
                stack.append(child)
    return None


def _clone_node(node: _Node) -> _Node:
    """Deep-copy a node (iterative stack)."""
    new_root = _Node(node.open)
    new_root.closed = node.closed
    stack: list[tuple[_Node, _Node, int]] = [(node, new_root, 0)]
    while stack:
        orig, new, idx = stack[-1]
        if idx >= len(orig.children):
            stack.pop()
            continue
        child = orig.children[idx]
        stack[-1] = (orig, new, idx + 1)
        if isinstance(child, _Node):
            new_child = _Node(child.open)
            new_child.closed = child.closed
            new.children.append(new_child)
            stack.append((child, new_child, 0))
        else:
            new.children.append(child)
    return new_root


# ── Public API ────────────────────────────────────────────────────────


def lightweight_tree_mutate(data: bytes, max_len: int = 65536, rng=None) -> bytes:
    """Apply a random tree mutation to *data* using Radamsa's heuristic.

    Args:
        data: Input bytes.
        max_len: Maximum output length.
        rng: Optional RandPool instance for fast random numbers.

    Returns:
        Mutated bytes, or original input if too short or mutation failed.
    """
    if len(data) < 4:
        return data

    root = partial_parse(data)
    nodes = _collect_nodes(root)

    n = len(nodes)
    if n < 1:
        return data

    # Choose a random mutation
    if rng is not None:
        op = _TREE_OPS[rng.randrange(4)]
    else:
        import random as _rand

        op = _TREE_OPS[_rand.randrange(4)]

    mutated = False
    if op == "del":
        mutated = mutate_tree_del(root, rng=rng)
    elif op == "dup":
        mutated = mutate_tree_dup(root, rng=rng)
    elif op == "swap":
        mutated = mutate_tree_swap(root, rng=rng)
    elif op == "stutter":
        mutated = mutate_tree_stutter(root, rng=rng)

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
