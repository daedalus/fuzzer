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

    def size(self) -> int:
        """Number of ``_Node`` descendants, including self (iterative).

        Kept iterative — like every other traversal in this module — so it
        stays safe on the pathologically deep trees the regression tests
        exercise (2000+ levels of nesting would blow the recursion limit).
        """
        total = 0
        stack = [self]
        while stack:
            node = stack.pop()
            total += 1
            for child in node.children:
                if isinstance(child, _Node):
                    stack.append(child)
        return total

    def byte_length(self) -> int:
        """Length of ``self.flatten()`` without building the bytes (iterative).

        Mirrors ``flatten()``'s exact accounting (open byte always counted
        if present, close byte only when ``closed``) so callers can budget
        against ``max_len`` before doing any cloning/serialization work.
        """
        total = 0
        stack = [self]
        while stack:
            node = stack.pop()
            if node.open is not None:
                total += 1
                if node.closed:
                    total += 1
            for child in node.children:
                if isinstance(child, bytes):
                    total += len(child)
                else:
                    stack.append(child)
        return total

    def depth(self) -> int:
        """Depth of the deepest ``_Node`` descendant below self (iterative).

        Observability only — not used to cap or reject mutations. Deep
        nesting is a deliberate, valuable fuzzing target (stack-overflow
        bugs in recursive-descent parsers), confirmed by
        ``TestDeepNesting`` in the test suite, so this mutator does not
        enforce a depth ceiling; a caller that wants one (e.g. for
        telemetry or a scheduler-level policy) has the primitive to build
        it on top of.
        """
        max_d = 0
        stack = [(self, 0)]
        while stack:
            node, d = stack.pop()
            for child in node.children:
                if isinstance(child, _Node):
                    if d + 1 > max_d:
                        max_d = d + 1
                    stack.append((child, d + 1))
        return max_d

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


# Bytes that can affect parse structure: any delimiter open byte, plus any
# byte that can close one. Every *other* byte unconditionally takes the
# literal-append path in partial_parse, so runs of them can be sliced in
# bulk instead of appended one at a time.
_INTERESTING = sorted(set(_DELIMITERS) | set(_DELIMITERS.values()))
_INTERESTING_LUT = bytes(1 if b in set(_INTERESTING) else 0 for b in range(256))

try:  # numpy is a hard dependency, but keep the scalar path usable without it
    import numpy as _np

    _INTERESTING_NP = _np.frombuffer(_INTERESTING_LUT, dtype=_np.uint8)
except Exception:  # pragma: no cover
    _np = None
    _INTERESTING_NP = None

# Below this length the numpy round-trip costs more than it saves.
_VECTOR_MIN_LEN = 64


def _interesting_positions(data: bytes) -> list[int]:
    """Offsets of bytes that can affect parse structure.

    One vectorized table lookup instead of a per-byte Python branch. Falls
    back to bytes.find for short inputs, where numpy's fixed overhead
    dominates, and when numpy is unavailable.
    """
    if _np is not None and len(data) >= _VECTOR_MIN_LEN:
        arr = _np.frombuffer(data, dtype=_np.uint8)
        return _np.flatnonzero(_INTERESTING_NP[arr]).tolist()
    out: list[int] = []
    for b in _INTERESTING:
        start = 0
        while True:
            j = data.find(b, start)
            if j < 0:
                break
            out.append(j)
            start = j + 1
    out.sort()
    return out


def partial_parse(data: bytes) -> _Node:
    """Parse *data* into a tree using delimiter matching.

    This is a best-effort parse: if delimiters are unmatched, the
    remaining bytes are appended as a raw tail.  The result always
    flattens back to the original bytes.

    Only 8 byte values can affect the tree structure (the delimiter opens
    and their closes). Rather than branch on every byte, locate those
    positions in one vectorized pass and copy the literal runs between
    them as whole slices — the structural logic below is unchanged and
    still runs one position at a time.
    """
    root = _Node()
    stack = [root]
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

    prev = 0
    for i in _interesting_positions(data):
        if i > prev:
            _append(data[prev:i])  # literal run, copied in bulk
        prev = i + 1

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

    if prev < n:
        _append(data[prev:])  # trailing literal run

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


def _collect_nodes_with_sizes(root: _Node) -> list[tuple[_Node, int]]:
    """Collect every non-root ``_Node`` with its subtree size, in one pass.

    Same candidate set as ``_collect_nodes`` (all descendant ``_Node``s,
    excluding the root), but computes each node's ``size()`` during a
    single post-order traversal instead of calling ``size()`` separately
    per node — O(n) total instead of O(n) per node (which is O(n^2) worst
    case on a deep, thin chain of nested delimiters).
    """
    results: list[tuple[_Node, int]] = []
    sizes: dict[_Node, int] = {}
    stack: list[tuple[_Node, int]] = [(root, 0)]
    while stack:
        node, idx = stack[-1]
        if idx < len(node.children):
            child = node.children[idx]
            stack[-1] = (node, idx + 1)
            if isinstance(child, _Node):
                stack.append((child, 0))
            continue
        stack.pop()
        total = 1
        for child in node.children:
            if isinstance(child, _Node):
                total += sizes[child]
        sizes[node] = total
        if node is not root:
            results.append((node, total))
    return results


def _weighted_index(weights: list[int], rng=None) -> int:
    """Pick an index with probability proportional to ``weights``.

    Uniform selection over all nodes over-samples the many small subtrees
    near the leaves of a Catalan-distributed tree and under-samples the
    few large, structurally interesting subtrees near the root — the same
    bias Koza's genetic-programming literature addresses with a 90/10
    internal/leaf crossover-point split. Weighting by subtree size
    corrects for it directly.
    """
    total = sum(weights)
    if total <= 0:
        idx = rng.randrange(len(weights)) if rng is not None else __import__("random").randrange(len(weights))
        return idx
    r = rng.randrange(total) if rng is not None else __import__("random").randrange(total)
    acc = 0
    for i, w in enumerate(weights):
        acc += w
        if r < acc:
            return i
    return len(weights) - 1


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
    """Delete a random node from the tree, biased toward larger subtrees."""
    pairs = _collect_nodes_with_sizes(root)
    if not pairs:
        return False
    idx = _weighted_index([w for _, w in pairs], rng)
    target = pairs[idx][0]
    _remove_child(root, target)
    return True


def mutate_tree_dup(root: _Node, rng=None) -> bool:
    """Duplicate a random node in-place, biased toward larger subtrees."""
    pairs = _collect_nodes_with_sizes(root)
    if not pairs:
        return False
    idx = _weighted_index([w for _, w in pairs], rng)
    target = pairs[idx][0]
    dup = _clone_node(target)
    _insert_after(root, target, dup)
    return True


def mutate_tree_swap(root: _Node, rng=None) -> bool:
    """Swap two random nodes in the tree, biased toward larger subtrees.

    Delimiter-type-agnostic by design: the two nodes need not share the
    same opening delimiter. Each node carries its own open/close byte, so
    the swap stays round-trip safe either way — this mutator has no
    grammar-rule concept to constrain the pick against (contrast with
    ``TreeMutator._tree_swap`` in grammar.py, which *is* constrained to
    same-rule nodes because it has grammar type information available).
    """
    pairs = _collect_nodes_with_sizes(root)
    n = len(pairs)
    if n < 2:
        return False
    weights = [w for _, w in pairs]
    i = _weighted_index(weights, rng)
    remaining = [k for k in range(n) if k != i]
    j = remaining[_weighted_index([weights[k] for k in remaining], rng)]
    _swap_nodes(root, pairs[i][0], pairs[j][0])
    return True


def mutate_tree_stutter(root: _Node, rng=None, max_len: int | None = None) -> bool:
    """Repeat a random subtree path multiple times, biased toward larger subtrees.

    When *max_len* is given, the repeat count is capped up front against
    the projected flattened size (via the cheap ``byte_length()`` counts)
    instead of building the full duplication and only discovering it was
    oversized after ``flatten()``.
    """
    pairs = _collect_nodes_with_sizes(root)
    if not pairs:
        return False
    idx = _weighted_index([w for _, w in pairs], rng)
    target = pairs[idx][0]
    if rng is not None:
        n_reps = rng.randint(2, 64)
    else:
        import random as _rand

        n_reps = _rand.randint(2, 64)

    if max_len is not None:
        subtree_len = target.byte_length()
        if subtree_len > 0:
            budget = max_len - root.byte_length()
            n_reps = min(n_reps, max(0, budget // subtree_len))
        if n_reps < 1:
            return False

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
        mutated = mutate_tree_stutter(root, rng=rng, max_len=max_len)

    if not mutated:
        return data

    result = root.flatten()
    if len(result) > max_len:
        return data
    return result


# ── Cycle-lemma Dyck path generator ─────────────────────────────────────

# Delimiter pairs eligible for synthesis (quotes excluded: their open/close
# byte is identical, so they don't behave as a U/D step pair under the
# cycle-lemma construction below — a run of quotes just toggles in/out
# rather than nesting).
_GEN_PAIRS: list[tuple[int, int]] = [(o, c) for o, c in _DELIMITERS.items() if o != c]


def has_bracket_delimiter(data: bytes) -> bool:
    """True if *data* contains at least one nesting delimiter byte.

    Checks only ``_GEN_PAIRS``' opening bytes -- ``() [] {}`` -- excluding
    quotes, since a quote run toggles in/out rather than nesting under the
    cycle-lemma construction (see the note on ``_GEN_PAIRS`` above). Used to
    gate the ``tree_generate`` operator: synthesizing a balanced-delimiter
    fragment only makes sense for a seed that already has some.
    """
    return any(bytes([o]) in data for o, _c in _GEN_PAIRS)


def cycle_lemma_dyck_bytes(n_pairs: int, rng=None) -> bytes:
    """Synthesize a uniformly random balanced-delimiter byte string.

    Generates an exact, uniform sample over the ``C_n`` (Catalan-many)
    balanced bracket strings of *n_pairs* pairs, in O(n) time and space,
    via the cycle lemma (Dvoretzky-Motzkin): take a uniformly random
    shuffle of n opens and n closes (as an undifferentiated +1/-1 walk),
    then rotate it to the unique cyclic shift whose running sum never
    dips below zero — found in one linear pass by tracking the position
    of the running minimum. This is cheaper than the recursive
    Catalan-decomposition sampler used to validate the depth ~ sqrt(n)
    claim in docs/handover/handover_trees.md §5 (no big-int Catalan-number
    table, no recursion), at the cost of only producing single-type
    bracket runs unless the caller mixes delimiter kinds itself (each
    opening step below independently draws a random pair from
    ``_GEN_PAIRS``, so kind is not tied to nesting position — see note
    below).

    This does not require ``partial_parse`` at all: the walk *is* the
    tree, so this is a generator, not a mutator, useful for synthesizing
    new nested seeds (JSON/XML/expression-like corpus entries) directly
    from the Dyck-path model rather than by mutating an existing input.

    Args:
        n_pairs: Number of delimiter pairs (>= 0).
        rng: Optional RandPool instance for fast random numbers.

    Returns:
        A balanced-delimiter byte string of length ``2 * n_pairs``, with
        delimiter *kind* (parens/brackets/braces) chosen independently per
        opening step -- so nesting is uniform over Dyck-path shapes, but
        bracket-kind matching within a shape is not itself the Catalan
        object being sampled (a close always matches its own open's kind,
        since we track opens on a stack; the randomness is only in *which*
        kind each open uses).
    """
    if n_pairs <= 0:
        return b""

    def _randrange(n: int) -> int:
        return rng.randrange(n) if rng is not None else __import__("random").randrange(n)

    # Step 1: uniform random shuffle of n (+1) and n (-1) steps
    # (Fisher-Yates over a list of {open, close} markers).
    steps = [1] * n_pairs + [-1] * n_pairs
    total = 2 * n_pairs
    for i in range(total - 1, 0, -1):
        j = _randrange(i + 1)
        steps[i], steps[j] = steps[j], steps[i]

    # Step 2: cycle lemma — find the rotation point via running minimum.
    prefix = 0
    min_val = 0
    min_idx = 0
    for i, s in enumerate(steps):
        prefix += s
        if prefix < min_val:
            min_val = prefix
            min_idx = i + 1
    rotated = steps[min_idx:] + steps[:min_idx]

    # Step 3: emit bytes, tracking an explicit close-stack per open so each
    # close reproduces the kind of the open it matches (round-trip valid),
    # while the *kind chosen* at each open is drawn independently.
    out = bytearray()
    close_stack: list[int] = []
    for s in rotated:
        if s == 1:
            o, c = _GEN_PAIRS[_randrange(len(_GEN_PAIRS))]
            out.append(o)
            close_stack.append(c)
        else:
            out.append(close_stack.pop())
    return bytes(out)


__all__ = [
    "partial_parse",
    "lightweight_tree_mutate",
    "mutate_tree_del",
    "mutate_tree_dup",
    "mutate_tree_swap",
    "mutate_tree_stutter",
    "has_bracket_delimiter",
    "cycle_lemma_dyck_bytes",
]
