"""Linear-time 2-SAT over input bits, plus unsigned-compare clause encoders.

Literals are DIMACS ints: ``+v`` / ``-v`` with ``v`` in ``1..n``. Each clause
``(a | b)`` adds the implications ``~a -> b`` and ``~b -> a``; a unit ``(a)``
adds ``~a -> a``. The formula is unsat iff some ``x`` and ``~x`` share a
strongly connected component::

    (x1|x2)(~x1|x2)(x1|~x2)     ~x2 -> x1 -> x2,  ~x1 -> x2 -> x1
                                x1, x2 forced True

Tarjan (iterative: a 200k-long chain must not hit the recursion limit)
numbers SCCs sinks first, so ``x = True`` iff ``comp(x) < comp(~x)``.

The encoders turn ``x REL c`` over a bit-vector into CNF. ``x >= c`` needs,
for every set bit ``i`` of ``c``, ``x_i`` or a higher bit where ``x`` beats
``c`` (``x_k = 1, c_k = 0``). It is 2-CNF only when ``c`` has at most one
zero above each set bit, e.g. ``x >= 0x40`` -> ``(x6 | x7)``.
"""

from __future__ import annotations

from collections.abc import Sequence

Clause = tuple[int, ...]
MAX_2SAT_WIDTH = 2
_UNSAT: list[Clause] = [()]  # the empty clause


def _node(lit: int) -> int:
    """Graph node of a literal: ``2(v-1)`` for ``+v``, ``2(v-1)+1`` for ``-v``."""
    return ((abs(lit) - 1) << 1) | (lit < 0)


def _check(n_vars: int, clause: Clause) -> None:
    if len(clause) > MAX_2SAT_WIDTH:
        raise ValueError(f"clause wider than {MAX_2SAT_WIDTH}: {clause}")
    for lit in clause:
        if lit == 0 or abs(lit) > n_vars:
            raise ValueError(f"literal {lit} outside 1..{n_vars}")


def _graph(n_vars: int, clauses: Sequence[Clause]) -> list[list[int]] | None:
    """Implication graph as adjacency lists; None on an empty clause."""
    adj: list[list[int]] = [[] for _ in range(n_vars << 1)]
    for clause in clauses:
        _check(n_vars, clause)
        if not clause:
            return None
        a = _node(clause[0])
        b = _node(clause[-1])  # unit: a == b, so ~a -> a
        adj[a ^ 1].append(b)
        if a != b:
            adj[b ^ 1].append(a)
    return adj


def _tarjan(adj: list[list[int]]) -> list[int]:
    """SCC id per node, sinks first. Iterative, O(nodes + edges)."""
    n = len(adj)
    index = [-1] * n
    low = [0] * n
    comp = [-1] * n
    stack: list[int] = []
    counter = 0
    n_comp = 0

    for root in range(n):
        if index[root] != -1:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        work = [(root, 0)]

        while work:
            v, i = work[-1]
            edges = adj[v]

            # Advance v's edge cursor; descend into unvisited children.
            if i < len(edges):
                work[-1] = (v, i + 1)
                w = edges[i]
                if index[w] == -1:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    work.append((w, 0))
                elif comp[w] == -1 and index[w] < low[v]:  # w still on stack
                    low[v] = index[w]
                continue

            # v finished: propagate low to its parent, pop a root's SCC.
            work.pop()
            if work:
                u = work[-1][0]
                if low[v] < low[u]:
                    low[u] = low[v]
            if low[v] != index[v]:
                continue
            while True:
                w = stack.pop()
                comp[w] = n_comp
                if w == v:
                    break
            n_comp += 1

    return comp


def solve_2sat(n_vars: int, clauses: Sequence[Clause]) -> list[bool] | None:
    """Model of a 2-CNF over vars ``1..n_vars`` (index ``v-1``), or None if unsat.

    Raises:
        ValueError: a clause is wider than 2 or a literal is out of range.
    """
    adj = _graph(n_vars, clauses)
    if adj is None:
        return None
    comp = _tarjan(adj)

    model = []
    for v in range(n_vars):
        pos, neg = comp[v << 1], comp[(v << 1) | 1]
        if pos == neg:
            return None
        model.append(pos < neg)
    return model


def _sat_by(model: list[bool], clause: Clause) -> bool:
    return any(model[abs(lit) - 1] == (lit > 0) for lit in clause)


def solve_one_wide(n_vars: int, clauses: Sequence[Clause], wide: Clause) -> list[bool] | None:
    """2-CNF plus one clause of any width, by case split on its literals.

    ``F & (l1 | ... | lk)`` is sat iff ``F & li`` is for some ``i``: at most
    ``k + 1`` linear solves. Example: ``x != original`` over a byte window.
    """
    if not wide:
        return None
    base = solve_2sat(n_vars, clauses)
    if base is None or _sat_by(base, wide):
        return base

    narrow = list(clauses)
    for lit in dict.fromkeys(wide):
        model = solve_2sat(n_vars, narrow + [(lit,)])
        if model is not None:
            return model
    return None


# ── unsigned bit-vector encoders (``bits``: var ids, LSB first) ─────────


def ge_clauses(bits: Sequence[int], c: int) -> list[Clause]:
    """CNF of ``x >= c``: per set bit ``i``, ``x_i`` or a higher zero bit of ``c``."""
    if c <= 0:
        return []
    if c >> len(bits):
        return list(_UNSAT)
    out: list[Clause] = []
    zeros: list[int] = []
    for i in range(len(bits) - 1, -1, -1):
        if c >> i & 1:
            out.append((bits[i], *zeros))
            continue
        zeros.append(bits[i])
    return out


def le_clauses(bits: Sequence[int], c: int) -> list[Clause]:
    """CNF of ``x <= c``: per clear bit ``i``, ``~x_i`` or ``~x_k`` for a higher set bit."""
    if c < 0:
        return list(_UNSAT)
    if c >= (1 << len(bits)) - 1:
        return []
    out: list[Clause] = []
    ones: list[int] = []
    for i in range(len(bits) - 1, -1, -1):
        if c >> i & 1:
            ones.append(-bits[i])
            continue
        out.append((-bits[i], *ones))
    return out


def gt_clauses(bits: Sequence[int], c: int) -> list[Clause]:
    """CNF of ``x > c``."""
    return ge_clauses(bits, c + 1)


def lt_clauses(bits: Sequence[int], c: int) -> list[Clause]:
    """CNF of ``x < c``."""
    if c <= 0:
        return list(_UNSAT)
    return le_clauses(bits, c - 1)


def eq_clauses(bits: Sequence[int], c: int) -> list[Clause]:
    """Units pinning ``x == c``."""
    if c < 0 or c >> len(bits):
        return list(_UNSAT)
    return [(b if c >> i & 1 else -b,) for i, b in enumerate(bits)]


def ne_clause(bits: Sequence[int], c: int) -> Clause:
    """The single clause ``x != c``: some bit differs.

    Raises:
        ValueError: ``c`` does not fit ``bits``.
    """
    if c < 0 or c >> len(bits):
        raise ValueError(f"constant {c:#x} does not fit {len(bits)} bits")
    return tuple(-b if c >> i & 1 else b for i, b in enumerate(bits))
