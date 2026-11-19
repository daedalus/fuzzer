"""Euler-walk seeds: every live ``format_fsm`` transition, fewest messages.

Directed Chinese postman, then Hierholzer. Virtual arcs close every message
into one circuit::

    S* ──0──► start ──label──► ... ──► final ──0──► T* ──M──► S*

Each live transition (on some start -> final path) is required once, and
``T* -> S*`` once (at least one message). Degree imbalance is repaired by
duplicating cheapest paths: a min-cost transportation over Dijkstra
distances, costs in bytes. ``M`` exceeds any byte total, so fewer messages
always wins, then fewer bytes. Hierholzer's circuit, cut at every
``T* -> S*``, yields the walks.

A strongly connected FSM yields one message. A DAG with parallel
transitions cannot fit them in one message; the result then holds several.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from fuzzer_tool.core.format_fsm import FormatFsm, Transition

# Bounds the transportation solve (Bellman-Ford, supplies x demands arcs).
MAX_EDGES = 1024

# Not \w+: cannot collide with a spec state name.
_SRC = "<S*>"
_SNK = "<T*>"


@dataclass(frozen=True)
class _Arc:
    src: str
    dst: str
    cost: int
    edge: Transition | None  # None: virtual


def _closure(roots, adj: dict[str, list[str]]) -> set[str]:
    seen = set(roots)
    stack = list(seen)
    while stack:
        for n in adj.get(stack.pop(), ()):
            if n not in seen:
                seen.add(n)
                stack.append(n)
    return seen


def _live_edges(fsm: FormatFsm) -> list[Transition]:
    """Transitions on some start -> final path; the rest cannot be walked."""
    fwd: dict[str, list[str]] = {}
    back: dict[str, list[str]] = {}
    for t in fsm.edges:
        fwd.setdefault(t.src, []).append(t.dst)
        back.setdefault(t.dst, []).append(t.src)

    reach = _closure(fsm.starts, fwd)
    coreach = _closure(fsm.finals, back)
    return [t for t in fsm.edges if t.src in reach and t.dst in coreach]


def _arcs(fsm: FormatFsm, live: list[Transition]) -> tuple[list[_Arc], _Arc]:
    """Real arcs, virtual entry/exit arcs, and the costly closing arc."""
    states = {t.src for t in live} | {t.dst for t in live}
    arcs = [_Arc(t.src, t.dst, t.label.width, t) for t in live]
    arcs += [_Arc(_SRC, s, 0, None) for s in dict.fromkeys(fsm.starts) if s in states]
    arcs += [_Arc(f, _SNK, 0, None) for f in sorted(fsm.finals) if f in states]

    # Any repair path is simple (<= every width once); at most |E|+1 of them.
    big = (len(live) + 1) * sum(t.label.width for t in live) + 1
    return arcs, _Arc(_SNK, _SRC, big, None)


def _dijkstra(src: str, out: dict[str, list[_Arc]]):
    """Cheapest distance and incoming arc per node from ``src``."""
    dist = {src: 0}
    via: dict[str, _Arc] = {}
    heap = [(0, src)]
    done: set[str] = set()
    while heap:
        d, v = heapq.heappop(heap)
        if v in done:
            continue
        done.add(v)
        for a in out.get(v, ()):
            nd = d + a.cost
            if nd < dist.get(a.dst, nd + 1):
                dist[a.dst] = nd
                via[a.dst] = a
                heapq.heappush(heap, (nd, a.dst))
    return dist, via


def _imbalance(required: list[_Arc]) -> dict[str, int]:
    """in - out per node over the required arcs."""
    b: dict[str, int] = {}
    for a in required:
        b[a.dst] = b.get(a.dst, 0) + 1
        b[a.src] = b.get(a.src, 0) - 1
    return b


def _transport(supply: list[int], demand: list[int], cost: list[list[int]]):
    """Min-cost transportation by successive shortest paths (Bellman-Ford).

    Returns ``flow[i][j]``. Uncapacitated i -> j arcs; balanced totals.
    """
    p, q = len(supply), len(demand)
    flow = [[0] * q for _ in range(p)]
    left_s = list(supply)
    left_d = list(demand)
    while any(left_s):
        # Nodes: supplies 0..p-1, demands p..p+q-1. Forward i -> j always
        # open; backward j -> i open where flow > 0 (cost negated).
        dist = [0 if left_s[i] else None for i in range(p)] + [None] * q
        prev: list[int | None] = [None] * (p + q)
        for _ in range(p + q):
            if not _relax(dist, prev, flow, cost, p, q):
                break

        j = min(
            (k for k in range(q) if left_d[k] and dist[p + k] is not None),
            key=lambda k: dist[p + k],
        )
        _augment(flow, prev, left_s, left_d, p, j)
    return flow


def _relax(dist, prev, flow, cost, p: int, q: int) -> bool:
    """One Bellman-Ford pass over the residual bipartite graph."""
    changed = False
    for i in range(p):
        if dist[i] is None:
            continue
        for k in range(q):
            nd = dist[i] + cost[i][k]
            if dist[p + k] is None or nd < dist[p + k]:
                dist[p + k], prev[p + k] = nd, i
                changed = True
    for k in range(q):
        if dist[p + k] is None:
            continue
        for i in range(p):
            if flow[i][k] <= 0:
                continue
            nd = dist[p + k] - cost[i][k]
            if dist[i] is None or nd < dist[i]:
                dist[i], prev[i] = nd, p + k
                changed = True
    return changed


def _augment(flow, prev, left_s, left_d, p: int, j: int) -> None:
    """Push the bottleneck along the alternating path ending at demand j."""
    path = []
    node = p + j
    while prev[node] is not None:
        path.append((prev[node], node))
        node = prev[node]
    root = node

    amount = min(left_s[root], left_d[j])
    for u, v in path:
        if u < p:  # forward supply -> demand
            continue
        amount = min(amount, flow[v][u - p])

    for u, v in path:
        if u < p:
            flow[u][v - p] += amount
        else:
            flow[v][u - p] -= amount
    left_s[root] -= amount
    left_d[j] -= amount


def _repairs(required: list[_Arc], arcs: list[_Arc], close: _Arc) -> list[_Arc]:
    """Duplicate arcs that balance in == out everywhere, cheapest total."""
    out: dict[str, list[_Arc]] = {}
    for a in (*arcs, close):
        out.setdefault(a.src, []).append(a)

    b = _imbalance(required)
    sup = [v for v in sorted(b) if b[v] > 0]
    dem = [v for v in sorted(b) if b[v] < 0]
    if not sup:
        return []

    paths = [_dijkstra(v, out) for v in sup]
    cost = [[dist[d] for d in dem] for dist, _ in paths]
    flow = _transport([b[v] for v in sup], [-b[v] for v in dem], cost)

    extra: list[_Arc] = []
    for i, (_, via) in enumerate(paths):
        for k, d in enumerate(dem):
            if flow[i][k]:
                extra += _path(via, sup[i], d) * flow[i][k]
    return extra


def _path(via: dict[str, _Arc], src: str, dst: str) -> list[_Arc]:
    """Arcs of the Dijkstra tree path src -> dst."""
    arcs: list[_Arc] = []
    while dst != src:
        a = via[dst]
        arcs.append(a)
        dst = a.src
    return arcs


def _circuit(arcs: list[_Arc]) -> list[_Arc]:
    """Hierholzer, iterative, from S*: every arc exactly once."""
    out: dict[str, list[_Arc]] = {}
    for a in reversed(arcs):  # pop() then takes arcs in input order
        out.setdefault(a.src, []).append(a)

    stack: list[tuple[str, _Arc | None]] = [(_SRC, None)]
    circuit: list[_Arc] = []
    while stack:
        v, a = stack[-1]
        nxt = out.get(v)
        if nxt:
            arc = nxt.pop()
            stack.append((arc.dst, arc))
            continue
        stack.pop()
        if a is not None:
            circuit.append(a)
    circuit.reverse()

    if len(circuit) != len(arcs):
        raise ValueError("euler walk: arc set is not connected")
    return circuit


def euler_walks(fsm: FormatFsm) -> list[list[Transition]]:
    """Fewest start -> final walks covering every live transition, then fewest bytes.

    Raises:
        ValueError: more than ``MAX_EDGES`` transitions.
    """
    if len(fsm.edges) > MAX_EDGES:
        raise ValueError(f"euler walk: {len(fsm.edges)} transitions > {MAX_EDGES}")

    live = _live_edges(fsm)
    if not live:
        return []

    arcs, close = _arcs(fsm, live)
    required = arcs[: len(live)] + [close]
    circuit = _circuit(required + _repairs(required, arcs, close))

    # Cut at every T* -> S*; drop virtual arcs.
    walks: list[list[Transition]] = []
    cur: list[Transition] = []
    for a in circuit:
        if a is close:
            walks.append(cur)
            cur = []
        elif a.edge is not None:
            cur.append(a.edge)
    return walks


def euler_seeds(fsm: FormatFsm, rng) -> list[bytes]:
    """One message per Euler walk; each label sampled from ``rng``."""
    return [b"".join(t.label.sample(rng) for t in w) for w in euler_walks(fsm)]
