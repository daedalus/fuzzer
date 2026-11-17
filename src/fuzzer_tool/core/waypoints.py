"""Waypoint race for directed fuzzing (soksolv ``hubstrata.go`` / ``hubrace.go``).

Waypoints are the target's dominator chain on the ICFG from ``main``'s entry
(``waypoint_stages``): every input reaching the target through a modelled
edge passes each one in order. Indirect calls have no ICFG edge, so a chain
can name a block an indirect path skips: waypoints steer energy, they prove
nothing. A seed's stratum
is the deepest waypoint it reached (the coarse signature); the shim marks
waypoints in a tiny node bitmap (one bit per stage).

One stratum at a time runs a trial: its seeds get an energy multiplier that
starts at ``BUDGET_START`` and doubles (up to ``BUDGET_CAP``) after every
window of ``WINDOW_EXECS`` execs whose minimum distance sets a new record.
A window without a record ends the trial, the budget resets and the next
stratum is picked:

    never-tried strata first, deepest first
    else max (prog + eps) / (tries + 1), x2 if it got a new seed since,
    ties to the deeper stratum

A failed stratum is only deprioritized (its tries grow), never pruned.
"""

import math

import numpy as np

from fuzzer_tool.core.horizon import _gather, _out_csr

# No waypoint reached.
NO_STRATUM = -1

# Execs per progress window.
WINDOW_EXECS = 256

# Energy multiplier of the active stratum at trial start, and its ceiling.
BUDGET_START = 2.0
BUDGET_CAP = 8.0

# Waypoints kept, nearest the target (soksolv strataMax).
MAX_STAGES = 64

# Score floor for a stratum with no progress yet (soksolv: prog + 0.01).
_PROG_EPS = 0.01

# Score boost for a tried stratum holding a seed newer than its last trial.
_FRESH_BOOST = 2.0


class WaypointRace:
    """Progress-gated budget doubling over waypoint strata."""

    def __init__(self, n_stages: int, window: int = WINDOW_EXECS):
        """
        Raises:
            ValueError: *n_stages* < 1.
        """
        if n_stages < 1:
            raise ValueError("need at least one waypoint stage")
        self._n = n_stages
        self._lim = (1 << n_stages) - 1  # stage bits of a mask
        self._window = window
        self._tries = [0] * n_stages
        self._prog = [0] * n_stages
        self._seen = [False] * n_stages
        self._fresh = [False] * n_stages
        self._best = math.inf  # record min distance over trial windows
        self._left = window
        self._win_min = math.inf
        self.active = NO_STRATUM
        self._budget = BUDGET_START

    def stratum_of(self, mask: bytes) -> int:
        """Deepest stage bit set in *mask*; NO_STRATUM when none."""
        # Highest set bit below n: bit_length beat a per-bit scan 25x (empty mask).
        return (int.from_bytes(mask, "little") & self._lim).bit_length() - 1

    def admit(self, stratum: int) -> None:
        """A corpus seed reached *stratum*."""
        if not 0 <= stratum < self._n:
            return
        self._seen[stratum] = True
        if self._tries[stratum]:
            self._fresh[stratum] = True
        if self.active == NO_STRATUM:
            self._start_trial()

    def multiplier(self, stratum: int) -> float:
        """Energy factor for a seed of *stratum*."""
        if stratum == self.active and stratum != NO_STRATUM:
            return self._budget
        return 1.0

    def note(self, distance: float | None) -> None:
        """One exec; *distance* None when this exec measured none."""
        if distance is not None and distance < self._win_min:
            self._win_min = distance
        self._left -= 1
        if self._left:
            return
        self._close_window()

    def _close_window(self) -> None:
        self._left = self._window
        win_min, self._win_min = self._win_min, math.inf
        a = self.active
        if a == NO_STRATUM:
            return

        # New record: the trial goes on with double budget.
        if win_min < self._best:
            self._best = win_min
            self._prog[a] += 1
            self._budget = min(self._budget * 2, BUDGET_CAP)
            return

        # No progress: trial failed, next stratum.
        self._tries[a] += 1
        self._start_trial()

    def _start_trial(self) -> None:
        self._budget = BUDGET_START
        self.active = self._pick()
        if self.active != NO_STRATUM:
            self._fresh[self.active] = False

    def _pick(self) -> int:
        seen = [k for k in range(self._n) if self._seen[k]]
        if not seen:
            return NO_STRATUM
        untried = [k for k in seen if not self._tries[k]]
        if untried:
            return untried[-1]
        return max(seen, key=lambda k: (self._score(k), k))

    def _score(self, k: int) -> float:
        sc = (self._prog[k] + _PROG_EPS) / (self._tries[k] + 1)
        return sc * _FRESH_BOOST if self._fresh[k] else sc

    def state_dict(self) -> dict:
        """Learned per-stratum record; the running trial is not kept."""
        return {
            "n": self._n,
            "tries": list(self._tries),
            "prog": list(self._prog),
            "seen": list(self._seen),
            "best": self._best,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore :meth:`state_dict`; ignored for a different chain length."""
        if state.get("n") != self._n:
            return
        self._tries = [int(x) for x in state["tries"]]
        self._prog = [int(x) for x in state["prog"]]
        self._seen = [bool(x) for x in state["seen"]]
        self._best = float(state["best"])
        self._start_trial()


def _reach(ptr: np.ndarray, nbr: np.ndarray, start: np.ndarray, n: int) -> np.ndarray:
    """Bool mask of nodes reachable from *start* (level-synchronous BFS)."""
    seen = np.zeros(n, dtype=bool)
    seen[start] = True
    frontier = np.unique(start)
    while frontier.size:
        kids = nbr[_gather(ptr[frontier], ptr[frontier + 1])]
        kids = np.unique(kids[~seen[kids]])
        seen[kids] = True
        frontier = kids
    return seen


def _rpo(succ: list[list[int]], root: int) -> list[int]:
    """Reverse postorder from *root*, iterative."""
    order: list[int] = []
    seen = {root}
    stack = [(root, 0)]
    while stack:
        x, i = stack[-1]
        if i < len(succ[x]):
            stack[-1] = (x, i + 1)
            y = succ[x][i]
            if y not in seen:
                seen.add(y)
                stack.append((y, 0))
            continue
        stack.pop()
        order.append(x)
    return order[::-1]


def _idom(succ: list[list[int]], pred: list[list[int]], root: int) -> list[int]:
    """Immediate dominators (Cooper-Harvey-Kennedy); -1 when unreachable."""
    order = _rpo(succ, root)
    num = [-1] * len(succ)
    for i, x in enumerate(order):
        num[x] = i
    idom = [-1] * len(succ)
    idom[root] = root

    # Iterate to a fixed point in reverse postorder.
    changed = True
    while changed:
        changed = False
        for x in order[1:]:
            new = -1
            for p in pred[x]:
                if idom[p] < 0:
                    continue
                if new < 0:
                    new = p
                    continue
                a, b = p, new
                while a != b:
                    while num[a] > num[b]:
                        a = idom[a]
                    while num[b] > num[a]:
                        b = idom[b]
                new = a
            if new != idom[x]:
                idom[x] = new
                changed = True
    return idom


def _subgraph(n: int, src: np.ndarray, dst: np.ndarray, keep: np.ndarray):
    """(nodes, global->local, succ, pred) of the subgraph induced by *keep*."""
    nodes = np.flatnonzero(keep)
    local = np.full(n, -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    inner = keep[src] & keep[dst]
    succ: list[list[int]] = [[] for _ in range(nodes.size)]
    pred: list[list[int]] = [[] for _ in range(nodes.size)]
    for u, v in zip(local[src[inner]].tolist(), local[dst[inner]].tolist(), strict=True):
        succ[u].append(v)
        pred[v].append(u)
    return nodes, local, succ, pred


def _common_chain(idom: list[int], root: int, live: list[int]) -> list[int]:
    """Common prefix of the root -> t dominator chains, targets excluded."""
    chains = []
    for t in live:
        x, chain = t, []
        while x != root:
            chain.append(x)
            x = idom[x]
        chain.append(root)
        chains.append(chain[::-1])
    tset = set(live)
    common: list[int] = []
    for col in zip(*chains, strict=False):
        if len(set(col)) != 1 or col[0] in tset:
            break
        common.append(col[0])
    return common


def waypoint_stages(
    n: int, src: np.ndarray, dst: np.ndarray, entry: int, targets: set[int]
) -> list[frozenset[int]]:
    """Dominator waypoints entry -> targets, one node set per stage.

    Only nodes on some entry -> target path matter for dominance, so the
    search runs on fwd(entry) & bwd(targets). Stages: the dominators common
    to every reachable target, then the targets themselves; at most
    ``MAX_STAGES``, nearest the target kept. Empty when no target is
    reachable.
    """
    if not targets or n == 0:
        return []
    ptr, nbr = _out_csr(n, src, dst)
    rptr, rnbr = _out_csr(n, dst, src)
    tgt = np.array(sorted(targets), dtype=np.int64)
    keep = _reach(ptr, nbr, np.array([entry], dtype=np.int64), n)
    keep &= _reach(rptr, rnbr, tgt, n)
    live = [int(t) for t in tgt if keep[t]]
    if not live:
        return []

    nodes, local, succ, pred = _subgraph(n, src, dst, keep)
    root = int(local[entry])
    idom = _idom(succ, pred, root)
    common = _common_chain(idom, root, [int(local[t]) for t in live])
    stages = [frozenset({int(nodes[x])}) for x in common]
    stages.append(frozenset(live))
    return stages[-MAX_STAGES:]


# Waypoints run from this function's entry block.
_ENTRY_FUNC = "main"


class WaypointPlan:
    """Stages resolved to shim probe keys for one target binary."""

    def __init__(self, n_stages: int, node_of: dict[int, int], dist: dict[int, float]):
        self.n_stages = n_stages
        self.node_of = node_of  # probe key -> stage index
        self.dist = dist  # probe key -> distance, for keys the table lacks


def plan_waypoints(td) -> WaypointPlan | None:
    """Stages for a loaded TargetDistance; None when none resolve.

    Builds the whole-program ICFG (cached CFGs). Caller-side waypoint sites
    carry no AFLGo block value, so ``dist`` gives them the function-level
    distance the Python path already falls back to.
    """
    from fuzzer_tool.core.icfg import build_interprocedural_cfg

    icfg = build_interprocedural_cfg(td)
    if icfg is None:
        return None
    entry = next((i for i, f in enumerate(icfg.node_funcs) if f == _ENTRY_FUNC), None)
    if entry is None:
        return None

    # Target block starts -> node indices (exact matches only).
    addrs = np.frombuffer(icfg.node_addrs, dtype=np.uint64)
    want = np.array(sorted(td.target_blocks()), dtype=np.uint64)
    pos = np.minimum(np.searchsorted(addrs, want), max(len(addrs) - 1, 0))
    targets = set(pos[addrs[pos] == want].tolist()) if want.size else set()

    stages = waypoint_stages(icfg.n_nodes, icfg.src, icfg.dst, entry, targets)
    blocks = [frozenset(int(addrs[i]) for i in s) for s in stages]
    node_of = td.waypoint_node_table(blocks)
    if not node_of:
        return None
    dist = {key: td.bb_distance(call) for key, call in td.waypoint_sites(node_of)}
    return WaypointPlan(len(stages), node_of, dist)
