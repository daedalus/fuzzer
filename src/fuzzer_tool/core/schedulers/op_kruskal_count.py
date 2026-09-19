"""OpKruskalCountScheduler: Kruskal-count coupling over the operator jump graph.

``core/schedulers/seed_kruskal_count.py`` (the seed-picker's ``kruskal_count``
Elo arm) builds its functional graph from a seed's own bytes: ``WALKER_COUNT``
walkers step through the seed's byte values, and the score is how fast/
completely they couple. That domain (byte offsets ``0..len(seed)-1``) can be
arbitrarily large, so it needs a fixed ``MAX_STEPS`` cap and a numpy-
vectorized batch path (``batch_scores``) to stay cheap across a corpus.

The operator domain differs in a way that simplifies things instead of
complicating them: ``n`` is the count of currently-offered ops, always small
(tens, not thousands), and there is no byte array to walk -- so the jump
table has to come from each op's own observed success rate instead:

    jump(i) = 1 + floor(rate_i * (n - 1))

A proven op skips far ahead in the index space; an op with no track record
barely moves. That is deliberately the same exploitation signal
``core/schedulers/op_katz.py``'s beta uses (raw success rate, not
``1 - rate``) -- see that module's docstring for why the seed-side and
op-side arms want opposite signs from the same family of graph algorithm --
just consumed by a discrete walk here instead of solved as a linear system.

Because ``f: {0..n-1} -> {0..n-1}`` is a functional graph over a domain of
size ``n``, every walker enters a cycle within at most ``n`` steps, and any
two walkers that ever meet must meet within about ``2n`` steps. So unlike
the seed version's fixed ``MAX_STEPS=256``, the walk here is capped at
``2*n`` computed per call, and there is no cyclic/acyclic split to worry
about the way ``seed_katz.py`` vs ``op_katz.py`` had to reconcile: a
functional graph on a finite domain is never anything but eventually
cyclic, so one code path covers every case. It also never needs numpy:
``op_katz.py``'s ``classical_katz_scores`` does an O(n^3) eigendecomposition
plus a linear solve on every ``scores()`` call; this does an O(n^2) integer
walk over plain lists.

``score(op_i) = rate_i * (1 + attractor_share_i)``, where
``attractor_share_i`` is the fraction of walker starts whose trajectory,
from the first pairwise coupling onward, visits index ``i`` before closing
its own cycle. An op that many different starting points funnel into gets
amplified above its raw rate alone -- the op-side analogue of Katz
centrality's "does it feed into other productive ops," discovered by
simulation rather than by solving for a fixed point.

Landed off by default, reached only via ``--elo`` (see the
``_FALLBACK_PRECEDENCE`` comment in ``services/operators.py``), same
posture as ``op_katz``/``op_tang``: this is an unproven exploratory arm,
not measured against the convergence harness yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fuzzer_tool.core.rand_pool import RandPool

WALKER_COUNT = 4


@dataclass
class WalkTrace:
    """First coupling step per walker pair, plus each walker's full path."""

    couple_steps: dict[tuple[int, int], int] = field(default_factory=dict)
    paths: list[list[int]] = field(default_factory=list)


def walker_starts(n: int) -> list[int]:
    """Distinct evenly spaced starts; fewer walkers than WALKER_COUNT when n is tiny."""
    k = min(n, WALKER_COUNT)
    if k <= 0:
        return []
    return [i * n // k for i in range(k)]


def jump_table(rates: list[float]) -> list[int]:
    """Per-index jump length from each op's own success rate, clamped to [0, 1]."""
    n = len(rates)
    if n <= 1:
        return [1] * n
    return [1 + int(max(0.0, min(1.0, r)) * (n - 1)) for r in rates]


def step(i: int, jumps: list[int]) -> int:
    """One functional-graph step: f(i) = (i + jump(i)) % n."""
    return (i + jumps[i]) % len(jumps)


def trace(jumps: list[int]) -> WalkTrace:
    """Advance all walkers until every pair has coupled or the domain cap is hit."""
    out = WalkTrace()
    n = len(jumps)
    pos = walker_starts(n)
    k = len(pos)
    total = k * (k - 1) // 2
    cap = max(2 * n, 1)
    out.paths = [[] for _ in range(k)]

    for s in range(1, cap + 1):
        pos = [step(p, jumps) for p in pos]
        for w in range(k):
            out.paths[w].append(pos[w])
        if len(out.couple_steps) == total:
            continue
        for i in range(k):
            for j in range(i + 1, k):
                if pos[i] != pos[j] or (i, j) in out.couple_steps:
                    continue
                out.couple_steps[(i, j)] = s
    return out


def attractor_shares(t: WalkTrace, n: int) -> list[float]:
    """Fraction of walker starts whose post-coupling trajectory visits each index.

    All zero when nothing has coupled -- e.g. a single walker (n < 2) has no
    pair to couple with, so its own private cycle carries no attractor
    signal, only its raw rate does.
    """
    if not t.couple_steps or not t.paths:
        return [0.0] * n

    earliest = min(t.couple_steps.values())
    hits = [0] * n
    for path in t.paths:
        seen: dict[int, None] = {}
        for idx, p in enumerate(path, start=1):
            if idx < earliest:
                continue
            if p in seen:
                break
            seen[p] = None
        for pos in seen:
            hits[pos] += 1
    k = len(t.paths)
    return [h / k for h in hits] if k else [0.0] * n


class OpKruskalCountScheduler:
    """Elo-arm operator scheduler: Kruskal-count coupling over the op jump graph.

    Lifecycle mirrors ``OpKatzScheduler``/``FPLScheduler``/``ReplicatorScheduler``:
    the fuzzer holds one instance, calls :meth:`record` on every outcome, and
    :meth:`select_op` picks from the offered arm list.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16).
    """

    supports_priors = False

    def __init__(self, rng: RandPool | None = None):
        if rng is None:
            raise ValueError("OpKruskalCountScheduler requires a RandPool (Hard Rule 16)")
        self._rng = rng
        self.successes: dict[str, float] = {}
        self.attempts: dict[str, float] = {}

    def record(self, op: str, success: bool, weight: float = 1.0) -> None:
        """Record one outcome. Mirrors the shared ``record(op, success, weight=...)``
        contract (see ``op_katz.py``) so this sits in the same reward fan-out
        loop. Unlike ``op_katz``, there is no transition edge to accumulate:
        the jump graph is rebuilt fresh from each op's own rate on every
        :meth:`scores` call rather than carried incrementally.
        """
        self.attempts[op] = self.attempts.get(op, 0.0) + 1.0
        if success:
            self.successes[op] = self.successes.get(op, 0.0) + max(weight, 0.0)

    def rates(self, ops: list[str]) -> list[float]:
        """Raw success rate per op, 0.0 for an op never attempted."""
        out = []
        for op in ops:
            attempts = self.attempts.get(op, 0.0)
            rate = self.successes.get(op, 0.0) / attempts if attempts > 0 else 0.0
            out.append(min(max(rate, 0.0), 1.0))
        return out

    def scores(self, ops: list[str]) -> dict[str, float]:
        """Kruskal-count score per op, restricted to the offered ``ops`` list.

        ``rate_i`` is exploitation-favoring like ``op_katz``'s beta (see
        module docstring); ``attractor_share_i`` is this arm's own
        propagation term, discovered by the coupling walk instead of solved
        for algebraically.
        """
        rates = self.rates(ops)
        jumps = jump_table(rates)
        t = trace(jumps)
        shares = attractor_shares(t, len(ops))
        return {op: r * (1.0 + s) for op, r, s in zip(ops, rates, shares, strict=True)}

    def select_op(self, ops: list[str]) -> str:
        """Softmax-free weighted pick: scores shifted non-negative, sampled by mass.

        Identical draw mechanism to ``OpKatzScheduler.select_op`` -- see that
        method's docstring for why a direct weighted draw is used instead of
        a temperature softmax here.
        """
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        s = self.scores(ops)
        vals = [s[op] for op in ops]
        lo = min(vals)
        shifted = [v - lo + 1e-9 for v in vals]
        total = sum(shifted)
        probs = [v / total for v in shifted] if total > 0 else [1.0 / len(ops)] * len(ops)
        r = self._rng.random()
        cumulative = 0.0
        for op, p in zip(ops, probs, strict=True):
            cumulative += p
            if r <= cumulative:
                return op
        return ops[-1]
