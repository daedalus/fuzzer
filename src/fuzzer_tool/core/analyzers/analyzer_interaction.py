"""Second-order byte interaction (perturbation theory, second order).

``ByteSensitivityTracker`` is the first-order term: how far the trace moves
when ONE byte changes. It cannot see coupled bytes -- a length and its
payload, a tag and its value, two guards that must both flip -- because
each looks ordinary alone. The second-order term exposes them.

Per edge ``e``, with ``x_S(e) = 1`` when ``e`` fires after changing the
bytes in ``S``, the discrete mixed partial is::

    d2_ij(e) = x_ij(e) - x_i(e) - x_j(e) + x_0(e)

It is zero for every edge exactly when bytes ``i`` and ``j`` act
independently. The pair's interaction strength is the L1 norm over the
union ``U`` of the four traces, normalised to ``[0, 2]``::

    I_ij = sum_e |d2_ij(e)| / |U|

Example: edge 99 fires only when both bytes change (an AND gate)::

    x_0 = 0, x_i = 0, x_j = 0, x_ij = 1  ->  d2 = +1

Cost: only the ``top_k`` offsets with the highest first-order score are
probed, ``k`` singles plus ``k*(k-1)/2`` pairs. Coupled pairs feed the
``coupled_pair`` operator. Not persisted across ``--resume``.
"""

from fuzzer_tool.core.rand_pool import RandPool, get_default_rand_pool

# Offsets probed per seed (by first-order score). 8 -> 8 + 28 = 36 execs.
INTERACTION_TOP_K = 8
# Seeds tracked before FIFO eviction; bounds memory.
INTERACTION_MAX_SEEDS = 50
# Minimum I_ij to call a pair coupled. Untuned: absorbs one flaky edge in
# a trace of ~2/INTERACTION_MIN edges.
INTERACTION_MIN = 0.02
# Seed key prefix, the ByteSensitivityTracker convention.
_KEY_LEN = 64


def mixed_difference(e0: set[int], ei: set[int], ej: set[int], eij: set[int]) -> float:
    """Normalised L1 norm of the discrete mixed partial over four traces."""
    union = e0 | ei | ej | eij
    if not union:
        return 0.0

    # Membership arithmetic per edge: [eij] - [ei] - [ej] + [e0].
    total = 0
    for e in union:
        total += abs((e in eij) - (e in ei) - (e in ej) + (e in e0))
    return total / len(union)


class ByteInteractionTracker:
    """Find coupled byte pairs per seed from second-order trace differences.

    Args:
        max_seeds: Seeds tracked before the oldest is evicted.
        top_k: First-order-hottest offsets probed per seed.
        rng: Shared RandPool (Hard Rule 16).
    """

    def __init__(
        self,
        max_seeds: int = INTERACTION_MAX_SEEDS,
        top_k: int = INTERACTION_TOP_K,
        rng: RandPool | None = None,
    ):
        self._max_seeds = max_seeds
        self._top_k = top_k
        self._rng = rng or get_default_rand_pool()
        self._pairs: dict[bytes, list[tuple[int, int]]] = {}

    def cost(self) -> int:
        """Worst-case executions analyze_seed spends on one seed."""
        k = self._top_k
        return k + k * (k - 1) // 2

    def analyzed(self, seed: bytes) -> bool:
        """True once *seed* was analyzed (and not evicted)."""
        return bytes(seed[:_KEY_LEN]) in self._pairs

    def has_data(self, seed: bytes) -> bool:
        """True when *seed* has at least one coupled pair."""
        return bool(self._pairs.get(bytes(seed[:_KEY_LEN])))

    def pairs(self, seed: bytes) -> list[tuple[int, int]]:
        """Coupled ``(i, j)`` pairs, ``i < j``, strongest first."""
        return self._pairs.get(bytes(seed[:_KEY_LEN]), [])

    def tracked(self) -> int:
        """Seeds currently held."""
        return len(self._pairs)

    def coupled(self) -> int:
        """Held seeds with at least one coupled pair."""
        return sum(1 for v in self._pairs.values() if v)

    def analyze_seed(
        self, seed: bytes, original_edges: set[int], exec_fn, scores
    ) -> list[tuple[int, int]]:
        """Probe the hottest offsets of *seed* and store its coupled pairs.

        Args:
            seed: Seed bytes.
            original_edges: Edge set of the unmodified seed.
            exec_fn: Callable(bytes) -> set[int] running one input.
            scores: First-order sensitivity per offset.

        Returns:
            Coupled pairs, strongest first.
        """
        key = bytes(seed[:_KEY_LEN])
        if key in self._pairs:
            return self._pairs[key]

        hot = self._hottest(seed, scores)
        found = self._probe(seed, original_edges, exec_fn, hot) if len(hot) >= 2 else []
        self._store(key, found)
        return found

    def _hottest(self, seed: bytes, scores) -> list[int]:
        # Top-k positive first-order offsets inside the seed, ascending.
        n = min(len(seed), len(scores))
        live = [i for i in range(n) if scores[i] > 0.0]
        live.sort(key=lambda i: (-scores[i], i))
        return sorted(live[: self._top_k])

    def _probe(self, seed: bytes, e0: set[int], exec_fn, hot: list[int]) -> list[tuple[int, int]]:
        # One value per offset, reused by singles and pairs so the mixed
        # partial compares like with like. Offset 1..255 always changes it.
        deltas = self._rng.randint_list(1, 255, len(hot))
        value = {i: (seed[i] + d) & 0xFF for i, d in zip(hot, deltas, strict=True)}

        def run(*offs: int) -> set[int]:
            buf = bytearray(seed)
            for o in offs:
                buf[o] = value[o]
            return exec_fn(bytes(buf))

        single = {i: run(i) for i in hot}

        # Score every pair; keep the coupled ones, strongest first.
        scored = []
        m = len(hot)
        for a in range(m):
            i = hot[a]
            for b in range(a + 1, m):
                j = hot[b]
                strength = mixed_difference(e0, single[i], single[j], run(i, j))
                if strength >= INTERACTION_MIN:
                    scored.append((-strength, i, j))
        scored.sort()
        return [(i, j) for _, i, j in scored]

    def _store(self, key: bytes, found: list[tuple[int, int]]) -> None:
        # FIFO eviction keeps memory bounded (Hard Rule 54).
        self._pairs[key] = found
        if len(self._pairs) > self._max_seeds:
            del self._pairs[next(iter(self._pairs))]
