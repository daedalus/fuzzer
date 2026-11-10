"""Window influence map (Chattopadhyay-Zuckerman, Def. 1.6).

influence(Q) = P[trace undetermined | bytes outside Q fixed, Q random].
Estimated per window Q: refill Q with FILLS random strings, outside = the
seed. Score = |union \\ intersection| / |union| over those traces plus the
original. Every byte inherits its window's score. Unlike the paper the
outside is the seed, not random, so this is a local estimate.

    seed:  [ w0 ][ w1 ][ w2 ]...      w1 refilled -> traces t1, t2
    score(w1) = |U \\ I| / |U|,  U = t0|t1|t2, I = t0&t1&t2
"""

from fuzzer_tool.core.analyzers.analyzer_sensitivity import ByteSensitivityTracker

WINDOW = 8
FILLS = 2


class WindowInfluenceTracker(ByteSensitivityTracker):
    """ByteSensitivityTracker with window-level, multi-fill scoring."""

    def _n_windows(self, seed: bytes) -> int:
        n = min(len(seed), self.max_bytes)
        return -(-n // WINDOW)

    def cost(self, seed: bytes) -> int:
        wins = self._n_windows(seed)
        if wins == 0:
            return 1
        return max(1, int(wins * self.sample_rate)) * FILLS

    def analyze_seed(self, seed: bytes, original_edges: set[int], exec_fn) -> list[float]:
        key = bytes(seed[:64])
        if key in self._analyzed:
            return self._sensitivity.get(key, [])

        n = min(len(seed), self.max_bytes)
        if n == 0:
            return []

        wins = self._n_windows(seed)
        picked = self._rng.sample(range(wins), max(1, int(wins * self.sample_rate)))
        scores = [0.0] * n
        for w in picked:
            lo = w * WINDOW
            hi = min(lo + WINDOW, n)
            scores[lo:hi] = [self._window_score(seed, lo, hi, original_edges, exec_fn)] * (hi - lo)

        self._sensitivity[key] = scores
        self._analyzed.add(key)
        self._cum_cache.pop(key, None)
        self._evict()
        return scores

    def _window_score(self, seed, lo, hi, original_edges, exec_fn) -> float:
        union = set(original_edges)
        common = set(original_edges)
        for _ in range(FILLS):
            buf = bytearray(seed)
            buf[lo:hi] = self._rng.randbytes(hi - lo)
            edges = exec_fn(bytes(buf))
            union |= edges
            common &= edges
        if not union:
            return 0.0
        return (len(union) - len(common)) / len(union)

    def _evict(self) -> None:
        if len(self._analyzed) <= self.max_seeds:
            return
        oldest = next(iter(self._analyzed))
        self._analyzed.discard(oldest)
        self._sensitivity.pop(oldest, None)
        self._cum_cache.pop(oldest, None)
