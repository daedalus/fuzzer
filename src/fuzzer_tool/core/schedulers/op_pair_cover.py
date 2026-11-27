"""PairCoverScheduler: 2-way operator-sequence coverage, then pair exploitation.

Combinatorial testing (Kuhn et al., NIST) says most interaction faults need
only two parameters to line up. Here the "parameters" are consecutive
operators: the unit of coverage is the pair (previous op, next op).

    row a:  a->a  a->b  a->c  ...   cover each pair cover_target times,
                                    then exploit the best pair

Phase 1, per row: while some pair (prev, op) in the candidate list has fewer
than ``cover_target`` pulls, pick uniformly among those, so K visits to one
row cover all K pairs whatever the RNG. Phase 2: Thompson sample each pair
from Beta(s + k m, f + k (1 - m)), where m is the op's marginal posterior mean
and k = ``prior_strength``, so a thin pair shrinks toward its marginal and a
dense one follows its own pair. The context-free bandits cannot see "b pays
after a but not after c"; this one can.

State follows the run (the last recorded op), as in ``op_qlearn``. A previous
op that was never registered has no row and falls back to the marginal. The
pair table is bounded by K^2 for the K arms ever registered. Experimental,
Elo-only (absent from ``_FALLBACK_PRECEDENCE``).
"""

from __future__ import annotations

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts, reward

#: State before the first record.
START_STATE = ""


class PairCoverScheduler(ArmCounts):
    """Cover every (prev, next) pair, then Thompson-sample the pair posterior."""

    #: Pair posteriors start from the op's marginal; no per-arm prior override.
    supports_priors = False

    def __init__(self, rng, cover_target: int = 1, prior_strength: float = 2.0) -> None:
        if rng is None:
            raise ValueError("PairCoverScheduler requires a RandPool (Hard Rule 16)")
        if cover_target < 1:
            raise ValueError(f"cover_target must be >= 1, got {cover_target}")
        if not prior_strength > 0.0:
            raise ValueError(f"prior_strength must be > 0, got {prior_strength}")

        super().__init__()
        self._rng = rng
        self._cover_target = cover_target
        self._prior_strength = prior_strength
        self._pair_n: dict[tuple[str, str], int] = {}  # (prev, next) -> pulls
        self._pair_s: dict[tuple[str, str], float] = {}  # (prev, next) -> reward sum
        self._pulls = 0
        self._covered = 0  # pairs with n >= cover_target
        self._prev = START_STATE  # row the next select reads
        self._sel_prev = START_STATE  # row the next record credits

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        super().record(name, success, weight)
        self._pulls += 1

        # Only registered rows hold pairs; this keeps the table within K^2.
        key = (self._sel_prev, name)
        if self._sel_prev in self._counts:
            n = self._pair_n.get(key, 0) + 1
            self._pair_n[key] = n
            self._pair_s[key] = self._pair_s.get(key, 0.0) + reward(success, weight)
            if n == self._cover_target:
                self._covered += 1
        self._prev = name
        self._sel_prev = name

    def select_op(self, ops: list[str]) -> str:
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]

        cands = list(dict.fromkeys(ops))
        counts = self._counts
        for op in cands:
            if op not in counts:
                self.init_arm(op)

        prev = self._prev
        self._sel_prev = prev
        if prev not in counts:
            return self._thompson(cands, None)

        pair_n = self._pair_n
        target = self._cover_target
        open_pairs = [op for op in cands if pair_n.get((prev, op), 0) < target]
        if open_pairs:
            return self._rng.choice(open_pairs)
        return self._thompson(cands, prev)

    def _thompson(self, cands: list[str], prev: str | None) -> str:
        """Best Beta draw; pair posteriors shrink toward the marginal mean."""
        k = self._prior_strength
        best, best_draw = cands[0], -1.0
        for op in cands:
            m = self.mean(op)
            s, n = 0.0, 0
            if prev is not None:
                s = self._pair_s.get((prev, op), 0.0)
                n = self._pair_n.get((prev, op), 0)
            draw = self._rng.betavariate(s + k * m, (n - s) + k * (1.0 - m))
            if draw > best_draw:
                best, best_draw = op, draw
        return best

    def bandit_stats(self) -> dict:
        arms = len(self._counts)
        return {
            "pc_success": dict(self._pair_s),
            "pc_pulls": self._pulls,
            "pc_pairs_covered": self._covered,
            "pc_coverage": self._covered / (arms * arms) if arms else 0.0,
        }
