"""Mutual information between input bytes and coverage edges.

Computes I(X_i; Y) where X_i is byte position i and Y is the edge bitmap.
High-MI bytes are the ones that actually control which code paths execute —
mutating them is more likely to discover new coverage.

Also provides:
- Conditional MI: I(X_i; Y | X_j) — MI of byte i given byte j is fixed
- Interaction information: I(X_i, X_j; Y) — synergy between two positions
- Per-position MI profiles for scheduling decisions
"""

import math
from array import array
from collections import defaultdict

import numpy as np

from fuzzer_tool.core.rand_pool import get_default_rand_pool

# Maximum edges tracked per (position, byte_val) pair in the joint
# distribution.  Without this cap the joint dict grows without bound
# and record() becomes O(positions × total_edges).  With the cap we
# bound it to O(positions × MAX_EDGES_PER_CELL).
MAX_EDGES_PER_CELL = 64

# Maximum total (position, byte_val, edge) cells across the joint.
# MAX_EDGES_PER_CELL alone is not enough: max_positions positions x 256
# byte values x MAX_EDGES_PER_CELL edges reaches tens of GB (the observed
# 796MB mi.json / multi-GB RSS blowup).  When the budget is exceeded the
# least-observed position is evicted (joint + marginals), bounding memory.
# Tuned down from 2M (2026-08): the joint is serialized verbatim to mi.json
# at every shutdown, and 2M nested-dict cells produced a ~100MB file (the
# other state jsons are KBs).  250k cells keeps both RSS (~30MB) and the
# on-disk snapshot modest while still covering the informative positions.
MAX_JOINT_CELLS = 250_000

# Maximum byte positions the tracker will ever track (independent of the
# fuzzer's max_len, which auto-grows to 65536).  Positions beyond this are
# skipped in record(); the cap keeps the joint bounded.
MI_MAX_POSITIONS = 4096


def _nlog2n(k: int) -> float:
    """k * log2(k), with 0 log 0 = 0."""
    return k * math.log2(k) if k > 0 else 0.0


def _fold_edges(hit_edges: set[int], map_size: int) -> set[int]:
    """Fold opaque edge hashes into [0, map_size): mask for power-of-two maps, else modulo."""
    if map_size & (map_size - 1) == 0:
        return {e & (map_size - 1) for e in hit_edges}
    return {e % map_size for e in hit_edges}


def _load_edge_marginal(em: dict | list) -> array:
    """Rebuild the dense edge_marginal array from its list or legacy sparse-dict form."""
    if not isinstance(em, dict):
        return array("Q", em)
    max_idx = max((int(k) for k in em), default=-1) + 1 if em else 0
    out = array("Q", [0]) * max_idx
    for k, v in em.items():
        out[int(k)] = v
    return out


def _load_joint(raw_joint: dict) -> defaultdict:
    """Rebuild the nested joint table from nested-dict or legacy packed-key form."""
    first_val = next(iter(raw_joint.values()))
    if isinstance(first_val, dict):
        return defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)),
            {
                int(pos): defaultdict(
                    lambda: defaultdict(int),
                    {
                        int(bv): defaultdict(int, {int(e): c for e, c in edges.items()})
                        for bv, edges in byte_vals.items()
                    },
                )
                for pos, byte_vals in raw_joint.items()
            },
        )

    # Legacy packed key: pos<<16 | byte_val<<8 | edge.
    joint: defaultdict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for k, v in raw_joint.items():
        key = int(k)
        pos = (key >> 16) & 0xFFFF
        bv = (key >> 8) & 0xFF
        edge = key & 0xFF
        joint[pos][bv][edge] = v
    return joint


class MutualInformationTracker:
    """Track mutual information between byte positions and coverage edges.

    Maintains joint distributions P(byte_val, edge_hit) incrementally.
    After sufficient observations, computes MI profiles that guide
    mutation scheduling toward high-impact byte positions.

    Args:
        max_positions: Maximum number of byte positions to track.
        min_observations: Minimum observations before computing MI.
    """

    def __init__(self, max_positions: int = 4096, min_observations: int = 50):
        self.max_positions = max_positions
        self.min_observations = min_observations
        self._total_edges: int | None = None  # cached sum(edge_marginal)
        # Zero-copy ndarray view over edge_marginal for vectorized sums;
        # rebuilt when the array grows (extend() may realloc) or is reloaded.
        self._edge_marginal_view = None
        self._edge_marginal_view_len = -1

        # Per-position: byte_value -> edge_index -> count
        # P(X_i = v, Y = e)
        self.joint: dict[int, dict[int, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        # Per-position: byte_value -> count
        self.byte_marginal: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # Global: edge_index -> count (dense array for O(1) access)
        self.edge_marginal: array = array("Q")
        self._edge_marginal_size = 0
        # Total observations per position
        self.position_counts: dict[int, int] = defaultdict(int)
        self.total_observations: int = 0
        # Live count of joint (position, byte_val, edge) cells, bounded by
        # MAX_JOINT_CELLS via least-observed-position eviction.
        self._joint_cells = 0
        self._reset_sums()

    def record(self, input_bytes: bytes, hit_edges: set[int], map_size: int = 65536) -> None:
        """Record one input-coverage pair.

        Args:
            input_bytes: The input that was executed.
            hit_edges: Set of edge IDs that were hit in this execution.
            map_size: Maximum edge index to consider.
        """
        self.total_observations += 1
        self._total_edges = None
        # The C shim emits full 32-bit edge hashes (caller_ctx ^ prev_loc ^
        # cur_loc) with collisions resolved by SHM linear probing, so edge IDs
        # are opaque hashes, not dense indices.  edge_marginal is a dense array
        # indexed by edge ID, so fold IDs into [0, map_size) before counting;
        # an unmasked high hash would force a multi-GB allocation on its first
        # sighting.  Power-of-two maps (AFL convention) mask; others modulo.
        if map_size > 0:
            hit_edges = _fold_edges(hit_edges, map_size)
        # Invalidate weighted_position cache when a new position appears
        if hasattr(self, "_wp_sorted_pos") and self._wp_sorted_pos is not None:
            self._drop_stale_wp(input_bytes)

        for pos, byte_val in enumerate(input_bytes):
            if pos >= self.max_positions:
                break
            self.position_counts[pos] += 1
            self.byte_marginal[pos][byte_val] += 1
            # Only update the expensive joint distribution once the position
            # has enough observations to compute MI.  New (position, byte_val,
            # edge) cells are rejected once MAX_JOINT_CELLS is exhausted so
            # the joint is hard-bounded; existing cells keep incrementing
            # (evicting least-observed positions instead would thrash, since
            # evicted positions are re-observed and immediately re-victimized).
            if (
                self.position_counts[pos] >= self.min_observations
                and self._joint_cells < MAX_JOINT_CELLS
                and hit_edges
            ):
                self._record_cells(pos, byte_val, hit_edges)

    def _record_cells(self, pos: int, byte_val: int, hit_edges: set[int]) -> None:
        """Count (pos, byte_val, edge) cells, keeping the profile sums in step.

        Per cell increment c -> c+1: sum f(c) of the position grows by
        f(c+1) - f(c), and K[pos, edge] by 1 (see ``mi_profile``).
        """
        cell = self.joint[pos][byte_val]
        k_slot, k_cnt = self._k_slot, self._k_cnt
        log2 = math.log2
        s_f = 0.0
        for bv_edges, edge in enumerate(hit_edges):
            if self._joint_cells >= MAX_JOINT_CELLS:
                break
            old = cell.get(edge, 0)
            if old == 0:
                self._joint_cells += 1
            cell[edge] = old + 1
            # Update edge marginal array
            if edge >= self._edge_marginal_size:
                self._grow_edge_marginal(edge)
            self.edge_marginal[edge] += 1

            s_f += (old + 1) * log2(old + 1) - (old * log2(old) if old else 0.0)
            key = pos << 32 | edge
            slot = k_slot.get(key)
            if slot is None:
                k_slot[key] = len(k_cnt)
                self._k_pos.append(pos)
                self._k_edge.append(edge)
                k_cnt.append(1)
            else:
                k_cnt[slot] += 1
            if bv_edges >= MAX_EDGES_PER_CELL:
                break
        self._s_f[pos] = self._s_f.get(pos, 0.0) + s_f

    def _reset_sums(self) -> None:
        """Empty the closed-form profile sums (sum f(c) per position, K slots)."""
        self._s_f: dict[int, float] = {}
        self._k_slot: dict[int, int] = {}
        self._k_pos = array("q")
        self._k_edge = array("q")
        self._k_cnt = array("q")

    def _rebuild_sums(self) -> None:
        """Recompute the profile sums from the joint (after a load)."""
        self._reset_sums()
        self._total_edges = None
        for pos, byte_vals in self.joint.items():
            s_f = 0.0
            for edges in byte_vals.values():
                for edge, c in edges.items():
                    s_f += _nlog2n(c)
                    key = pos << 32 | edge
                    slot = self._k_slot.get(key)
                    if slot is None:
                        self._k_slot[key] = len(self._k_cnt)
                        self._k_pos.append(pos)
                        self._k_edge.append(edge)
                        self._k_cnt.append(c)
                    else:
                        self._k_cnt[slot] += c
            self._s_f[pos] = s_f

    def _drop_stale_wp(self, input_bytes: bytes) -> None:
        """Clear the weighted_position cache if this input's last position is new."""
        max_pos = len(input_bytes) - 1 if input_bytes else 0
        if max_pos >= self.max_positions:
            max_pos = self.max_positions - 1
        # Below min_observations the position cannot enter the cache, so a
        # rebuild changes nothing structural (was 69% of rebuilds).
        if self.position_counts.get(max_pos, 0) < self.min_observations:
            return
        import bisect

        if (
            bisect.bisect_left(self._wp_sorted_pos, max_pos) == len(self._wp_sorted_pos)
            or self._wp_sorted_pos[
                min(
                    bisect.bisect_left(self._wp_sorted_pos, max_pos),
                    len(self._wp_sorted_pos) - 1,
                )
            ]
            != max_pos
        ):
            self._wp_sorted_pos = None

    def _grow_edge_marginal(self, edge: int) -> None:
        """Extend the dense edge_marginal array to cover ``edge`` (cold path)."""
        # Release the view first: array.array refuses to
        # resize while a frombuffer view exports its buffer.
        self._edge_marginal_view = None
        self._edge_marginal_view_len = -1
        self.edge_marginal.extend(array("Q", [0]) * (edge + 1 - self._edge_marginal_size))
        self._edge_marginal_size = edge + 1

    def _evict_least_observed(self) -> None:
        """Drop the least-observed position with joint cells.

        Used only to trim an oversized state loaded from disk: the joint is
        hard-capped during recording, so this runs at most once per load.
        """
        if not self.joint:
            return
        victim = min(self.joint, key=lambda p: self.position_counts.get(p, 0))
        joint_pos = self.joint.pop(victim)
        for byte_vals in joint_pos.values():
            for edge, count in byte_vals.items():
                self.edge_marginal[edge] -= count
                self._joint_cells -= 1
        self.byte_marginal.pop(victim, None)
        self.position_counts.pop(victim, None)
        self._total_edges = None
        if hasattr(self, "_wp_sorted_pos"):
            self._wp_sorted_pos = None

    def _edge_marginal_sum(self) -> int:
        """sum(edge_marginal) via a zero-copy numpy view.

        Python-level sum() over an array("Q") boxes every C value; a
        frombuffer view shares the buffer (no copy) and sums vectorized
        (~40x faster at 64K entries). Rebuilt when the array grows or after
        a reload; in-place value mutations (eviction) are visible through
        the view since it shares memory.
        """
        return int(self._marginal_view().sum())

    def _marginal_view(self) -> np.ndarray:
        """Zero-copy uint64 view over edge_marginal, rebuilt after growth or reload."""
        if self._edge_marginal_view is None or (
            len(self.edge_marginal) != self._edge_marginal_view_len
        ):
            self._edge_marginal_view = np.frombuffer(self.edge_marginal, dtype=np.uint64)
            self._edge_marginal_view_len = len(self.edge_marginal)
        return self._edge_marginal_view

    def mi(self, position: int) -> float:
        """Compute I(X_pos; Y) in bits.

        I(X; Y) = sum_{x,y} P(x,y) * log2(P(x,y) / (P(x) * P(y)))

        Returns 0.0 if insufficient data or position not observed.
        """
        n = self.position_counts.get(position, 0)
        if n < self.min_observations:
            return 0.0

        if self._total_edges is None:
            self._total_edges = self._edge_marginal_sum()
        total_edges = self._total_edges
        if total_edges == 0:
            return 0.0

        mi_value = 0.0
        byte_counts = self.byte_marginal.get(position, {})
        joint_pos = self.joint.get(position, {})

        for byte_val, bv_count in byte_counts.items():
            p_x = bv_count / n
            for edge, joint_count in joint_pos.get(byte_val, {}).items():
                p_xy = joint_count / n
                p_y = (
                    self.edge_marginal[edge] / total_edges if edge < self._edge_marginal_size else 0
                )
                if p_xy > 0 and p_y > 0:
                    mi_value += p_xy * math.log2(p_xy / (p_x * p_y))

        return max(0.0, mi_value)

    def mi_profile(self, input_length: int | None = None) -> dict[int, float]:
        """Compute MI for all tracked positions.

        Args:
            input_length: Only compute for positions < input_length.

        Returns:
            Dict mapping position -> MI in bits.
        """
        if input_length is None:
            input_length = max(self.position_counts.keys()) + 1 if self.position_counts else 0
        positions = [pos for pos in range(input_length) if pos in self.position_counts]
        profile = self._closed_profile(positions)
        if profile is None:
            return {pos: self.mi(pos) for pos in positions}
        return profile

    def _closed_profile(self, positions: list[int]) -> dict[int, float] | None:
        """``mi()`` of each position from the running sums; None if they cannot apply.

        n*MI = sum f(c) + C*log2(T) - sum_x J_x*log2(b_x) - sum_e K_e*log2(m_e),
        f(k) = k*log2(k), C = sum c, J_x = sum_e c, K_e = sum_x c. The edge
        term is one bincount; the byte term one pass over (byte -> edges)
        dicts, not over cells. None (per-cell fallback) when a cell names an
        edge with no marginal or a byte with no count: loaded state only.
        """
        out = dict.fromkeys(positions, 0.0)
        if self._total_edges is None:
            self._total_edges = self._edge_marginal_sum()
        total_edges = self._total_edges
        if total_edges == 0:
            return out

        # Edge term for every position at once.
        k_edge = np.frombuffer(self._k_edge, dtype=np.int64)
        if len(k_edge) and int(k_edge.max()) >= self._edge_marginal_size:
            return None
        m = self._marginal_view()[k_edge].astype(np.float64)
        if (m <= 0).any():
            return None
        k_pos = np.frombuffer(self._k_pos, dtype=np.int64)
        k_cnt = np.frombuffer(self._k_cnt, dtype=np.int64)
        edge_term = np.bincount(
            k_pos, weights=k_cnt * np.log2(m), minlength=max(positions, default=0) + 1
        ).tolist()

        log2_t = math.log2(total_edges)
        for pos in positions:
            n = self.position_counts[pos]
            joint_pos = self.joint.get(pos)
            if n < self.min_observations or not joint_pos:
                continue
            byte_counts = self.byte_marginal.get(pos, {})
            c_total = 0
            byte_term = 0.0
            for byte_val, edges in joint_pos.items():
                j = sum(edges.values())
                if not j:
                    continue
                b = byte_counts.get(byte_val, 0)
                if b <= 0:
                    return None
                c_total += j
                byte_term += j * math.log2(b)
            value = self._s_f.get(pos, 0.0) + c_total * log2_t - byte_term - edge_term[pos]
            out[pos] = max(0.0, value / n)
        return out

    def top_positions(
        self, k: int = 10, input_length: int | None = None
    ) -> list[tuple[int, float]]:
        """Return the k positions with highest MI.

        Returns:
            List of (position, mi_bits) sorted by MI descending.
        """
        profile = self.mi_profile(input_length)
        sorted_pos = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        return sorted_pos[:k]

    def mutation_weight(
        self, position: int, input_length: int, mi_profile: dict[int, float] | None = None
    ) -> float:
        """Compute a mutation weight for a position based on MI.

        Returns a weight in [0.1, 5.0]:
        - High MI → weight near 5.0 (mutate this position aggressively)
        - Low MI → weight near 0.1 (skip this position)

        Normalizes MI to [0, 1] using the maximum observed MI across positions,
        then maps to the weight range.

        Args:
            position: Byte position to weight.
            input_length: Only consider positions < input_length.
            mi_profile: Optional precomputed ``{position: mi}`` mapping.  When
                supplied the caller avoids redundant ``mi()`` work; when omitted
                the profile is computed on demand.
        """
        if mi_profile is None:
            mi_profile = self.mi_profile(input_length)

        mi_val = mi_profile.get(position, 0.0)
        if mi_val <= 0:
            return 0.1

        max_mi = max(mi_profile.values()) if mi_profile else 0.0
        if max_mi <= 0:
            return 1.0

        normalized = mi_val / max_mi
        return 0.1 + 4.9 * normalized

    def weighted_position(self, input_length: int) -> int | None:
        """Sample a byte position weighted by MI.

        Uses MI-weighted roulette wheel selection. Returns a position
        in [0, input_length) that is more likely to be information-rich.
        Precomputes sorted positions + cumulative weights; uses bisect
        for O(log n) filtering instead of rebuilding lists each call.
        """
        if not self.position_counts:
            return None

        if not hasattr(self, "_wp_sorted_pos"):
            self._wp_sorted_pos = None
            self._wp_cum_weights = None
            self._wp_total = 0.0

        if self._wp_sorted_pos is None:
            # Compute MI profile once instead of calling mi() twice per
            # position via mutation_weight()'s old max-scan path.
            mi_profile = self.mi_profile(min(input_length, self.max_positions))
            # Build sorted (position, weight) pairs and cumulative sum
            pairs = []
            for pos in self.position_counts:
                if self.position_counts[pos] >= self.min_observations and pos < input_length:
                    pairs.append(
                        (pos, self.mutation_weight(pos, input_length, mi_profile=mi_profile))
                    )
            if not pairs:
                self._wp_sorted_pos = []
                self._wp_cum_weights = []
                self._wp_total = 0.0
                return None
            pairs.sort(key=lambda x: x[0])
            self._wp_sorted_pos = [p for p, _ in pairs]
            cum = 0.0
            cum_list = []
            for _, w in pairs:
                cum += w
                cum_list.append(cum)
            self._wp_cum_weights = cum_list
            self._wp_total = cum

        sorted_pos = self._wp_sorted_pos
        cum_w = self._wp_cum_weights
        total = self._wp_total

        if not sorted_pos or total <= 0:
            return None

        # Find cutoff index: positions < input_length
        import bisect

        if input_length >= self.max_positions:
            n = len(sorted_pos)
        else:
            n = bisect.bisect_left(sorted_pos, input_length)

        if n == 0:
            return None

        # Weighted sampling using precomputed cumulative sum
        r = get_default_rand_pool().random() * cum_w[n - 1]
        idx = bisect.bisect_right(cum_w, r, hi=n)
        return sorted_pos[min(idx, n - 1)]

    def conditional_mi(self, position_a: int, position_b: int) -> float:
        """Compute I(X_a; Y | X_b) — MI of position a given position b is observed.

        This captures the *additional* information position a provides
        beyond what position b already tells us about coverage.

        Uses the chain rule: I(X_a; Y | X_b) = H(X_a | X_b) + H(Y | X_b) - H(X_a, Y | X_b)

        Simplified approximation: if positions are independent given coverage,
        this equals I(X_a; Y). Significant deviation indicates correlation.
        """
        mi_a = self.mi(position_a)
        mi_b = self.mi(position_b)
        if mi_a == 0 or mi_b == 0:
            return mi_a

        # Joint MI of both positions
        joint = self._joint_mi_two(position_a, position_b)
        # I(X_a; Y | X_b) ≈ I(X_a, X_b; Y) - I(X_b; Y)
        return max(0.0, joint - mi_b)

    def _joint_mi_two(self, pos_a: int, pos_b: int) -> float:
        """Compute I(X_a, X_b; Y) — joint MI of two positions with coverage.

        Approximated by treating (byte_a, byte_b) as a combined symbol.
        Only works when both positions have sufficient observations.
        """
        n_a = self.position_counts.get(pos_a, 0)
        n_b = self.position_counts.get(pos_b, 0)
        if n_a < self.min_observations or n_b < self.min_observations:
            return 0.0

        # Build joint distribution: (byte_a, byte_b) -> edge -> count
        # This is approximate — we assume independent marginals
        # A more exact version would require tracking all position pairs
        mi_a = self.mi(pos_a)
        mi_b = self.mi(pos_b)
        # Upper bound: I(X_a,X_b;Y) <= I(X_a;Y) + I(X_b;Y)
        # Lower bound: I(X_a,X_b;Y) >= max(I(X_a;Y), I(X_b;Y))
        # Use sum as approximation (independence assumption)
        return mi_a + mi_b

    def interaction_information(self, pos_a: int, pos_b: int) -> float:
        """Compute interaction information: I(X_a, X_b; Y) - I(X_a; Y) - I(X_b; Y).

        Positive = synergy (together they explain more than sum of parts)
        Negative = redundancy (they explain overlapping coverage)
        Zero = independent
        """
        joint = self._joint_mi_two(pos_a, pos_b)
        mi_a = self.mi(pos_a)
        mi_b = self.mi(pos_b)
        return joint - mi_a - mi_b

    def to_dict(self) -> dict:
        """Serialize tracker state to a dict (for StateStore pickle)."""
        return {
            "max_positions": self.max_positions,
            "min_observations": self.min_observations,
            "total_observations": self.total_observations,
            "position_counts": dict(self.position_counts),
            "edge_marginal": self.edge_marginal.tolist(),
            "byte_marginal": {
                str(pos): {str(bv): c for bv, c in counts.items()}
                for pos, counts in self.byte_marginal.items()
            },
            "joint": {
                str(pos): {
                    str(bv): {str(e): c for e, c in edges.items()}
                    for bv, edges in byte_vals.items()
                }
                for pos, byte_vals in self.joint.items()
            },
        }

    def from_dict(self, data: dict) -> None:
        """Restore tracker state from a serialized dict."""
        self.max_positions = data.get("max_positions", self.max_positions)
        self.min_observations = data.get("min_observations", self.min_observations)
        self.total_observations = data.get("total_observations", 0)
        self.position_counts = defaultdict(
            int, {int(k): v for k, v in data.get("position_counts", {}).items()}
        )
        self.edge_marginal = _load_edge_marginal(data.get("edge_marginal", []))
        self._edge_marginal_size = len(self.edge_marginal)
        self._edge_marginal_view = None
        self._edge_marginal_view_len = -1
        self.byte_marginal = defaultdict(
            lambda: defaultdict(int),
            {
                int(pos): defaultdict(int, {int(bv): c for bv, c in counts.items()})
                for pos, counts in data.get("byte_marginal", {}).items()
            },
        )
        raw_joint = data.get("joint", {})
        if raw_joint:
            self.joint = _load_joint(raw_joint)
        self._joint_cells = sum(
            len(edges) for pos_vals in self.joint.values() for edges in pos_vals.values()
        )
        while self._joint_cells > MAX_JOINT_CELLS:
            self._evict_least_observed()
        self._rebuild_sums()

    def save(self, path: str) -> bool:
        """Save tracker state to JSON (legacy interface)."""
        import json

        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"))
            return True
        except OSError:
            return False

    def load(self, path: str) -> bool:
        """Load tracker state from JSON (legacy interface)."""
        import json

        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        self.from_dict(data)
        return True
