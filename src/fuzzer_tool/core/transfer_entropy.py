"""Transfer entropy for directional information flow analysis.

Transfer entropy (Schreiber 2000) measures the directional flow of
information between time series:

  T_{X→Y} = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)

It answers: "does knowing X's past improve prediction of Y's future,
beyond what Y's own past already tells us?"

Applied to fuzzing:
- Byte positions as X, coverage edges as Y → "which bytes causally
  influence which code paths?"
- Edge_i as X, Edge_j as Y → "which code paths lead to other code paths?"
- Operator as X, coverage as Y → "which operators causally drive discoveries?"

Unlike mutual information (which is symmetric and non-directional),
transfer entropy captures *causal* asymmetry: X→Y ≠ Y→X.

Provides:
- Transfer entropy estimation via k-nearest neighbors (Kraskov-Stögbauer-Grassberger)
- Directional information flow maps
- Causal chain detection
"""

import math
from collections import defaultdict


class TransferEntropy:
    """Estimate transfer entropy between discrete signals.

    Uses bin-based estimation for discrete data (byte values, edge indices).

    Args:
        history_length: Number of past values to condition on (k in T_{X→Y}).
        n_bins: Number of bins for discretization (if data isn't already discrete).
    """

    def __init__(self, history_length: int = 1, n_bins: int = 256):
        self.k = history_length
        self.n_bins = n_bins

    def transfer_entropy(
        self,
        source: list[int],
        target: list[int],
    ) -> float:
        """Compute T_{source → target} in bits.

        T_{X→Y} = H(Y_{t+1} | Y_t^{(k)}) - H(Y_{t+1} | Y_t^{(k)}, X_t)

        where Y_t^{(k)} = (Y_t, Y_{t-1}, ..., Y_{t-k+1}) is the k-length
        history of Y.

        Args:
            source: Time series of source values (X).
            target: Time series of target values (Y). Must be same length.

        Returns:
            Transfer entropy in bits. Positive means X→Y information flow.
            Zero means no directed influence.
        """
        n = min(len(source), len(target))
        if n < self.k + 2:
            return 0.0

        # Build joint distributions
        # P(Y_{t+1}, Y_t^{(k)}) — target alone
        joint_target = defaultdict(int)
        # P(Y_{t+1}, Y_t^{(k)}, X_t) — with source
        joint_both = defaultdict(int)
        # Marginal counts
        count_target = 0
        count_both = 0

        for t in range(self.k, n - 1):
            y_future = target[t + 1]
            y_hist = tuple(target[t - self.k + 1 : t + 1])
            x_present = source[t]

            joint_target[(y_future, y_hist)] += 1
            count_target += 1

            joint_both[(y_future, y_hist, x_present)] += 1
            count_both += 1

        if count_target == 0 or count_both == 0:
            return 0.0

        # Compute conditional entropies
        # H(Y_{t+1} | Y_t^{(k)})
        h_target = self._conditional_entropy_target(joint_target, count_target)
        # H(Y_{t+1} | Y_t^{(k)}, X_t)
        h_both = self._conditional_entropy_both(joint_both, count_both)

        te = h_target - h_both
        return max(0.0, te)  # TE should be non-negative

    def _conditional_entropy_target(
        self, joint: dict, count: int
    ) -> float:
        """Compute H(Y_{t+1} | Y_t^{(k)}) from joint distribution."""
        # Group by y_hist
        hist_groups: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for (y_future, y_hist), c in joint.items():
            hist_groups[y_hist][y_future] += c

        h = 0.0
        for _y_hist, outcomes in hist_groups.items():
            hist_count = sum(outcomes.values())
            p_hist = hist_count / count
            # H(Y | this_hist) = -sum(p(y|hist) * log(p(y|hist)))
            h_given = 0.0
            for _y_future, c in outcomes.items():
                p_y = c / hist_count
                if p_y > 0:
                    h_given -= p_y * math.log2(p_y)
            h += p_hist * h_given
        return h

    def _conditional_entropy_both(
        self, joint: dict, count: int
    ) -> float:
        """Compute H(Y_{t+1} | Y_t^{(k)}, X_t) from joint distribution."""
        # Group by (y_hist, x_present)
        context_groups: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for (y_future, y_hist, x_present), c in joint.items():
            context_groups[(y_hist, x_present)][y_future] += c

        h = 0.0
        for _context, outcomes in context_groups.items():
            ctx_count = sum(outcomes.values())
            p_ctx = ctx_count / count
            h_given = 0.0
            for _y_future, c in outcomes.items():
                p_y = c / ctx_count
                if p_y > 0:
                    h_given -= p_y * math.log2(p_y)
            h += p_ctx * h_given
        return h

    def directed_information(
        self,
        source: list[int],
        target: list[int],
    ) -> float:
        """Compute directed information I(X → Y).

        Directed information is the sum of transfer entropy over all time steps.
        For stationary processes, this equals the transfer entropy rate.
        """
        return self.transfer_entropy(source, target)

    def transfer_entropy_matrix(
        self,
        signals: dict[str, list[int]],
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise transfer entropy for all signal pairs.

        Args:
            signals: Dict mapping signal_name -> time_series.

        Returns:
            Dict mapping (source, target) -> TE value.
        """
        result = {}
        names = sorted(signals.keys())
        for src_name in names:
            for tgt_name in names:
                if src_name == tgt_name:
                    continue
                te = self.transfer_entropy(signals[src_name], signals[tgt_name])
                result[(src_name, tgt_name)] = te
        return result

    def causal_chains(
        self,
        signals: dict[str, list[int]],
        threshold: float = 0.01,
    ) -> list[list[str]]:
        """Detect causal chains in the transfer entropy graph.

        Finds paths where information flows: A → B → C → ...
        Chains are sequences where each consecutive pair has TE > threshold.

        Args:
            signals: Dict mapping signal_name -> time_series.
            threshold: Minimum TE to consider an edge present.

        Returns:
            List of causal chains (each is a list of signal names).
        """
        # Build adjacency from TE matrix
        te_matrix = self.transfer_entropy_matrix(signals)
        adjacency: dict[str, set[str]] = defaultdict(set)
        for (src, tgt), te_val in te_matrix.items():
            if te_val > threshold:
                adjacency[src].add(tgt)

        # Find all paths via DFS
        chains: list[list[str]] = []
        visited: set[str] = set()

        def dfs(node: str, path: list[str]):
            if len(path) >= 2:
                chains.append(list(path))
            for neighbor in sorted(adjacency.get(node, set())):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
                    visited.discard(neighbor)

        for node in sorted(adjacency):
            visited.add(node)
            dfs(node, [node])
            visited.discard(node)

        return chains

    def byte_to_edge_flow(
        self,
        input_bytes: list[bytes],
        edge_bitmaps: list[bytes],
        map_size: int = 65536,
        max_positions: int = 64,
    ) -> dict[int, float]:
        """Compute transfer entropy from byte positions to coverage edges.

        For each byte position, computes TE(position → dominant_edges).
        This reveals which byte positions causally influence which code regions.

        Args:
            input_bytes: List of input byte sequences (same length).
            edge_bitmaps: List of corresponding edge bitmaps.
            map_size: Edge bitmap size.
            max_positions: Maximum byte positions to analyze.

        Returns:
            Dict mapping byte_position -> TE to coverage (in bits).
        """
        n = min(len(input_bytes), len(edge_bitmaps))
        if n < 3:
            return {}

        result = {}
        for pos in range(min(max_positions, min(len(b) for b in input_bytes if b))):
            # Source: byte value at position pos over time
            source = [b[pos] if pos < len(b) else 0 for b in input_bytes[:n]]
            # Target: dominant edge index (most-hit edge) at each step
            target = []
            for eb in edge_bitmaps[:n]:
                if eb:
                    # Find the most-hit edge
                    max_edge = 0
                    max_val = 0
                    for i, v in enumerate(eb[:map_size]):
                        if v > max_val:
                            max_val = v
                            max_edge = i
                    target.append(max_edge)
                else:
                    target.append(0)

            te = self.transfer_entropy(source, target)
            if te > 0:
                result[pos] = te

        return result

    def edge_to_edge_flow(
        self,
        edge_bitmaps: list[bytes],
        map_size: int = 65536,
        top_k: int = 10,
    ) -> dict[tuple[int, int], float]:
        """Compute transfer entropy between the top-k most-hit edges.

        Reveals causal chains in code execution: "hitting edge A tends to
        lead to hitting edge B in the next execution."

        Args:
            edge_bitmaps: List of edge bitmaps over time.
            map_size: Edge bitmap size.
            top_k: Only analyze edges with highest total hit count.

        Returns:
            Dict mapping (source_edge, target_edge) -> TE value.
        """
        n = len(edge_bitmaps)
        if n < 3:
            return {}

        # Find top-k edges by total hit count
        total_hits: dict[int, int] = defaultdict(int)
        for eb in edge_bitmaps:
            for i, v in enumerate(eb[:map_size]):
                if v > 0:
                    total_hits[i] += v
        top_edges = sorted(total_hits.keys(), key=lambda e: total_hits[e], reverse=True)[:top_k]

        if not top_edges:
            return {}

        # Build time series for each edge (binary: hit or not)
        edge_series: dict[int, list[int]] = {
            e: [1 if e < len(eb) and eb[e] > 0 else 0 for eb in edge_bitmaps]
            for e in top_edges
        }

        # Compute pairwise TE
        result = {}
        for src_edge in top_edges:
            for tgt_edge in top_edges:
                if src_edge == tgt_edge:
                    continue
                te = self.transfer_entropy(edge_series[src_edge], edge_series[tgt_edge])
                if te > 0:
                    result[(src_edge, tgt_edge)] = te

        return result

    def save(self, path: str) -> bool:
        """Save configuration."""
        import json
        try:
            with open(path, "w") as f:
                json.dump({"k": self.k, "n_bins": self.n_bins}, f)
            return True
        except OSError:
            return False

    def load(self, path: str) -> bool:
        """Load configuration."""
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            self.k = data.get("k", self.k)
            self.n_bins = data.get("n_bins", self.n_bins)
            return True
        except (OSError, json.JSONDecodeError):
            return False
