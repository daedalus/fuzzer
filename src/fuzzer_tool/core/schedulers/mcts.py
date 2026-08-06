"""Monte Carlo Tree Search seed scheduling over the mutation lineage tree.

Every other seed strategy in ``seed_picker`` scores the corpus as a *flat
pool*: each seed gets a weight from its own metadata and the best-weighted
one wins. None of them exploit the fact that ``core/lineage.py`` already
maintains the full parent/child genealogy of how seeds were derived.

That structure carries information a flat score cannot express. A seed whose
own coverage looks unremarkable may sit at the root of a subtree that keeps
producing discoveries; conversely a seed that found a lot on its own may have
a subtree that has gone completely sterile. MCTS descends the tree by UCT,
so budget flows toward *regions* of the lineage that are still paying off
rather than toward individually high-scoring seeds.

Structure (standard UCT, adapted to an externally-grown tree):

- **Selection** — from each root, walk down by UCT until reaching a node with
  no visited children. Unlike textbook MCTS there is no expansion step: the
  fuzzer grows the tree for us via ``LineageTree.insert``, so a node's
  children appear on their own as mutations of it land in the corpus.
- **Simulation** — none. The rollout is the fuzzer actually executing the
  selected seed, which is far more informative than a random playout.
- **Backpropagation** — ``update()`` credits the selected node and all its
  ancestors, which is what makes a productive subtree lift its whole
  ancestor chain.

Reward is new edges found, squashed to [0, 1] so UCT's exploration constant
stays meaningful; raw edge counts vary by orders of magnitude between targets
and would otherwise swamp the exploration term.

Reference: Kocsis & Szepesvári, "Bandit based Monte-Carlo Planning" (ECML
2006) for UCT; Alphuzz (ACSAC 2021) for the application to seed scheduling.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuzzer_tool.core.lineage import LineageTree

# UCT exploration constant. sqrt(2) is the standard value for rewards in
# [0, 1], which is the range _squash produces.
DEFAULT_EXPLORATION = math.sqrt(2.0)

# Reward squashing scale: edges/exec is small and heavy-tailed, so a
# saturating transform keeps a single lucky seed from dominating the tree.
_REWARD_SCALE = 8.0

# Nodes start with one virtual visit so an unvisited node is not treated as
# infinitely promising forever; without this, UCT spends the early run
# cycling every leaf exactly once regardless of how bad they look.
_PRIOR_VISITS = 1.0
_PRIOR_VALUE = 0.5


def _squash(edges: float) -> float:
    """Map a raw new-edge count into [0, 1].

    Saturating rather than linear: the difference between finding 1 and 5 new
    edges matters much more than between 100 and 104, and UCT's exploration
    term is only calibrated when rewards are bounded. Very large counts reach
    exactly 1.0 once the exponential underflows, which is harmless — the
    bound is what UCT needs, not strict inequality.
    """
    if edges <= 0:
        return 0.0
    return 1.0 - math.exp(-edges / _REWARD_SCALE)


class MCTSSeedScheduler:
    """UCT over ``LineageTree``, selecting which corpus seed to fuzz next.

    Visit counts and values are held here rather than on ``LineageNode`` so
    the lineage tree stays a pure structural record with no scheduler state
    bolted onto it, and so this scheduler can be enabled or disabled without
    changing what lineage persists.
    """

    def __init__(
        self,
        exploration: float = DEFAULT_EXPLORATION,
        max_depth: int = 64,
        rng: random.Random | None = None,
    ):
        self.exploration = exploration
        self.max_depth = max_depth
        self._rng = rng or random.Random()

        # key -> accumulated visits / squashed reward, including priors.
        # Aggregate stats cover a node *and everything below it*, and drive the
        # descent. Self stats cover only the outcome of fuzzing that exact seed
        # and decide whether to stop at it — separating the two is what lets an
        # interior node be chosen over its own descendants.
        self.visits: dict[str, float] = {}
        self.values: dict[str, float] = {}
        self.self_visits: dict[str, float] = {}
        self.self_values: dict[str, float] = {}

        # The path from a root down to the node handed out by select(),
        # retained so update() can backpropagate without re-walking.
        self._last_path: list[str] = []
        self._last_selected: str | None = None

        self.selections = 0
        self.updates = 0

    # ── statistics ─────────────────────────────────────────────────────

    def _visits(self, key: str) -> float:
        return self.visits.get(key, _PRIOR_VISITS)

    def _value(self, key: str) -> float:
        """Mean squashed reward for *key*, prior-initialized."""
        return self.values.get(key, _PRIOR_VALUE * _PRIOR_VISITS) / self._visits(key)

    def _uct(self, key: str, parent_visits: float) -> float:
        """UCT score over the subtree rooted at *key* (drives the descent)."""
        exploit = self._value(key)
        explore = self.exploration * math.sqrt(
            math.log(max(parent_visits, 1.0)) / self._visits(key)
        )
        return exploit + explore

    def _self_visits(self, key: str) -> float:
        return self.self_visits.get(key, _PRIOR_VISITS)

    def _self_value(self, key: str) -> float:
        return self.self_values.get(key, _PRIOR_VALUE * _PRIOR_VISITS) / self._self_visits(key)

    def _self_uct(self, key: str, parent_visits: float) -> float:
        """UCT score for fuzzing *key* itself rather than descending past it."""
        exploit = self._self_value(key)
        explore = self.exploration * math.sqrt(
            math.log(max(parent_visits, 1.0)) / self._self_visits(key)
        )
        return exploit + explore

    # ── selection ──────────────────────────────────────────────────────

    def select(self, tree: LineageTree, eligible: set[str]) -> str | None:
        """Descend the lineage tree by UCT and return a seed key to fuzz.

        Args:
            tree: The lineage forest to walk.
            eligible: Keys currently backed by a live corpus entry. A node may
                outlive its seed (corpus minimization drops entries while the
                lineage node persists as a soft-deleted ancestor), so the walk
                may pass *through* an ineligible node but must not return one.

        Returns:
            A key from *eligible*, or None if the tree offers no reachable
            eligible node — the caller should fall back to another strategy.
        """
        if not eligible:
            return None

        roots = self._roots(tree, eligible)
        if not roots:
            return None

        total = sum(self._visits(r) for r in roots)
        node_key = max(roots, key=lambda k: self._uct(k, total))

        path = [node_key]
        best_eligible = node_key if node_key in eligible else None

        for _ in range(self.max_depth):
            children = [c for c in tree._children.get(node_key, ()) if c in tree.nodes]
            if not children:
                break
            parent_visits = self._visits(node_key)
            best_child = max(children, key=lambda k: self._uct(k, parent_visits))

            # Stopping here is a real action, not a fallback: every lineage
            # node is a corpus seed we could fuzz. If it did not compete
            # against its children the descent would always bottom out at a
            # leaf and interior seeds — including the imported roots — would
            # never be selected at all.
            if node_key in eligible and self._self_uct(node_key, parent_visits) >= self._uct(
                best_child, parent_visits
            ):
                best_eligible = node_key
                break

            node_key = best_child
            path.append(node_key)
            if node_key in eligible:
                best_eligible = node_key

        if best_eligible is None:
            # Walked into a region with no live seeds. Record the visit so UCT
            # deprioritizes this branch next time instead of looping on it.
            self._last_path = path
            self._last_selected = None
            self.update(0.0)
            return None

        # Trim the path at the returned node: crediting nodes below the seed
        # actually fuzzed would attribute the outcome to descendants that had
        # no part in it.
        self._last_path = path[: path.index(best_eligible) + 1]
        self._last_selected = best_eligible
        self.selections += 1
        return best_eligible

    def _roots(self, tree: LineageTree, eligible: set[str]) -> list[str]:
        """Root keys of the lineage forest, restricted to useful subtrees.

        A node is a root when its parent is absent from the tree — the
        forest is genuinely multi-rooted, since every imported corpus seed is
        inserted with ``parent_key=None``.
        """
        roots = []
        for key, node in tree.nodes.items():
            if node.parent_key is None or node.parent_key not in tree.nodes:
                roots.append(key)
        if not roots:
            # Malformed/cyclic parent pointers: fall back to eligible keys
            # present in the tree so selection still makes progress.
            roots = [k for k in eligible if k in tree.nodes]
        return roots

    # ── backpropagation ────────────────────────────────────────────────

    def update(self, new_edges: float) -> None:
        """Backpropagate the outcome of the last ``select()`` up the path.

        Crediting every ancestor is the mechanism that makes a productive
        subtree raise its whole chain: a discovery deep in the tree lifts the
        UCT score of the region containing it, so subsequent selections are
        drawn back toward it.
        """
        if not self._last_path:
            return
        reward = _squash(new_edges)
        for key in self._last_path:
            self.visits[key] = self._visits(key) + 1.0
            self.values[key] = self.values.get(key, _PRIOR_VALUE * _PRIOR_VISITS) + reward
        # Self stats go only to the seed actually fuzzed, so "is this node
        # worth stopping at" is learned from its own outcomes and not from
        # its descendants'.
        if self._last_selected is not None:
            key = self._last_selected
            self.self_visits[key] = self._self_visits(key) + 1.0
            self.self_values[key] = self.self_values.get(key, _PRIOR_VALUE * _PRIOR_VISITS) + reward
        self.updates += 1
        self._last_path = []
        self._last_selected = None

    # ── maintenance ────────────────────────────────────────────────────

    def prune(self, live_keys: set[str]) -> int:
        """Drop statistics for keys no longer in the tree.

        Corpus minimization removes seeds permanently; without this the two
        dicts grow for the whole run. Returns the number of entries dropped.
        """
        stale = [k for k in self.visits if k not in live_keys]
        for key in stale:
            self.visits.pop(key, None)
            self.values.pop(key, None)
        for key in [k for k in self.self_visits if k not in live_keys]:
            self.self_visits.pop(key, None)
            self.self_values.pop(key, None)
        return len(stale)

    def stats(self) -> dict:
        """Scheduler diagnostics for the stats line / convergence report."""
        tracked = len(self.visits)
        return {
            "selections": self.selections,
            "updates": self.updates,
            "tracked_nodes": tracked,
            "mean_value": (sum(self._value(k) for k in self.visits) / tracked if tracked else 0.0),
        }

    # ── persistence ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "visits": self.visits,
            "values": self.values,
            "self_visits": self.self_visits,
            "self_values": self.self_values,
            "selections": self.selections,
            "updates": self.updates,
        }

    def from_dict(self, data: dict) -> None:
        self.visits = {str(k): float(v) for k, v in data.get("visits", {}).items()}
        self.values = {str(k): float(v) for k, v in data.get("values", {}).items()}
        self.self_visits = {str(k): float(v) for k, v in data.get("self_visits", {}).items()}
        self.self_values = {str(k): float(v) for k, v in data.get("self_values", {}).items()}
        self.selections = int(data.get("selections", 0))
        self.updates = int(data.get("updates", 0))
        self._last_path = []
        self._last_selected = None
