# DAG math in the fuzzer: inventory, ranked by value

**Status:** survey only, no code changed. Ranking = measured evidence first
(`docs/AB_LEDGER.md`), then wiring depth, then leverage.

## Core property

Acyclic ⇒ topological order ⇒ one pass is exact and iteration count is
bounded by depth. Every item below cashes in a variant of that.

## Ranked findings

| # | Area | Module | Wired | Evidence |
|---|------|--------|-------|----------|
| 1 | Katz seed scheduling | `core/horizon.py`, `core/schedulers/seed_katz.py`, `services/katz_channel.py` | yes | #71 WIN (wall, fuzzgoat), #48/#78 LEAN+ |
| 2 | Coupled field repair | `core/field_constraints.py`, `core/structural_constraints.py` | yes (`services/operators.py`) | no A/B |
| 3 | Lineage forest | `core/lineage.py` | yes (`seed_picker`, `stats`, `report`, `seed_mcts`) | no A/B |
| 4 | Dominator gating | `core/dominators.py` | opt-in `--gate-bonus`, default 0 | no A/B |
| 5 | Precedence job ordering | `core/job_scheduling.py` | yes (`maintenance`, `seed_picker`) | no A/B |
| 6 | ICFG min-cut / centrality | `core/mincut.py`, `core/centrality.py` | only via `core/icfg.py` | no A/B |
| 7 | Seed poset height/width | `core/poset.py` | **no** (only `tools/edge_matrix_modes.py`) | none |

### 1. Katz centrality over the horizon DAG
- Tarjan SCC contracts the unvisited ICFG; intra-SCC edges dropped ⇒ DAG.
- Katz fixed point reached exactly at round `depth` (no path longer than depth).
- One Kahn walk proves acyclicity and yields `depth`. Replaced fixed 30
  rounds: 5e-2 abs error on a depth-59, 100k-node graph.
- Cycle ⇒ raises, instead of divergent scores poisoning seed energy.

### 2. Topological field repair
- Lengths/CRCs/offsets form a dependency graph; topo order makes all
  constraints hold in one pass. Piecewise repair does not converge.
- Nested TLV: all levels fixed innermost-first; parser no longer rejects
  at the outer frame.
- Only cycles (self-covering checksum) reach z3.

### 3. Lineage forest
- Subtree-weight pruning, `batch_lca_distances` diversity term in seed
  scoring, causal crash-chain replay for tmin, Strahler branch ranking.

### 4. Dominators
- Mandatory gates to a target, which BFS distance conflates with "near".
- Semi-NCA: ~20 ms vs ~40 s (CHK) on irreducible 4096-block CFG.

### 5. Job ordering
- EDD/MDD under precedence; acyclicity check guarantees a ready job.

### 6. Min-cut / centrality
- Frontier→target min edge cut (Menger); Brandes betweenness/closeness.
  Not reaching a scheduler beyond ICFG construction.

### 7. Poset
- Containment order on seed edge sets; width via Dilworth (n − max
  matching) = corpus diversity. Unwired.

## Next steps
- [ ] A/B `--gate-bonus` > 0 vs 0 on a directed fuzzgoat target (ledger row).
- [ ] A/B field repair on/off on PNG target.
- [ ] Decide wire-or-remove for `core/poset.py` (Hard Rule 7).
- [ ] Trace whether min-cut/centrality output reaches any consumer.
