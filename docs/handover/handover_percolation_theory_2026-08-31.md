# Handover — Percolation Theory Applied to Coverage-Guided Fuzzing

**Date:** 2026-08-31
**Base:** `fuzzer-new`
**Status: PLANNING ONLY. No code changed.** This maps percolation theory onto the
fuzzer's state-space exploration and proposes six concrete implementation
modules, ordered by actionability. Each section states what exists today,
what percolation adds, and a concrete first step.

---

## 1. The framing: fuzzing as a percolation process

A fuzzer explores a program's state-space graph `G = (V, E)` where vertices
are program states and edges are transitions (branches). Each input is a
"bond" that, with some probability `p`, opens a path to new coverage. The
set of reachable states at time `t` is a cluster. The fuzzer's job is to
grow that cluster until it **percolates** — reaches a spanning component
covering all reachable regions.

Percolation theory tells us this process has a **phase transition** at
threshold `p_c`:
- `p < p_c` (subcritical): small isolated clusters, exponential decay of
  reach. The fuzzer is stuck — each new input discovers only a tiny island.
- `p > p_c` (supercritical): giant connected cluster. The fuzzer is
  compounding — each input unlocks multiple new paths.
- `p = p_c` (critical): power-law behavior, maximum sensitivity. This is
  where the fuzzer is most responsive to small changes in strategy.

The six modules below make this framing operational.

---

## 2. Module 1: Bootstrap Percolation Corpus Minimization

### What exists

`CorpusManager.auto_minimize_corpus()` (`services/corpus_manager.py:735`)
already does greedy set cover: iteratively remove seeds whose edge coverage
is fully subsumed by the remaining corpus. `EdgeTracker._maybe_prune()`
(`core/edge_tracker.py:901`) evicts cheapest-first by unique edge loss. The
system is already "percolation-aware" in spirit but not in structure.

### What percolation adds

Bootstrap percolation (`wikipedia/Bootstrap_percolation`) formalizes this:
a seed is removed if it has fewer than `k` "active" neighbors, iterated until
fixed point. The remaining set is the **rigid core** — the minimal corpus
that cannot be further reduced without breaking connectivity.

The existing greedy set cover is single-pass and greedy. Bootstrap
percolation is iterative and captures transitive redundancy: seed A
subsumes B, B subsumes C, but A does not directly subsume C. Greedy keeps
A and C; bootstrap removes both B and C once A absorbs the cluster.

### Implementation plan

**New file:** `src/fuzzer_tool/core/percolation.py`

```python
def bootstrap_minimize_corpus(corpus, edge_tracker, k=2):
    """Iteratively remove seeds subsumed by >=k neighbors.

    A seed is "active" (kept) if it owns an edge no other active seed owns.
    Otherwise it becomes "inactive" (removed). Iterate until no seed
    changes state. What remains is the rigid core — the smallest corpus
    that preserves the full coverage graph's connected components.

    Returns: (kept_seeds, removed_seeds)
    """
```

**Integration point:** Call from `auto_minimize_corpus()` as a post-pass
after the existing greedy reduction. The greedy pass reduces `O(n)` to
`O(edges)`; bootstrap cleans up the residual redundancy in `O(iterations × edges)`.

**Test:** `tests/test_percolation.py` — construct a known graph where
greedy keeps 3 seeds but bootstrap correctly reduces to 2, assert the
smaller set preserves cumulative coverage.

**First step:** Implement `bootstrap_minimize_corpus()` with a fixed `k=2`
on the existing `EdgeTracker.seed_edges` dict, validate on a live corpus
from a `png_read` run.

---

## 3. Module 2: Coverage Phase Transition Detection

### What exists

`EdgeTracker.record_edges()` tracks per-seed discoveries.
`compute_coverage_proximity()` (`core/edge_tracker.py:1633`) measures how
close a seed is to the current frontier. No existing code detects whether
the fuzzer is in subcritical vs. supercritical regime.

### What percolation adds

In the subcritical regime, the probability of reaching distance `r` from
any origin decays exponentially. For the fuzzer, this means new-coverage
events follow exponential decay over time — each generation discovers fewer
edges than the last, and the rate itself is dropping. In the supercritical
regime, coverage compounds — each discovery unlocks multiple new paths.

Detecting which regime the fuzzer is in enables **automated strategy
switching**: if subcritical, increase mutation intensity, switch to
grammar-aware mode, or reseed. If supercritical, stay the course.

### Implementation plan

**Extend:** `EdgeTracker` with a regime tracker.

```python
class CoverageRegime:
    """Detects subcritical vs. supercritical coverage phase."""

    def update(self, new_edges: int, timestamp: float) -> None:
        """Record a coverage discovery event."""

    @property
    def regime(self) -> Literal["subcritical", "critical", "supercritical"]:
        """Classify current phase from discovery rate decay."""

    @property
    def p_estimate(self) -> float:
        """Estimated percolation probability (fraction of 'open' bonds)."""
```

**Algorithm:** Fit the inter-discovery time series to exponential decay.
- If decay constant `λ > threshold` → subcritical (exponential decay)
- If decay constant `λ ≈ 0` → supercritical (constant or accelerating)
- If power-law tail → critical (most sensitive, smallest changes matter)

**Integration point:** The main fuzzing loop (`Fuzzer.run()`) checks
`regime` every N executions and adjusts strategy. Emit a log warning on
subcritical detection — it's the signal that the fuzzer is wasting time
with the current operator set.

**First step:** Log the ratio `edges_this_window / edges_last_window` over
a campaign and plot it. If it decays exponentially, the signal is real and
worth automating.

---

## 4. Module 3: Target Difficulty Estimation via `p_c` Formulas

### What exists

The fuzzer has no pre-fuzz static analysis. All targets get the same
default strategy and time budget.

### What percolation adds

For an Erdős-Rényi random graph, `p_c = 1/⟨k⟩` where `⟨k⟩` is the average
degree. For clustered networks, `p_c = 1/((1-C) · g_1'(1))` where `C` is
the clustering coefficient. Higher `p_c` = harder to percolate = harder
to fuzz.

Estimate these statically from the target binary's CFG:
- `⟨k⟩` ≈ average branching factor per basic block
- `C` ≈ density of redundant paths (multiple paths to same block)

Targets with high `p_c` (deeply nested conditionals, many redundant paths)
need more time, more seeds, and grammar-aware mutation. Targets with low
`p_c` (linear parsers, flat CFG) can be fuzzed with random mutation and
shorter runs.

### Implementation plan

**New file:** `src/fuzzer_tool/core/target_difficulty.py`

```python
def estimate_percolation_threshold(elf_path: str) -> dict:
    """Static estimate of p_c from the target's CFG.

    Returns:
        avg_branching: float — mean out-degree per BB
        clustering: float — fraction of BBs reachable by >=2 paths
        p_c_estimate: float — estimated percolation threshold
        difficulty: Literal["low", "medium", "high", "extreme"]
    """
```

**Integration point:** Called at fuzzer startup before the first
execution. Drives:
1. Time budget allocation (high `p_c` → longer runs)
2. Initial corpus size (high `p_c` → more seeds)
3. Operator preselection (high `p_c` → grammar-aware from the start)

**First step:** Extract `⟨k⟩` from the `.so` targets already in `targets/`
(png, jpeg, zlib, lz4, grep, gzip) using `objdump -d` and a basic-block
parser. Rank them by estimated `p_c` and see if the ranking matches
observed fuzzing difficulty.

---

## 5. Module 4: Invasion Percolation for Operator Selection

### What exists

The Elo meta-scheduler (`_register_arms`) already arbitrates between
operator arms. `SeedPicker._pick_boltzmann_seed()` selects seeds by
energy. No existing framework treats operator selection as a percolation
process.

### What percolation adds

Invasion percolation (`wikipedia/Invasion_percolation`) grows a cluster
by always adding the bond with the lowest resistance — greedy, no
backtracking, no global view. It naturally finds the path of least
resistance through a random medium.

For the fuzzer, this means: always select the operator whose
"resistance" (inverse success rate) is lowest **at the current frontier**.
This is already approximately what the Elo scheduler does. Framing it as
invasion percolation gives a theoretical bound on when the process gets
stuck: when the cluster is fully surrounded by high-resistance bonds
(i.e., every operator's resistance exceeds a threshold).

### Implementation plan

**Extend:** `services/seed_picker.py` with a resistance-aware wrapper.

```python
def invasion_select(operator_stats: dict, frontier_edges: set) -> str:
    """Select the operator with lowest resistance on the current frontier.

    Resistance = (trials - successes) / trials for edges adjacent to
    frontier. If no operator has resistance < threshold, signal that the
    cluster is stuck (all bonds are high-resistance) — trigger a reseed
    or strategy switch.
    """
```

**Integration point:** Used inside the Elo scheduler as an additional
signal: when invasion percolation reports "stuck," the scheduler injects
a random seed or switches to grammar-aware mode.

**First step:** Log operator resistance per-window during a campaign.
Plot the resistance trajectory — if it plateaus, the cluster is stuck
and the framework is validated.

---

## 6. Module 5: First Passage Percolation for Time Budgeting

### What exists

The fuzzer runs for a fixed number of iterations (`--iters`) or timeout.
No per-target time optimization.

### What percolation adds

First passage percolation (`wikipedia/First_passage_percolation`) assigns
random weights (times) to bonds and asks: how long until the cluster
reaches a given point?

For the fuzzer: estimate the time to reach the next uncovered region
given the current coverage frontier. If expected time is high, allocate
more time to this target. If low, move on.

### Implementation plan

**Extend:** `percolation.py` (from Module 1) with time estimation.

```python
def estimate_time_to_next_discovery(
    edge_tracker, operator_stats, coverage_regime
) -> float:
    """Expected number of executions to reach the next uncovered edge.

    Uses the current operator success rate and the number of edges
    adjacent to the frontier. Returns float('inf') if the cluster is
    stuck (subcritical with high-resistance operators).
    """
```

**Integration point:** Used in multi-target fuzzing to dynamically
allocate time budgets. Also feeds into Module 2's regime detection —
if estimated time is rising, the fuzzer is entering subcritical regime.

**First step:** Compute the empirical average time-to-discovery over a
campaign and compare it to the theoretical estimate from operator success
rates. If they correlate, the estimator is predictive.

---

## 7. Module 6: Universality → Strategy Transfer

### What exists

Operator weights and scheduler parameters are learned per-target and
discarded. No cross-target transfer.

### What percolation adds

The universality principle (`wikipedia/Universality_(dynamical_systems)`)
states that critical exponents depend only on dimension `d`, not on
lattice details. For the fuzzer: strategy effectiveness should depend
only on abstract properties of the target (branching factor, clustering)
not on the specific binary.

This means: learn operator weights on an easy target, then **transfer**
the learned percolation dynamics to a hard target. The hard target
doesn't need to learn from scratch — it inherits the dynamics from
a structurally similar easy target.

### Implementation plan

**New file:** `src/fuzzer_tool/core/strategy_transfer.py`

```python
def transfer_strategy(source_profile: dict, target_profile: dict,
                      source_weights: dict) -> dict:
    """Transfer operator weights from source to target.

    Scales weights by the ratio of percolation thresholds:
    weight_target = weight_source * (p_c_source / p_c_target).

    Higher p_c target → higher weights on exploratory operators
    (grammar, structure, havoc), lower on exploitative ones.
    """
```

**Integration point:** At fuzzer startup, if a profile exists for a
similar target, pre-initialize operator weights instead of starting
from uniform. The Elo scheduler then refines from a warm start.

**First step:** Profile two structurally similar targets (e.g., `png_read`
and `jpeg_read`) and compare learned operator weights. If they correlate
after normalizing by `p_c`, universality holds and transfer is viable.

---

## 8. Recommended implementation order

| Order | Module | Effort | Payoff | Dependency |
|-------|--------|--------|--------|------------|
| 1 | Bootstrap minimization | Low | Medium | None |
| 2 | Phase transition detection | Medium | High | None |
| 3 | Target difficulty estimation | Medium | Medium | None |
| 4 | Invasion percolation | Low | Low-Medium | Module 2 |
| 5 | First passage time budgeting | Medium | Medium | Modules 2, 4 |
| 6 | Universality strategy transfer | High | High (long-term) | Modules 3, 4 |

Modules 1-3 are independent and can be developed in parallel. Module 1
ships first because it's contained (new file + small integration), has
no upstream dependencies, and the existing `auto_minimize_corpus` provides
a direct baseline for A/B comparison.

---

## 9. What could falsify this framing

1. **The coverage graph is not random.** If the program's state-space
   graph has heavy-tailed degree distribution (scale-free), the Erdős-Rényi
   `p_c = 1/⟨k⟩` formula does not apply. Scale-free networks have
   `p_c → 0` for infinite size — meaning the fuzzer never hits a phase
   transition, it just slowly grinds. Check: plot the degree distribution
   of the coverage graph for a real target. If it's heavy-tailed, Modules
   3-6 need different formulas (use Molloy-Reed for configuration models).

2. **Bootstrap minimization removes useful diversity.** If the rigid core
   preserves coverage edges but loses behavioral diversity (e.g., seeds
   that are edge-equivalent but reach edges via different paths), the
   fuzzer loses ability to combine paths. Check: compare crash-finding
   rate of minimized corpus vs. full corpus on a target with known bugs.

3. **Phase transition detection has too much noise.** If coverage
   discovery is inherently bursty (one input unlocks 50 edges at once),
   the exponential decay signal may be undetectable above noise. Check:
   measure the coefficient of variation of inter-discovery times. If
   CV >> 1, the Poisson assumption breaks and the detector needs a
   different model.

---

## 10. First concrete step

```sh
# 1. Extract target CFGs and estimate p_c
python3 tools/estimate_pc.py targets/png_read.so targets/jpeg_read.so \
    targets/zlib_read.so targets/lz4_read.so targets/grep_read.so \
    targets/gzip_read.so

# 2. Build corpus for png_read (existing flow)
python3 -m fuzzer_tool fuzz --target targets/png_read.so \
    --iters 50000 --corpus ~/corpus_png

# 3. Run existing minimization, log size before/after
python3 -m fuzzer_tool minimize --corpus ~/corpus_png

# 4. Run bootstrap minimization on the same corpus, compare
python3 -m fuzzer_tool minimize --corpus ~/corpus_png --bootstrap

# 5. Validate: re-run minimized corpora, compare coverage
```

If bootstrap produces a smaller corpus with equal coverage, Module 1 is
validated and ships. If not, the greedy set cover is already near-optimal
and the percolation framing needs adjustment for this target class.

---

## 11. Files referenced

| File | Relevance |
|------|-----------|
| `src/fuzzer_tool/services/corpus_manager.py:735` | `auto_minimize_corpus()` — existing greedy set cover, integration point for bootstrap |
| `src/fuzzer_tool/core/edge_tracker.py:901` | `_maybe_prune()` — existing eviction logic |
| `src/fuzzer_tool/core/edge_tracker.py:689` | `record_edges()` — per-seed coverage tracking |
| `src/fuzzer_tool/services/seed_picker.py` | Boltzmann seed picker — integration point for invasion percolation |
| `src/fuzzer_tool/services/fuzzer.py` | Main fuzzing loop — integration point for phase detection |
| `tools/eval_set.py` | Target sets — where `direct_lite` lives, for p_c estimation |
| `docs/TODO.md` | Existing roadmap — items on collaborative scheduling and cost-based eviction relate to Modules 4-5 |
