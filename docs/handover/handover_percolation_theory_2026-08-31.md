# Handover — Percolation Theory Applied to Coverage-Guided Fuzzing

**Date:** 2026-08-31 (updated 2026-09-01)
**Base:** `fuzzer-new`
**Status: Modules 1 (Bootstrap Percolation) and 2 (Coverage Phase Transition
Detection) IMPLEMENTED.** The remaining modules (3, 4, 5, 6) are still
planning proposals. Each section states what exists today, what percolation
adds, and — for Modules 1 and 2 — what was built and how to verify it.

---

## 0 Rule 1.

The file where all the percolation primitives live is `core/percolation.py`.


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

The six modules below make this framing operational. Module 2 is live;
modules 1, 3, 4, 5, 6 remain as proposed designs.

---

## 2. Module 1: Bootstrap Percolation Corpus Minimization — ✅ IMPLEMENTED

### What exists

`CorpusManager.auto_minimize_corpus()` (`services/corpus_manager.py:742`)
already does greedy set cover: iteratively remove seeds whose edge coverage
is fully subsumed by the remaining corpus. `EdgeTracker._maybe_prune()`
(`core/edge_tracker.py:901`) evicts cheapest-first by unique edge loss. The
system is already "percolation-aware" in spirit but not in structure.

### What was built

**New file:** `src/fuzzer_tool/core/percolation.py`

`bootstrap_minimize_corpus()` iteratively removes seeds with fewer than `k`
singleton edges (edges covered by no other seed in the corpus). After each
removal round, unique-edge counts are recomputed and the process repeats
until no seed changes state. The result is the **k-rigid core** — the
smallest corpus where every seed has at least k singleton edges.

This captures transitive redundancy that single-pass greedy set-cover
misses. Example: A={1,2}, B={2,3}, C={3,4}, D={4,5}. Greedy set-cover
keeps A, C, D (each has a unique edge). Bootstrap removes B and C in
round 1 (0 unique edges each), leaving A and D — a smaller corpus with
the same coverable edges.

```python
def bootstrap_minimize_corpus(corpus, edge_tracker, k=1):
    """Iteratively remove seeds with < k unique edges to fixed point."""
```

**Integration:** Post-pass in `auto_minimize_corpus()` after the existing
greedy reduction and coverage recovery. Controlled by `--bootstrap` flag
(default off) and `--bootstrap-k` (default 1).

**Tests:** `tests/test_percolation.py` — 10 tests covering transitive
redundancy, k-value filtering, edge cases, idempotency, and the
`CoverageRegime` enum import.

### Files touched

| File | Change |
|---|---|
| `src/fuzzer_tool/core/percolation.py` | **New** — `bootstrap_minimize_corpus()` + `CoverageRegime` enum |
| `src/fuzzer_tool/services/corpus_manager.py` | **Modify** — post-pass call in `auto_minimize_corpus()` |
| `src/fuzzer_tool/services/fuzzer.py` | **Modify** — `_use_bootstrap` / `_bootstrap_k` init |
| `src/fuzzer_tool/cli/commands.py` | **Modify** — `--bootstrap` / `--bootstrap-k` CLI args |
| `tests/test_percolation.py` | **New** — 10 unit tests |

---

## 3. Module 2: Coverage Phase Transition Detection — ✅ IMPLEMENTED

### What exists (pre-implementation)

| Component | Location | Purpose |
|---|---|---|
| `CriticalSlowingDown` | `core/critical_slowing.py:51` | Tracks discovery-rate variance/autocorrelation/skewness; detects "approaching transition" |
| `CoverageHomogeneityDetector` | `core/critical_slowing.py:233` | Chi-squared spatial uniformity test on per-column coverage |
| Stall recovery | `fuzzer.py:4747` | Triggers reseed when no new edges for `_stall_threshold` execs |
| Allan variance detector | `fuzzer.py:5701` | Tracks per-interval edge delta |
| CSD signal emission | `stats.py:583-586` | Logs `[CSD: variance ...]` to stats line — informational only |

### What was built

**New file:** `src/fuzzer_tool/core/coverage_regime.py`

`CoverageRegimeDetector` wraps the existing CSD, homogeneity, and stall
detectors into a single phase classifier:

```python
class CoverageRegime(enum.Enum):
    SUBCRITICAL = "subcritical"    # exponential decay; fuzzer stuck
    CRITICAL = "critical"          # power-law; near transition
    SUPERCRITICAL = "supercritical"  # compounding; healthy exploration

class CoverageRegimeDetector:
    def observe(self, discovery_rate, allan_delta, homogeneity_result,
                execs_since_edge, exec_count) -> CoverageRegime: ...
    @property
    def actionable(self) -> bool: ...     # True only on regime transitions
    def acknowledge(self) -> None: ...    # consume the signal
    def save(self) -> dict: ...           # for resume persistence
    def load(self, data) -> None: ...
```

**Classification logic (precedence order):**
1. `execs_since_edge >= stall_threshold` → SUBCRITICAL
2. CSD `is_approaching_transition()` fires → CRITICAL
3. Homogeneity rejects uniform (clustered coverage) → SUBCRITICAL
4. Otherwise → SUPERCRITICAL

**Key design decisions:**
- The detector **reads** the existing detectors' state but does **not** re-feed
  them. `CriticalSlowingDown` is fed by the stats reporter (`stats.py:583`);
  `CoverageHomogeneityDetector` is fed by the main loop (`fuzzer.py:5735`).
  Double-feeding would advance their windows out of lockstep.
- `actionable` is `True` only on the first observation of a new regime, preventing
  re-triggering on every tick. The loop calls `.acknowledge()` after acting.
- `save()` persists CSD state + regime history; homogeneity is rebuilt from
  replay buffer on resume (its sliding window is transient by design).

### Integration

**`src/fuzzer_tool/services/fuzzer.py`:**

```python
# __init__ (line ~1740):
self._regime = CoverageRegimeDetector(
    csd=self._csd, homogeneity=self._homogeneity,
    stall_threshold=self._stall_threshold,
)

# Main loop stats block (line ~5770):
self._regime.observe(
    discovery_rate=self._stats.discovery_rate(),
    allan_delta=delta,
    homogeneity_result=homogeneity_result,
    execs_since_edge=execs_since_edge,
    exec_count=self.exec_count,
)
if self._regime.actionable:
    if regime is CoverageRegime.SUBCRITICAL:
        if not self._stall_recovery_active:
            self._maybe_trigger_stall_recovery(execs_since_edge)
        # Bump havoc energy up to 5x
    elif regime is CoverageRegime.CRITICAL:
        pass  # preserve strategy — CSD means we're near a jump
    elif regime is CoverageRegime.SUPERCRITICAL:
        self._stall_recovery_active = False
    self._regime.acknowledge()
```

**`src/fuzzer_tool/services/corpus_manager.py`:**

```python
# save_state():
store.set("regime", f._regime.save())

# load_state():
if regime_data is not None and hasattr(f, "_regime"):
    f._regime.load(regime_data)
```

### Tests

**`tests/test_coverage_regime.py`** — 12 tests, all green:
- Stall threshold → SUBCRITICAL
- Actionable fires once per transition, consumed by acknowledge()
- Healthy exploration → SUPERCRITICAL
- Homogeneity rejection → SUBCRITICAL
- CSD firing → CRITICAL
- Save/load round-trip preserves history
- `test_critical_preserves_strategy`, `test_supercritical_resets_stall_state`,
  `test_subcritical_invokes_stall_recovery` verify the strategy branches

### How to verify on a live run

```sh
python3 -m fuzzer_tool fuzz --target targets/png_read.so \
    --iters 10000 --corpus /tmp/regime_test
```

Expected log lines:
```
REGIME: supercritical — healthy compounding
REGIME: critical — approaching transition (productive): variance Xx, ...
REGIME: subcritical — stall (N execs without new edge)
REGIME: subcritical — clustered coverage (χ²=..., p=...)
```

Resume test: Ctrl+C mid-run, then `--resume`; verify `regime` and `history`
are restored from `state.json`.

### Files touched

| File | Change |
|---|---|
| `src/fuzzer_tool/core/coverage_regime.py` | **New** — detector class |
| `tests/test_coverage_regime.py` | **New** — 12 unit + integration tests |
| `src/fuzzer_tool/services/fuzzer.py` | Instantiate, feed, wire strategy actions |
| `src/fuzzer_tool/services/corpus_manager.py` | Persist/restore on save/load |

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

### Implementation plan

**New file:** `src/fuzzer_tool/core/target_difficulty.py`

```python
def estimate_percolation_threshold(elf_path: str) -> dict:
    """Static estimate of p_c from the target's CFG."""
```

**Integration point:** Called at fuzzer startup. Drives time budget, initial
corpus size, and operator preselection.

**First step:** Extract `⟨k⟩` from existing targets using `objdump -d` and
rank by estimated `p_c`.

---

## 5. Module 4: Invasion Percolation for Operator Selection

### What exists

The Elo meta-scheduler (`_register_arms`) already arbitrates between
operator arms. No existing framework treats operator selection as a
percolation process.

### What percolation adds

Invasion percolation grows a cluster by always adding the bond with the
lowest resistance — greedy, no backtracking. For the fuzzer: always select
the operator whose resistance (inverse success rate) is lowest at the
current frontier. When all operators exceed a resistance threshold, the
cluster is stuck — trigger a reseed or strategy switch.

### Implementation plan

**Extend:** `services/seed_picker.py` with a resistance-aware wrapper.

```python
def invasion_select(operator_stats: dict, frontier_edges: set) -> str:
    """Select the operator with lowest resistance on the current frontier."""
```

**Integration point:** Used inside the Elo scheduler as an additional signal.

---

## 6. Module 5: First Passage Percolation for Time Budgeting

### What exists

The fuzzer runs for a fixed number of iterations (`--iters`) or timeout.
No per-target time optimization.

### What percolation adds

First passage percolation assigns random weights (times) to bonds and asks:
how long until the cluster reaches a given point? For the fuzzer:
estimate time to reach the next uncovered region. If high, allocate more
time. If low, move on.

### Implementation plan

**Extend:** `percolation.py` (from Module 1) with time estimation.

```python
def estimate_time_to_next_discovery(
    edge_tracker, operator_stats, coverage_regime
) -> float:
    """Expected executions to reach the next uncovered edge."""
```

**Integration point:** Used in multi-target fuzzing to dynamically allocate
time budgets.

---

## 7. Module 6: Universality → Strategy Transfer

### What exists

Operator weights and scheduler parameters are learned per-target and
discarded. No cross-target transfer.

### What percolation adds

The universality principle states that critical exponents depend only on
dimension `d`, not on lattice details. For the fuzzer: strategy
effectiveness should depend only on abstract properties (branching factor,
clustering), not on the specific binary. Learn operator weights on an
easy target, then **transfer** the learned percolation dynamics to a
hard target.

### Implementation plan

**New file:** `src/fuzzer_tool/core/strategy_transfer.py`

```python
def transfer_strategy(source_profile: dict, target_profile: dict,
                      source_weights: dict) -> dict:
    """Transfer operator weights from source to target."""
```

**Integration point:** At fuzzer startup, pre-initialize operator weights
from a structurally similar target.

---

## 8. Recommended implementation order

| Order | Module | Status | Effort | Payoff | Dependency |
|-------|--------|--------|--------|--------|------------|
| 1 | Bootstrap minimization | **DONE** | Low | Medium | None |
| 2 | Phase transition detection | **DONE** | Medium | High | None |
| 3 | Target difficulty estimation | Proposed | Medium | Medium | None |
| 4 | Invasion percolation | Proposed | Low | Low-Medium | Module 2 |
| 5 | First passage time budgeting | Proposed | Medium | Medium | Modules 2, 4 |
| 6 | Universality strategy transfer | Proposed | High | High (long-term) | Modules 3, 4 |

Modules 1 and 2 are complete. Module 3 (Target Difficulty Estimation) is
independent and can be developed next. Module 4 builds on Module 2's
regime signal.

---

## 9. What could falsify this framing

1. **The coverage graph is not random.** If the program's state-space
   graph has heavy-tailed degree distribution (scale-free), the Erdős-Rényi
   `p_c = 1/⟨k⟩` formula does not apply. Check: plot the degree distribution
   of the coverage graph for a real target. If it's heavy-tailed, Modules
   3-6 need different formulas (Molloy-Reed for configuration models).

2. **Bootstrap minimization removes useful diversity.** If the rigid core
   preserves coverage edges but loses behavioral diversity, the fuzzer
   loses ability to combine paths. Check: compare crash-finding rate of
   minimized corpus vs. full corpus.

3. **Phase transition detection has too much noise.** If coverage
   discovery is inherently bursty (one input unlocks 50 edges at once),
   the exponential decay signal may be undetectable above noise. Check:
   measure the coefficient of variation of inter-discovery times. If
   CV >> 1, the Poisson assumption breaks.

---

## 10. Files referenced

| File | Relevance |
|---|---|
| `src/fuzzer_tool/core/percolation.py` | **NEW** — `bootstrap_minimize_corpus()` + `CoverageRegime` enum |
| `src/fuzzer_tool/core/coverage_regime.py` | `CoverageRegimeDetector` (imports `CoverageRegime` from percolation) |
| `tests/test_percolation.py` | **NEW** — 10 tests for bootstrap percolation |
| `tests/test_coverage_regime.py` | 12 tests for the regime detector |
| `src/fuzzer_tool/services/fuzzer.py:1740` | Regime detector instantiation |
| `src/fuzzer_tool/services/fuzzer.py:5770` | Main-loop feeding + strategy wiring |
| `src/fuzzer_tool/services/corpus_manager.py:1096` | Bootstrap post-pass in `auto_minimize_corpus()` |
| `src/fuzzer_tool/services/corpus_manager.py:290` | `save_state()` persistence |
| `src/fuzzer_tool/services/corpus_manager.py:396` | `load_state()` restore |
| `src/fuzzer_tool/core/critical_slowing.py:51` | `CriticalSlowingDown` — wrapped by regime detector |
| `src/fuzzer_tool/core/critical_slowing.py:233` | `CoverageHomogeneityDetector` — wrapped by regime detector |
| `src/fuzzer_tool/core/edge_tracker.py:901` | `_maybe_prune()` — existing eviction logic |
| `src/fuzzer_tool/services/seed_picker.py` | Integration point for invasion percolation |
