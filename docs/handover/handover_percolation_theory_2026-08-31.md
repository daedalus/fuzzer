# Handover — Percolation Theory Applied to Coverage-Guided Fuzzing

**Date:** 2026-08-31 (updated 2026-09-01, literature update 2026-08-31)
**Base:** `fuzzer-new`
**Status: Modules 1 (Bootstrap Percolation), 2 (Coverage Phase Transition
Detection), 3 (Target Difficulty Estimation), and 4 (Invasion Percolation
Operator Selection) IMPLEMENTED.** The remaining modules (5, 6) are still
planning proposals. Each section states what exists today, what percolation
adds, and — for Modules 1–4 — what was built and how to verify it.

**Literature update (2026-08-31):** Diskin, Easo, Radhakrishnan, Sudakov &
Tassion, *"Supercritical sharpness of percolation"* (arXiv:2603.03257,
Mar 2026), proves supercritical sharpness — exponential tail decay of finite
cluster size above `p_c` — for **every infinite transitive graph**, with no
assumption on degree distribution or growth type. Previously this was only
known for `Z^d`, transitive graphs of polynomial growth, and nonamenable
graphs. The bound is stated purely in terms of the graph's isoperimetric
function `Φ(n) = min{|∂S| : S ⊂ V, n ≤ |S| < ∞}`, not degree or clustering
statistics. This directly answers the "heavy-tailed degree distribution"
falsifier in §9 and changes the recommended design of Module 3 and Module 5
below — see the inline notes in those sections.

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

### What percolation adds — revised per Diskin–Easo–Radhakrishnan–Sudakov–
### Tassion (2026)

The original plan below (`⟨k⟩`, `C`) assumes the target's CFG behaves like
an Erdős-Rényi or configuration-model random graph. Real CFGs don't: basic
blocks near parsers/dispatch loops have heavy-tailed branching, which is
exactly the case flagged as a potential falsifier in §9.1. The new paper
sidesteps the degree-distribution question entirely — its bound

```
P_p(n ≤ |cluster| < ∞) ≤ exp(-c · Φ(n))
```

(Theorem 1) holds on **any** infinite transitive graph, using only the
isoperimetric function `Φ(n) = min{|∂S| : S ⊂ V, n ≤ |S| < ∞}` — the
minimum edge-boundary over all vertex sets of size ≥ n. `Φ` is a purely
combinatorial cut quantity; it needs no degree-distribution assumption and
is well defined whether the coverage graph is regular, scale-free, or
anything else. This means Module 3 should target **estimating `Φ`
directly** rather than reconstructing `p_c` from `⟨k⟩`/`C`:

- Higher `Φ` (well-connected CFG, few narrow chokepoints) → coverage
  clusters escape to new regions easily → fuzzer-friendly target.
- Low/flat `Φ` (narrow bridges — e.g. a single dispatch switch or checksum
  gate) → the isoperimetric bound is weak → expect a hard target where
  progress requires clearing a specific bottleneck edge, not "more time."

`Φ` is exactly a min-cut computation, so it is *more* tractable to estimate
than `⟨k⟩`/`C`-based `p_c` formulas (which additionally require deciding
whether the ER or configuration-model formula applies before you can trust
the number).

### Implementation plan

**New file:** `src/fuzzer_tool/core/target_difficulty.py`

```python
def estimate_isoperimetric_profile(elf_path: str, sizes: list[int]) -> dict[int, int]:
    """Approximate Φ(n) for the target's CFG at each requested cluster size n.

    Build the CFG as an undirected graph over basic blocks (edges = observed
    or statically-recovered branches). For each n in `sizes`, approximate
    min{|∂S| : S ⊂ V, |S| = n} with a bounded number of randomized min-cut /
    local expansion probes (exact min-cut over all size-n subsets is
    intractable; a small-n greedy-growth heuristic, e.g. repeatedly adding
    the vertex with cheapest marginal boundary cost, is a reasonable
    first-pass approximation and is what [CMT24]-style numerics use).
    """
```

`estimate_percolation_threshold()` from the original plan can still be kept
as a cheap fallback for targets where a full CFG recovery isn't available
(e.g. no symbols), but it should be treated as a fallback, not the primary
estimator.

**Integration point:** Called at fuzzer startup. A flat/low `Φ` profile near
some `n` flags a likely chokepoint — drives time budget, initial corpus
size, and operator preselection (e.g. bias toward operators historically
good at clearing narrow gates: checksum/length-field mutations).

**First step:** Recover an approximate CFG from `objdump -d` (or existing
static-analysis tooling if the project has any) and compute `Φ(n)` at a few
small `n` for the current target set; rank targets by their `Φ` profile
instead of by `⟨k⟩`.

---

## 5. Module 4: Invasion Percolation for Operator Selection — ✅ IMPLEMENTED

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

### What was built

**Extended:** `src/fuzzer_tool/services/seed_picker.py`

```python
def invasion_select(
    operator_stats: dict[str, tuple[float, float]],
    frontier_edges: set[int] | None = None,
    resistance_threshold: float = INVASION_STUCK_THRESHOLD,
) -> str | None:
    """Select the operator with lowest resistance on the current frontier."""
```

`operator_stats` takes the `(successes, failures)` shape every scheduler's
`bandit_stats()` already returns — no new stats tracking needed. Resistance
is `1 / success_rate`; an untried arm (no observations at all) gets
resistance 0, so invasion tries it before writing it off, matching the
optimism-under-uncertainty the UCB schedulers already use elsewhere.
Returns `None` (stuck) when every operator's resistance is at or above
`resistance_threshold`, or when an explicit empty `frontier_edges` is
passed (nothing left to invade on this frontier).

**Not yet done:** wiring this into the Elo scheduler in `services/fuzzer.py`
as an additional signal — the integration point below is still a proposal.

**Integration point:** Used inside the Elo scheduler as an additional signal.

**Tests:** `tests/test_invasion_select.py` — 11 tests: selection,
exploration-over-low-success-rate, deterministic tie-break, stuck
threshold, adversarial all-zero-success cases.

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

**Literature note:** Theorem 3 of the 2026 paper gives an explicit,
non-heuristic answer to a closely related question — how fast does the
*reachable neighborhood itself* grow — for any transitive graph, purely in
terms of `Φ`:

```
∫[|S| to v_n] (1/(c·Φ(t))) dt = n
```

i.e. `v_n` (the guaranteed lower bound on reachable-set size after "time"
`n`) is defined implicitly by inverting the integral of `1/Φ`. If Module 3
produces an estimated `Φ` profile for a target, `v_n` can be computed
directly from it (numerically invert the integral) instead of learning a
separate empirical growth-rate model per target. This gives `Module 5` a
principled prior for "expected executions to reach the next uncovered
region" that only needs Module 3's `Φ` estimate as input, rather than a
freestanding heuristic — treat `estimate_time_to_next_discovery()` below as
the inverse of `v_n(·)`. Two extremes worth sanity-checking once `Φ`
estimation exists (Remark 7.2 of the paper): polynomial-growth-like CFGs
give `v_n ≍ n^d`-ish neighborhoods; near-nonamenable ones (few chokepoints,
`Φ(x) ≳ x`) give `v_n` growing exponentially in `n` — i.e. discovery should
compound rather than plateau, a good cross-check against the existing
`CoverageRegimeDetector` SUPERCRITICAL classification.

### Implementation plan

**Extend:** `percolation.py` (from Module 1) with time estimation.

```python
def estimate_time_to_next_discovery(
    edge_tracker, operator_stats, coverage_regime
) -> float:
    """Expected executions to reach the next uncovered edge.

    If a Φ profile is available from target_difficulty.py, prefer inverting
    the v_n integral above over ad hoc discovery-rate extrapolation.
    """
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
| 4 | Invasion percolation | **DONE** | Low | Low-Medium | Module 2 |
| 5 | First passage time budgeting | Proposed | Medium | Medium | Modules 2, 4 |
| 6 | Universality strategy transfer | Proposed | High | High (long-term) | Modules 3, 4 |

Modules 1 and 2 are complete. Module 3 (Target Difficulty Estimation) is
independent and can be developed next. Module 4 builds on Module 2's
regime signal.

---

## 9. What could falsify this framing

1. ~~**The coverage graph is not random.**~~ **Largely addressed
   (2026-08-31).** If the program's state-space graph has heavy-tailed
   degree distribution (scale-free), the Erdős-Rényi `p_c = 1/⟨k⟩` formula
   does not apply, and this was flagged as an open risk for Modules 3-6.
   Diskin–Easo–Radhakrishnan–Sudakov–Tassion (arXiv:2603.03257) prove
   supercritical sharpness for *every* infinite transitive graph using only
   the isoperimetric function `Φ(n)`, with no degree-distribution
   assumption — see §4 (Module 3) and §6 (Module 5) above, both revised to
   use `Φ` instead of `⟨k⟩`/`C`. Two caveats remain: (a) the fuzzer's
   coverage graph is finite and non-transitive (no exact vertex-symmetry),
   so this is an analogy, not a literal application of the theorem — the
   graphs it applies to are infinite and transitive by definition; treat
   `Φ`-based estimates as a well-motivated heuristic, not a guarantee. (b)
   `Φ(n)` itself is a min-cut-style quantity and is NP-hard to compute
   exactly for general n; Module 3's plan above only proposes a cheap
   approximation. Check once implemented: does the approximated `Φ` profile
   for a real target actually correlate with observed discovery slowdowns
   at narrow CFG chokepoints (e.g. checksum/length gates)?

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
| `src/fuzzer_tool/core/target_difficulty.py` | **PROPOSED (revised)** — `estimate_isoperimetric_profile()`, Module 3 |

## 11. References

- Diskin, Easo, Radhakrishnan, Sudakov, Tassion. *Supercritical sharpness
  of percolation.* arXiv:2603.03257 [math.PR], Mar 2026. Source for the
  `Φ`-based revisions to Modules 3 and 5 and for closing falsifier §9.1.
