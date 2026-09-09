# Handover — RO / RD temporal-orientation integration (paper-inspired)

**Date:** 2026-09-08 (updated 2026-09-08 with wiring plan)  
**Status:** primitives implemented + unit-tested (unwired); wiring plan below  
**Constraint:** **Touching `src/fuzzer_tool/adapters/afl_shim.c` is forbidden.**  
All work stays in Python (`core/`, `services/`, `cli/`, schedulers, seed metadata, report path).  
**Paper citation:** G. Du, *A Systematic Solution to the Arrow of Time Problem: Reversal of Objective Temporal Orientation Is Not Dynamical History Reversal* (81 pp., attached as `paper.pdf` / document id DKV4j). Key distinctions: RO vs RD (eq. 1.2), Temporal Quantity-Type Existence + Time-Causal Structure postulates (Postulates 2.1–2.2, structure \(T_{\mathrm{phys}}=(Q_T,T_{\mathrm{loc}})\)), finite-time occupation measures (Sec. 3), horizontal vs longitudinal statistical support, typicality/concentration and the conditional form of the local second law \(P(H_{t+\tau}\mid L_t)\approx 1\) (eq. 1.3 / A4).

**Base for this plan:** live source as of the campaign zip / HEAD that contains `core/analyzer_registry.py` (migration complete). Re-verify every file:line anchor against HEAD before coding.

**Companion reading (repo):**  
- `core/analyzer_registry.py` — **single source of truth** for pluggable analyzers; all construction of detectors/estimators goes here (`AnalyzerSpec` + `REGISTRY.wire_all`). See also `docs/handover/handover_analyzer_registry_2026-09-07.md`.  
- `core/operator_registry.py` — mutation operator dispatcher (`path_negate` already registered).  
- `core/transfer_entropy.py` — directed \(T_{X\to Y}\) with surrogate bias correction; already an analyzer (`transfer_entropy`).  
- `core/edge_tracker.py` — per-seed edge sets, hit-count distributions, Chao2, Wasserstein, F0.  
- `core/renyi.py`, `core/seed_quality.py`, `core/execution_time.py`.  
- `core/schedulers/*` and Elo arbitration.  
- Lineage / `--lineage` and path-negation / SMT paths.  
- Existing handovers under `docs/handover/` for style and priority language.

**Landed primitives (unwired, 41 unit tests):**  
- `core/occupation.py` — `OccupationMeasure`, `LongitudinalRarity`  
- `core/ro_rd.py` — `OrientationClass`, `classify_operator_name`, `rd_reverse_lineage`, `tag_operator_record`  
- `core/causal_sector.py` — `CausalSectorGraph`, stability / asymmetry  
- `tests/test_{occupation,ro_rd,causal_sector}.py`

---

## 0. Goal and non-goals

### Goal
Import the paper’s structural distinctions into the fuzzer’s *decision* and *scoring* layers so that:

1. Objective temporal / causal orientation (the paper’s Time-Causal Structure) is treated as a primitive that *selects a sector*, not something derived from entropy increase or coverage growth alone.
2. Reversal of objective orientation (**RO**) is kept distinct from dynamical-history reversal (**RD**) inside a fixed orientation.
3. Finite-time occupation (discrete visit counts already exported by the shim) becomes a first-class longitudinal statistical support, distinct from horizontal repeated-seed hit frequencies.
4. Seed and operator scoring can use conditional typicality (“low-measure preparation \(L\) → high-measure region \(H\)”) rather than unconditional rarity or entropy growth.
5. The gap between theoretical probability objects (bandit / Markov / MI) and actual statistical support (occupation / hit frequencies) is measured and optionally scored (“bridge deficit”).

### Non-goals / hard constraints
- **No edits to `afl_shim.c`**, no SHM layout changes, no new edge-timing channels, no new cmplog record types.
- No per-edge wall-clock or cycle occupancy (would require shim or ptrace). We use the existing per-run `{edge_id, count}` map as a *discrete* occupation measure (visits, not time).
- No cosmological entropy arguments, unit-covariance formalisms, or axiomatic measurement theory beyond the operational clocks already present.
- No new universal prior over microstates; every probability / typicality claim must state its object, partition, and support (paper Sec. 2.1 / A3–A4).
- Follow existing conventions: surgical changes, **register analyzers only via `analyzer_registry`**, match surrounding code style, do not invent parallel construction paths in `Fuzzer.__init__`.
- Default campaign behaviour must remain bit-identical when new flags are off / weights are 0.

### Why this is feasible without the shim
The shim already exports, per execution:
- sparse edge table with **counts** (not merely bits),
- `path_hash`, stack depth,
- (optional) comparison records.

Whole-run wall time and optional `perf_event` instruction/branch counters already exist. Transfer entropy, Rényi, MI, lineage, and edge-tracker distributions are pure Python. That is sufficient for discrete occupation, causal orientation, RO/RD tagging, conditional scoring, and bridge-deficit reporting.

---

## 1. Conceptual mapping (paper → fuzzer)

| Paper concept | Fuzzer analogue | Data already present | New Python surface |
|---------------|-----------------|----------------------|--------------------|
| RO (objective orientation reversal) | Path-negation / branch inversion / SMT opposite-branch solving | `path_negate` in `operator_registry` + cmplog / SMT | `OrientationClass.RO`; sector-crossing arm |
| RD (history reversal, orientation fixed) | Reverse recorded operator sequence on a seed’s lineage and re-execute | `--lineage`, mutation history | `rd_reverse_lineage` (pure); later tmin |
| Time-Causal Structure / objective \(A\to B\) | Stable directed TE graph | `TransferEntropy` analyzer | `CausalSectorGraph` analyzer |
| Finite-time occupation \(\mu_I\) | Per-run edge **count** vector | SHM `{edge_id, count}` | `OccupationMeasure` analyzer |
| Macroscopic push-forward \(\mu^{\mathrm{mac}}_I\) | Coarse map (function / module / BB group) | Edge IDs; optional CFG | `OccupationMeasure.push_forward` |
| Horizontal support | Repeated seeds / hit frequencies | EdgeTracker owner counts, Chao2 | Keep; label explicitly |
| Longitudinal support | Occupation along one execution history | Per-run counts | `LongitudinalRarity` |
| Typicality / concentration | Rényi, rarity, F0, Chao2 | Existing modules | Conditional \(P(H\mid L)\) form (late) |
| Conditional second law | Prefer \(L\to H\) transitions | TE, Markov, bandits | Re-weighting policy (late, weight 0 default) |
| Theoretical vs actual support | Bandit / Markov / MI vs counts / occupation | Both exist | Bridge-deficit metric + report |
| Sector selection | TE-graph stability gates RO soft weight | — | `CausalSectorGraph.stable` |

**Formula-level reminder (paper eq. 1.2):** \(\Pi_{R_D}=\Pi_{R_O}=R\Pi\) but \(R_D\not= R_O\). Never treat `path_negate` as merely another history-reversal mutator.

---

## 2. Architecture rule: analyzer_registry is the wiring surface

The analyzer-registry migration is **complete**. Historically each detector was constructed ad hoc in `Fuzzer.__init__`; every such component now lives behind one `AnalyzerSpec`:

| Field | Role |
|-------|------|
| `name` | Stable id (`transfer_entropy`, `occupation`, …) |
| `category` | Taxonomy bucket (e.g. `mutual_information`, `coverage`) |
| `available(f)` | Reads a gating flag already set on the fuzzer (`getattr(f, "_use_x", False)`). `None` = always on |
| `activate(f)` | Construct, optional state-store restore, one-time log |
| `deactivate(f)` | Off defaults (usually `attr = None`) |
| `phase` | `"main"` (default `wire_all`) or `"early"` (rare ordering constraint) |
| `swallow_errors` | Only for fail-open legacy cases (e.g. checksum_learner) |

`REGISTRY.wire_all(fuzzer, phase="main")` runs once from `Fuzzer.__init__` after `_state_store` / `max_len` and after all gating flags are assigned. A second call with `phase="early"` exists only for order-sensitive analyzers (currently `sensitivity`).

**Rule for this work:** anything that is “constructed when a flag is set and torn down when not” is registered in `core/analyzer_registry.py`. `Fuzzer.__init__` must not import `occupation` / `causal_sector` directly.

**Parallel rule:** RO/RD *labels* belong with operators (`operator_registry` / `classify_operator_name`). RD *reverse* stays a pure helper until lineage/tmin asks for it — not an analyzer.

Existing TE pattern to copy (`analyzer_registry.py`):

```python
def _activate_transfer_entropy(f: FuzzerLike) -> None:
    from fuzzer_tool.core.transfer_entropy import TransferEntropy
    f._te = TransferEntropy(history_length=1)
    f._te_input_history = []
    f._te_edge_history = []
    f._te_history_max = 500
    log.info("Transfer entropy tracking enabled")

def _deactivate_transfer_entropy(f: FuzzerLike) -> None:
    f._te = None

REGISTRY.register(
    AnalyzerSpec(
        name="transfer_entropy",
        category="mutual_information",
        available=lambda f: bool(getattr(f, "_use_transfer_entropy", False)),
        activate=_activate_transfer_entropy,
        deactivate=_deactivate_transfer_entropy,
    )
)
```

Gating flag assignment lives in `Fuzzer.__init__` next to `self._use_transfer_entropy = transfer_entropy` (kwargs default `False`).

---

## 3. Implementation status and phases

### Done — Phase P (primitives, no wiring)

| Module | Contents | Tests |
|--------|----------|-------|
| `core/occupation.py` | `OccupationMeasure.from_counts`, push-forward, Shannon/Rényi, `sparse_snapshot`, `LongitudinalRarity` | `tests/test_occupation.py` |
| `core/ro_rd.py` | `OrientationClass`, `classify_operator_name`, `is_invertible_operator`, `MutationStep` / `LineageRecord`, `rd_reverse_lineage`, `tag_operator_record` | `tests/test_ro_rd.py` |
| `core/causal_sector.py` | `CausalSectorGraph` (decay, caps, asymmetry, stability windows, `aligns`, `snapshot`) | `tests/test_causal_sector.py` |

**41 unit tests, all pure — no Fuzzer, no CLI, no shim.**

### Phase 0 — Anchors (½ day, no behaviour change)

Re-verify against HEAD before any wiring commit:

- `core/analyzer_registry.py` — `AnalyzerSpec`, `REGISTRY.wire_all`, TE registration, category strings, registration order (esp. anything `coverage_regime`-style that depends on earlier specs).
- `services/fuzzer.py` — kwargs for `transfer_entropy`, assignment of `_use_transfer_entropy`, both `wire_all` call sites (`early` / `main`), edge-record / seed-outcome path (~coverage record and `seed_quality.record_outcome`), TE consumption sites.
- `core/operator_registry.py` — `path_negate` availability (`_has_branch_records`), `OperatorSpec` shape.
- `core/edge_tracker.py` — per-seed memory ceilings (comments on ~95 KiB / ~592 KiB per seed); do not attach heavy occupation blobs without a budget.
- Lineage / tmin consumers if planning RD reverse later.
- Report / stats-line path.

Invariant: every new probability or typicality number must name (a) theoretical object, (b) macroscopic partition if any, (c) statistical support (horizontal or longitudinal).

---

### Phase A — Register analyzers only (construct + deactivate)

**Goal:** objects exist on the fuzzer when flags are on; default campaigns unchanged.

#### A1. `occupation` analyzer

```text
name:        occupation
category:    coverage          # confirm against REGISTRY.categories() usage
available:   lambda f: bool(getattr(f, "_use_occupation", False))
activate:    attach LongitudinalRarity + last-run snapshot slots
deactivate:  f._occupation_rarity = None; f._last_occupation = None
```

**Activate should:**

- `f._occupation_rarity = LongitudinalRarity()`
- `f._last_occupation = None`
- `f._occupation_max_edges = 256`  # bound for sparse_snapshot
- log once: `"Occupation tracking enabled"`

**Gating flag:** add `occupation: bool = False` to `Fuzzer.__init__` kwargs (near `transfer_entropy=False`); assign `self._use_occupation = occupation` in the same block as other analyzer flags (immediately before `wire_all(self)`).

**CLI (same PR or follow-up):** `--occupation` → passes the flag. Default off.

**Do not** import `occupation` inside `Fuzzer.__init__` body — only via registry activate.

#### A2. `causal_sector` analyzer

```text
name:        causal_sector
category:    mutual_information   # same family as transfer_entropy
available:   lambda f: (
                 bool(getattr(f, "_use_causal_sector", False))
                 and bool(getattr(f, "_use_transfer_entropy", False))
             )
activate:    f._causal_sector = CausalSectorGraph(...)
deactivate:  f._causal_sector = None
```

**Registration order:** register **after** the existing `transfer_entropy` spec so that when both flags are on in one `wire_all` pass, `_te` already exists (same pattern as `coverage_regime` depending on `csd` / `garch` / etc.).

**Soft-require TE:** do not activate an empty sector with no TE source. Prefer the `available` conjunction above.

**Gating:** `_use_causal_sector`, CLI `--causal-sector`, default off. Document that effective enablement also needs `--transfer-entropy` (or whatever the existing TE flag is).

#### A3. Registry patch shape

One block at the end of `analyzer_registry.py` (after existing registrations):

```python
def _activate_occupation(f: FuzzerLike) -> None:
    from fuzzer_tool.core.occupation import LongitudinalRarity
    f._occupation_rarity = LongitudinalRarity()
    f._last_occupation = None
    f._occupation_max_edges = 256
    log.info("Occupation tracking enabled")

def _deactivate_occupation(f: FuzzerLike) -> None:
    f._occupation_rarity = None
    f._last_occupation = None

REGISTRY.register(
    AnalyzerSpec(
        name="occupation",
        category="coverage",
        available=lambda f: bool(getattr(f, "_use_occupation", False)),
        activate=_activate_occupation,
        deactivate=_deactivate_occupation,
    )
)

def _activate_causal_sector(f: FuzzerLike) -> None:
    from fuzzer_tool.core.causal_sector import CausalSectorGraph
    f._causal_sector = CausalSectorGraph()
    log.info("Causal-sector graph enabled")

def _deactivate_causal_sector(f: FuzzerLike) -> None:
    f._causal_sector = None

REGISTRY.register(
    AnalyzerSpec(
        name="causal_sector",
        category="mutual_information",
        available=lambda f: bool(getattr(f, "_use_causal_sector", False))
        and bool(getattr(f, "_use_transfer_entropy", False)),
        activate=_activate_causal_sector,
        deactivate=_deactivate_causal_sector,
    )
)
```

#### A4. Tests for Phase A

- Extend registry / `wire_all` tests: activation matrix for `occupation` and `causal_sector`.
- Flags off → attributes `None` after `wire_all`.
- `occupation` on → `_occupation_rarity` is a `LongitudinalRarity`.
- `causal_sector` on without TE → **not** activated.
- Both on → `_causal_sector` constructed.
- No behavioural change to campaigns when flags are default False.

---

### Phase B — Record hooks (still no selection policy)

**Goal:** fill the objects from data the loop already produces.

#### B1. Occupation after coverage record

At the site that already turns a run’s sparse edge table into tracker updates (near EdgeTracker / seed outcome recording):

```python
if getattr(self, "_occupation_rarity", None) is not None:
    from fuzzer_tool.core.occupation import OccupationMeasure
    occ = OccupationMeasure.from_counts(edge_counts)  # sparse (id, count)
    self._last_occupation = occ.sparse_snapshot(self._occupation_max_edges)
    self._occupation_rarity.observe(occ)
```

**Memory:** Phase B keeps process-global `LongitudinalRarity` + `_last_occupation` only. Do **not** yet attach per-seed occupation blobs to EdgeTracker without an explicit byte budget and prune interaction review.

#### B2. Causal sector from existing TE history

TE already maintains `_te_edge_history` (and related) and is consulted on the exec path. At the same cadence TE already updates (or a periodic throttle):

```python
if getattr(self, "_causal_sector", None) is not None and self._te is not None:
    flow = self._te.edge_to_edge_flow(self._te_edge_history, ...)
    self._causal_sector.observe_flow(flow)
```

Reuse TE’s observation cadence; do not recompute full pairwise TE every iteration.

#### B3. Policy still frozen

No change to operator selection weights, seed selection, or `path_negate` frequency. Optional debug log of `snapshot().stable` behind the same flags.

---

### Phase C — RO/RD on the operator side (not analyzer_registry)

`path_negate` is already registered in `operator_registry` (availability `_has_branch_records`).

#### C1. Tagging (metadata only)

**Preferred first step:** call `classify_operator_name(op_name)` / `tag_operator_record` at record and report sites only — zero change to `OperatorSpec` schema.

**Optional later:** add `orientation: str | None = None` on `OperatorSpec`, set `"ro"` for `path_negate` (and SMT path-neg aliases). Only if many call sites need the field.

#### C2. RD reverse

Keep `rd_reverse_lineage` pure. Wire into lineage/tmin only when lineage records can supply `MutationStep(operator, site)`. That is a **services/tmin** change, not an analyzer.

Refuse non-invertible steps and RO-containing lineages (already implemented in the primitive).

#### C3. Soft RO gating (optional, last)

Only after causal_sector is stable in real campaigns:

- when `_causal_sector.stable`, apply a weight multiplier or separate arm for RO-class operators in Elo / bandit paths
- **never** hard-disable coverage-positive RO moves

---

### Phase D — Scoring / seed_quality (optional, weight 0 default)

`BayesianSeedQuality` remains the success/failure model. Do **not** fold occupation into Beta counts without a written contract.

Safer pattern:

- optional additive term when ranking seeds for the queue, e.g.  
  `score += occupation_weight * longitudinal_rarity_bonus(...)`  
  with `occupation_weight=0` by default
- bridge deficit / conditional typicality as **report metrics first**, then weights

Still no shim changes.

---

### Phase E — Report / CLI / CHANGELOG

- CLI: `--occupation`, `--causal-sector` (and document TE dependency for the latter). Defaults off.
- `--report` / live stats: occupation support size / entropy, longitudinal rarity summary, sector stability, top directed TE pairs, RO vs RD application counts (classification only).
- `CHANGELOG.md` entries per landed phase.
- When phases complete, move status in this file or write `handover_RoRd_done_*.md` with a decision ledger.

---

## 4. What goes where (summary)

| Concern | Mechanism |
|---------|-----------|
| Construct occupation / causal_sector | **`analyzer_registry`** `AnalyzerSpec` + gating flags |
| Construct TE (already done) | existing `transfer_entropy` spec |
| Feed occupation from edge counts | one call site after coverage record |
| Feed causal_sector from TE | same cadence as TE history update |
| RO vs RD label on ops | `ro_rd.classify_operator_name` / optional `OperatorSpec` field |
| RD lineage reverse | pure helper → later tmin/lineage |
| Seed score terms | optional weights default 0; **not** inside analyzer `activate` |
| CLI flags | thin pass-through to `Fuzzer(..., occupation=..., causal_sector=...)` |
| `afl_shim.c` | **never** |

**One-line rule:** anything “constructed when a flag is set and torn down when not” goes through `AnalyzerSpec` in `analyzer_registry.py`; RO/RD *labels* stay with operators; RD *reverse* stays pure until lineage/tmin asks for it.

---

## 5. File-level touch list

### Already landed (primitives PR)
- `src/fuzzer_tool/core/occupation.py` (new)
- `src/fuzzer_tool/core/ro_rd.py` (new)
- `src/fuzzer_tool/core/causal_sector.py` (new)
- `tests/test_occupation.py`, `tests/test_ro_rd.py`, `tests/test_causal_sector.py` (new)

### Phase A (next)
- `src/fuzzer_tool/core/analyzer_registry.py` — two `REGISTRY.register(...)` blocks
- `src/fuzzer_tool/services/fuzzer.py` — kwargs + `_use_occupation` / `_use_causal_sector` assignment only (no direct imports of the new modules)
- Registry / wire_all tests

### Phase B
- `services/fuzzer.py` (or the exact coverage-record helper) — occupation observe + causal_sector `observe_flow`
- Tests with synthetic edge counts / TE history

### Phase C–E
- Optional: `operator_registry.py` orientation field
- Optional: tmin/lineage RD reverse
- `cli/commands.py` flags
- Report / stats path
- `CHANGELOG.md`, this handover status

### Forbidden
- `adapters/afl_shim.c` and any SHM layout change
- Ad-hoc analyzer construction inside `Fuzzer.__init__` outside `wire_all`
- Default-on flags or non-zero policy weights without an explicit decision

---

## 6. Testing and equivalence policy

- Every wiring phase ships with tests that, under default / flag-off configuration, preserve existing behaviour (registry activation matrix, no new attrs, unchanged operator selection under fixed RNG when weights are 0).
- Prefer property / equivalence tests over large golden corpora.
- Memory: if per-seed occupation is ever added, assert prune still runs and RSS stays in the same ballpark as EdgeTracker’s documented ceilings.
- Registry tests must prove `causal_sector` does not activate without TE.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Per-seed occupation blows EdgeTracker memory | Phase B = process-global only; per-seed needs explicit budget |
| TE graph cost | Cap nodes/edges in `CausalSectorGraph`; reuse TE cadence |
| RO gating reduces path-negation discoveries | Soft weights only; never hard-disable coverage-positive RO |
| RD reverse undefined for many operators | Primitive already refuses; document invertible subset |
| Ad-hoc wiring drifts from registry | Code review rule: no new analyzer imports in `Fuzzer.__init__` |
| “Entropy increase = progress” fallacy | Conditional form only; report theoretical vs actual support |

---

## 8. Priority relative to existing handovers

**P3 design / P2 wiring** in the language of `handover_pending_2026-09-06.md`:

- Not a P0 defect in shipped code.
- Not a measured P1 win with an equivalence oracle already in hand.
- Design with genuine content; wiring is cheap once flags + `AnalyzerSpec` land.

Do not schedule ahead of open P0/P1 items unless a campaign is blocked on causal / rarity scoring quality and the owner explicitly prioritises this thread.

Respect the analyzer-registry handover: adding analyzers means **one `REGISTRY.register` call** — nothing else in the constructor.

---

## 9. Suggested commit series

1. `docs: add/update handover_RoRd.md (primitives status + analyzer_registry wiring)`  
2. `feat(core): RO/RD occupation and causal-sector primitives (unwired)` — already prepared as a format-patch  
3. `feat(analyzers): register occupation and causal_sector (flags default off)`  
4. `feat(fuzzer): record occupation and causal-sector observations`  
5. `feat(cli/report): occupation / causal-sector flags and report sections`  
6. `feat(policy): optional RO soft-weight and occupation seed term (weight 0)` — optional  
7. `docs: mark handover_RoRd phases done + CHANGELOG`

Each commit leaves the tree green and default behaviour unchanged when flags are off.

---

## 10. Citation and references

**Primary**  
G. Du, *A Systematic Solution to the Arrow of Time Problem: Reversal of Objective Temporal Orientation Is Not Dynamical History Reversal*.  
Especially: RO vs RD (Introduction, eq. 1.2); Postulates 2.1–2.2 and \(T_{\mathrm{phys}}\); complete-input audit; finite-time occupation (Sec. 3); horizontal vs longitudinal support; conditional local second law; probability-object / typicality / actual-support triad.

**Fuzzer-internal**  
- `core/analyzer_registry.py` + `docs/handover/handover_analyzer_registry_2026-09-07.md`  
- `core/operator_registry.py` (`path_negate`)  
- `core/transfer_entropy.py`, `core/edge_tracker.py`, `core/seed_quality.py`  
- Lineage, path-negation / SMT, Elo / bandit schedulers  

**External (already reflected in TE module)**  
Schreiber, *Measuring Information Transfer* (2000); Marschinski & Kantz effective transfer entropy bias correction.

---

## 11. Acceptance checklist

### Primitives
- [x] `core/occupation.py`, `core/ro_rd.py`, `core/causal_sector.py` + 41 unit tests  
- [x] No `afl_shim.c` changes  

### Wiring
- [ ] Phase A: `AnalyzerSpec` for `occupation` and `causal_sector`; flags default off; registry tests green  
- [ ] Phase B: record hooks; synthetic integration tests; still no selection change  
- [ ] Phase C: RO/RD classification at report/record sites; RD reverse only when lineage-ready  
- [ ] Phase D: optional scores with weight 0 default  
- [ ] Phase E: CLI / report / CHANGELOG  
- [ ] **No modifications to `afl_shim.c` or SHM layout**  
- [ ] **No analyzer construction outside `analyzer_registry.wire_all`**  
- [ ] Default campaigns bit-identical with flags off  
- [ ] Done document (or update here) records decisions and rejected alternatives  

---

## 12. One-paragraph summary for the next agent

Primitives for occupation, RO/RD, and causal-sector are landed and unit-tested but **unwired**. Wire them the same way as `transfer_entropy`: gating flags on `Fuzzer`, `AnalyzerSpec` + `activate`/`deactivate` in `core/analyzer_registry.py`, `REGISTRY.wire_all` only — never ad-hoc imports in `Fuzzer.__init__`. Soft-require TE for `causal_sector`. Record edge-count occupation and TE flows in Phase B without changing selection policy. Keep RO/RD labels on the operator side; keep `rd_reverse_lineage` pure until tmin/lineage needs it. Default flags off, policy weights 0, no `afl_shim.c`. Re-verify anchors against HEAD before editing.
