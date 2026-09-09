# Handover — RO / RD temporal-orientation integration (paper-inspired)

**Date:** 2026-09-08  
**Status:** design + implementation plan (no code landed yet)  
**Constraint:** **Touching `src/fuzzer_tool/adapters/afl_shim.c` is forbidden.**  
All work stays in Python (`core/`, `services/`, `cli/`, schedulers, seed metadata, report path).  
**Paper citation:** G. Du, *A Systematic Solution to the Arrow of Time Problem: Reversal of Objective Temporal Orientation Is Not Dynamical History Reversal* (81 pp., attached as `paper.pdf` / document id DKV4j). Key distinctions: RO vs RD (eq. 1.2), Temporal Quantity-Type Existence + Time-Causal Structure postulates (Postulates 2.1–2.2, structure \(T_{\mathrm{phys}}=(Q_T,T_{\mathrm{loc}})\)), finite-time occupation measures (Sec. 3), horizontal vs longitudinal statistical support, typicality/concentration and the conditional form of the local second law \(P(H_{t+\tau}\mid L_t)\approx 1\) (eq. 1.3 / A4).

**Base for this plan:** live source as of the zip that contained this handover (edge_tracker, transfer_entropy, renyi, execution_time, schedulers/*, seed_quality, lineage, path-negation / SMT paths). Re-verify every file:line anchor against HEAD before coding.

**Companion reading (repo):**  
- `core/transfer_entropy.py` — already implements directed \(T_{X\to Y}\) with surrogate bias correction (Schreiber / Marschinski–Kantz style).  
- `core/edge_tracker.py` — per-seed edge sets, hit-count distributions, Chao2, Wasserstein, F0.  
- `core/renyi.py`, `core/seed_quality.py`, `core/execution_time.py`.  
- `core/schedulers/*` and Elo arbitration.  
- Lineage / `--lineage` and path-negation / SMT paths (operators + cmplog consumers).  
- Existing handovers under `docs/handover/` for style and priority language.

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
- Follow existing conventions: surgical changes, register new schedulers in the established places, match surrounding code style, do not invent parallel mechanisms.

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
| RO (objective orientation reversal) | Path-negation / branch inversion / SMT opposite-branch solving | Existing operators + cmplog / SMT | Explicit RO tag; sector-crossing arm |
| RD (history reversal, orientation fixed) | Reverse recorded operator sequence on a seed’s lineage and re-execute | `--lineage`, mutation history | RD helper + intra-sector mutator / tmin use |
| Time-Causal Structure / objective \(A\to B\) | Stable directed TE graph (edge→edge, byte→edge, op→coverage) | `TransferEntropy` + surrogate bias | Causal-sector graph + soft constraints |
| Finite-time occupation \(\mu_I\) | Per-run edge **count** vector (normalised or raw) | SHM `{edge_id, count}` | `OccupationMeasure` / snapshot on seed |
| Macroscopic push-forward \(\mu^{\mathrm{mac}}_I\) | Coarse map (function / module / BB group) | Edge IDs; optional CFG | Optional coarse occupation |
| Horizontal support | Repeated seeds / hit frequencies across corpus | EdgeTracker owner counts, Chao2 | Keep; label explicitly |
| Longitudinal support | Occupation along one execution history | Per-run counts | New longitudinal rarity / occupation score |
| Typicality / concentration | Rényi, rarity, F0, Chao2 | Existing modules | Conditional \(P(H\mid L)\) form |
| Conditional second law | Prefer mutations that start in low-measure \(L\) and transition toward high-measure \(H\) | TE, Markov, bandits | Re-weighting policy |
| Theoretical probability object vs actual support | Bandit posteriors / Markov / MI vs empirical counts / occupation | Both exist | Bridge-deficit metric + report |
| Sector selection | Orientation chosen by Time-Causal Structure; RO leaves the sector | — | Gating of RO arm by TE-graph stability |

**Formula-level reminder (paper eq. 1.2):** the same ordinary reversal map \(R\) lifts in two ways, \(\Pi_{R_D}=\Pi_{R_O}=R\Pi\), but \(R_D\not= R_O\). The fuzzer must never treat path-negation (RO-class) as merely “another history-reversal mutator”.

---

## 2. Implementation plan (phased, no shim)

### Phase 0 — Anchors and invariants (½ day, no behaviour change)

1. Re-verify live anchors (do not trust this document’s line numbers after further commits):
   - `core/transfer_entropy.py` — `TransferEntropy.transfer_entropy`, surrogate correction, `edge_to_edge_flow`.
   - `core/edge_tracker.py` — `record_edges`, per-seed edge maps, hit-count distributions, prune ceilings, Chao2 / F0 paths.
   - Seed metadata / corpus store — where extra per-seed blobs can be attached without blowing the tracked-seed memory bound (see edge_tracker comments on ~95 KiB / ~592 KiB per seed).
   - Operator registry and path-negation / SMT entry points.
   - Lineage storage and `--lineage` consumers (tmin, auto-minimize).
   - Elo / bandit arbitration and `_OPERATOR_STRATEGY_NAMES` (or current equivalent).
   - Report / stats-line path (`--report`, live stat line).

2. Add a short note to `docs/learnings/` or this handover once anchors are confirmed.

3. Invariant for all later phases: **every new probability or typicality number must name (a) the theoretical object, (b) the macroscopic partition if any, (c) the statistical support (horizontal or longitudinal).**

### Phase 1 — Discrete occupation measure (1 day)

**Goal.** Treat the per-run edge count vector as the paper’s finite-time occupation \(\mu_I\) (visits instead of Lebesgue time).

**Work:**

1. New small module `core/occupation.py` (or a clearly named helper inside edge_tracker if the surface is tiny):
   - Input: the non-zero `(edge_id, count)` pairs from one execution (already available after the run).
   - Outputs:
     - raw visit vector / sparse map,
     - normalised \(\mu_I(e)=\mathrm{count}(e)/\sum\mathrm{count}\),
     - optional entropy of the occupation distribution (Shannon or Rényi),
     - optional “dominant macroregion” under a trivial identity partition (paper Sec. 3.2 style `arg max`).
   - Keep memory bounded: store a sparse dict or a fixed-size sketch (top-K edges by count, or a Count-Min / simple hash sketch) rather than a full dense map for every seed.

2. Hook point: after a successful (or interesting) execution, snapshot occupation into seed metadata or an auxiliary structure owned by EdgeTracker / CorpusManager. Respect the existing tracked-seed ceiling and prune logic; occupation must not turn the 90% low-water prune into an OOM.

3. Longitudinal rarity score:
   - For each edge (or seed), compute a longitudinal rarity from the occupation mass it receives across the seeds that hit it (e.g. average or median \(\mu_I(e)\) among hitting seeds, or inverse).
   - Expose `occupation_rarity(seed)` and/or per-edge longitudinal rarity alongside the existing horizontal rarity / Chao2.

4. Tests:
   - Unit tests on synthetic count vectors (normalisation, empty run, single-edge run, heavy-tailed occupation).
   - Integration: occupation snapshot does not change coverage decisions when the new scores are disabled (feature flag or weight 0).
   - Memory: occupation storage stays inside the documented per-seed budget or is explicitly opt-in.

5. CLI / config: `--occupation` / `--occupation-weight` (default off or weight 0 so existing campaigns are byte-identical).

**Out of scope for Phase 1:** true time-based occupation, CFG-based macroscopic maps (Phase 3 optional), writing occupation into the SHM itself.

### Phase 2 — RO / RD tagging and lineage helpers (1–1.5 days)

**Goal.** Make the paper’s RO ≠ RD distinction operational in the mutation and minimisation layers.

**Work:**

1. **Tagging**
   - Classify existing operators:
     - **RO-class:** path-negation, SMT opposite-branch / path-negation solves, any explicit “invert this comparison outcome” operator.
     - **RD-class:** ordinary mutators; plus a new (or helper) “lineage reverse” that replays the inverse operator sequence recorded for a seed (where inverses are well-defined: bit flips, many arithmetic ops, some block ops; skip or approximate non-invertible ones).
   - Store the class on the operator registry entry or on the mutation record so schedulers and reports can see it without string matching names.

2. **RD helper**
   - Given a seed’s lineage (parent + operator sequence + sites), produce a candidate reversed history and re-execute under the *same* coverage orientation.
   - Use cases: tmin / auto-minimize unproductive-branch pruning; diversity checks; “is this crash path RD-reversible?”.
   - Failure modes: non-invertible operators, length changes that break FrameShift invariants — document and skip rather than invent silent approximations.

3. **RO gating (soft)**
   - Do **not** remove RO operators from the pool.
   - Optionally gate their *selection probability* or treat them as a separate arm whose reward is only fully credited when a causal orientation has been declared stable (Phase 3). Until Phase 3 lands, tag and report only.

4. **Tests**
   - Round-trip RD on a synthetic invertible lineage.
   - RO-tagged operators still appear in the registry and can be forced via existing “select this operator” test hooks.
   - No change to default campaign behaviour when RO/RD policy weights are zero.

5. **Docs / report**
   - Live stats or `--report` section: counts of RO vs RD applications, RO success rate, RD-reversible crash fraction (when lineage is on).

### Phase 3 — Causal-sector graph from transfer entropy (1.5–2 days)

**Goal.** Realise the paper’s Time-Causal Structure as a lightweight, decaying directed graph over edges (and optionally operators / byte positions) built from the existing TE estimator.

**Work:**

1. **Graph maintenance**
   - Consume `TransferEntropy.edge_to_edge_flow` (and any byte→edge / op→coverage TE already computed).
   - Maintain a sparse directed graph: nodes = edge ids (top-K by hit mass or by TE participation), weighted edges = bias-corrected TE, with exponential decay or sliding window so the graph tracks the current campaign.
   - Stability predicate: e.g. the set of high-TE directions has not flipped for \(N\) observation windows, or the asymmetry \(T_{A\to B}-T_{B\to A}\) stays above a threshold with the same sign.

2. **Soft constraints**
   - Prefer operators / seeds whose recent discoveries align with the current orientation.
   - When stability holds, optionally boost or isolate the RO arm (Phase 2 tags) so that orientation-reversing moves are deliberate probes rather than noise.
   - Never hard-block coverage-increasing RO moves; the paper’s sector is a selector, not a censorship regime.

3. **Integration points**
   - Scheduler policy or Elo meta-feature: “causal alignment” score.
   - Seed quality: slight bonus for seeds that extend a stable causal chain (new edges that are TE-successors of already oriented edges).

4. **Tests**
   - Synthetic TE series with a known direction; graph recovers it and marks stability.
   - Surrogate / shuffled series → no stable orientation.
   - Feature-flag off → zero effect on decisions (equivalence).

5. **Memory / cost**
   - Cap nodes and edges hard; TE is already the expensive part. Do not recompute full pairwise TE every iteration; reuse existing observation cadence.

### Phase 4 — Conditional typicality scoring and bridge deficit (1–1.5 days)

**Goal.** Implement the paper’s conditional form of the local second law and the theoretical-vs-actual bridge.

**Work:**

1. **Low / high measure regions**
   - \(L\): seeds or edge-sets with high Rényi / high horizontal+longitudinal rarity / low F0-normalised mass (nonequilibrium preparation).
   - \(H\): dense, typical coverage regions (high measure under the current reference, e.g. longitudinal occupation mass or horizontal hit frequency).
   - Keep definitions explicit and configurable; do not silently identify “more edges” with “higher entropy” or with “progress”.

2. **Conditional score**
   - Prefer mutations that:
     - start from an \(L\)-like seed, **and**
     - have high estimated conditional probability (from TE, Markov, or recent empirical transitions) of reaching new \(H\)-like or previously unseen edges.
   - This is a re-weighting of existing signals, not a new search algorithm. Wire as an optional term in seed_quality / operator reward with a weight defaulting to 0.

3. **Bridge deficit**
   - Theoretical object: bandit posteriors, Markov transition matrix, MI estimates, or any other formal probability the campaign already maintains.
   - Actual support: horizontal hit frequencies and/or longitudinal occupation masses.
   - Deficit = divergence (KL, total variation, or simple relative error on the support of interest).
   - Report it; optionally down-weight theoretical-guided decisions when deficit is large (model has lost contact with observed histories).

4. **Tests**
   - Synthetic: L→H transitions scored higher than H→L or L→L under the conditional weight.
   - Bridge deficit rises when theoretical model is deliberately desynchronised from counts.
   - Default weights 0 → bit-identical behaviour.

### Phase 5 — Reporting, CLI, and documentation (½–1 day)

1. CLI flags (all default safe / off):
   - `--occupation` / `--occupation-weight`
   - `--ro-rd-policy` / `--ro-arm-weight` (or similar)
   - `--causal-sector` / `--te-graph`
   - `--conditional-typicality` / `--bridge-deficit`
   - Or a single umbrella `--arrow-of-time` that enables a documented subset.

2. `--report` and live stats:
   - Occupation entropy / dominant edges.
   - RO vs RD counts and yields.
   - Causal-orientation stability and top directed TE pairs.
   - Bridge deficit summary.
   - Explicit statement of which probability objects and supports were used (paper discipline).

3. Update:
   - `CHANGELOG.md` (Added section).
   - Short learning note under `docs/learnings/` if non-obvious pitfalls appear.
   - This handover → move to a `*_done_*.md` when phases complete, with a removal / decision ledger.

4. AGENTS.md / SPEC: only if new public CLI surface or hard rules appear; otherwise keep the change local.

---

## 3. File-level sketch (expected touch list)

**New**
- `src/fuzzer_tool/core/occupation.py` — discrete occupation measure, normalisation, longitudinal rarity helpers.
- `tests/test_occupation.py`
- Optionally `src/fuzzer_tool/core/causal_sector.py` — TE graph + stability (or keep inside transfer_entropy / a scheduler helper if small).

**Likely modified (surgical)**
- `core/edge_tracker.py` — optional occupation snapshot hook; do not break prune / memory ceilings.
- `core/seed_quality.py` — optional occupation / conditional / bridge terms.
- `core/transfer_entropy.py` — only if small helpers for graph export are cleaner here than a new module.
- Operator registry / mutation recording path — RO/RD class tags.
- Lineage / tmin path — RD reverse helper.
- Scheduler arbitration or one policy module — soft RO gating and causal alignment feature.
- `cli/commands.py` — flags and wiring.
- Report / stats path — new sections.
- `CHANGELOG.md`, this handover (status updates).

**Forbidden**
- `adapters/afl_shim.c` and any other C shim that defines SHM layout or edge recording.
- Changes that alter default campaign semantics when the new flags are off.

---

## 4. Testing and equivalence policy

- Every phase must ship with tests that, under default / weight-0 configuration, preserve existing behaviour (corpus decisions, operator selection distribution under fixed RNG seed, report keys that already exist).
- Prefer property / equivalence tests over golden large corpora when possible.
- Memory: occupation and TE graph must respect documented ceilings; add a regression that fills many seeds and asserts the prune path still runs and RSS stays in the same ballpark.
- No requirement for A/B fuzzing campaigns to land the feature, but a short offline evaluation script (synthetic or small real target) that shows L→H conditional scoring and RO/RD statistics is desirable before declaring Phase 4 done.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Per-seed occupation blows EdgeTracker memory | Sparse / top-K / sketch; opt-in; respect existing prune ceiling and low-water mark |
| TE graph cost | Cap nodes; reuse existing TE observation cadence; decay old edges |
| RO gating reduces path-negation discoveries | Soft weights only; never hard-disable coverage-positive RO moves |
| RD reverse undefined for many operators | Document invertible subset; skip or approximate explicitly; do not invent silent inverses |
| Conceptual overload in stats line | Gate new numbers behind the same flags; keep default line unchanged |
| “Entropy increase = progress” fallacy | Conditional form only; report theoretical vs actual support; cite paper discipline in comments |

---

## 6. Priority relative to existing handovers

This work is **P3-design / P2-wiring** in the language of `handover_pending_2026-09-06.md`:

- It is not a P0 defect in shipped code.
- It is not a measured P1 win with an equivalence oracle already in hand.
- It is design with genuine content, gated on the structural claims of the paper, and then cheap wiring once the objects exist.

Do not schedule it ahead of open P0/P1 items unless a campaign is already blocked on causal / rarity scoring quality and the owner explicitly prioritises this thread.

---

## 7. Suggested commit series (for the implementer)

1. `docs: add handover_RoRd.md (arrow-of-time / RO-RD plan)`  
2. `feat(occupation): discrete finite-time occupation from edge counts` (+ tests, flag)  
3. `feat(operators): RO/RD tags and lineage RD-reverse helper` (+ tests)  
4. `feat(te): causal-sector graph and stability predicate` (+ tests)  
5. `feat(scoring): conditional typicality and bridge deficit` (+ tests)  
6. `feat(cli/report): wire arrow-of-time flags and report sections`  
7. `docs: mark handover_RoRd phases done + CHANGELOG`

Each commit must leave the tree green and default behaviour unchanged when flags are off.

---

## 8. Citation and references

**Primary**  
G. Du, *A Systematic Solution to the Arrow of Time Problem: Reversal of Objective Temporal Orientation Is Not Dynamical History Reversal*.  
Especially: distinction RO vs RD (Introduction, eq. 1.2); Postulates 2.1–2.2 and \(T_{\mathrm{phys}}\); complete-input audit / orientation-recovery dichotomy; finite-time occupation (Sec. 3, classical \(\mu_I\) and macroscopic push-forward); horizontal vs longitudinal support; conditional local second law; probability-object / typicality / actual-support triad; limits of global Boltzmann extrapolation (Sec. 7 / Appendix C).

**Fuzzer-internal**  
- `core/transfer_entropy.py` (Schreiber TE + surrogate bias)  
- `core/edge_tracker.py`, `core/renyi.py`, `core/seed_quality.py`, `core/execution_time.py`  
- Lineage, path-negation / SMT, Elo / bandit schedulers  
- Existing handovers under `docs/handover/` for process and priority language  

**External (already reflected in TE module)**  
Schreiber, *Measuring Information Transfer* (2000); Marschinski & Kantz effective transfer entropy bias correction.

---

## 9. Acceptance checklist (when this handover can be closed)

- [ ] `docs/handover/handover_RoRd.md` present and anchors re-verified against HEAD  
- [ ] Phase 1 occupation module + tests + opt-in flag; default behaviour unchanged  
- [ ] Phase 2 RO/RD tags + RD lineage helper + tests; default unchanged  
- [ ] Phase 3 causal-sector graph + soft constraints + tests; default unchanged  
- [ ] Phase 4 conditional scoring + bridge deficit + tests; default unchanged  
- [ ] Phase 5 CLI/report/CHANGELOG; learning note if needed  
- [ ] **No modifications to `afl_shim.c` or SHM layout**  
- [ ] Memory and prune regressions green  
- [ ] Done document (or update to this file) records decisions and any rejected alternatives  

---

## 10. One-paragraph summary for the next agent

Implement the paper’s RO ≠ RD distinction, discrete finite-time occupation from existing per-run edge **counts**, a TE-derived causal-sector graph, and conditional typicality / bridge-deficit scoring **entirely in Python**. Do not touch `afl_shim.c`. Keep every new score behind flags with default weight 0 so existing campaigns remain bit-identical. Follow the phased plan above, match existing scheduler/seed/report conventions, and re-verify every file:line anchor before editing. When finished, mark phases done and leave a short decision ledger.
