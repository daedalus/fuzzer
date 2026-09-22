# Architecture Contract

This document is the **canonical layer model**. It exists because the project
already had three descriptions of its own shape — `docs/architecture.dot`
(wiring diagram), `README.md#architecture` (prose walkthrough), and
`_print_enabled_features()` in `services/fuzzer.py` (the runtime's own
7-bucket grouping) — and they disagree with each other in a few places. This
file is the tie-breaker. The other three keep their jobs (diagram, pitch,
runtime banner); this one says *where a new piece of code belongs* and *why*.

## Relationship to the other docs

| Doc | Answers | Authority |
|---|---|---|
| `docs/architecture.dot` | "which files talk to which" (data-flow wiring) | wiring only |
| `README.md#architecture` | "what does this project do" (pitch) | prose only |
| `_print_enabled_features()` | "what's on for *this* run" | runtime reflection of the model below |
| **this file** | "which layer owns this code, and why" | **canonical** |

When the diagram or the banner disagrees with this file, this file wins and
the other should be updated to match — that's a bug in the diagram/banner,
not a reason to re-litigate the layer.

## The layers

Nine layers. Seven map onto the campaign loop in order; two (0 and 8) are
cross-cutting and outside the loop.

### 0 — Foundation (cross-cutting, no flags)
Shared primitives with no fuzzing-domain semantics of their own, consumed by
≥2 other layers: `rand_pool.py`, `running_stats.py`, `bloom.py`, `cuckoo.py`,
`state_store.py`, `crc32.py`, `fast_json.py`, `kalman.py`, `gaussian.py`,
`circular_stats.py`.
**Placement rule:** if it's math/data-structure/IO infra that doesn't know
what an "edge" or a "seed" is, it goes here, not in the layer that happens to
call it first.

### 1 — Startup / Ingest (runs once, before the loop)
`cli/commands.py`, `target_profiler.py`, `cfg.py`, `icfg.py`, `dwarf.py`,
`elf.py`. Produces static facts (CFG, DWARF lines, AFLGo distance tables,
dictionary tokens) that later layers read but never recompute mid-campaign.

### 2 — Scheduling ("what to fuzz next")
Two sub-layers, kept textually separate because they're different files and
different questions, even though the diagram draws them as one node:
- **2a Operator scheduling** — `core/schedulers/op_*.py`, `shapley.py`,
  Elo rating. Picks *which mutation operator* runs this iteration.
- **2b Seed scheduling** — `services/seed_picker.py`, `schedules.py`,
  `core/schedulers/seed_*.py`, `ga.py`, `qea.py`. Picks *which corpus entry*
  runs this iteration.
**Placement rule:** output is "pick X now," not "produce bytes" (→3) and not
"measure a signal" (→5).

### 3 — Mutation ("produce or repair the candidate")
- **3a** Generic byte-level ops — `operator_registry.py` dispatch,
  `services/operators.py` (bit/byte/block/dict/radamsa/adaptive-havoc).
- **3b** Format-aware structural ops — `core/mutations/*.py` (36+ container
  formats: png/jpeg/gif/webp/tlv/grammar/tree_mutator/frameshift/markov/
  regularity).
- **3c** Constraint-driven **generation** — `wfc.py`, `wfc_chunks.py`,
  corpus-boost. Kept as its own bucket even though it's wired through the
  same operator registry as 3a/3b, because it *synthesizes* new structured
  content from constraints rather than perturbing existing bytes — a
  different operation, same plumbing.
- **3d** Checksum/constraint repair — `berlekamp_massey.py`,
  `int_checksum_solver.py`, `smt_solver.py`, `xor_map_solver.py`,
  `z3_lifecycle.py`, `gradient_cmp.py`, `gradient_descent.py`. Repairs a
  candidate so it survives validation; never decides which seed/operator to
  use, never executes anything.
**Placement rule:** produces or repairs the next candidate input. Anything
that runs the target instead belongs in layer 4.

### 4 — Execution & Coverage (deliberately one layer)
`adapters/*` (forkserver, inprocess, network, ptrace, perf_event, lbr_trace,
shm.py), `services/runner.py`, `edge_tracker.py`, `op_edge_tracker.py`,
`cmplog.py`, `intel_pt.py`.
**Explicit resolution:** the `.dot` diagram splits HW-perf/PT/LBR coverage
into a separate "Feedback" cluster from "Execution." This file follows the
README's own combined "Coverage & Execution" heading instead: anything that
touches the target process, a fork, a socket, or reads back raw HW/edge/cmp
signal is Execution & Coverage, full stop. If you're adding a new coverage
source, it goes here, not in Analysis.

### 5 — Analysis ("turn raw signal into a decision input")
Everything registered in `core/analyzer_registry.py` **whose output is
read by layer 2 or 3, or is explicitly intended to be**: `mi.py`,
`transfer_entropy.py`, `renyi.py`, `sensitivity.py`, `structure_function.py`,
`causal_sector.py`, `occupation.py`, `garch.py`, `coverage_regime.py`,
`discovery_uniformity.py`, `kuramoto_sync.py`, `distance.py` (AFLGo),
`checksum_learner.py` (feeds 3d), `corpus_compression.py`, `csd.py`,
`continuum.py`. Elo is cross-listed here and in 2a — it's wired as an
analyzer but its output is consumed as a scheduling weight.
**Two named exceptions, and why they're exceptions:** `format_learner.py`
and `analyzer_trace.py` are also wired through `analyzer_registry.py` but
live in layer 8, not here — see below.

### 6 — Corpus lifecycle
`services/corpus_manager.py` (admission/prune/sync), `rate_distortion.py`,
`state_store.py` (resume), `minimal_seeds.py`, `target_difficulty.py`. Owns
the corpus itself, independent of which seed gets picked next (that's 2b).

### 7 — Crash pipeline
`sanitizer.py`, `crash_metadata.py`, `root_cause.py`, `tmin.py`,
`minimize.py`, `differential.py`. Runs only after a crash/hang signal —
never touches normal-iteration scoring.

### 8 — Output / Observability (cross-cutting, no fuzzing decisions read this)
`services/stats.py`, `stats_reporter.py`, `report.py`, `sendmail.py`, plus:
- **`format_learner.py`** ("learn-format" in the banner) — today it only
  *records* observations (`record_transition`, `record_liveness`); nothing
  reads its hypotheses back into layer 2/3 yet (`live_bit_mask.py:69` notes
  the wiring as a TODO). It is filed here, not in Analysis, precisely
  because it doesn't feed a decision yet.
- **`analyzer_trace.py` / `CrashTracer`** ("trace-crashes") — writes a
  report to `crashes_dir`; that's artifact generation, not feedback into
  the loop, even though it's dispatched through the same registry as the
  layer-5 analyzers.
**Standing rule:** the moment either of these two starts feeding a
scheduling or mutation decision, move it to layer 5 in this doc, the
banner group in `_print_enabled_features()`, and `analyzer_registry.py`'s
own docstring, in the same commit. Don't leave the three disagreeing again.

### 9 — Developer Tooling (outside the runtime entirely)
`tools/*.py`, `tools/*.sh` — benchmarks (`bench_*.py`), corpus/target
generation (`corpus_fuzzgoat.py`, `gen_*`, `vendor_*.sh`), diagnostics
(`edge_diagnostic.py`, `edge_matrix_analysis.py`, `phantom_edge_probe.py`).
Never imported by `src/fuzzer_tool/`; runs against a checkout, not inside a
campaign. New one-off scripts belong here, not in `core/`, even if they
import fuzzer internals for analysis.

## Adding something new: the checklist

1. **Pick the layer** using the placement rules above, not the registry you
   happen to wire it through (registry ≠ layer — see the layer-8 exceptions).
2. **Register it** in the matching mechanism: `operator_registry.py` (3a/3b),
   `analyzer_registry.py` (5 or 8), a `core/schedulers/` dispatch table
   (2a/2b), or nothing (0, 1, 4, 6, 7, 9 mostly wire in directly).
3. **Add the banner entry** in `_print_enabled_features()` under the same
   layer's group name (Scheduling/Seed selection/Mutation/Generation/
   Execution/Analysis/Output). If your feature doesn't fit an existing
   banner group, that's a signal the group list itself needs a new bucket —
   update this doc first, then the banner, not the other way round.
4. **State whether it's read back.** If layer 5 or layer 3d, say in the
   docstring what specifically consumes the output. "Might be useful later"
   is a layer-8 feature until something actually reads it.
5. **Update `docs/architecture.dot`** only if the wiring — not the
   conceptual layer — changed. The two files answer different questions;
   don't let one drift into the other's job.

## Known debt (tracked, not yet fixed)

- `docs/architecture.dot` still shows HW-perf counters under a separate
  "Feedback" cluster from "Execution" (see layer 4's resolution above) —
  needs a diagram edit to stop disagreeing with the README and this file.
- `analyzer_registry.py`'s module docstring calls every wired component a
  generic "detector/estimator," which is what caused the format_learner /
  trace-crashes ambiguity in the first place. Consider splitting the
  docstring's own list into "feeds a decision" vs "produces an artifact" so
  the registry's self-description stops implying they're all layer 5.
