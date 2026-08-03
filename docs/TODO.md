# TODO — fuzzer-tool Roadmap

> **Status note**: This roadmap tracks aspirational and in-progress work. Many original "TODO" items have been implemented since the list was first drafted — those are marked [x] with the resolved‑by feature. Items without [x] are still pending.

## Coverage & Instrumentation
- [ ] **Call stack coverage**: distinguish `f()→g()` from `h()→g()` by encoding caller context into the edge ID, not just `prev_loc ^ cur_loc`. Would improve edge resolution for shared-library targets.
- [x] **Forkserver mode** — implemented via `ForkserverRunner`, auto-selected for standalone executables. 2-5x throughput improvement over per-call subprocess.
- [x] **Deep coverage with x86-64 decoder BB discovery** — Capstone-based basic block discovery available via `--deep-coverage`.
- [x] **Sanitizer coverage** — `-fsanitize-coverage=trace-pc-guard` support via `--clang-scov`, with auto-detection of sancov counters in `.so` targets.
- [x] **Cmplog/comparison coverage** — symbol-based (cmplog_shim.c, LD_PRELOAD) and compiler-IR (trace-cmp, tracecmp_shim.c) both implemented.

## Mutation
- [x] **Radamsa-style structural mutations** — tree mutator (`lightweight_tree_mutate`) with delimiter-based delete/duplicate/swap/stutter.
- [x] **Token-level mutations for text protocols** — grammar integration with grammar-aware mutations.
- [ ] **Havoc stage weighting by per-operator success history** — current havoc uses uniform operator pool; could weight toward operators with higher historical success rate.

## Scheduling
- [x] **Chi-squared operator heterogeneity test** (`--chi2-operator-interval`) — periodic diagnostic that tests whether operators have significantly different success rates, driven by contingency-table independence test with Cramér's V effect size. Implemented as an independent chi-squared module (`chi_squared.py`) with four test families (goodness-of-fit, homogeneity, independence, p-value via regularized incomplete gamma).
- [x] **Coverage-column homogeneity detector** — `CoverageHomogeneityDetector` in `critical_slowing.py` tracks per-column edge discovery and tests spatial uniformity of coverage via chi-squared goodness-of-fit.
- [x] **Seed energy burst on discovery, decay over time** — AFL++ power schedules (FAST/COE/RARE/MMOPT/LIN/QUAD) via `SeedScorer`, plus Honggfuzz power factors.
- [x] **Boltzmann seed selection** (`--boltzmann`) — thermodynamic seed weighting: `P(seed) ∝ exp(-E/T)` with `E = log(fuzz_count+1)`, reusing the SA temperature for annealing. Collapses most of the seven hand-tuned schedules into one formula with one tunable knob.
- [x] **Metropolis corpus admission** (`--metropolis`) — probabilistic acceptance of non-improving inputs: `P = exp(-ΔE/T)`. Exploratory junk admitted at high T; strict coverage-only rule at low T.
- [x] **AFLGo distance-annealed schedule** (`--schedule go`) — wires the already-computed `avg_distance` and `_anneal_progress` into `SeedScorer.score()` so mutation budget tightens toward target seeds during exploitation phase.
- [x] **Bayesian seed quality feedback** (`--bayesian`) — `record_outcome()` now actually feeds the Beta-Bernoulli posteriors (was previously a dead code path with all seeds stuck at Beta(1,1)). Thompson sampling now drives genuine explore/exploit balance.
- [ ] **Fluctuation theorems for fuzzing** (research-y) — Jarzynski/Crooks relations could estimate the "difficulty" of reaching rare corpus regions from biased/accelerated trajectories. Requires a well-defined work functional over mutation trajectories, which doesn't exist yet. Flagged as speculative, not a sprint task.
- [ ] **Collaborative scheduling across parallel workers** — parallel workers currently sync corpus but don't coordinate scheduling decisions. Could share exploration/exploitation state.

## Crash Analysis
- [x] **Automated crash bucketing** — Levenshtein crash clustering groups crashes by stack-trace similarity.
- [x] **Exploitability scoring** — `ASAN_EXPLOITABILITY` classification in reports.
- [ ] **Root cause diff** — show minimal byte diff from nearest non-crashing input to root-cause bytes.

## Performance
- [x] **Forkserver mode** — see Coverage section above.
- [x] **Cmplog optimization** — adaptive periodic collection (1 in 20 iterations once pool exceeds 2000 entries).
- [x] **Corpus distillation on-the-fly** — `--max-corpus` triggers auto-minimization when corpus exceeds threshold.
- [x] **CLI hotpath profiling** — `--profile-hotpath` cProfile integration into fuzz mode (tottime/cumtime/ncalls tables + `.prof` dump via `--profile-out`).
- [x] **array.array for cold bounded histories** — corpus-size history, the four tuple histories (discovery/crash-rate/entropy/coverage-timeline), elo prediction-error lists, redqueen pair-length index converted to `array("Q")`/`array("d")`/`array("I")` (7-9x memory per container, runtime verified neutral). EdgeTracker edge-count maps deliberately left as sparse dicts (array.array loses on scalar RMW + iteration; reads already numpy-vectorized).
- [x] **Bound the MI tracker's memory** — the joint (position x byte x edge) was unbounded; `--elo all` on a real corpus OOM-killed at 7.5GB/796MB mi.json with EPS collapsing 1500->20. Now: hard cell cap (MAX_JOINT_CELLS) rejecting new cells at budget (no eviction thrash), MI_MAX_POSITIONS cap independent of max_len, and mi.json loaded only with `--resume`. Measured: 7.5GB -> flat 229MB, EPS 11 -> 180+, run completes where it previously OOM'd at ~1.2k execs.

## Infrastructure
- [ ] **Dockerfile** for reproducible builds and CI
- [ ] **Structured logging** (e.g. `--log-json`) for machine-parseable output
- [ ] **`fuzzer-tool-asan` wrapper** — CLI wrapper that sets `LD_PRELOAD=libasan.so.8` and exec's into the real fuzzer (mentioned in ASAN-LIMITATION.md but not yet generated as a installable entry point)
- [ ] **Persist `invocation` into `state.json`** — `fuzzer.invocation` (sys.argv, captured in `cmd_fuzz` for the report exec lines) is not saved on shutdown; a `--resume` run therefore reports only the resumed command, not the original one. Saving it into `state.json` would let reports on resumed sessions carry the original invocation.

## Pending Bugs
- [x] `parse_protobuf` crashes on deeply nested group fields (>16 levels) — `_parse_fields` returned bare `None` at the depth limit instead of `(None, [])`, causing `TypeError: cannot unpack non-iterable NoneType`
- [ ] `_apply_single_mutation` havoc doesn't enforce `max_len` strictly (allows +1 byte per insert, up to +8 total)
- [ ] `parse_dict_line` triple-encode chain fragile for bytes > 0x7F
