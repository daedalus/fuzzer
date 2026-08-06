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
- [x] **Fault-address extraction (PTRACE_GETSIGINFO)** — the ptrace runner captures `si_addr` + registers at fatal-signal stops; non-sanitizer crash signatures become `signal:N@0xaddr` so same-signal crashes at different addresses dedup separately; `--trace-crashes` reports show both `RIP:` (`crash_rip`) and `Fault:` (GDB `$_siginfo`). Extended to all `.so` crash paths: the persistent loader's grandchild self-traces (`PTRACE_TRACEME` + `WUNTRACED` stop loop, relayed over the RC line), and direct_lite re-runs crashing inputs through the ptrace-attached loader script (`TargetRunner._run_triage_ptrace`) for full triage.
- [x] **GDB crash replay in the crash report** — every saved crash `.txt` embeds a `=== GDB crash replay ===` section automatically (when gdb is installed): signal, RIP/Fault, registers, backtrace, and DWARF source (`file:line` + the crashing source line). `.so` targets are driven via a ctypes harness calling the probed fuzz entry point (stdin input), so non-ASAN shared-library crashes get a real gdb backtrace in the report.
- [ ] **Root cause diff** — show minimal byte diff from nearest non-crashing input to root-cause bytes.

## Performance
- [x] **Forkserver mode** — see Coverage section above.
- [x] **Cmplog optimization** — adaptive periodic collection (1 in 20 iterations once pool exceeds 2000 entries).
- [x] **Corpus distillation on-the-fly** — `--max-corpus` triggers auto-minimization when corpus exceeds threshold.
- [x] **CLI hotpath profiling** — `--profile-hotpath` cProfile integration into fuzz mode (tottime/cumtime/ncalls tables + `.prof` dump via `--profile-out`).
- [x] **array.array for cold bounded histories** — corpus-size history, the four tuple histories (discovery/crash-rate/entropy/coverage-timeline), elo prediction-error lists, redqueen pair-length index converted to `array("Q")`/`array("d")`/`array("I")` (7-9x memory per container, runtime verified neutral). EdgeTracker edge-count maps deliberately left as sparse dicts (array.array loses on scalar RMW + iteration; reads already numpy-vectorized).
- [x] **Bound the MI tracker's memory** — the joint (position x byte x edge) was unbounded; `--elo all` on a real corpus OOM-killed at 7.5GB/796MB mi.json with EPS collapsing 1500->20. Now: hard cell cap (MAX_JOINT_CELLS) rejecting new cells at budget (no eviction thrash), MI_MAX_POSITIONS cap independent of max_len, and mi.json loaded only with `--resume`. Measured: 7.5GB -> flat 229MB, EPS 11 -> 180+, run completes where it previously OOM'd at ~1.2k execs.
- [x] **Stabilize the first stats line's EPS** — the first `[*] execs` line could show inflated EPS (bursty warm-up, startup time in the denominator), and the stats-interval calculation fed that single raw reading back as `10 * last_eps`, stretching the gap to the next line. Now the first tick prints at ~1 second of work (1x EPS), and the effective tick spacing is `10 * mean(last 10 avg-eps samples)` (falling back to the fixed `stats_interval` while the window fills).
- [x] **Branch-pruning hot-path micro-opts** — profiled `png_read_dist.so` (2k execs): per-exec Python cost is `mutate`/`available`/`_check` ~44µs, seed picking ~50-97µs (weight-cached), `_check_new_coverage` ~23µs, `_compute_crps` ~15µs. Landed: CRPS fused to `np.dot` (one less temp array/exec, numpy import hoisted) and `select_position` gates the sensitivity call like MI/TE. ~2-3% wall-clock faster (100k-exec A/B, alternating identical corpus). Open lever: `REGISTRY.available()` re-evaluates data-independent predicates per exec — memoize on state fingerprint.

## Infrastructure
- [ ] **Dockerfile** for reproducible builds and CI
- [ ] **Structured logging** (e.g. `--log-json`) for machine-parseable output
- [ ] **`fuzzer-tool-asan` wrapper** — CLI wrapper that sets `LD_PRELOAD=libasan.so.8` and exec's into the real fuzzer (mentioned in ASAN-LIMITATION.md but not yet generated as a installable entry point)
- [ ] **Persist `invocation` into `state.json`** — `fuzzer.invocation` (sys.argv, captured in `cmd_fuzz` for the report exec lines) is not saved on shutdown; a `--resume` run therefore reports only the resumed command, not the original one. Saving it into `state.json` would let reports on resumed sessions carry the original invocation.

## Pending Bugs
- [x] `parse_protobuf` crashes on deeply nested group fields (>16 levels) — `_parse_fields` returned bare `None` at the depth limit instead of `(None, [])`, causing `TypeError: cannot unpack non-iterable NoneType`
- [x] Ptrace breakpoint handler read `rsp` from `user_regs_struct` offset 176 (`gs_base`, 0 for the main thread) instead of 152 — every function-entry breakpoint skipped its first instruction, corrupting the tracee (spurious stack-address SIGSEGVs, zero edge coverage)
- [x] Ptrace wait loop conflated "no event" `(0, 0)` with a clean exit `(pid, 0)` (tested `status == 0` instead of the PID) — every rc=0 exit in ptrace mode was misreported as `-2` and its input saved to the corpus
- [x] Subprocess loader's standalone-exec branch clamped signal-killed executables to exit 0 (`sys.exit(max(0, min(proc.returncode, 125)))` — a SIGSEGV'd target's `-11` became a clean 0, invisible to `is_crash`). Now exits `128+signum` (139/134), which `SIGNAL_CRASH_CODES` recognizes; timeouts exit 137 which is not a crash code.
- [ ] `_apply_single_mutation` havoc doesn't enforce `max_len` strictly (allows +1 byte per insert, up to +8 total)
- [x] `targets/test_target.c` crash paths vanish under `-O2` — clang eliminated the direct `((void(*)())0)()` NULL-jump as UB (fuzz_test compiled to just `ret`). Fixed by calling through a `void (*volatile)` pointer, which the compiler must keep; the target now crashes at `-O2` too (verified: 66 crashes in a 5k e2e vs 0 before).
- [ ] `parse_dict_line` triple-encode chain fragile for bytes > 0x7F
- [x] `fuzzer-tool` startup hang for ASAN+UBSAN targets — `ldpreload_wrapper` preloaded libasan + the UBSAN standalone together into the interpreter, tripping an ASAN init CHECK (`sanitizer_signal_interceptors.inc`) and hanging before the fuzz loop; preloading libasan onto targets that link their own runtime (defined `__asan_init`, e.g. `png_read`) also silently killed the child's AFL SHM coverage. Detection now keys off strong undefined (`U`) symbols only: defined (`T`) and weak-undefined (`w`) references never trigger a preload, and the fuzzer's existing per-child LD_PRELOAD fallback covers in-process modes. Verified: `fuzzer-tool fuzz targets/png_read` starts, SHM edges climb 43→102, corpus grows 202→249.
- [x] `--elo all` corpus frozen at 1 seed — QEA's corpus bypass (`if not f.qea: f.corpus.append()`) froze the in-memory corpus at its initial size while Elo arbitrated corpus-based seed strategies (weighted/pareto/bayesian/boltzmann) that read `f.corpus`: those arms starved onto the initial seeds, so large campaigns showed `corpus: 1`, `rate: 0.0 ed/kexec`, `P(stall): 100%` and stalled. Seeds were still written to disk (1→12 files) but never entered the picker pool. Fix: keep the bypass only for standalone `--qea`; under Elo arbitration (`_use_elo`) append applies. Verified: `--elo all` corpus 10→25→40, discovery resumes.
- [x] Resume-inflated EPS — `--resume` restores `exec_count` from state while `start_time` is the fresh process start, so the display `eps = exec_count/elapsed` reported absurd rates (2.335M execs over 1s of wall time, decaying on every line). `load_state` now records the restored count as the session baseline (`_resume_baseline_exec`) and resets the Kalman interval counter (`_last_eps_count`); the stats line, peak-eps, dict-eps window, and first-tick stats interval all subtract the baseline so they measure only the current process.
