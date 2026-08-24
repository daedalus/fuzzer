# Port candidates — internet survey (2026-08-25)

Four parallel web-research passes: (A) production-fuzzer feature parity
(AFL++ 4.x, libFuzzer/clang, honggfuzz, E9Patch), (B) academic 2022–2026
(USENIX Sec / S&P / CCS / NDSS / ICSE / ISSTA / arXiv), (C) grammar/semantic/
LLM-guided fuzzing, (D) execution substrate (snapshots, HW tracing,
emulation, crash triage). Raw yield ~45 candidates; deduplicated to the rows
below.

**Audit before writing.** Grep of `src/` + `docs/TODO.md` for every candidate:
all absent from the tree except two caught by the audit — **K-Scheduler**
(already audited, `docs/kscheduler_centrality_port.md`; W1 artifacts wired via
`core/cfg_cache.py`) and SymCC (comment-level reference only,
`core/path_constraints.py:20`). K-Scheduler is excluded from the tables.

Same caveat as `six_source_technique_port.md`: drafted from sources alone.
Effort estimates are guesses until a per-item audit happens against live code.

## Status

**2026-08-24**: Items #1, #2, #3 landed (see `docs/TODO.md` Scheduling section). #3
(`cluster_crashes`) already existed in `core/crash_metadata.py` but was dead code —
it is now wired into `services/report.py::_crash_signatures`. #4–7 remain: #4/#5
require `afl_shim.c` changes (excluded from this pass); #6/#7 are `L` effort
needing new feedback plumbing, not yet started.

**2026-08-24 (later)**: #8 (Subtree-population crossover) landed. `TreeMutator`
already documented a "subtree splice" op that was never implemented; added
`SubtreePopulation` (bounded reservoir per rule name) and `_tree_splice`,
wired into `mutate_tree` and incrementally populated from the corpus in
`services/operators.py::_op_grammar_tree_mutate`.

### Tier 1 — quick wins

| # | Candidate | Source | Mechanism | Lands in | Effort |
|---|---|---|---|---|---|
| 1 | autotokens (✅ landed 2026-08-24) | AFL++ | Tokenize ASCII corpus into whole-token dictionary entries; no grammar needed | `import_corpus.py::build_autotoken_dictionary`/`extract_tokens`, `import --autotokens` | Trivial |
| 2 | Entropic power schedule (✅ landed 2026-08-24) | libFuzzer `-entropic` | Energy ∝ log(rare-feature count), updated from feature-frequency histograms we already collect | `SeedScorer._entropic_factor` (`schedules.py`), `--schedule entropic` | Trivial |
| 3 | Stack-hash crash clustering (✅ landed 2026-08-24) | ClusterFuzz; CASR/LibCASR (embeddable Python API, maintained 2024–25) | Exact-match then LCS-distance stack similarity + hierarchical clustering over ASAN replay output | `report.py::_crash_signatures` now calls the pre-existing `core/crash_metadata.py::cluster_crashes` | Trivial–L |
| 4 | trace-div / trace-gep | clang `-fsanitize-coverage=trace-div,trace-gep` | Callbacks carry every non-constant divisor/GEP index — dynamic operand feedback complementing static DIV extraction | cmplog-style handlers in `afl_shim.c` plumbing | L |
| 5 | N-gram edge coverage | AFL++ (`AFL_NGRAM_ENV`, RAID'19) | Hash last-N executed edges into the map key; separates deep parser states that collide under lone-edge hashing | map-update arithmetic in `afl_shim.c` (CTX already default-on; n-gram is the missing sibling) | L |
| 6 | Zest validity channel | Zest/JQF (ISSTA'19) | Input *validity* (parser acceptance rate) as second fitness channel alongside coverage; saves valid-and-new inputs to steer past syntax checks into semantic stages | outcome recording in `runner.py`/`fuzzer.py` + scheduler feature | L |
| 7 | SGFuzz enum-state variables | SGFuzz (USENIX Sec '22) | Regex/AST-extract enum-typed state variables, instrument assignments, runtime State Transition Tree as feedback; upstream ships a Python instrumentation script; composes with Markov generation | new instrumentation pass + feedback feature | L |

### Tier 2 — medium effort, high value

| # | Candidate | Source | Mechanism | Lands in | Effort |
|---|---|---|---|---|---|
| 8 | Subtree-population crossover (✅ landed 2026-08-24) | GRIIN (ASE '23); Grammarinator×AFL++ (2026) | Global subtree population for grammar-aware tree crossover — measured as the biggest single win of that integration | `core/grammar.py::SubtreePopulation`/`TreeMutator._tree_splice`, wired in `services/operators.py::_op_grammar_tree_mutate` | L–M |
| 9 | FormatFuzzer decision seeds | FormatFuzzer (USENIX Sec '21, `uds-se/FormatFuzzer`) | Compiles community 010 Editor binary templates (170+ formats incl. MP4/PNG/AVI/ZIP) into parser+generator pairs; byte fuzzer mutates choice bits while output stays valid | template-driven generator family generalizing the hand-written mutators | M |
| 10 | Grammar-aware reduction | Perses (ICSE '18); ProbDD/WDD (ICSE '25) | Reduce along grammar/token trees (~2% of ddmin output size); ddmin-to-fixpoint alone shrinks ~68% further | `tmin.py` | L–M |
| 11 | Reaching-probability directed mode | SelectFuzz (IEEE S&P '23) | Block fitness = averaged successor reaching probability instead of graph distance; instrument only target-relevant blocks (<2% of reachable BBs) so irrelevant coverage never pollutes feedback | `distance.py` math + one LLVM pass | L–M |
| 12 | EcoFuzz energy allocation | EcoFuzz (USENIX Sec '20) | Adversarial-MAB estimating per-seed new-path reward probability against observed average cost; −32% execs at equal coverage | scheduler arm in `schedulers/` + `SeedScorer` | L |
| 13 | Reachable-uncovered weighting | PrescientFuzz (arXiv 2024) | Per-seed BFS-counted reachable-uncovered blocks from its trace, inverse-rarity + depth weighted — nearly free on our CFG/DWARF pipeline | seed-scoring feature atop `cfg.py`/`cfg_cache.py` | L |
| 14 | Concolic query piping | symcc/SymQEMU (bundled with AFL++) | Branch constraints collected during execution piped into solver loop; goes beyond redqueen encodings on nested comparisons | feed `smt_solver.py`/`z3_lifecycle.py`; concolic loop already sketched at `path_constraints.py:20` | M |
| 15 | Deviating-block probing taint | WindRanger (ICSE '22) | Static+dynamic identification of "deviation basic blocks" en route to targets; effector-map probing maps stubborn branches to controlling offsets — cmplog solves comparisons globally, this localizes them | probing pass + mutation-position bias | M |
| 16 | Learned mutation-field gradients | IDFuzz (USENIX Sec '25) | Small network trained on historically productive mutations near targets; gradients locate critical input fields; −91.9% ineffective mutations, 2.48× faster CVE repro | mutation position/value selection | M |
| 17 | Static rewriting of uninstrumented binaries | E9Patch/E9Tool/E9AFL (`GJDuck/e9afl`) | Instruction-punning trampoline injection into stripped x86-64 ELFs at near-native speed — vendor/shipped binaries without rebuild | offline e9tool step reusing shim runtime (grep-class targets) | M |
| 18 | userfaultfd write-protect snapshots | uffd-fuzz (2022) | Pure-userland dirty-page tracking (wp-mode): pristine bytes memcpy'd back per iteration, restore <2 µs, ~1.8× median over persistent-mode fork reset; x86-64-only | extension of `afl_shim.c` + `inprocess.py`; must intercept mmap/mprotect | M |
| 19 | LBR branch sampling | perf `PERF_SAMPLE_BRANCH_STACK` (Haswell+) | Last-32-branches statistical samples at near-zero cost; secondary signal for uninstrumented dependency code | one more event type in `perf_event.py` | L |

### Tier 3 — heavier bets

| # | Candidate | Source | Mechanism | Lands in | Effort |
|---|---|---|---|---|---|
| 20 | DataFlowTrace taint + focus_function | libFuzzer/OSS-Fuzz (DFSan) | Byte→function taint traces concentrate mutations on relevant bytes and energy on a focus function — exact mapping vs colorization's trial-and-error | second dfsan-instrumented build + Python trace collector | M–H |
| 21 | Input Processing Tree auto-repair | NestFuzz (CCS '23) | DFSan-taint-derived inter-field/nesting dependencies; cascading mutation auto-repairs length/offset/container fields when mutating nested structures | automates what `frameshift.py` + hand-written mutators do manually | M–H |
| 22 | Plateau-triggered LLM generation | ChatAFL (NDSS '24) | Local LLM extracts message grammars, enriches corpus with missing types, generates targeted inputs when coverage saturates (+47% states vs AFLNet) | orchestrator hook + local LLM server | M |
| 23 | LLM-written generator programs | SeedMind/SeedSmith (arXiv 2411.18143, 2607.08949) | LLM writes *generator programs* refined by execution/coverage feedback (<$0.5/harness); works with any downstream fuzzer | complements hand-written mutators for uncovered formats | L–M |
| 24 | Fuzzer-space evolution | ELFuzz (USENIX Sec '25) | LLM-driven evolution of generation-based fuzzers embedding grammar+semantic constraints; ran fully locally on CodeLlama-13B, found 5 cvc5 0-days | alternative maintenance path for the mutator set | M |
| 25 | JIT constraint evaluation | JIGSAW (IEEE S&P '22) | Branch constraints JIT-compiled to native functions, Angora-style gradient descent at ~600K–12M evals/sec; evidence suggests demoting Z3 to fallback | `smt_solver.py` repositioning; cheap first step = pure-Python numeric-gradient evaluator | M–H |
| 26 | Root-cause crash clustering | Igor (CCS '21) | Minimize PoC traces via coverage-reduction fuzzing, cluster by CFG similarity (Weisfeiler-Lehman kernel); kept 39 real bugs distinct where stack hashes inflate counts 10–100× | beyond #3's dedup; reuses edge maps + #10 minimizer | M–H |
| 27 | Intel PT coverage | honggfuzz `--linux_perf_ipt_block` + libipt; PTrix (AsiaCCS '19) | Module-free PT via perf AUX mmap buffers on kernel ≥4.2; indirect-block-only decoding is the proven stability baseline (~10–15% overhead) | `perf_shim.c` extension + libipt decode thread | M–H |
| 28 | Generator fault-injection | Fuzztruction (USENIX Sec '23) | Fault-inject compile-time-instrumented *generator* programs so outputs are almost-valid — bypasses parsing/CRC/encryption checks wholesale (different route than the checksum-learning stack) | per-target generator/consumer pairing | H |
| 29 | Full-VM snapshot reset | Nyx / AFL++ Nyx mode (`nyx-fuzz`) | KVM full-state snapshots thousands/sec; runs instrumented userland targets on vanilla kernel ≥5.11; `nyx_packer.py` is Python | only if slow-init/stateful targets dominate | H |

## Cross-cutting findings

- **Hidden-edge undercount**: AFL++ 4.35a pcguard rewrite found ~5–8% of edges
  missed by vanilla LLVM sancov ("hidden" decisions). Our shim sits on the same
  callbacks — worth a probe against a vendored ffmpeg build once available.
- **AFL++ 4.20c changed the forkserver protocol** (new targets incompatible
  with old afl-fuzz). Matters only if we interop with stock AFL++ binaries.
- **Directed fuzzing is the field's active frontier**: ≥4 papers at USENIX Sec
  '25 alone (IDFuzz, Lyso, WDFuzz, ELFuzz).
- **Local small LLMs are validated in-loop**: ELFuzz ran CodeLlama-13B on one
  GPU; SeedMind <$0.5/harness — cost-effective without API dependence.
- **Crash-count ground truth (Igor)**: coverage-profile dedup inflates bug
  counts 2–3 orders of magnitude; stack hashes 1–2 orders; LCS-based
  clustering (#3) is the cheapest accuracy upgrade.

## Excluded after review

- **CoreSight/ETM** (AFL++ coresight_mode, RAID'24 Stalker) — requires ARM64
  SoCs + u-dma-buf module; inapplicable to our x86-64 host.
- **PEBS sampling** — dominated by LBR (#19) for coverage purposes.
- **Snappy adaptive snapshots, GPTrace (ICSE '26)** — immature/heavyweight;
  watch-list only.
- **DARWIN, MobFuzz, ParmeSan, DAFL/DeepGo/Lyso/Prospector, HyLLfuzz** —
  overlap existing GA/MOpt/hierarchical-bandit/directed-distance stack.
  DeepGo's RL path-transition model worth revisiting if directed mode plateaus.

## Sources

| Tag | Source |
|---|---|
| A | AFL++ docs/changelog 4.x; LLVM SanitizerCoverage docs; honggfuzz source/docs; E9Patch/E9AFL guides |
| B | SelectFuzz S&P'23; SGFuzz USENIX'22; WindRanger ICSE'22; JIGSAW S&P'22; BEACON S&P'22; IDFuzz USENIX'25; NestFuzz CCS'23; K-Scheduler S&P'22 (audited separately); EcoFuzz USENIX'20; Fuzztruction USENIX'23; PrescientFuzz arXiv'24 |
| C | FormatFuzzer USENIX'21/TOSEM'23; Zest ISSTA'19; GRIIN ASE'23 + Grammarinator×AFL++ '26; Superion ICSE'19; ChatAFL NDSS'24; SeedMind arXiv'24; SeedSmith arXiv'26; ELFuzz USENIX'25; TitanFuzz ISSTA'23/FuzzGPT ICSE'24; Fuzz4All ICSE'24; GLADE ICSE'17/REINAM ASE'19; Autogram/Mimid/ISLearn; AFLNet/StateAFL lineage |
| D | CASR/LibCASR (ISPRAS); Perses + ProbDD/WDD; qemuafl/frida-afl; uffd-fuzz '22; kAFL/PTrix/Honeybee; nyx-fuzz; Igor CCS'21; AURORA USENIX'20; unicornafl v3 |

Not ported from research but noted for context: Superion-style structure-aware
havoc (#8 adjacent), stateful protocol feedback à la AFLNet (file-format analog
= demuxer progression states; med-high effort, parked).
