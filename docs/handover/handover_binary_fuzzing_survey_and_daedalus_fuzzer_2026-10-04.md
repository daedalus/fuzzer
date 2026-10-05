# Handover — Binary Fuzzing Survey + daedalus/fuzzer Snapshot

**Date**: 2026-10-04 (updated evening with ROSA)  
**Author**: Grok (survey + clone attempt + ROSA analysis); Claude (clone verification + FuzzBench port analysis, section 5)  
**Status**: Complete survey; ROSA (ICSE'25) incorporated; FuzzBench port analysis present  
**Related**: `https://github.com/daedalus/fuzzer` (master, ~2.3k commits, MIT)

> Standing note: This document records an external survey of recent binary-only fuzzing techniques, a first-pass inspection of the daedalus/fuzzer (fuzzer-tool) repository, and an analysis of ROSA (ICSE 2025). It is written in the style of the existing `docs/handover/` series so it can be dropped into the tree without format friction.

---

## 1. Repository Snapshot — daedalus/fuzzer

**One-line description** (from README):  
Information-dense, coverage-guided binary fuzzer: 148 mutation operators across 9 categories, 16 bandit/optimizer scheduler modules under Elo arbitration, AFL-style forkserver + SHM edge coverage, comparison tracing down to the individual call site, and information-theoretic seed scoring.

**Honest caveat** (author):  
Probably the most complex fuzzer from an information-theory standpoint, and also the slowest raw-throughput. The tradeoff is speed for edge-discovery novelty. For production fuzzing at scale, AFL-family tools remain the best choice.

### Core Capabilities (condensed)

| Area | Highlights |
|------|------------|
| **Mutation** | 148 ops in 9 categories (bit / byte / block / dict / structural / radamsa / format / regularity / adaptive). Heavy format-aware support: PNG, JPEG, BMP, gzip/zlib, PGS, ISO-BMFF, NAL, MPEG-TS, ADTS, MP3, Ogg, FLV, ASF, RIFF, Protobuf, GIF, WebP, WebM, ZIP, AVIF, SQLite, DER, ELF, x86/ARM. FrameShift length repair, Berlekamp-Massey checksum learning, adaptive havoc. |
| **Coverage & Exec** | AFL SHM bitmap (sparse), forkserver (`afl_shim.c`), ptrace+Capstone for closed-source, in-process (ctypes), hardware perf counters, n-gram edges, PerfFuzz-style hit-count maxima, Bloom dedup, persistent mode, network targets, differential multi-target. |
| **Scheduling** | Elo arbitration over Thompson, CEM, MOpt, EXP3, hierarchical, GP-UCB, CMA-ES, contextual LinUCB, discounted/sliding/combinatorial UCB, MCTS, Boltzmann, Kruskal-count, entropy-KL, Good-Turing, etc. Information-theoretic seed scoring (rarity, Chao2, residual risk). |
| **Other** | RedQueen/colorization/SMT comparison tracing, AFLGo-style directed, ASAN/MSAN/TSAN/UBSAN, state persistence, 5 400+ tests claimed, real trophies (FFmpeg, fgrep, …). |

Architecture is a campaign loop with three feedback colours: coverage/operator reward (green), comparison signal (blue), corpus/seed selection (purple). Source of truth for layering lives in `docs/ARCHITECTURE.md` + `docs/architecture.dot`.

**Clone status**:  
The earlier shallow-clone timeouts no longer reproduce: `git clone --depth 1` and `git pull` of `daedalus/fuzzer` completed cleanly (checked at `2b51500`, then pulled to `36c85b0`), and `src/fuzzer_tool/` is fully materialised. The capability table above still comes from the README; section 5 was checked against the source.

---

## 2. Survey of Recent Binary Fuzzing Techniques (2024–2026)

Focus: binary-only / COTS techniques that improve on classic QEMU/FRIDA/ZAFL baselines.

### 2.1 Lightweight Feedback that Avoids Rewrite / Emulation / Hardware

- **SPFuzz / SPFuzz++** (USENIX Security 2025)  
  System-call *pattern* coverage as the primary feedback signal.  
  No binary rewriting, no emulator, no Intel-PT.  
  Works on any binary that issues syscalls.  
  Results: comparable (sometimes superior) branch coverage vs classic edge coverage; up to 41× faster than AFL-QEMU on some targets; six new CVEs (incl. CUDA tooling).  
  Handles binaries that six baseline binary-only fuzzers could not run at all.  
  Paper: https://www.usenix.org/system/files/usenixsecurity25-xiao-jifan.pdf

- **TraceLib** (arXiv 2026) — system-call bitmap for language-agnostic web fuzzing; projects selected syscalls + argument hashes into an AFL-style 65 536-slot map.

### 2.2 Static Binary Rewriting Advances

- **PeAR** (arXiv 2026, GTIRB-based)  
  Static instrumentation for AFL++ (Linux x64/ARM64) and WinAFL (Windows).  
  Supports deferred initialisation, persistent mode, shared-memory test-case delivery.  
  Instruments ~88 % of FuzzBench; median ~4× throughput gain with persistent+SHM; coverage competitive with compiler instrumentation.  
  Demonstrates that modern SBI frameworks (GTIRB) make static rewriting practical again.  
  https://arxiv.org/abs/2606.02126 · https://github.com/avncharlie/PeAR

- Continues the ZAFL / RetroWrite / StochFuzz line. AFL++ still recommends ZAFL, RetroWrite, FRIDA persistent, QEMU persistent, Nyx as the practical binary-only toolkit.

### 2.3 Data-flow & Hybrid Guidance

- **FuzzRDUCC** (arXiv Sep 2025)  
  Reconstructs def-use chains from binaries via selective symbolic execution + heuristics; feeds them as coverage. Finds unique crashes on binutils.

- **VSGFuzz** (2025) — RL-guided mutation that scores functions by estimated vulnerable-state probability + coverage reward.

- Multi-angle schemes that combine hardware tracing, static path-complexity metrics and concolic execution for energy allocation.

- **KBinCov** — kernel binary coverage via memory-access pattern abstraction; integrated into Syzkaller; more bugs found than kcov/StateFuzz/IJON baselines.

### 2.4 Specialised / LLM-Augmented

- **Bin2Wrong** (USENIX ATC 2025)  
  Unified mutation of source + compiler flags + optimisation + executable format for *decompiler* testing.  
  10–17× higher binary diversity, more decompiler coverage, 48 new bugs (30 confirmed); one bug forced a major Binary Ninja redesign.

- LLM + reverse-analysis pipelines (CFG + dynamic input-dependency trajectories → structured text → fine-tuned LLM predicts high-risk regions). Reported ~34 % higher vulnerability discovery rate vs AFL on LAVA-M + closed-source binaries.

- **LATTE** — first fully automated static binary taint analysis powered by LLM; found 37 new firmware bugs (7 CVEs) that prior tools missed.

- GitHub Security Lab Taskflow Agent (2026) — autonomous harness generation → AFL++ → coverage-gap chasing → crash triage, with LLM only in the decision loop.

- **BinSleuth** — automated sink-to-source slicing + path-reduction for under-constrained symbolic execution; large speed-ups and better real-world detection than Arbiter / hybrid baselines.


### 2.5 Backdoor Detection via Fuzzing — ROSA (ICSE 2025)

**Paper**: *ROSA: Finding Backdoors with Fuzzing*  
**Venue**: ICSE 2025 (Best Artifact Award)  
**Authors**: Dimitri Kokkonis, Michaël Marcozzi, Emilien Decoux (CEA List), Stefano Zacchiroli (Télécom Paris)  
**Nutshell**: https://binsec.github.io/nutshells/icse-25.html  
**PDF**: https://binsec.github.io/assets/publications/papers/2025-icse.pdf  
**Tool**: https://github.com/binsec/rosa (Rust + patched AFL++/QEMU-AFL)  
**Benchmark**: https://github.com/binsec/rosarum (ROSARUM)  
**arXiv**: 2505.08544

**Problem**  
Code-level backdoors (hard-coded credentials, logic bombs, etc.) have been injected via supply-chain attacks (PHP, ProFTPD, vsFTPd, xz, router firmware). Manual reverse-engineering does not scale; prior automated approaches still require substantial binary RE and cover only a narrow class of backdoors.

**Core idea**  
Pair AFL++ with a **metamorphic test oracle** that detects anomalous runtime behaviour:

1. **Phase 1** (≈ 1 min) — collect *representative inputs* that cover new CFG edges together with their system-call traces.
2. **Phase 2** (hours) — for every new input, search the representative set for one that covers a similar edge set *and* emits the same syscall pattern. No match → flag as suspicious (possible backdoor trigger).
3. **Post-processing** — human inspects the small set of suspicious inputs under `strace`. Because the full triggering input *and* the divergent syscall trace are already provided, no manual binary reverse-engineering is required.

**Classic illustration (Sudo)**  
Normal wrong passwords produce similar failure + syscall traces. The backdoor password `"let_me_in"` produces authentication success + fork/exec → clearly divergent syscalls → flagged.

**Evaluation (ROSARUM)**  
- 17 authentic + synthetic backdoors of different varieties.  
- 10 × 8 h runs per program.  
- **Detects all 17** backdoors in **1 h 30 min** on average.  
- Average **≈ 7** suspicious inputs to vet.  
- vs. prior SOTA (RE-based): finds all (prior found 4/17), **44× fewer** suspicious inputs, no manual RE needed.

**Strengths**  
Fully binary-only, high automation, low false-positive burden, strong open artifact + Docker images, works on closed-source firmware.

**Limitations**  
- Assumes the backdoor produces a *noticeably different* syscall footprint (silent/stealthy backdoors may evade).  
- Still needs a short human vetting step.  
- Currently Linux x86_64 only.  
- Representative-input selection received minor corrections in the camera-ready (v2 PDF).

**Relevance to daedalus/fuzzer**  
ROSA sits in the same family as SPFuzz / TraceLib (syscall-pattern signals) but uses them as a *metamorphic oracle* rather than pure coverage. A natural experiment would be to add a syscall-trace similarity / outlier channel under the existing Elo arbitrator, turning daedalus/fuzzer into a dual-purpose vulnerability + backdoor hunter.

### 2.6 Practical Baseline Recommendation (2026)


| Scenario | Recommended starting point |
|----------|---------------------------|
| Linux binary, can rewrite | PeAR or ZAFL + AFL++ persistent/SHM |
| Linux binary, cannot rewrite | FRIDA persistent or QEMU persistent; fallback SPFuzz-style syscall patterns |
| Full-system / kernel | Nyx / KVM snapshot + KBinCov-style feedback |
| Closed-source Windows | WinAFL via PeAR or DynamoRIO |
| Backdoor hunting (binary-only) | ROSA (AFL++ + metamorphic syscall oracle) |
| Research / novelty | daedalus/fuzzer-style information-theoretic + multi-bandit schedulers |

---

## 3. Open Questions / Follow-ups for daedalus/fuzzer

1. ~~Full shallow clone still needed~~ Resolved 2026-10-04, see Clone status.  
2. How many of the 16+ bandit modules are actually exercised under the default Elo arbitrator on real targets (png, grep, FFmpeg, fuzzgoat)?  
3. Measured overhead of the regularity / statistical operators vs classic havoc.  
4. Whether syscall-pattern feedback (SPFuzz) or def-use reconstruction (FuzzRDUCC) can be bolted on as additional channels without destroying the existing Elo reward model.  
5. Trophy-case reproduction scripts — confirm they still build under current `tools/build_targets.sh`.

---

## 4. References (primary)

- daedalus/fuzzer README & tree — https://github.com/daedalus/fuzzer  
- SPFuzz — USENIX Security 2025, Xiao et al.  
- PeAR — arXiv:2606.02126  
- FuzzRDUCC — arXiv:2509.04967  
- Bin2Wrong — USENIX ATC 2025, Yang & Nagy  
- **ROSA** — ICSE 2025, Kokkonis et al.  
  - Nutshell: https://binsec.github.io/nutshells/icse-25.html  
  - PDF: https://binsec.github.io/assets/publications/papers/2025-icse.pdf  
  - Tool: https://github.com/binsec/rosa  
  - Benchmark: https://github.com/binsec/rosarum  
  - arXiv: 2505.08544  
- AFL++ binary-only guide (current recommendations)

---

6. Feasibility of a ROSA-inspired "backdoor mode": collect representative (edge, syscall-trace) pairs, then flag inputs whose syscall footprint diverges from nearest neighbours while coverage stays similar.

---

## 5. FuzzBench Port Analysis (2026-10-04)

Compared `google/fuzzbench` (72 fuzzer definitions under `fuzzers/`) and the upstream sources of the less common ones (Pythia, TortoiseFuzz, LearnPerfFuzz, WingFuzz, FaFuzz, DARWIN, AFL randomized_top_rated) against `src/fuzzer_tool/`. Method: keyword grep over `src/`, plus reading the upstream diffs. This is a technique comparison, not a line-by-line audit.

### 5.1 Already covered (do not port)

| FuzzBench fuzzer | Where it lives here |
|---|---|
| FairFuzz | `core/schedulers/pos_rare_mask.py` |
| AFLFast / lin / quad / coe | `core/schedules.py` (`SeedScorer`) |
| MOpt, EcoFuzz, Weizz, Neuzz | `op_*` / `seed_*` / `pos_*` schedulers, `core/weizz_tags.py` |
| Redqueen / cmplog / laf-intel (CompCov) | `core/cmplog.py`, `afl_shim.c` |
| Centipede data-flow features | `afl_shim.c` trace-loads/stores, `tools/build_targets.sh` |
| Pythia (residual risk) | Good-Turing in `report.py` / `stats.py` |
| DARWIN (evolving operator distribution) | `core/schedulers/op_cmaes.py` (assumed; DARWIN's own algorithm was not read) |
| PerfFuzz / LearnPerfFuzz max-hit maxima | `core/edge_tracker.py` (`max_hit_count`), `services/fuzzer.py` `_perf_novelty`, `services/fuzz_round.py` |
| AFL virgin map / count classes | `core/count_class.py`, `adapters/shm.py` |

**Correction**: an earlier pass of this analysis listed per-edge max hit counts (PerfFuzz) as missing. That was wrong; it is implemented and gated by `--perf-novelty`. The first grep missed it because the code says "performance novelty", not "PerfFuzz".

### 5.2 Candidates to port, in priority order

1. **TortoiseFuzz coverage accounting.** Weights each edge by security impact (calls to `memcpy`/`memmove`/`memset`/`memcmp`, allocators, memory ops) and favours seeds that reach new high-impact edges. Upstream: `func_metric/` and `bb_metric/` LLVM passes. Here only the format learner tracks `sensitive_ops`; nothing weights edges or the favored set by impact. Feasible with the existing sancov pc-table plus trace-loads/stores.
2. **WingFuzz compare filtering.** `instrument/LoadCmpTracer.cc` drops compares on loop back-edges (dominator-tree test) and outside 8-64 bit integer widths, and splits var/ord/const operands. The build already uses trace-cmp; filtering would cut cmplog noise. No back-edge or width filter was found in `afl_shim.c` or `tools/`.
3. **Randomized top_rated.** Upstream (`Practical-Formal-Methods/AFL-public`, branch `randomized_top_rated`) breaks ties for the favored set with a per-seed random number instead of a fixed cost. `_compute_favored` in `services/fuzzer.py` is deterministic (`exec_us * input_size`, then edge id), so the same seed always wins a given edge. A few-line change; effect on coverage is unmeasured here.

### 5.3 Not worth porting

- `aflplusplus_um_*`: needs source-level mutants (`mutate`, `prioritize_mutants`), C/C++ only.
- KLEE, SymCC, SymSAN, SymQEMU: overlap with `core/smt_solver.py` and `core/path_constraints.py`.

### 5.4 Not verified

- DARWIN's algorithm (README has no description; the `op_cmaes` equivalence is an assumption).
- FaFuzz's custom havoc stage (`fa_havoc_fuzzing_one`) and `aflpp_random_wrs*` were not read.
- HasteFuzz upstream is a README-only snapshot.
- None of the three candidates was prototyped or benchmarked.

---

*End of handover (updated with ROSA ICSE'25). Drop into `docs/handover/` and link from `docs/TODO.md` or `CHANGELOG.md` if desired.*
