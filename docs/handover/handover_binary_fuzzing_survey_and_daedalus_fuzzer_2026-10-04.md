# Handover — Binary Fuzzing Survey + daedalus/fuzzer Snapshot

**Date**: 2026-10-04  
**Author**: Grok (survey + clone attempt)  
**Status**: Complete survey; partial clone only (environment timeouts on large repo)  
**Related**: `https://github.com/daedalus/fuzzer` (master, ~2.3k commits, MIT)

> Standing note: This document records an external survey of recent binary-only fuzzing techniques and a first-pass inspection of the daedalus/fuzzer (fuzzer-tool) repository. It is written in the style of the existing `docs/handover/` series so it can be dropped into the tree without format friction.

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

**Clone status in this environment**:  
Shallow clones repeatedly timed out or left incomplete trees (only top-level docs + `.git` remnants survived). Full source under `src/fuzzer_tool/` was never fully materialised. All technical claims above are taken from the public README and GitHub tree listing.

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

### 2.5 Practical Baseline Recommendation (2026)

| Scenario | Recommended starting point |
|----------|---------------------------|
| Linux binary, can rewrite | PeAR or ZAFL + AFL++ persistent/SHM |
| Linux binary, cannot rewrite | FRIDA persistent or QEMU persistent; fallback SPFuzz-style syscall patterns |
| Full-system / kernel | Nyx / KVM snapshot + KBinCov-style feedback |
| Closed-source Windows | WinAFL via PeAR or DynamoRIO |
| Research / novelty | daedalus/fuzzer-style information-theoretic + multi-bandit schedulers |

---

## 3. Open Questions / Follow-ups for daedalus/fuzzer

1. Full shallow clone still needed (environment was timing out at ~1.6 k files).  
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
- AFL++ binary-only guide (current recommendations)

---

*End of handover. Drop into `docs/handover/` and link from `docs/TODO.md` or `CHANGELOG.md` if desired.*
