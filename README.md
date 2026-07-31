# fuzzer-tool

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/daedalus/fuzzer)

**Information-dense, coverage-guided binary fuzzer** with Markov generation, Monte Carlo optimization, grammar-aware mutations, and 40+ scheduling strategies — including Elo arbitration, evolutionary algorithms, and information-theoretic scoring.

> **Honest caveat**: This is the most complex fuzzer from an information-theory standpoint, and also the slowest raw-throughput. The tradeoff is speed for edge-discovery novelty. For production fuzzing at scale, AFL family fuzzers remain the best choice.

---

## Quick Start

```bash
pip install -e ".[dev]"

# Basic fuzzing
fuzzer-tool fuzz ./target

# Coverage-guided with dictionary
fuzzer-tool fuzz -c -D dictionary.txt ./target

# In-process mode (fastest for .so targets)
fuzzer-tool fuzz libfoo.so --inprocess -c

# With Markov generation + Monte Carlo bandit
fuzzer-tool fuzz --markov --markov-gen --mc-bandit --mc-cem ./target

# Resume a previous session
fuzzer-tool fuzz ./target -c --resume
```

---

## Core Capabilities

### Mutation Engine
40+ mutation operators: bit/byte flips, arithmetic (1/2/4/8-byte LE/BE), block insert/delete/duplicate/swap, havoc with stall-recovery escalation, TLV-aware, token shuffle, security-sensitive string injection, punctuation insertion, compound dictionary, and **FrameShift** auto-adjusting length fields.

**Grammar-aware**: format-specific structural mutations for PNG (IHDR/IDAT/CRC/filter/interlace), JPEG (SOF/DHT/DQT/DRI/SOS), BMP (header/pixel), gzip/zlib (CMF/FLG/Adler-32), **PGS** (PCS/WDS/PDS/ODS/END segment mutations), **ISO-BMFF** (box-type/size/container nesting/codec/handler mutations for MP4/MOV), **NAL** (H.264/H.265 NAL unit type/ref_idc/SPS/PPS/slice mutations). **Format lock**: magic-prefix detection with protected-byte-tail-havoc for autoprobe targets. **Tree mutator**: Radamsa-style delimiter-based mutations (delete, duplicate, swap, stutter) with correct round-trip invariants.

### Coverage & Execution
- **AFL SHM bitmap** with sparse 8-byte entry hash table — no silent bucket collisions
- **Ptrace edge coverage** + Capstone x86-64 decoder for closed-source binaries
- **In-process direct** (`--inprocess-direct`): ctypes calls at 2k–34k eps with sigsetjmp crash survival
- **Multi-target**: fuzz multiple binaries with shared corpus and weighted round-robin
- **Hardware perf counters** via `perf_event_open`: instruction, branch, branch-miss counts
- **AFLGo directed fuzzing**: harmonic call-graph + CFG distance to targets (function names, addresses, or `file.c:line` via pure-Python DWARF), with the exact AFLGo power schedule (`--schedule aflgo --t-x N`) and a runtime SHM-tail distance channel on `trace-pc` builds (`build_targets.sh --distance`)

### Scheduling Intelligence
| Strategy | Flag | Description |
|----------|------|-------------|
| **Elo arbitration** | `--elo` | Combined operator + seed scheduling via Bayesian Elo rating; all strategies run in shadow |
| Thompson sampling | `--mc-bandit` | Beta-posterior bandit with Brier calibration |
| CEM byte distribution | `--mc-cem` | Cross-Entropy Method for value-level learning |
| MOpt PSO | `--mopt` | Particle swarm optimization over operator distributions |
| Replicator dynamics | `--replicator` | Evolutionary game theory — operators grow by fitness |
| EXP3 | `--exp3` | Adversarial bandit for non-stationary rewards |
| Epsilon-greedy | `--eps-greedy` | Classic exploration/exploitation with annealing |
| Hierarchical bandit | `--hierarchical-bandit` | Two-level: category → operator Thompson sampling |
| GP-UCB | `--gp-ucb` | Gaussian Process UCB with RBF kernel covariance |
| AFL++ power schedules | `--schedule` | FAST/COE/RARE/MMOPT/LIN/QUAD/GO/AFLGO seed-level energy |
| AFLGo directed annealing | `--schedule aflgo` | Exact AFLGo power factor — symmetric 32×/1/32× energy by distance-to-target with time-based cooling (`--t-x`, `--aflgo-cooling`) |
| Seed strategies | — | Weighted, Pareto, format-aware, GA, QEA, Bayesian, Markov-gen |

### Information-Theoretic Scoring
- **Mutual information** (`--mi-guided`): I(byte_position; coverage) guides mutation to positions that control code paths
- **Rényi entropy** (`--renyi-weight`): boosts seeds exercising rare edges
- **Transfer entropy** (`--transfer-entropy`): directional causal flow: byte→edge
- **Shannon entropy rate**: stall detection via flat vs. redistributing entropy
- **Index of Dispersion**: Fano factor resolves genuine stall from bursty exploration
- **Shapley value** (`--shapley`): fair operator credit via co-occurrence frequency
- **Rate-distortion corpus minimization** (`--rate-distortion`): optimal compression preserving diversity

### Evolutionary Lifecycles
- **GA** (`--ga`): bounded population, speciation (MinHash LSH), tournament selection, generational replacement with elitism
- **QEA** (`--qea`): quantum-inspired amplitude encoding, rotation gate feedback, collapse-only evaluation
- **WFC** (`--wfc`): Wave Function Collapse constraint-satisfaction generation for chunk reordering and 2D pixel data

### Comparison Tracing (cmplog)
- **Symbol-based**: intercepts `memcmp`/`strcmp`/`memmem`/`strstr` via LD_PRELOAD
- **Compiler-IR** (`trace-cmp`): clang `-fsanitize-coverage=trace-cmp` catches inlined/folded comparisons
- **SMT solving** (`--enable-smt-z3`): Z3 arithmetic constraint solving from cmplog pairs

### Crash Analysis
- ASAN/MSAN/TSAN/LSAN/UBSAN auto-classification
- Levenshtein crash clustering, stack-hash dedup, exploitability tiers
- Crash minimization (delta-debugging with signature pinning)
- Blocklist/allowlist and smaller-crash replacement

### Static Target Analysis (`TargetProfiler`)
- ELF analysis: string constants, magic bytes, DIV/IDIV constant extraction
- Auto-populated dictionary from `.rodata`
- Format-aware seed generation (PNG, JSON, XML, HTML from inferred format)
- Hot-function weighting and crash ETA estimation

### Observability
Live stat line shows: GA/QEA gen/pop/species, MI observations, Elo meta-strategy, Replicator dynamics, Jaccard redundancy, Wasserstein diversity, mutual information, transfer entropy, FrameShift relations, Markov context count. Full explainability report via `--report`.

### Distribution Diagnostics
Online running statistics (Welford/Pébay) for mean, variance, skewness, excess kurtosis — detect tail-risk inputs, critical slowing down, corpus bloat pre-warning, and per-operator reward stability.

---

## Coverage Modes

| Mode | Flag | Throughput |
|------|------|-----------|
| SHM bitmap | `-c` (default) | 65–200 eps |
| In-process subprocess | `--inprocess` | 65–120 eps |
| In-process direct | `--inprocess-direct` | 2k–34k eps |
| Ptrace basic | `-c --no-shm` | ~20 eps |
| Ptrace deep | `-c --no-shm --deep-coverage` | ~18 eps |

---

## Subcommands

| Command | Description |
|---------|-------------|
| `fuzz` | Run coverage-guided fuzzing |
| `rank` | Rank seeds by edge coverage, rarity, subsumption |
| `minimize` | Greedy set-cover corpus pruning |
| `sweep` | Linear seed replay — find missed crashes |
| `tmin` | Crash delta-debugging minimization |
| `replay` | Replay a crash input |
| `verify` | Confirm crashes with ASAN target |
| `estimate` | Crash ETA via Good-Turing + calibration |
| `import` | Import corpus from AFL/libFuzzer/honggfuzz |

---

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md) | Full reference: all features, options, API, building targets |
| [`docs/ASAN-LIMITATION.md`](docs/ASAN-LIMITATION.md) | ASAN in-process limitation & root cause analysis |
| [`docs/tracecmp-howto.md`](docs/tracecmp-howto.md) | Compiler-IR comparison tracing vendor build guide |
| [`docs/inprocess-limitations.md`](docs/inprocess-limitations.md) | In-process execution mode design notes |
| [`docs/TODO.md`](docs/TODO.md) | Roadmap and pending work |
| [`docs/learnings/`](docs/learnings/) | Research spikes and engineering learnings |
| [`docs/FINDINGS/`](docs/FINDINGS/) | Bug discovery reports (fgrep, FFmpeg) |
| [`AGENTS.md`](AGENTS.md) | Development guide and project conventions |
| [`SPEC.md`](SPEC.md) | Original specification |

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
ruff format src/ tests/
```

Current test suite: 2500+ tests including 67+ regression tests for historical bugfixes.

---

## Benchmarking

```bash
# 4-way configuration comparison
tools/bench.sh targets/png_read 10000

# Exhaustive feature/combination sweep
tools/bench_sweep.sh
```

---

## License

MIT
