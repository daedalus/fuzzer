# fuzzer-tool

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/daedalus/fuzzer)

**Information-dense, coverage-guided binary fuzzer** with Markov generation, Monte Carlo optimization, grammar-aware mutations, and 40+ scheduling strategies — including Elo arbitration, evolutionary algorithms, and information-theoretic scoring.

> **Honest caveat**: This is the most complex fuzzer from an information-theory standpoint, and also the slowest raw-throughput. The tradeoff is speed for edge-discovery novelty. For production fuzzing at scale, AFL family fuzzers remain the best choice.

---

## Quick Start

```bash
pip install -e ".[dev]"

# Basic fuzzing — coverage-guided by default
fuzzer-tool fuzz ./target

# With a dictionary
fuzzer-tool fuzz -D dictionary.txt ./target

# In-process mode (fastest for .so targets)
fuzzer-tool fuzz libfoo.so --inprocess

# Blind mutation, no edge bitmap (crash detection still works)
fuzzer-tool fuzz ./target --no-coverage

# With Markov generation + Monte Carlo bandit
fuzzer-tool fuzz --markov --markov-gen --mc-bandit --mc-cem ./target

# Resume a previous session
fuzzer-tool fuzz ./target --resume
```

---

## Core Capabilities

### Mutation Engine
40+ mutation operators: bit/byte flips, arithmetic (1/2/4/8-byte LE/BE), block insert/delete/duplicate/swap, havoc with stall-recovery escalation, TLV-aware, token shuffle, security-sensitive string injection, punctuation insertion, compound dictionary, and **FrameShift** auto-adjusting length fields.

**Grammar-aware**: format-specific structural mutations for PNG (IHDR/IDAT/CRC/filter/interlace), JPEG (SOF/DHT/DQT/DRI/SOS), BMP (header/pixel), gzip/zlib (CMF/FLG/Adler-32), **PGS** (PCS/WDS/PDS/ODS/END segment mutations), **ISO-BMFF** (box-type/size/container nesting/codec/handler mutations for MP4/MOV), **NAL** (H.264/H.265 NAL unit type/ref_idc/SPS/PPS/slice mutations), **Protobuf** (tag/wire-type/varint/length re-encode, field splice/duplicate/delete), **GIF** (LSD/GCT/image/extension/trailer), **WebP** (RIFF chunk type/size/VP8/VP8L/VP8X/ANMF), **WebM** (EBML element ID/size-vint/codec/nest-unnest), **ZIP** (LFH/CD/EOCD fields, crc/name/method), **x86/x86-64** (opcode-class, modrm, imm/disp, insn delete/duplicate/swap/splice), **ARM** (A32/T32 word-level). **Format lock**: magic-prefix detection with protected-byte-tail-havoc for autoprobe targets. **Tree mutator**: Radamsa-style delimiter-based mutations (delete, duplicate, swap, stutter) with correct round-trip invariants.

**Checksum learning** (Berlekamp-Massey): recovers unknown linear checksum polynomials from observed (data, checksum) pairs — format-aware PNG/ZIP/GZIP extraction plus cmplog heuristics — then the `crc_learn` operator patches checksum fields with the recovered model. Targets using non-standard CRC polynomials or proprietary linear checksums become fuzzable instead of opaque. All CRC-32 computation routes through a configurable wrapper (`core/crc32.py`): hardware `zlib.crc32` for the standard polynomial, software LFSR for recovered non-standard ones.

### Coverage & Execution
- **AFL SHM bitmap** with sparse 8-byte entry hash table — no silent bucket collisions
- **Forkserver** (default, `--no-forkserver` to opt out): `afl_shim.c` installs an AFL-style
  forkserver in its constructor, so the target is exec'd once and each input costs a `fork()`
  from a fully initialised process — 2.77x end to end, 5.27x on a light target
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
| **Mutation lineage tree** | `--lineage` | Weighted parent/ops/sites forest per seed: unproductive-branch pruning in auto-minimize, causal crash-path replay in `tmin`, LCA-based diversity scoring |

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

Coverage-guided mode is **on by default**. `--no-coverage` turns it off:
crash and timeout detection still work, but the edge bitmap stays empty, so
corpus growth and coverage-guided scheduling are inactive. Measured cost of
having it on, 1500 execs x 2 reps: 1.4% throughput on `targets/test_target`,
7.8% on `targets/png_read` — against a corpus that otherwise never grows past
its seeds. `-c`/`--coverage` are still accepted and are now no-ops.

If the target was not built with instrumentation the bitmap cannot fill, and
the run reports that at startup rather than looking healthy while discovering
nothing. Build with `tools/build_targets.sh`.

| Mode | Flag | Throughput |
|------|------|-----------|
| SHM bitmap + forkserver | *(default)* | 0.5k–1.4k eps |
| SHM bitmap, spawn per exec | `--no-forkserver` | 65–500 eps |
| In-process subprocess | `--inprocess` | 65–120 eps |
| In-process direct | `--inprocess-direct` | 2k–34k eps |
| Ptrace basic | `--no-shm` | ~20 eps |
| Ptrace deep | `--no-shm --deep-coverage` | ~18 eps |
| Blind (no edge bitmap) | `--no-coverage` | as target allows |

---

## Feature Compatibility Matrix

Not every feature works in every execution mode — the constraints come from
where a *process boundary* exists. A mode that loads the target into the
fuzzer's own address space has no signal-carrying boundary to attach a
tracer to; a mode that forks per call does.

### Execution modes × features

| Feature | direct (`--inprocess-direct`) | subprocess (`--inprocess`) | SHM / exec (default) | ptrace (`--no-shm`) |
|---|---|---|---|---|
| Edge coverage (AFL bitmap) | ✅ | ✅ | ✅ | ✅ (via breakpoints) |
| Fault address (`si_addr`) | ✅ ¹ | ✅ | ✅ | ✅ |
| Register capture at crash | ✅ ¹ | ✅ | ✅ | ✅ |
| cmplog / redqueen | ✅ ² | ✅ | ✅ | ✅ |
| trace-cmp (compiler IR) | ✅ ² | ✅ | ✅ | ✅ |
| ASAN targets | ⚠️ ³ | ✅ | ✅ | ✅ |
| UBSAN targets | ✅ | ✅ | ✅ | ✅ |
| MSAN targets | ❌ ⁴ | ❌ ⁴ | ✅ | ✅ |
| TSAN targets | ❌ ⁴ | ❌ ⁴ | ✅ | ✅ |
| Sanitizer report parsing (stderr) | ✅ | ✅ | ✅ | ✅ |
| Persistent (no re-exec) | ✅ | ✅ | ❌ | ❌ |

¹ direct mode has no process boundary to attach at, so a crashing input is
re-run once through a ptrace'd one-shot loader for triage. Costs an extra
execution only on the rare crashing input, not on the hot path.

² Requires the cmplog/trace-cmp shim to be *compiled into* the target `.so`
(`build_targets.sh --cmplog` / `--tracecmp`, on by default), or externally
`LD_PRELOAD`ed. `LD_PRELOAD` alone cannot work for a `.so` opened with
`ctypes.CDLL` into an already-running process — the fuzzer auto-detects this
and falls back to the persistent subprocess loader.

³ ASAN `.so` targets need `libasan` preloaded before the interpreter starts;
without it the fuzzer automatically falls back to the subprocess loader.

⁴ MSAN and TSAN instrument the *whole process*. Loading such a target into
the uninstrumented CPython host reports on Python's own memory, so these are
built as standalone executables only (`--msan` / `--tsan`) and run through
the exec path.

### Sanitizer build variants

| Sanitizer | Flag | Finds | Notes |
|---|---|---|---|
| ASAN | (default) | heap/stack overflow, UAF, double-free | executables + `.so` |
| UBSAN | (with ASAN) | integer overflow, bad shifts, misaligned access | `.so`, clang |
| MSAN | `--msan` | use-of-uninitialized-value | clang only, executables only; **ASAN cannot detect this class at all**. Targets linking uninstrumented system libs (libpng/libz/libjpeg) are skipped — they would report false positives unless those libraries are rebuilt instrumented. |
| TSAN | `--tsan` | data races, lock-order inversion, thread leaks | executables only; built with clang (gcc also supports `-fsanitize=thread` if you build manually). Relevant for threaded targets. |

### Compiler support

`clang` is the default compiler, and it matters: **it is the only compiler
that can produce full automatic edge coverage here.**

| Capability | gcc | clang |
|---|---|---|
| Target + `.so` builds | ✅ | ✅ (default) |
| AFL edge instrumentation shim | ✅ | ✅ |
| cmplog shim (libc interposition) | ✅ | ✅ (preferred) |
| Manual `__afl_map_edge()` coverage | ✅ | ✅ |
| **Automatic edge coverage** (`-fsanitize-coverage=trace-pc-guard`) | ❌ ⁵ | ✅ required |
| trace-cmp instrumentation | ⚠️ ⁵ | ✅ required |
| MSAN (`-fsanitize=memory`) | ❌ | ✅ required |
| TSAN (`-fsanitize=thread`) | ✅ | ✅ (used) |
| DWARF 4 + DWARF 5 line tables | ✅ | ✅ |

#### ⁵ The gcc edge-coverage limitation

gcc's `-fsanitize-coverage=` accepts only `trace-pc` and `trace-cmp` — not
the `trace-pc-guard` variant the AFL shim's edge callbacks are built on. The
one gcc-compatible callback the shim does implement,
`__sanitizer_cov_trace_pc()`, is compiled into every shim build
(`__AFL_DISTANCE_MODE` defaults to 1; `-D__AFL_DISTANCE_MODE=0` drops it).
It depends on the AFLGo distance SHM but degrades gracefully without one:
an unmapped segment leaves the lookup table NULL and the callback returns
early. There is currently no measured standalone `trace-pc` path for gcc.

gcc targets therefore fall back to the hand-placed `__afl_map_edge()` calls
in the target wrapper sources. Those see the wrapper's own branching, but not
the branching inside the library being fuzzed. Measured on
`targets/png_read.c` with an identical seed:

| Build | Instrumented call sites | Bitmap slots populated |
|---|---|---|
| gcc — manual `__afl_map_edge` only | 0 automatic | 43 |
| clang `--clang-scov` (`trace-pc-guard`) | 192 | **127** |

gcc still builds every target correctly and remains a supported fallback; it
just yields shallower coverage. Everything downstream that consumes the edge
signal — seed scheduling, MI/TE/sensitivity position weighting, Elo/bandit
operator scheduling, stall detection — is only as good as that signal.

> **Note:** automatic edge coverage is opt-in via `--clang-scov`. A plain
> `build_targets.sh` produces manual-instrumentation-only targets on *either*
> compiler. Pass `--clang-scov` when coverage depth matters — verified on
> `targets/png_read`: 0 instrumented call sites without it, 194 with.

Regardless of the default, the build script selects `clang` automatically for
the features that require it (trace-pc-guard, trace-cmp, UBSAN, MSAN, sancov),
so those work even when `DEFAULT_CC` is gcc.

#### Vendored libraries: required for real coverage depth

Compiler flags only instrument code the build actually compiles. A target
linking a *system* library (`-lpng`, `-lz`, `-ljpeg`) gets zero edges from
inside that library no matter which compiler or coverage flag is used — the
library was built by your distro without instrumentation, and the target only
sees it across a `.so` boundary. What remains instrumented is the thin target
wrapper: `targets/png_read.c` is 175 lines that open a file, call
`png_read_png`, and check the error path.

`vendor/` is gitignored, so a fresh clone silently falls back to system libs.
Nothing fails and the numbers look plausible — the ceiling is just quietly
much lower. Fetch the sources to fix it:

```bash
git clone --depth 1 --branch v1.3.1  https://github.com/madler/zlib.git   vendor/zlib
git clone --depth 1 --branch v1.6.43 https://github.com/pnggroup/libpng.git vendor/libpng
bash tools/build_targets.sh   # look for: "Using vendored trace-cmp libraries"
```

No extra flag is needed — `--cmplog` is on by default, and
`build_simple_so_targets()` links the vendored archives automatically once
they exist. Measured on `targets/png_read.so`, identical corpus and 45s budget:

| Build | `NEEDED` | png syms defined / undefined | Bitmap | Edges (45 s) |
|---|---|---|---|---|
| System libs | `libpng16.so.16`, `libz.so.1` | 2 / 17 | 8 KB | 304 |
| Vendored | *(none)* | 418 / 0 | 16 KB | **1,444** |

To check which one you have:

```bash
readelf -d targets/png_read.so | grep NEEDED     # libpng16.so.16 present => system libs
nm targets/png_read.so | grep -c ' [Tt] png_'    # 2 => system libs, ~418 => vendored
```

A suspiciously low edge plateau on a large, branchy library is usually this —
verify the link before tuning schedulers or dictionaries. See
[`docs/learnings/2026-08-07-uninstrumented-system-libs-coverage.md`](docs/learnings/2026-08-07-uninstrumented-system-libs-coverage.md).

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
