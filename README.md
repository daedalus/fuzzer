# fuzzer-tool

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/daedalus/fuzzer)

Coverage-guided binary fuzzer with static target analysis, statistical novelty scoring, Markov chain generation, Monte Carlo mutations, kernel crash verification, and format-aware grammar mutations.

Honest Caveat: Probably the most complex and dense fuzzer from the information theory standpoint but also the slowest. The tradeoff is speed for edge discovery novelty.
For production and sensitive binaries using AFL family fuzzers is the best course of action. 

## Features

### Mutation & Generation
- **Mutation operators**: bit flip, byte flip, interesting values (8/16/32-bit, signed + unsigned boundary), arithmetic (1/2/4/8-byte, LE/BE), block insert/delete/duplicate, bit-offset flip/span (arbitrary bit positions for DEFLATE/JPEG), havoc mode
- **Operator performance**: `type_replace` uses a precomputed 256-byte translate table (184x faster), PNG/BMP random generation uses `random.randbytes()` instead of Python loops (16x faster), `colorization` uses a module-level lookup table
- **Length boundary operator**: systematically tries input lengths at boundary values (0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024, 4096) — discovers length-sensitive unsigned integer underflows
- **Unsigned boundary values**: interesting values include small values (0-5) and unsigned max values (0xFF, 0xFFFF, 0xFFFFFFFF) for triggering unsigned arithmetic underflows
- **Crash-MI-guided mutation**: CrashMITracker identifies byte positions and values correlated with crashes, biasing mutation position selection and interesting value selection toward crash-relevant bytes
- **CrashMITracker memory pruning**: automatically caps per-position byte-value tracking to top 32 entries every 500 execs — prevents tracker JSON from growing unbounded (was 7.5MB per 15k execs, now ~500KB)
- **Weighted length distribution**: length_boundary operator weights small lengths (0-16) 10:1 over large ones (512+) — 4096-byte inputs dropped from 4.7% to 0.5% of picks, stabilizing EPS
- **Corrupted state recovery**: seed_meta entries with keys > 256 chars (tracker JSON loaded as corpus) are skipped on load and save — self-heals bloated state.json on first run (15MB → 6KB)
- **Grammar-aware mutations**: format-specific structure-aware mutations for PNG (IHDR, IDAT, CRC, filter types, interlace), JPEG (SOF, DHT, DQT, DRI, SOS, scan data), BMP (header fields, pixel data), gzip (header flags, deflate stream, trailer, extra fields), and zlib (CMF/FLG header, deflate stream, Adler-32 trailer)
- **FrameShift**: automatic length-field tracking — discovers and adjusts length/count fields during insertions/deletions, applied as universal post-processing after every mutation
- **Dictionary support**: inject protocol tokens from dictionary files
- **Markov chain**: learn byte-level transition probabilities from corpus, generate statistically similar inputs, persist across runs
- **Monte Carlo scheduling**: Thompson sampling bandit for operator selection + Cross-Entropy Method for byte distribution learning
- **Surprisal-weighted rewards**: all scheduling mechanisms (bandit, MOpt, Replicator, Elo) weight discovery credit by `1 - bitmap_density` — rare discoveries in sparse coverage regions get more credit than discoveries near already-saturated areas
- **Perplexity-gated generation**: model quality dynamically scales generation rate (more generation when model is lost, less when well-calibrated); rejects extreme-perplexity outputs as pure noise

### Static Target Analysis
- **TargetProfiler**: ELF static analysis at startup — extracts string constants, function boundaries, magic bytes, and input format hints
- **Auto-populated dictionary**: interesting strings (format specifiers, error messages, keywords) and magic bytes extracted from `.rodata`
- **Format-aware seed generation**: produces structurally meaningful initial seeds (PNG headers, text protocols, JSON, XML, HTML) based on inferred format
- **Informative Bayesian priors**: `format_operator_priors()` seeds the Thompson-sampling bandit's Beta prior toward structure-aware operators (e.g. `png_chunk_mutate`) and dictionary operators when static analysis detects a matching format or extractable tokens, instead of always starting from the uninformative Beta(1, 1)
- **Hot-function weighting**: seeds exercising high-branch-density functions get a proportional boost in selection
- **Crash ETA estimation**: blends static risky density (keyword heuristic) with dynamic I(byte_position; crash) mutual information from actual executions, plus Good-Turing edge estimates and calibrated discovery rate — the MI signal strengthens as fuzzing accumulates near-miss data

### Coverage & Scoring
- **AFL count-class bucketization**: `classify_counts()` collapses raw hit counts into 9 logarithmic buckets (0, 1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+) before coverage comparison — eliminates noise from count-magnitude jitter and provides cleaner signal for JS-divergence/Wasserstein diversity scoring; `new_bits()` provides AFL-style overlap/new-edge detection on classified bitmaps
- **Morris probabilistic counting (a=30)**: log-scale edge hit counters prevent overflow and provide frequency information for scheduler decisions; estimate formula `a * ((1+1/a)^v - 1)` converts back to approximate counts
- **AFL SHM bitmap** coverage for instrumented targets (~65-200 eps)
- **Ptrace edge coverage** with deep capstone disassembly for closed-source binaries (~18-20 eps)
- **In-process execution**: persistent subprocess mode (~65-120 eps) with auto-restart on crash
- **Length-edge tracking**: correlates input length with coverage edge discovery — biases seed selection and length-changing mutations toward productive lengths
- **Per-target SHM coverage**: multi-target mode tracks coverage independently per target binary
- **Cross-target seed scoring**: seeds that found edges in the least-covered target get boosted proportionally to the coverage gap
- **Branch density**: per-target static analysis metric (conditional branches/KB) with average across targets
- **Auto-sized edge bitmap**: `estimate_map_size()` from branch density × .text size replaces hardcoded 65536
- **Good-Turing estimation**: prospective edge discovery count with saturation confidence
- **KS significance testing**: replaces fixed JS thresholds with sample-size-aware p-values
- **CRPS scoring**: proper scoring rule for execution time calibration (fixed indicator direction bug)

### Distribution Diagnostics
- **Running statistics** (`core/running_stats.py`): Welford/Pébay online algorithm for O(1) mean, variance, stddev, skewness, and excess kurtosis — unbounded or sliding-window variants
- **Execution time tail-risk detection**: skewness > 2.0 flags algorithmic-complexity inputs (regex backtracking, hash-flood) that occasionally trigger big execution-time excursions
- **Critical slowing down skewness tier**: three-signal detector (variance + autocorrelation + skewness) — rising right skew upgrades the verdict to "approaching transition, and it looks productive"
- **Per-operator reward moments**: UCB-style exploration bonus (`mean + k * stddev`) with kurtosis-scaled stability guard — high-kurtosis operators require more observations before trusting their stddev-based bonus
- **Format learner z-score gate**: replaces fixed `delta != 0` threshold with z-score-based outlier detection; MAD fallback under high kurtosis for robustness against zero-inflated coverage deltas
- **Corpus bloat early-warning**: rising right skew in seed file sizes is a leading indicator of bloat that precedes RSS threshold tripping
- **Bounded memory structures**: all accumulative data structures (correlation matrix, coverage timeline, cmplog tokens/pairs, kernel crashes, Shapley attribution edges, stderr buffer, seed secretary, seen hashes) are capped via module-level constants — RSS plateaus instead of growing linearly with exec count
- **Report distribution diagnostics**: stddev, skewness, and kurtosis for exec time, discovery rate, per-operator rewards, and seed sizes

### Scheduling Intelligence
- **Jaccard index**: average pairwise edge-set overlap (xxhash-fast) for corpus redundancy monitoring
- **Subsumption weighting**: MinHash-approximated Jaccard for continuous seed deprioritization
- **Hitcount diversity (JS divergence)**: seeds with unusual frequency profiles get boosted
- **Wasserstein spatial diversity**: seeds exploring different code regions get boosted
- **Weight caching**: recomputes only when corpus/edge-count changes (733x speedup on 200+ seeds)
- **Perplexity (MDL codelength)**: structurally novel seeds get 1.0-2.0x weight
- **NCD similarity**: Normalized Compression Distance between corpus entries
- **Simulated annealing**: temperature-scaled exploration/exploitation balance
- **Hamming bitmap distance**: fast byte-level seed-to-seed similarity on edge bitmaps
- **Near-duplicate detection**: finds seed pairs with near-identical coverage via Hamming + LSH

### Information Theory
- **Mutual information** (`--mi-guided`): I(byte_position; coverage) guides mutation toward positions that actually control code paths
- **Rényi entropy** (`--renyi-weight`): generalized entropy spectrum for seed weighting — boosts seeds exercising rare (cold) edges
- **Rate-distortion corpus minimization** (`--rate-distortion`): optimal compression of corpus preserving coverage diversity
- **Transfer entropy** (`--transfer-entropy`): directional causal flow between byte positions and coverage edges
- **Shannon entropy rate tracking**: global edge-hit distribution entropy sampled periodically; confirms genuine stall (no new edges + flat entropy rate) vs. transient redistribution before activating random-mode recovery

### Game Theory
- **Shapley value** (`--shapley`): per-edge frequency-weighted operator attribution — credit distributed proportional to co-occurrence frequency, not naive full credit to all stacked operators
- **Replicator dynamics** (`--replicator`): evolutionary game theory scheduling — operators grow proportionally to fitness, converging to evolutionarily stable strategies
- **MOpt PSO** (`--mopt`): particle swarm optimization over operator distributions (alternative to Thompson sampling)

### Genetic Algorithm Lifecycle (`--ga`)
- **Finite population**: replaces monotonically growing corpus with bounded, evolving population (`--ga-pop-size`)
- **Unified fitness function**: single score combining novelty (edge coverage), diversity (Wasserstein distance), freshness (recency), and mutation potential
- **Fitness-proportional parent selection**: tournament selection for crossover parents instead of random corpus picks
- **Speciation**: MinHash LSH-based species partitioning prevents dominant lineages from monopolizing selection
- **Generational replacement**: periodic evolution cycles with elitism — top fraction always survives, low-fitness individuals culled
- **Crash preservation**: crash-triggering seeds get infinite fitness bonus, never culled
- **State persistence**: population and generation state saved to `ga.json`, survives `--resume`

### Quantum-Inspired Evolutionary Algorithm (`--qea`)
- **Amplitude encoding**: each bit represented as a qubit-like probability amplitude pair (α, β) with α² + β² = 1 — "this bit is P(0)=α² likely to be 0" rather than a committed value
- **Rotation gate feedback**: amplitudes incrementally updated after each evaluation — nudging toward or away from collapsed values depending on coverage outcome
- **Collapse-only evaluation**: concrete bytes sampled from amplitudes at evaluation time, preserving uncertainty between generations
- **Built on GA infrastructure**: reuses the existing FitnessFunction, Speciation (MinHash LSH), and generation lifecycle
- **Breeding by collapse + crossover**: parents' amplitudes collapsed to bytes, two-point crossover applied, child amplitudes biased toward result
- **Diversity preservation**: continuous per-bit uncertainty maintains diversity longer than committed-value GA or batched CEM refits
- **State persistence**: population and amplitude state saved to `qea.json`, survives `--resume`
- **Note**: `--qea` and `--ga` are mutually exclusive; `--qea` takes precedence if both are set

### Wave Function Collapse (`--wfc`)
- **Constraint-satisfaction generation**: WFC solves local adjacency constraints via min-entropy collapse and AC-3 propagation, producing novel-but-valid chunk orderings for structured formats
- **1D chunk reordering**: replaces random chunk swaps in PNG/JPEG/gzip structural mutations with WFC-valid orderings that respect format-specific adjacency rules (IHDR first, IEND last, ancillary interposition, IDAT contiguity)
- **2D pixel generation**: per-row WFC for locally-coherent pixel data in BMP/PNG raw payloads, using adjacency learned from existing corpus pixels
- **No topology assumptions**: unlike `grammar.py` (recursive CFG) and `markov.py` (causal left-to-right), WFC handles arbitrary flat adjacency constraints without causal ordering or recursion depth limits
- **Defensive posture**: bounded backtrack (max 3 restarts), capped AC-3 iterations (5000) with greedy fallback, seeded RNG for tmin reproducibility
- **Guarded integration**: WFC operators run at warm tier (per-input format mutation, not per-execution bit flip), and are disabled when `--wfc` is not set
- **Independent mode**: `--wfc` is orthogonal to `--ga`/`--qea` — it controls structural generation inside format mutators, not seed selection

### Multi-Target Fuzzing
- **Shared corpus**: fuzz multiple binaries with the same corpus — inputs that find coverage in one target can discover paths in others
- **Glob expansion**: `targets/fuzz_*` expands to all matching executables, automatically skips non-binaries (`.c`, `.py`, `.sh`, etc.)
- **Per-target SHM**: each target gets its own shared memory region for independent edge tracking
- **Weighted round-robin**: targets with fewer discovered edges get proportionally more execution time
- **Cross-target seed scoring**: seeds productive for the least-covered target get boosted in selection
- **Per-target stats**: startup shows `[AFL]`/`[no-AFL]` detection, branch density per target; live stats show edge counts per target
- **AFL detection**: binary checked for `__afl_area`/`__afl_map_shm` symbols via `nm` at startup

### Corpus Management
- **Delta-encoded corpus**: parent-child diffs for small mutations (< 25% change), periodic full snapshots every 20 generations
- **xxhash dedup**: ~13x faster than SHA-256 for corpus deduplication
- **Delta snapshotting**: caps chain depth at 20 hops, prevents unbounded reconstruction cost
- **Auto-minimize**: corpus pruning guided by Wasserstein spatial diversity
- **Hamming fuzzy dedup**: near-duplicate detection via Hamming distance on equal-length seeds (`--fuzzy-dedup N`)

### Crash Analysis
- **Sanitizer detection**: automatic ASAN/MSAN/TSAN/LSAN/UBSAN crash classification
- **Kernel crash verification**: async dmesg streaming for kernel-level crash detection
- **Crash minimization**: delta-debugging with signature-matching to prevent drift to unrelated bugs
- **Corpus minimization**: greedy set-cover over SHM edge bitmaps (`minimize` subcommand)
- **Crash exploitability tiers**: ASAN_EXPLOITABILITY classification in reports
- **Levenshtein crash clustering**: groups crashes with similar stack traces (same root cause, different offsets)
- **Fuzzy corpus similarity**: Hamming + Levenshtein + 4-gram Jaccard for crash-to-corpus nearest-neighbor search

### Observability
- **Branch density**: per-target static analysis at startup (`cond branches/KB`) with average across targets
- **Per-target coverage stats**: live display shows `targets: name1:N name2:N name3:N` (edge counts per target)
- **AFL detection**: binary checked for `__afl_area`/`__afl_map_shm` symbols via `nm` — shows `[AFL]`/`[no-AFL]` per target
- **Jaccard index**: corpus redundancy metric (`| jac: 0.XX`)
- **Diversity score**: Wasserstein spatial diversity (`| div: N`)
- **`--report` flag**: full explainability report with coverage, mutations, perplexity, corpus health, edge map
- **`--replay-N` flag**: background crash reproducibility scoring
- **Per-seed cost tracking**: wall-clock time per seed for cost-aware scheduling
- **Discovery rate**: edges per 1k execs over sliding window
- **Bitmap density**: map occupancy percentage (saturation detection)
- **Dup rejection rate**: duplicate-rejection as saturation signal

### Performance
- **Weight caching**: 733x speedup on `_pick_seed` with 200+ seeds
- **Lazy watchdog**: `Event.wait(timeout)` eliminates busy-poll overhead on fast processes
- **xxhash dedup**: 13x faster than SHA-256 for corpus operations

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# Basic fuzzing
fuzzer-tool fuzz ./target

# Coverage-guided with dictionary
fuzzer-tool fuzz -c -D dictionary.txt ./target

# In-process mode (fastest for .so targets)
fuzzer-tool fuzz libfoo.so --inprocess --inprocess-func target_func -c

# File-based target
fuzzer-tool fuzz -F -A "{file}" ./target

# With Markov and Monte Carlo
fuzzer-tool fuzz --markov --markov-gen --mc-bandit --mc-cem ./target

# Grammar-aware PNG fuzzing
fuzzer-tool fuzz targets/png_read -c -D dictionaries/png.dict -g dictionaries/png.gram

# ASAN fuzzing (auto-detected, catches heap-buffer-overflow, use-after-free, etc.)
fuzzer-tool fuzz targets/asan_target

# Cmplog comparison tracing test (memcmp/strcmp/strncmp/memchr/strcasecmp/strncasecmp/memmem/strstr/strcasestr)
fuzzer-tool fuzz targets/cmplog_exercise --cmplog

# Compiler-IR comparison tracing test (requires clang -fsanitize-coverage=trace-cmp)
fuzzer-tool fuzz targets/tracecmp_target --cmplog

# fgrep SIMD/regex/BMH search fuzzing
fuzzer-tool fuzz targets/fgrep_read

# fgrep-specific fuzz targets (ASAN-instrumented)
# Regex compilation — adversarial patterns against regcomp()
fuzzer-tool fuzz targets/fuzz_regex_compile

# Pattern matching — fixed patterns, fuzzed data against regexec/SIMD search
fuzzer-tool fuzz targets/fuzz_pattern_match

# Full search pipeline — end-to-end search_data() with SIMD, regex, output
fuzzer-tool fuzz targets/fuzz_search_pipeline

# Multi-target: fuzz multiple binaries with shared corpus (glob supported)
fuzzer-tool fuzz targets/fuzz_regex_compile targets/fuzz_pattern_match targets/fuzz_search_pipeline -c -d corpus/fgrep

# Tailslayer hedged reader fuzzing (in-process .so mode, ~66 eps)
fuzzer-tool fuzz targets/tailslayer_read.so -c --inprocess

# Multi-target with glob — skips .c/.h/.py automatically
fuzzer-tool fuzz 'targets/fuzz_*' -c -d corpus/fgrep

# Two-pass workflow: fast fuzz without ASAN, then verify crashes with ASAN
fuzzer-tool fuzz targets/fuzz_*_nosan -c -d corpus/fast/
fuzzer-tool verify targets/fuzz_search_pipeline corpus/fast/crashes/

# Resume a previous fuzzing session
fuzzer-tool fuzz ./target -c --resume

# Full report after run
fuzzer-tool fuzz ./target -c -n 5000 --report report.txt

# Rank corpus seeds by interestingness
fuzzer-tool rank ./target -d corpus -n 20

# Dump top 10 most interesting seeds to files
fuzzer-tool rank ./target -d corpus -n 10 --dump top_seeds
```

## Fuzzing Options

| Flag | Description |
|------|-------------|
| `-c` | Enable coverage-guided mode |
| `--no-shm` | Skip AFL SHM, force ptrace |
| `--deep-coverage` | Capstone-based basic block discovery |
| `-F` | File mode (write input to temp file) |
| `-D FILE` | Load dictionary tokens |
| `-g GRAMMAR` | Grammar-aware mutations (built-in: png, json, http_request, elf) |
| `--cmplog` | Comparison tracing via LD_PRELOAD (or compile `cmplog_shim.c` into target .so for direct_lite compatibility) |
| `--markov-gen` | Markov-generated seeds (rate adapts to model quality via perplexity) |
| `--mc-bandit` | Thompson sampling operator selection (Brier score calibration) |
| `--mc-cem` | Cross-Entropy Method byte distribution |
| `--mopt` | MOpt PSO operator scheduling (alternative to bandit) |
| `--replicator` | Replicator dynamics operator scheduling (evolutionary game theory) |
| `--shapley` | Shapley value operator attribution (fair credit distribution) |
| `--mi-guided` | Mutual information guided mutation (target high-MI byte positions) |
| `--renyi-weight` | Rényi entropy weighting in seed selection (boost cold-edge seeds) |
| `--transfer-entropy` | Transfer entropy causal tracking (byte→edge influence detection) |
| `--inprocess` | Persistent subprocess mode (auto-restart on crash) |
| `--resume` | Resume from saved state |
| `--crash-codes N` | Additional exit codes to treat as crashes |
| `-j N` | Parallel fuzzing with N workers |
| `--max-corpus N` | Auto-minimize corpus at N entries |
| `--replay-n N` | Replay each crash N times for reproducibility scoring |
| `--report [FILE]` | Generate explainability report (stdout or file) |
| `--stats-interval N` | Print live stats and dump stats file every N iterations (default: 1000) |

## Subcommands

| Command | Description |
|---------|-------------|
| `fuzz` | Run coverage-guided fuzzing (default) |
| `rank` | Rank corpus seeds by interestingness (edge coverage, rarity, subsumption) |
| `minimize` | Minimize corpus by removing redundant inputs |
| `tmin` | Minimize a crash to smallest reproducer |
| `replay` | Replay a crash input against the target |
| `verify` | Re-run crashes with ASAN target to confirm memory bugs |
| `estimate` | Estimate execs to first crash via static analysis + calibration |
| `import` | Import corpus from AFL/libFuzzer/honggfuzz |

### Rank Seeds

Rank corpus seeds by a composite interestingness score based on edge coverage, singleton edge rarity, subsumption (irreplaceability), and coverage proximity.

```bash
fuzzer-tool rank <target> -d <corpus> [-n TOP] [--dump PREFIX]
```

| Flag | Description |
|------|-------------|
| `-d DIR` | Corpus directory |
| `-n N` | Number of top seeds to show (default 10) |
| `--dump PREFIX` | Dump top seeds to files `PREFIX.0`, `PREFIX.1`, ... |

### Estimate Crash ETA

Estimate executions to first crash using static risky density, Good-Turing edge estimates, and optional calibration runs.

```bash
fuzzer-tool estimate <target> --corpus <dir> [--calibrate N]
```

| Flag | Description |
|------|-------------|
| `--corpus DIR` | Corpus directory for Good-Turing edge estimation |
| `--calibrate N` | Number of calibration executions (default: 1000) |

## Coverage Modes

| Mode | Flag | Throughput | Notes |
|------|------|-----------|-------|
| SHM bitmap | `-c` (default) | 65–200 eps | For AFL-instrumented targets |
| In-process | `--inprocess` | 65–120 eps | Persistent loader with crash recovery |
| In-process direct | `--inprocess-direct` | 2k–34k eps | No crash isolation |
| Ptrace basic | `-c --no-shm` | ~20 eps | Function-entry breakpoints |
| Ptrace deep | `-c --no-shm --deep-coverage` | ~18 eps | Capstone BB discovery |

## State Persistence

```bash
fuzzer-tool fuzz ./target -c -n 10000
fuzzer-tool fuzz ./target -c --resume -n 10000
```

State files:
- `state.json` — exec counts, crash sigs, op stats, seed metadata, lineage depths
- `edge_tracker.json` — per-seed edge coverage, cumulative edges, global hit counts, hit counts
- `markov.json` — persisted Markov chain transitions
- `mi.json` — mutual information tracker (byte-to-coverage correlations)

## In-Process Execution

### Direct ctypes (`--inprocess-direct`)
Calls target function directly via `ctypes.CDLL`. Catches SIGSEGV/SIGABRT via signal handler. ~2k–34k eps.

### Persistent subprocess (`--inprocess`)
Keeps one Python subprocess alive. Fork-per-call with `os.setsid()` for process group isolation. Timeout enforced via outer threaded readline. Auto-restarts on subprocess death. ~65–120 eps.

### ASAN support
Automatically detects ASAN-instrumented targets by checking for `__asan_init` symbols. Falls back from `--inprocess-direct` to subprocess mode when ASAN is detected (ASAN calls `_exit()` which kills in-process targets).

### SHM resize in inprocess mode
When collision risk exceeds the threshold, the bitmap SHM is resized. In inprocess mode, this patches the target's `__afl_area` pointer to the new SHM segment and invalidates the cached SHM attachment, so coverage writes don't go to freed memory. The target's compiled-in `__afl_map_mask` is not updated (static variable), so the target underutilizes the new bitmap — but writes remain in-bounds.

### Timeout in direct mode
`--inprocess-direct` and direct_lite mode enforce timeout via `SIGALRM` + `setitimer`. Previously these modes had no timeout protection — a hanging target would freeze the fuzzer.

## Corpus Minimization

```bash
# Basic minimization (greedy set-cover)
fuzzer-tool minimize ./target -d corpus -c

# Rate-distortion optimal pruning (preserves coverage diversity)
fuzzer-tool minimize ./target -d corpus -c --rate-distortion --target-frac 0.95
```

## Test Suite

1494 tests covering all modules. Run with:

```bash
pip install -e ".[dev]"
pytest
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
ruff format src/ tests/
```

## Benchmarking

Compare fuzzer configurations on a target:

```bash
# 4-way comparison: baseline vs enhanced vs enhanced+ vs optimal
tools/bench.sh targets/png_read 10000
```

Configurations:
- **baseline**: no features
- **enhanced**: elo + bandit + mopt
- **enhanced+**: all enhanced + markov + replicator + shapley + renyi + transfer-entropy + grammar
- **optimal**: elo + mopt + replicator + markov ensemble (orders 0,1,2,3) + markov-gen
  - Best edge coverage at -n 1k (sweep-validated: 74 edges vs 61 baseline, 70 enhanced+)

For a broader sweep across individual features and many combinations (instead
of these four named configurations), use `tools/bench_sweep.sh`. Both scripts
share common helpers (SHM cleanup, log metric extraction, coverage
verification) from `tools/lib/bench_common.sh`.

### SMT Solver Evaluation (`--enable-smt-z3`)

The SMT solver (Z3) attempts to solve arithmetic constraints discovered by cmplog, generating inputs that satisfy specific branch conditions rather than relying solely on random mutations.

**30k-iteration comparison, zero corpus, `targets/png_read_tracecmp_asan.so`:**

| Metric | SMT (z3) | No SMT |
|--------|----------|--------|
| Avg EPS | 485.4 | 444.3 |
| Corpus growth | 1→139 entries | 1→126 entries |
| SHM max edge IDs | 273 | 245 |
| Stalls | 1 (0.6% recovery) | 4 (16.2% recovery) |
| SMT solve rate | 10/39 (26%) | N/A |

**Verdict**: From a cold start, SMT provides a modest but real advantage — ~9% higher throughput, ~75% fewer stalls, and slightly higher edge coverage (273 vs 245 max edge IDs). The solver fires on ~26% of cmplog arithmetic constraints when starting from a fresh corpus (higher solve rate on simpler constraints). The effect is amplified over the pre-warmed case, where stale constraints reduce the solve rate to ~6%. At this scale the advantage is incremental, not transformative — the SMT overhead is negligible, so there is no reason to leave it off when cmplog is already enabled.

## Building Targets

```bash
# Build all targets (ASAN + no-ASAN executables and .so shared libraries)
tools/build_targets.sh

# ASAN only
tools/build_targets.sh --asan

# No-ASAN only (faster)
tools/build_targets.sh --fast

# Build .so targets with cmplog compiled in (for direct_lite compatibility)
tools/build_targets.sh --cmplog
tools/build_targets.sh --asan --cmplog        # ASAN + cmplog

# Build with compiler-inserted edge coverage and compiler-IR comparison tracing (requires clang)
tools/build_targets.sh --clang-scov
tools/build_targets.sh --asan --clang-scov    # ASAN + compiler-inserted coverage

# Build vendored (libpng+zlib) targets with compiler-IR comparison tracing
tools/build_targets.sh --vendor-tracecmp
tools/build_targets.sh --vendor-tracecmp --asan   # With ASAN (two-step build)
```

The build script compiles every target as both an executable and a `.so` shared library, in ASAN and no-ASAN variants. The no-ASAN `.so` variants (`*_nosan.so`) are suitable for high-throughput in-process fuzzing without sanitizer overhead.

### Build-time Cmplog for .so Targets

By default, `--cmplog` uses `LD_PRELOAD` to intercept comparison functions, which requires a process boundary (fork+exec). For `.so` targets in `direct_lite` mode, this doesn't work — no exec occurs.

**Solution: compile cmplog into your .so at build time.** Link `cmplog_shim.c` (needs `-ldl`) alongside your target:

```bash
gcc -shared -fPIC -O2 \
    -include src/fuzzer_tool/adapters/afl_shim.c \
    src/fuzzer_tool/adapters/cmplog_shim.c \
    targets/lz4_read.c \
    -o targets/lz4_read.so \
    -llz4 -ldl
```

The fuzzer auto-detects the built-in cmplog by scanning for the `__cmplog_reset` symbol and keeps using `direct_lite` mode (no fork overhead). The log file is truncated between executions via `__cmplog_reset()`, which the fuzzer calls via ctypes after reading tokens.

To verify cmplog is active from the .so itself, check the startup output:
```
[*] Cmplog: compiled into target .so (direct_lite compatible)
```

### Compiler-IR Comparison Tracing (trace-cmp)

Symbol-based cmplog intercepts libc functions, but GCC -O2 inlines small constant-length `memcmp` into integer compares — no libc call exists to intercept. This is exactly the pattern for format-signature detection (PNG magic, protocol headers, etc.).

**Performance note**: cmplog can produce thousands of comparison pairs per execution
when the library code is heavily instrumented (e.g., with trace-pc-guard coverage).
Each execution writes CMP lines to a log file, and the Python `collect_tokens()`
parses them all (14-23ms for 5000 pairs). To prevent an EPS cliff, the fuzzer
uses **adaptive periodic collection**: once the pair pool exceeds 2000 entries,
cmplog data is collected only 1 in 20 iterations, amortizing the parsing cost
to ~1ms per iteration while still discovering new tokens.

**trace-cmp** solves this by using Clang's `-fsanitize-coverage=trace-cmp` instrumentation, which inserts callbacks at the IR level — after the compiler has already inlined/folded comparisons. This catches every `icmp` that survives optimization.

Both shims coexist: symbol-based (cmplog_shim.c) for explicit libc calls + compiler-based (tracecmp_shim.c) for inlined comparisons. They export different symbols, write to the same `_CMPLOG_OUT` file, and the collector parses both transparently.

```bash
# Build targets with trace-cmp (requires clang)
tools/build_targets.sh --tracecmp --clang

# Build with both cmplog and trace-cmp
tools/build_targets.sh --asan --cmplog --tracecmp --clang
```

The trace-cmp shim intercepts:
- `__sanitizer_cov_trace_cmp{1,2,4,8}` — typed comparison callbacks
- `__sanitizer_cov_trace_const_cmp{1,2,4,8}` — constant-operand variants
- `__sanitizer_cov_trace_switch` — switch statement tracing

#### Vendored trace-cmp targets (libpng + zlib)

The `--vendor-tracecmp` flag rebuilds zlib and libpng from `vendor/` with
`-fsanitize-coverage=trace-cmp,trace-pc-guard`, then links targets against
the instrumented static libraries:

```bash
tools/build_targets.sh --vendor-tracecmp            # Non-ASAN .so targets
tools/build_targets.sh --vendor-tracecmp --asan     # ASAN + tracecmp two-step build
```

Output goes to `targets/png_read_tracecmp.so`, `targets/zlib_read_tracecmp.so`,
`targets/gzip_read_tracecmp.so`, and (with `--asan`)
`targets/png_read_asan_tracecmp.so`. The separate `_tracecmp` suffix avoids
clobbering regular builds.

#### ASAN + tracecmp two-step build

When a target is compiled with both `-fsanitize=address` and
`-fsanitize-coverage=trace-cmp`, ASAN's LD_PRELOAD provides its own
`__sanitizer_cov_trace_cmp*` no-op stubs that would override the tracecmp
shim's logging implementations. The fix: compile `tracecmp_shim.c` with
`-fvisibility=hidden` and link it INTO the target `.so` so the callbacks
resolve locally rather than through the PLT/GOT:

```bash
# Step 1: compile tracecmp shim with hidden visibility
clang -O2 -g -fsanitize=address -fvisibility=hidden -fPIC -c \
    src/fuzzer_tool/adapters/tracecmp_shim.c -o /tmp/tracecmp_shim.o

# Step 2: compile target + link shim together
clang -O2 -g -fsanitize=address -fsanitize-coverage=trace-cmp,trace-pc-guard \
    -shared -fPIC -include src/fuzzer_tool/adapters/afl_shim.c \
    -o targets/my_target_asan.so targets/my_target.c /tmp/tracecmp_shim.o \
    vendor/libpng/.libs/libpng16.a vendor/zlib/libz.a -lm
```

The fuzzer auto-detects the compiled-in tracecmp symbols and uses
`direct_lite` mode with ASAN LD_PRELOAD:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 \
python -m fuzzer_tool fuzz targets/my_target_asan.so -c --cmplog -d corpus
```

#### Auto-detection

The fuzzer auto-detects tracecmp targets by scanning for
`__sanitizer_cov_trace_cmp1` in the binary. When found:
- Direct_lite mode is used (no subprocess overhead)
- The tracecmp shim is preloaded into the process before the target .so
- `_CMPLOG_OUT` is set before loading so the constructor opens the log file
- After each execution, the 256KB internal buffer is flushed to disk and
  comparison operands are extracted as dictionary tokens

### Compiler-Inserted Edge Coverage (`--clang-scov`)

The default coverage scheme uses hand-placed `__afl_map_edge()` calls in
wrapper targets. This only covers the wrapper code — the fuzzer cannot see
which internal code path executed inside the library being fuzzed.

**Clang `-fsanitize-coverage=trace-pc-guard`** solves this by having the compiler
insert `__sanitizer_cov_trace_pc_guard()` at every edge, which the runtime shim
(`afl_shim.c`) delegates to `__afl_map_edge()` → the AFL SHM bitmap.

**Important**: Clang zero-initializes guard variables by default. The shim's
`__sanitizer_cov_trace_pc_guard_init` **must** assign each guard a unique non-zero
value. Without this, `__sanitizer_cov_trace_pc_guard` returns immediately on
`*guard == 0` and every edge is silently skipped — the most common reason for
"0 edges discovered" with trace-pc-guard.

The `--clang-scov` flag or the default `.so` build (when clang is available) passes
`-fsanitize-coverage=trace-pc-guard` to fgrep library compilation. Fgrep `.so`
targets now get full compiler-inserted edge coverage automatically.

```bash
# Build with compiler-inserted edge coverage (requires clang)
tools/build_targets.sh --clang-scov

# Combined with ASAN
tools/build_targets.sh --asan --clang-scov
```

This builds two variants of library-wrapping targets (png_read, zlib_read,
gzip_read, jpeg_read) using vendored library sources compiled with sancov
instrumentation. The vendored sources live in `vendor/` and are compiled
as `.o` files with the same flags, then linked statically.

The existing manual `__afl_map_edge()` calls in wrappers remain — they become
named semantic checkpoints on top of full automatic coverage.

For targets with source already compiled by the build script (fgrep, tailslayer),
the sancov flag is applied directly to their compilation — no vendoring needed.

## Troubleshooting

### Zero edges discovered (ASan + LD_PRELOAD conflict)

If the fuzzer runs but reports `edges: 0` and `map: 0.0%`, the target is likely crashing before AFL instrumentation initializes. The most common cause is `LD_PRELOAD` entries (e.g. `ksm_preload.so`) that load before the ASan runtime, triggering the error:

```
ASan runtime does not come first in initial library list
```

The fuzzer strips conflicting `LD_PRELOAD` entries automatically, but if you set `LD_PRELOAD` manually, ensure it does not contain sanitizer-incompatible libraries. Verify by running:

```bash
python3 -c "
import os, sys; sys.path.insert(0, 'src')
from fuzzer_tool.adapters.process import _clean_env
print(_clean_env(os.environ).get('LD_PRELOAD', '(stripped)'))
"
```

If this prints `(stripped)`, the environment is clean.

## License

MIT
