# fuzzer-tool

Coverage-guided binary fuzzer with static target analysis, statistical novelty scoring, Markov chain generation, Monte Carlo mutations, kernel crash verification, and format-aware grammar mutations.

## Features

### Mutation & Generation
- **Mutation operators**: bit flip, byte flip, interesting values (8/16/32-bit), arithmetic (1/2/4/8-byte, LE/BE), block insert/delete/duplicate, havoc mode
- **Grammar-aware mutations**: format-specific mutations for PNG (IHDR fields, IDAT splitting, filter types, interlace, CRC corruption, color/depth combos, large inputs)
- **Dictionary support**: inject protocol tokens from dictionary files
- **Markov chain**: learn byte-level transition probabilities from corpus, generate statistically similar inputs, persist across runs
- **Monte Carlo scheduling**: Thompson sampling bandit for operator selection + Cross-Entropy Method for byte distribution learning
- **Perplexity-gated generation**: model quality dynamically scales generation rate (more generation when model is lost, less when well-calibrated); rejects extreme-perplexity outputs as pure noise

### Static Target Analysis
- **TargetProfiler**: ELF static analysis at startup — extracts string constants, function boundaries, magic bytes, and input format hints
- **Auto-populated dictionary**: interesting strings (format specifiers, error messages, keywords) and magic bytes extracted from `.rodata`
- **Format-aware seed generation**: produces structurally meaningful initial seeds (PNG headers, text protocols, JSON, XML, HTML) based on inferred format
- **Hot-function weighting**: seeds exercising high-branch-density functions get a proportional boost in selection

### Coverage & Scoring
- **AFL SHM bitmap** coverage for instrumented targets (~65-200 eps)
- **Ptrace edge coverage** with deep capstone disassembly for closed-source binaries (~18-20 eps)
- **In-process execution**: persistent subprocess mode (~65-120 eps) with auto-restart on crash
- **Branch density**: static analysis metric (conditional branches/KB) for target complexity estimation
- **Auto-sized edge bitmap**: `estimate_map_size()` from branch density × .text size replaces hardcoded 65536
- **Good-Turing estimation**: prospective edge discovery count with saturation confidence
- **KS significance testing**: replaces fixed JS thresholds with sample-size-aware p-values
- **CRPS scoring**: proper scoring rule for execution time calibration (fixed indicator direction bug)

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

### Game Theory
- **Shapley value** (`--shapley`): fair operator credit distribution accounting for synergistic effects between mutation operators
- **Replicator dynamics** (`--replicator`): evolutionary game theory scheduling — operators grow proportionally to fitness, converging to evolutionarily stable strategies
- **MOpt PSO** (`--mopt`): particle swarm optimization over operator distributions (alternative to Thompson sampling)

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
- **Branch density**: static analysis at startup (`cond branches/KB`)
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
| `--cmplog` | Comparison tracing via LD_PRELOAD |
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

## Subcommands

| Command | Description |
|---------|-------------|
| `fuzz` | Run coverage-guided fuzzing (default) |
| `rank` | Rank corpus seeds by interestingness (edge coverage, rarity, subsumption) |
| `minimize` | Minimize corpus by removing redundant inputs |
| `tmin` | Minimize a crash to smallest reproducer |
| `replay` | Replay a crash input against the target |
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
Calls target function directly via `ctypes.CDLL`. Catches SIGSEGV via signal handler. ~2k–34k eps.

### Persistent subprocess (`--inprocess`)
Keeps one Python subprocess alive. Fork-per-call with `os.setsid()` for process group isolation. Timeout enforced via outer threaded readline. Auto-restarts on subprocess death. ~65–120 eps.

## Corpus Minimization

```bash
# Basic minimization (greedy set-cover)
fuzzer-tool minimize ./target -d corpus -c

# Rate-distortion optimal pruning (preserves coverage diversity)
fuzzer-tool minimize ./target -d corpus -c --rate-distortion --target-frac 0.95
```

## Test Suite

1024 tests covering all modules. Run with:

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

## License

MIT
