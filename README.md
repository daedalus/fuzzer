# fuzzer-tool

Coverage-guided binary fuzzer with statistical novelty scoring, Markov chain generation, Monte Carlo mutations, and kernel crash verification.

## Features

### Mutation & Generation
- **Mutation operators**: bit flip, byte flip, interesting values (8/16/32-bit), arithmetic (1/2/4/8-byte, LE/BE), block insert/delete/duplicate, havoc mode
- **Dictionary support**: inject protocol tokens from dictionary files
- **Markov chain**: learn byte-level transition probabilities from corpus, generate statistically similar inputs, persist across runs
- **Monte Carlo scheduling**: Thompson sampling bandit for operator selection + Cross-Entropy Method for byte distribution learning
- **Perplexity-gated generation**: model quality dynamically scales generation rate (more generation when model is lost, less when well-calibrated); rejects extreme-perplexity outputs as pure noise

### Coverage & Scoring
- **AFL SHM bitmap** coverage for instrumented targets (~65-200 eps)
- **Ptrace edge coverage** with deep capstone disassembly for closed-source binaries (~18-20 eps)
- **In-process execution**: persistent subprocess mode (~65-120 eps) with auto-restart on crash
- **Good-Turing estimation**: prospective edge discovery count with saturation confidence
- **KS significance testing**: replaces fixed JS thresholds with sample-size-aware p-values
- **CRPS scoring**: proper scoring rule for execution time calibration (fixed indicator direction bug)

### Scheduling Intelligence
- **Subsumption weighting**: seeds fully covered by others get deprioritized
- **Hitcount diversity (JS divergence)**: seeds with unusual frequency profiles get boosted
- **Wasserstein spatial diversity**: seeds exploring different code regions get boosted
- **Perplexity (MDL codelength)**: structurally novel seeds get 1.0-2.0x weight
- **NCD similarity**: Normalized Compression Distance between corpus entries
- **Weight caching**: edge tracker recomputes only when coverage changes

### Crash Analysis
- **Sanitizer detection**: automatic ASAN/MSAN/TSAN/LSAN/UBSAN crash classification
- **Kernel crash verification**: async dmesg streaming for kernel-level crash detection
- **Crash minimization**: delta-debugging to smallest reproducer (`tmin` subcommand)
- **Corpus minimization**: greedy set-cover over SHM edge bitmaps (`minimize` subcommand)
- **Crash exploitability tiers**: ASAN_EXPLOITABILITY classification in reports

### Observability
- **`--report` flag**: full explainability report with coverage, mutations, perplexity, corpus health, edge map
- **`--replay-N` flag**: background crash reproducibility scoring
- **Per-seed cost tracking**: wall-clock time per seed for cost-aware scheduling
- **Discovery rate**: edges per 1k execs over sliding window
- **Bitmap density**: map occupancy percentage (saturation detection)
- **Dup rejection rate**: duplicate-rejection as saturation signal

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

# Generate PNG corpus for libpng fuzzing
python tools/corpus_png.py --out corpus --download
fuzzer-tool fuzz libpng_read.so --inprocess --inprocess-func fuzz_png -c -D dictionaries/png.dict -d corpus/

# Resume a previous fuzzing session
fuzzer-tool fuzz ./target -c --resume

# Run with crash reproducibility scoring
fuzzer-tool fuzz ./target -c --replay-n 3

# Full report after run
fuzzer-tool fuzz ./target -c -n 5000 --report report.txt
```

## Fuzzing Options

| Flag | Description |
|------|-------------|
| `-c` | Enable coverage-guided mode |
| `--no-shm` | Skip AFL SHM, force ptrace |
| `--deep-coverage` | Capstone-based basic block discovery |
| `-F` | File mode (write input to temp file) |
| `-D FILE` | Load dictionary tokens |
| `-g GRAMMAR` | Grammar-aware mutations (built-in: json, http_request, elf) |
| `--cmplog` | Comparison tracing via LD_PRELOAD |
| `--markov-gen` | Markov-generated seeds (rate adapts to model quality via perplexity) |
| `--mc-bandit` | Thompson sampling operator selection (Brier score calibration) |
| `--mc-cem` | Cross-Entropy Method byte distribution |
| `--inprocess` | Persistent subprocess mode (auto-restart on crash) |
| `--resume` | Resume from saved state |
| `-j N` | Parallel fuzzing with N workers |
| `--max-corpus N` | Auto-minimize corpus at N entries |
| `--replay-n N` | Replay each crash N times for reproducibility scoring |
| `--report [FILE]` | Generate explainability report (stdout or file) |

## Coverage Modes

| Mode | Flag | Throughput | Notes |
|------|------|-----------|-------|
| SHM bitmap | `-c` (default) | 65–200 eps | For AFL-instrumented targets |
| In-process | `--inprocess` | 65–120 eps | Persistent loader with crash recovery |
| In-process direct | `--inprocess-direct` | 2k–34k eps | No crash isolation |
| Ptrace basic | `-c --no-shm` | ~20 eps | Function-entry breakpoints |
| Ptrace deep | `-c --no-shm --deep-coverage` | ~18 eps | Capstone BB discovery |

## Statistical Metrics

### Good-Turing Coverage Estimation
Estimates undiscovered edges from the frequency spectrum (N₁²/2N₂). Reports saturation percentage and confidence level. Replaces the old "we found N edges" with "we estimate X edges remain."

### KS Significance Testing
Replaces fixed JS divergence thresholds (JS < 0.01 / JS > 0.1) with sample-size-aware critical values. Early in a run (few samples), thresholds are high; later (many samples), thresholds drop to catch subtle changes.

### CRPS (Continuous Ranked Probability Score)
Proper scoring rule for execution time calibration. Tracks the empirical CDF of execution times and scores new observations. Rising CRPS trend = target runtime behavior is drifting (possible ASLR, cache effects, or behavioral change).

### Perplexity
PP = 2^(codelength_per_byte) from the Markov model. Used to:
- Gate generation rate (high PP → generate more, low PP → generate less)
- Filter generated inputs (reject PP > 512 as pure noise)
- Boost structurally novel seeds in scheduling

## State Persistence

```bash
fuzzer-tool fuzz ./target -c -n 10000
fuzzer-tool fuzz ./target -c --resume -n 10000
```

State files:
- `state.json` — exec counts, crash sigs, op stats, seed metadata
- `edge_tracker.json` — per-seed edge coverage, cumulative edges, global hit counts
- `markov.json` — persisted Markov chain transitions

## In-Process Execution

### Direct ctypes (`--inprocess-direct`)
Calls target function directly via `ctypes.CDLL`. No crash isolation. ~2k–34k eps.

### Persistent subprocess (`--inprocess`)
Keeps one Python subprocess alive. Fork-per-call for crash isolation. Auto-restarts on subprocess death (SIGSEGV from target). No bitmap pipe transfer — parent reads SHM directly. ~65–120 eps.

## Test Suite

528 tests covering all modules. Run with:

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
