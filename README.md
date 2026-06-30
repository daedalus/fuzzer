# fuzzer-tool

Coverage-guided binary fuzzer with Markov chain, Monte Carlo mutations, kernel crash verification, and state persistence.

## Features

- **Mutation operators**: bit flip, byte flip, interesting values (8/16/32-bit), arithmetic (1/2/4/8-byte, LE/BE), block insert/delete/duplicate, havoc mode
- **Dictionary support**: inject protocol tokens from dictionary files
- **Markov chain**: learn byte-level transition probabilities from corpus and generate statistically similar inputs; persists across runs via `markov.json`
- **Monte Carlo scheduling**: Thompson sampling bandit for operator selection + Cross-Entropy Method for byte distribution learning
- **Sanitizer detection**: automatic crash classification via ASAN/MSAN/TSAN/LSAN/UBSAN output parsing
- **Kernel crash verification**: async dmesg streaming for kernel-level crash detection (segfaults, OOM, KASAN, etc.)
- **Coverage-guided mode**: ptrace-based edge coverage with deep capstone disassembly for closed-source binaries
- **In-process execution**: bypass subprocess overhead for maximum throughput
- **File mode**: fuzz targets that read from files instead of stdin
- **State persistence**: save/resume fuzzer state (corpus metadata, edge tracker, exec counts) across runs
- **Corpus tools**: built-in PNG corpus generator under `tools/`

## Installation

```bash
pip install -e ".[dev]"

# Optional: with capstone for deep coverage
pip install -e ".[dev,capstone]"
```

## Quick Start

```bash
# Basic fuzzing
fuzzer-tool fuzz ./target

# With coverage and dictionary
fuzzer-tool fuzz -c -D dictionary.txt ./target

# File-based target
fuzzer-tool fuzz -F -A "{file}" ./target

# With Markov and Monte Carlo
fuzzer-tool fuzz --markov --markov-gen --mc-bandit --mc-cem ./target

# Generate PNG corpus for libpng fuzzing
python tools/corpus_png.py --out corpus_png --download
fuzzer-tool fuzz ./png_read -c --no-shm -F -D dictionaries/png.dict -d corpus_png/

# Resume a previous fuzzing session
fuzzer-tool fuzz ./target -c --resume
```

## Fuzzing Options

| Flag | Description |
|------|-------------|
| `-c` | Enable coverage-guided mode |
| `--no-shm` | Skip AFL SHM, force ptrace (for uninstrumented binaries) |
| `--deep-coverage` | Capstone-based basic block discovery |
| `-F` | File mode (write input to temp file) |
| `-D FILE` | Load dictionary tokens |
| `-g GRAMMAR` | Grammar-aware mutations (built-in: json, http_request, elf) |
| `--cmplog` | Comparison tracing via LD_PRELOAD |
| `--markov-gen` | Markov-generated seeds (15% of selections) |
| `--mc-bandit` | Thompson sampling operator selection |
| `--mc-cem` | Cross-Entropy Method byte distribution |
| `--inprocess` | Persistent subprocess mode |
| `--resume` | Resume from saved state |
| `-j N` | Parallel fuzzing with N workers |
| `--max-corpus N` | Auto-minimize corpus at N entries |

## Coverage Modes

| Mode | Flag | Throughput | Notes |
|------|------|-----------|-------|
| Ptrace deep | `-c --no-shm --deep-coverage` | ~18 eps | Best coverage, needs uninstrumented binary |
| Ptrace basic | `-c` | ~20 eps | Function-entry breakpoints only |
| SHM | `-c` (default) | ~20 eps | For AFL-instrumented targets |
| In-process | `--inprocess` | ~73 eps | No deep coverage |
| In-process direct | `--inprocess-direct` | ~2k-34k eps | No crash isolation |

## State Persistence

Fuzzer state (seed metadata, edge tracker, execution counts, crash signatures) is saved to `{corpus_dir}/state.json` on shutdown. Use `--resume` to continue from where you left off:

```bash
# Run 1: fuzz for a while
fuzzer-tool fuzz ./target -c -n 10000

# Run 2: resume with accumulated state
fuzzer-tool fuzz ./target -c --resume -n 10000
```

State files:
- `state.json` — exec counts, crash sigs, op stats, seed metadata
- `edge_tracker.json` — per-seed edge coverage for subsumption scheduling
- `markov.json` — persisted Markov chain transitions

## Kernel Crash Verification

The fuzzer polls `dmesg` asynchronously to verify crashes at the kernel level. This catches crashes that userspace exit codes might miss (OOM kills, kernel BUGs, etc.):

```
[*] Kernel-verified crashes: 3
    segfault: 2
    oom: 1
```

Requires root or `CAP_SYSLOG`. Warns gracefully if unavailable.

## Corpus Tools

```bash
# Generate diverse PNG corpus for libpng fuzzing
python tools/corpus_png.py --out corpus_png --download

# Options:
#   --out DIR       Output directory (default: corpus_png)
#   --count N       Max seeds to generate (0=all)
#   --download      Also fetch real-world PNGs from the internet
```

## In-Process Execution Modes

### Direct ctypes (`--inprocess-direct`)

Calls the target function directly via `ctypes.CDLL` — zero subprocess overhead.

```bash
fuzzer-tool fuzz libfoo.so --inprocess-direct
```

| Metric | Value |
|--------|-------|
| Throughput | ~2k–34k execs/sec |
| Crash isolation | None (crash kills fuzzer) |
| Coverage | No |

### Persistent subprocess (`--inprocess`)

Keeps one subprocess alive across iterations, communicating via pipes.

```bash
fuzzer-tool fuzz ./target --inprocess
```

| Metric | Value |
|--------|-------|
| Throughput | ~350 execs/sec (C loader) |
| Crash isolation | Full |
| Coverage | Yes (standalone executables) |

## Performance Summary

| Mode | execs/sec | Crash isolation | Coverage |
|------|-----------|-----------------|----------|
| Default subprocess | ~20 | Full | Yes (instrumented targets) |
| `--inprocess-direct` | ~2k–34k | None | No |
| `--inprocess` (C loader) | ~350 | Full | Yes |
| Ptrace deep | ~18 | Full | Yes (all basic blocks) |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
ruff format src/ tests/
```

## License

MIT
