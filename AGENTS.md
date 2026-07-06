# AGENTS.md — fuzzer-tool

## Overview

Coverage-guided binary fuzzer with ASAN/MSAN/TSAN/UBSAN detection, dictionary mutations, Markov chain generation, Monte Carlo optimization, kernel crash verification, and state persistence. CLI tool for fuzzing arbitrary binaries.

## Commands

| Command | Description |
|---------|------------|
| `pytest` | Run test suite |
| `ruff format src/ tests/` | Format code |
| `ruff check src/ tests/` | Lint code |
| `fuzzer-tool --help` | Show CLI help |
| `python tools/corpus_png.py --out corpus --download` | Generate PNG corpus |

## Development

```bash
# Setup
pip install -e ".[test]"

# Test
pytest

# Format
ruff format src/ tests/

# Lint
ruff check src/ tests/
```

## Project Structure

```
src/fuzzer_tool/
├── core/           # Domain logic
│   ├── markov.py       # Byte-level Markov chain (with save/load persistence)
│   ├── montecarlo.py   # Thompson sampling + CEM
│   ├── mutations.py    # Mutation operators
│   ├── sanitizer.py    # ASAN/MSAN/TSAN output parsing
│   ├── edge_tracker.py # Per-seed coverage tracking (with save/load)
│   ├── dmesg.py        # Kernel crash verification via dmesg
│   ├── cmplog.py       # Comparison tracing via LD_PRELOAD
│   ├── grammar.py      # Grammar-aware mutations
│   ├── bloom.py        # Bloom filter for dedup
│   ├── crash_metadata.py # Crash enrichment
│   └── elf.py          # ELF parsing utilities
├── adapters/       # Process execution, filesystem operations
├── services/       # Fuzzer orchestration (fuzzer.py, parallel.py, etc.)
└── cli/            # CLI entry point

tools/
├── corpus_png.py      # PNG corpus generator for libpng fuzzing
└── release.sh         # Release automation

dictionaries/
└── png.dict           # PNG format tokens

targets/
├── png_read.c         # libpng fuzz target
├── png_read           # Compiled target
├── test_target.c      # Minimal crash target
└── test_target        # Compiled target
```

## Key Concepts

### State Persistence
Fuzzer state is saved to `{corpus_dir}/state.json` on shutdown. Use `--resume` to continue:
- `state.json` — exec counts, crash sigs, op stats, seed metadata
- `edge_tracker.json` — per-seed edge coverage
- `markov.json` — Markov chain transitions

### Coverage Modes
- `--no-shm` — forces ptrace for uninstrumented binaries
- `--deep-coverage` — capstone disassembly for basic block discovery
- Default SHM — for AFL-instrumented targets

### Kernel Crash Verification
- Async dmesg streaming (`dmesg -l err,warn --json -w`)
- PID-filtered crash attribution
- Requires root or CAP_SYSLOG

### Markov Persistence
- Markov chain saved to `markov.json` on exit
- Loaded on init; skip retrain if loaded to avoid double-counting
- Transitions accumulate across sessions

## Code Style

- Format: ruff format
- Lint: ruff check
- Docstrings: Google style
- Type hints: strict mypy

## Rules

- **Always improve the corpus, never delete it.** Corpus files represent discovered coverage and crash triggers. Only add new inputs, never remove existing ones. Use `fuzzer-tool minimize` to prune redundancies — removed inputs are moved to `corpus/pruned/`, not deleted. The active corpus keeps only inputs that produce the most edges.
- **Do not clean the corpus between runs.** The corpus directory accumulates discovered inputs across sessions. Running `rm -rf corpus/*` destroys coverage history and forces the fuzzer to rediscover everything from scratch. Always use `--resume` to continue. When generating a new corpus (e.g. `corpus_png.py`), write to a fresh directory, not an existing one.
- **Verify claims against code.** Before acting on behavior, type, or API shape, read the source. Don't infer from names.
- **Run the full test suite after changes.** `pytest` must pass before considering any change complete.
