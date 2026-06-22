# fuzzer-tool

Coverage-guided binary fuzzer with Markov chain and Monte Carlo mutations.

## Features

- **Mutation operators**: bit flip, byte flip, interesting values (8/16/32-bit), block insert/delete/duplicate, havoc mode
- **Dictionary support**: inject protocol tokens from dictionary files
- **Markov chain**: learn byte-level transition probabilities from corpus and generate statistically similar inputs
- **Monte Carlo scheduling**: Thompson sampling bandit for operator selection + Cross-Entropy Method for byte distribution learning
- **Sanitizer detection**: automatic crash classification via ASAN/MSAN/TSAN/LSAN/UBSAN output parsing
- **Coverage-guided mode**: ptrace-based edge coverage for closed-source binaries (with optional capstone disassembly for deep basic block discovery)
- **File mode**: fuzz targets that read from files instead of stdin

## Installation

```bash
pip install -e ".[dev]"

# Optional: with capstone for deep coverage
pip install -e ".[dev,capstone]"
```

## Usage

```bash
# Basic fuzzing
fuzzer-tool ./target

# With coverage and dictionary
fuzzer-tool -c -D dictionary.txt ./target

# File-based target
fuzzer-tool -F -A "{file}" ./target

# With Markov and Monte Carlo
fuzzer-tool --markov --markov-gen --mc-bandit --mc-cem ./target
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
mypy src/
```

## License

MIT
