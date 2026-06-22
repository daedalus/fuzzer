# AGENTS.md — fuzzer-tool

## Overview

Coverage-guided binary fuzzer with ASAN/MSAN/TSAN/UBSAN detection, dictionary mutations, Markov chain generation, and Monte Carlo optimization. CLI tool for fuzzing arbitrary binaries.

## Commands

| Command | Description |
|---------|------------|
| `pytest` | Run test suite |
| `ruff format src/ tests/` | Format code |
| `ruff check src/ tests/` | Lint code |
| `fuzzer-tool --help` | Show CLI help |

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
├── core/           # Domain logic (MarkovChain, MonteCarlo, Sanitizer, Mutations)
├── adapters/       # Process execution, filesystem operations
├── services/       # Fuzzer orchestration
└── cli/            # CLI entry point
```

## Code Style

- Format: ruff format
- Lint: ruff check
- Docstrings: Google style
- Type hints: strict mypy
