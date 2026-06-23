# fuzzer-tool

Coverage-guided binary fuzzer with Markov chain, Monte Carlo mutations, and in-process execution modes.

## Features

- **Mutation operators**: bit flip, byte flip, interesting values (8/16/32-bit), block insert/delete/duplicate, havoc mode
- **Dictionary support**: inject protocol tokens from dictionary files
- **Markov chain**: learn byte-level transition probabilities from corpus and generate statistically similar inputs
- **Monte Carlo scheduling**: Thompson sampling bandit for operator selection + Cross-Entropy Method for byte distribution learning
- **Sanitizer detection**: automatic crash classification via ASAN/MSAN/TSAN/LSAN/UBSAN output parsing
- **Coverage-guided mode**: ptrace-based edge coverage for closed-source binaries, SHM bitmap for instrumented targets
- **In-process execution**: bypass subprocess overhead for maximum throughput
- **File mode**: fuzz targets that read from files instead of stdin

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
```

## In-Process Execution Modes

In-process modes bypass the default subprocess-per-iteration model for higher throughput.

### Direct ctypes (`--inprocess-direct`)

Calls the target function directly via `ctypes.CDLL` — zero subprocess overhead.

```bash
# Target must handle errors internally (setjmp/longjmp or sanitizers)
fuzzer-tool fuzz libfoo.so --inprocess-direct
```

| Metric | Value |
|--------|-------|
| Throughput | ~2k–34k execs/sec |
| Crash isolation | None (crash kills fuzzer) |
| Coverage | No |

### Persistent subprocess (`--inprocess`)

Keeps one subprocess alive across iterations, communicating via pipes. The subprocess loads the target once and calls it repeatedly.

```bash
fuzzer-tool fuzz ./target --inprocess
```

| Metric | Value |
|--------|-------|
| Throughput | ~350 execs/sec (C loader) |
| Crash isolation | Full |
| Coverage | Yes (standalone executables) |

The C loader (`fuzz_loader.c`) compiles at startup for maximum speed. Falls back to Python loader if gcc/clang unavailable.

### Coverage-guided in-process

Coverage works with standalone executables that dump the coverage bitmap to a file. The target must:
1. Be compiled with `-fsanitize-coverage=inline-8bit-counters`
2. Implement `__sanitizer_cov_8bit_counters_init` to capture the bitmap pointer
3. Write the bitmap to `_COV_BITMAP_OUT` before exit

```bash
# Build with coverage dump
clang -O1 -fsanitize-coverage=inline-8bit-counters \
  -o fuzz_target fuzz_target.c -lpng -lz

# Fuzz with coverage
fuzzer-tool fuzz ./fuzz_target --inprocess -c -d corpus/ -D png.dict
```

**Target template:**
```c
static uint8_t *cov_bitmap = NULL;
static size_t cov_size = 0;

void __sanitizer_cov_8bit_counters_init(uint8_t *start, uint8_t *stop) {
    cov_bitmap = start; cov_size = stop - start;
}

int main(void) {
    // ... run target ...
    const char *out = getenv("_COV_BITMAP_OUT");
    if (out && cov_bitmap) {
        FILE *f = fopen(out, "wb");
        fwrite(cov_bitmap, 1, cov_size, f);
        fclose(f);
    }
}
```

## Performance Summary

| Mode | execs/sec | Crash isolation | Coverage |
|------|-----------|-----------------|----------|
| Default subprocess | ~20 | Full | Yes (instrumented targets) |
| `--inprocess-direct` | ~2k–34k | None | No |
| `--inprocess` (C loader) | ~350 | Full | Yes |
| `--inprocess` (Python) | ~21 | Full | Yes |

## Limitations

See [docs/inprocess-limitations.md](docs/inprocess-limitations.md) for detailed coverage limitations, architecture notes, and known issues.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
mypy src/
```

## License

MIT
