# SPEC.md — fuzzer-tool

## Purpose

Coverage-guided binary fuzzer with ASAN/MSAN/TSAN/UBSAN detection, dictionary mutations, Markov chain generation, and Monte Carlo optimization. Provides a CLI tool for fuzzing arbitrary binaries via stdin or file mode, with automatic crash deduplication and signature tracking.

## Scope

### In scope
- CLI fuzzer targeting arbitrary binaries (stdin and file mode)
- Mutation operators: bit flip, byte flip, interesting values (8/16/32-bit), random bytes, block insert/delete/duplicate, havoc
- Dictionary-based mutations from external token files
- Markov chain byte-level generation trained on corpus
- Monte Carlo Thompson sampling bandit for mutation operator selection
- Monte Carlo cross-entropy method for byte distribution learning
- Sanitizer output parsing (ASAN, MSAN, TSAN, LSAN, UBSAN)
- Crash deduplication via SHA-256 hashing
- Crash signature tracking with stack frame extraction
- Corpus management (load, save, deduplicate)
- Configurable timeouts, max input length, mutations per input
- Coverage-guided mode (AFL_MAP_SIZE passthrough)
- Ptrace-based edge coverage with basic block discovery
- Per-mutation-op usage statistics
- Timeout/crash rate tracking
- Memory usage tracking (peak RSS)
- Periodic stats dump to JSON file

### NOT in scope
- Network-based fuzzing
- GUI or web interface
- Distributed/multi-process fuzzing
- Custom compiler instrumentation (beyond AFL_MAP_SIZE)
- Hypervisor-based execution

## Public API / Interface

### CLI

```
fuzzer-tool <target> [options]
```

Arguments:
- `target` (required): Path to target binary

Options:
- `-d, --corpus DIR`: Corpus directory
- `-o, --crashes DIR`: Crashes directory
- `-m, --max-len N`: Max input length (default: 4096)
- `-t, --timeout SEC`: Timeout in seconds (default: 5)
- `-n, --iterations N`: Number of iterations, 0=infinite (default: 0)
- `-M, --mutations N`: Mutations per input (default: 8)
- `-c, --coverage`: Enable coverage-guided mode
- `--deep-coverage`: Enable capstone-based BB discovery
- `--max-bps N`: Max breakpoints for deep coverage (default: 50000)
- `-D, --dict FILE`: Dictionary file
- `-F, --file-mode`: Write input to temp file instead of stdin
- `-A, --target-args ...`: Target arguments ({file} placeholder)
- `--markov`: Enable Markov chain mutation
- `--markov-gen`: Enable Markov chain seed generation
- `--markov-order N`: Markov chain order (default: 1)
- `--mc-bandit`: Enable Thompson sampling bandit
- `--mc-cem`: Enable cross-entropy method
- `--mc-elite-frac FLOAT`: CEM elite fraction (default: 0.1)
- `--mc-refit-int N`: CEM refit interval (default: 1000)
- `--stats-file FILE`: Save stats to JSON file periodically
- `--stats-interval N`: Stats dump interval (default: 1000)

### Core Classes

#### `MarkovChain`
- `__init__(order=1, smoothing=1e-6)`: Initialize with n-gram order and Laplace smoothing
- `train(data: bytes)`: Learn byte transitions from data
- `train_corpus(corpus: list[bytes])`: Train on multiple inputs
- `generate(length: int) -> bytes`: Generate input from learned distribution
- `sample_byte(ctx: bytes) -> int`: Sample one byte given context
- `is_trained() -> bool`: Check if any transitions observed

#### `MonteCarloScheduler`
- `__init__(elite_frac=0.1, refit_interval=1000)`: Initialize with CEM parameters
- `init_arm(name: str)`: Register a mutation operator arm
- `select_op(ops: list[str]) -> str`: Thompson sample to select operator
- `record(name: str, success: bool)`: Update arm statistics
- `add_elite(data: bytes, score: int)`: Add to elite set (bounded to 200)
- `maybe_refit()`: Refit CEM distribution if interval reached
- `cem_byte(pos: int) -> int`: Sample byte at position from CEM distribution
- `cem_sample(length: int) -> bytes`: Generate full input from CEM
- `bandit_stats() -> dict[str, tuple[float, float]]`: Get arm success/failure counts

#### `SanitizerReport`
- `parse(stderr: str) -> SanitizerReport | None`: Parse sanitizer output
- `is_valid() -> bool`: Check if report has valid sanitizer and error type
- Attributes: sanitizer, error_type, fault_addr, frames, raw, signature

#### `Fuzzer`
- `__init__(target, corpus_dir, crashes_dir, ...)`: Initialize fuzzer with all options
- `fuzz_one(data: bytes) -> bool`: Mutate, execute, check result
- `mutate(data: bytes) -> bytes`: Apply mutation operators
- `run(iterations=0)`: Main fuzzing loop

### Module-level Functions

- `load_dictionary(path: str) -> list[bytes]`: Parse dictionary file
- `parse_dict_line(line: str) -> bytes | None`: Parse single dictionary line

## Data Formats

### Dictionary File Format
One token per line. Lines starting with `#` are comments. Empty lines ignored.
Formats: `NAME=value` (name ignored, value used) or raw bytes.

### Crash Metadata File
Text file alongside crash binary with:
- returncode, sanitizer info, error type, fault address
- Signature (sanitizer:type@frame1@frame2...)
- Stack trace (up to 12 frames)
- Raw stderr

### Corpus Files
Binary files named `id_{sha256_prefix}` in corpus directory.

### Stats JSON File
Periodic dump with: timestamp, exec_count, crash_count, timeout_count,
corpus_size, eps, peak_rss_kb, op_counts, op_success, bandit_stats, cem state.

## Edge Cases

1. Empty input buffer: fuzzer generates random bytes of random length (1-32)
2. Target binary not found or not executable: exit with error message
3. Target timeout: process group killed, counted as timeout (returncode -1)
4. Empty corpus: seeds with `b"AAAAAAAA"` default
5. Duplicate crash: deduplicated by SHA-256 hash, only first saved
6. Markov chain with 0-length data: no transitions learned, remains untrained
7. CEM with empty elite set: no distribution fitted, `cem_bytes` not offered as mutation
8. Bandit with single arm: always selects that arm (degenerate case)
9. Max input length reached: block_insert skipped
10. Dictionary with invalid escape sequences: handled via `errors="replace"`

## Performance & Constraints

- No external dependencies (stdlib only, capstone optional)
- O(1) per mutation operation
- O(n) corpus loading where n = number of corpus files
- Memory: corpus held in memory, elite set bounded to 200 entries
- CEM byte_freq: sparse dict-of-dicts, bounded by elite input lengths
