# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `ShmCoverage`: hit counts are now part of the novelty decision. The primary
  coverage path decided interestingness by `ids - _seen_edge_ids` — set
  membership only — so an input driving a loop 2 times and one driving it 128
  times were the same coverage, and loop-count-guarded branches (`if (n > 16)`,
  buffer-growth paths, parser backtrack limits) were invisible. The `count`
  field was maintained faithfully by the shim and read back by
  `get_edge_counts()`; nothing consulted it.
- `core/count_class.py`: added `bucket_bit`/`bucket_bits`, AFL's
  `count_class_lookup8` as *bits*. `classify_single` returns representative
  values (0, 1, 2, 3, 4, 8, …) which are not disjoint — class 3 is `0b11`,
  class 1 OR'd with class 2 — so a virgin map built from them silently drops
  a hit count of exactly 3 from any edge already seen once and twice.
  `classify_*` semantics are unchanged; the new ladder is separate.
- cmplog shim: `memmem`/`strstr`/`strcasestr` passed a hardcoded `-1` as the
  comparison result, so a *successful* substring match bypassed `log_cmp`'s
  `result == 0` filter and was pooled as an unsolved comparison.
- `runner`: the bounded wait for a child's first ptrace stop charged the full
  per-exec `timeout`, which the run loop then charged again — worst case 2x the
  configured budget. Capped independently by `_INITIAL_STOP_TIMEOUT` (1.0s).
- `CmplogCollector.start`: superseded digest-keyed shim objects and the legacy
  fixed-name `fuzz_cmplog_shim.so` are now pruned once the current object is on
  disk, instead of accumulating in `~/.cache/fuzzer_cmplog/` forever.

### Changed
- The SHM virgin bucket map is indexed by `edge_id` into a dense `uint8` array
  rather than a dict, because it runs on every coverage-changing execution.
  Measured on 200k active edges: 1.8ms direct-indexed against 28.7ms via a
  sorted-array `searchsorted` and 108ms via a per-entry dict loop — 6-12% on
  top of the `set(...) - _seen_edge_ids` diff already on that path, against
  ~500% for the dict loop. Affordable because guard values are small
  sequential integers, so `edge_id = prev_loc ^ cur_loc` stays in a range of
  roughly `2 * guard_count` and XOR with a context term cannot widen it past
  its wider operand. `__AFL_CTX_BITS` in the 24..32 range is the exception and
  falls back to a dict (`VIRGIN_DENSE_MAX`).
- Comparison constants are now visible on optimized targets. `-fno-builtin-*`
  (`$NOBUILTIN_CMP`) keeps `memcmp`/`strcmp` at the PLT so the libc layer sees
  their operands at `-O2`; `-fsanitize-coverage=trace-cmp` cannot recover them
  at any optimization level, because SanitizerCoverage instruments IR `icmp`
  and clang's `ExpandMemCmp` runs after it. Measured on
  `targets/cmplog_exercise.c`: 0/10 constants at `-O2`, 10/10 with the flags.
- trace-cmp targets link `cmplog_shim.o` instead of relying on `LD_PRELOAD`.
  `-fsanitize-coverage` links compiler-rt's sancov runtime, whose weak no-op
  `__sanitizer_cov_trace_*cmp*` stubs win the symbol lookup against a
  preloaded shim; the callbacks fired 20 times and logged nothing.
- `WITH_TRACECMP` defaults to on (`--no-tracecmp` opts out) and now covers
  `cmplog_exercise` as well as `tracecmp_target`, built as `*_tcg`.
- mypy is ratcheted rather than permanently red: `strict = true` remains the
  target, the 114 modules that cannot yet satisfy it are exempted by name in
  `[[tool.mypy.overrides]]`, and the other 17 are checked strictly. New modules
  are strict by default. `tests/test_regression_mypy_ratchet.py` enforces that
  the list only shrinks.
- CI installs clang and the `smt` extra, and fails if either is missing. Those
  105 tests previously skipped silently.
- The initial-ptrace-stop regression test runs on an injected virtual clock
  instead of a wall-clock threshold, making the exec budget directly
  observable and the assertion coverage-insensitive.

## [0.1.0] - 2025-01-01

### Added
- Core mutation operators (bit flip, byte flip, interesting values, block ops, havoc)
- Dictionary support with token injection
- Markov chain byte-level generation and mutation
- Thompson sampling bandit for operator selection
- Cross-entropy method for per-position byte distribution learning
- Sanitizer output parsing (ASAN, MSAN, TSAN, LSAN, UBSAN)
- Crash deduplication via signature generation
- Coverage-guided mode with ptrace breakpoints
- Deep coverage via x86-64 decoder disassembly
- File-mode execution for file-reading targets
- CLI with argparse
- pytest test suite
- CI pipeline with GitHub Actions
