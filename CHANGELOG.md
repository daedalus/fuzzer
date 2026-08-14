# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Forkserver on the default execution path** (`--no-forkserver` to opt out).
  `afl_shim.c` now installs an AFL-style forkserver at the end of its
  constructor (`__afl_start_forkserver()`, AFL's protocol on fds 198/199), so
  the target is exec'd once and each input costs a `fork()` from a process
  already past its ELF load, dynamic linker, libc init and ASAN init.
  Measured: 5.27x on `test_target`, 1.38x on an ASAN target with heavy static
  init, **2.77x end to end through the CLI** (484 -> 1341 eps). The ASAN
  figure is the one to plan around — forking a process carrying ASAN's shadow
  mapping is itself expensive, and every target here is built with ASAN.
  Enabled only for the set `run_target_fast` already handled; in-process,
  persistent, network, ptrace, cmplog, perf-counter, `file_mode`,
  `target_args` and multi-target runs are untouched. Targets must be rebuilt
  to benefit — an older target silently takes the fork+exec fallback.
  Note coverage from pre-fork init is no longer re-recorded per execution, so
  edge sets are **not comparable across `--no-forkserver`**; verified to be a
  strict subset differing by a constant, input-independent init set.

### Fixed
- **`fuzz_loader.c` was never a forkserver.** It did `fork()` + `execl()` per
  input, so every execution paid the full ELF load + linker + libc + ASAN init
  anyway. `docs/edge-coverage-analysis.md` §1 prescribed deleting its
  bitmap-file round-trip; that was necessary but measured **0.99x** on an ASAN
  target on its own. See `docs/learnings/2026-08-14-forkserver-that-execs.md`.
- Forkserver coverage never reached the fuzzer: the loader read a bitmap from a
  file while the target wrote to SHM. The child inherits `__AFL_SHM_ID` and the
  shim's constructor attaches on its own, so the bitmap, the `_COV_BITMAP_OUT`
  setenv and the caller-side `memmove` are all gone.
- The forkserver parent recorded its own control flow into the coverage map
  after the per-exec reset, and advanced `__afl_prev_loc` between forks, so the
  same input produced different edge ids on its first executions (5, then 6,
  then a stable 6). `__afl_area` is now detached in the parent, which suppresses
  recording through the null check already at the top of `__afl_map_edge` — no
  added hot-path cost, and `prev_loc` is left untouched.
- The forkserver dropped the child's stderr. ASAN exits 1, so
  `SanitizerReport.parse(stderr)` is the only crash signal `is_crash()` has
  there — the path would have been silently blind to every ASAN finding.
- A zero-length `RUN` produced no reply, blocking the caller until its join
  timeout and then tearing down and restarting the loader.
- An oversized `RUN` was skipped without consuming its body, leaving the payload
  in the pipe to be parsed as the next command.
- Forkserver timeouts reported the raw wait status (`-SIGKILL`). `-9` is in
  `SIGNAL_CRASH_CODES`, so every slow input would have been filed as a
  fatal-signal crash. Now `-1`.
- `ForkserverRunner.__del__` raised `ValueError: write to closed file` through
  the ignored-exception path on every clean exit (`stop()` runs twice).
- The loader piped stdin only, silently moving targets that read `argv[1]` when
  `argc == 2` (`png_read.c`, `grep_read.c`) onto their 64KB stdin path. It now
  stages each input into a file and execs `<target> <file>` with stdin
  redirected from it, mirroring `run_target_fast`.
- `runner.py`'s forkserver branch never reset the edge map, so
  `is_new_coverage_with_edges()` would have seen every execution since start
  accumulated together.
- `ShmCoverage.resize()` allocates a new segment and removes the old one, but
  the loader's environment is only read at exec time — its children kept
  attaching to the removed segment. Both resize sites now respawn the loader.
- The loader was hardcoded to `gcc`; it now prefers `clang` (hard rule 4).
- **cmplog/edge shim merge: a preloaded `cmplog_shim.so` could silently zero a
  target's coverage.** `cmplog_shim.c` carried a second copy of the edge
  machinery behind `weak` definitions of `__afl_map_shm` / `__afl_map_reset` /
  `__sanitizer_cov_trace_pc_guard{,_init}`. `weak` only loses to a strong
  definition at *static* link time; at dynamic link time the first definition
  in the global lookup scope wins regardless of binding, and `LD_PRELOAD`
  precedes dependency `.so`s. Measured on a `.so` target built without
  `-Wl,-Bsymbolic`: `__afl_area` = `0x7f4c757d6018` with no preload, `(nil)`
  with the shim preloaded — the run recorded zero edges. The four
  `_tracecmp.so` targets (`png_read`, `zlib_read`, `gzip_read`, `jpeg_read`)
  are built without `-Bsymbolic`, so they were exposed. The comparison layers
  now live in `afl_shim.c` behind `-D__AFL_CMPLOG=1`, and the `LD_PRELOAD`
  artifact is built from the same source with `-D__AFL_PRELOAD_ONLY`, which
  defines none of the `__afl_*` symbols.
- cmplog shim: the coverage segment was attached twice per exec (2 `shmat`
  against 1 for the edge shim alone) — the shim's own constructor re-entered
  `__afl_map_shm`, which in a combined link resolved to the strong definition.
- cmplog shim: the comparison buffer was flushed on the **first** crash only.
  Its crash handler restored the previous disposition permanently, so every
  later crash in a persistent/`direct_lite` loop lost up to 256KB of buffered
  records. The flush now lives in `__afl_crash_handler`, before the
  `siglongjmp`, and runs on every crash.
- cmplog shim: `AFL_MAP_SIZE` was read as *bytes* there and as *entries* in
  `afl_shim.c`. A live `__afl_map_reset` from the old shim would have zeroed
  the 24-byte SHM header and the first ~1021 entries. Only one definition
  survives the merge, so the unit is unambiguous.
- cmplog shim: `real_memcmp` and friends were resolved in the constructor and
  dereferenced unconditionally — a NULL call for any comparison reaching the
  interceptor before that constructor ran. Tolerable for an `LD_PRELOAD`
  object that loads early; not for one constructor among many in the target.
  Resolution is now lazy, with a re-entrancy guard (`dlsym` calls the very
  functions being interposed) and naive fallbacks.
- The comparison logger is no longer instrumented by the coverage it enables.
  With `-include` the layer lands in the *target's* translation unit, so
  `-fsanitize-coverage=trace-cmp` instruments the record writer, whose own
  comparisons call back into it — unbounded recursion arriving as a
  stack-overflow SIGSEGV at startup. Reproduced on `gcc -D__AFL_CMPLOG=1
  -fsanitize-coverage=trace-cmp`; fixed with `__AFL_NO_COV`
  (`no_sanitize_coverage` / `no_sanitize("coverage")`) on every function in
  the layer, plus a thread-local re-entrancy flag for toolchains without the
  attribute. The old build dodged this by compiling the shim as a separate
  uninstrumented object; that protection does not survive the merge.
- `CmplogCollector.collect_tokens`: the optional PC field was parsed with
  `int(s)` — base 10 — while the shim writes it in the `%p` convention
  (`0x55f65c387346`). Every parse raised `ValueError` into a `suppress()`, so
  `pc` was silently `None` for every record ever written and `_pair_pc` has
  always been empty. Now `int(s, 0)`, which still reads plain decimal, so
  existing logs keep parsing.
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
- `src/fuzzer_tool/adapters/cmplog_shim.c` is **removed**. Its libc
  interposition and trace-cmp layers are in `afl_shim.c`; enable with
  `-D__AFL_CMPLOG=1` (needs `-ldl`). Off by default, which keeps the
  interposers and the `-ldl` dependency out of targets that do not want them,
  and keeps `__cmplog_reset` out of their symbol tables — that symbol is what
  `services/fuzzer.py::_detect_cmplog` reads to decide whether `direct_lite`
  is safe, so a layer that always defined it would make the probe a constant.
- `tools/build_targets.sh`: `$CMPLOG_SHIM` and the per-target `cmplog_shim.o`
  compile/link/cleanup dance are replaced by `$CMPLOG_CFLAGS` /
  `$CMPLOG_LIBS`. The `tailslayer_read` C++ target keeps cmplog off (the
  interceptors use C signatures; C++ overloads their const-ness — the old
  build compiled the shim as a separate C object precisely to dodge that).
  MSAN/TSAN targets keep cmplog off as before: unmeasured rather than assumed
  safe.
- The trace-cmp callbacks are compiled `visibility("hidden")` in
  `-D__AFL_CMPLOG=1` builds, so nothing — libasan's weak stubs, an older
  preloaded shim — can interpose them. `nm` reports them as `t` (local)
  rather than `T`; the build script's post-link check accepts both.
- The comparison log is written through a raw fd and `write(2)` rather than
  `FILE*`/`fwrite`. The pre-crash flush runs inside a signal handler, where
  stdio is not async-signal-safe; this also drops the stdio lock from a path
  that runs on every intercepted comparison. Record format is unchanged.
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
- trace-cmp targets compile the callbacks in instead of relying on `LD_PRELOAD`.
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
