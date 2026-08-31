# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **`os.environ` writes were never restored, leaking state across runs in
  the same process.** Bug report 2026-08-21 HIGH #10. `run()` only ever
  undid the cmplog shim's own `LD_PRELOAD` edit (`self._cmplog.restore_env()`)
  — `__AFL_DIST_SHM_ID`, `__AFL_SHM_ID`, `AFL_MAP_SIZE`, the ASAN `LD_PRELOAD`
  injection, and `UBSAN_OPTIONS` were all written directly into the process
  environment and never put back. That leaked into whatever ran next in the
  same process: the next target in a multi-target session, a caller
  embedding `Fuzzer` as a library, or the next test in a pytest run. Added a
  process-wide snapshot taken once in `__init__` (before any of the above
  can mutate it) and a restore that runs both at the natural end of `run()`
  and via `atexit` as a safety net for crashes/SIGTERM/SIGINT. Regression
  tests in `tests/test_regression_environ_restore.py`.
- **Findings #12 and #13 (2026-08-21 bug report) fixed and pruned from
  `docs/bugreport_2026-08-21_merged.md`** — multi-target mode reading edges
  from the wrong SHM segment, and `--inprocess-direct` freezing permanently
  on the first wild-pointer crash. See `a927e62` and `ffc4437` /
  `ca9fdd0`.
- **`RandPool.sample()` raised `TypeError` on a `range` population.** The
  dispatch was `isinstance(population, list | tuple | bytes)`, so a `range`
  missed the sequence branch, fell through to the int branch, and died on
  `k > population` comparing an int to a range. `rng.sample(range(n), k)` is
  the idiomatic `random.sample` call and roughly a dozen structure-aware
  mutators use it (`asf.py`, `riff.py`, `mp3.py`, `adts.py`, …). It only
  fires when the pooled RNG is active rather than stdlib `random`, so it
  presented as a seed-dependent flake in `test_mutate_includes_splice`
  rather than as what it is: an exception raised mid-campaign from an
  ordinary call. `bytearray` was missing from the same tuple and is added
  with it.
- **`DEFAULT_CC` swallowed the gcc-fallback warning on any box without clang,
  breaking every vendored-library compile.** `_pick_cc()` emitted its warning
  through `warn()`, which writes to *stdout*, and the whole of its stdout is
  captured by `DEFAULT_CC="$(_pick_cc)"` — so `DEFAULT_CC` became the warning
  text followed by `gcc`, which is not a command. Every helper that compiles
  with `$cc` redirects stderr to `/dev/null`, so the only visible symptom was a
  run of "objects failed" warnings and targets silently missing from the
  output. Redirected to stderr, with a regression test in
  `tests/test_regression_build_flags.py`.
- **`run_target_fast` had no timeout, deadlocked on chatty targets, and leaked
  the child it failed on.** Bug report 2026-08-21 CRITICAL #3. This is the
  *default* spawn-fallback path — `run_target` picks it whenever the run is
  neither `file_mode` nor cmplog — and it was the only backend not honouring
  `f.timeout`. Three defects, each independently fatal:

  | defect | old behaviour | measured |
  |---|---|---|
  | `os.waitpid(pid, 0)`, no deadline | one looping input hangs the campaign | SIGKILLed at 15s, never returned |
  | stderr read after the reap | >64 KiB fills the pipe; child blocks in `write()`, parent in `waitpid()` | SIGKILLed at 15s, never returned |
  | `except: return -2, str(e), 0` | crash attribution lost, child never reaped | — |

  After: the looper returns `rc=-1` at the deadline, and the 400 KiB-stderr
  target returns `rc=3` with 65536 B captured in 0.77s.

  The bound is enforced by `poll()` on the stderr pipe, which doubles as the
  liveness wait — one syscall per wakeup and **no threads**. A watchdog thread
  (as `run_target_stdin`/`run_target_file` use) would have been the obvious
  match, but this path exists specifically to create no threads, and this
  process forks elsewhere; see E3 and the note in `tests/conftest.py`. The
  child is also spawned into its own process group (`setpgroup=0`, the
  posix_spawn equivalent of the siblings' `preexec_fn=os.setsid`) so a timeout
  kill reaches grandchildren.

  No throughput cost: 980.3 eps vs 989.2 eps median over 5 interleaved repeats
  of 400 execs, ranges fully overlapping. `timeout=None` keeps the old
  unbounded behaviour for callers that pass nothing; `runner.py` forwards
  `f.timeout`, pinned by a wiring test.

  `_TRACKED_PIDS` and friends moved out of the "Stdin mode" section, since all
  three modes now track their children.

- **cmplog left its shim on the process-global `LD_PRELOAD` forever.** Bug
  report 2026-08-21 E2. `setup_env_for_run()` is called before *every*
  execution and set `_CMPLOG_OUT` and `LD_PRELOAD` with no way to undo them.
  The shim conflicts with the ASAN runtime, so any later subprocess exec ran at
  full speed and found **zero crashes** — a quiet wrong answer, not an error.
  `restore_env()` reverts both from a snapshot taken before the *first*
  mutation; re-snapshotting per call would capture our own preload and make
  restore a no-op, which is the original defect wearing the fix's clothes.
  Wired into `stop()` (previously dead code) and the end of `Fuzzer.run()`.

- **`_clean_env({})` returned the full parent environment.** Bug report
  2026-08-21 E2. `dict(env or os.environ)` is falsy for an empty dict, so a
  caller asking for a scrubbed environment silently got the opposite. `None`
  now means "inherit", any dict means "use exactly this".

- **A bare `pytest` could hang forever.** Bug report 2026-08-21 E1. 300s
  default per test, applied in `pytest_configure` rather than `addopts` so a
  dev env without `pytest-timeout` still runs. The method is `signal`, not
  `thread`: the thread method arms a `threading.Timer` per test and makes the
  pytest process multi-threaded for the entire session, which makes every fork
  in the suite riskier (E3, and `docs/handover/test_shm_hang_2026-08-14.md`).
  Measured — under `thread`, CPython emits its multi-threaded-fork
  DeprecationWarning on a test that is otherwise silent. The Z3 modules, which
  block inside native code where SIGALRM is never delivered, opt into `thread`
  individually via `pytest_collection_modifyitems`.

  Also added: an autouse `_env_isolation` fixture restoring the five
  `os.environ` keys production code mutates, with `--env-leak-strict` to fail
  and name the leaking test rather than quietly repairing it. This is what
  fixes the two ASAN tests that passed in isolation and failed in a full run.

### Changed
- **Coverage-guided mode is the default; `--no-coverage` opts out.** `fuzz`
  required `-c/--coverage`, and forgetting it failed silently: no SHM bitmap
  was created, so seed scheduling, MI/TE/sensitivity position weighting,
  Elo/bandit operator scheduling, stall detection and corpus admission all ran
  on a constant-zero signal while the run reported healthy throughput.
  Measured, 1500 execs x 2 repeats:

  | target | coverage on | coverage off | cost | corpus on -> off |
  |---|---|---|---|---|
  | `targets/test_target` | 158.4 eps | 160.7 eps | 1.4% | 3.0 -> 1.0 |
  | `targets/png_read` | 128.1 eps | 138.9 eps | 7.8% | 14.5 -> 1.0 |
  | uninstrumented gcc build | 587.0 eps | 592.4 eps | 0.9% | 1.0 -> 1.0 |

  The corpus column is the decision: without coverage it never grows past its
  seeds, on any target, so the default mode of a coverage-guided fuzzer was
  blind mutation. `-c`/`--coverage` are still accepted and are now no-ops, so
  every existing script, README line and doc example keeps working.
  `--no-coverage` is a real mode, not a compatibility shim — crash and timeout
  detection do not need the bitmap.

  Scoped to `fuzz`. `tmin`, `rc` and `minimize` keep `-c` opt-in because it
  means something different there: in `tmin`/`rc` it only sets `AFL_MAP_SIZE`
  in the child env, and in `minimize` it selects a different algorithm — so
  flipping those would silently change what the command *does*, not how fast
  it runs. The four `use_coverage: bool = False` service signatures are also
  unchanged; the CLI is the layer that expresses this policy.

  Rejected on measurement: auto-selecting ptrace for uninstrumented targets.
  73.9 eps against 587 for the SHM path on the same binary — an 8x silent
  slowdown to buy 2 function-entry edges. It stays behind explicit `--no-shm`.

### Added
- **Vendored SQLite fuzz target (`targets/sqlite_read.c`).** Wraps the SQLite
  amalgamation (`tools/vendor_sqlite.sh` → `vendor/sqlite/`) as a `.so` target
  built like lz4_read and secp256k1_read: `sqlite3.c` compiled as its own TU
  without the shim, linked into the wrapper, `$SQLITE_DEFINES` shared by both
  sides so header and library cannot disagree about `SQLITE_*` options.

  The input carries **no mode-selector byte**, unlike `lz4_read.c`. The
  `sqlite_chunk_mutate` sniffer is `len(d) >= 100 and d[:16] == b"SQLite
  format 3\x00"`, so a prefix byte would shift the magic to offset 1, stop the
  sniffer firing, and flat-byte-mutate every database in the corpus while the
  campaign kept reporting edges — a total loss of structure-awareness with no
  symptom. The dispatch uses those same two conditions: magic → database image
  via `sqlite3_deserialize()`, anything else → SQL text.
  `tests/test_regression_sqlite_target.py` pins the agreement.

  DB path: `PRAGMA integrity_check(4)`, `sqlite_master` read, then up to 24
  table scans reading every column value (without a column read the b-tree
  walk stops at the cell boundary and the record decoder never runs).
  In-process safety, since `direct_lite` shares the fuzzer's process:
  `:memory:` only, `DEFENSIVE` + `TRUSTED_SCHEMA=0`, extension loading off,
  authorizer denying ATTACH/DETACH/PRAGMA on the SQL path, progress-handler
  opcode budget, and `sqlite3_hard_heap_limit64`. Verified over 3,500 execs of
  corrupt, truncated and random inputs: no crashes, RSS flat.
- **Startup warns when the target has no edge instrumentation.** With coverage
  on by default the common failure is no longer "forgot `-c`" but "target was
  never built instrumented", and both produce the identical symptom. `run()`
  previously printed `AFL instrumentation: detected` with no negative branch,
  so a bare target ran with `[*] Coverage: AFL SHM bitmap` on screen and found
  nothing. `Fuzzer._warn_uninstrumented()` names the target, the consequence
  and both ways out (rebuild, or `--no-coverage` on purpose); it fires once per
  run, per instance, and never when coverage was explicitly disabled. Also
  wired into the multi-target banner. This generalizes
  `_warn_no_coverage()`, which covered only in-process `.so` targets.
- **`afl_instrumentation_status()` replaces the boolean `_detect_afl` for any
  decision that warns.** It returns `present` / `absent` / `unknown`, because
  `nm` reports no symbols at all for a stripped binary and a boolean cannot
  tell "not instrumented" from "symbol table removed" — a stripped,
  instrumented target is normal, and warning on it is the fastest way to train
  the warning out of people. `_detect_afl` remains as the boolean face for the
  three call sites that only decide whether to print `[AFL]`. Rule verified
  against every target shape in `targets/`: instrumented ELF and `.so` ->
  present, plain gcc build -> absent, stripped (either kind) and missing file
  -> unknown.

### Known issues
- **ptrace coverage reports no ASAN crashes.** Measured on
  `targets/asan_target.c` (gcc `-fsanitize=address`), same seed and settings:
  0 crashes at both n=100 and n=400 under `--no-shm`, against 32 for the SHM
  path and 25 for `--no-coverage`. Pre-existing and unrelated to this change in
  cause — but it was invisible until now, because `_setup_ptrace` is gated on
  `use_coverage`: the `ptrace` case of `test_asan_all_modes` passes no `-c`, so
  `--no-shm` was inert and that parametrization silently duplicated
  `default_subprocess`. Making coverage the default turned the case honest and
  it failed immediately. Marked `xfail(strict=True)` so it reports loudly when
  ptrace crash reporting is fixed; the fix belongs with the ptrace runner, not
  with a CLI default.

### Removed
- **`_add_common_args()` in `cli/commands.py`, which had zero production call
  sites.** AST-checked: only `tests/test_commands.py` called it. It is named
  "arguments shared by fuzz and subcommands" and declared its own
  `-c/--coverage`, making it the obvious place to edit when flipping this
  default — an edit that would have changed nothing reachable by a user while
  turning `test_commands.py` red, a wrong signal twice over. The live flags are
  declared per-subparser. `TestGetDirs` now builds its namespace directly,
  which is what it was actually testing.

- **XOR-checksum recovery solves by GF(2) elimination instead of SAT; the 32-bit
  rung is live.** `xor_map_solver` ran one Z3 `Solver` per output bit, and at a
  32-bit field the per-bit solve blew past `_SOLVER_TIMEOUT_MS` — recorded in
  `docs/edge-coverage-analysis.md` as a deliberate miss on cost, with a real
  32-bit XOR checksum deferred to an offline pass. It was never a cost problem.
  Every constraint the module builds is `XOR_{i in S} w_i == c`, one linear
  equation over `F2`, and the coefficient matrix is the same for every output
  bit — only the right-hand side changes with `j`. One incremental Gauss-Jordan
  elimination over the augmented matrix, rows as Python ints with all output
  bits packed into the RHS word, recovers the whole map in `O(pairs * rank)`.
  Measured on one box, 32x32 over 64 pairs: **151 s** for the SAT path with the
  timeout lifted (4.7 s/bit), **0.5 ms** for elimination; at the default budget
  the old path returned `(None, False)` and recovered nothing. Three
  consequences beyond speed: the full 8/16/32 ladder is now reached on every
  call; recovery no longer needs the optional `smt` extra, so it works on any
  install; and a rejection is now a consistency proof rather than an expired
  timeout — `solve()` has no inconclusive third outcome. `IncrementalXorMapSolver`
  keeps its public API. `timeout_ms` is still accepted and stored but no longer
  gates anything, and `_SOLVER_TIMEOUT_MS` survives only for callers reading it.

### Added
- **`recover_xor_model` requires a full-rank system before accepting a model**
  (`require_determined=True`, new `IncrementalXorMapSolver.rank` /
  `.is_determined`). Reaching the 32-bit rung exposed a defect that cost had been
  hiding: an underdetermined system reproduces **every pair it was fitted on**,
  for any assignment of the free variables, so `verify_xor_model` — which checks
  against those same pairs — cannot reject it. Measured over 100 trials per cell:
  at 24 pairs, a 32-bit fit to CRC-32 or Adler-32 data was accepted **100/100**
  times, with 0.4% accuracy on held-out inputs. That is worse than recovering
  nothing, because every input the fuzzer then "repairs" carries a wrong checksum
  while the operator looks healthy. Requiring full rank makes acceptance mean "the
  observations admit exactly one linear map". Over 2100 trials across 8/16/32
  bits: **0 wrong models**, 100% held-out accuracy on every accepted model, and no
  true positives lost — the gate converts insufficient-evidence cases from *wrong*
  to *abstain*. Pass `require_determined=False` for the old behaviour.

  Not verified: neither the 32-bit recovery nor the gate has been A/B'd for edge
  coverage on a real target. The mechanism is tested; the coverage claim is not.

- **`estimate_map_size()` now reports which tier produced its answer.** Every
  call logs the block count, the source (`sancov_guards`, `sancov_cntrs`,
  `profile`, `branch_density`, `default`), whether that source is exact or an
  estimate, and whether the cap bound. `estimate_map_size_detail()` returns the
  same as a `MapSizeEstimate` for callers and tests that need to assert the
  tier rather than the number. A silent fallback from "exact" to "estimated" is
  what hid the `__sancov_cntrs`/`__sancov_guards` mismatch below for the entire
  life of the function: tier 3 always returns a plausible number, so nothing
  downstream could tell a measurement from a guess.
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
- **An unguarded `execve`/`execv` in the fork+exec launch paths (`PersistentRunner.start`,
  `TargetRunner`'s ptrace launch) let a failed exec turn the child into an
  orphaned duplicate of the entire parent process.** `fork()` duplicates the
  whole process, including the interpreter's call stack; the child continued
  straight into `execve`/`execv` with no `try`/`except` around it. When exec
  failed (missing/non-executable target), the raised exception unwound back
  through that *inherited* stack instead of hitting the `os._exit(127)` meant
  to catch it — which sat unreachable one statement later, after the
  exec call, not around it. In a pytest run this meant the child kept
  executing as a second, orphaned copy of the whole test session: it
  re-entered pytest's own test loop and re-ran every remaining test
  concurrently with the real parent, producing duplicated test output and,
  at full-suite scale, hangs from the two copies contending for the same
  shm ids and temp files. Both launch paths now wrap the setup/exec sequence
  in the forked child in `try: ... except BaseException: os._exit(127)`,
  matching the existing convention in `ptrace_available()`. Regression test:
  `tests/test_regression_persistent_execve_failure_exits.py` runs the
  offending test in an isolated pytest subprocess and asserts it reports
  exactly once and completes promptly.

- **Every un-stopped `ForkserverRunner` leaked a process, a thread and a SHM
  segment.** The stderr-drain thread was started as `target=self._drain_stderr`
  — a bound method, so the thread held the runner. That reference is
  self-sustaining: the thread blocks reading the child's stderr until the
  child exits; the child exits when `stop()` sends QUIT; `stop()` runs from
  `__del__`; and `__del__` cannot run while the thread keeps the runner
  reachable. Nothing broke the loop. Measured: two test files left **98 live
  runners with 98 live children**, and `gc.collect()` freed none of them —
  they were reachable, not garbage. A full-suite run peaked at ~185 orphaned
  `fuzz_loader`/target processes, each pinning a SHM segment in `dest` state.
  The thread now targets a module-level `_drain_stream(stream, sink)` and
  holds no reference to the runner; after the fix the same suite run peaks at
  0 orphans and 1 segment. This is a production defect, not only a test
  artifact: any `Fuzzer` not explicitly stopped leaked the same three
  resources.
- **A failed SHM attach was silent, on both sides.** All three early returns
  in `__afl_map_shm()` (`__AFL_SHM_ID` absent, unparseable, or `shmat()`
  failing) left `__afl_area` NULL, after which the target ran its full input
  and exited 0 having recorded nothing, and the fuzzer read back an all-zero
  header. Success and total failure were indistinguishable. This is the
  "Loose thread" in `docs/edge-coverage-analysis.md`, unresolved across four
  sightings for precisely that reason; the instrumented assertion caught the
  fourth (`rc=0 edge_count=0 diag=0x00000000 dropped=0 occupied=0 stderr=b''`),
  and `diag == 0` identifies it as a child that never attached rather than a
  parent that raced the read. A target that was asked for coverage and could
  not attach now writes one line to stderr naming which return fired, with
  `errno`. Running standalone (no `__AFL_SHM_ID`) stays silent. The wording
  contains none of the tokens `ExecutionRunner.is_crash()` scans stderr for,
  so a diagnostic cannot be misread as a crashing input.
- **Map sizing read the wrong sancov section, so it never once used a real
  block count.** `estimate_map_size()` lists "sancov guard count (exact)" as
  its first priority, but the only parser it had — `parse_sancov_offsets()` —
  matches `__start/__stop___sancov_cntrs`, the *inline-8bit-counters* section.
  Every target in this tree is built with `-fsanitize-coverage=trace-pc-guard`,
  which emits `__sancov_guards` instead, so priority 1 matched nothing and every
  target fell through to branch-density estimation. That estimate ran 4–16x
  high: `test_target` (91 guards) asked for 131072 entries instead of 8192, a
  1 MiB table memset before every execution to hold 4 edges. Added
  `parse_sancov_guard_count()` and wired it ahead of the counters path. Reset
  cost per exec on the affected targets: 40.8 µs → 4.1 µs, which is noise under
  fork+exec (0.997x on `test_target`) and ~12% of a 305 µs forkserver exec. The
  same over-estimate also fed `recommended_map_size()`.
- `estimate_map_size()` divided the `__sancov_cntrs` length by 4 "because guards
  are uint32_t". That section is 8-bit counters, one byte per block, so the
  fallback path under-sized by 4x on any externally built target that did carry
  it. No target here has the section, so nothing in-tree was affected.
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
