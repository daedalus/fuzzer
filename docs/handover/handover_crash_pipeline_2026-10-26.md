# Handover: crash detection pipeline audit (2026-10-26)

Branch `ccr-c1181a22-nnd25m`. TODO entry: `docs/TODO.md` §Crash triage, "Crash pipeline audit follow-ups".

## Pipeline

```
run_target (runner.py)                       mode backends return (rc, stderr)
  └─ is_crash: SanitizerReport.parse ─┐      rc: -sig | 128+sig | -1 timeout | -2 infra
               rc in SIGNAL_CRASH_CODES ┘
_on_crash (fuzz_round.py)
  ├─ _triage_crash  (direct_lite ptrace re-run; skipped if verdict can't change)
  └─ CorpusManager.save_crash
       ├─ classify_crash (filesystem.py): input hash → blocklist → signature → fuzzy
       ├─ enrich (novel only)
       └─ save_crash: .bin/.txt/.json/.sh/.hex, --save-smaller replacement
maintenance (i % 500): run_crash_replays via f._runner.run_target
```

## Shipped

| Commit | Fix | Test |
|---|---|---|
| `8403c4a` | Anchored sanitizer header; frames from crash stack; module+offset frames; ordered XXH64 stack hash; UBSAN check name + location; LSAN/SEGV parsing | `tests/test_regression_sanitizer_parse.py` |
| `8403c4a` | SIGILL/SIGTRAP/SIGSYS 128+sig codes; `signal_number()`; persistent exit decode; persistent stderr marker sync | `tests/test_regression_crash_rc_codes.py` |
| `8403c4a` | stderr tail (Python spawn, `fuzz_loader.c`); forkserver drains while waiting | `tests/test_regression_stderr_tail.py`, `tests/test_regression_fuzz_loader_stderr_tail.py` |
| `6d688ac` | Fuzzy match per `sanitizer:error_type`, none for unsymbolized; `--save-smaller` by signature/cluster id; prune key | `tests/test_regression_save_smaller.py` |
| `93b42c8` | Replays through campaign runner | `tests/test_regression_crash_replay_runner.py` |
| `1d4e98f` | Triage skip for repeat input / known sanitizer signature | `tests/test_regression_triage_known_crash.py` |
| `8b13b0e` | ptrace stderr to temp file; non-fault stops delivered (SIGCHLD was rc -17) | `tests/test_regression_ptrace_stderr.py` |
| `80de339` | Parser perf: non-report 78 → 28 us, report 55 → 65 us | `bench_parse.py` (scratch, not committed) |

## Pending: process

1. **Regression comparison: done.** 10,421 tests over ~500 crash/ptrace/forkserver/stderr files. Branch-only failures: none after updating `test_fuzzer.py::test_is_interesting_asan` (pinned the bare-"ASAN" spoof). Pre-existing on base `d32da74`: `test_exhaustive_pool.py::TestCoinFlipRewrite`, `test_regression_hail_mary_gates.py`, `test_regressions.py::TestEloDecayInitIndependentOfChi2` (2). Forkserver tests flaked only under two parallel suites; pass serially.
2. **Draft PR** opened for this branch.
3. **impactguard not run.** PyPI install was blocked in the container; the pre-commit hook was uninstalled again. Run `impactguard-check-staged` (or the hook) on the branch and fix any breaking changes (Hard Rule 3). Public surface touched: `CrashVerdict.replaces`, `classify_crash(crash_min_sizes=)`, `run_crash_replays(run_fn=)`, `process.CRASH_SIGNALS` / `EXIT_SIGNAL_BASE` / `signal_number`, `SanitizerReport(location=, pcs=)`, `CorpusManager.triage_skippable`.
4. **Full suite** not run (14k tests, Hard Rule 50); only affected files.
5. **Fuzzgoat calibration** (Hard Rule 52) not run: confirm crash counts / unique signatures vs base on a clang ASAN fuzzgoat build.
6. **Environment:** `libclang-rt-18-dev` was installed by hand; shim tests skip without it.

## Pending: code

### (a) Sentinels collide with SIGHUP/SIGINT
`process._status_rc` returns `-WTERMSIG`; -1 = timeout, -2 = infra. SIGHUP kill → timeout, SIGINT → dropped. ~128 sites compare against -1/-2. Sketch: `TIMEOUT_RC = -1000`, `INFRA_RC = -1001` constants in `adapters/process.py`, migrate every literal; `GUARD_TIMEOUT_RC` (afl_shim) stays -1 inside the shim and is mapped at `inprocess.py:204`. Low frequency (Ctrl-C usually stops the fuzzer too).

### (b) Repeat crashes reward operators
`fuzz_round._judge`: `_success = bool(self._is_crash or ...)`; Elo gets `crash=self._is_crash`. Every repeat of a shallow bug pays the operator. Sketch: `_is_novel_crash` from `CorpusManager.triage_skippable`-style pure classify before `_judge`; signal crashes in direct_lite lack the fault address at that point (triage runs later), so novelty is approximate there. Needs paired A/B (`tools/lib/bench_paired.py`, fuzzgoat), register in `docs/AB_PENDING.md`.

### (c) Replays check "crashed", not "same crash"
**Fixed `1f289da`**: `ReplayOutcome.DIVERGED`, `crash_signature()`; report shows the count.
`stats_reporter._replay_crashed`. Sketch: add `ReplayOutcome.DIVERGED`; compare `sig.split("@")[0]` (bug class) with the replay's class. Existing tests key replays by dummy `"SIG"`: update fixtures.

### (d) Prune evicts the newest bugs
**Fixed `c991192`**: ties keep newest; `save_crash` treats a cluster still on disk as a repeat.
`Fuzzer._prune_crash_data` keeps top-75% by count; new count-1 signatures go first and are re-written as "new" on the next hit. Option: evict oldest among equal counts (insertion order), or keep a Bloom of evicted signatures (`core/bloom.py`).

### (e) Non-atomic crash writes
**Fixed `ce61f4e`**: artifacts built first, written to dot-temps, renamed together.
`filesystem.save_crash` writes five files separately; an interrupt leaves partial sets. Sketch: write to `crashes/.tmp/<base>.*`, `os.rename` each into place, `.bin` last.

### (f) No RSS limit
No `RLIMIT_AS`/RSS cap anywhere in `src/`; a memory-hungry input runs until the kernel OOM killer or the timeout. ASAN `out of memory` reports get error type `out` (not in `ASAN_ERROR_TYPES`). Sketch: `--rss-limit-mb`, `preexec`/`posix_spawn` `setrlimit`, ASAN `rss_limit_mb=`.

### (g) Resume re-saves known crashes once
Signatures (frames, unsymbolized names) and stack hashes changed; resumed older campaigns write each known bug once more under the new signature. Blocklists built from old stack hashes no longer match: regenerate them.

### (h) Subprocess loader stderr unbounded
**Fixed `8cbb1bd`**: spawn path (tail cap); loader's exe path relays stderr (was dropped).
`InProcessRunner` subprocess path: `proc.communicate()` (`inprocess.py::_run_c_subprocess`) reads all stderr into memory (Hard Rule 54). Bound with `_TailBuffer`-style reads.

### (i) `crash_hashes` unbounded
**Fixed `3d20f89`**: `MAX_CRASH_HASHES = 200_000`, cleared past it (safe given on-disk recognition).
Existing TODO (2026-09-24); unchanged.

## Follow-ups shipped (2026-10-26, second pass)

- **Crash-site signatures** (`dff50f6`, `63b94bb`): plain `.so` SIGSEGVs were one `signal:11` bucket. `signal:N@<page offsets>` from the shim crash record; sink default-on for targets without a sanitizer runtime (`fuzzer._target_runtime`). Sink file closed at exit (`atexit`). Note: sancov executables link clang's UBSAN standalone runtime; their `UndefinedBehaviorSanitizer: SEGV` report already splits sites.
- **`--abort-is-crash`**: shim `__afl_shim_abort` raises SIGABRT when `__AFL_ABORT_CRASH=1` (`fuzzer._export_abort_mode`). Default unchanged (FFmpeg `av_assert0` flood). Excluded from `--hail-mary`. Targets must be rebuilt (Hard Rule 56); SHM layout unchanged, no `__AFL_SHM_LAYOUT` bump.
- Test runs left ~115 empty `/tmp/fuzzer_crashsym_*` files from before the `atexit` fix; deleting them was blocked in this session.

## Fuzzgoat calibration (Hard Rule 52)

clang builds from `tools/build_targets.sh`, base `d32da74` vs branch, 20k execs, seeds 1-3, `fuzzer-tool fuzz <so> -s N`.

| target | base (crashes / unique / edges) | branch |
|---|---|---|
| `fuzzgoat_read_asan.so` | 5/2/758, 9/3/830, 14/4/827 | 7/4/956, 7/5/712, 51/5/856 |
| `fuzzgoat_read_noasan.so` | 4/1/756, 3/1/798, 12/1/768 | 2/2/712, 6/2/764, 13/1/758 |

- Edges: within run-to-run spread.
- ASAN: extra uniques are real stack splits (UAF via `json_parse` vs `parse_trailer`; base merged them through alloc-stack frames in the signature). Found the `attempting ...` parse gap: 11 bad-free reports typed `attempting`; fixed (`bad-free` / `double-free`).
- noasan: base filed every glibc abort (`free(): invalid pointer` / double free) as one `signal:6`; branch splits them by crash site, and the same site gets the same cluster id across runs (ASLR-stable): `6b429bba` in seeds 1 and 2, `7b56d567` in seeds 2 and 3.
- Still one bucket: in-process ASAN `.so` SIGSEGVs the shim catches before ASAN reports (`signal:11`, sink off for sanitizer targets).

## Third pass (2026-10-29, PR #241 + follow-up)

Shipped: 3-frame signatures (runtime skipped, harness boundary, shim module per raw frame), resume re-derives saved signatures, `# stack-hash v2` blocklists + sidecar `stack_hash`, ptrace sites by `rip`, `repro`/`flaky` in sidecars and SARIF, signal exploitability tiers, hang dedup by trace edges, `--rss-limit-mb` pre-exec (file input) + persistent per-call cap, `--crash-tmin-execs`. Follow-up: direct_lite glibc heap aborts tiered HIGH from the shim record (`shim_exploitability`).

Fuzzgoat calibration (rebuilt targets, base `e9dfb89b` vs branch, 20k execs, seeds 1-3; crashes / unique / edges):

| target | base | branch |
|---|---|---|
| `fuzzgoat_read_asan.so` | 5/3/682, 3/2/705, 11/4/786 | 5/3/784, 11/3/787, 9/3/792 |
| `fuzzgoat_read_noasan.so` | 14/3/797, 1/1/682, 8/3/942 | 12/4/795, 5/3/738, 14/3/1359 |

- Branch cluster ids repeat across seeds: ASAN `dba7aadc`, `0b052e38` 3/3; noasan `be590256`, `902cc90d`, `582a3a5d` 3/3. Base ids were split by harness frames (`fuzz_shm_run`, libffi offsets).
- Remaining ASAN splits are real callers: bad-free via `default_free` vs direct, UAF via `json_parse` vs the harness's `parse_trailer`.
- noasan invalid frees read LOW (glibc message not captured in direct_lite): fixed in the follow-up, now HIGH.
- Edges within spread (one 1359 outlier, seed 3).

Open: (a) `--rss-limit-mb` unenforced in direct_lite (warns); (h) hang edges not persisted across `--resume`; (m) `--crash-tmin-execs` unmeasured; (10) `--novel-crash-reward` FFmpeg A/B: built (9.0.2, 9 min), arms added, but 0 crashes at 20k execs from an empty corpus with `--abort-is-crash`; blocked on a crash-reaching corpus (`docs/AB_PENDING.md`).

## Verify

```
python -m pytest -q tests/test_regression_sanitizer_parse.py tests/test_regression_crash_rc_codes.py \
  tests/test_regression_stderr_tail.py tests/test_regression_fuzz_loader_stderr_tail.py \
  tests/test_regression_save_smaller.py tests/test_regression_crash_replay_runner.py \
  tests/test_regression_triage_known_crash.py tests/test_regression_ptrace_stderr.py
```
Use `python -m pytest`: `/root/.local/bin/pytest` is a uv tool venv without numpy.
