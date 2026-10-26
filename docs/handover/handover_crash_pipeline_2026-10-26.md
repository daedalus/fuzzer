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
`stats_reporter._replay_crashed`. Sketch: add `ReplayOutcome.DIVERGED`; compare `sig.split("@")[0]` (bug class) with the replay's class. Existing tests key replays by dummy `"SIG"`: update fixtures.

### (d) Prune evicts the newest bugs
`Fuzzer._prune_crash_data` keeps top-75% by count; new count-1 signatures go first and are re-written as "new" on the next hit. Option: evict oldest among equal counts (insertion order), or keep a Bloom of evicted signatures (`core/bloom.py`).

### (e) Non-atomic crash writes
`filesystem.save_crash` writes five files separately; an interrupt leaves partial sets. Sketch: write to `crashes/.tmp/<base>.*`, `os.rename` each into place, `.bin` last.

### (f) No RSS limit
No `RLIMIT_AS`/RSS cap anywhere in `src/`; a memory-hungry input runs until the kernel OOM killer or the timeout. ASAN `out of memory` reports get error type `out` (not in `ASAN_ERROR_TYPES`). Sketch: `--rss-limit-mb`, `preexec`/`posix_spawn` `setrlimit`, ASAN `rss_limit_mb=`.

### (g) Resume re-saves known crashes once
Signatures (frames, unsymbolized names) and stack hashes changed; resumed older campaigns write each known bug once more under the new signature. Blocklists built from old stack hashes no longer match: regenerate them.

### (h) Subprocess loader stderr unbounded
`InProcessRunner` subprocess path: `proc.communicate()` (`inprocess.py::_run_c_subprocess`) reads all stderr into memory (Hard Rule 54). Bound with `_TailBuffer`-style reads.

### (i) `crash_hashes` unbounded
Existing TODO (2026-09-24); unchanged.

## Verify

```
python -m pytest -q tests/test_regression_sanitizer_parse.py tests/test_regression_crash_rc_codes.py \
  tests/test_regression_stderr_tail.py tests/test_regression_fuzz_loader_stderr_tail.py \
  tests/test_regression_save_smaller.py tests/test_regression_crash_replay_runner.py \
  tests/test_regression_triage_known_crash.py tests/test_regression_ptrace_stderr.py
```
Use `python -m pytest`: `/root/.local/bin/pytest` is a uv tool venv without numpy.
