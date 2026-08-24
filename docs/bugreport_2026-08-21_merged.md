# Merged bug report — fuzzer-tool (2026-08-21)

Consolidation of two independent read-only audits of the whole tree
(`src/fuzzer_tool/`, ~71k lines Python + ~2.5k lines C) plus dynamic testing.
Findings found by both audits are marked **[corroborated]**; findings confirmed
by execution or line-trace during the merge are marked **[verified]**.

Method: audit 1 — five parallel subsystem reviews (main loop, mutation engine,
adapters/coverage, scheduler math, parsing/services) guided by
`docs/refs/bug-classes.md`, cross-checked against callers. Audit 2 — six parallel
reviews (main loop, mutations, adapters + C shims, coverage/scheduling math,
binary-analysis algorithms, CLI/services). Merge pass — every CRITICAL and all
disputed/unique high-impact claims re-verified line-by-line before inclusion;
dynamic evidence from full-suite pytest runs, an env-mutation probe plugin, and
py-spy stack dumps of a hung run.

Cleared by both audits (checked, no bug): Elo update algebra, Kalman/Welford
algebra, NIST test battery, PNG/ZIP/GZIP round-trips, splice bounds, tmin loop
termination, fast_json (thin orjson wrapper), checksum_learner, registry↔handler
completeness (134/134), builtin `hash()` in persisted keys.

---

## TEST-SUITE EVIDENCE (dynamic)

**E1. Plain `pytest` can hang forever — no timeout configured.** **FIXED 2026-08-22.**
`tests/conftest.py` sets a 300s default via `pytest_configure`, and
`pytest-timeout` is now a declared `dev` dependency. The method is `signal`, not
`thread`: the thread method arms a `threading.Timer` per test, which makes the
pytest process multi-threaded for the whole session and so makes every fork in
E3 riskier. The Z3 modules — the ones that actually block in native code, where
SIGALRM never lands — opt into `thread` individually via
`pytest_collection_modifyitems`. Set programmatically rather than in `addopts`
so a dev env without the plugin still runs.
`tests/test_structural_constraints.py:295` wedged >9 min inside `solver.add()`
(`core/structural_constraints.py:186`); Z3's `timeout` parameter bounds only
`check()`, not assertion processing. SIGTERM could not kill it (stuck in native
code); SIGKILL required. `pyproject.toml` has no pytest-timeout setting.
**[verified]**

**E2. Production code leaks `os.environ`; three tests fail as a result.**
**FIXED 2026-08-22**, both halves. `CmplogCollector.restore_env()` undoes
`setup_env_for_run()`'s `_CMPLOG_OUT`/`LD_PRELOAD` mutations from a snapshot
taken before the *first* mutation (re-snapshotting would capture our own
preload and make restore a no-op); wired into `stop()` and the end of
`Fuzzer.run()`. `adapters/process.py:50` now distinguishes `None` (inherit)
from `{}` (empty), so a caller asking for a scrubbed env gets one. An autouse
`_env_isolation` fixture in `tests/conftest.py` restores the five leaked keys
after every test, with `--env-leak-strict` to name the leaking test instead of
quietly repairing it. Regression test `tests/test_regression_env_leak.py`.
The `adapters/inprocess.py` / `stats.py` SHM-id writers are unchanged: still
global, now contained at the test boundary rather than at the source.
Probe-attributed leaks: `LD_PRELOAD=<cmplog shim>.so` + `_CMPLOG_OUT`
(written with no restore at `core/cmplog.py:315-321`) and
`__AFL_SHM_ID` / `__AFL_DIST_SHM_ID` / `AFL_MAP_SIZE` (`fuzzer.py:4417`,
`stats.py:501`, `inprocess.py:237`). Consequences, all reproduced:
- `test_asan_finds_heap_buffer_overflow` and
  `test_asan_all_modes[default_subprocess]`: 101 execs at eps 337, **0 crashes**
  (`shm: 0`, `map: 0.0%`) — every exec inherits the cmplog shim preload, which
  conflicts with the ASAN runtime. Passes in isolation. **[verified]**
- `test_process.py::TestCleanEnv::test_no_preload` fails via an aggravating
  production bug: `adapters/process.py:50` uses `dict(env or os.environ)` — an
  explicitly *empty* env dict is falsy, so callers asking for a scrubbed
  environment silently get the full parent env. **[verified]**

**E3. Fork-in-multithreaded-process hazard.** `persistent.py:72`,
`runner.py:311`, inprocess loader call `os.fork()` while threads exist
(CPython DeprecationWarning; deadlock risk). Also breaks
`test_regression_persistent_execve_failure_exits` deterministically: the warning
summary reprints the inner test name, so its `count(...) == 1` assertion sees 2.
**[verified]**

---

## CRITICAL

1. **`adapters/persistent.py:61`** [corroborated, verified] — **FIXED
   2026-08-21.** `libc.shmat()` without `restype=c_void_p`. Default ctypes
   restype truncates the 64-bit address to 32 bits; `memmove` through the
   truncated pointer segfaults or silently corrupts memory on first use of
   `PersistentRunner`. Now routed through `adapters/libc_shm.py`; regression
   test `tests/test_regression_shmat_restype.py`.

2. **`cli/commands.py:297-304` + `services/parallel.py:216`** [verified] — **FIXED
   2026-08-21** (`6f7a866`). `cmd_fuzz` passes `contextual*`/`lineage_backtrack` kwargs that
   `run_parallel()` does not accept. Every `--jobs > 1` CLI run dies with
   `TypeError` before spawning a worker; parallel fuzzing is unreachable from
   the CLI.

3. **`adapters/process.py:227` via `services/runner.py:228`** [corroborated] — **FIXED
   2026-08-22.** All three halves: `run_target_fast` now takes `timeout` (forwarded
   from `f.timeout` by `runner.py`), drains stderr concurrently via `poll()` instead
   of after the reap, spawns into its own process group so the kill reaches
   grandchildren, and returns the real pid on exception after killing and reaping the
   child. Thread-free by design — a watchdog thread would re-open the multi-threaded
   fork hazard (E3). Measured: no throughput cost (980 vs 989 eps, overlapping
   ranges). Regression test `tests/test_regression_fast_path_timeout.py`.
   Original text follows.
   Default spawn-fallback path (`run_target_fast`) enforces **no timeout**
   (`os.waitpid(pid, 0)`), returns `pid=0` on exception (crash attribution lost,
   child leaked), and drains stderr only after reaping (64 KiB pipe deadlock).
   One infinite-looping or chatty target hangs the campaign forever. Every other
   backend honors `f.timeout`.

4. **`services/minimize.py:27,140`** [corroborated, verified] — **FIXED
   2026-08-21.** Second/third missing-restype `shmat`: truncated pointer fed to
   `string_at` → segfault or garbage bitmaps driving greedy set-cover cmin.
   Reference-correct site: `adapters/shm.py:88`. Both sites now use
   `adapters/libc_shm.py`. Note #5 is independent and still open, so
   `minimize -c` against an uninstrumented target still prunes the corpus.

5. **`services/minimize.py:126-146`** [verified] — **FIXED 2026-08-21** (`c6fa0ce`): minimize now discovers sharded corpora and refuses to prune on an all-zero bitmap. Original text follows. — SHM failure or uninstrumented
   target yields all-zero bitmaps; zero-bitmap files contribute no edges, so set
   cover moves them to `pruned/`. Against an uninstrumented target the **entire
   corpus** is pruned. Combined with #4, `minimize -c` is broken end-to-end on
   x86-64.

## HIGH

6. **`services/runner.py:395-464`** [corroborated, verified] — **FIXED
   2026-08-23** (`7ba1054`): deadline expiry is now an explicit `timed_out`
   flag rather than a return code reconstructed from the last consumed
   wait status, and returns the `(-1, "timeout")` pair `is_timeout` tests
   for (`fuzzer.py:3298`). The blind-`PTRACE_CONT` wait is a bounded poll
   against the same deadline instead of a blocking `waitpid(pid, 0)`.
   `tests/test_regression_ptrace_timeout.py` (6 tests). Original text
   follows. — ptrace-mode
   timeouts are never reported as timeouts. On deadline expiry `status` holds the
   last consumed event: with ≥1 breakpoint handled, the post-loop
   `waitpid(WNOHANG)` returns `(0,0)`, stale SIGTRAP status yields `rc=-5`
   ("crash signal 5"); with none, the else-branch SIGKILLs and the SIGKILL death
   status yields `rc=-9`. `is_timeout` (`rc==-1`) can never fire; slow inputs
   flood `crashes/` and poison signature dedup. Additionally the post-deadline
   blocking `waitpid(pid, 0)` after blind `PTRACE_CONT` can hang forever and
   swallows fatal signals delivered at that point.

7. **`core/rand_pool.py:146-157`** [verified empirically] — **FIXED 2026-08-21** (`3712812`): `randint` width-256 fast path adds the offset `a` back. Original text follows. — `randint(a,b)`
   silently drops offset `a` when `b-a+1 == 256` (fast path returns
   `self._m256_l[pos]` without `a + ...`). Verified: `randint(-128,127)` → zero
   negative draws in 5000. Live call site: `operators.py:562`.

8. **`core/grammar.py:726-790`** [corroborated, verified] — **FIXED 2026-08-21** (`3712812`): `hierarchical_shrink` returns `best` instead of None. Original text follows. —
   `TreeMutator.hierarchical_shrink` has no `return best`; always returns None →
   `TypeError` in `tmin.py:191`. Grammar-mode crash minimization fails 100%.

9. **`core/schedules.py:227-229`** [verified] — **FIXED 2026-08-21** (`3712812`): COE skip returns floor energy, not `max_mult*100`. Original text follows. — COE power schedule inverted:
   seeds that should be *skipped* (`coe_skip() == True`) receive max energy
   (`max_mult * 100`). Exactly backwards.

10. **`cli/commands.py` env/global state** [verified] — see E2/E3: import-time
    signal handlers + child-killing atexit (`fuzzer.py:159-168`), unrestored
    `os.environ` writes across cmplog/fuzzer/stats/inprocess, fork-in-threaded
    process. Breaks test isolation, library reuse, and reproducibility
    simultaneously.

11. **`services/fuzzer.py:1080-1086 et al.`** [verified] — **FIXED 2026-08-23.**
    Both halves. `StateStore.start_empty()` marks the store loaded-and-empty so
    `get()` cannot lazy-load; `Fuzzer.__init__` calls it on the non-resume
    branch. Gating `load()` on `self.resume` had only DEFERRED the read to the
    first `get()`, not prevented it — the lazy load in `get()` is wanted by the
    standalone readers (`report.py`, `tmin.py`, `cli/commands.py`), so the opt-out
    belongs at the fuzzing call site rather than in `get()`. The legacy-JSON
    migration path was the same hazard by another route and is closed with it.
    The GA restore and its banner moved out of the `if self._diff_target:` block
    into `if self._ga_enabled:`, next to `ga.initialize()`, mirroring the QEA
    block directly below. `tests/test_regression_fresh_run_state.py` (9 tests);
    the GA half is asserted against `run()`'s AST, since which block a statement
    sits in is exactly what was wrong and constructing a Fuzzer needs a built
    target. Original text follows. — persisted component
    state restored on *non-resume* runs: `StateStore.get()` lazy-loads
    `state.pkl.gz` even without `--resume`, so a second "fresh" run inherits the
    previous campaign's Markov model (skips retraining), Elo ratings, crash-MI
    counters — poisoning A/B schedule comparisons. Related: GA restore nested
    inside the differential block (`fuzzer.py:4840-4855`) →
    `--differential-target` without `--ga` crashes at startup;
    `--ga --resume` silently restarts GA.

12. **`services/fuzzer.py:3369` (also :3795, :5061)** — multi-target mode reads
    per-run edges from `self.shm_cov` instead of the per-target segment in
    `_target_shm_covs`; seed edges/momentum/stall detection run on empty data.

13. **`adapters/inprocess.py:443-494`** [corroborated, verified] —
    `_run_c_direct` cannot survive real faults: returning Python SIGSEGV handler
    re-executes the faulting instruction (infinite fault loop); `timed_out` flag
    checked only after the blocking ctypes call returns, so a looping native
    target can never be stopped. First wild-pointer input under
    `--inprocess-direct` freezes the fuzzer permanently.

14. **`adapters/persistent_loader.py:511-529`** [verified] — **FIXED 2026-08-21** (`3712812`): throughput watchdog stops the old process before `start()`. Original text follows. — slowdown watchdog
    "restart" is a no-op: sets `_ready=False` then calls `start()`, which
    early-returns because the old loader is alive-but-slow. All subsequent
    `run_one` return `-2` forever, silently.

15. **`services/parallel.py:199-213`** [corroborated, verified] — **FIXED
    2026-08-23.** Both halves, plus two more found while fixing. `_sync_corpus_in`
    now walks `<worker>/seeds/**/id_*` instead of listing the worker directory
    non-recursively, and the index cursor is replaced by a per-sibling set of
    consumed FILENAMES — seed names are content hashes, so their sort order is
    unrelated to creation order and any insertion before the cursor was skipped
    permanently. Because the filename IS the hash, a seed already held is now
    skipped without being read. Also: `pruned/` is excluded, matching
    `load_corpus` (re-importing a sibling's pruned entries would undo its
    minimization); a worker no longer syncs from its own directory; and a
    name/content hash mismatch is treated as a torn read from a sibling's
    non-atomic write and retried next round rather than imported. Deltas are
    deliberately not transferred — a delta names a parent hash the importing
    worker may not hold. **Every test in `tests/test_parallel.py` and
    `tests/test_parallel_unit.py` wrote seeds flat at the worker-dir top level,
    a layout `save_to_corpus` has never produced**, so 14 passing tests
    certified a transfer path that moved zero seeds; both files were rewritten
    onto the shipping writer. Original text follows. — `-j N` corpus
    sync has two defects: (a) it filters top-level *files*, but seeds live under
    `seeds/<hh>/id_*` — zero seeds ever transfer, and the only top-level file,
    `state.pkl.gz`, is imported as a garbage seed by every sibling worker;
    (b) even once fixed, its index cursor over a `sorted()` listing of
    hash-named files permanently skips any insertion sorting before the cursor.
    Worker corpus sharing has never worked.

16. **`core/transfer_entropy.py:84-105`** [verified empirically] — plug-in TE
    estimator reports ~2.6 bits for *independent* uniform byte streams (n=1500);
    no bias correction for context cardinality. `byte_to_edge_flow` /
    `causal_chains` produce spurious causal edges on essentially any input.

17. **`core/crash_eta.py:66-135`** — `CrashMITracker` records only crashing
    inputs, so `byte_total == joint_crash` always; "MI" degenerates to position
    frequency × log₂(1/p_crash). Crash ETA and mutation targeting driven by noise.

18. **`core/schedulers/monte_carlo.py:195-263`** [corroborated] — **FIXED
    2026-08-23.** `_prev_op` is now advanced by `record()` alone, at the end,
    unconditionally. Two defects in one: the sole assignment sat in `select_op`
    *after* the early return taken whenever the matrix is empty, so it could
    never run; and it assigned the operator being SELECTED, so had it run,
    `record()` for that same operator would have hit the `_prev_op != name`
    guard. Measured on a harness where the reward exists only on a specific
    transition (`b` pays only after `a`), 8 seeds x 3000 steps:
    `pairwise_blend=0.6` scores 83.0 mean vs 64.2 for pure Thompson — before the
    fix the two configurations were bit-identical, since the blend branch was
    unreachable. Note the residual limitation, not a defect: only successful
    transitions are counted and the unconditional arm posterior still punishes
    a setup move that never pays off on its own, so a required predecessor stays
    under-selected. `tests/test_regression_pairwise_bootstrap.py` (6 tests).
    **All nine pairwise tests in `tests/test_montecarlo.py` assign `mc._prev_op`
    by hand before calling `record()`**, pre-loading exactly the internal state
    the production path cannot reach; they are left in place as unit tests of
    `record()`, with the new file driving the public interface only. Original
    text follows. — pairwise
    transition tracking can never bootstrap: `record()` needs `_prev_op`, which
    only `select_op()` sets inside the branch requiring non-empty transitions.
    Blending/stationary/spectral_gap dead on fresh runs.

19. **`adapters/shm.py:313` + `adapters/afl_shim.c:810-815`** — **FIXED
    2026-08-23** (`189e387`): `ShmCoverage.reset_edge_map()` wipes the edge
    table when the tag returns to 0. The wipe already existed in
    `__afl_map_reset()` with the same reasoning, but that function still
    has no callers, so it sat on a dead path while the live reset is the
    Python one; fixing it there also keeps a single writer of the header.
    Reproduced first (edge read as live again at exactly N = 256, 512),
    and `tests/test_regression_generation_wrap.py` (5 tests) asserts the
    aliasing period rather than "no ghosts eventually". Original text
    follows. — generation tag
    wraps at 256 execs; the anti-wrap table wipe exists in C
    (`__afl_map_reset`) but has **zero callers**, so ghost edges from 256 execs
    ago re-enter the live set every wrap.

20. **`adapters/inprocess.py:513-595` + `runner.py:154`** — `direct_lite`
    (hardcoded-on default) never resets SHM between iterations → every exec
    reports the cumulative union of all prior coverage; per-exec attribution and
    stability calibration meaningless. Compounded by `inprocess.py:391-414`
    memsetting entry-*count* as bytes (wipes header + ⅛ of table, forcing
    generation to 0 so stale entries look current).

21. **`services/fuzzer.py:3224`** [verified] — **FIXED 2026-08-21** (`3712812`): `rc == -1` is treated as a timeout regardless of stderr text. Original text follows. — `is_timeout = rc==-1 and
    stderr=="timeout"` matches only some backends: forkserver (the default)
    returns `(-1,"")`, the C loader reports `RC -1 <n>` with target stderr.
    Default-run hangs are never counted; honggfuzz timeout penalty inert; under
    `--metropolis` hung inputs eligible for corpus admission.

22. **`services/fuzzer.py:3751,3756` + `stats_reporter.py:66-85`** [verified] —
    crash-replay keys use `crash_sigs.get(crash_name, crash_name)` (signature
    map fed filenames); fallback `stem.startswith(sig[:12])` matches any crash
    within a ~2.7h window → reproducibility scores computed from the wrong
    crash's input.

23. **`core/elf.py:1159-1837` (4 sites)** [verified] — unguarded
    `struct.unpack_from` on attacker-controlled `e_shoff`/section offsets in
    `branch_density`/`_text_size`/`extract_constants_pure`/
    `extract_div_constants` → malformed target ELF crashes the fuzzer at startup
    (violates repo's own bounds-check rule).

24. **`core/grammar.py:167-221`** [verified] — per-token repeats clamped but not
    their product: chained `{32}` rules expand fully before `max_len`
    truncation → crafted grammar file OOMs/hangs the fuzzer itself.

25. **`stats_reporter.py:30-32` + `edge_tracker.py:943-947` +
    `report.py:1115-1144`** [corroborated] — snapshot trims shrink exec/edge
    arrays but never timestamps → arrays desync, temporal join pairs wrong
    indices, and past snapshot caps an uncaught `IndexError` aborts report
    generation.

26. **`services/tmin.py:41-59` + `corpus_manager.py:173 vs 434`** [verified] —
    lineage walk looks up xxhash-prefix keys in a content-hex-keyed dict → chain
    walk never passes the immediate parent; advertised root-shrink never happens.

27. **`services/corpus_manager.py:507-566`** [corroborated] — trim replaces seeds
    in memory only: original file later pruned from disk while trimmed bytes are
    never written → seed lost entirely on resume; `seen_hashes` still holds the
    old hash so regenerated originals are rejected as dupes.

28. **`mutations/x86.py:146-148,187,253`** [verified empirically] — decoder
    treats accumulator-immediate ALU ops (no ModRM) and far call/jmp (6-byte
    operand, reads 4) as ModRM/4-byte forms → all later instruction boundaries
    corrupt; imm/disp rewrites smash neighboring real instructions.

29. **`mutations/generic.py:1628-1637`** [corroborated, verified empirically] —
    ULEB128 rewrite path doubly broken: width loop always exits at width=1, so
    the "rewrite" degrades to insert-before-original (value duplicated, e.g.
    `\xff\xff` → `\xff\xff\x01\xff`); when wider widths do engage, the slice
    covers `width-1` bytes of a `width`-byte field (stale top byte retained) and
    the max_len guard compares the wrong expression.

30. **`qea.py:267,361` + `monte_carlo.py:778,895`** [verified] — **FIXED
    2026-08-22.** `Fuzzer._seed_global_numpy()` seeds the legacy global
    `np.random` state from `__init__` (next to `random.seed`) and from
    `_reseed_after_stall`, folding wider seeds into `[0, 2**32)`. The
    stall-reseed docstring's claim that `np.random` backs `RandPool` was false
    and is corrected — `RandPool` owns an independent `default_rng`, which is
    precisely why the global went unseeded. Regression test
    `tests/test_regression_numpy_global_seed.py` (10 tests) asserts draw-sequence
    reproducibility, not that a seeding call was made. Original text follows. — global numpy
    RNG never seeded anywhere in `src/` → `--seed` reproducibility broken
    whenever QEA is active; stall-reseed docstring falsely claims `np.random`
    backs RandPool.

31. **`services/differential.py:78`** [verified] — **FIXED 2026-08-22.** The
    branch now sets `diverged = True`, but only when neither side produced a
    valid sanitizer report: when both crashed with the SAME `error_type` the
    branches above have already adjudicated them as matching, and their stderr
    still differs every run by allocation addresses, pids and thread ids —
    flagging on that would report a divergence for every identical crash pair.
    `tests/test_differential.py::test_different_stderr` ASSERTED THE BUG
    (`assert not diverged  # stderr differs but not diverged`, written from
    observed output rather than the documented contract) and is corrected in the
    same commit — same shape as the `test_hex_escape` case in docs/TODO.md.
    Regression test `tests/test_regression_stderr_divergence.py` (11 tests).
    Original text follows. — stderr divergence appends a
    reason but never sets `diverged=True`; documented contract says stderr must
    match. (Supersedes an earlier audit note that cleared this file.)

32. **`services/report.py:763,770`** [verified] — **FIXED 2026-08-22.**
    `_crash_analysis` filters on `CRASH_INPUT_SUFFIX` (`.bin`), the extension
    `save_crash` gives the input; `.txt`/`.sh`/`.hex` are sidecars. This also
    repaired the size histogram and the sample list, which were being fed
    sidecar TEXT as though it were crash input — the count was the visible
    symptom, not the whole defect. Regression tests in
    `tests/test_regression_crash_count_and_rss.py`. Original text follows. — crash counting iterates all
    files in `crashes_dir`; `.txt/.sh/.hex` sidecars + sanitizer JSONs inflate
    "Total crashes" ~4-5×.

33. **`adapters/filesystem.py:608-627`** — a blocklisted crash poisons its coarse
    signature in dedup state; a different later crash sharing only the top-frame
    signature is silently discarded without being saved.

34. **`adapters/afl_shim.c:1394-1444,1592`** — permanent crash handlers installed
    in the fuzzer's own process incl. SIGPIPE (which CPython deliberately
    ignores) and unconditional `siglongjmp` through a possibly-stale
    `sigjmp_buf` → UB on ordinary EPIPE/faults outside guarded calls.

35. **`adapters/fuzz_loader.c:595-611`** — timeout longjmps out of the target
    mid-execution (locks/state possibly held) yet keeps serving RUNs from the
    poisoned process; AFL++ respawns workers here.

36. **`adapters/persistent.py:136-161`** [corroborated] — protocol never sends
    SIGCONT per its own docstring; after the first `run_one` the target stays
    SIGSTOPped, every later iteration times out and SIGKILLs it — persistent
    mode is single-shot.

37. **`adapters/persistent_loader.py:80`** [corroborated] — **STALE ENTRY, NOT A
    BUG as of 2026-08-22.** Verified against the code, not against commit
    messages: the embedded loader source spans lines 55-266 and contains no
    `log` reference at all; all five `log.` call sites (354, 357, 374, 423, 491)
    are in the OUTER module, where `log = logging.getLogger(__name__)` is
    defined at line 28. Nothing to fix. Original text follows. — embedded loader error
    handler references undefined `log` → NameError kills the whole loader on any
    SHM attach failure; parent sees EOF and reports `-2` thereafter.

38. **`adapters/persistent_loader.py:369`** [verified] — stderr drain thread
    targets a bound method → self-sustaining reference cycle; every abandoned
    runner leaks process+thread (exact leak `forkserver.py:204-227` already
    fixed and documents).

## MEDIUM

39. **`fuzzer.py:776,5093`** [verified] — `_last_new_edge_exec` not restored on
    `--resume` → every resume immediately false-triggers stall recovery.

40. **`runner.py:584`** [verified] — **FIXED 2026-08-21** (`3712812`): `is_interesting` excludes rc `-2` as infrastructure, like `-1`. Original text follows. — `is_interesting` treats rc `-2`
    (infrastructure/exec-failure sentinel) as discovery → junk corpus entries
    with phantom coverage credit (`is_crash` correctly excludes it).

41. **`core/kalman.py:370-388`** [verified] — RobustKF computes adaptive `_r_eff`
    but never feeds it into filtering; advertised self-tuning measurement noise
    is inert.

42. **`markov.py:479-485`** [verified] — ensemble never truncates context for
    order-0 chains → trained unigram unreachable; ~22% of picks degenerate to
    most-common byte.

43. **`monte_carlo.py:360-364`** [corroborated] — once `elite_set ≥ 10`, the
    refit-interval gate short-circuits; CEM refits (O(elite×len) rebuild + JS)
    on *every* interesting event; adaptive interval is dead code.

44. **`core/mutations/zlib.py:105-111,210-264`** — `serialize_zlib`'s final
    `% 31` destroys FLEVEL/FDICT bits it claims to preserve (FDICT streams
    round-trip invalid; DICTID dropped then re-emitted as zeros);
    `_mutate_flevel`/`_mutate_window_size` are silent no-ops (serializer never
    reads those attrs) — counted mutations that mutate nothing.

45. **`core/tree_mutator.py:347-357`** — `_swap_nodes` on ancestor/descendant
    pairs creates a cycle and silently drops content (verified: `(a(b)c)` →
    `(b)`, 300/300).

46. **`core/frameshift.py:102-112,272-280`** — discovered relations seeded with
    `val=0`, then unconditionally `apply_to_buffer` zeroes real field contents
    on every mutant after calibration; `on_delete` discards `_rel_on_remove`'s
    invalid flag (unlike `on_insert`), so disabled relations keep patching stale
    offsets; resizing ops never notify frameshift at all.

47. **`edge_tracker.py:1336-1390`** — JS-divergence vs aggregate omits KL(Q‖M)
    terms and Wasserstein walks only seed-edge positions — both diversity
    metrics systematically biased low, contradicting their docstrings and the
    correct sibling implementation.

48. **`berlekamp_massey.py:347,358-362`** — GCD-of-syndromes masked to `width`
    bits: when syndromes share a cofactor (~50% of text-like inputs, measured
    98/200) a meaningless fragment is returned instead of the generator;
    syndrome XOR also applies `init` unshifted (wrong for nonzero CRC preload).

49. **`inprocess.py:128-143`** — legacy loader treats `AFL_MAP_SIZE` (entries)
    as bytes starting at the header → coverage record header-contaminated and
    ~8× short.

50. **`perf_event.py:87,326-334`** — `1 << 11` is `inherit_stat`, not
    `enable_on_exec` (bit 12; masked today by explicit ioctl fallback);
    `reset_counters` issues `ioctl(fd, 0)` instead of `PERF_EVENT_IOC_RESET`
    (0x2403) → EINVAL suppressed, counters never reset in hardware, next delta
    inflated by full accumulation.

51. **`forkserver.py:195-258` + `persistent_loader.py:359-383`** [verified] —
    startup `stdout.readline()` handshake has no timeout (hung dlopen hangs the
    fuzzer); failed INIT returns False without killing/reaping the loader →
    orphaned process holding SHM. `forkserver.py:108` additionally builds
    `fuzz_loader` at a fixed shared path, not PID-namespaced → `-j N`
    ETXTBSY/races (shim_factory does this correctly).

52. **Grandchild leakage on teardown** [corroborated] —
    `persistent.py:182-192` setsid's its child but only ever `kill()`s (never
    `killpg`) so forked targets survive cleanup; `forkserver.py:195-201,441-447`
    spawns without setsid and kills only the loader, orphaning the exec'd hung
    target; `afl_shim.c:1554` + `fuzz_loader.c:161-167` timeout kills only the
    direct child.

53. **`services/report.py:1192-1196`** — exploitability section reads an
    `"exploitability"` JSON key nothing writes (real tier lives in unread .txt
    sidecars) — always UNKNOWN.

54. **`core/sanitizer.py:30-52,269-284`** [verified] — common ASAN layouts
    "attempting double-free"/"attempting free…" classify error_type as
    `"attempting"` → misgraded exploitability buckets and mislabeled signatures.

55. **`core/dwarf.py:422-437,493,595,617`** — one malformed CU (`line_range ==
    0` → ZeroDivisionError) aborts the entire CU loop under a broad except,
    silently disabling DWARF resolution for all later valid CUs. (One audit
    cleared DWARF generally; this specific path was line-verified by the other.)

56. **`seed_picker.py:206,391,840-861,886-926`** [verified] — `randint(4,
    min(64, max_len))` raises ValueError with `--max-len < 4` and empty corpus;
    3-D Pareto sweep excludes genuinely non-dominated seeds (running maxima from
    different items); Pareto sampling uses global `random` (breaks seeded
    reproducibility); front cache keyed on `len(corpus)` goes stale after
    in-place trim.

57. **`import_corpus.py:196-213`** [corroborated] — **FIXED 2026-08-22.**
    `--format` now defaults to None so argparse can distinguish "unspecified"
    from an explicit `--format afl`, and detection moved into
    `_detect_source_format()`, checked most-specific-first (honggfuzz markers,
    then AFL `queue/`/`fuzzer_stats`, then flat = libFuzzer) because an AFL
    output tree also has top-level files and would otherwise look like a
    libFuzzer corpus. Regression tests in
    `tests/test_regression_crash_count_and_rss.py`. Original text follows. — format auto-detect is dead
    code (`args.format == "afl" or …` always true on default); libFuzzer corpora
    import 0 seeds with a success message.

58. **`generic.py:1498` + `grammar.py:204,218`** [verified] — radamsa_num draws
    from module-global `random` despite injected RNG plumbing (~1/10 of draws
    break `-s` reproducibility); same leak in grammar versifier paths.

59. **`fuzzer.py:2667-2680`** [verified] — **FIXED 2026-08-22.** New module-level
    `_current_rss_kb()` reads resident pages from `/proc/self/statm` and converts
    via `SC_PAGE_SIZE`, returning None (check skipped) when /proc is unreadable
    or unparseable. The docstring claiming `getrusage` reports current RSS is
    corrected. Regression tests in `tests/test_regression_crash_count_and_rss.py`
    include a falsification that allocates and frees 200 MiB and asserts the
    reading drops back BELOW `ru_maxrss`. Original text follows. — memory prune keyed off peak RSS
    (`ru_maxrss`, monotonic) labeled as current RSS → pruner arms forever after
    one spike; warning prints stale numbers as current usage.

60. **`fuzzer.py:2318-2329`** [verified] — `_check_differential` discards
    computed results and records hardcoded zeros; drift stats meaningless
    whenever `--differential-target` is used.

61. **`fuzzer.py:4785-5122`** [verified] — **FIXED 2026-08-22.** `run()` now
    also catches `Exception`, logs the traceback via `log.exception` (loud, not
    swallowed — Hard Rule 20) and sets `_aborted_by_error`, so every
    `_state_store.set`, both `_save_state` calls and the ablation fd close below
    the try are reached. The end-of-run summary says "aborted by an unexpected
    error" rather than "stopped" so an aborted run is not mistaken for a clean
    one. `BaseException` deliberately still propagates. Regression test
    `tests/test_regression_end_of_run_persistence.py` (10 tests). Original text
    follows. — main loop catches only
    `(KeyboardInterrupt, SystemExit, OSError)`; any other exception skips all
    end-of-run persistence (`_dump_stats`, every `_state_store.set`, both
    `_save_state`) and leaks the ablation fd — hours of campaign state lost.

62. **`fuzzer.py:2593-2612 vs :3756-3762`** [verified] — `_prune_crash_data`
    evicts `_crash_replays` but not `_crash_sanitizer_replays` (pins full input
    bytes indefinitely).

63. **`fuzzer.py:4347-4363`** [verified] — Allan-noise-adaptive stall thresholds
    unreachable (caller gates on the unreduced threshold first); advertised
    early-warning inert.

64. **`elo.py:685-698`** [verified] — BayesianElo adaptive temperature never
    called; win-rate bookkeeping costs cycles for nothing.

65. **`inprocess.py:544-551`** [verified] — direct_lite installs process-global
    SIGALRM handler once, never restores (stop()/__del__ don't); second runner
    captures the first's handler as its "old" one; later `signal.alarm` users
    silently deliver into the stale handler.

66. **`filesystem.py:463-466`** [verified] — wholesale `seen_hashes.clear()` at
    the 200k cap resets in-memory dedup and per-seed statistics (fuzz_count=0)
    every 200k unique seeds.

67. **`minimize.py:75-77` + `root_cause.py:25-38`** [verified] — **FIXED
    2026-08-23.** Both halves, but they were fixed nine days apart and that is
    the point of this entry. `minimize.py` was corrected as fallout from
    findings 4/5; `root_cause.py` was named in the same line of the same
    finding and sat untouched, so `root-cause --corpus-dir <real corpus>`
    listed the directory non-recursively, found no seeds under
    `seeds/<hh>/id_*`, and fell back to the only top-level regular files a live
    corpus holds. It then printed `state.pkl.gz` as the "nearest corpus seed"
    and diffed the crash against gzip bytes — a complete, confident,
    meaningless root-cause report. Original text follows. — corpus scan
    ignores the standard `seeds/<hh>/` layout; replays/offers `state.pkl.gz` as
    the only "seed".

    The walk now lives once, in `adapters/filesystem.discover_seed_files()`,
    which is where the layout constants already were; `minimize` and
    `root_cause` pass their own exclusions to it. The exclusions genuinely
    differ and must not be collapsed: `minimize` drops `irreplaceable/`
    because those entries are never-prune and so are not prune candidates,
    while `root_cause` keeps them because they are ordinary seeds and make
    perfectly good baselines. Both drop `crashing/`, for different reasons.
    `tests/test_regression_seed_discovery_layout.py` (14 tests) asserts on
    the returned BYTES; four of them fail against the pre-fix source, and a
    test asserting only `len(seeds) > 0` passes against it, which is how this
    survived.

68. **`state_store.py:190-194` + `stats.py:349-406`** — temp-file+rename without
    fsync (power loss can persist empty/truncated `state.pkl.gz`); stats and
    coverage JSON written non-atomically (partial write on kill corrupts file).

69. **`distance.py:55,337-346,416`** — call graph built by scanning raw `0xE8`
    bytes → phantom CALL edges distort AFLGo distances (accurate decoder exists
    in-tree); penalty distance for unreachable functions derived from `visited`
    left over from the last target's BFS (unstable across runs).

70. **`chi_squared.py:55`** [verified] — **FIXED 2026-08-22.** The spurious
    `0.5 * _LOG_2 * 7.0` term is gone; `_log_gamma` now agrees with
    `math.lgamma` to ~1e-15 (it was off by a constant 2.426). `_LOG_2` became
    unused and was removed with it. Still dead on CPython, so the regression
    test `tests/test_regression_log_gamma_constant.py` (20 tests) calls
    `_log_gamma` directly and derives expectations from closed-form identities
    (Gamma(n)=(n-1)!, Gamma(1/2)=sqrt(pi), Euler reflection) rather than from a
    recorded run. Original text follows. — Lanczos `_log_gamma` fallback adds
    spurious `+ln 2^3.5` (dead code today; garbage p-values if `lgamma` absent).

71. **`edge_tracker.py:781`** [verified] — `record_edge_lifetimes` fed two
    different time axes (cumulative-edge count vs exec_count) → lifetime stats
    mix units, overstated by orders of magnitude.

## LOW

72. `corpus_manager.py:174,247` [verified] — resume metadata skipped for seeds
    ≥128 bytes (`len(hex) >= 256`); most real seeds reset `fuzz_count=0`.
73. `rand_pool.py:173-191` [verified] — `randbytes(n)` replays consumed pool
    when n ≡ 0 mod pool size; batch methods silently cap at 4096 items
    (`rand_pool.py:94-141`, latent).
74. `jpeg.py:669` [verified] — unclamped `randint(1, negative)` for small
    max_len; currently masked by RandPool's silent lo-return on empty ranges
    (itself a masking hazard: converts class-#1 crashes into silent degradation).
75. `generic.py:1723-1728` [corroborated] — versifier never emits decimal digits
    (no else for base 10).
76. `generic.py:1242` [verified] — ascii_num_replace negative-token handling
    unreachable (spans contain digits only).
77. `generic.py:2033-2054` — `_structure_keyvalue` duplicates the value node
    (`k:v` → `k:vv`).
78. `operators.py:1480-1483` + `zip.py:402` — truncate ops can draw a
    delete-nothing bound yet still count as applied/timed.
79. `webm.py:380-384` — timecode-scale payload encoded with EBML size-vint
    framing; intended values never reach the wire.
80. `isobmff.py:442-454` — bootstrap writes nested containers with declared size
    0 ("to EOF"), illegal for non-top-level boxes.
81. `bmp.py:150` (+ gzip/zlib siblings) — size==2 arithmetic path lacks the upper
    clamp its size==4 siblings have (`struct.error` waiting for the next size-2
    field; latent).
82. `operator_registry.py:413` — sniffer exceptions swallowed unlogged.
83. `count_class.py:41-49` — extra "64" bucket vs AFL's merged [32-127] class;
    63→64 registers as novel where AFL wouldn't.
84. `shapley.py:174-176` — `operator_synergy` formula ≤0 by construction.
85. `gf2_common.py:192` [verified] — `e %= self.m if a != 0 else e` parses as
    `e %= (…)`; `pow(0, e>0)` returns 1.
86. `te_position.py:48` [verified] — returns `max(byte_edges.keys())` (highest
    offset); TE weights never consulted despite docstring and dedicated tests
    that only check bounds.
87. `periodicity.py:152-153` — latent IndexError when caller passes `max_lag` >
    analyzed window.
88. `qea.py:609-617` — empty population at generation boundary raises ValueError
    (empty-corpus start + 500 execs kills the loop).
89. `monte_carlo.py:265-285,536-544` — Brier logs post-update prediction
    (systematically optimistic); `cem_byte` residual mass spills into uniform-
    over-all-256 instead of unobserved-only (TV ≈ 0.04 from true predictive).
90. `randomness.py:338-358` — kmer_occupancy operates at λ≈0.14, not designed
    λ≈2 (statistic far from discriminative regime).
91. `fuzzer.py:4111` — builtin `hash(data)` as colorization taint-cache key
    (in-memory only, but the banned pattern).
92. `report.py:1197` — broad `except Exception: continue` hides permission/disk
    errors in exploitability scan.
93. `persistent_loader.py:572-575` + `inprocess.py:707-716` + `sigguard.c:30-41`
    — killpg on possibly-recycled PID from stale pidfile; resize fallback leaves
    diag pointers at unmapped old SHM; sigguard lacks SA_ONSTACK/reentrancy
    safety (currently unused).
94. `fuzzer.py:3793-3806` — `except Exception: pass` around sensitivity analysis
    (production path, zero diagnostics).

---

## Audit 2026-08-22 (high-ROI pass)

Seven findings fixed in this series (30, 31, 32, 57, 59, 61, 70) and one
(37) found to be a stale entry that was never a bug. Each was verified
against the source before being touched, per the standing warning in
docs/TODO.md that an open-looking entry is not evidence of anything — a
warning that earned its keep again here: 37 read as open and was already
correct.

Two observations worth carrying forward:

- **The count was not the bug.** Findings 32 and 59 were both filed on a
  wrong NUMBER (crash totals ~4x high; pruner arming forever). In both cases
  the number was the visible edge of a wrong SOURCE — sidecar text feeding a
  size histogram, a monotonic high-water mark feeding a threshold check.
  Fixing only the reported symptom would have left both half-broken.

- **A test asserted the defect, again.** `test_different_stderr` pinned
  finding 31 in place with `assert not diverged  # stderr differs but not
  diverged`, contradicting the module's own docstring. That is the second
  instance of this pattern in this tree after `test_hex_escape`. Both were
  written by recording what the code returned. The value-free-assertion sweep
  recorded in docs/TODO.md (573 of ~4,956 tests) measures a related but
  distinct hazard: these two assertions were specific and strong, and wrong.
  Worth a separate sweep for tests whose comments explain away a surprising
  expected value.

## Audit 2026-08-23 (parallel / scheduling / state pass)

Three HIGH findings closed: 11, 15, 18. Each was reproduced through the
production call path before being touched, and each new regression file was
run against the pre-fix source to confirm it fails there.

The theme is one pattern, now seen five times in this tree:

- **A test that pre-loads unreachable internal state.** Finding 18's nine
  existing tests set `mc._prev_op = "a"` and then asserted `record()` counted
  the transition. `record()` was correct. Nothing could ever set `_prev_op`,
  so the feature was dead and the tests were green. Finding 15's fourteen
  tests wrote seeds flat at the worker-dir top level — a directory shape
  `save_to_corpus` has never produced — and asserted the sync function found
  them. This is the same failure as `test_hex_escape` and
  `test_different_stderr`, one level up: not "assert the value the code
  returns" but "assert against the arrangement the code expects". Both
  survived the value-free-assertion sweep recorded in docs/TODO.md, because
  both assert real values. **A unit test that constructs its own fixture is
  only as good as that fixture's fidelity to what the system produces.**
  Where a writer exists, tests should call it rather than hand-rolling the
  layout — that alone would have caught finding 15.

- **A gate that reads as authoritative and is not.** Finding 11's
  `if self.resume: self._state_store.load()` looks like it decides whether
  state is read. It decides only *when*, because `get()` lazy-loads. A guard
  that omits an action does not prevent that action if something downstream
  performs it on demand. Worth a sweep for other `if flag: do_x()` sites whose
  callee has a lazy or self-healing path.

## Cross-cutting patterns worth regression tests

- **ctypes hygiene**: every libc function returning a pointer needs explicit
  `restype` (offenders: persistent.py, minimize.py ×2; shm.py is
  reference-correct). `enable_on_exec` is attr bit 12. *Addressed for the SysV
  SHM calls: `adapters/libc_shm.py` is now the single binding site and
  `tests/test_regression_shmat_restype.py` scans the package for re-binders.
  Sibling hazard, **fixed 2026-08-23**: the `(void *) -1` failure sentinel does
  NOT compare equal to `-1` once `restype=c_void_p` is declared, so adding the
  restype without rewriting the `== -1` guard silences the failure path.
  `adapters/inprocess.py` was in exactly that half-fixed state at all three of
  its attach sites — restype declared (so it passed the scan), guard left as
  `if ptr and ptr != -1` (so it admitted every failure). `reset_bitmap()`
  therefore called `memset()` through `0xffffffffffffffff`, which SIGSEGVs the
  fuzzer and is not catchable by the enclosing `except Exception`; the sentinel
  was then cached, pinning the runner to the dead pointer. All three now route
  through `libc_shm`, which returns None. The scan was extended to reject
  `ptr == -1` / `ptr != -1` in any module that calls `shmat` — a scan that
  checks only for the declared restype certifies the visible half of the fix
  and is what let this sit. See
  `docs/learnings/2026-08-23-shmat-sentinel-and-header-clobber.md`.*
- **Timeout invariant**: a wait ending without definitive status must yield -1
  (timeout), never a stale-status-derived "crash" nor an eternal freeze. Three
  backends violate it today (ptrace post-loop, run_target_fast, direct modes);
  flags checked after blocking ctypes calls enforce nothing.
- **Global-state writes need restore**: `os.environ`, signal handlers, numpy
  global RNG. One mechanism fixes test isolation, library reuse, and `-s`
  reproducibility simultaneously.
- **`AFL_MAP_SIZE` is entries, not bytes** — violated in three places;
  persistent_loader.py is the reference implementation.
- **Generation-tag reset protocol**: Python `reset_edge_map` is the only writer;
  C-side `__afl_map_reset` (incl. essential wrap-at-256 wipe) has zero callers.
  *Partly wrong, corrected 2026-08-23: `reset_edge_map` was NOT the only writer.
  `InProcessRunner.reset_bitmap()` memset the segment base, wiping the whole
  diag word at offset 4 — generation, dropped-edge counter and ctx width —
  every execution, immediately after `reset_edge_map()` had bumped the
  generation. Both sides then sat at generation 0, so entries left over from
  earlier executions read as live coverage. Its comment justified the memset
  with "the C shim's `__afl_map_reset()` rewrites the header after the target
  executes", citing the very function this entry records as having no callers.
  Now zeroes the edge table only. The wrap-at-256 half of this entry stands:
  `__afl_map_reset` still has zero callers.*
- **Index cursors over sorted listings of hash-named files** reorder on
  insertion and permanently skip files. Key on filename instead.
- **Flat listing of a sharded corpus directory** — `save_to_corpus` writes
  `seeds/<hh>/id_<hash>`, and the only top-level regular file a live corpus
  holds is `state.pkl.gz`. A consumer that lists the directory therefore does
  not merely find nothing: it finds the state file and uses it as a seed. Four
  modules had this — `minimize` (empty corpus, exit 0), the parallel worker
  sync (gzip bytes imported into every sibling), `root_cause` (state file
  reported as the nearest seed), and `report` (fixed earlier). *Closed
  2026-08-23 by moving the walk into
  `adapters/filesystem.discover_seed_files()`, the module that already owned
  the layout constants. Each consumer passes its own subtree exclusions; the
  exclusions differ legitimately and were the reason the earlier fixes were
  copied rather than shared.* Every instance was certified by a passing test
  that asserted only shape, never bytes.
- **Parallel time-series arrays must be trimmed in lockstep** — two consumers
  already assume equal lengths.
- **Memory/disk divergence in corpus mutations** (trim, near-dup removal) breaks
  resume and dedup; persistence belongs inside the mutating function.
- **CLI↔service signature drift**: adding a Fuzzer kwarg doesn't add it to
  `run_parallel` (#2); the parallel call site needs its own audit.
- **Fixed classes recur in sibling files**: restype (fixed in shm.py, broken in
  persistent.py + minimize.py), bound-method thread leak (fixed in
  forkserver.py, repeated in persistent_loader.py), non-PID-namespaced build
  paths. A "check the neighbor file" pass catches these cheaply.
- **Advertised-but-unwired adaptivity**: six mechanisms compute values nothing
  consumes (refit pacing, pairwise transitions, RobustKF R, Elo temperature,
  Allan thresholds, TE weighting in te_position).
- **Destructive fallback chains**: read-failure → zeroed data → prune/delete
  turns infrastructure hiccups into corpus loss (minimize, trim_new_coverage).
