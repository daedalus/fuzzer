# Merged bug report — fuzzer-tool (2026-08-21)

Consolidation of two independent read-only audits of the whole tree
(`src/fuzzer_tool/`, ~71k lines Python + ~2.5k lines C) plus dynamic testing.
Findings found by both audits are marked **[corroborated]**; findings confirmed
by execution or line-trace during the merge are marked **[verified]**.

**Closed findings are pruned from this file.** What remains is open work only.
Finding numbers are the ORIGINAL ones and are deliberately not renumbered, so
the gaps are closed items and any reference to a number from a commit message,
a learnings note or a source comment still resolves to the same finding in git
history.

**Line numbers are as of 2026-08-21 and have drifted.** The tree has moved a
great deal since; spot-checking the surviving findings shows several now point
at unrelated code. Locate a finding by the symbol it names, not by its line —
and confirm the defect is still there before working on it, for the same reason
`docs/TODO.md` says an unchecked box is not evidence of anything. Finding 83 is
the worked example: `core/count_class.py` now documents the extra "64" bucket
as deliberate, with the two ladders pinned separately in
`tests/test_count_class_exhaustive.py`, so what was filed as a defect has since
been settled as intended behaviour without the entry being touched.

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

**E3. Fork-in-multithreaded-process hazard.** `persistent.py:72`,
`runner.py:311`, inprocess loader call `os.fork()` while threads exist
(CPython DeprecationWarning; deadlock risk). Also breaks
`test_regression_persistent_execve_failure_exits` deterministically: the warning
summary reprints the inner test name, so its `count(...) == 1` assertion sees 2.
**[verified]**

---

## HIGH

10. **`cli/commands.py` env/global state** [verified] — see E3: import-time
    signal handlers + child-killing atexit (`fuzzer.py:159-168`), unrestored
    `os.environ` writes across cmplog/fuzzer/stats/inprocess, fork-in-threaded
    process. Breaks test isolation, library reuse, and reproducibility
    simultaneously.

12. **`services/fuzzer.py:3369` (also :3795, :5061)** — multi-target mode reads
    per-run edges from `self.shm_cov` instead of the per-target segment in
    `_target_shm_covs`; seed edges/momentum/stall detection run on empty data.

13. **`adapters/inprocess.py:443-494`** [corroborated, verified] —
    `_run_c_direct` cannot survive real faults: returning Python SIGSEGV handler
    re-executes the faulting instruction (infinite fault loop); `timed_out` flag
    checked only after the blocking ctypes call returns, so a looping native
    target can never be stopped. First wild-pointer input under
    `--inprocess-direct` freezes the fuzzer permanently.

16. **`core/transfer_entropy.py:84-105`** [verified empirically] — plug-in TE
    estimator reports ~2.6 bits for *independent* uniform byte streams (n=1500);
    no bias correction for context cardinality. `byte_to_edge_flow` /
    `causal_chains` produce spurious causal edges on essentially any input.

17. **`core/crash_eta.py:66-135`** — `CrashMITracker` records only crashing
    inputs, so `byte_total == joint_crash` always; "MI" degenerates to position
    frequency × log₂(1/p_crash). Crash ETA and mutation targeting driven by noise.

20. **`adapters/inprocess.py:513-595` + `runner.py:154`** — `direct_lite`
    (hardcoded-on default) never resets SHM between iterations → every exec
    reports the cumulative union of all prior coverage; per-exec attribution and
    stability calibration meaningless. Compounded by `inprocess.py:391-414`
    memsetting entry-*count* as bytes (wipes header + ⅛ of table, forcing
    generation to 0 so stale entries look current).

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

56. **`seed_picker.py:206,840-861,886-926`** [verified] — three sub-findings,
    all open (the `randint(4, min(64, max_len))` half was fixed 2026-08-24 and
    is pruned): the 3-D Pareto sweep excludes genuinely non-dominated seeds
    (running maxima taken from different items); Pareto sampling uses global
    `random`, breaking seeded reproducibility; the front cache is keyed on
    `len(corpus)` and goes stale after an in-place trim.

60. **`fuzzer.py:2318-2329`** [verified] — `_check_differential` discards
    computed results and records hardcoded zeros; drift stats meaningless
    whenever `--differential-target` is used.

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

68. **`state_store.py:190-194` + `stats.py:349-406`** — temp-file+rename without
    fsync (power loss can persist empty/truncated `state.pkl.gz`); stats and
    coverage JSON written non-atomically (partial write on kill corrupts file).

69. **`distance.py:55,337-346,416`** — call graph built by scanning raw `0xE8`
    bytes → phantom CALL edges distort AFLGo distances (accurate decoder exists
    in-tree); penalty distance for unreachable functions derived from `visited`
    left over from the last target's BFS (unstable across runs).

71. **`edge_tracker.py:781`** [verified] — `record_edge_lifetimes` fed two
    different time axes (cumulative-edge count vs exec_count) → lifetime stats
    mix units, overstated by orders of magnitude.

## LOW

72. `corpus_manager.py:174,247` [verified] — resume metadata skipped for seeds
    ≥128 bytes (`len(hex) >= 256`); most real seeds reset `fuzz_count=0`.
73. `rand_pool.py:173-191` [verified] — `randbytes(n)` replays consumed pool
    when n ≡ 0 mod pool size; batch methods silently cap at 4096 items
    (`rand_pool.py:94-141`, latent).
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

## Patterns the closed findings left behind

The closed findings themselves are pruned — `CHANGELOG.md` and git history hold
them. These are the patterns that kept recurring across them, and each one has
already cost time more than once.

- **The count was not the bug.** Two separate findings were filed on a wrong
  NUMBER (crash totals ~4x high; the memory pruner arming forever). In both
  cases the number was the visible edge of a wrong SOURCE — sidecar text
  feeding a size histogram, a monotonic high-water mark feeding a threshold
  check. Fixing only the reported symptom would have left both half-broken.

- **A test asserted the defect.** `test_different_stderr` pinned a divergence
  bug in place with `assert not diverged  # stderr differs but not diverged`,
  contradicting the module's own docstring; `test_hex_escape` did the same for
  the grammar escape bug. Both were written by recording what the code
  returned. The value-free-assertion sweep in `docs/TODO.md` measures a related
  but distinct hazard: these assertions were specific and strong, and wrong.
  Worth a sweep for tests whose comments explain away a surprising expected
  value.

- **A test that pre-loads unreachable internal state.** The Monte-Carlo
  transition finding had nine existing tests that set `mc._prev_op = "a"` and
  then asserted `record()` counted the transition. `record()` was correct;
  nothing could ever set `_prev_op`, so the feature was dead and the tests were
  green. The parallel corpus-sync finding had fourteen tests writing seeds flat
  at the worker-dir top level — a directory shape `save_to_corpus` has never
  produced. Not "assert the value the code returns" but "assert against the
  arrangement the code expects", and both survived the value-free sweep because
  both assert real values. **A unit test that constructs its own fixture is only
  as good as that fixture's fidelity to what the system produces.** Where a
  writer exists, call it rather than hand-rolling the layout.

- **A gate that reads as authoritative and is not.**
  `if self.resume: self._state_store.load()` looks like it decides whether
  state is read. It decides only *when*, because `get()` lazy-loads. A guard
  that omits an action does not prevent that action if something downstream
  performs it on demand. Worth a sweep for other `if flag: do_x()` sites whose
  callee has a lazy or self-healing path.

- **Verify before starting.** Every pass over this file has found at least one
  entry that read as open and was already correct. Check the source, not the
  marker.

## Cross-cutting patterns worth regression tests

- **ctypes hygiene**: every libc function returning a pointer needs an explicit
  `restype`. `adapters/libc_shm.py` is the single binding site for the SysV SHM
  calls and `tests/test_regression_shmat_restype.py` scans the package for
  re-binders. **Sibling hazard, and the reason the scan alone is not enough:**
  the `(void *) -1` failure sentinel does NOT compare equal to `-1` once
  `restype=c_void_p` is declared, so adding the restype without rewriting the
  `== -1` guard silences the failure path — a module can pass the restype scan
  while admitting every attach failure, then `memset()` through
  `0xffffffffffffffff`, which SIGSEGVs the fuzzer and is not catchable by an
  enclosing `except Exception`. The scan now rejects `ptr == -1` / `ptr != -1`
  in any module that calls `shmat`. Any new pointer-returning binding needs both
  halves. `enable_on_exec` is attr bit 12. See
  `docs/learnings/2026-08-23-shmat-sentinel-and-header-clobber.md`.
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
