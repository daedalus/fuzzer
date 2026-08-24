# TODO — fuzzer-tool Roadmap

> **Status note**: This roadmap tracks aspirational and in-progress work. Items without [x] are still pending.

> **Audit 2026-08-22 (second pass).** The reconciliation below was run in the
> morning; this second pass, run the same day, found SEVEN MORE entries that
> were marked pending but already implemented: the bounded probe window,
> deterministic stages + SkipDet, call stack coverage, sanitizer coverage,
> per-format tuning of the regularity band, root cause diff, and the
> `fuzzer-tool-asan` wrapper (which is not merely done but obsolete — the
> shipping entry point is strictly better than what the entry asked for).
>
> That is twelve stale entries in one day, and the bounded probe window was
> the top open item under *Coverage & Instrumentation* — the first thing a
> planning pass would pick up. The pattern is now established well enough to
> state plainly: **an unchecked `[ ]` in this file is not evidence of anything.**
> Verify against the code, and when closing an item, close it in the same
> commit as the implementation. Nine entries remain open; each was checked
> against the code on 2026-08-22 and is genuinely unimplemented.

> **Audit 2026-08-22.** Five entries were marked pending but had already been
> implemented, some for over a week. All five were verified against the code,
> not against commit messages: `favored`/`cull_queue` (called at `fuzzer.py:5058`),
> hit-count bucketing on the SHM path (`shm.py:597`), the fast-path
> empty-edge-set bug (`shm.py:168-172`), cmplog comparison coverage
> (`afl_shim.c:1286-1295`), and havoc `max_len` enforcement (`aa94dd7`,
> 2026-08-12, with a passing regression test). The havoc entry was also
> duplicated between *Performance* and *Pending Bugs*; the duplicate is gone.
>
> This mattered: `favored`/`cull_queue` was recommended as a top-priority open
> item in a planning pass earlier the same day, on the strength of this file
> alone. **Check the code before picking work off this list.** The same drift
> had already been found in `docs/bugreport_2026-08-21_merged.md`, where eight
> closed findings still read as open.

## Coverage & Instrumentation
- [ ] **Leak severity is conditional on edge count** (2026-08-23) — the fixed header clobber only carries stale coverage forward on targets with more than `(map_size - 24) / 8` live edges (8,189 at the default 65,536-entry map). Measured: entry index IS the edge id, assigned sequentially by `__sanitizer_cov_trace_pc_guard_init`, and `png_read.so` has ~4,600 guards, so nothing in `targets/` reaches the threshold. Verified deterministically by planting an entry at index 20,000: reported as live coverage pre-fix, filtered post-fix. A vendored ffmpeg or grep build would cross it naturally — worth re-running the probe (`docs/learnings/2026-08-23-shmat-sentinel-and-header-clobber.md`) once one is available.
- [ ] **`reset_bitmap()` is redundant on the runner path** (2026-08-23) — `services/runner.py` calls `shm.reset_edge_map()` immediately before `run_one()`, and `run_one` has exactly one call site, so the full-table memset in `InProcessRunner.reset_bitmap()` (reached from `_run_c_direct`) resets a table the generation bump already invalidated. Dropping the call would save a `shm_size * 8` byte memset per execution on the hot path of the fastest execution mode. Not done here because it wants measuring on a built target, and clang was unavailable in the authoring environment. The header-clobber half of this was a real bug and IS fixed — see `docs/learnings/2026-08-23-shmat-sentinel-and-header-clobber.md`.
- [x] **`read_bitmap()` / `runner.py` disagree with the SHM layout, inertly** (FIXED 2026-08-23) — fixed on all three axes together, as the entry required. `read_bitmap()` now returns `shm_size * SIZEOF_ENTRY` bytes from `_shm_ptr + SHM_METADATA_SIZE`; `runner.py` bounds the copy with `len(bitmap) <= shm.table_bytes` and memmoves to `shm._ptr + SHM_METADATA_SIZE`. Still a self-copy in the shipping configuration, so the fix is not observable end to end — `tests/test_regression_read_bitmap_offset.py` (5 tests) asserts the offset and the length directly, and its last test drives a SEPARATE source segment, which is the case the self-copy hid. Two things found while verifying: both non-SHM sources feeding `read_bitmap` are dead (`fuzz_loader.c` does not carry coverage over its protocol, so `bmp_len` is always 0; nothing in the C writes `_COV_BITMAP_OUT`), and `test_real_segment_round_trips` in `tests/test_regression_shmat_sentinel_guard.py` was pinning the base-offset read as its expected value — corrected in `9511042`.
- [x] **Forkserver on default execution path** (IMPLEMENTED 2026-08-14) — the win did NOT come from re-enabling `ForkserverRunner`: `fuzz_loader.c` did fork+**exec** per input, so uncommenting it measured 0.99× on an ASAN target. `afl_shim.c` now installs a real AFL-style forkserver in its constructor (`__afl_start_forkserver()`, fds 198/199); the loader drives it and falls back to fork+exec for targets built against an older shim. 5.27× on `test_target`, 1.38× on an ASAN target with heavy static init, 2.77× end to end. Opt out with `--no-forkserver`. Targets must be rebuilt to benefit. See `docs/learnings/2026-08-14-forkserver-that-execs.md`.
- [x] **Bounded probe window for `__afl_map_edge`** (IMPLEMENTED, verified 2026-08-22) — `__AFL_PROBE_MAX` (`afl_shim.c:333`, default 64) bounds the linear probe on BOTH lookup and insertion; bounding one without the other would let a bounded lookup miss an edge a bounded insert had placed. Drops are counted via `__afl_note_drop()` into the diag word and surfaced by `ShmCoverage.read_dropped_edges()` (four consumers: `report.py:326`, `:808`, `stats.py:605`, `fuzzer.py:4403`). The window is 64 rather than the 8–16 this entry proposed because a dropped edge is permanently invisible, not merely delayed: on `ffmpeg_read` (load 0.77) window 16 drops 1.60% and window 64 drops 0.04%. Every other instrumented target sits near 13% load, where all of these windows drop nothing.
- [x] **Hit-count bucketing on the SHM path** (IMPLEMENTED, verified 2026-08-22) — `shm.py` imports `bucket_bits` from `core/count_class.py` and folds it in through `_update_virgin_buckets`, called from the live coverage path (`shm.py:597`, `:631`). OR'd with the set-membership result rather than replacing it, so a new edge with count 0 still counts as an edge.
- [x] **`favored` / `cull_queue` minimal-set-cover** (IMPLEMENTED, verified 2026-08-22) — `Fuzzer._cull_queue()` (`fuzzer.py:2791`) computes the AFL-style top_rated/favored set; called every stats interval when `seed_edges` is populated (`:5058`) and passed through as `favored=(seed_key in self._favored)` (`:5023`). Power schedules are no longer permanently unfavored.
- [x] **Deterministic stages + SkipDet** (IMPLEMENTED, verified 2026-08-22) — `SkipDetector` is imported at `fuzzer.py:52` and constructed at `:1348`; `_deterministic_mutation_stream` walks the operators at `operators.py:229`, gated through `SkipDetector.should_det_fuzz` at `:3248`. Covered by `tests/test_skipdet.py` (31 tests).
- [x] **Fast path empty-edge-set bug** (FIXED, verified 2026-08-22) — the fast path returns `self._last_ids`, the edge set as of the last slow-path scan, rather than an empty `set()`. Callers that snapshot the return value (e.g. `Fuzzer._prev_edge_set`) now get a real "same as before" set instead of one that reads as "the target fired zero edges". See the comment at `shm.py:168-172`.
- [x] **Call stack coverage** (IMPLEMENTED, verified 2026-08-22; DEFAULT-ON since 2026-08-24) — `__AFL_CTX_SENSITIVE` folds `__afl_get_caller_ctx()` into the edge ID (`afl_shim.c`), masked to `__AFL_CTX_BITS` (default 8) so context cannot inflate the live ID count without bound. The width is advertised in the symbol NAME (`__afl_ctx_bits_N`) so `elf.detect_ctx_bits()` can size the map before the first execution. **The shim now defaults `__AFL_CTX_SENSITIVE=1` and `__AFL_DISTANCE_MODE=1`** (both opt-out with `=0`); every shim-linked build in `tools/build_targets.sh` carries `-fno-omit-frame-pointer`, without which the walk yields junk-or-zero context rather than real callers (hardened walk: degraded signal, never a crash). Check `read_dropped_edges()` rather than guessing when raising the width.
- [x] **Ctx + distance defaults flipped on** (DONE 2026-08-24) — `afl_shim.c` compiles call-stack-sensitive hashing and the AFLGo SHM-tail distance channel into EVERY target by default (`#ifndef … 1` gates; value-checked `#if __AFL_DISTANCE_MODE` so `-D__AFL_DISTANCE_MODE=0` opts out). Consequences worth knowing: `_detect_distance()` now reports "detected" for every shim build (cosmetic — directed mode still requires `--target-functions`; tail reads degrade to no-data with count==0); `detect_ctx_bits()==8` inflates auto map sizing ~4× by design; tests that assert literal context-free edge IDs pin `-D__AFL_CTX_SENSITIVE=0` explicitly (`test_regression_shm_stale_reclaim`, `test_ctx_and_map_size::TestEdgeIdZeroRegression`, `test_synthetic_target`, resize drivers[0]). Contract test: `test_ctx_and_map_size.py::test_default_compile_enables_ctx_and_distance`.
- [x] **CFG decode cache + parallel decode** (DONE 2026-08-24) — `core/cfg_cache.py`: build-id-keyed on-disk cache (one gzip pickle per binary+decoder under `~/.cache/fuzzer_cfgcache`) behind `TargetDistance._build_cfgs`, with per-function accumulate-merge so different `--target-functions` runs share one artifact. Invalidation = NT_GNU_BUILD_ID (`elf.build_id()`, sha256-file fallback) ⊕ decoder source fingerprint ⊕ SCHEMA_VERSION ⊕ size. Misses decode via fork-context ProcessPoolExecutor above 512 KiB/16-function thresholds, serial below. Safe-unpickler allowlist extended by exactly FunctionCFG/BasicBlock; corruption warns once and recomputes. Opt-outs: `--no-cfg-cache`, `FUZZER_DISABLE_CFG_CACHE`. Tests: `tests/test_cfg_cache.py` (13, incl. falsification: second-run hit / byte-change / schema-bump invalidation / parallel≡serial; adversarial: corrupt artifact / hostile pickle global). K-Scheduler W1 reuses identity/load/store for whole-program ICFG.
- [x] **Sanitizer coverage** (IMPLEMENTED, verified 2026-08-22) — `tools/build_targets.sh --clang-scov` (parsed into `WITH_CLANG_SCOV`, `:136`) builds with `-fsanitize-coverage=trace-pc-guard`; `elf.parse_sancov_guard_count()` and `parse_sancov_offsets()` auto-detect the guard/counter sections for exact map sizing, and the sizing path tells the user to rebuild with the flag when it has to estimate instead (`elf.py:1754`). Regression-tested in `tests/test_regression_build_flags.py`.
- [x] **Cmplog/comparison coverage** (IMPLEMENTED) — symbol-based (libc interposition) and compiler-IR (`trace-cmp`) both live in `afl_shim.c` behind `-D__AFL_CMPLOG=1`; `__sanitizer_cov_trace_cmp{1,2,4,8}` present at `afl_shim.c:1286-1295`. The entry's own text already said "both implemented" — only the checkbox was stale.

## Mutation
- [x] **Per-format tuning of the regularity band** (IMPLEMENTED, verified 2026-08-22) — all three named operators are gated in `_FORMAT_SNIFFERS` (`operator_registry.py:335-337`): `spectral_peak` on `_sniff_dct_transform_coded`, `degenerate_geometry` on `_sniff_mesh_or_vector_geometry`, `rank_deficient` on `_sniff_rar`. `invariant_break` is gated separately on `_has_corpus_samples` (`:454`). Note the `_FORMAT_BOOTSTRAP_RATE` caveat below before reading anything into their measured yield.

## Scheduling
- [x] **Fluctuation theorems for fuzzing** (IMPLEMENTED 2026-08-13) — Jarzynski/Crooks relations implemented for mutation trajectories. Work functional `w_i = -log(max(p_i, ε))` over operator-selection probabilities; free-energy estimator `ΔF̂` printed via `StatsReporter.print_stats`. Phase 1 is diagnostics-only: estimates are not fed back into scheduling. CLI flag: `--fluctuation-theorems`. See `src/fuzzer_tool/core/fluctuation.py`, `services/fuzzer.py`, `services/stats.py`.
- [ ] **Collaborative scheduling across parallel workers** — parallel workers currently sync corpus but don't coordinate scheduling decisions. Could share exploration/exploitation state.
- [ ] **Port Tier-1 candidates from the 2026-08 web survey** — 7 low-effort items (autotokens, entropic schedule, CASR crash clustering, trace-div/gep, n-gram edges, Zest validity channel, SGFuzz state variables). Survey + audit: `docs/web_research_port_candidates_2026-08.md`; per-item audit against live code still required before starting any of them.

## Crash Analysis
- [x] **Root cause diff** (IMPLEMENTED, verified 2026-08-22) — `core/root_cause.py` plus the `services/root_cause.py` CLI wrapper, exposed as the `root-cause` subcommand; `format_root_cause_report()` renders the diff, and the flaky-target guard refuses to report when the crash does not reproduce reliably (`services/root_cause.py:207`).

## Performance
- [x] **`_apply_single_mutation` havoc `max_len` enforcement** (FIXED 2026-08-12, `aa94dd7`) — both insert paths in `services/operators.py:_apply_single_mutation` are guarded by `len(buf) < self.f.max_len`, so no sequence of sub-mutations can overshoot. Regression test `tests/test_regression_havoc_max_len.py` (4 tests, passing), which covers the boundary case of buffers starting *at* `max_len`.

## Infrastructure
- [x] **Dockerfile** for reproducible builds and CI (2026-08-22) — `Dockerfile` + `.dockerignore`, ubuntu:24.04 to match the CI runner. Installs `.[dev,smt]` (z3 is not pulled by `dev`, and omitting it is the silent-skip case), builds `fuzz_loader` at image-build time (gitignored, and without it the suite HANGS rather than fails), and fails the build if clang/nm/z3 are absent. **Never actually built — docker was unavailable in the authoring environment. Build it once before relying on it.**
- [x] **Structured logging** (IMPLEMENTED 2026-08-22) — `--log-json FILE` (`-` for stderr) writes one JSON object per stats interval: execs, raw and Kalman-filtered eps, corpus, crashes, distinct sigs, timeouts, peak RSS, dict size, plus edges/dropped_edges and cmplog counts when those are active. Built from fuzzer attributes, not by scraping the human stats line (~30 conditional fragments). Append mode, so `--resume` extends the series; emit failures are swallowed so telemetry cannot abort a run. `tests/test_log_json.py` (8 tests).
- [x] **`fuzzer-tool-asan` wrapper** (OBSOLETE — do not build this, verified 2026-08-22) — superseded by something better. `cli/ldpreload_wrapper` is already the `fuzzer-tool` entry point itself (`pyproject.toml:40`), and it DETECTS the needed runtime from the target's undefined symbols (`__asan_init` / `__ubsan_handle_*` via `nm -D`) rather than requiring the user to pick a wrapper. A separate `-asan` entry point would be a manual, ASAN-only regression on what already ships.
- [x] **Persist `invocation` into `state.json`** (FIXED 2026-08-22) — `save_state` persists it and `load_state` restores it onto `original_invocation`, a SEPARATE attribute: `cmd_fuzz` assigns `f.invocation = argv` after the Fuzzer is constructed, i.e. after `load_state` has already run, so restoring onto `invocation` is clobbered a moment later. `save_state` prefers `original_invocation` when present, so the first command survives a chain of resumes. Report and HTML plot print `Started as:` only when it differs from the current command. `cmd_fuzz` now uses `shlex.join` rather than `" ".join`, so an argument containing a space survives being pasted back into a shell. Regression test `tests/test_regression_invocation_persisted.py` (6 tests).

## Integer-Modulus Checksum Recovery (follow-ons)
- [ ] **Weighted-sum multiplier sweep is a fixed candidate list** (`_MULTIPLIER_CANDIDATES`) — a target using an unlisted multiplier is missed entirely. Recovering `k` properly means root-finding mod `N`; Coppersmith's bound (`N^(1/deg)`) is useless at realistic data lengths, so the list is the pragmatic answer for now. Consider deriving candidates from cmplog constants instead of hardcoding.
- [ ] **`_extract_zlib_adler_pairs` only fires on valid streams** — `decompressobj` raises on an Adler mismatch, so mutated PNGs yield no pair. Pairs therefore come only from corpus seeds and successful recompressions. Reading the trailer without validating would widen the source but needs a raw-deflate path.
- [ ] **Fletcher-32 word endianness is swept, not detected** — both LE and BE are tried and verification arbitrates. Fine, but it doubles the general-path work for that family.
- [ ] **No format-aware patcher for integer checksums** — `_op_crc_learn` patches only the generic trailing field when an integer model is active. A real zlib/IDAT Adler patcher belongs in the `recompress_zlib` mutator, not here.
- [ ] **`field_constraints.py` bounded-integer pre-pass** (handover §1, deprioritized) — z3 is already fast on these small bitwidth systems, so the win is thin. Revisit only if the integer-checksum pattern proves out.

## Grammar (`.gram`) — fixed 2026-08-22

- [x] **Quoted literals did not decode escapes** — `"\xFF\xD8"` was eight ASCII characters, not the JPEG SOI marker. All 35 rules of `jpeg.gram` are quoted marker definitions, so the entire grammar emitted literal `\xFF` text and could never generate a JPEG. Found by auditing the parser class that `parse_dict_line` belonged to. **A test asserted the buggy value** (`test_hex_escape`, commented "Parser treats backslash escapes as literal characters") — written by observing output rather than intent, so it held the defect in place.
- [x] **Undefined rule references expanded to `b"?"` at DEBUG level** — invisible. Bare words are rule references, so `signature = \x89PNG\r\n\x1a\n` asked for a nonterminal named `PNG`; `png.gram` shipped five (PNG, IHDR, IDAT, IEND, PLTE), putting ~8 stray `?` bytes in every generation and no valid chunk name in any. `rar.gram` had two, one of which (`header_chain`) was referenced by the START rule and never defined at all. Now logs at WARNING; the grammars quote their literals and `header_chain = header+` is defined.

### Test-quality note

A sweep found **573 of ~4,956 tests (11.6%) whose every assertion is
value-free** — `is not None`, `isinstance`, `len(...) > 0`. That is what let
the dictionary and grammar bugs through: output was well-typed and
semantically wrong. An invariant audit over all 134 operators (18,090
invocations across nine input formats, checking type / max_len / exceptions)
found **zero** violations, so the operators themselves are sound; the risk
concentrates in PARSERS, where a wrong byte is still a valid byte. The
`test_hex_escape` case is the worse variant: an assertion strong enough to
pass review that pins the defect. Prefer asserting values, and derive the
expected value from the spec, not from a run.

## Pending Bugs
- [x] `parse_dict_line` — **the enclosing-quote contract WAS broken. Fixed 2026-08-22.** The earlier spot-check was right that high-byte decoding is fine; the deferred half of the check is where the bug was. The AFL format encloses the token in double quotes, and the quotes were kept as content, so `"IDAT"` produced `b'"IDAT"'` and matched nothing in the target. **12,169 of the 18,311 tokens in `dictionaries/` (66.5%) were affected**, including every PNG/GIF/ZIP magic. Three further defects fell out of the same rewrite:
  - splitting on the first `=` mangled any token containing one (`"a=b"` → `b'b"'`), and destroyed the bare unquoted token lists in `ruby.dict` and `rar.dict` (5,770 lines, of which 296 contain an `=`: `!=` → `b''`, `==` → `b'='`). An unquoted line is now taken whole — AFL only puts an unquoted `=` between a name and its *quoted* value, so a line with no quotes cannot be a name/value pair.
  - the `\xNN` regex sweep matched inside an escaped backslash, so `\\x41` decoded as backslash + `A` instead of backslash + `x41`. Replaced by a single left-to-right scan, since the escapes are not independent.
  - `\\` and `\"` were not decoded at all.

  This survived because the existing tests asserted only that the result was non-None `bytes`, never what the bytes were. `tests/test_regression_dict_quotes.py` (23 tests) asserts values, and checks the shipped dictionaries directly.

## Operator-yield triage (FFmpeg session follow-up)

Closed out the "persistently low-yield operators" item. Two of the five were
real defects; the rest were measurement artifacts.

- [x] `invariant_break` (0/234, 0.0%) — **real bug, fixed.** `invariant_mask`
  spanned `min(len(s) for s in samples)`, so the shortest corpus entry set the
  mask width. Fuzzing accumulates trimmed inputs, so one short (or empty) entry
  collapsed the mask to nothing. Now spans the largest prefix `min_samples`
  entries reach, with only those entries contributing.
- [x] `jpeg_crc_fix` (1.0%), `line_mutate` (2.6%), `tree_mutate` (3.2%) — **not
  bugs.** Measured on matching input: 99.3% / 98.0% / 97.3% (JSON) and 76.0%
  (XML) over 300 trials each. Their run-time rates reflect a corpus with no
  JPEG or text/tree-structured entries, not broken mutators.

### Caveat for any future per-operator rate measurement

`_format_available` offers a never-yet-seen format on non-matching input 2% of
the time (`_FORMAT_BOOTSTRAP_RATE`), so a sniffer-gated operator's *denominator*
includes selections where it was never going to fire. On a corpus containing
none of its format, an operator's reported success rate is essentially the
trickle rate, and says nothing about correctness. Compare against a corpus that
actually contains the format before concluding an operator is broken.

This also made the no-op regression sweep seed-dependent: a trickle-offered
operator that correctly declines to mutate looks like a pure no-op. Fixed by
giving every sniffer-gated operator a matching sample in the battery; keep it
that way when adding operators.

## Verified not-a-bug

- `--cmplog` under `--inprocess-direct` (reported as "Cmplog: disabled" in the
  FFmpeg session). The wiring is correct: `_detect_cmplog` finds the exported
  `__cmplog_reset`, and the direct_lite path prints "compiled into target .so
  (direct_lite compatible)" and collects pairs. Reproduced on
  `targets/cmplog_exercise.c` built with `-D__AFL_CMPLOG=1`: 420 tokens /
  450 pairs.
  The session recipe simply never enabled it — `-c` is `--coverage`, and
  `--cmplog` has no short form. Add `--cmplog` explicitly to the FFmpeg recipe.
