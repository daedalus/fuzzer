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
- [x] **Forkserver on default execution path** (IMPLEMENTED 2026-08-14) — the win did NOT come from re-enabling `ForkserverRunner`: `fuzz_loader.c` did fork+**exec** per input, so uncommenting it measured 0.99× on an ASAN target. `afl_shim.c` now installs a real AFL-style forkserver in its constructor (`__afl_start_forkserver()`, fds 198/199); the loader drives it and falls back to fork+exec for targets built against an older shim. 5.27× on `test_target`, 1.38× on an ASAN target with heavy static init, 2.77× end to end. Opt out with `--no-forkserver`. Targets must be rebuilt to benefit. See `docs/learnings/2026-08-14-forkserver-that-execs.md`.
- [x] **Bounded probe window for `__afl_map_edge`** (IMPLEMENTED, verified 2026-08-22) — `__AFL_PROBE_MAX` (`afl_shim.c:333`, default 64) bounds the linear probe on BOTH lookup and insertion; bounding one without the other would let a bounded lookup miss an edge a bounded insert had placed. Drops are counted via `__afl_note_drop()` into the diag word and surfaced by `ShmCoverage.read_dropped_edges()` (four consumers: `report.py:326`, `:808`, `stats.py:605`, `fuzzer.py:4403`). The window is 64 rather than the 8–16 this entry proposed because a dropped edge is permanently invisible, not merely delayed: on `ffmpeg_read` (load 0.77) window 16 drops 1.60% and window 64 drops 0.04%. Every other instrumented target sits near 13% load, where all of these windows drop nothing.
- [x] **Hit-count bucketing on the SHM path** (IMPLEMENTED, verified 2026-08-22) — `shm.py` imports `bucket_bits` from `core/count_class.py` and folds it in through `_update_virgin_buckets`, called from the live coverage path (`shm.py:597`, `:631`). OR'd with the set-membership result rather than replacing it, so a new edge with count 0 still counts as an edge.
- [x] **`favored` / `cull_queue` minimal-set-cover** (IMPLEMENTED, verified 2026-08-22) — `Fuzzer._cull_queue()` (`fuzzer.py:2791`) computes the AFL-style top_rated/favored set; called every stats interval when `seed_edges` is populated (`:5058`) and passed through as `favored=(seed_key in self._favored)` (`:5023`). Power schedules are no longer permanently unfavored.
- [x] **Deterministic stages + SkipDet** (IMPLEMENTED, verified 2026-08-22) — `SkipDetector` is imported at `fuzzer.py:52` and constructed at `:1348`; `_deterministic_mutation_stream` walks the operators at `operators.py:229`, gated through `SkipDetector.should_det_fuzz` at `:3248`. Covered by `tests/test_skipdet.py` (31 tests).
- [x] **Fast path empty-edge-set bug** (FIXED, verified 2026-08-22) — the fast path returns `self._last_ids`, the edge set as of the last slow-path scan, rather than an empty `set()`. Callers that snapshot the return value (e.g. `Fuzzer._prev_edge_set`) now get a real "same as before" set instead of one that reads as "the target fired zero edges". See the comment at `shm.py:168-172`.
- [x] **Call stack coverage** (IMPLEMENTED, verified 2026-08-22) — `-D__AFL_CTX_SENSITIVE=1` folds `__afl_get_caller_ctx()` into the edge ID (`afl_shim.c:571`), masked to `__AFL_CTX_BITS` (default 8) so context cannot inflate the live ID count without bound. The width is advertised in the symbol NAME (`__afl_ctx_bits_N`) so `elf.detect_ctx_bits()` can size the map before the first execution. Enabled by `tools/build_targets.sh`; check `read_dropped_edges()` rather than guessing when raising the width.
- [x] **Sanitizer coverage** (IMPLEMENTED, verified 2026-08-22) — `tools/build_targets.sh --clang-scov` (parsed into `WITH_CLANG_SCOV`, `:136`) builds with `-fsanitize-coverage=trace-pc-guard`; `elf.parse_sancov_guard_count()` and `parse_sancov_offsets()` auto-detect the guard/counter sections for exact map sizing, and the sizing path tells the user to rebuild with the flag when it has to estimate instead (`elf.py:1754`). Regression-tested in `tests/test_regression_build_flags.py`.
- [x] **Cmplog/comparison coverage** (IMPLEMENTED) — symbol-based (libc interposition) and compiler-IR (`trace-cmp`) both live in `afl_shim.c` behind `-D__AFL_CMPLOG=1`; `__sanitizer_cov_trace_cmp{1,2,4,8}` present at `afl_shim.c:1286-1295`. The entry's own text already said "both implemented" — only the checkbox was stale.

## Mutation
- [x] **Per-format tuning of the regularity band** (IMPLEMENTED, verified 2026-08-22) — all three named operators are gated in `_FORMAT_SNIFFERS` (`operator_registry.py:335-337`): `spectral_peak` on `_sniff_dct_transform_coded`, `degenerate_geometry` on `_sniff_mesh_or_vector_geometry`, `rank_deficient` on `_sniff_rar`. `invariant_break` is gated separately on `_has_corpus_samples` (`:454`). Note the `_FORMAT_BOOTSTRAP_RATE` caveat below before reading anything into their measured yield.

## Scheduling
- [x] **Fluctuation theorems for fuzzing** (IMPLEMENTED 2026-08-13) — Jarzynski/Crooks relations implemented for mutation trajectories. Work functional `w_i = -log(max(p_i, ε))` over operator-selection probabilities; free-energy estimator `ΔF̂` printed via `StatsReporter.print_stats`. Phase 1 is diagnostics-only: estimates are not fed back into scheduling. CLI flag: `--fluctuation-theorems`. See `src/fuzzer_tool/core/fluctuation.py`, `services/fuzzer.py`, `services/stats.py`.
- [ ] **Collaborative scheduling across parallel workers** — parallel workers currently sync corpus but don't coordinate scheduling decisions. Could share exploration/exploitation state.

## Crash Analysis
- [x] **Root cause diff** (IMPLEMENTED, verified 2026-08-22) — `core/root_cause.py` plus the `services/root_cause.py` CLI wrapper, exposed as the `root-cause` subcommand; `format_root_cause_report()` renders the diff, and the flaky-target guard refuses to report when the crash does not reproduce reliably (`services/root_cause.py:207`).

## Performance
- [x] **`_apply_single_mutation` havoc `max_len` enforcement** (FIXED 2026-08-12, `aa94dd7`) — both insert paths in `services/operators.py:_apply_single_mutation` are guarded by `len(buf) < self.f.max_len`, so no sequence of sub-mutations can overshoot. Regression test `tests/test_regression_havoc_max_len.py` (4 tests, passing), which covers the boundary case of buffers starting *at* `max_len`.

## Infrastructure
- [ ] **Dockerfile** for reproducible builds and CI
- [ ] **Structured logging** (e.g. `--log-json`) for machine-parseable output
- [x] **`fuzzer-tool-asan` wrapper** (OBSOLETE — do not build this, verified 2026-08-22) — superseded by something better. `cli/ldpreload_wrapper` is already the `fuzzer-tool` entry point itself (`pyproject.toml:40`), and it DETECTS the needed runtime from the target's undefined symbols (`__asan_init` / `__ubsan_handle_*` via `nm -D`) rather than requiring the user to pick a wrapper. A separate `-asan` entry point would be a manual, ASAN-only regression on what already ships.
- [x] **Persist `invocation` into `state.json`** (FIXED 2026-08-22) — `save_state` persists it and `load_state` restores it onto `original_invocation`, a SEPARATE attribute: `cmd_fuzz` assigns `f.invocation = argv` after the Fuzzer is constructed, i.e. after `load_state` has already run, so restoring onto `invocation` is clobbered a moment later. `save_state` prefers `original_invocation` when present, so the first command survives a chain of resumes. Report and HTML plot print `Started as:` only when it differs from the current command. `cmd_fuzz` now uses `shlex.join` rather than `" ".join`, so an argument containing a space survives being pasted back into a shell. Regression test `tests/test_regression_invocation_persisted.py` (6 tests).

## Integer-Modulus Checksum Recovery (follow-ons)
- [ ] **Weighted-sum multiplier sweep is a fixed candidate list** (`_MULTIPLIER_CANDIDATES`) — a target using an unlisted multiplier is missed entirely. Recovering `k` properly means root-finding mod `N`; Coppersmith's bound (`N^(1/deg)`) is useless at realistic data lengths, so the list is the pragmatic answer for now. Consider deriving candidates from cmplog constants instead of hardcoding.
- [ ] **`_extract_zlib_adler_pairs` only fires on valid streams** — `decompressobj` raises on an Adler mismatch, so mutated PNGs yield no pair. Pairs therefore come only from corpus seeds and successful recompressions. Reading the trailer without validating would widen the source but needs a raw-deflate path.
- [ ] **Fletcher-32 word endianness is swept, not detected** — both LE and BE are tried and verification arbitrates. Fine, but it doubles the general-path work for that family.
- [ ] **No format-aware patcher for integer checksums** — `_op_crc_learn` patches only the generic trailing field when an integer model is active. A real zlib/IDAT Adler patcher belongs in the `recompress_zlib` mutator, not here.
- [ ] **`field_constraints.py` bounded-integer pre-pass** (handover §1, deprioritized) — z3 is already fast on these small bitwidth systems, so the win is thin. Revisit only if the integer-checksum pattern proves out.

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
