# TODO — fuzzer-tool Roadmap

> **Status note**: This roadmap tracks aspirational and in-progress work. Items without [x] are still pending.

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
- [ ] **Bounded probe window for `__afl_map_edge`** — linear-probe cost is O(map_size) on saturated tables. Bound to 8–16 slots and count drops via the SHM header so the trade is observable.
- [x] **Hit-count bucketing on the SHM path** (IMPLEMENTED, verified 2026-08-22) — `shm.py` imports `bucket_bits` from `core/count_class.py` and folds it in through `_update_virgin_buckets`, called from the live coverage path (`shm.py:597`, `:631`). OR'd with the set-membership result rather than replacing it, so a new edge with count 0 still counts as an edge.
- [x] **`favored` / `cull_queue` minimal-set-cover** (IMPLEMENTED, verified 2026-08-22) — `Fuzzer._cull_queue()` (`fuzzer.py:2791`) computes the AFL-style top_rated/favored set; called every stats interval when `seed_edges` is populated (`:5058`) and passed through as `favored=(seed_key in self._favored)` (`:5023`). Power schedules are no longer permanently unfavored.
- [ ] **Deterministic stages + SkipDet** — the effector map is dead because no code walks the deterministic operators systematically across a seed. Add a per-seed deterministic pass gated by `SkipDetector`.
- [x] **Fast path empty-edge-set bug** (FIXED, verified 2026-08-22) — the fast path returns `self._last_ids`, the edge set as of the last slow-path scan, rather than an empty `set()`. Callers that snapshot the return value (e.g. `Fuzzer._prev_edge_set`) now get a real "same as before" set instead of one that reads as "the target fired zero edges". See the comment at `shm.py:168-172`.
- [ ] **Call stack coverage** — distinguish `f()→g()` from `h()→g()` by encoding caller context into the edge ID, not just `prev_loc ^ cur_loc`. Would improve edge resolution for shared-library targets.
- [ ] **Sanitizer coverage** — `-fsanitize-coverage=trace-pc-guard` support via `--clang-scov`, with auto-detection of sancov counters in `.so` targets.
- [x] **Cmplog/comparison coverage** (IMPLEMENTED) — symbol-based (libc interposition) and compiler-IR (`trace-cmp`) both live in `afl_shim.c` behind `-D__AFL_CMPLOG=1`; `__sanitizer_cov_trace_cmp{1,2,4,8}` present at `afl_shim.c:1286-1295`. The entry's own text already said "both implemented" — only the checkbox was stale.

## Mutation
- [ ] **Per-format tuning of the regularity band** — the operators are currently offered unconditionally (except `invariant_break`). Several are format-shaped in practice: `spectral_peak` matters for DCT codecs, `degenerate_geometry` for vector/mesh parsers, `rank_deficient` for erasure coders. A sniffer gate like `_FORMAT_SNIFFERS` would stop them burning budget on targets that cannot use them.

## Scheduling
- [x] **Fluctuation theorems for fuzzing** (IMPLEMENTED 2026-08-13) — Jarzynski/Crooks relations implemented for mutation trajectories. Work functional `w_i = -log(max(p_i, ε))` over operator-selection probabilities; free-energy estimator `ΔF̂` printed via `StatsReporter.print_stats`. Phase 1 is diagnostics-only: estimates are not fed back into scheduling. CLI flag: `--fluctuation-theorems`. See `src/fuzzer_tool/core/fluctuation.py`, `services/fuzzer.py`, `services/stats.py`.
- [ ] **Collaborative scheduling across parallel workers** — parallel workers currently sync corpus but don't coordinate scheduling decisions. Could share exploration/exploitation state.

## Crash Analysis
- [ ] **Root cause diff** — show minimal byte diff from nearest non-crashing input to root-cause bytes.

## Performance
- [x] **`_apply_single_mutation` havoc `max_len` enforcement** (FIXED 2026-08-12, `aa94dd7`) — both insert paths in `services/operators.py:_apply_single_mutation` are guarded by `len(buf) < self.f.max_len`, so no sequence of sub-mutations can overshoot. Regression test `tests/test_regression_havoc_max_len.py` (4 tests, passing), which covers the boundary case of buffers starting *at* `max_len`.

## Infrastructure
- [ ] **Dockerfile** for reproducible builds and CI
- [ ] **Structured logging** (e.g. `--log-json`) for machine-parseable output
- [ ] **`fuzzer-tool-asan` wrapper** — CLI wrapper that sets `LD_PRELOAD=libasan.so.8` and exec's into the real fuzzer (mentioned in ASAN-LIMITATION.md but not yet generated as a installable entry point)
- [ ] **Persist `invocation` into `state.json`** — `fuzzer.invocation` (`sys.argv`, captured in `cmd_fuzz` for the report exec lines) is not saved on shutdown; a `--resume` run therefore reports only the resumed command, not the original one. Saving it into `state.json` would let reports on resumed sessions carry the original invocation.

## Integer-Modulus Checksum Recovery (follow-ons)
- [ ] **Weighted-sum multiplier sweep is a fixed candidate list** (`_MULTIPLIER_CANDIDATES`) — a target using an unlisted multiplier is missed entirely. Recovering `k` properly means root-finding mod `N`; Coppersmith's bound (`N^(1/deg)`) is useless at realistic data lengths, so the list is the pragmatic answer for now. Consider deriving candidates from cmplog constants instead of hardcoding.
- [ ] **`_extract_zlib_adler_pairs` only fires on valid streams** — `decompressobj` raises on an Adler mismatch, so mutated PNGs yield no pair. Pairs therefore come only from corpus seeds and successful recompressions. Reading the trailer without validating would widen the source but needs a raw-deflate path.
- [ ] **Fletcher-32 word endianness is swept, not detected** — both LE and BE are tried and verification arbitrates. Fine, but it doubles the general-path work for that family.
- [ ] **No format-aware patcher for integer checksums** — `_op_crc_learn` patches only the generic trailing field when an integer model is active. A real zlib/IDAT Adler patcher belongs in the `recompress_zlib` mutator, not here.
- [ ] **`field_constraints.py` bounded-integer pre-pass** (handover §1, deprioritized) — z3 is already fast on these small bitwidth systems, so the win is thin. Revisit only if the integer-checksum pattern proves out.

## Pending Bugs
- [ ] `parse_dict_line` triple-encode chain fragile for bytes > 0x7F — **possibly stale.** Spot-checked 2026-08-22: high-byte decoding is correct (`\xff`, `\x80\x9f\xfe` and UTF-8 `é` all round-trip to the right bytes) and the implementation encodes raw UTF-8 rather than chaining encodes. Not closed, because the enclosing-quote contract was not checked against the caller. Re-verify before spending time on it.

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
