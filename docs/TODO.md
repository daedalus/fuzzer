# TODO — fuzzer-tool Roadmap

> **Status note**: This roadmap tracks aspirational and in-progress work. Items without [x] are still pending.

## Coverage & Instrumentation
- [x] **Forkserver on default execution path** (IMPLEMENTED 2026-08-14) — the win did NOT come from re-enabling `ForkserverRunner`: `fuzz_loader.c` did fork+**exec** per input, so uncommenting it measured 0.99× on an ASAN target. `afl_shim.c` now installs a real AFL-style forkserver in its constructor (`__afl_start_forkserver()`, fds 198/199); the loader drives it and falls back to fork+exec for targets built against an older shim. 5.27× on `test_target`, 1.38× on an ASAN target with heavy static init, 2.77× end to end. Opt out with `--no-forkserver`. Targets must be rebuilt to benefit. See `docs/learnings/2026-08-14-forkserver-that-execs.md`.
- [ ] **Bounded probe window for `__afl_map_edge`** — linear-probe cost is O(map_size) on saturated tables. Bound to 8–16 slots and count drops via the SHM header so the trade is observable.
- [ ] **Hit-count bucketing on the SHM path** — `count_class.py` is already implemented but only used by the ptrace fallback. Wire it into `ShmCoverage._check_new_coverage` so loop-count-guarded branches become visible.
- [ ] **`favored` / `cull_queue` minimal-set-cover** — power schedules always run in unfavored mode because `favored` is never computed. Implement `cull_queue` over `EdgeTracker.seed_edges` and pass `favored` through the scheduler.
- [ ] **Deterministic stages + SkipDet** — the effector map is dead because no code walks the deterministic operators systematically across a seed. Add a per-seed deterministic pass gated by `SkipDetector`.
- [ ] **Fast path empty-edge-set bug** — `_check_new_coverage` returns `(False, set())` on unchanged input, but callers cache that as the real edge set, making the next diff report all edges as new.
- [ ] **Call stack coverage** — distinguish `f()→g()` from `h()→g()` by encoding caller context into the edge ID, not just `prev_loc ^ cur_loc`. Would improve edge resolution for shared-library targets.
- [ ] **Sanitizer coverage** — `-fsanitize-coverage=trace-pc-guard` support via `--clang-scov`, with auto-detection of sancov counters in `.so` targets.
- [ ] **Cmplog/comparison coverage** — symbol-based (libc interposition) and compiler-IR (`trace-cmp`) both implemented, in `afl_shim.c` behind `-D__AFL_CMPLOG=1`.

## Mutation
- [ ] **Per-format tuning of the regularity band** — the operators are currently offered unconditionally (except `invariant_break`). Several are format-shaped in practice: `spectral_peak` matters for DCT codecs, `degenerate_geometry` for vector/mesh parsers, `rank_deficient` for erasure coders. A sniffer gate like `_FORMAT_SNIFFERS` would stop them burning budget on targets that cannot use them.

## Scheduling
- [x] **Fluctuation theorems for fuzzing** (IMPLEMENTED 2026-08-13) — Jarzynski/Crooks relations implemented for mutation trajectories. Work functional `w_i = -log(max(p_i, ε))` over operator-selection probabilities; free-energy estimator `ΔF̂` printed via `StatsReporter.print_stats`. Phase 1 is diagnostics-only: estimates are not fed back into scheduling. CLI flag: `--fluctuation-theorems`. See `src/fuzzer_tool/core/fluctuation.py`, `services/fuzzer.py`, `services/stats.py`.
- [ ] **Collaborative scheduling across parallel workers** — parallel workers currently sync corpus but don't coordinate scheduling decisions. Could share exploration/exploitation state.

## Crash Analysis
- [ ] **Root cause diff** — show minimal byte diff from nearest non-crashing input to root-cause bytes.

## Performance
- [ ] **`_apply_single_mutation` havoc `max_len` enforcement** — havoc doesn't enforce `max_len` strictly (allows +1 byte per insert, up to +8 total).

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
- [ ] `_apply_single_mutation` havoc doesn't enforce `max_len` strictly (allows +1 byte per insert, up to +8 total)
- [ ] `parse_dict_line` triple-encode chain fragile for bytes > 0x7F

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
